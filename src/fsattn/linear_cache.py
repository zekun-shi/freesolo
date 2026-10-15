from __future__ import annotations

import torch
import torch.nn.functional as F

from .prefix_topp import stable_topk_slots


def rotary_at(attn, x, start):
    freq = attn.rope.inv_freq
    pos = torch.arange(x.shape[2], device=x.device, dtype=freq.dtype) + start
    phase = torch.outer(pos, freq)
    cos, sin = phase.cos()[None, None], phase.sin()[None, None]
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def _gather_cells(x, slots):
    shape = slots.shape + x.shape[3:]
    index = slots.reshape(slots.shape + (1,) * (x.ndim - 3)).expand(shape)
    return x.gather(2, index)


def _clone_tree(obj):
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {key: _clone_tree(value) for key, value in obj.items()}
    return obj


class LinearFSCache:
    bounded = True

    def __init__(self, attn):
        attn._assert_cacheable(phase="prefill")
        if attn._is_dense_lr():
            raise ValueError("LinearFSCache requires native prefix pruning.")
        if attn.use_rope and attn.rope.inv_freq.dtype not in (torch.float32, torch.float64):
            raise ValueError("Keep RoPE frequencies in FP32; use autocast for bf16 inference.")
        if attn.use_rope and getattr(attn.rope, "mscale", 1.0) != 1.0:
            raise NotImplementedError("Linear cache supports the deployed unscaled local RoPE.")
        self.attn = attn
        self.s = attn.cell_stride
        self.Bq = attn._cell_block
        self.W = attn.window_size
        self.P = attn._prune_cfg("cell_prune_pool", "FS_PRUNE_POOL", 64)
        self.Wc = attn._prune_cfg("cell_prune_window", "FS_PRUNE_WINDOW", 8)
        if min(self.s, self.Bq, self.P) <= 0 or min(self.W, self.Wc) < 0:
            raise ValueError("Invalid linear-cache capacities.")
        self.pos = 0
        self.eligible = 0
        self.block_start = -1
        self._top = None
        self._recent = None
        self._pending_k = self._pending_v = None
        self._sr_k = self._sr_v = None
        self.pool = None
        self.max_merge_cells = 0
        self.merged_cells = 0
        self.updates = 0

    def _cells(self, k, v, first):
        b, h, n, d = k.shape
        count = n // self.s
        kc = k.reshape(b, h, count, self.s, d)
        vc = v.reshape(b, h, count, self.s, d)
        ids = torch.arange(first, first + count, device=k.device)
        return dict(
            ids=ids.view(1, 1, -1).expand(b, h, -1),
            priority=kc.float().norm(dim=-1).amax(dim=-1),
            mean=kc.mean(dim=3).float(), k=kc, v=vc,
        )

    @staticmethod
    def _concat(left, right):
        if left is None:
            return right
        return {name: torch.cat((left[name], value), dim=2)
                for name, value in right.items()}

    def prepare(self):
        start = self.pos // self.Bq * self.Bq
        if start == self.block_start:
            return
        target = start // self.s
        count = target - self.eligible
        if count:
            n = count * self.s
            new = self._cells(self._pending_k[:, :, :n],
                              self._pending_v[:, :, :n], self.eligible)
            candidates = self._concat(self._top, new)
            width = candidates["ids"].shape[-1]
            self.max_merge_cells = max(self.max_merge_cells, width)
            self.merged_cells += width
            self.updates += 1
            slots = stable_topk_slots(candidates["priority"], candidates["ids"], self.P)
            self._top = {name: _gather_cells(value, slots)
                         for name, value in candidates.items()}
            if self.Wc:
                recent = self._concat(self._recent, new)
                self._recent = {name: value[:, :, -self.Wc:].clone()
                                for name, value in recent.items()}
            self._pending_k = self._pending_k[:, :, n:].clone()
            self._pending_v = self._pending_v[:, :, n:].clone()
            self.eligible = target
        self.block_start = start
        if self._top is None:
            self.pool = None
            return
        recent = (None if self._recent is None else
                  {name: value.flip(2) for name, value in self._recent.items()})
        pool = self._top if recent is None else self._concat(self._top, recent)
        ids = pool["ids"]
        c = ids.shape[-1]
        earlier = torch.ones(c, c, device=ids.device, dtype=torch.bool).tril(-1)
        duplicate = ((ids[..., :, None] == ids[..., None, :]) & earlier).any(-1)
        self.pool = dict(pool, valid=~duplicate)

    def ingest(self, k, v):
        n = k.shape[2]
        if n > self.Bq - self.pos % self.Bq:
            raise ValueError("ingest must not cross a query-block boundary")
        for name, value in (("_pending_k", k), ("_pending_v", v),
                            ("_sr_k", k), ("_sr_v", v)):
            old = getattr(self, name)
            joined = value if old is None else torch.cat((old, value), dim=2)
            if name.startswith("_sr"):
                joined = joined[:, :, -self.W:] if self.W else joined[:, :, :0]
            setattr(self, name, joined.detach().clone())
        self.pos += n

    def _short_range(self, q, k, v):
        a = self.attn
        old = 0 if self._sr_k is None else self._sr_k.shape[2]
        kw = k if old == 0 else torch.cat((self._sr_k, k), dim=2)
        vw = v if old == 0 else torch.cat((self._sr_v, v), dim=2)
        if a.use_rope:
            q = rotary_at(a, q, self.pos)
            kw = rotary_at(a, kw, self.pos - old)
        qi = torch.arange(q.shape[2], device=q.device) + self.pos
        kj = torch.arange(kw.shape[2], device=q.device) + self.pos - old
        dist = qi[:, None] - kj[None, :]
        valid = (dist >= 0) & (dist <= self.W)
        if a.fs_uniform_sr:
            mask = torch.zeros_like(dist, dtype=q.dtype).masked_fill(~valid, -torch.inf)
            mask = mask[None, None]
        else:
            arg = dist.abs().float()[None] / (2 * a.sigma.float()[:, None, None])
            mask = torch.erfc(arg).clamp_min(1e-10).log().masked_fill(~valid, -torch.inf)[None]
        return F.scaled_dot_product_attention(q, kw, vw, attn_mask=mask, scale=a.scale)

    def select(self, q):
        pool = self.pool
        if pool is None:
            return None, None
        a = self.attn
        content = torch.einsum("bhqd,bhcd->bhqc", q.float(), pool["mean"]) * a.scale
        score = content
        if not a.sel_content_only:
            pos = torch.arange(q.shape[2], device=q.device).float() + self.pos
            center = (pool["ids"].float() + 0.5) * self.s
            dist = (pos[None, None, :, None] - center[:, :, None]).abs()
            gate = torch.erf(dist / (2 * a.sigma.float()[None, :, None, None]))
            score = content + a.sel_scale.float()[None, :, None, None] * pool["priority"][:, :, None] * gate
        score = score.masked_fill(~pool["valid"][:, :, None], -torch.inf)
        ids = pool["ids"][:, :, None].expand_as(score)
        ids = ids.masked_fill(~pool["valid"][:, :, None], -1)
        slots = stable_topk_slots(score, ids, min(a.cell_topk, self.eligible))
        return slots, ids.gather(-1, slots)

    def _long_range(self, q):
        if self.pool is None:
            return torch.zeros_like(q)
        a, pool = self.attn, self.pool
        slots, ids = self.select(q)
        b, h, n, _ = q.shape
        c = pool["ids"].shape[-1]
        k = pool["k"].flatten(2, 3)
        v = pool["v"].flatten(2, 3)
        if q.is_cuda and a._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0):
            virtual_pos = ((c * self.s + self.Bq - 1) // self.Bq) * self.Bq
            selected = slots.to(torch.int32).masked_fill(ids < 0, -1).contiguous()
            if k.shape[2] < n:
                k = F.pad(k, (0, 0, 0, n - k.shape[2]))
                v = F.pad(v, (0, 0, 0, n - v.shape[2]))
            return a._cell_fused_lr(q.contiguous(), k, v, selected,
                                   self.s, self.Bq, q_pos=virtual_pos)
        mask = torch.zeros(b, h, n, c, dtype=torch.bool, device=q.device)
        mask.scatter_(-1, slots, ids >= 0)
        mask = mask.repeat_interleave(self.s, dim=-1)
        scores = torch.einsum("bhqd,bhjd->bhqj", q, k).float() * a.scale
        weights = scores.masked_fill(~mask, -1e9).softmax(-1).to(v.dtype)
        out = torch.einsum("bhqj,bhjd->bhqd", weights, v)
        out = out * mask.any(-1, keepdim=True).to(out.dtype)
        return a.lr_norm(out.to(q.dtype))

    def attend(self, q, k, v):
        if torch.is_grad_enabled():
            raise RuntimeError("LinearFSCache is inference-only; use torch.no_grad().")
        outputs = []
        offset = 0
        while offset < q.shape[2]:
            self.prepare()
            n = min(q.shape[2] - offset, self.Bq - self.pos % self.Bq)
            qc, kc, vc = (x[:, :, offset:offset + n] for x in (q, k, v))
            sr = self._short_range(qc, kc, vc)
            lr = self._long_range(qc)
            outputs.append(sr + self.attn.long_range_scale.unsqueeze(0) * lr)
            self.ingest(kc, vc)
            offset += n
        return torch.cat(outputs, dim=2)

    def fork(self):
        result = object.__new__(type(self))
        result.__dict__ = {name: (value if name == "attn" else _clone_tree(value))
                           for name, value in self.__dict__.items()}
        return result

    def state_bytes(self):
        seen = set()
        total = 0

        def visit(value):
            nonlocal total
            if torch.is_tensor(value):
                storage = value.untyped_storage()
                key = (value.device, storage.data_ptr())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)

        for name, value in self.__dict__.items():
            if name != "attn":
                visit(value)
        return total
