from __future__ import annotations

import torch

from decode_graph import GraphedDecoder
from kv_cache import make_caches, linear_prefill
from longctx_infer import _lm_head_fp32
from src.fsattn.prefix_topp import stable_topk_slots


def _rotate(attn, x, positions):
    phase = positions[:, None].to(attn.rope.inv_freq.dtype) * attn.rope.inv_freq[None]
    cos, sin = phase.cos()[None, None], phase.sin()[None, None]
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def graph_attention(attn, x, buf):
    b = x.shape[0]
    h, d = attn.n_heads, attn.head_dim
    q, k, v = (projection(x).view(b, 1, h, d).transpose(1, 2)
               for projection in (attn.W_q, attn.W_k, attn.W_v))
    pos = buf["pos_t"]
    slot = (pos % buf["kbuf"].shape[2]).reshape(1)
    buf["kbuf"].index_copy_(2, slot, k)
    buf["vbuf"].index_copy_(2, slot, v)
    kw = buf["kbuf"].index_select(2, buf["win_idx"])
    vw = buf["vbuf"].index_select(2, buf["win_idx"])
    qs = _rotate(attn, q, pos.reshape(1)) if attn.use_rope else q
    ks = _rotate(attn, kw, buf["win_pos"]) if attn.use_rope else kw
    scores = torch.einsum("bhd,bhld->bhl", qs[:, :, 0], ks).float() * attn.scale
    if not attn.fs_uniform_sr:
        dist = pos.float() - buf["win_pos"]
        bias = torch.erfc(dist.abs()[None] / (2 * attn.sigma.float()[:, None])).clamp_min(1e-10).log()
        scores = scores + bias[None]
    weights = scores.softmax(-1).to(vw.dtype)
    sr = torch.einsum("bhl,bhld->bhd", weights, vw)[:, :, None]

    pool = buf["pool"]
    content = torch.einsum("bhqd,bhcd->bhqc", q.float(), pool["mean"]) * attn.scale
    score = content
    if not attn.sel_content_only:
        center = (pool["ids"].float() + 0.5) * attn.cell_stride
        gate = torch.erf((pos.float() - center).abs() / (2 * attn.sigma.float()[None, :, None]))
        score = content + attn.sel_scale.float()[None, :, None, None] * pool["priority"][:, :, None] * gate[:, :, None]
    score = score.masked_fill(~pool["valid"][:, :, None], -torch.inf)
    ids = pool["ids"].masked_fill(~pool["valid"], -1)[:, :, None]
    chosen = stable_topk_slots(score, ids, min(attn.cell_topk, pool["C"]))
    good = ids.expand_as(score).gather(-1, chosen) >= 0
    if attn._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0):
        selected = chosen.to(torch.int32).masked_fill(~good, -1).contiguous()
        virtual = ((pool["C"] * attn.cell_stride + attn._cell_block - 1) // attn._cell_block) * attn._cell_block
        lr = attn._cell_fused_lr(q.contiguous(), pool["k"], pool["v"], selected,
                                attn.cell_stride, attn._cell_block, q_pos=virtual)
    else:
        keep = torch.zeros_like(score, dtype=torch.bool).scatter(-1, chosen, good)
        keep = keep.repeat_interleave(attn.cell_stride, dim=-1)
        scores = torch.einsum("bhqd,bhjd->bhqj", q, pool["k"]).float() * attn.scale
        weights = scores.masked_fill(~keep, -1e9).softmax(-1).to(pool["v"].dtype)
        lr = torch.einsum("bhqj,bhjd->bhqd", weights, pool["v"])
        lr = attn.lr_norm(lr * keep.any(-1, keepdim=True).to(lr.dtype))
    out = sr + attn.long_range_scale[None] * lr
    return attn.W_o(out.transpose(1, 2).reshape(b, 1, attn.d_model))


class LinearGraphedDecoder(GraphedDecoder):
    def __init__(self, model, prompt_ids, Lmax, device="cuda", dtype=torch.bfloat16):
        if prompt_ids.ndim > 2 or (prompt_ids.ndim == 2 and prompt_ids.shape[0] != 1):
            raise ValueError("LinearGraphedDecoder supports one prompt at a time.")
        super().__init__(model, prompt_ids, Lmax, device=device, dtype=dtype)

    def _alloc(self, prompt_length):
        if prompt_length < self.W:
            raise ValueError("Graph decode requires a full local window; use eager for short prompts.")
        b, h, d = 1, self.H, self.D
        self.pos_t = torch.zeros((), dtype=torch.long, device=self.dev)
        self.tok_static = torch.zeros((b, 1), dtype=torch.long, device=self.dev)
        self.win_idx = torch.zeros(self.W + 1, dtype=torch.long, device=self.dev)
        self.win_pos = torch.zeros(self.W + 1, dtype=torch.float32, device=self.dev)
        self.layer_bufs = []
        for a in self.attn:
            count = a._prune_cfg("cell_prune_pool", "FS_PRUNE_POOL", 64) + a._prune_cfg("cell_prune_window", "FS_PRUNE_WINDOW", 8)
            pool = dict(
                C=count, empty=False,
                ids=torch.full((b, h, count), -1, dtype=torch.long, device=self.dev),
                valid=torch.zeros(b, h, count, dtype=torch.bool, device=self.dev),
                priority=torch.zeros(b, h, count, dtype=torch.float32, device=self.dev),
                mean=torch.zeros(b, h, count, d, dtype=torch.float32, device=self.dev),
                k=torch.zeros(b, h, count * self.s, d, dtype=self.dt, device=self.dev),
                v=torch.zeros(b, h, count * self.s, d, dtype=self.dt, device=self.dev),
            )
            self.layer_bufs.append(dict(
                kbuf=torch.zeros(b, h, self.W + 1, d, dtype=self.dt, device=self.dev),
                vbuf=torch.zeros(b, h, self.W + 1, d, dtype=self.dt, device=self.dev),
                pos_t=self.pos_t, win_idx=self.win_idx, win_pos=self.win_pos, pool=pool,
            ))

    @torch.no_grad()
    def prefill(self):
        seq = self.prompt.view(1, -1)
        self._alloc(seq.shape[1])
        self.cache = make_caches(self.m, linear=True)
        with torch.autocast("cuda", dtype=self.dt, enabled=self.dt != torch.float32):
            hidden = linear_prefill(self.m, seq, self.cache)
            logits = _lm_head_fp32(self.m, hidden, 4096)[:, 0]
        self.next_tok = logits.argmax(-1)
        self.pos = seq.shape[1]
        for cache, buf in zip(self.cache, self.layer_bufs):
            n = cache._sr_k.shape[2]
            slots = torch.arange(self.pos - n, self.pos, device=self.dev) % (self.W + 1)
            buf["kbuf"].index_copy_(2, slots, cache._sr_k.to(self.dt))
            buf["vbuf"].index_copy_(2, slots, cache._sr_v.to(self.dt))
        self._refresh_block_pool()
        self._refresh_window()
        self.prompt = None

    @torch.no_grad()
    def _refresh_block_pool(self):
        self.cur_blk = self.pos // self.Bq * self.Bq
        for cache, buf in zip(self.cache, self.layer_bufs):
            cache.prepare()
            target = buf["pool"]
            target["ids"].fill_(-1)
            target["valid"].zero_()
            if cache.pool is None:
                continue
            source = cache.pool
            count = source["ids"].shape[-1]
            for name in ("ids", "valid", "priority", "mean"):
                target[name][:, :, :count].copy_(source[name])
            for name in ("k", "v"):
                target[name][:, :, :count * self.s].copy_(source[name].flatten(2, 3))

    @torch.no_grad()
    def _refresh_window(self):
        positions = torch.arange(self.pos - self.W, self.pos + 1, device=self.dev)
        self.win_idx.copy_(positions % (self.W + 1))
        self.win_pos.copy_(positions)
        self.pos_t.fill_(self.pos)

    def _trunk_graph(self):
        x = self.m.drop(self.m.tok_emb(self.tok_static))
        for block, attn, buf in zip(self.blocks, self.attn, self.layer_bufs):
            x = x + graph_attention(attn, block.ln1(x), buf)
            x = x + block.ffn(block.ln2(x))
        return _lm_head_fp32(self.m, self.m.ln_final(x), 4096)

    @torch.no_grad()
    def _advance_cache_bookkeeping(self):
        slot = self.pos % (self.W + 1)
        for cache, buf in zip(self.cache, self.layer_bufs):
            cache.ingest(buf["kbuf"][:, :, slot:slot + 1], buf["vbuf"][:, :, slot:slot + 1])
