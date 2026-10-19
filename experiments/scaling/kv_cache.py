from __future__ import annotations
import os
import torch

from longctx_infer import _as_2d_long, _resolve_device, _autocast_ctx, _lm_head_fp32


def _fast_cell_stats(K, ncell, s):
    if not K.is_cuda:
        return None
    try:
        from src.fsattn.fsl_select_cuda import cache_cell_stats
    except Exception:
        return None
    return cache_cell_stats(K, ncell, s)


class FSKVCache:
    def __init__(self, attn, lean_prefill=False):
        attn._assert_cacheable(phase="prefill")
        self.attn = attn
        self.s = attn.cell_stride
        self.W = attn.window_size
        self.pos = 0
        self.lean_prefill = lean_prefill
        self.reserve = 0
        self._prealloc = None
        self._k_buf = None; self._v_buf = None; self._len = 0
        self._k_hist = None
        self._v_hist = None
        self._kbar_buf = None
        self._ceil_buf = None
        self._kbar = None
        self._ceil = None
        self._cell_k = None
        self.fsb_leaf_radius = bool(getattr(attn, "fsb_leaf_radius", False))
        self._spread = None
        self._prefill_cell_stats = None     # the attention's prefill hand-off, consumed by prefill()

    def _grow(self, k_t, v_t):
        B, H, _, D = k_t.shape
        if self._k_hist is None:
            cap = 1024
            self._k_buf = k_t.new_zeros(B, H, cap, D)
            self._v_buf = v_t.new_zeros(B, H, cap, D)
            self._len = 0
        if self._len == self._k_buf.shape[2]:
            cap = self._k_buf.shape[2] * 2
            nk = self._k_buf.new_zeros(B, H, cap, D); nk[:, :, :self._len] = self._k_buf
            nv = self._v_buf.new_zeros(B, H, cap, D); nv[:, :, :self._len] = self._v_buf
            self._k_buf, self._v_buf = nk, nv
        off = self._len * self._k_buf.stride(2) * self._k_buf.element_size()
        if k_t.data_ptr() != self._k_buf.data_ptr() + off:
            self._k_buf[:, :, self._len:self._len + 1] = k_t
        if v_t.data_ptr() != self._v_buf.data_ptr() + off:
            self._v_buf[:, :, self._len:self._len + 1] = v_t
        self._len += 1
        self._k_hist = self._k_buf[:, :, :self._len]
        self._v_hist = self._v_buf[:, :, :self._len]

    def preallocate(self, B, N, dtype, device, keep=False):
        cap = N + max(0, int(self.reserve))
        if cap < N or (cap == N and not keep):
            return
        H, D = self.attn.n_heads, self.attn.head_dim
        kb = torch.zeros(B, H, cap, D, dtype=dtype, device=device)
        vb = torch.zeros(B, H, cap, D, dtype=dtype, device=device)
        max_cells = cap // self.s
        kbar = torch.empty(B, H, max_cells, D, dtype=torch.float32, device=device) if max_cells > 0 else None
        ceil = torch.empty(B, H, max_cells, dtype=torch.float32, device=device) if max_cells > 0 else None
        self._prealloc = (B, N, kb, vb, kbar, ceil)
        self._keep_prealloc = bool(keep)

    def prefill(self, K, V, cell_stats=None):
        # cell_stats (or self._prefill_cell_stats, set by the attention's prefill): (K, Kbar, ceil) from the layer's
        # long-range prefill, computed by the same ops on this K view. (A plain attribute, reset here: torch.compile
        # traces this method, and dynamo cannot trace a pop from __dict__.)
        handed, self._prefill_cell_stats = getattr(self, "_prefill_cell_stats", None), None
        cell_stats = handed if cell_stats is None else cell_stats
        B, H, N, D = K.shape
        cap = N + max(0, int(self.reserve))
        pre = self._prealloc
        if not getattr(self, "_keep_prealloc", False):
            self._prealloc = None
        if pre is not None and not (pre[0] == B and pre[1] == N and tuple(pre[2].shape) == (B, H, cap, D)
                                    and pre[2].dtype == K.dtype and pre[2].device == K.device):
            pre = None
        if cap > N or pre is not None:
            if pre is not None:
                self._k_buf, self._v_buf = pre[2], pre[3]
            else:
                self._k_buf = K.new_zeros(B, H, cap, D); self._v_buf = V.new_zeros(B, H, cap, D)
            if self._k_buf.data_ptr() != K.data_ptr():
                self._k_buf[:, :, :N] = K
            if self._v_buf.data_ptr() != V.data_ptr():
                self._v_buf[:, :, :N] = V
        else:
            self._k_buf = K.contiguous(); self._v_buf = V.contiguous()
        self._len = N
        self._k_hist = self._k_buf[:, :, :N]; self._v_hist = self._v_buf[:, :, :N]
        ncell = N // self.s
        self._kbar_buf = pre[4] if pre is not None else None
        self._ceil_buf = pre[5] if pre is not None else None
        if ncell > 0:
            Kc = K[:, :, :ncell * self.s, :].reshape(B, H, ncell, self.s, D)
            # same bits three ways, cheapest first: the statistics the layer's long-range prefill handed over, one
            # CUDA pass (FSL_FAST_CACHE_STATS), the torch ops
            if (cell_stats is not None and cell_stats[0] is K and tuple(cell_stats[1].shape) == (B, H, ncell, D)
                    and tuple(cell_stats[2].shape) == (B, H, ncell)):
                st = (cell_stats[1], cell_stats[2])
            else:
                st = _fast_cell_stats(K, ncell, self.s)
            if st is not None:
                self._kbar, self._ceil = st
            else:
                self._kbar = Kc.mean(dim=3).float()
            if st is None:
                self._ceil = Kc.float().norm(dim=-1).amax(dim=-1)
            if self._kbar_buf is not None and self._kbar_buf.shape[2] >= ncell:
                self._kbar_buf[:, :, :ncell].copy_(self._kbar)
                self._ceil_buf[:, :, :ncell].copy_(self._ceil)
                self._kbar = self._kbar_buf[:, :, :ncell]
                self._ceil = self._ceil_buf[:, :, :ncell]
            if self.fsb_leaf_radius:
                self._spread = self._cell_spread(Kc, self._kbar)
        r = N - ncell * self.s
        if r > 0:
            self._cell_k = K[:, :, ncell * self.s:N, :].contiguous()
        self.pos = N

    def _finish_cell(self, cell):
        st = None if self.fsb_leaf_radius else _fast_cell_stats(cell, 1, self.s)
        if st is not None:
            kbar, ceil = st
            sp = None
        else:
            cell = cell.contiguous()
            kbar = cell.mean(dim=2, keepdim=True).float()
            ceil = cell.float().norm(dim=-1).amax(dim=-1, keepdim=True)
            sp = self._cell_spread(cell[:, :, None], kbar) if self.fsb_leaf_radius else None
        idx = 0 if self._kbar is None else self._kbar.shape[2]
        if (self._kbar_buf is not None and self._ceil_buf is not None
                and idx < self._kbar_buf.shape[2] and sp is None):
            self._kbar_buf[:, :, idx:idx + 1].copy_(kbar)
            self._ceil_buf[:, :, idx:idx + 1].copy_(ceil)
            self._kbar = self._kbar_buf[:, :, :idx + 1]
            self._ceil = self._ceil_buf[:, :, :idx + 1]
        elif self._kbar is None:
            self._kbar, self._ceil, self._spread = kbar, ceil, sp
        else:
            self._kbar_buf = None
            self._ceil_buf = None
            self._kbar = torch.cat([self._kbar, kbar], dim=2)
            self._ceil = torch.cat([self._ceil, ceil], dim=2)
            if sp is not None:
                self._spread = torch.cat([self._spread, sp], dim=2)
        self._cell_k = None

    def append(self, k_t, v_t):
        self._grow(k_t, v_t)
        n = self._len
        rem = n % self.s
        if rem != 0:
            self._cell_k = self._k_buf[:, :, n - rem:n]
        else:
            self._finish_cell(self._k_buf[:, :, n - self.s:n])
        self.pos += 1

    def advance_in_place(self):
        n = self._len + 1
        self._len = n
        self.pos = n
        if n % self.s == 0:
            self._k_hist = self._k_buf[:, :, :n]
            self._v_hist = self._v_buf[:, :, :n]
            self._finish_cell(self._k_buf[:, :, n - self.s:n])

    def sr_window(self, at=None):
        if self._k_hist is not None and self._k_hist.shape[2] != self._len:
            self._k_hist = self._k_buf[:, :, :self._len]
            self._v_hist = self._v_buf[:, :, :self._len]
        end = (self.pos if at is None else at + 1)
        L = min(self.W + 1, end)
        k = self._k_hist[:, :, end - L:end, :]
        v = self._v_hist[:, :, end - L:end, :]
        p = torch.arange(end - L, end, device=k.device)
        return k, v, p

    def completed_cells(self):
        if self._kbar is None:
            return None, None, 0
        return self._kbar, self._ceil, self._kbar.shape[2]

    def completed_spread(self):
        return self._spread

    @staticmethod
    def _cell_spread(Kc, kbar):
        from src.fsattn.fsb_tree import _det_sum
        d = Kc.float() - kbar[..., None, :]
        return _det_sum(d * d, -1).amax(dim=-1).sqrt()

    def lr_tokens(self):
        if self._k_hist is not None and self._k_hist.shape[2] != self._len:
            self._k_hist = self._k_buf[:, :, :self._len]
            self._v_hist = self._v_buf[:, :, :self._len]
        return self._k_hist, self._v_hist


def make_caches(model, *, lean_prefill=False, reserve=0, linear=None) -> list:
    linear = os.environ.get("FSL_LINEAR_CACHE", "0") == "1" if linear is None else linear
    caches = []
    for blk in model.blocks:
        at = blk.attn.attn_type
        if at != "fsattn":
            raise NotImplementedError(f"KV cache for attn_type={at!r} is not implemented")
        if linear:
            from src.fsattn.linear_cache import LinearFSCache
            c = LinearFSCache(blk.attn.fsattn)
        else:
            c = FSKVCache(blk.attn.fsattn, lean_prefill=lean_prefill)
            c.reserve = reserve
        caches.append(c)
    return caches


@torch.no_grad()
def cached_generate(model, prompt_ids, max_new_tokens, device="cuda", *,
                    dtype="bfloat16", eos_token_id=None, lean_prefill=False,
                    linear=None, prefill_chunk_size=2048):
    device = _resolve_device(device)
    model.eval()
    seq = _as_2d_long(prompt_ids, device)
    B = seq.shape[0]
    dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    if (os.environ.get("FSB_DECODE_WHOLE", "1") == "1" and os.environ.get("FSB_DECODE_GRAPH", "1") != "0"
            and os.environ.get("FSB_DECODE_GRAPH_CHECK", "0") != "1" and B == 1 and device.type == "cuda"
            and not linear and max_new_tokens > 0):
        from decode_graph import FSBGraphedDecoder, fsb_graphed_generate
        if FSBGraphedDecoder.supported(model, seq.shape[1]):
            # FS-B: one CUDA graph per decode step (same operations as the loop below)
            return fsb_graphed_generate(model, seq, max_new_tokens, device=device, dtype=dt,
                                        eos_token_id=eos_token_id, lean_prefill=lean_prefill)
    cache = make_caches(model, lean_prefill=lean_prefill, reserve=max_new_tokens, linear=linear)
    if os.environ.get("KV_PREALLOC", "1") == "1" and seq.shape[1] > 1:
        kv_dt = dt if (device.type == "cuda" and dt != torch.float32) else next(model.parameters()).dtype
        for c in cache:
            if isinstance(c, FSKVCache):
                c.preallocate(B, seq.shape[1], kv_dt, device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    step_in = seq
    generated = []
    with _autocast_ctx(device, dt):
        for _ in range(max_new_tokens):
            if cache and getattr(cache[0], "bounded", False) and step_in.shape[1] > 1:
                hidden = linear_prefill(model, step_in, cache, chunk_size=prefill_chunk_size)
            else:
                hidden = _trunk_cached(model, step_in, cache)
            logits = _lm_head_fp32(model, hidden[:, -1:, :], 4096)[:, 0, :]
            nxt = logits.argmax(dim=-1)
            if eos_token_id is not None:
                nxt = torch.where(finished, torch.full_like(nxt, eos_token_id), nxt)
                finished = finished | (nxt == eos_token_id)
            generated.append(nxt)
            step_in = nxt.unsqueeze(1)
            if eos_token_id is not None and bool(finished.all()):
                break
    gen = torch.stack(generated, dim=1) if generated else seq[:, :0]
    return torch.cat([seq, gen], dim=1)


@torch.no_grad()
def linear_prefill(model, idx, cache, *, chunk_size=2048):
    if chunk_size <= 0 or idx.shape[1] == 0:
        raise ValueError("linear_prefill needs a positive chunk size and nonempty input")
    if not all(getattr(c, "bounded", False) for c in cache):
        raise ValueError("linear_prefill requires bounded caches in every layer")
    device = model.tok_emb.weight.device
    for start in range(0, idx.shape[1], chunk_size):
        hidden = _trunk_cached(model, idx[:, start:start + chunk_size].to(device), cache)
    return hidden[:, -1:].clone()


def _trunk_cached(model, idx, cache, *, last_n: int | None = None):
    x = model.drop(model.tok_emb(idx))
    for i, block in enumerate(model.blocks):
        x = block(x, cache=cache[i])
    if last_n is not None and x.shape[1] > last_n:
        x = x[:, -last_n:, :].contiguous()
    return model.ln_final(x)
