from __future__ import annotations
import os
import torch

from longctx_infer import _as_2d_long, _resolve_device, _autocast_ctx, _lm_head_fp32
from src.fsattn import kv_int8 as _kvi


def _fast_cell_stats(K, ncell, s):
    if not K.is_cuda:
        return None
    try:
        from src.fsattn.fsl_select_cuda import cache_cell_stats
    except Exception:
        return None
    return cache_cell_stats(K, ncell, s)


class FSKVCache:
    def __init__(self, attn, lean_prefill=False, retain_kv: bool = True):
        attn._assert_cacheable(phase="prefill")
        self.attn = attn
        self.s = attn.cell_stride
        self.W = attn.window_size
        self.pos = 0
        self.lean_prefill = lean_prefill
        self.retain_kv = bool(retain_kv)
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
        self._int8_ok = False               # FSBGraphedDecoder flips this on via enable_int8 (FS_KV_INT8=1)
        self.use_int8 = False
        self._qk_buf = None; self._qv_buf = None     # int8 (B,H,cap,D), completed cells
        self._ks_buf = None; self._vs_buf = None     # bf16 scales (B,H,cap//s)
        self._k_hot = None; self._v_hot = None       # bf16 (B,H,s,D), the open cell
        self._n_q = 0                                # quantized rows ([0, _n_q) is the int8 region)
        self._nq_t = None                            # device int64 scalar holding _n_q, read inside decode graphs
        self._deq = None                             # eager materialization scratch (built lazily)

    def enable_int8(self):
        self._int8_ok = True

    def row_capacity(self) -> int:
        if self.use_int8:
            return 0 if self._qk_buf is None else self._qk_buf.shape[2]
        return 0 if self._k_buf is None else self._k_buf.shape[2]

    def graph_kv(self):
        # the read state a captured decode step gathers KV through: (mode, args); mode "bf16" gives (K, V)
        if self.use_int8:
            return ("int8", (self._qk_buf, self._qv_buf, self._ks_buf, self._vs_buf,
                             self._k_hot, self._v_hot, self._nq_t, self.s))
        if self._k_buf is not None:
            return ("bf16", (self._k_buf, self._v_buf))
        return (None, None)

    def rerank_keys(self):
        # the tokK the rerank scores read token keys from: the int8 read state in int8 mode, else the bf16 history
        if self.use_int8:
            return _kvi.Int8Keys(self._qk_buf, self._ks_buf, self._k_hot, self._nq_t, self.s)
        return self._k_buf if self._k_buf is not None else self.lr_tokens()[0]

    def read_full_kv(self):
        # bf16 (K, V) over the whole stored history: the buffers themselves, or materialized in int8 mode
        if self.use_int8:
            return self.lr_tokens()
        if self._k_buf is not None:
            return self._k_buf, self._v_buf
        return self.lr_tokens()

    def _grow(self, k_t, v_t):
        B, H, _, D = k_t.shape
        if self.use_int8:
            off = self._len - self._n_q
            self._k_hot[:, :, off:off + 1] = k_t
            self._v_hot[:, :, off:off + 1] = v_t
            self._len += 1
            return
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
        max_cells = cap // self.s
        kbar = torch.empty(B, H, max_cells, D, dtype=torch.float32, device=device) if max_cells > 0 else None
        ceil = torch.empty(B, H, max_cells, dtype=torch.float32, device=device) if max_cells > 0 else None
        if self._int8_ok and _kvi.int8_supported(dtype, device):
            qk = torch.zeros(B, H, cap, D, dtype=torch.int8, device=device)
            qv = torch.zeros(B, H, cap, D, dtype=torch.int8, device=device)
            sk = torch.empty(B, H, max_cells, dtype=torch.bfloat16, device=device) if max_cells > 0 else None
            sv = torch.empty(B, H, max_cells, dtype=torch.bfloat16, device=device) if max_cells > 0 else None
            kh = torch.zeros(B, H, self.s, D, dtype=dtype, device=device)
            vh = torch.zeros(B, H, self.s, D, dtype=dtype, device=device)
            self._prealloc = (B, N, (qk, qv, sk, sv, kh, vh), kbar, ceil)
            self._keep_prealloc = bool(keep)
            return
        kb = torch.zeros(B, H, cap, D, dtype=dtype, device=device)
        vb = torch.zeros(B, H, cap, D, dtype=dtype, device=device)
        self._prealloc = (B, N, kb, vb, kbar, ceil)
        self._keep_prealloc = bool(keep)

    def prefill(self, K, V, cell_stats=None):
        # cell_stats (or self._prefill_cell_stats, set by the attention's prefill): (K, Kbar, ceil) from the layer's
        # long-range prefill, computed by the same ops on this K view. (A plain attribute, reset here: torch.compile
        # traces this method, and dynamo cannot trace a pop from __dict__.)
        handed, self._prefill_cell_stats = getattr(self, "_prefill_cell_stats", None), None
        if not getattr(self, "retain_kv", True):
            if not getattr(self, "_keep_prealloc", False):
                self._prealloc = None
            self._len = K.shape[2]
            self.pos = K.shape[2]
            return
        cell_stats = handed if cell_stats is None else cell_stats
        B, H, N, D = K.shape
        cap = N + max(0, int(self.reserve))
        pre = self._prealloc
        if not getattr(self, "_keep_prealloc", False):
            self._prealloc = None
        self.use_int8 = bool(self._int8_ok and _kvi.int8_supported(K.dtype, K.device)
                             and K.stride(-1) == 1 and V.stride(-1) == 1)
        pre_i8 = isinstance(pre[2], tuple) if pre is not None else False
        if pre is not None and pre_i8 != self.use_int8:
            pre, pre_i8 = None, False
        if pre_i8 and not (pre[0] == B and pre[1] == N and tuple(pre[2][0].shape) == (B, H, cap, D)
                           and pre[2][0].device == K.device):
            pre, pre_i8 = None, False
        if pre is not None and not pre_i8 and not (pre[0] == B and pre[1] == N and tuple(pre[2].shape) == (B, H, cap, D)
                                                   and pre[2].dtype == K.dtype and pre[2].device == K.device):
            pre = None
        ncell = N // self.s
        if self.use_int8:
            if pre is not None:
                (self._qk_buf, self._qv_buf, self._ks_buf, self._vs_buf, self._k_hot, self._v_hot) = pre[2]
            else:
                self._qk_buf = K.new_zeros((B, H, cap, D), dtype=torch.int8)
                self._qv_buf = K.new_zeros((B, H, cap, D), dtype=torch.int8)
                mc = cap // self.s
                self._ks_buf = torch.empty(B, H, mc, dtype=torch.bfloat16, device=K.device) if mc > 0 else None
                self._vs_buf = torch.empty(B, H, mc, dtype=torch.bfloat16, device=K.device) if mc > 0 else None
                self._k_hot = K.new_zeros(B, H, self.s, D)
                self._v_hot = V.new_zeros(B, H, self.s, D)
            n_q = ncell * self.s
            if n_q > 0:
                _kvi.quantize_cells_fast(K[:, :, :n_q], self.s, self._qk_buf[:, :, :n_q],
                                         self._ks_buf[:, :, :ncell])
                _kvi.quantize_cells_fast(V[:, :, :n_q], self.s, self._qv_buf[:, :, :n_q],
                                         self._vs_buf[:, :, :ncell])
            self._n_q = n_q
            self._nq_t = torch.full((), n_q, dtype=torch.int64, device=K.device)
            self._len = N
            self._k_buf = None; self._v_buf = None
            self._k_hist = None; self._v_hist = None
            self._kbar_buf = pre[3] if pre is not None else None
            self._ceil_buf = pre[4] if pre is not None else None
        elif cap > N or pre is not None:
            if pre is not None:
                self._k_buf, self._v_buf = pre[2], pre[3]
            else:
                self._k_buf = K.new_zeros(B, H, cap, D); self._v_buf = V.new_zeros(B, H, cap, D)
            if self._k_buf.data_ptr() != K.data_ptr():
                self._k_buf[:, :, :N] = K
            if self._v_buf.data_ptr() != V.data_ptr():
                self._v_buf[:, :, :N] = V
            self._len = N
            self._k_hist = self._k_buf[:, :, :N]; self._v_hist = self._v_buf[:, :, :N]
            self._kbar_buf = pre[4] if pre is not None else None
            self._ceil_buf = pre[5] if pre is not None else None
        else:
            self._k_buf = K.contiguous(); self._v_buf = V.contiguous()
            self._len = N
            self._k_hist = self._k_buf[:, :, :N]; self._v_hist = self._v_buf[:, :, :N]
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
            if self.use_int8:
                self._k_hot[:, :, :r].copy_(K[:, :, ncell * self.s:N, :])
                self._v_hot[:, :, :r].copy_(V[:, :, ncell * self.s:N, :])
                self._cell_k = self._k_hot[:, :, :r]
            else:
                self._cell_k = K[:, :, ncell * self.s:N, :].contiguous()
        self.pos = N

    def _finish_cell(self, cell, v_cell=None):
        if self.use_int8:
            k_cell, v_cell = self._k_hot, self._v_hot
            st = None if self.fsb_leaf_radius else _fast_cell_stats(k_cell, 1, self.s)
            if st is None:
                cell = k_cell
            idx = self._n_q // self.s
            c0 = self._n_q
            _kvi.quantize_cells_fast(k_cell, self.s, self._qk_buf[:, :, c0:c0 + self.s],
                                     self._ks_buf[:, :, idx:idx + 1])
            _kvi.quantize_cells_fast(v_cell, self.s, self._qv_buf[:, :, c0:c0 + self.s],
                                     self._vs_buf[:, :, idx:idx + 1])
            self._n_q = c0 + self.s
            self._nq_t.fill_(self._n_q)
        else:
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
            self._cell_k = self._k_hot[:, :, self._n_q:n] if self.use_int8 else self._k_buf[:, :, n - rem:n]
        else:
            self._finish_cell(self._k_hot if self.use_int8 else self._k_buf[:, :, n - self.s:n])
        self.pos += 1

    def advance_in_place(self):
        n = self._len + 1
        self._len = n
        self.pos = n
        if n % self.s == 0:
            if self.use_int8:
                self._finish_cell(self._k_hot)
            else:
                self._k_hist = self._k_buf[:, :, :n]
                self._v_hist = self._v_buf[:, :, :n]
                self._finish_cell(self._k_buf[:, :, n - self.s:n])

    def _materialize(self):
        # int8 mode: dequantize the completed cells and append the open cell, into a persistent bf16 scratch
        n = self._len
        n_q = self._n_q
        if self._deq is None or self._deq[0].shape[2] < n:
            B, H, _, D = self._qk_buf.shape
            cap = max(n, self._qk_buf.shape[2])
            dt, dev = self._k_hot.dtype, self._qk_buf.device
            self._deq = (torch.empty(B, H, cap, D, dtype=dt, device=dev),
                         torch.empty(B, H, cap, D, dtype=dt, device=dev))
        dk, dv = self._deq
        if n_q > 0:
            _kvi.dequant_cells(self._qk_buf[:, :, :n_q], self._ks_buf, self.s, out=dk)
            _kvi.dequant_cells(self._qv_buf[:, :, :n_q], self._vs_buf, self.s, out=dv)
        if n > n_q:
            dk[:, :, n_q:n] = self._k_hot[:, :, :n - n_q]
            dv[:, :, n_q:n] = self._v_hot[:, :, :n - n_q]
        return dk[:, :, :n], dv[:, :, :n]

    def sr_window(self, at=None):
        if self.use_int8:
            kh, vh = self._materialize()
        else:
            if self._k_hist is not None and self._k_hist.shape[2] != self._len:
                self._k_hist = self._k_buf[:, :, :self._len]
                self._v_hist = self._v_buf[:, :, :self._len]
            kh, vh = self._k_hist, self._v_hist
        end = (self.pos if at is None else at + 1)
        L = min(self.W + 1, end)
        k = kh[:, :, end - L:end, :]
        v = vh[:, :, end - L:end, :]
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
        if self.use_int8:
            return self._materialize()
        if self._k_hist is not None and self._k_hist.shape[2] != self._len:
            self._k_hist = self._k_buf[:, :, :self._len]
            self._v_hist = self._v_buf[:, :, :self._len]
        return self._k_hist, self._v_hist


def make_caches(model, *, lean_prefill=False, reserve=0, linear=None, retain_kv: bool = True) -> list:
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
            c = FSKVCache(blk.attn.fsattn, lean_prefill=lean_prefill, retain_kv=retain_kv)
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
