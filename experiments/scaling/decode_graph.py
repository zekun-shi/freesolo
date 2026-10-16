from __future__ import annotations

import os

import torch
from kv_cache import _trunk_cached, make_caches
from longctx_infer import _autocast_ctx, _lm_head_fp32, extend_rope_cache
from src.utils import trunk_triton as _tt


def _norm_cast(ln, x):
    if _tt.rms_norm_ok(x, ln.weight):
        return _tt.rms_norm_cast(x, ln.weight, ln.eps, _tt.autocast_dtype())
    return ln(x)


def _norm_fp32(ln, x):
    if _tt.rms_norm_ok(x, ln.weight):
        return _tt.rms_norm_cast(x, ln.weight, ln.eps, torch.float32)
    return ln(x)


class GraphedDecoder:
    def __init__(self, model, prompt_ids, Lmax, device="cuda", dtype=torch.bfloat16):
        self.m = model
        self.dev = torch.device(device)
        self.dt = dtype
        self.Lmax = int(Lmax)
        self.blocks = list(model.blocks)
        self.nL = len(self.blocks)
        self.attn = [b.attn.fsattn for b in self.blocks]
        if any(getattr(a, "fsb_enabled", False) for a in self.attn):
            raise NotImplementedError(
                "GraphedDecoder does not support FS-B: use FSBGraphedDecoder (cached_generate's default for FS-B)")
        a0 = self.attn[0]
        self.H = a0.n_heads; self.D = a0.head_dim
        self.W = a0.window_size
        self.s = a0.cell_stride
        self.Bq = a0._cell_block
        self.d_model = a0.d_model
        self.prompt = prompt_ids
        self.cache = None
        self.graph = None
        self._captured_C = None

    def _alloc(self, P):
        B = 1
        dev = self.dev
        self.pos_t = torch.zeros((), dtype=torch.int64, device=dev)
        self.tok_static = torch.zeros((B, 1), dtype=torch.int64, device=dev)
        self.win_idx = torch.zeros(self.W + 1, dtype=torch.int64, device=dev)
        self.win_pos = torch.zeros(self.W + 1, dtype=torch.float32, device=dev)
        self._ar = torch.arange(-self.W, 1, dtype=torch.int64, device=dev)
        # each layer's kbuf / vbuf is bound in prefill(): to the cache's own preallocated buffers when their
        # capacity covers Lmax (one KV copy, as the FS-B decoder does), else to fresh graph-owned buffers
        self.layer_bufs = []
        for _ in range(self.nL):
            self.layer_bufs.append({
                "pos_t": self.pos_t, "win_idx": self.win_idx, "win_pos": self.win_pos,
                "pool": {"empty": True},
            })

    @torch.no_grad()
    def prefill(self):
        m = self.m
        seq = self.prompt.to(self.dev).view(1, -1)
        P = seq.shape[1]
        extend_rope_cache(m, self.Lmax + 8)
        if os.environ.get("FSL_TRUNK_FUSED", "1") != "0":
            for blk, a in zip(self.blocks, self.attn):
                if hasattr(blk.ffn, "_ensure_gu_cache") and blk.ffn.w_gate.bias is None and blk.ffn.w_up.bias is None:
                    blk.ffn._ensure_gu_cache()
                if hasattr(a, "_ensure_qkv_cache") and a.W_q.bias is None and a.W_k.bias is None and a.W_v.bias is None:
                    a._ensure_qkv_cache()
        self._alloc(P)
        self.cache = make_caches(m, linear=False, reserve=max(0, self.Lmax - P))
        kv_dt = self.dt if self.dt != torch.float32 else next(m.parameters()).dtype
        if os.environ.get("KV_PREALLOC", "1") == "1":       # as cached_generate: buffers before the prefill peak
            for c in self.cache:
                c.preallocate(1, P, kv_dt, self.dev)
        with torch.amp.autocast("cuda", dtype=self.dt):
            hidden = _trunk_cached(m, seq, self.cache, last_n=1)
            logits = _lm_head_fp32(m, hidden[:, -1:, :], 4096)[:, 0, :]
        self.next_tok = logits.argmax(dim=-1)
        for i in range(self.nL):
            c = self.cache[i]
            buf = self.layer_bufs[i]
            if (c._k_buf is not None and c._k_buf.shape[2] >= self.Lmax and c._k_buf.dtype == self.dt
                    and c._k_buf.shape[3] == self.D):
                buf["kbuf"], buf["vbuf"] = c._k_buf, c._v_buf
            else:
                kbuf = torch.zeros(1, self.H, self.Lmax, self.D, dtype=self.dt, device=self.dev)
                vbuf = torch.zeros(1, self.H, self.Lmax, self.D, dtype=self.dt, device=self.dev)
                Kh, Vh = c.lr_tokens()
                n = Kh.shape[2]
                kbuf[:, :, :n, :].copy_(Kh.to(self.dt))
                vbuf[:, :, :n, :].copy_(Vh.to(self.dt))
                buf["kbuf"], buf["vbuf"] = kbuf, vbuf
        self.pos = P
        self._refresh_block_pool()
        self._refresh_window()

    @torch.no_grad()
    def _refresh_block_pool(self):
        from src.fsattn import cell_select_triton as _cst
        blk_start = (self.pos // self.Bq) * self.Bq
        self.cur_blk = blk_start
        for i in range(self.nL):
            attn = self.attn[i]
            buf = self.layer_bufs[i]
            need_tok = (_cst.triton is None) or (attn.cell_topk > 16)
            sp = buf.get("pool")
            if attn._is_dense_lr() and not need_tok:
                pc = attn._build_dense_decode_pool(self.cache[i], blk_start, self.s, self.dev, out=sp)
            else:
                pc = attn._build_decode_pool(self.cache[i], blk_start, self.s, self.dev,
                                             need_tokens=need_tok, out=sp if not need_tok else None)
            if pc.get("empty", False):
                buf["pool"] = {"empty": True}
                continue
            if pc is sp:
                continue
            if pc.get("dense", False):
                buf["pool"] = pc
                continue
            C = pc["C"]
            if (not sp) or sp.get("empty", True) or sp.get("dense", False) or sp.get("C") != C or ("tok_idx" in pc) != ("tok_idx" in sp):
                sp = {
                    "empty": False, "M_vis": pc["M_vis"], "C": C,
                    "Kbar_cand": torch.zeros_like(pc["Kbar_cand"]),
                    "ceil_cand": torch.zeros_like(pc["ceil_cand"]),
                    "cen_cand": torch.zeros_like(pc["cen_cand"]),
                    "ok_cell": torch.zeros_like(pc["ok_cell"]),
                    "cand_c": torch.zeros_like(pc["cand_c"]),
                }
                if "tok_idx" in pc:
                    sp["tok_idx"] = torch.zeros_like(pc["tok_idx"])
                buf["pool"] = sp
            sp["M_vis"] = pc["M_vis"]
            sp["Kbar_cand"].copy_(pc["Kbar_cand"]); sp["ceil_cand"].copy_(pc["ceil_cand"])
            sp["cen_cand"].copy_(pc["cen_cand"]); sp["ok_cell"].copy_(pc["ok_cell"])
            sp["cand_c"].copy_(pc["cand_c"])
            if "tok_idx" in pc:
                sp["tok_idx"].copy_(pc["tok_idx"])

    @torch.no_grad()
    def _refresh_window(self):
        pos = self.pos
        idx = self._ar + pos
        self.win_idx.copy_(idx)
        self.win_pos.copy_(idx)
        self.pos_t.fill_(pos)

    def _trunk_graph(self):
        m = self.m
        x = m.drop(m.tok_emb(self.tok_static))
        for i, block in enumerate(self.blocks):
            attn = self.attn[i]; buf = self.layer_bufs[i]
            x = x + attn.graph_decode_layer(_norm_cast(block.ln1, x), buf)
            x = x + block.ffn(_norm_cast(block.ln2, x))
        x = _norm_fp32(m.ln_final, x)
        return _lm_head_fp32(m, x, 4096)         # the cached_generate head, as in FSBGraphedDecoder

    def _pool_cap_key(self):
        p0 = self.layer_bufs[0]["pool"]
        if p0.get("dense", False):
            return ("dense", p0.get("C"), p0["Kbar_all"].data_ptr(), min(self.attn[0].cell_topk, p0["M_vis"]))
        c = p0.get("C")
        if "M_vis" in p0 and c is not None:
            return (c, min(self.attn[0].cell_topk, p0["M_vis"], c))
        return c

    @torch.no_grad()
    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        for blk in self.blocks:
            blk.ffn._use_gu_cache = True
        try:
            _tt._GRAPH_WARMUP = True
            try:
                with torch.cuda.stream(s), torch.amp.autocast("cuda", dtype=self.dt):
                    for _ in range(3):
                        _ = self._trunk_graph()
            finally:
                _tt._GRAPH_WARMUP = False
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.amp.autocast("cuda", dtype=self.dt):
                self.static_logits = self._trunk_graph()
        finally:
            for blk in self.blocks:
                blk.ffn._use_gu_cache = False
        self._captured_C = self._pool_cap_key()

    @torch.no_grad()
    def step(self):
        self.tok_static.copy_(self.next_tok.view(1, 1))
        self._refresh_window()
        if self.pos % self.Bq == 0:
            self._refresh_block_pool()
            if (self._captured_C is None) or (self._pool_cap_key() != self._captured_C):
                self._capture()
        if self.graph is None:
            self._capture()
        self.graph.replay()
        logits = self.static_logits[:, 0, :].float()
        self.next_tok = logits.argmax(dim=-1)
        self._advance_cache_bookkeeping()
        self.pos += 1
        return self.next_tok

    @torch.no_grad()
    def _advance_cache_bookkeeping(self):
        p = self.pos
        for i in range(self.nL):
            c = self.cache[i]
            buf = self.layer_bufs[i]
            if buf["kbuf"] is c._k_buf and buf["vbuf"] is c._v_buf:
                c.advance_in_place()
            else:
                k_t = buf["kbuf"][:, :, p:p + 1, :]
                v_t = buf["vbuf"][:, :, p:p + 1, :]
                c.append(k_t, v_t)


class FSBGraphedDecoder:
    """Whole-step CUDA-graph decode for FS-B: every layer's token step (projections, window, FS-B read, FFN) and the
    fp32 head replay as one graph. At each query block the forests are brought up to date eagerly
    (fsb_decode_prepare); the step graph is re-captured only when the tensors it reads moved (arena growth, a live
    flip, a path switch): the prepare refreshes its forest slot tensors in place, so one capture serves every block
    whose registry is unchanged. The KV cache's cell bookkeeping runs after each replay. The operations are those
    of cached_generate's decode, so the tokens match it."""

    def __init__(self, model, prompt_ids, max_new_tokens, device="cuda", dtype=torch.bfloat16, lean_prefill=False):
        self.m = model
        self.dev = torch.device(device)
        self.dt = dtype
        self.blocks = list(model.blocks)
        self.attn = [b.attn.fsattn for b in self.blocks]
        a0 = self.attn[0]
        self.H, self.D, self.W, self.Bq = a0.n_heads, a0.head_dim, a0.window_size, a0._cell_block
        self.prompt = prompt_ids
        self.max_new = int(max_new_tokens)
        self.lean_prefill = lean_prefill
        self.graph = None
        self._reg = None

    def _registry(self):
        from src.fsattn.fsb_tree import fsb_graph_registry
        return tuple(fsb_graph_registry(a, c) for a, c in zip(self.attn, self.cache))

    @staticmethod
    def supported(model, prompt_len) -> bool:
        a = [b.attn.fsattn for b in model.blocks]
        return (all(getattr(x, "fsb_enabled", False) and getattr(x, "fsb_tree", "forest") == "forest" for x in a)
                and torch.cuda.is_available() and prompt_len >= a[0].window_size + 1)

    @torch.no_grad()
    def prefill(self):
        m = self.m
        seq = self.prompt.to(self.dev).view(1, -1)
        P = seq.shape[1]
        # the graph indexes the RoPE table by position (the eager loop extends it on demand): cover every position now,
        # so no captured graph ever refers to a table that is rebuilt later
        extend_rope_cache(m, P + self.max_new + 8)
        if os.environ.get("FSL_TRUNK_FUSED", "1") != "0":
            for blk, a in zip(self.blocks, self.attn):
                if hasattr(blk.ffn, "_ensure_gu_cache") and blk.ffn.w_gate.bias is None and blk.ffn.w_up.bias is None:
                    blk.ffn._ensure_gu_cache()
                if hasattr(a, "_ensure_qkv_cache") and a.W_q.bias is None and a.W_k.bias is None and a.W_v.bias is None:
                    a._ensure_qkv_cache()
        self.cache = make_caches(m, lean_prefill=self.lean_prefill, reserve=self.max_new + 8, linear=False)
        for c in self.cache:
            c._fsb_graph_m_cap = (P + self.max_new + 8) // c.attn.cell_stride
        if os.environ.get("KV_PREALLOC", "1") == "1":       # as cached_generate: buffers before the prefill peak
            kv_dt = self.dt if self.dt != torch.float32 else next(m.parameters()).dtype
            for c in self.cache:
                c.preallocate(1, P, kv_dt, self.dev)
        with _autocast_ctx(self.dev, self.dt):
            hidden = _trunk_cached(m, seq, self.cache, last_n=1)
            logits = _lm_head_fp32(m, hidden[:, -1:, :], 4096)[:, 0, :]
        self.next_tok = logits.argmax(dim=-1)
        self.pos = P
        self.pos_t = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.win_idx = torch.zeros(self.W + 1, dtype=torch.int64, device=self.dev)
        self.win_pos = torch.zeros(self.W + 1, dtype=torch.float32, device=self.dev)
        self.tok_static = torch.zeros((1, 1), dtype=torch.int64, device=self.dev)
        self.buf = {"pos_t": self.pos_t, "win_idx": self.win_idx, "win_pos": self.win_pos}
        self._ar = torch.arange(-self.W, 1, dtype=torch.int64, device=self.dev)
        self._blk = None

    def _step_body(self):
        m = self.m
        x = m.drop(m.tok_emb(self.tok_static))
        for i, blk in enumerate(self.blocks):
            x = x + self.attn[i].graph_decode_layer_fsb(_norm_cast(blk.ln1, x), self.cache[i], self.buf)
            x = x + blk.ffn(_norm_cast(blk.ln2, x))
        x = _norm_fp32(m.ln_final, x)
        return _lm_head_fp32(m, x, 4096)[:, 0, :]

    @torch.no_grad()
    def _capture(self):
        from src.fsattn.fsb_tree import fsb_decode_prepare
        for a, c in zip(self.attn, self.cache):
            fsb_decode_prepare(a, c, 1, self.H, self.D, self.dev, self.pos)
        accs = [getattr(a, "_fsb_read_acc", None) for a in self.attn]
        for a in self.attn:                     # the warm-up step must not add to the read statistics
            a._fsb_read_acc = None
        for blk in self.blocks:
            blk.ffn._use_gu_cache = True
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            _tt._GRAPH_WARMUP = True
            try:
                with torch.cuda.stream(s), _autocast_ctx(self.dev, self.dt):
                    self._step_body()
            finally:
                _tt._GRAPH_WARMUP = False
            torch.cuda.current_stream().wait_stream(s)
            for a, acc in zip(self.attn, accs):
                a._fsb_read_acc = acc
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), _autocast_ctx(self.dev, self.dt):
                self.static_logits = self._step_body()
        finally:
            for blk in self.blocks:
                blk.ffn._use_gu_cache = False
        self._blk = self.pos // self.Bq
        self._reg = self._registry()

    @torch.no_grad()
    def step(self):
        p = self.pos
        self.tok_static.copy_(self.next_tok.view(1, 1))
        idx = self._ar + p
        self.win_idx.copy_(idx)
        self.win_pos.copy_(idx)
        self.pos_t.fill_(p)
        if self.graph is None:
            self._capture()                     # warms up (writes this token's K / V) and captures; same values
        elif p // self.Bq != self._blk:
            from src.fsattn.fsb_tree import fsb_decode_prepare
            for a, c in zip(self.attn, self.cache):
                fsb_decode_prepare(a, c, 1, self.H, self.D, self.dev, p)
            if self._registry() != self._reg:
                self._capture()                 # the tensors the graph reads moved: re-capture over the new state
            else:
                self._blk = p // self.Bq        # slots refreshed in place: the capture survives the block change
        self.graph.replay()
        self.next_tok = self.static_logits.argmax(dim=-1)
        for c in self.cache:                    # cell bookkeeping of the token the graph wrote at position p
            c.advance_in_place()
        self.pos = p + 1
        return self.next_tok


@torch.no_grad()
def fsb_graphed_generate(model, prompt_ids, max_new_tokens, device="cuda", dtype="bfloat16", eos_token_id=None,
                         lean_prefill=False):
    dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    seq = prompt_ids.to(device).view(1, -1)
    if max_new_tokens <= 0:
        return seq
    gd = FSBGraphedDecoder(model, seq, max_new_tokens, device=device, dtype=dt, lean_prefill=lean_prefill)
    gd.prefill()
    gen = [gd.next_tok.clone()]
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and bool((gen[-1] == eos_token_id).all()):
            break
        gen.append(gd.step().clone())
    return torch.cat([seq, torch.stack(gen, dim=1)], dim=1)


@torch.no_grad()
def graphed_generate(model, prompt_ids, max_new_tokens, device="cuda", dtype="bfloat16",
                     eos_token_id=None, linear=None):
    dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    seq = prompt_ids.to(device).view(1, -1)
    P = seq.shape[1]
    if max_new_tokens <= 0:
        return seq
    linear = os.environ.get("FSL_LINEAR_CACHE", "0") == "1" if linear is None else linear
    decoder = GraphedDecoder
    if linear:
        from linear_decode_graph import LinearGraphedDecoder
        decoder = LinearGraphedDecoder
    gd = decoder(model, seq, Lmax=P + max_new_tokens + 8, device=device, dtype=dt)
    gd.prefill()
    gen = [gd.next_tok.clone()]
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and bool((gen[-1] == eos_token_id).all()):
            break
        gen.append(gd.step().clone())
    out = torch.stack(gen, dim=1)
    return torch.cat([seq, out], dim=1)


class GraphedPrefiller:
    """Whole-forward CUDA-graph prefill for FS / FS-L / FS-B at a fixed (B, L) shape: all KV-cache buffers
    (_k_buf, _v_buf, _kbar, _ceil), the static token input (B, L) and output logits are hoisted before capture;
    one eager warm-up pass compiles/caches all kernels and task tables before stream capture."""

    def __init__(self, model, L: int, B: int = 1, device="cuda", dtype=torch.bfloat16, reserve: int = 0,
                 tail_tokens: int = 64):
        self.m = model
        self.dev = torch.device(device)
        self.dt = dtype
        self.B = int(B)
        self.L = int(L)
        self.reserve = int(reserve)
        self.tail_tokens = min(int(tail_tokens), self.L) if tail_tokens > 0 else 1
        self.graph = None
        self.x_static = None
        self.static_logits = None
        self.static_tail = None
        self.cache = None

    def _body(self):
        for c in self.cache:
            c.pos = 0
        h = _trunk_cached(self.m, self.x_static, self.cache, last_n=self.tail_tokens)
        tail = h[:, -self.tail_tokens:, :].clone()
        logits = _lm_head_fp32(self.m, h[:, -1:, :], 4096)[:, 0, :]
        return logits, tail

    @torch.no_grad()
    def capture(self, sample_ids: torch.Tensor | None = None):
        m = self.m
        extend_rope_cache(m, self.L + max(self.reserve, 256))
        self.x_static = torch.zeros((self.B, self.L), dtype=torch.int64, device=self.dev)
        if sample_ids is not None:
            self.x_static.copy_(sample_ids.to(self.dev).view(self.B, self.L))
        self.cache = make_caches(m, reserve=self.reserve, linear=False)
        kv_dt = self.dt if self.dt != torch.float32 else next(m.parameters()).dtype
        for c in self.cache:
            c.preallocate(self.B, self.L, kv_dt, self.dev, keep=True)
        accs = [getattr(b.attn.fsattn, "_fsb_read_acc", None) for b in m.blocks]
        for b in m.blocks:
            b.attn.fsattn._fsb_read_acc = None
        s = torch.cuda.Stream(device=self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s), _autocast_ctx(self.dev, self.dt):
            self._body()
        torch.cuda.current_stream(self.dev).wait_stream(s)
        del s
        for b in m.blocks:
            b.attn.fsattn._fsb_stats = None
        torch.cuda.synchronize(self.dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.dev)
        for b, acc in zip(m.blocks, accs):
            b.attn.fsattn._fsb_read_acc = acc
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), _autocast_ctx(self.dev, self.dt):
            self.static_logits, self.static_tail = self._body()
        return self

    @torch.no_grad()
    def run(self, prompt_ids: torch.Tensor):
        if self.graph is None:
            self.capture(prompt_ids)
        else:
            self.x_static.copy_(prompt_ids.to(self.dev).view(self.B, self.L))
        self.graph.replay()
        for c in self.cache:
            c.pos = self.L
        return self.static_logits

    @torch.no_grad()
    def tail_logits(self):
        with _autocast_ctx(self.dev, self.dt):
            return _lm_head_fp32(self.m, self.static_tail, 4096)

