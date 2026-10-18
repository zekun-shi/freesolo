import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..utils.compile_guard import compiling
from ..utils.local_attention import local_attention


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:, :d].unsqueeze(0).unsqueeze(0)
    sin = sin[:, :d].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


_WINDOW_STREAMS: dict = {}


def _window_stream(dev):
    s = _WINDOW_STREAMS.get(dev)
    if s is None:
        s = _WINDOW_STREAMS[dev] = torch.cuda.Stream(device=dev)
    return s


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.scale


class FSAttn(nn.Module):
    VARIANTS = ("cell_topk",)

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 128,
        sigma_init: float = 4.0,
        learnable_sigma: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
        long_range_variant: str = "cell_topk",
        use_rope: bool = False,
        rope_long_range: bool = True,
        max_seq_len: int = 8192,
        lr_scale_mode: str = "learn",
        cell_stride: int = 64,
        cell_topk: int = 1,
        cell_select: str = "score",
        **kwargs,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert long_range_variant in self.VARIANTS, (
            f"Unknown variant '{long_range_variant}', choose from {self.VARIANTS}"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.window_size = window_size
        self.causal = causal
        self.long_range_variant = long_range_variant
        self.disable_short_range = kwargs.get("disable_short_range", False)
        self.disable_long_range = kwargs.get("disable_long_range", False)
        self.fs_uniform_sr = bool(kwargs.get("fs_uniform_sr", False))
        self.sel_content_only = bool(kwargs.get("sel_content_only", False))
        self.use_rope = use_rope
        self.rope_long_range = rope_long_range
        self.cell_stride = cell_stride if cell_stride > 0 else window_size
        self.cell_topk = max(1, cell_topk)
        self._cell_block = 128
        assert cell_select in ("score",), f"bad cell_select '{cell_select}'"
        self.cell_select = cell_select
        self.cell_prune_kernel = kwargs.get("cell_prune_kernel", None)
        self.cell_prune_pool = kwargs.get("cell_prune_pool", None)
        self.cell_prune_window = kwargs.get("cell_prune_window", None)
        self.cell_prune_blkchunk = kwargs.get("cell_prune_blkchunk", None)
        self.sel_strict_causal = kwargs.get("sel_strict_causal", None)
        self.fs_fused_attn = kwargs.get("fs_fused_attn", None)
        self.cell_fused_select = kwargs.get("cell_fused_select", None)
        self.fsb_enabled = bool(kwargs.get("fsb_enabled", False))
        if self.fsb_enabled:
            assert long_range_variant == "cell_topk" and cell_select == "score", (
                "FS-B requires long_range_variant='cell_topk' and cell_select='score'"
            )
            from .fsb_tree import fsb_init
            fsb_init(self, n_heads, self.head_dim, kwargs)
        self.sel_scale = nn.Parameter(torch.ones(n_heads))

        log_sigma = torch.full((n_heads,), math.log(sigma_init))
        if learnable_sigma and not self.fs_uniform_sr:
            self.log_sigma = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigma", log_sigma)

        if lr_scale_mode == "learn":
            self.long_range_scale = nn.Parameter(torch.full((n_heads, 1, 1), 0.1))
        else:
            self.register_buffer("long_range_scale", torch.ones(n_heads, 1, 1))

        self.lr_norm = RMSNorm(self.head_dim)

        if use_rope:
            from experiments.scaling.yarn_rope import build_rope
            self.rope = build_rope(
                self.head_dim, max_seq_len,
                use_yarn=kwargs.get("use_yarn", False),
                yarn_scale=kwargs.get("yarn_scale", 1.0),
                yarn_orig_max_seq_len=kwargs.get("yarn_orig_max_seq_len", None),
                yarn_beta_fast=kwargs.get("yarn_beta_fast", 32.0),
                yarn_beta_slow=kwargs.get("yarn_beta_slow", 1.0),
            )

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        if cache is not None:
            return self._forward_cached(x, cache)
        B, N, _ = x.shape
        H, D = self.n_heads, self.head_dim

        Q = self.W_q(x).view(B, N, H, D).transpose(1, 2)
        K = self.W_k(x).view(B, N, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, D).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope(N)
            Q_rope, K_rope = self._rope_pair(Q, K, cos, sin, window_only=not getattr(self, "rope_long_range", True))
        else:
            Q_rope = Q
            K_rope = K

        sigma = self.sigma

        if self.disable_short_range:
            out_short = None
        else:
            out_short = self._short_range(Q_rope, K_rope, V, sigma)
            out_short = self.dropout(out_short)

        Q_long = Q_rope if getattr(self, "rope_long_range", True) else Q
        K_long = K_rope if (self.use_rope and getattr(self, "rope_long_range", True)
                            and self.long_range_variant == "cell_topk") else K

        if getattr(self, "recompute_long_range", True) and self.training:
            out_long = torch.utils.checkpoint.checkpoint(
                self._long_range, Q_long, K_long, V, sigma, N, use_reentrant=False,
            )
        elif self._lr_handoff_off():
            out_long = self._long_range(Q_long, K_long, V, sigma, N)
        else:
            self._begin_lr(None if self.disable_long_range else out_short)
            try:
                out_long = self._long_range(Q_long, K_long, V, sigma, N)
            finally:
                fused, _ = self._end_lr()
            if fused:
                return self.W_o(out_long)

        if self.disable_long_range:
            out = out_short
        elif out_short is None:
            out = self.long_range_scale.unsqueeze(0) * out_long
        else:
            out = out_short + self.long_range_scale.unsqueeze(0) * out_long

        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.W_o(out)

    def _short_range(self, Q, K, V, sigma):
        sr_sigma = None if self.fs_uniform_sr else sigma
        # Q and K arrive rotated, so the chunked band-mask path is the same computation as the generic loop (bit for
        # bit: tests/test_window_paths.py), with larger chunks and one mask per chunk
        return local_attention(
            Q, K, V, self.window_size, scale=self.scale, causal=self.causal, sigma=sr_sigma, fast_sliding=True,
        )

    def _short_range_prefill(self, Q, K, V, sigma):
        # The window branch of a prefill. With FSL_WINDOW_STREAM=1 it runs on a side stream, overlapping the
        # long-range branch (same kernels and inputs, so the same values); the returned event marks its completion
        # and every consumer of the output waits for it (_await_window). Not under torch.compile.
        if not (os.environ.get("FSL_WINDOW_STREAM", "0") == "1" and Q.is_cuda and not torch.is_grad_enabled()
                and not compiling() and not torch.cuda.is_current_stream_capturing()):
            return self.dropout(self._short_range(Q, K, V, sigma)), None
        side = _window_stream(Q.device)
        side.wait_stream(torch.cuda.current_stream(Q.device))
        with torch.cuda.stream(side):
            out = self.dropout(self._short_range(Q, K, V, sigma))
        for t in (Q, K, V):
            t.record_stream(side)
        ev = torch.cuda.Event()
        ev.record(side)
        return out, ev

    @staticmethod
    def _await_window(out_short, ev):
        if ev is not None:
            main = torch.cuda.current_stream(out_short.device)
            main.wait_event(ev)
            out_short.record_stream(main)

    def _begin_lr(self, out_short, want_cell_stats=False, sr_event=None, inplace_sr=False):
        # Prefill only. Hand the window output to the cell readout, which may return W_o's input with the combine
        # fused in (readout_triton; FSL_READOUT_FUSED=0 keeps the eager combine), and ask the long-range branch for
        # its cell statistics when FSKVCache.prefill would recompute exactly them. Not under torch.compile (the
        # callers run the long-range branch without it there: _lr_handoff_off).
        fuse = (out_short is not None and out_short.is_cuda and not torch.is_grad_enabled()
                and os.environ.get("FSL_READOUT_FUSED", "1") != "0")
        from . import fsb_rows as _frows
        self._lr_combine_sr = out_short if fuse else None
        self._lr_combine_ev = sr_event
        self._lr_inplace_sr = bool(inplace_sr and fuse and (not self.fsb_enabled or _frows.UCAP >= 512))
        self._lr_combined = False
        self._cell_stats_out = [] if want_cell_stats else None

    @staticmethod
    def _lr_handoff_off() -> bool:
        # Under torch.compile the long-range branch runs without the hand-off (_begin_lr / _end_lr): the compiled
        # graph keeps the eager combine and the cache's own cell statistics, as before them. And without the try
        # block around it: dynamo cannot resume inside a try block after the readout's graph break
        # (indexed_cell_attention runs outside the graph), so the whole forward would run uncompiled.
        return compiling()

    def _end_lr(self):
        fused, stats = getattr(self, "_lr_combined", False), getattr(self, "_cell_stats_out", None)
        self._lr_combine_sr = None
        self._lr_combine_ev = None
        self._lr_inplace_sr = False
        self._lr_combined = False
        self._cell_stats_out = None
        return fused, (stats[0] if stats else None)

    def _combine(self, out_short, out_long, fused, B, N, sr_event=None):
        # W_o's input: transpose(out_short + long_range_scale * out_long), unless the readout already returned it
        if fused:
            return out_long
        self._await_window(out_short, sr_event)
        out = out_short + self.long_range_scale.unsqueeze(0) * out_long
        return out.transpose(1, 2).contiguous().view(B, N, self.d_model)

    def _long_range(self, Q, K, V, sigma, seq_len):
        if self.fsb_enabled:
            from .fsb_tree import fsb_long_range
            return fsb_long_range(self, Q, K, V, sigma, seq_len)
        return self._lr_cell_topk(Q, K, V, sigma, seq_len)

    def _prune_cfg(self, attr, env, dflt):
        v = getattr(self, attr, None)
        return int(v) if v is not None else int(os.environ.get(env, str(dflt)))

    def _cell_sel_cells(self, topi, cell_end, blk_start, B, H, Npad):
        valid = cell_end[topi] <= blk_start.view(1, 1, -1, 1, 1)
        sel = topi.to(torch.int32).masked_fill(~valid, -1)
        return sel.reshape(B, H, Npad, topi.shape[-1]).contiguous()

    def _cell_fused_lr(self, Q, K, V, sel_cells, s, Bq, bias=None, q_pos=0):
        sr = getattr(self, "_lr_combine_sr", None)
        if sr is not None:
            from .readout_triton import readout_ok, readout_combine
            if readout_ok(Q, K, V, sel_cells, sr, cell_size=s, query_block=Bq, bias=bias, q_pos=q_pos):
                # the prefill caller handed over the window output (_begin_lr): return W_o's input directly
                inplace = bool(getattr(self, "_lr_inplace_sr", False))
                self._lr_combine_sr = None
                self._lr_inplace_sr = False
                self._lr_combined = True
                self._await_window(sr, getattr(self, "_lr_combine_ev", None))
                return readout_combine(Q, K, V, sel_cells, sr, self.lr_norm.scale, self.long_range_scale,
                                       self.lr_norm.eps, scale=self.scale, cell_size=s, query_block=Bq,
                                       inplace=inplace)
        from .cell_select_triton import indexed_cell_attention
        out = indexed_cell_attention(Q, K, V, sel_cells, cell_size=s, query_block=Bq, scale=self.scale,
                                     bias=bias, q_pos=q_pos)
        return self.lr_norm(out.to(Q.dtype))

    def _lr_cell_topk(self, Q, K, V, sigma, N):
        if getattr(self, "fsb_enabled", False):
            from .fsb_tree import fsb_long_range
            return fsb_long_range(self, Q, K, V, sigma, N)
        B, H, _, D = Q.shape
        s = self.cell_stride
        dev = Q.device
        _strict = bool(self._prune_cfg("sel_strict_causal", "FS_STRICT_CAUSAL", 0))
        _prune = bool(self._prune_cfg("cell_prune_kernel", "FS_PRUNE_KERNEL", 0))
        _fused = bool(self._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0))
        if _strict and _prune:
            return self._cell_score_On_chunked_strict(Q, K, V, sigma, N)
        if _prune and self._prune_cfg("cell_prune_blkchunk", "FS_PRUNE_BLKCHUNK", 0) > 0:
            _Bq = min(self._cell_block, N)
            _nb = (N + _Bq - 1) // _Bq
            if self._prune_cfg("cell_prune_blkchunk", "FS_PRUNE_BLKCHUNK", 0) < _nb:
                return self._cell_score_On_chunked(Q, K, V, sigma, N)
        M = (N + s - 1) // s
        pad = M * s - N
        kk = min(self.cell_topk, M)
        Bq = min(self._cell_block, N)
        nb = (N + Bq - 1) // Bq
        Npad = nb * Bq
        qpad = Npad - N
        _fused_select = _fused and bool(self._prune_cfg("cell_fused_select", "FS_FUSED_SELECT", 0))
        if _strict and not _prune and self.causal and _fused_select and self.cell_select == "score":
            from .fsl_select_cuda import dense_fast_select_ok, fsl_dense_fast_select
            Kp_fast = torch.cat([K, K.new_zeros(B, H, pad, D)], dim=2) if pad else K
            if dense_fast_select_ok(Q, Kp_fast, kk, Bq, s, N):
                st = [] if (getattr(self, "_cell_stats_out", None) is not None and not pad and M <= 16384) else None
                sel_full = fsl_dense_fast_select(
                    Q, Kp_fast, N=N, M=M, s=s, Bq=Bq, nb=nb, kk=kk,
                    lam=self.sel_scale, sigv=sigma, scale=self.scale,
                    content_only=self.sel_content_only, stats_out=st,
                )
                if sel_full is not None:
                    res = self._cell_fused_lr(Q, K, V, sel_full, s, Bq, bias=None)
                    del sel_full
                    if st:
                        self._cell_stats_out.append((K, st[0][0].float(), st[0][1]))
                        del st
                    return res
        Qf = Q.float()
        if pad:
            Kp = torch.cat([K, K.new_zeros(B, H, pad, D)], dim=2)
            Vp = torch.cat([V, V.new_zeros(B, H, pad, D)], dim=2)
        else:
            Kp, Vp = K, V
        Kbar = Kp.view(B, H, M, s, D).mean(dim=3).float()
        cen = (torch.arange(M, device=dev, dtype=torch.float32) + 0.5) * s
        sig = sigma.to(torch.float32).view(H, 1, 1)
        NEG = -1e9
        cell_end = (torch.arange(M, device=dev, dtype=torch.float32) + 1.0) * s
        if _strict:
            topi = self._cell_select_blk_logits(
                Qf, Kbar, Kp, cen, sig, N, M, s, B, H, D, dev, Bq, nb, Npad, cell_end, per_query=True, topk_k=kk)
            if _fused:
                blk_start = torch.arange(nb, device=dev, dtype=torch.float32) * Bq
                sel = self._cell_sel_cells(topi, cell_end, blk_start, B, H, Npad)
                return self._cell_fused_lr(Q, K, V, sel, s, Bq, bias=None)
            blk_start = torch.arange(nb, device=dev, dtype=torch.float32) * Bq
            cell_sel = torch.zeros(B, H, nb, Bq, M, dtype=torch.bool, device=dev)
            cell_sel.scatter_(-1, topi, torch.ones_like(topi, dtype=torch.bool))
            cell_ok = cell_sel & (cell_end.view(1, 1, 1, 1, M) <= blk_start.view(1, 1, nb, 1, 1))
            tok_ok = cell_ok.repeat_interleave(s, dim=-1)
            Qb = Q if qpad == 0 else torch.cat([Q, Q.new_zeros(B, H, qpad, D)], dim=2)
            Qb = Qb.view(B, H, nb, Bq, D)
            scores = torch.einsum("bhnqd,bhjd->bhnqj", Qb, Kp).float() * self.scale
            if self.causal:
                jpos = torch.arange(M * s, device=dev)
                qpos = torch.arange(Npad, device=dev).view(nb, Bq)
                tok_ok = tok_ok & (jpos.view(1, 1, 1, 1, -1) < qpos.view(1, 1, nb, Bq, 1))
            scores = scores.masked_fill(~tok_ok, NEG)
            w = torch.softmax(scores, dim=-1).to(Vp.dtype)
            ob = torch.einsum("bhnqj,bhjd->bhnqd", w, Vp)
            ob = ob * tok_ok.any(dim=-1, keepdim=True).to(ob.dtype)
            out = ob.reshape(B, H, Npad, D)
            if qpad:
                out = out[:, :, :N, :]
            return self.lr_norm(out.to(Q.dtype))
        if _prune:
            topi = self._cell_select_score_On(
                Qf, Kbar, Kp, cen, cell_end, sig, N, M, s, B, H, D, dev, Bq, nb, Npad, kk)
        else:
            blk_logits = self._cell_select_blk_logits(
                Qf, Kbar, Kp, cen, sig, N, M, s, B, H, D, dev, Bq, nb, Npad, cell_end)
            topi = blk_logits.topk(kk, dim=-1).indices
        off = torch.arange(s, device=dev)
        tok_idx = (topi.unsqueeze(-1) * s + off).reshape(B, H, nb, kk * s)
        K = kk * s
        gidx = tok_idx.reshape(B, H, nb * K, 1).expand(-1, -1, -1, D)
        Kc = torch.gather(Kp, 2, gidx).reshape(B, H, nb, K, D)
        Vc = torch.gather(Vp, 2, gidx).reshape(B, H, nb, K, D)
        Qb = Q if qpad == 0 else torch.cat([Q, Q.new_zeros(B, H, qpad, D)], dim=2)
        Qb = Qb.view(B, H, nb, Bq, D)
        scores = torch.einsum("bhcqd,bhckd->bhcqk", Qb, Kc).float() * self.scale
        if self.causal:
            qpos = torch.arange(Npad, device=dev).view(nb, Bq)
            fut = tok_idx[:, :, :, None, :] >= qpos[None, None, :, :, None]
            scores = scores.masked_fill(fut, NEG)
        w = torch.softmax(scores, dim=-1).to(Vc.dtype)
        ob = torch.einsum("bhcqk,bhckd->bhcqd", w, Vc)
        if self.causal:
            ob = ob * (~fut).any(dim=-1, keepdim=True).to(ob.dtype)
        out = ob.reshape(B, H, Npad, D)
        if qpad:
            out = out[:, :, :N, :]
        return self.lr_norm(out.to(Q.dtype))

    def _cell_select_blk_logits(self, Qf, Kbar, Kp, cen, sig, N, M, s, B, H, D, dev, Bq, nb, Npad, cell_end, per_query=False, topk_k: int = 0):
        NEG = -1e9
        sc = self.scale
        blk_vis = None
        if self.causal:
            blk_start = torch.arange(nb, device=dev, dtype=torch.float32) * Bq
            blk_vis = cell_end[None, :] <= blk_start[:, None]
        ceil_c = Kp.float().norm(dim=-1).view(B, H, M, s).amax(dim=3)
        if getattr(self, "_cell_stats_out", None) is not None and M * s == N and M <= 16384:
            self._cell_stats_out.append((Kp, Kbar, ceil_c))
        QCH = int(getattr(self, "_cell_select_chunk", 8192)) or Npad
        if per_query and topk_k > 0:
            _max_els = 128 * 1024 * 1024
            _cap_q = max(Bq, (_max_els // max(1, B * H * M) // Bq) * Bq)
            QCH = min(QCH, _cap_q)
        QCH = max(Bq, (QCH // Bq) * Bq)
        blk = None if per_query else Qf.new_full((B, H, nb, M), NEG)
        topi_pq = torch.empty((B, H, nb, Bq, topk_k), dtype=torch.int64, device=dev) if (per_query and topk_k > 0) else None
        blk_pq = Qf.new_full((B, H, nb, Bq, M), NEG) if (per_query and topk_k <= 0) else None
        for q0 in range(0, Npad, QCH):
            q1 = min(q0 + QCH, Npad)
            ch = q1 - q0
            nbc = ch // Bq
            rq1 = min(q1, N)
            cl = Qf.new_full((B, H, ch, M), NEG)
            if q0 < N:
                nr = rq1 - q0
                qf = Qf[:, :, q0:rq1, :]
                posr = torch.arange(q0, rq1, device=dev, dtype=torch.float32)
                distr = (posr[:, None] - cen[None, :]).abs()
                content = torch.einsum("bhnd,bhmd->bhnm", qf, Kbar) * sc
                erf_fac = torch.erf(distr[None] / (2.0 * sig))
                lam = self.sel_scale.to(torch.float32).view(1, H, 1, 1)
                clc = content if self.sel_content_only else content + lam * ceil_c.unsqueeze(2) * erf_fac.unsqueeze(0)
                cl[:, :, :nr, :] = clc
            cl = cl.view(B, H, nbc, Bq, M)
            if self.causal:
                vis = blk_vis[q0 // Bq: q0 // Bq + nbc]
                cl = cl.masked_fill(~vis[None, None, :, None, :], NEG)
            if per_query:
                if topk_k > 0:
                    topi_pq[:, :, q0 // Bq: q0 // Bq + nbc, :, :] = cl.topk(topk_k, dim=-1).indices
                else:
                    blk_pq[:, :, q0 // Bq: q0 // Bq + nbc, :, :] = cl
            else:
                blk[:, :, q0 // Bq: q0 // Bq + nbc, :] = cl.amax(dim=3)
        _pp = self._prune_cfg("cell_prune_pool", "FS_PRUNE_POOL", 0)
        if (not per_query) and _pp > 0 and ceil_c is not None:
            _pw = self._prune_cfg("cell_prune_window", "FS_PRUNE_WINDOW", 8)
            blk_start = torch.arange(nb, device=dev, dtype=torch.float32) * Bq
            vis = cell_end[None, :] <= blk_start[:, None]
            near = vis & (cen[None, :] >= (blk_start[:, None] - _pw * s))
            ceil_bhnm = ceil_c.unsqueeze(2).expand(B, H, nb, M).masked_fill(
                ~vis[None, None], NEG)
            topP = ceil_bhnm.topk(min(_pp, M), dim=-1).indices
            cand = torch.zeros(B, H, nb, M, dtype=torch.bool, device=dev)
            cand.scatter_(-1, topP, True)
            cand = (cand | near[None, None]) & vis[None, None]
            blk = blk.masked_fill(~cand, NEG)
        if per_query:
            return topi_pq if topk_k > 0 else blk_pq
        return blk

    def _cell_select_score_On(self, Qf, Kbar, Kp, cen, cell_end, sig, N, M, s, B, H, D, dev, Bq, nb, Npad, kk):
        NEG = -1e9
        sc = self.scale
        P = self._prune_cfg("cell_prune_pool", "FS_PRUNE_POOL", 64)
        Wc = self._prune_cfg("cell_prune_window", "FS_PRUNE_WINDOW", 8)
        ceil_c = Kp.float().norm(dim=-1).view(B, H, M, s).amax(dim=3)
        lam = self.sel_scale.to(torch.float32).view(1, H, 1, 1, 1)
        sigv = sig.to(torch.float32).view(1, H, 1, 1, 1)
        Pk = min(P, M)
        topP = ceil_c.topk(Pk, dim=-1).indices
        blk_start = torch.arange(nb, device=dev, dtype=torch.float32) * Bq
        hi = (blk_start / s).floor().long() - 1
        near = hi[:, None] - torch.arange(Wc, device=dev)[None, :]
        cand = torch.cat([topP[:, :, None, :].expand(B, H, nb, Pk),
                          near[None, None, :, :].expand(B, H, nb, Wc)], dim=-1)
        C = cand.shape[-1]
        cand_valid = cand >= 0
        cand_c = cand.clamp(min=0)
        gidx = cand_c.reshape(B, H, nb * C, 1).expand(-1, -1, -1, D)
        Kbar_cand = torch.gather(Kbar, 2, gidx).reshape(B, H, nb, C, D)
        ceil_cand = torch.gather(ceil_c, 2, cand_c.reshape(B, H, nb * C)).reshape(B, H, nb, C)
        cen_cand = cen[cand_c]
        cend_cand = cell_end[cand_c]
        Qpad = Qf if Npad == N else torch.cat([Qf, Qf.new_zeros(B, H, Npad - N, D)], dim=2)
        Qb = Qpad.view(B, H, nb, Bq, D)
        qpos = torch.arange(Npad, device=dev, dtype=torch.float32).view(1, 1, nb, Bq, 1)
        content = torch.einsum("bhnqd,bhncd->bhnqc", Qb, Kbar_cand) * sc
        dist = (qpos - cen_cand[:, :, :, None, :]).abs()
        erf = torch.erf(dist / (2.0 * sigv))
        score = content if self.sel_content_only else content + lam * ceil_cand[:, :, :, None, :] * erf
        vis = (cend_cand <= blk_start[None, None, :, None]) & cand_valid
        score = score.masked_fill(~vis[:, :, :, None, :], NEG)
        blk_cand = score.amax(dim=3)
        sort_idx = cand_c.argsort(dim=-1, stable=True)
        sorted_g = torch.gather(cand_c, -1, sort_idx)
        is_dup = torch.zeros_like(sorted_g, dtype=torch.bool)
        is_dup[..., 1:] = sorted_g[..., 1:] == sorted_g[..., :-1]
        dup = torch.zeros_like(is_dup).scatter_(-1, sort_idx, is_dup)
        blk_cand = blk_cand.masked_fill(dup, NEG)
        sel = blk_cand.topk(min(kk, blk_cand.shape[-1]), dim=-1).indices
        topi = torch.gather(cand_c, -1, sel)
        return topi

    def _cell_score_On_chunked(self, Q, K, V, sigma, N):
        B, H, _, D = Q.shape
        s = self.cell_stride
        dev = Q.device
        NEG = -1e9
        sc = self.scale
        P = self._prune_cfg("cell_prune_pool", "FS_PRUNE_POOL", 64)
        Wc = self._prune_cfg("cell_prune_window", "FS_PRUNE_WINDOW", 8)
        NBCH = self._prune_cfg("cell_prune_blkchunk", "FS_PRUNE_BLKCHUNK", 256)
        M = (N + s - 1) // s
        pad = M * s - N
        if pad:
            Kp = torch.cat([K, K.new_zeros(B, H, pad, D)], dim=2)
            Vp = torch.cat([V, V.new_zeros(B, H, pad, D)], dim=2)
        else:
            Kp, Vp = K, V
        cen = (torch.arange(M, device=dev, dtype=torch.float32) + 0.5) * s
        cell_end = (torch.arange(M, device=dev, dtype=torch.float32) + 1.0) * s
        kk = min(self.cell_topk, M)
        Bq = min(self._cell_block, N)
        nb = (N + Bq - 1) // Bq
        lam = self.sel_scale.to(torch.float32).view(1, H, 1, 1, 1)
        sigv = sigma.to(torch.float32).view(1, H, 1, 1, 1)
        Kpv = Kp.view(B, H, M, s, D)
        Kbar = Kp.new_zeros(B, H, M, D, dtype=torch.float32)
        ceil_c = Kp.new_zeros(B, H, M, dtype=torch.float32)
        _MCH = 16384
        for _m0 in range(0, M, _MCH):
            _m1 = min(_m0 + _MCH, M)
            seg = Kpv[:, :, _m0:_m1]
            Kbar[:, :, _m0:_m1] = seg.mean(dim=3).float()
            ceil_c[:, :, _m0:_m1] = seg.float().norm(dim=-1).amax(dim=-1)
        Pk = min(P, M)
        topP = ceil_c.topk(Pk, dim=-1).indices
        Kk = kk * s
        off = torch.arange(s, device=dev)
        ar_wc = torch.arange(Wc, device=dev)
        ar_bq = torch.arange(Bq, device=dev, dtype=torch.float32)
        arbq_l = torch.arange(Bq, device=dev)
        out = Q.new_zeros(B, H, nb * Bq, D)
        for b0 in range(0, nb, NBCH):
            b1 = min(b0 + NBCH, nb)
            nbc = b1 - b0
            blk_start = torch.arange(b0, b1, device=dev, dtype=torch.float32) * Bq
            n_real = max(0, min(b1 * Bq, N) - b0 * Bq)
            Qc = Q.new_zeros(B, H, nbc * Bq, D)
            if n_real > 0:
                Qc[:, :, :n_real, :] = Q[:, :, b0 * Bq:b0 * Bq + n_real, :]
            Qb = Qc.float().view(B, H, nbc, Bq, D)
            Qbo = Qc.view(B, H, nbc, Bq, D)
            hi = (blk_start / s).floor().long() - 1
            near = hi[:, None] - ar_wc[None, :]
            cand = torch.cat([topP[:, :, None, :].expand(B, H, nbc, Pk),
                              near[None, None, :, :].expand(B, H, nbc, Wc)], dim=-1)
            C = cand.shape[-1]
            cand_valid = cand >= 0
            cand_c = cand.clamp(min=0)
            gidx = cand_c.reshape(B, H, nbc * C, 1).expand(-1, -1, -1, D)
            Kbar_cand = torch.gather(Kbar, 2, gidx).reshape(B, H, nbc, C, D)
            ceil_cand = torch.gather(ceil_c, 2, cand_c.reshape(B, H, nbc * C)).reshape(B, H, nbc, C)
            cen_cand = cen[cand_c]
            cend_cand = cell_end[cand_c]
            qpos = (blk_start[None, None, :, None, None] +
                    ar_bq.view(1, 1, 1, Bq, 1))
            content = torch.einsum("bhnqd,bhncd->bhnqc", Qb, Kbar_cand) * sc
            dist = (qpos - cen_cand[:, :, :, None, :]).abs()
            erf = torch.erf(dist / (2.0 * sigv))
            score = content if self.sel_content_only else content + lam * ceil_cand[:, :, :, None, :] * erf
            vis = (cend_cand <= blk_start[None, None, :, None]) & cand_valid
            score = score.masked_fill(~vis[:, :, :, None, :], NEG)
            blk_cand = score.amax(dim=3)
            sort_idx = cand_c.argsort(dim=-1, stable=True)
            sorted_g = torch.gather(cand_c, -1, sort_idx)
            is_dup = torch.zeros_like(sorted_g, dtype=torch.bool)
            is_dup[..., 1:] = sorted_g[..., 1:] == sorted_g[..., :-1]
            dup = torch.zeros_like(is_dup).scatter_(-1, sort_idx, is_dup)
            blk_cand = blk_cand.masked_fill(dup, NEG)
            sel = blk_cand.topk(min(kk, C), dim=-1).indices
            topi = torch.gather(cand_c, -1, sel)
            tok_idx = (topi.unsqueeze(-1) * s + off).reshape(B, H, nbc, Kk)
            gidx2 = tok_idx.reshape(B, H, nbc * Kk, 1).expand(-1, -1, -1, D)
            Kc = torch.gather(Kp, 2, gidx2).reshape(B, H, nbc, Kk, D)
            Vc = torch.gather(Vp, 2, gidx2).reshape(B, H, nbc, Kk, D)
            scores = torch.einsum("bhcqd,bhckd->bhcqk", Qbo, Kc).float() * self.scale
            if self.causal:
                qposi = (blk_start[:, None].long() + arbq_l[None, :])
                fut = tok_idx[:, :, :, None, :] >= qposi[None, None, :, :, None]
                scores = scores.masked_fill(fut, NEG)
            w = torch.softmax(scores, dim=-1).to(Vc.dtype)
            ob = torch.einsum("bhcqk,bhckd->bhcqd", w, Vc)
            if self.causal:
                ob = ob * (~fut).any(dim=-1, keepdim=True).to(ob.dtype)
            out[:, :, b0 * Bq:b1 * Bq, :] = ob.reshape(B, H, nbc * Bq, D)
        if nb * Bq > N:
            out = out[:, :, :N, :]
        return self.lr_norm(out.to(Q.dtype))

    def _cell_score_On_chunked_strict(self, Q, K, V, sigma, N):
        B, H, _, D = Q.shape
        s = self.cell_stride
        dev = Q.device
        NEG = -1e9
        P = self._prune_cfg("cell_prune_pool", "FS_PRUNE_POOL", 64)
        Wc = self._prune_cfg("cell_prune_window", "FS_PRUNE_WINDOW", 8)
        NBCH = self._prune_cfg("cell_prune_blkchunk", "FS_PRUNE_BLKCHUNK", 32)
        if s <= 0 or self.cell_topk <= 0:
            raise ValueError(f"strict O(N) cell selection: cell_stride ({s}) and cell_topk ({self.cell_topk}) must be positive")
        if P <= 0 or Wc < 0:
            raise ValueError(f"strict O(N) cell selection: cell_prune_pool ({P}) must be positive and cell_prune_window ({Wc}) non-negative")
        M = (N + s - 1) // s
        pad = M * s - N
        Kp = torch.cat([K, K.new_zeros(B, H, pad, D)], dim=2) if pad else K
        kk = min(self.cell_topk, M)
        Bq = min(self._cell_block, N)
        nb = (N + Bq - 1) // Bq
        if NBCH <= 0:
            NBCH = nb
        sc = self.scale
        _fused = bool(self._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0)) and self.cell_select == "score"
        _fused_select = _fused and bool(self._prune_cfg("cell_fused_select", "FS_FUSED_SELECT", 0))
        if _fused and NBCH > 0:
            _budget = self._prune_cfg("cell_fused_chunk_gib", "FS_FUSED_CHUNK_GIB", 4)
            C_est = min(P, M) + Wc
            _bytes_per_blk = B * H * Bq * C_est * 4
            _fit = max(1, int(_budget * (1024 ** 3) / max(1, _bytes_per_blk)))
            NBCH = min(nb, max(NBCH, _fit))
        if _fused_select and not (self._prune_cfg("cell_prune_incremental", "FS_PRUNE_INCREMENTAL", 0)
                                  or self._prune_cfg("cell_prune_stable_ties", "FS_PRUNE_STABLE_TIES", 0)):
            # same cells as the torch glue + Triton select below, from a few CUDA kernels (FSL_FAST_SELECT=0: legacy)
            from .fsl_select_cuda import fast_select_ok, fsl_fast_select
            if fast_select_ok(Q, Kp, min(P, M), Wc, kk, Bq, N):
                st = [] if (getattr(self, "_cell_stats_out", None) is not None and not pad and M <= 16384) else None
                sel_full = fsl_fast_select(Q, Kp, N=N, M=M, s=s, Bq=Bq, nb=nb, NBCH=NBCH, Pk=min(P, M), Wc=Wc, kk=kk,
                                           lam=self.sel_scale, sigv=sigma, scale=sc,
                                           content_only=self.sel_content_only, pool=P, stats_out=st)
                if sel_full is not None:
                    res = self._cell_fused_lr(Q, K, V, sel_full, s, Bq)
                    del sel_full
                    if st:
                        # FSKVCache.prefill: its own ops' bits (torch's bf16 mean, then .float(), as below)
                        self._cell_stats_out.append((K, st[0][0].float(), st[0][1]))
                        del st
                    return res
        Vp = torch.cat([V, V.new_zeros(B, H, pad, D)], dim=2) if pad else V
        cen = (torch.arange(M, device=dev, dtype=torch.float32) + 0.5) * s
        cell_end = (torch.arange(M, device=dev, dtype=torch.float32) + 1.0) * s
        lam = self.sel_scale.to(torch.float32).view(1, H, 1, 1, 1)
        sigv = sigma.to(torch.float32).view(1, H, 1, 1, 1)
        sel_full = torch.full((B, H, nb * Bq, kk), -1, dtype=torch.int32, device=dev) if _fused else None
        Kpv = Kp.view(B, H, M, s, D)
        Kbar = Kp.new_zeros(B, H, M, D, dtype=torch.float32)
        ceil_c = Kp.new_zeros(B, H, M, dtype=torch.float32)
        _MCH = 16384
        for _m0 in range(0, M, _MCH):
            _m1 = min(_m0 + _MCH, M)
            seg = Kpv[:, :, _m0:_m1]
            Kbar[:, :, _m0:_m1] = seg.mean(dim=3).float()
            ceil_c[:, :, _m0:_m1] = seg.float().norm(dim=-1).amax(dim=-1)
        if getattr(self, "_cell_stats_out", None) is not None and not pad and M <= _MCH:
            self._cell_stats_out.append((K, Kbar, ceil_c))      # FSKVCache.prefill: the same ops on the same view
        Pk = min(P, M)
        incremental = bool(self._prune_cfg("cell_prune_incremental", "FS_PRUNE_INCREMENTAL", 0))
        stable_ties = incremental or bool(self._prune_cfg("cell_prune_stable_ties", "FS_PRUNE_STABLE_TIES", 0))
        prefix_pool = None
        if stable_ties:
            from .prefix_topp import stable_topk_slots
            _fused_select = False
        if incremental:
            from .prefix_topp import causal_prefix_topp_incremental
            with torch.no_grad():
                prefix_pool = causal_prefix_topp_incremental(ceil_c, s, Bq, nb, Pk)
        off = torch.arange(s, device=dev)
        ar_wc = torch.arange(Wc, device=dev)
        arbq_l = torch.arange(Bq, device=dev)
        ar_bq = torch.arange(Bq, device=dev, dtype=torch.float32)
        _Cpool = Pk + Wc
        _dup_tri = torch.tril(torch.ones(_Cpool, _Cpool, dtype=torch.bool, device=dev), diagonal=-1)
        out = Q.new_zeros(B, H, nb * Bq, D)
        for b0 in range(0, nb, NBCH):
            b1 = min(b0 + NBCH, nb)
            nbc = b1 - b0
            blk_start = torch.arange(b0, b1, device=dev, dtype=torch.float32) * Bq
            n_real = max(0, min(b1 * Bq, N) - b0 * Bq)
            Qc = Q.new_zeros(B, H, nbc * Bq, D)
            if n_real > 0:
                Qc[:, :, :n_real, :] = Q[:, :, b0 * Bq:b0 * Bq + n_real, :]
            Qb = Qc.float().view(B, H, nbc, Bq, D)
            Qbo = Qc.view(B, H, nbc, Bq, D)
            hi = (blk_start / s).floor().long() - 1
            if prefix_pool is not None:
                topP = prefix_pool[:, :, b0:b1]
            else:
                visM = cell_end[None, :] <= blk_start[:, None]
                ceil_pref = ceil_c[:, :, None, :].expand(B, H, nbc, M).masked_fill(
                    ~visM[None, None], NEG)
                if stable_ties:
                    cell_ids = torch.arange(M, device=dev).view(1, 1, 1, M)
                    topP = stable_topk_slots(ceil_pref, cell_ids, Pk)
                    topP = topP.masked_fill(ceil_pref.gather(-1, topP) == NEG, -1)
                else:
                    topP = ceil_pref.topk(Pk, dim=-1).indices
            near = (hi[:, None] - ar_wc[None, :])
            cand = torch.cat([topP,
                              near[None, None, :, :].expand(B, H, nbc, Wc)], dim=-1)
            C = cand.shape[-1]
            cand_valid = cand >= 0
            cand_c = cand.clamp(min=0)
            gidx = cand_c.reshape(B, H, nbc * C, 1).expand(-1, -1, -1, D)
            Kbar_cand = torch.gather(Kbar, 2, gidx).reshape(B, H, nbc, C, D)
            ceil_cand = torch.gather(ceil_c, 2, cand_c.reshape(B, H, nbc * C)).reshape(B, H, nbc, C)
            cen_cand = cen[cand_c]
            cend_cand = cell_end[cand_c]
            vis = (cend_cand <= blk_start[None, None, :, None]) & cand_valid
            dup = ((cand_c[..., None, :] == cand_c[..., :, None]) & _dup_tri).any(dim=-1)
            ok_cell = vis & ~dup
            if _fused_select:
                from .cell_select_triton import cell_fused_select
                sel_full[:, :, b0 * Bq:b1 * Bq, :] = cell_fused_select(
                    Qb, Kbar_cand, ceil_cand, cen_cand, ok_cell, cand_c,
                    blk_start, self.sel_scale.to(torch.float32), sigv.reshape(H),
                    kk=kk, scale=sc, content_only=self.sel_content_only, neg=NEG,
                )
                continue
            qpos = (blk_start[None, None, :, None, None] +
                    ar_bq.view(1, 1, 1, Bq, 1))
            content = torch.einsum("bhnqd,bhncd->bhnqc", Qb, Kbar_cand) * sc
            dist = (qpos - cen_cand[:, :, :, None, :]).abs()
            erf = torch.erf(dist / (2.0 * sigv))
            score = content if self.sel_content_only else content + lam * ceil_cand[:, :, :, None, :] * erf
            score = score.masked_fill(~ok_cell[:, :, :, None, :], NEG)
            if stable_ties:
                ids = cand_c.masked_fill(~ok_cell, -1)[:, :, :, None, :]
                sel_pq = stable_topk_slots(score, ids, min(kk, C))
            else:
                sel_pq = score.topk(min(kk, C), dim=-1).indices
            cellmask = torch.zeros(B, H, nbc, Bq, C, dtype=torch.bool, device=dev)
            cellmask.scatter_(-1, sel_pq, True)
            cellmask = cellmask & ok_cell[:, :, :, None, :]
            if _fused:
                _exp = (B, H, nbc, Bq, C)
                topi_g = torch.gather(cand_c[:, :, :, None, :].expand(_exp), 4, sel_pq)
                sel_ok = torch.gather(ok_cell[:, :, :, None, :].expand(_exp), 4, sel_pq)
                topi_g = topi_g.to(torch.int32).masked_fill(~sel_ok, -1)
                sel_full[:, :, b0 * Bq:b1 * Bq, :] = topi_g.reshape(B, H, nbc * Bq, kk)
                continue
            tok_idx = (cand_c.unsqueeze(-1) * s + off).reshape(B, H, nbc, C * s)
            gidx2 = tok_idx.reshape(B, H, nbc * C * s, 1).expand(-1, -1, -1, D)
            Kc = torch.gather(Kp, 2, gidx2).reshape(B, H, nbc, C * s, D)
            Vc = torch.gather(Vp, 2, gidx2).reshape(B, H, nbc, C * s, D)
            scores = torch.einsum("bhcqd,bhckd->bhcqk", Qbo, Kc).float() * self.scale
            tokcell = cellmask.repeat_interleave(s, dim=-1)
            allow = tokcell
            if self.causal:
                qposi = (blk_start[:, None].long() + arbq_l[None, :])
                fut = tok_idx[:, :, :, None, :] >= qposi[None, None, :, :, None]
                allow = allow & ~fut
            scores = scores.masked_fill(~allow, NEG)
            w = torch.softmax(scores, dim=-1).to(Vc.dtype)
            ob = torch.einsum("bhcqk,bhckd->bhcqd", w, Vc)
            ob = ob * allow.any(dim=-1, keepdim=True).to(ob.dtype)
            out[:, :, b0 * Bq:b1 * Bq, :] = ob.reshape(B, H, nbc * Bq, D)
        if _fused:
            return self._cell_fused_lr(Q, K, V, sel_full, s, Bq)
        if nb * Bq > N:
            out = out[:, :, :N, :]
        return self.lr_norm(out.to(Q.dtype))


    def _is_dense_lr(self):
        return not bool(self._prune_cfg("cell_prune_kernel", "FS_PRUNE_KERNEL", 0))

    def _rope_abs(self, x, pos0):
        n = x.shape[2]
        end = pos0 + n
        if end > self.rope.cos_cached.shape[0]:
            self.rope._build_cache(end)
        cos = self.rope.cos_cached[pos0:end]
        sin = self.rope.sin_cached[pos0:end]
        return _apply_rotary_emb(x, cos, sin)

    def _rope_pair(self, Q, K, cos, sin, window_only=False):
        # _apply_rotary_emb of Q and K. Without autograd on CUDA: one fused pass each, bit for bit (trunk_triton.rope).
        # When the rotated tensors feed only the window's SDPA (window_only) and autocast is on, they are written in
        # the autocast dtype directly: the same values SDPA's autocast cast would produce.
        if Q.shape[2] > 1 and Q.is_cuda and not torch.is_grad_enabled():
            from ..utils import trunk_triton as tt
            if tt.rope_ok(Q, cos) and tt.rope_ok(K, cos) and sin.stride(-1) == 1 and sin.shape == cos.shape:
                od = None
                adt = tt.autocast_dtype()
                if window_only and adt is not None and os.environ.get("FSL_SLIDING_MODE", "chunked").lower() != "flex":
                    od = adt
                return tt.rope(Q, cos, sin, od), tt.rope(K, cos, sin, od)
        return _apply_rotary_emb(Q, cos, sin), _apply_rotary_emb(K, cos, sin)

    def _assert_cacheable(self, phase="decode"):
        if not self.causal:
            raise NotImplementedError("FSKVCache requires causal=True.")
        if self.use_rope and getattr(self, "rope_long_range", True):
            raise NotImplementedError(
                "FSKVCache with use_rope=True requires rope_long_range=False "
                "(SR-rope + NoPE-LR, the deployed rope-local config). rope_long_range=True "
                "rotates the long-range keys (K_long=K_rope), which needs rotated-K cell "
                "summaries — not implemented.")
        if self.disable_short_range or self.disable_long_range:
            raise NotImplementedError("FSKVCache requires both SR and LR branches enabled.")
        if self.long_range_variant != "cell_topk" or self.cell_select != "score":
            raise NotImplementedError(
                "FSKVCache only supports long_range_variant='cell_topk', cell_select='score' "
                f"(got {self.long_range_variant!r}/{self.cell_select!r}).")
        _strict = bool(self._prune_cfg("sel_strict_causal", "FS_STRICT_CAUSAL", 0))
        _fused = bool(self._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0))
        if not _strict:
            raise NotImplementedError(
                "FSKVCache requires sel_strict_causal=1 (block-causal path is non-cacheable).")

    def _forward_cached(self, x, cache):
        B, N, _ = x.shape
        self._assert_cacheable(phase="prefill" if (cache.pos == 0 and N > 1) else "decode")
        H, D = self.n_heads, self.head_dim
        is_std_prefill = (cache.pos == 0 and N > 1 and not getattr(cache, "lean_prefill", False)
                          and not getattr(cache, "bounded", False))
        pre_ok = not self.fsb_enabled or (self.cell_stride == 32 and D == 64)
        pre = getattr(cache, "_prealloc", None) if (is_std_prefill and not torch.is_grad_enabled()
                                                    and pre_ok and not compiling()) else None
        cap = N + max(0, int(getattr(cache, "reserve", 0))) if pre is not None else 0
        if (pre is not None and pre[0] == B and pre[1] == N and tuple(pre[2].shape) == (B, H, cap, D)
                and pre[2].device == x.device):
            wk = self.W_k(x).view(B, N, H, D).transpose(1, 2)
            if wk.dtype == pre[2].dtype:
                K = pre[2][:, :, :N]
                K.copy_(wk)
                del wk
                wv = self.W_v(x).view(B, N, H, D).transpose(1, 2)
                V = pre[3][:, :, :N]
                V.copy_(wv)
                del wv
                Q = self.W_q(x).view(B, N, H, D).transpose(1, 2)
            else:
                Q = self.W_q(x).view(B, N, H, D).transpose(1, 2)
                K = wk
                del wk
                V = self.W_v(x).view(B, N, H, D).transpose(1, 2)
        elif is_std_prefill and not torch.is_grad_enabled() and pre_ok and not compiling() and N > 4096:
            K = self.W_k(x).view(B, N, H, D).transpose(1, 2).contiguous()
            V = self.W_v(x).view(B, N, H, D).transpose(1, 2).contiguous()
            Q = self.W_q(x).view(B, N, H, D).transpose(1, 2)
        else:
            Q = self.W_q(x).view(B, N, H, D).transpose(1, 2)
            K = self.W_k(x).view(B, N, H, D).transpose(1, 2)
            V = self.W_v(x).view(B, N, H, D).transpose(1, 2)
        if getattr(x, "_can_release", False) and not torch.is_grad_enabled():
            x.set_(x.new_empty(0))
        del x

        if getattr(cache, "bounded", False):
            if self.fsb_enabled:
                raise NotImplementedError("FS-B has no bounded-cache (linear) path; use FSKVCache.")
            out = cache.attend(Q, K, V)
            return self.W_o(out.transpose(1, 2).contiguous().view(B, N, self.d_model))

        _rope = self.use_rope
        _rope_lr = _rope and getattr(self, "rope_long_range", True)
        if is_std_prefill:
            sigma = self.sigma
            used_chunk_rope = False
            if (_rope and not _rope_lr and self.fs_uniform_sr and self.causal
                    and N > 16384 and Q.is_cuda and not torch.is_grad_enabled() and not compiling()
                    and os.environ.get("FSL_SLIDING_MODE", "chunked").lower() != "flex"
                    and os.environ.get("FSL_WINDOW_STREAM", "0") != "1"):
                end = cache.pos + N
                if end > self.rope.cos_cached.shape[0]:
                    self.rope._build_cache(end)
                cos = self.rope.cos_cached[cache.pos:end]
                sin = self.rope.sin_cached[cache.pos:end]
                from ..utils import trunk_triton as tt
                from ..utils.local_attention import _batched_sliding_window_rope
                adt = tt.autocast_dtype()
                dt_sr = adt or Q.dtype
                T_blk = 64 if D <= 64 else 128
                if (tt.rope_ok(Q, cos) and tt.rope_ok(K, cos) and sin.stride(-1) == 1 and sin.shape == cos.shape
                        and D <= 128 and D % 8 == 0 and self.window_size % T_blk == 0 and N >= self.window_size
                        and dt_sr in (torch.bfloat16, torch.float16) and V.stride(-1) == 1):
                    out_short = self.dropout(_batched_sliding_window_rope(
                        Q, K, V, cos, sin, adt, self.window_size, self.scale, T=T_blk
                    ))
                    ev = None
                    Q_lr, K_lr = Q, K
                    used_chunk_rope = True
            if not used_chunk_rope:
                if _rope:
                    end = cache.pos + N
                    if end > self.rope.cos_cached.shape[0]:
                        self.rope._build_cache(end)
                    Q_sr, K_sr = self._rope_pair(
                        Q, K, self.rope.cos_cached[cache.pos:end], self.rope.sin_cached[cache.pos:end],
                        window_only=not _rope_lr,
                    )
                else:
                    Q_sr, K_sr = Q, K
                Q_lr = Q_sr if _rope_lr else Q
                K_lr = K_sr if _rope_lr else K
                out_short, ev = self._short_range_prefill(Q_sr, K_sr, V, sigma)
                if not _rope_lr and ev is None and not torch.is_grad_enabled():
                    del Q_sr, K_sr
            if self._lr_handoff_off():
                out_long = self._long_range(Q_lr, K_lr, V, sigma, N)
                fused, cell_stats = False, None
            else:
                # FSL_PREFILL_CELL_REUSE=0: FSKVCache.prefill recomputes the cell means / ceilings itself
                self._begin_lr(out_short, want_cell_stats=(os.environ.get("FSL_PREFILL_CELL_REUSE", "1") != "0"
                                                           and not torch.is_grad_enabled()),
                               sr_event=ev, inplace_sr=(ev is None and not torch.is_grad_enabled()))
                try:
                    out_long = self._long_range(Q_lr, K_lr, V, sigma, N)
                finally:
                    fused, cell_stats = self._end_lr()
            y = self._combine(out_short, out_long, fused, B, N, sr_event=ev)
            if not torch.is_grad_enabled():
                del Q, Q_lr, out_short, out_long
            if cell_stats is not None:
                cache._prefill_cell_stats = cell_stats          # read (and cleared) by FSKVCache.prefill
            cache.prefill(K, V)
            if not torch.is_grad_enabled():
                del K, K_lr, V, cell_stats
                out = self.W_o(y)
                del y
                return out
            return self.W_o(y)

        if _rope:
            end = cache.pos + N
            if end > self.rope.cos_cached.shape[0]:
                self.rope._build_cache(end)
            Q_sr, K_sr = self._rope_pair(Q, K, self.rope.cos_cached[cache.pos:end], self.rope.sin_cached[cache.pos:end],
                                         window_only=not _rope_lr)
        else:
            Q_sr, K_sr = Q, K
        Q_lr = Q_sr if _rope_lr else Q
        K_lr = K_sr if _rope_lr else K

        if cache.pos == 0 and N > 1:
            sigma = self.sigma
            out_short, ev = self._short_range_prefill(Q_sr, K_sr, V, sigma)
            cache.prefill(K, V)
            if self._lr_handoff_off():
                out_long = self._lr_prefill_blocks(Q_lr, cache, sigma, N)
                fused = False
            else:
                self._begin_lr(out_short, sr_event=ev)
                try:
                    out_long = self._lr_prefill_blocks(Q_lr, cache, sigma, N)
                finally:
                    fused, _ = self._end_lr()
            return self.W_o(self._combine(out_short, out_long, fused, B, N, sr_event=ev))

        outs = []
        for t in range(N):
            k_t = K[:, :, t:t + 1, :]
            v_t = V[:, :, t:t + 1, :]
            q_t = Q[:, :, t:t + 1, :]
            o_t = self._decode_one(q_t, k_t, v_t, cache)
            outs.append(o_t)
        out = torch.cat(outs, dim=2)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, N, self.d_model))

    def _decode_one(self, q_t, k_t, v_t, cache):
        H, D = self.n_heads, self.head_dim
        pos = cache.pos
        sigma = self.sigma
        cache.append(k_t, v_t)
        out_short = self._sr_decode_step(q_t, cache, sigma, pos)
        _fused = bool(self._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0))
        if self.fsb_enabled:
            from .fsb_tree import fsb_decode_step
            out_long = fsb_decode_step(self, q_t, cache, sigma, pos)
        elif _fused and not self._is_dense_lr():
            out_long = self._lr_decode_step_fused(q_t, cache, sigma, pos)
        else:
            out_long = self._lr_decode_step(q_t, cache, sigma, pos)
        out = out_short + self.long_range_scale.unsqueeze(0) * out_long
        return out

    def _sr_decode_step(self, q_t, cache, sigma, pos):
        B, H = q_t.shape[0], self.n_heads
        W = self.window_size
        Kw, Vw, posw = cache.sr_window(at=pos)
        dev = q_t.device
        if self.use_rope:
            k0 = pos - (Kw.shape[2] - 1)
            q_t = self._rope_abs(q_t, pos)
            Kw = self._rope_abs(Kw, k0)
        dist = (pos - posw.to(torch.float32))
        band = dist.abs() <= W
        if self.fs_uniform_sr:
            zero = torch.zeros((), device=dev, dtype=torch.float32)
            neg = torch.full((), float("-inf"), device=dev, dtype=torch.float32)
            m = torch.where(band.unsqueeze(0), zero, neg).expand(H, -1)
        else:
            arg = dist.abs().unsqueeze(0) / (2.0 * sigma.to(torch.float32).view(H, 1))
            bias = torch.log(torch.erfc(arg).clamp(min=1e-10))
            neg = torch.full((), float("-inf"), device=dev, dtype=bias.dtype)
            m = torch.where(band.unsqueeze(0), bias, neg)
        scores = torch.einsum("bhd,bhld->bhl", q_t[:, :, 0, :], Kw).float() * self.scale
        scores = scores + m.unsqueeze(0)
        w = torch.softmax(scores, dim=-1).to(Vw.dtype)
        out = torch.einsum("bhl,bhld->bhd", w, Vw).unsqueeze(2)
        return out

    def _lr_decode_step_dense(self, q_t, cache, sigma, pos):
        B, H, _, D = q_t.shape
        s = self.cell_stride
        dev = q_t.device
        NEG = -1e9
        Bq = self._cell_block
        blk_start = (pos // Bq) * Bq
        Kbar_all, ceil_all, ncell = cache.completed_cells()
        M_vis = blk_start // s
        if M_vis <= 0 or ncell <= 0:
            return q_t.new_zeros(B, H, 1, D)
        M_vis = min(M_vis, ncell)
        cen = (torch.arange(M_vis, device=dev, dtype=torch.float32) + 0.5) * s
        Kbar = Kbar_all[:, :, :M_vis, :]
        ceil_c = ceil_all[:, :, :M_vis]
        lam = self.sel_scale.to(torch.float32).view(1, H, 1)
        sigv = sigma.to(torch.float32).view(1, H, 1)
        sc = self.scale
        content = torch.einsum("bhd,bhcd->bhc", q_t[:, :, 0, :].float(), Kbar) * sc
        dist = (float(pos) - cen).abs()
        erf_fac = torch.erf(dist.view(1, 1, M_vis) / (2.0 * sigv))
        score = content if self.sel_content_only else content + lam * ceil_c * erf_fac
        kk = min(self.cell_topk, M_vis)
        topi = score.topk(kk, dim=-1).indices
        if bool(self._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0)):
            sel = topi.to(torch.int32).unsqueeze(2).contiguous()
            Khist, Vhist = cache.lr_tokens()
            return self._cell_fused_lr(q_t.contiguous(), Khist, Vhist, sel, s, Bq, bias=None, q_pos=pos)
        off = torch.arange(s, device=dev)
        tok_idx = (topi.unsqueeze(-1) * s + off).reshape(B, H, kk * s)
        Kp, Vp = cache.lr_tokens()
        gidx = tok_idx.unsqueeze(-1).expand(B, H, kk * s, D)
        Kc = torch.gather(Kp, 2, gidx)
        Vc = torch.gather(Vp, 2, gidx)
        scores = torch.einsum("bhd,bhkd->bhk", q_t[:, :, 0, :], Kc).float() * self.scale
        allow = tok_idx < pos
        scores = scores.masked_fill(~allow, NEG)
        w = torch.softmax(scores, dim=-1).to(Vc.dtype)
        ob = torch.einsum("bhk,bhkd->bhd", w, Vc).unsqueeze(2)
        ob = ob * allow.any(dim=-1).view(B, H, 1, 1).to(ob.dtype)
        return self.lr_norm(ob.to(q_t.dtype))

    def _lr_decode_step(self, q_t, cache, sigma, pos):
        if self._is_dense_lr():
            return self._lr_decode_step_dense(q_t, cache, sigma, pos)
        B, H, _, D = q_t.shape
        s = self.cell_stride
        dev = q_t.device
        NEG = -1e9
        Bq = self._cell_block
        blk_start = (pos // Bq) * Bq
        pc = getattr(cache, "_lr_decode_pool", None)
        if pc is None or pc["blk_start"] != blk_start or (not pc.get("empty", False) and "Kc" not in pc):
            pc = self._build_decode_pool(cache, blk_start, s, dev, need_tokens=True)
            pc["blk_start"] = blk_start
            cache._lr_decode_pool = pc
        if pc["empty"]:
            return q_t.new_zeros(B, H, 1, D)
        M_vis = pc["M_vis"]; C = pc["C"]
        Kbar_cand = pc["Kbar_cand"]; ceil_cand = pc["ceil_cand"]; cen_cand = pc["cen_cand"]
        ok_cell = pc["ok_cell"]; tok_idx = pc["tok_idx"]; Kc = pc["Kc"]; Vc = pc["Vc"]
        lam = self.sel_scale.to(torch.float32).view(1, H, 1)
        sigv = sigma.to(torch.float32).view(1, H, 1)
        sc = self.scale
        content = torch.einsum("bhd,bhcd->bhc", q_t[:, :, 0, :].float(), Kbar_cand) * sc
        dist = (float(pos) - cen_cand).abs()
        erf = torch.erf(dist / (2.0 * sigv))
        score = content + lam * ceil_cand * erf
        score = score.masked_fill(~ok_cell, NEG)
        kk = min(self.cell_topk, M_vis)
        sel = score.topk(min(kk, C), dim=-1).indices
        cellmask = torch.zeros(B, H, C, dtype=torch.bool, device=dev)
        cellmask.scatter_(-1, sel, True)
        cellmask = cellmask & ok_cell
        scores = torch.einsum("bhd,bhkd->bhk", q_t[:, :, 0, :], Kc).float() * self.scale
        tokcell = cellmask.repeat_interleave(s, dim=-1)
        allow = tokcell & (tok_idx < pos)
        scores = scores.masked_fill(~allow, NEG)
        w = torch.softmax(scores, dim=-1).to(Vc.dtype)
        ob = torch.einsum("bhk,bhkd->bhd", w, Vc).unsqueeze(2)
        ob = ob * allow.any(dim=-1).view(B, H, 1, 1).to(ob.dtype)
        return self.lr_norm(ob.to(q_t.dtype))

    def _lr_decode_step_fused(self, q_t, cache, sigma, pos):
        B, H, _, D = q_t.shape
        s = self.cell_stride
        dev = q_t.device
        NEG = -1e9
        Bq = self._cell_block
        blk_start = (pos // Bq) * Bq
        pc = getattr(cache, "_lr_decode_pool", None)
        if pc is None or pc["blk_start"] != blk_start:
            pc = self._build_decode_pool(cache, blk_start, s, dev, need_tokens=False)
            pc["blk_start"] = blk_start
            cache._lr_decode_pool = pc
        if pc["empty"]:
            return q_t.new_zeros(B, H, 1, D)
        M_vis = pc["M_vis"]; C = pc["C"]
        Kbar_cand = pc["Kbar_cand"]; ceil_cand = pc["ceil_cand"]; cen_cand = pc["cen_cand"]
        ok_cell = pc["ok_cell"]; cand_c = pc["cand_c"]
        lam = self.sel_scale.to(torch.float32).view(1, H, 1)
        sigv = sigma.to(torch.float32).view(1, H, 1)
        sc = self.scale
        content = torch.einsum("bhd,bhcd->bhc", q_t[:, :, 0, :].float(), Kbar_cand) * sc
        dist = (float(pos) - cen_cand).abs()
        erf = torch.erf(dist / (2.0 * sigv))
        score = content + lam * ceil_cand * erf
        score = score.masked_fill(~ok_cell, NEG)
        kk = min(self.cell_topk, M_vis)
        sel_slot = score.topk(min(kk, C), dim=-1).indices
        pick_ok = torch.gather(ok_cell, -1, sel_slot)
        gcell = torch.gather(cand_c, -1, sel_slot)
        sel = gcell.to(torch.int32).masked_fill(~pick_ok, -1)
        sel = sel.unsqueeze(2).contiguous()
        Khist, Vhist = cache.lr_tokens()
        q_in = q_t.contiguous()
        out = self._cell_fused_lr(q_in, Khist, Vhist, sel, s, Bq, bias=None, q_pos=pos)
        return out

    def _build_dense_decode_pool(self, cache, blk_start, s, dev, out: dict | None = None):
        Kbar_all, ceil_all, ncell = cache.completed_cells()
        M_vis = blk_start // s
        if M_vis <= 0 or ncell <= 0:
            return {"empty": True}
        B, H, _, _ = Kbar_all.shape
        M_vis = min(M_vis, ncell)
        Kbar_buf = getattr(cache, "_kbar_buf", None)
        ceil_buf = getattr(cache, "_ceil_buf", None)
        if Kbar_buf is None or ceil_buf is None or Kbar_buf.shape[2] < M_vis:
            Kbar_buf = Kbar_all[:, :, :M_vis]
            ceil_buf = ceil_all[:, :, :M_vis]
        if (isinstance(out, dict) and out.get("dense", False)
                and out.get("Kbar_all") is Kbar_buf and out.get("ceil_all") is ceil_buf):
            out["M_vis"] = M_vis
            out["m_vis_t"].fill_(M_vis)
            return out
        max_cells = Kbar_buf.shape[2]
        m_vis_t = torch.tensor(M_vis, device=dev, dtype=torch.int32)
        split_m = 16
        kk = min(max(1, int(self.cell_topk)), 16)
        part_s = torch.empty((B, H, split_m * kk), device=dev, dtype=torch.float32)
        part_c = torch.empty((B, H, split_m * kk), device=dev, dtype=torch.int32)
        return {
            "empty": False,
            "dense": True,
            "M_vis": M_vis,
            "C": max_cells,
            "m_vis_t": m_vis_t,
            "Kbar_all": Kbar_buf,
            "ceil_all": ceil_buf,
            "part_s": part_s,
            "part_c": part_c,
        }

    def _build_decode_pool(self, cache, blk_start, s, dev, need_tokens: bool = True, out: dict | None = None):
        P = self._prune_cfg("cell_prune_pool", "FS_PRUNE_POOL", 64)
        Wc = self._prune_cfg("cell_prune_window", "FS_PRUNE_WINDOW", 8)
        Kbar_all, ceil_all, ncell = cache.completed_cells()
        M_vis = blk_start // s
        if M_vis <= 0 or ncell <= 0:
            # Nothing visible yet. Check this before touching Kbar_all: it is None until the
            # cache has a completed cell (e.g. right after a prompt of <= cell_stride - 2 tokens).
            return {"empty": True}
        B, H, _, D = Kbar_all.shape
        M_vis = min(M_vis, ncell)
        Pk = min(P, M_vis)
        ceil_c = ceil_all[:, :, :M_vis]
        topP = ceil_c.topk(Pk, dim=-1).indices
        if not need_tokens:
            from ..utils import trunk_triton as _tt
            if _tt.fsl_pool_refresh_ok(Kbar_all, ceil_all, M_vis, Pk, Wc, D):
                return _tt.fsl_pool_refresh(topP, Kbar_all, ceil_all, M_vis, blk_start, s, Pk, Wc, out=out)
        cen = (torch.arange(M_vis, device=dev, dtype=torch.float32) + 0.5) * s
        cell_end = (torch.arange(M_vis, device=dev, dtype=torch.float32) + 1.0) * s
        Kbar = Kbar_all[:, :, :M_vis, :]
        hi = blk_start // s - 1
        near = hi - torch.arange(Wc, device=dev)
        near_e = near.view(1, 1, Wc).expand(B, H, Wc)
        cand = torch.cat([topP, near_e], dim=-1)
        C = cand.shape[-1]
        cand_valid = cand >= 0
        cand_c = cand.clamp(min=0)
        gidx = cand_c.unsqueeze(-1).expand(B, H, C, D)
        Kbar_cand = torch.gather(Kbar, 2, gidx)
        ceil_cand = torch.gather(ceil_c, 2, cand_c)
        cen_cand = cen[cand_c]
        cend_cand = cell_end[cand_c]
        vis = (cend_cand <= float(blk_start)) & cand_valid
        sort_idx = cand_c.argsort(dim=-1, stable=True)
        sorted_g = torch.gather(cand_c, -1, sort_idx)
        is_dup = torch.zeros_like(sorted_g, dtype=torch.bool)
        is_dup[..., 1:] = sorted_g[..., 1:] == sorted_g[..., :-1]
        dup = torch.zeros_like(is_dup).scatter_(-1, sort_idx, is_dup)
        ok_cell = vis & ~dup
        res = {"empty": False, "M_vis": M_vis, "C": C,
               "Kbar_cand": Kbar_cand, "ceil_cand": ceil_cand, "cen_cand": cen_cand,
               "ok_cell": ok_cell, "cand_c": cand_c}
        if need_tokens:
            off = torch.arange(s, device=dev)
            tok_idx = (cand_c.unsqueeze(-1) * s + off).reshape(B, H, C * s)
            Kp, Vp = cache.lr_tokens()
            gidx2 = tok_idx.unsqueeze(-1).expand(B, H, C * s, D)
            res["tok_idx"] = tok_idx
            res["Kc"] = torch.gather(Kp, 2, gidx2)
            res["Vc"] = torch.gather(Vp, 2, gidx2)
        return res

    def _sr_decode_from_buf(self, q_t, Kw, Vw, posw, pos_t, sigma):
        B, H = q_t.shape[0], self.n_heads
        W = self.window_size
        dev = q_t.device
        if self.use_rope:
            Lw = Kw.shape[2]
            q_t = self._rope_abs_t(q_t, pos_t)
            Kw = self._rope_abs_t(Kw, pos_t - (Lw - 1))
        dist = (pos_t.to(torch.float32) - posw)
        band = dist.abs() <= W
        if self.fs_uniform_sr:
            zero = torch.zeros((), device=dev, dtype=torch.float32)
            neg = torch.full((), float("-inf"), device=dev, dtype=torch.float32)
            m = torch.where(band.unsqueeze(0), zero, neg).expand(H, -1)
        else:
            arg = dist.abs().unsqueeze(0) / (2.0 * sigma.to(torch.float32).view(H, 1))
            bias = torch.log(torch.erfc(arg).clamp(min=1e-10))
            neg = torch.full((), float("-inf"), device=dev, dtype=bias.dtype)
            m = torch.where(band.unsqueeze(0), bias, neg)
        scores = torch.einsum("bhd,bhld->bhl", q_t[:, :, 0, :], Kw).float() * self.scale
        scores = scores + m.unsqueeze(0)
        w = torch.softmax(scores, dim=-1).to(Vw.dtype)
        out = torch.einsum("bhl,bhld->bhd", w, Vw).unsqueeze(2)
        return out

    def _ensure_qkv_cache(self):
        wq, wk, wv = self.W_q.weight, self.W_k.weight, self.W_v.weight
        dm = self.d_model
        key = (wq.data_ptr(), wq._version, wk.data_ptr(), wk._version, wv.data_ptr(), wv._version, wq.dtype)
        c = getattr(self, "_w_qkv_cache", None)
        if c is None or c[0] != key:
            qkv_w = torch.cat([wq.data, wk.data, wv.data], dim=0)
            self.W_q.weight.data = qkv_w[:dm]
            self.W_k.weight.data = qkv_w[dm:2 * dm]
            self.W_v.weight.data = qkv_w[2 * dm:]
            wq, wk, wv = self.W_q.weight, self.W_k.weight, self.W_v.weight
            key = (wq.data_ptr(), wq._version, wk.data_ptr(), wk._version, wv.data_ptr(), wv._version, wq.dtype)
            c = (key, qkv_w)
            self._w_qkv_cache = c
        return c[1], dm

    def _qkv_decode(self, x, B, N, H, D):
        if (x.is_cuda and B * N == 1 and not torch.is_grad_enabled()
                and self.W_q.bias is None and self.W_k.bias is None and self.W_v.bias is None
                and os.environ.get("FSL_TRUNK_FUSED", "1") != "0"):
            qkv_w, _ = self._ensure_qkv_cache()
            return F.linear(x, qkv_w).view(3, B, H, N, D).unbind(0)
        return (self.W_q(x).view(B, N, H, D).transpose(1, 2),
                self.W_k(x).view(B, N, H, D).transpose(1, 2),
                self.W_v(x).view(B, N, H, D).transpose(1, 2))

    def _sr_kv_and_decode(self, Q, K, V, kbuf, vbuf, buf, sigma):
        from ..utils import trunk_triton as _tt
        pos_t = buf["pos_t"]
        win_idx = buf["win_idx"]
        win_pos = buf["win_pos"]
        cos = self.rope.cos_cached if self.use_rope else None
        sin = self.rope.sin_cached if self.use_rope else None
        if _tt.sr_decode_ok(Q, K, V, kbuf, vbuf, win_idx, win_pos, cos):
            return _tt.sr_decode_fused(Q, K, V, kbuf, vbuf, pos_t, win_idx, win_pos, sigma,
                                       self.scale, self.window_size, self.fs_uniform_sr, self.use_rope, cos, sin)
        kbuf.index_copy_(2, pos_t.reshape(1), K)
        vbuf.index_copy_(2, pos_t.reshape(1), V)
        Kw = kbuf.index_select(2, win_idx)
        Vw = vbuf.index_select(2, win_idx)
        return self._sr_decode_from_buf(Q, Kw, Vw, win_pos, pos_t, sigma)

    def graph_decode_layer_fsb(self, x, cache, buf):
        # _decode_one for FS-B inside a whole-step decode graph: K / V go into the KV cache's own buffers at pos_t (the
        # FS-B read gathers from them), the window as in graph_decode_layer, the block's FS-B read (forests prepared
        # outside the graph by fsb_decode_prepare); the cache's cell bookkeeping runs after the replay
        from .fsb_tree import fsb_decode_token
        B, N, _ = x.shape
        H, D = self.n_heads, self.head_dim
        Q, K, V = self._qkv_decode(x, B, N, H, D)
        pos_t = buf["pos_t"]
        fc = getattr(cache, "_fsb_forest", None)
        tk = getattr(fc, "_tok", None)
        sig = tk.get("sigma") if tk is not None and tk.get("sigma") is not None else self.sigma
        out_short = self._sr_kv_and_decode(Q, K, V, cache._k_buf, cache._v_buf, buf, sig)
        win_pos = buf.get("win_pos")
        pos_f = win_pos[-1:] if win_pos is not None else None
        out_long = fsb_decode_token(self, Q, cache, sig, pos_t, pos_f=pos_f, sr=out_short)
        if out_long.dim() == 3:
            return self.W_o(out_long)
        out = out_short + self.long_range_scale.unsqueeze(0) * out_long
        return self.W_o(out.transpose(1, 2).contiguous().view(B, N, self.d_model))

    def graph_decode_layer(self, x, buf):
        from ..utils import trunk_triton as _tt
        B, N, _ = x.shape
        H, D = self.n_heads, self.head_dim
        Q, K, V = self._qkv_decode(x, B, N, H, D)
        kbuf = buf["kbuf"]; vbuf = buf["vbuf"]
        pos_t = buf["pos_t"]
        sigma = buf.get("sigma")
        if sigma is None:
            sigma = self.sigma
        out_short = self._sr_kv_and_decode(Q, K, V, kbuf, vbuf, buf, sigma)
        pool = buf["pool"]
        if pool.get("empty", False):
            out_long = Q.new_zeros(B, H, 1, D)
        elif _tt.fsl_lr_decode_combine_ok(self, Q, kbuf, vbuf, out_short, pool, pos_t, sigma):
            return self.W_o(_tt.fsl_lr_decode_combine(self, Q, kbuf, vbuf, out_short, pool, pos_t, sigma))
        else:
            out_long = self._lr_decode_from_pool_buf(Q, kbuf, vbuf, pool, pos_t, sigma)
        out = out_short + self.long_range_scale.unsqueeze(0) * out_long
        return self.W_o(out.transpose(1, 2).contiguous().view(B, N, self.d_model))

    def _lr_decode_from_pool_buf(self, q_t, kbuf, vbuf, pool, pos_t, sigma):
        B, H, _, D = q_t.shape
        s = self.cell_stride
        NEG = -1e9
        if pool.get("dense", False):
            Kbar = pool["Kbar_all"]
            ceil_c = pool["ceil_all"]
            M_cap = Kbar.shape[2]
            m_vis_t = pool["m_vis_t"]
            c_idx = torch.arange(M_cap, device=q_t.device, dtype=torch.int32)
            ok_cell = (c_idx < m_vis_t.view(1)).view(1, 1, M_cap)
            cen = (c_idx.to(torch.float32) + 0.5) * s
            lam = self.sel_scale.to(torch.float32).view(1, H, 1)
            sigv = sigma.to(torch.float32).view(1, H, 1)
            content = torch.einsum("bhd,bhcd->bhc", q_t[:, :, 0, :].float(), Kbar) * self.scale
            dist = (pos_t.to(torch.float32) - cen).abs()
            erf_fac = torch.erf(dist.view(1, 1, M_cap) / (2.0 * sigv))
            score = content if self.sel_content_only else content + lam * ceil_c * erf_fac
            score = score.masked_fill(~ok_cell, NEG)
            kk = min(self.cell_topk, M_cap)
            sel = score.topk(kk, dim=-1).indices
            pick_ok = ok_cell.expand(B, H, M_cap).gather(-1, sel)
            selt = sel.to(torch.int32).masked_fill(~pick_ok, -1).unsqueeze(2).contiguous()
            from . import cell_select_triton as cst
            out, _ = cst.cell_indexed_attn_fwd_posp(q_t, kbuf, vbuf, selt, cell_size=s,
                                                    query_block=self._cell_block, scale=self.scale,
                                                    bias=None, q_pos_ptr=pos_t)
            return self.lr_norm(out.to(q_t.dtype))
        M_vis = pool["M_vis"]; C = pool["C"]
        Kbar_cand = pool["Kbar_cand"]; ceil_cand = pool["ceil_cand"]; cen_cand = pool["cen_cand"]
        ok_cell = pool["ok_cell"]; tok_idx = pool.get("tok_idx")
        lam = self.sel_scale.to(torch.float32).view(1, H, 1)
        sigv = sigma.to(torch.float32).view(1, H, 1)
        sc = self.scale
        content = torch.einsum("bhd,bhcd->bhc", q_t[:, :, 0, :].float(), Kbar_cand) * sc
        dist = (pos_t.to(torch.float32) - cen_cand).abs()
        erf = torch.erf(dist / (2.0 * sigv))
        score = content + lam * ceil_cand * erf
        score = score.masked_fill(~ok_cell, NEG)
        kk = min(self.cell_topk, M_vis)
        sel = score.topk(min(kk, C), dim=-1).indices
        cand_c = pool.get("cand_c")
        if cand_c is not None and min(kk, C) <= 16 and q_t.is_cuda:
            from . import cell_select_triton as cst
            if cst.triton is not None:
                # the selected cells the _lr_decode_step_fused scoring above picks, attended by one indexed
                # kernel directly from the KV buffers (the torch gather/mask/softmax path below is the fallback)
                pick_ok = torch.gather(ok_cell, -1, sel)
                gcell = torch.gather(cand_c, -1, sel)
                selt = gcell.to(torch.int32).masked_fill(~pick_ok, -1).unsqueeze(2).contiguous()
                out, _ = cst.cell_indexed_attn_fwd_posp(q_t, kbuf, vbuf, selt, cell_size=s,
                                                        query_block=self._cell_block, scale=self.scale,
                                                        bias=None, q_pos_ptr=pos_t)
                return self.lr_norm(out.to(q_t.dtype))
        cellmask = torch.zeros(B, H, C, dtype=torch.bool, device=q_t.device)
        cellmask.scatter_(-1, sel, True)
        cellmask = cellmask & ok_cell
        gidx = tok_idx.unsqueeze(-1).expand(B, H, C * s, D)
        Kc = torch.gather(kbuf, 2, gidx)
        Vc = torch.gather(vbuf, 2, gidx)
        scores = torch.einsum("bhd,bhkd->bhk", q_t[:, :, 0, :], Kc).float() * self.scale
        tokcell = cellmask.repeat_interleave(s, dim=-1)
        allow = tokcell & (tok_idx < pos_t)
        scores = scores.masked_fill(~allow, NEG)
        w = torch.softmax(scores, dim=-1).to(Vc.dtype)
        ob = torch.einsum("bhk,bhkd->bhd", w, Vc).unsqueeze(2)
        ob = ob * allow.any(dim=-1).view(B, H, 1, 1).to(ob.dtype)
        return self.lr_norm(ob.to(q_t.dtype))

    def _rope_abs_t(self, x, pos0_t):
        n = x.shape[2]
        idx = pos0_t.reshape(1) + torch.arange(n, device=x.device)
        cos = self.rope.cos_cached.index_select(0, idx)
        sin = self.rope.sin_cached.index_select(0, idx)
        return _apply_rotary_emb(x, cos, sin)

    def _lr_prefill_blocks(self, Q, cache, sigma, N):
        K, V = cache.lr_tokens()
        if self.fsb_enabled:
            from .fsb_tree import fsb_long_range
            kvp = None
            kb, vb = getattr(cache, "_k_buf", None), getattr(cache, "_v_buf", None)
            if kb is not None and vb is not None and getattr(cache, "_len", -1) == N and kb.shape[2] > N:
                kvp = (kb, vb)
            return fsb_long_range(self, Q, K[:, :, :N], V[:, :, :N], sigma, N, kv_padded=kvp)
        if not self._is_dense_lr():
            return self._cell_score_On_chunked_strict(Q, K, V, sigma, N)
        if bool(self._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0)):
            return self._lr_cell_topk(Q, K[:, :, :N], V[:, :, :N], sigma, N)
        B, H, _, D = Q.shape
        s = self.cell_stride
        dev = Q.device
        NEG = -1e9
        Bq = min(self._cell_block, N)
        nb = (N + Bq - 1) // Bq
        NBCH = self._prune_cfg("cell_prune_blkchunk", "FS_PRUNE_BLKCHUNK", 16) or nb
        kk_max = self.cell_topk
        lam = self.sel_scale.to(torch.float32).view(1, H, 1, 1)
        sigv = sigma.to(torch.float32).view(1, H, 1, 1)
        sc = self.scale
        Kbar_all, ceil_all, ncell = cache.completed_cells()
        off = torch.arange(s, device=dev)
        arbq = torch.arange(Bq, device=dev)
        out = Q.new_zeros(B, H, nb * Bq, D)
        for b0 in range(0, nb, NBCH):
            b1 = min(b0 + NBCH, nb)
            for b in range(b0, b1):
                blk_start = b * Bq
                M_vis = blk_start // s
                if M_vis <= 0 or ncell <= 0:
                    continue
                M_vis = min(M_vis, ncell)
                q0 = blk_start
                q1 = min(blk_start + Bq, N)
                nq = q1 - q0
                if nq <= 0:
                    continue
                qb = Q[:, :, q0:q1, :]
                qposv = (blk_start + arbq[:nq]).to(torch.float32)
                cen = (torch.arange(M_vis, device=dev, dtype=torch.float32) + 0.5) * s
                Kbar = Kbar_all[:, :, :M_vis, :]
                ceil_c = ceil_all[:, :, :M_vis]
                content = torch.einsum("bhqd,bhcd->bhqc", qb.float(), Kbar) * sc
                dist = (qposv.view(1, 1, nq, 1) - cen.view(1, 1, 1, M_vis)).abs()
                erf_fac = torch.erf(dist / (2.0 * sigv))
                score = content + lam * ceil_c.unsqueeze(2) * erf_fac
                kk = min(kk_max, M_vis)
                topi = score.topk(kk, dim=-1).indices
                tok_idx = (topi.unsqueeze(-1) * s + off).reshape(B, H, nq, kk * s)
                gidx = tok_idx.reshape(B, H, nq * kk * s, 1).expand(-1, -1, -1, D)
                Kc = torch.gather(K, 2, gidx).reshape(B, H, nq, kk * s, D)
                Vc = torch.gather(V, 2, gidx).reshape(B, H, nq, kk * s, D)
                sscore = torch.einsum("bhqd,bhqkd->bhqk", qb, Kc).float() * sc
                qabs = (blk_start + arbq[:nq]).view(1, 1, nq, 1)
                allow = tok_idx < qabs
                sscore = sscore.masked_fill(~allow, NEG)
                w = torch.softmax(sscore, dim=-1).to(Vc.dtype)
                ob = torch.einsum("bhqk,bhqkd->bhqd", w, Vc)
                ob = ob * allow.any(dim=-1, keepdim=True).to(ob.dtype)
                out[:, :, q0:q1, :] = ob.to(Q.dtype)
        if nb * Bq > N:
            out = out[:, :, :N, :]
        return self.lr_norm(out.to(Q.dtype))
