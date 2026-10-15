"""Prefill long-range readout with the window combine fused in (FS-B and FS-L, inference only).

Replaces, for a prefill layer under bf16 autocast:
    o = indexed_cell_attention(Q, K, V, sel)                                  # bf16 (B, H, N, D)
    y = (sr + long_range_scale * lr_norm(o)).transpose(1, 2).contiguous()     # fp32, cast to bf16 by W_o's autocast
with one Triton kernel that returns y as the bf16 (B, N, H * D) tensor W_o reads.

One program owns G consecutive queries of one (batch, head); G divides the 128-query block, so all its queries share
the block's cell-validity bound. It sorts the G * K selected cells once, drops invalid and repeated ones, and walks
the distinct cells in ascending order: each cell's 32 keys and values are loaded once for the G queries, scored with
one tensor-core dot, and folded into a per-query online softmax restricted to the queries that selected the cell.
The epilogue rounds o to bf16 (as the old kernel's output), applies lr_norm, the per-head long_range_scale and the
window output sr in fp32 with torch's operation order (no FMA contraction), and stores bf16.

Numerics (tolerance class): the selection is untouched; o differs from the old kernel by summation order (and is more
accurate: the old kernel rounded each p * v product to bf16), and the RMS mean by its reduction order."""
import os

import torch

from ..utils.compile_guard import compiling

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["N", "n_cells"])
    def _readout_combine_kernel(
        Q, K, V, SEL, SR, W, LRS, OUT, SCR,
        sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, ssb, ssh, ssn, ssk, srb, srh, srn,
        N, n_cells, H, scale, eps,
        D: tl.constexpr, CS: tl.constexpr, BQ: tl.constexpr, KK: tl.constexpr, KS: tl.constexpr,
        G: tl.constexpr,
    ):
        pg = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh - b * H
        r0 = pg * G
        rows = r0 + tl.arange(0, G)
        row_ok = rows < N
        bs = (r0 // BQ) * BQ
        slots = tl.arange(0, KS)
        dcol = tl.arange(0, D)

        sel = tl.load(SEL + b * ssb + h * ssh + rows[:, None] * ssn + slots[None, :] * ssk,
                      mask=row_ok[:, None] & (slots[None, :] < KK), other=-1).to(tl.int32)
        ok = (sel >= 0) & (sel < n_cells) & ((sel + 1) * CS <= bs)
        selv = tl.where(ok, sel, 1073741824)

        # distinct valid cells, ascending, compacted into this program's scratch slice
        srt = tl.sort(tl.reshape(selv, (G * KS,)))
        idx = tl.arange(0, G * KS)
        scr = SCR + (bh.to(tl.int64) * tl.num_programs(0) + pg) * (G * KS)
        tl.store(scr + idx, srt)
        tl.debug_barrier()
        prev = tl.load(scr + idx - 1, mask=idx > 0, other=-1)
        is_new = (srt != prev) & (srt < 1073741824)
        pos = tl.cumsum(is_new.to(tl.int32), axis=0) - 1
        n_u = tl.sum(is_new.to(tl.int32), axis=0)
        tl.debug_barrier()
        tl.store(scr + pos, srt, mask=is_new)
        tl.debug_barrier()

        q = tl.load(Q + b * sqb + h * sqh + rows[:, None] * sqn + dcol[None, :], mask=row_ok[:, None], other=0.0)
        m_i = tl.full((G,), float("-inf"), tl.float32)
        l_i = tl.zeros((G,), tl.float32)
        acc = tl.zeros((G, D), tl.float32)
        toks = tl.arange(0, CS)
        kbase = K + b * skb + h * skh + dcol[None, :]
        vbase = V + b * svb + h * svh + dcol[None, :]
        for t in range(0, n_u):
            u = tl.load(scr + t)
            mem = tl.sum((selv == u).to(tl.int32), axis=1) > 0
            tok = (u * CS + toks).to(tl.int64)
            kt = tl.load(kbase + tok[:, None] * skn)
            s = tl.dot(q, tl.trans(kt)) * scale
            s = tl.where(mem[:, None], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_use)
            p = tl.exp(s - m_use[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            vt = tl.load(vbase + tok[:, None] * svn)
            acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
            m_i = m_new

        o = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
        ob = o.to(Q.dtype.element_ty).to(tl.float32)
        ms = tl.sum(ob * ob, axis=1) * (1.0 / D)
        r = tl.math.rsqrt(ms + eps)
        w = tl.load(W + dcol).to(tl.float32)
        g = tl.load(LRS + h).to(tl.float32)
        lr = (ob * r[:, None]) * w[None, :]
        sr = tl.load(SR + b * srb + h * srh + rows[:, None] * srn + dcol[None, :], mask=row_ok[:, None],
                     other=0.0).to(tl.float32)
        y = sr + g * lr
        optr = OUT + (b * N + rows[:, None]).to(tl.int64) * (H * D) + h * D + dcol[None, :]
        tl.store(optr, y.to(OUT.dtype.element_ty), mask=row_ok[:, None])


_G = int(os.environ.get("FSL_READOUT_G", "64"))
_WARPS = int(os.environ.get("FSL_READOUT_WARPS", "4"))
_STAGES = int(os.environ.get("FSL_READOUT_STAGES", "2"))


def readout_ok(Q, K, V, sel, sr, *, cell_size, query_block, bias=None, q_pos=0) -> bool:
    if compiling() or triton is None or os.environ.get("FSL_READOUT_FUSED", "1") == "0" or torch.is_grad_enabled():
        return False                          # (under torch.compile the eager readout and combine, as before)
    if bias is not None or q_pos != 0 or sr is None or not Q.is_cuda:
        return False
    if not torch.is_autocast_enabled():
        return False
    adt = torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else torch.get_autocast_gpu_dtype()
    B, H, N, D = Q.shape
    G = min(_G, query_block)
    # G queries per program: tl.arange needs a power of two and tl.dot at least 16 rows. Prompts shorter than the
    # 128-query block have query_block = N (2-63 tokens other than 16 and 32 failed to compile), and FSL_READOUT_G
    # may be anything: those take the eager readout.
    return (adt == torch.bfloat16 and Q.dtype == K.dtype == V.dtype == sr.dtype == torch.bfloat16
            and G >= 16 and (G & (G - 1)) == 0 and query_block % G == 0
            and D in (32, 64, 128) and cell_size in (16, 32, 64) and N > 1
            and sel.dim() == 4 and sel.shape[0] == B and sel.shape[1] == H and sel.shape[2] >= N
            and 1 <= sel.shape[3] <= 8 and sel.dtype in (torch.int32, torch.int64)
            and K.shape[:2] == (B, H) and K.shape[2] >= N and V.shape == K.shape and tuple(sr.shape) == (B, H, N, D)
            and Q.stride(-1) == K.stride(-1) == V.stride(-1) == sr.stride(-1) == 1)


def readout_combine(Q, K, V, sel, sr, norm_w, lrs, eps, *, scale, cell_size, query_block):
    """y (B, N, H * D) bf16 = bf16(sr + lrs[h] * lr_norm(bf16(cell readout))) for every query, from one kernel."""
    B, H, N, D = Q.shape
    G = min(_G, query_block)
    KK = sel.shape[3]
    KS = triton.next_power_of_2(KK)
    n_kv = K.shape[2]
    ng = triton.cdiv(N, G)
    out = torch.empty(B, N, H * D, device=Q.device, dtype=torch.bfloat16)
    scr = torch.empty(B * H * ng * G * KS, device=Q.device, dtype=torch.int32)
    lrs = lrs.reshape(-1)
    norm_w = norm_w.reshape(-1)
    _readout_combine_kernel[(ng, B * H)](
        Q, K, V, sel, sr, norm_w, lrs, out, scr,
        Q.stride(0), Q.stride(1), Q.stride(2), K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2), sel.stride(0), sel.stride(1), sel.stride(2), sel.stride(3),
        sr.stride(0), sr.stride(1), sr.stride(2),
        N, (n_kv + cell_size - 1) // cell_size, H, float(scale), float(eps),
        D=D, CS=cell_size, BQ=query_block, KK=KK, KS=KS, G=G, num_warps=_WARPS, num_stages=_STAGES,
        enable_fp_fusion=False)
    return out
