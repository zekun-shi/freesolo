"""Bit-exact fused replacements for eager inference glue (no-grad, CUDA):

- rope(x, cos, sin): _apply_rotary_emb in one pass. Every product, difference and sum is rounded to the dtype torch
  would compute it in (promote(x, cos)): bf16 cos (bf16 model) rounds each op to bf16, fp32 cos (fp32 parameters under
  autocast) keeps fp32. The result can be cast straight to the dtype autocast would hand SDPA.
- silu_mul(g, u): F.silu(g) * u with torch's own formula (x / (1 + expf(-x)) in fp32 with libdevice expf and a
  correctly rounded, subnormal-preserving division, rounded to the input dtype, then the product rounded to it): one
  pass instead of two (all 65536 bf16 gate values are checked in tests/test_trunk_fused.py).
- rms_norm_cast(x, w, eps, dtype): experiments/scaling/model.RMSNorm under autocast followed by the cast autocast
  applies at the next linear. For rows of 256-2048 values (power of two) and at least 16 rows it is one pass that sums
  x^2 in exactly the order of torch's CUDA mean for that shape; otherwise x.pow(2) is one fused square, the mean is
  torch's own reduction, and rsqrt, x * r * w and the cast are one fused pass. The one-pass kernel unrolls the row
  (D / 128 float4 vectors per lane) and its compile time grows quickly with D: 1.3 s at 256, 2.1 s at 1024 and 20 s at
  2048 on a cold Triton cache (A100, Triton 3.1); 4096 did not finish compiling in 13 minutes, so wider rows take the
  fallback (1.6 s to compile at 4096).

All kernels compile with enable_fp_fusion=False, so no product is contracted into an FMA.

The *_ok gates take these paths only without autograd and outside torch.compile (compile_guard; dynamo would trace
the launches), and the norm and silu * mul only on inputs of at least MIN_ROWS rows: on fewer rows (single-token
decode) the extra Triton launches cost more host time than the eager ops they replace."""
import os

import torch

from .compile_guard import compiling

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as _libdevice
except ImportError:  # pragma: no cover
    triton = None
    tl = None

_RND = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}

if triton is not None:

    @triton.jit
    def _rnd(x, RND: tl.constexpr):
        # round an fp32 value to the compute dtype of the emulated torch op (RND 0: fp32, 1: bf16, 2: fp16)
        if RND == 1:
            x = x.to(tl.bfloat16).to(tl.float32)
        elif RND == 2:
            x = x.to(tl.float16).to(tl.float32)
        return x

    @triton.jit(do_not_specialize=["N"])
    def _rope_kernel(X, C, S, O, N, H, sxb, sxh, sxn, scn, ssn, sob, soh, son,
                     HALF: tl.constexpr, BN: tl.constexpr, RND: tl.constexpr):
        pn = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh - b * H
        rows = pn * BN + tl.arange(0, BN)
        cols = tl.arange(0, HALF)
        m = rows[:, None] < N
        xp = X + b * sxb + h * sxh + rows[:, None] * sxn + cols[None, :]
        x1 = tl.load(xp, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(xp + HALF, mask=m, other=0.0).to(tl.float32)
        c = tl.load(C + rows[:, None] * scn + cols[None, :], mask=m, other=0.0).to(tl.float32)
        s = tl.load(S + rows[:, None] * ssn + cols[None, :], mask=m, other=0.0).to(tl.float32)
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1), one rounding per torch op
        r1 = _rnd(_rnd(x1 * c, RND) - _rnd(x2 * s, RND), RND)
        r2 = _rnd(_rnd(x2 * c, RND) + _rnd(x1 * s, RND), RND)
        op = O + b * sob + h * soh + rows[:, None] * son + cols[None, :]
        tl.store(op, r1.to(O.dtype.element_ty), mask=m)
        tl.store(op + HALF, r2.to(O.dtype.element_ty), mask=m)

    @triton.jit
    def _div_rn_f32(a, b):
        # IEEE division that keeps subnormals (libdevice's div_rn is compiled flush-to-zero in Triton)
        return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True,
                                         pack=1)

    @triton.jit
    def _mul_rn_f32(a, b):
        return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True,
                                         pack=1)

    @triton.jit
    def _silu_mul_kernel(G, U, O, n, BLOCK: tl.constexpr, RND: tl.constexpr):
        # torch: silu(x) = x / (1 + exp(-x)) in fp32 (CUDA expf, IEEE division), rounded to the tensor dtype; the
        # product with u in fp32, rounded again. Subnormal x and products keep torch's (non-flushing) arithmetic.
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(U + offs, mask=m, other=0.0).to(tl.float32)
        sg = _rnd(_div_rn_f32(g, 1.0 + _libdevice.exp(-g)), RND)
        tl.store(O + offs, _mul_rn_f32(sg, u).to(O.dtype.element_ty), mask=m)

    @triton.jit
    def _square_kernel(X, Y, n, BLOCK: tl.constexpr):
        # x.pow(2) of the fp32 input autocast hands pow (torch computes base * base for exponent 2)
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + offs, x * x, mask=m)

    @triton.jit
    def _rms_out_kernel(X, MS, W, O, eps, D, BLOCK: tl.constexpr):
        # (x * rsqrt(ms + eps)) * w in fp32 (each product rounded, as torch's two multiplies), then the output cast
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        m = cols < D
        x = tl.load(X + row * D + cols, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        ms = tl.load(MS + row)
        r = tl.math.rsqrt(ms + eps)
        y = (x * r) * w
        tl.store(O + row * D + cols, y.to(O.dtype.element_ty), mask=m)


    @triton.jit
    def _rms_fused_kernel(X, W, O, eps, J: tl.constexpr):
        # One pass per row of D = 128 * J values: the mean of x^2 in the exact order of torch's CUDA mean for this
        # shape (rows >= 16, 256 <= D <= 2048: 32 lanes per row, each loading float4 vectors t, t + 32, ...,
        # accumulating each vector lane sequentially from 0, then ((a0 + a1) + a2) + a3, then the lanes' shuffle-down
        # tree with offsets 1, 2, 4, 8, 16, then * (1 / D)), then rsqrt(mean + eps), (x * r) * w and the cast.
        row = tl.program_id(0).to(tl.int64)
        jj = tl.arange(0, J)
        tt = tl.arange(0, 32)
        ii = tl.arange(0, 4)
        offs = jj[:, None, None] * 128 + tt[None, :, None] * 4 + ii[None, None, :]
        x = tl.load(X + row * (J * 128) + offs).to(tl.float32)
        sq = x * x
        acc = tl.zeros((32, 4), tl.float32)
        for j in tl.static_range(J):
            acc = acc + tl.sum(tl.where(jj[:, None, None] == j, sq, 0.0), axis=0)
        v = tl.sum(tl.where(ii[None, :] == 0, acc, 0.0), axis=1)
        for i in tl.static_range(1, 4):
            v = v + tl.sum(tl.where(ii[None, :] == i, acc, 0.0), axis=1)
        v = tl.sum(tl.reshape(v, (16, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (8, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (4, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (2, 2)), axis=1)
        tot = tl.sum(v, axis=0)
        r = tl.math.rsqrt(tot * (1.0 / (J * 128)) + eps)
        w = tl.load(W + offs).to(tl.float32)
        y = (x * r) * w
        tl.store(O + row * (J * 128) + offs, y.to(O.dtype.element_ty))

    @triton.jit
    def _rms_1row_kernel(X, W, O, eps, D: tl.constexpr, T: tl.constexpr, V: tl.constexpr):
        # Single-row (rows == 1) one-pass RMSNorm reproducing ATen's reduce_kernel<512, 1, MeanOps> reduction order
        # (D = 128: V = 1, T = 128; D in 256..2048: V = 4, T = D // 4; inter-warp halving tree first, then intra-warp
        # shfl_down offsets 1, 2, 4, 8, 16), followed by rsqrt(mean + eps), (x * r) * w and the output cast.
        NW: tl.constexpr = T // 32
        ww = tl.arange(0, NW)
        tt = tl.arange(0, 32)
        if V == 4:
            ii = tl.arange(0, 4)
            offs = ww[:, None, None] * 128 + tt[None, :, None] * 4 + ii[None, None, :]
            x = tl.load(X + offs).to(tl.float32)
            sq = x * x
            v = tl.sum(tl.where(ii[None, None, :] == 0, sq, 0.0), axis=2)
            for i in tl.static_range(1, 4):
                v = v + tl.sum(tl.where(ii[None, None, :] == i, sq, 0.0), axis=2)
        else:
            offs = ww[:, None] * 32 + tt[None, :]
            x = tl.load(X + offs).to(tl.float32)
            v = x * x
        v = tl.sum(tl.reshape(v, (2, NW // 2, 32)), axis=0)
        if NW >= 4:
            v = tl.sum(tl.reshape(v, (2, NW // 4, 32)), axis=0)
        if NW >= 8:
            v = tl.sum(tl.reshape(v, (2, NW // 8, 32)), axis=0)
        if NW >= 16:
            v = tl.sum(tl.reshape(v, (2, NW // 16, 32)), axis=0)
        v = tl.reshape(v, (32,))
        v = tl.sum(tl.reshape(v, (16, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (8, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (4, 2)), axis=1)
        v = tl.sum(tl.reshape(v, (2, 2)), axis=1)
        tot = tl.sum(v, axis=0)
        r = tl.math.rsqrt(tot * (1.0 / D) + eps)
        w = tl.load(W + offs).to(tl.float32)
        y = (x * r) * w
        tl.store(O + offs, y.to(O.dtype.element_ty))

    @triton.jit
    def _sr_kv_rope_kernel(
        Q, K, V, Kbuf, Vbuf, Cos, Sin, PosT, WinIdx,
        Qrot, KwOut, VwOut,
        Lw, H: tl.constexpr, HALF: tl.constexpr,
        sqb, sqh, skb, skh, svb, svh,
        skbb, skbh, skbn, svbb, svbh, svbn,
        scn, ssn,
        sqrb, sqrh, skwb, skwh, skwl, svwb, svwh, svwl,
        BN: tl.constexpr, RND: tl.constexpr, USE_ROPE: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh - b * H
        cols = tl.arange(0, HALF)
        pos_t = tl.load(PosT)
        if pid_n == 0:
            kp = K + b * skb + h * skh + cols
            k1 = tl.load(kp)
            k2 = tl.load(kp + HALF)
            kbp = Kbuf + b * skbb + h * skbh + pos_t * skbn + cols
            tl.store(kbp, k1)
            tl.store(kbp + HALF, k2)

            vp = V + b * svb + h * svh + cols
            v1 = tl.load(vp)
            v2 = tl.load(vp + HALF)
            vbp = Vbuf + b * svbb + h * svbh + pos_t * svbn + cols
            tl.store(vbp, v1)
            tl.store(vbp + HALF, v2)

            qp = Q + b * sqb + h * sqh + cols
            q1 = tl.load(qp).to(tl.float32)
            q2 = tl.load(qp + HALF).to(tl.float32)
            if USE_ROPE:
                c = tl.load(Cos + pos_t * scn + cols).to(tl.float32)
                s = tl.load(Sin + pos_t * ssn + cols).to(tl.float32)
                rq1 = _rnd(_rnd(q1 * c, RND) - _rnd(q2 * s, RND), RND)
                rq2 = _rnd(_rnd(q2 * c, RND) + _rnd(q1 * s, RND), RND)
            else:
                rq1, rq2 = q1, q2
            qrp = Qrot + b * sqrb + h * sqrh + cols
            tl.store(qrp, rq1.to(Qrot.dtype.element_ty))
            tl.store(qrp + HALF, rq2.to(Qrot.dtype.element_ty))

        rows = pid_n * BN + tl.arange(0, BN)
        m = rows < Lw
        w_idx = tl.load(WinIdx + rows, mask=m, other=0)
        is_cur = (w_idx == pos_t)[:, None]

        kbp_w = Kbuf + b * skbb + h * skbh + w_idx[:, None] * skbn + cols[None, :]
        kw1 = tl.load(kbp_w, mask=m[:, None], other=0.0)
        kw2 = tl.load(kbp_w + HALF, mask=m[:, None], other=0.0)
        kp0 = K + b * skb + h * skh + cols[None, :]
        k1_cur = tl.load(kp0)
        k2_cur = tl.load(kp0 + HALF)
        kw1 = tl.where(is_cur, k1_cur, kw1).to(tl.float32)
        kw2 = tl.where(is_cur, k2_cur, kw2).to(tl.float32)

        if USE_ROPE:
            r_idx = (pos_t - (Lw - 1)) + rows
            c = tl.load(Cos + r_idx[:, None] * scn + cols[None, :], mask=m[:, None], other=0.0).to(tl.float32)
            s = tl.load(Sin + r_idx[:, None] * ssn + cols[None, :], mask=m[:, None], other=0.0).to(tl.float32)
            rkw1 = _rnd(_rnd(kw1 * c, RND) - _rnd(kw2 * s, RND), RND)
            rkw2 = _rnd(_rnd(kw2 * c, RND) + _rnd(kw1 * s, RND), RND)
        else:
            rkw1, rkw2 = kw1, kw2
        kw_out_p = KwOut + b * skwb + h * skwh + rows[:, None] * skwl + cols[None, :]
        tl.store(kw_out_p, rkw1.to(KwOut.dtype.element_ty), mask=m[:, None])
        tl.store(kw_out_p + HALF, rkw2.to(KwOut.dtype.element_ty), mask=m[:, None])

        vbp_w = Vbuf + b * svbb + h * svbh + w_idx[:, None] * svbn + cols[None, :]
        vw1 = tl.load(vbp_w, mask=m[:, None], other=0.0)
        vw2 = tl.load(vbp_w + HALF, mask=m[:, None], other=0.0)
        vp0 = V + b * svb + h * svh + cols[None, :]
        v1_cur = tl.load(vp0)
        v2_cur = tl.load(vp0 + HALF)
        vw1 = tl.where(is_cur, v1_cur, vw1)
        vw2 = tl.where(is_cur, v2_cur, vw2)
        vw_out_p = VwOut + b * svwb + h * svwh + rows[:, None] * svwl + cols[None, :]
        tl.store(vw_out_p, vw1, mask=m[:, None])
        tl.store(vw_out_p + HALF, vw2, mask=m[:, None])

    @triton.jit
    def _sr_softmax_cast_kernel(
        Scores, PosT, WinPos, Sigma, WOut,
        scale, W_win, Lw,
        H: tl.constexpr, UNIFORM: tl.constexpr, LOG2_L: tl.constexpr,
    ):
        row = tl.program_id(0)
        NIT: tl.constexpr = (1 << LOG2_L) // 32
        ii = tl.arange(0, NIT)
        tt = tl.arange(0, 32)
        idx = ii[:, None] * 32 + tt[None, :]
        valid = idx < Lw

        s_raw = tl.load(Scores + row * Lw + idx, mask=valid, other=0.0).to(tl.float32)
        s_scaled = s_raw * scale

        pos_t = tl.load(PosT).to(tl.float32)
        pos_w = tl.load(WinPos + idx, mask=valid, other=0.0).to(tl.float32)
        dist = tl.abs(pos_t - pos_w)
        band = dist <= W_win

        if UNIFORM:
            m = tl.where(band, 0.0, float("-inf"))
        else:
            sig = tl.load(Sigma + (row % H)).to(tl.float32)
            arg = _div_rn_f32(dist, 2.0 * sig)
            erfc_v = _libdevice.erfc(arg)
            clamped = tl.where(erfc_v < 1e-10, 1e-10, erfc_v)
            bias = _libdevice.log(clamped)
            m = tl.where(band, bias, float("-inf"))

        el = tl.where(valid, s_scaled + m, float("-inf"))
        mx = tl.max(el, axis=0)
        mx = tl.max(mx, axis=0)

        ex = tl.where(valid, _libdevice.exp(el - mx), 0.0)
        acc = tl.sum(tl.where(ii[:, None] == 0, ex, 0.0), axis=0)
        for it in tl.static_range(1, NIT):
            acc = acc + tl.sum(tl.where(ii[:, None] == it, ex, 0.0), axis=0)
        acc = tl.sum(tl.reshape(acc, (2, 16)), axis=0)
        acc = tl.sum(tl.reshape(acc, (2, 8)), axis=0)
        acc = tl.sum(tl.reshape(acc, (2, 4)), axis=0)
        acc = tl.sum(tl.reshape(acc, (2, 2)), axis=0)
        tot = tl.sum(acc, axis=0)

        w = _div_rn_f32(ex, tot)
        tl.store(WOut + row * Lw + idx, w.to(WOut.dtype.element_ty), mask=valid)

    @triton.jit
    def _fsl_pool_refresh_kernel(
        TOPP, KBAR_ALL, CEIL_ALL,
        KBAR_OUT, CEIL_OUT, CEN_OUT, OK_OUT, CAND_OUT,
        skab, skah, skac, scab, scah,
        skob, skoh, skoc, scob, scoh,
        hi, blk_start_f, s_f, Pk, C,
        H: tl.constexpr, D: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        h = bh - b * H
        c = tl.arange(0, BLOCK_C)
        cmask = c < C

        topp_val = tl.load(TOPP + bh * Pk + c, mask=c < Pk, other=0).to(tl.int64)
        near_val = hi.to(tl.int64) - (c - Pk).to(tl.int64)
        cand = tl.where(c < Pk, topp_val, near_val)
        cand_valid = (cand >= 0) & cmask
        cc = tl.maximum(cand, 0)

        cen = (cc.to(tl.float32) + 0.5) * s_f
        cend = (cc.to(tl.float32) + 1.0) * s_f
        vis = (cend <= blk_start_f) & cand_valid

        eq = (cc[:, None] == cc[None, :]) & (c[None, :] < c[:, None]) & cmask[:, None] & cmask[None, :]
        dup = tl.sum(eq.to(tl.int32), axis=1) > 0
        ok = vis & (~dup)

        out_off = b * scob + h * scoh + c
        tl.store(CEIL_OUT + out_off, tl.load(CEIL_ALL + b * scab + h * scah + cc, mask=cmask, other=0.0), mask=cmask)
        tl.store(CEN_OUT + out_off, cen, mask=cmask)
        tl.store(OK_OUT + out_off, ok, mask=cmask)
        tl.store(CAND_OUT + out_off, cc, mask=cmask)

        d = tl.arange(0, BLOCK_D)
        dmask = d < D
        kbar_vals = tl.load(
            KBAR_ALL + b * skab + h * skah + cc[:, None] * skac + d[None, :],
            mask=cmask[:, None] & dmask[None, :],
            other=0.0,
        )
        tl.store(
            KBAR_OUT + b * skob + h * skoh + c[:, None] * skoc + d[None, :],
            kbar_vals,
            mask=cmask[:, None] & dmask[None, :],
        )

    @triton.jit
    def _fsl_lr_decode_combine_kernel(
        Q, KBAR, CEIL, CEN, OK, CAND, POS, LAM, SIG,
        K, V, SR, W_NORM, LRS, O,
        sqb, sqh, skbb, skbh, skbc, scb, sch,
        skb, skh, skn, svb, svh, svn, srb, srh,
        N_KV, sc, eps, C,
        H: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
        CELL_SIZE: tl.constexpr, QUERY_BLOCK: tl.constexpr, K_SELECT: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
        NORM_FP32: tl.constexpr, W_BF16: tl.constexpr, LRS_BF16: tl.constexpr,
        AUTOCAST_BF16: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        h = bh - b * H
        c = tl.arange(0, BLOCK_C)
        cmask = c < C

        q_base = Q + b * sqb + h * sqh
        kb_base = KBAR + b * skbb + h * skbh + c * skbc

        if AUTOCAST_BF16:
            dot = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for tx in tl.static_range(0, 16):
                q0 = tl.load(q_base + (0 + tx)).to(tl.bfloat16).to(tl.float32)
                q1 = tl.load(q_base + (16 + tx)).to(tl.bfloat16).to(tl.float32)
                q2 = tl.load(q_base + (32 + tx)).to(tl.bfloat16).to(tl.float32)
                q3 = tl.load(q_base + (48 + tx)).to(tl.bfloat16).to(tl.float32)

                k0 = tl.load(kb_base + (0 + tx), mask=cmask, other=0.0).to(tl.bfloat16).to(tl.float32)
                k1 = tl.load(kb_base + (16 + tx), mask=cmask, other=0.0).to(tl.bfloat16).to(tl.float32)
                k2 = tl.load(kb_base + (32 + tx), mask=cmask, other=0.0).to(tl.bfloat16).to(tl.float32)
                k3 = tl.load(kb_base + (48 + tx), mask=cmask, other=0.0).to(tl.bfloat16).to(tl.float32)

                t_tx = (((q0 * k0) + (q1 * k1)) + (q2 * k2)) + (q3 * k3)
                dot = dot + t_tx
            content = (dot.to(tl.bfloat16) * sc.to(tl.bfloat16)).to(tl.float32)
        else:
            acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
            acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
            acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
            acc3 = tl.zeros((BLOCK_C,), dtype=tl.float32)

            for i in tl.static_range(0, 16):
                q0 = tl.load(q_base + (4 * i + 0)).to(tl.float32)
                q1 = tl.load(q_base + (4 * i + 1)).to(tl.float32)
                q2 = tl.load(q_base + (4 * i + 2)).to(tl.float32)
                q3 = tl.load(q_base + (4 * i + 3)).to(tl.float32)

                k0 = tl.load(kb_base + (4 * i + 0), mask=cmask, other=0.0).to(tl.float32)
                k1 = tl.load(kb_base + (4 * i + 1), mask=cmask, other=0.0).to(tl.float32)
                k2 = tl.load(kb_base + (4 * i + 2), mask=cmask, other=0.0).to(tl.float32)
                k3 = tl.load(kb_base + (4 * i + 3), mask=cmask, other=0.0).to(tl.float32)

                acc0 = tl.math.fma(q0, k0, acc0)
                acc1 = tl.math.fma(q1, k1, acc1)
                acc2 = tl.math.fma(q2, k2, acc2)
                acc3 = tl.math.fma(q3, k3, acc3)

            dot = (acc0 + acc2) + (acc1 + acc3)
            content = dot * sc

        q_pos = tl.load(POS)
        pos_f = q_pos.to(tl.float32)
        cen = tl.load(CEN + b * scb + h * sch + c, mask=cmask, other=0.0).to(tl.float32)
        ceil_v = tl.load(CEIL + b * scb + h * sch + c, mask=cmask, other=0.0).to(tl.float32)
        ok = tl.load(OK + b * scb + h * sch + c, mask=cmask, other=False).to(tl.int1)
        cand = tl.load(CAND + b * scb + h * sch + c, mask=cmask, other=0).to(tl.int32)

        lam = tl.load(LAM + h).to(tl.float32)
        sig = tl.load(SIG + h).to(tl.float32)

        dist = tl.abs(pos_f - cen)
        erf_v = _libdevice.erf(dist / (2.0 * sig))
        score = content + (lam * ceil_v) * erf_v
        score = tl.where(ok & cmask, score, -1e9)
        score = tl.where(cmask, score, -float("inf"))

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        q = tl.load(q_base + offs_d, mask=d_mask, other=0.0)

        block_start = (q_pos // QUERY_BLOCK) * QUERY_BLOCK
        n_cells = (N_KV + CELL_SIZE - 1) // CELL_SIZE
        offs_s = tl.arange(0, BLOCK_S)

        m_i = tl.full((1,), -float("inf"), tl.float32)
        l_i = tl.zeros((1,), tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)
        seen_cells = tl.full((BLOCK_C,), -2, dtype=tl.int32)

        for slot in tl.static_range(0, K_SELECT):
            max_s = tl.max(score, axis=0)
            is_max = (score == max_s) & cmask
            best_c = tl.min(tl.where(is_max, c, BLOCK_C), axis=0)
            picked_ok = tl.sum(tl.where(c == best_c, ok.to(tl.int32), 0), axis=0) != 0
            picked_cell = tl.sum(tl.where(c == best_c, cand, 0), axis=0)
            cell = tl.where(picked_ok, picked_cell, -1)
            score = tl.where(c == best_c, -float("inf"), score)

            cell_ok = (
                (cell >= 0)
                & (cell < n_cells)
                & ((cell + 1) * CELL_SIZE <= block_start)
            )
            dup_cell = tl.sum(tl.where((c < slot) & (seen_cells == cell), 1, 0), axis=0) != 0
            cell_ok = cell_ok & (~dup_cell)
            seen_cells = tl.where(c == slot, tl.where(cell_ok, cell, -2), seen_cells)

            safe_cell = tl.maximum(cell, 0)
            tok = safe_cell * CELL_SIZE + offs_s
            tok_ok = (
                cell_ok
                & (offs_s < CELL_SIZE)
                & (tok < N_KV)
                & (tok < q_pos)
            )

            k_ptrs = K + b * skb + h * skh + tok[:, None] * skn + offs_d[None, :]
            kval = tl.load(k_ptrs, mask=tok_ok[:, None] & d_mask[None, :], other=0.0)
            scores = tl.sum(kval.to(tl.float32) * q[None, :], axis=1) * SCALE
            scores = tl.where(tok_ok, scores, -float("inf"))

            has_token = tl.sum(tok_ok.to(tl.int32), axis=0) > 0
            tile_m = tl.max(scores, axis=0)
            m_new = tl.where(has_token, tl.maximum(m_i, tile_m), m_i)
            alpha = tl.where(has_token, tl.exp(m_i - m_new), 1.0)
            p = tl.where(tok_ok, tl.exp(scores - m_new), 0.0)

            v_ptrs = V + b * svb + h * svh + tok[:, None] * svn + offs_d[None, :]
            vval = tl.load(v_ptrs, mask=tok_ok[:, None] & d_mask[None, :], other=0.0)
            acc_new = acc * alpha + tl.sum(p[:, None].to(vval.dtype) * vval, axis=0).to(tl.float32)
            l_new = l_i * alpha + tl.sum(p, axis=0)

            acc = tl.where(has_token, acc_new, acc)
            l_i = tl.where(has_token, l_new, l_i)
            m_i = m_new

        row_ok = l_i > 0.0
        out = tl.where(row_ok, acc / l_i, 0.0).to(Q.dtype.element_ty)

        xf = out.to(tl.float32)
        wq = tl.load(W_NORM + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        sr = tl.load(SR + b * srb + h * srh + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        lrs = tl.load(LRS + h).to(tl.float32)
        if NORM_FP32:
            sq = xf * xf
            ms = tl.sum(sq, axis=0) * (1.0 / D)
            rms = tl.math.rsqrt(ms + eps)
            out_long = (xf * rms) * wq
            y = (sr + lrs * out_long).to(O.dtype.element_ty)
        else:
            sq = (xf * xf).to(Q.dtype.element_ty).to(tl.float32)
            ms = (tl.sum(sq, axis=0) * (1.0 / D)).to(Q.dtype.element_ty).to(tl.float32)
            rms = tl.math.rsqrt((ms + eps).to(Q.dtype.element_ty).to(tl.float32)).to(Q.dtype.element_ty).to(tl.float32)
            if W_BF16:
                out_long = ((xf * rms).to(Q.dtype.element_ty).to(tl.float32) * wq).to(Q.dtype.element_ty).to(tl.float32)
            else:
                out_long = (xf * rms).to(Q.dtype.element_ty).to(tl.float32) * wq
            if W_BF16 and LRS_BF16:
                scaled_lr = (lrs * out_long).to(Q.dtype.element_ty).to(tl.float32)
                y = (sr + scaled_lr).to(O.dtype.element_ty)
            else:
                scaled_lr = lrs * out_long
                y = (sr + scaled_lr).to(O.dtype.element_ty)

        tl.store(O + b * (H * D) + h * D + offs_d, y, mask=d_mask)

    @triton.jit
    def _fs_dense_decode_topk_split_kernel(
        Q, KBAR, CEIL, POS, M_VIS_PTR, LAM, SIG,
        PART_S, PART_C,
        sqb, sqh,
        skbb, skbh, skbc,
        scb, sch,
        sc, max_cells,
        H: tl.constexpr, D: tl.constexpr, CELL_SIZE: tl.constexpr,
        K_SELECT: tl.constexpr, SPLIT_M: tl.constexpr, BLOCK_C: tl.constexpr,
        AUTOCAST_BF16: tl.constexpr,
    ):
        bh = tl.program_id(0)
        sm = tl.program_id(1)
        b = bh // H
        h = bh - b * H
        m_vis = tl.load(M_VIS_PTR).to(tl.int32)
        c = sm * BLOCK_C + tl.arange(0, BLOCK_C)
        cmask = (c < m_vis) & (c < max_cells)

        q_base = Q + b * sqb + h * sqh
        kb_base = KBAR + b * skbb + h * skbh + c * skbc

        if AUTOCAST_BF16:
            dot = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for tx in tl.static_range(0, 16):
                q0 = tl.load(q_base + (0 + tx)).to(tl.bfloat16).to(tl.float32)
                q1 = tl.load(q_base + (16 + tx)).to(tl.bfloat16).to(tl.float32)
                q2 = tl.load(q_base + (32 + tx)).to(tl.bfloat16).to(tl.float32)
                q3 = tl.load(q_base + (48 + tx)).to(tl.bfloat16).to(tl.float32)

                k0 = tl.load(kb_base + (0 + tx), mask=cmask, other=0.0).to(tl.bfloat16).to(tl.float32)
                k1 = tl.load(kb_base + (16 + tx), mask=cmask, other=0.0).to(tl.bfloat16).to(tl.float32)
                k2 = tl.load(kb_base + (32 + tx), mask=cmask, other=0.0).to(tl.bfloat16).to(tl.float32)
                k3 = tl.load(kb_base + (48 + tx), mask=cmask, other=0.0).to(tl.bfloat16).to(tl.float32)

                t_tx = (((q0 * k0) + (q1 * k1)) + (q2 * k2)) + (q3 * k3)
                dot = dot + t_tx
            content = (dot.to(tl.bfloat16) * sc.to(tl.bfloat16)).to(tl.float32)
        else:
            acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
            acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
            acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
            acc3 = tl.zeros((BLOCK_C,), dtype=tl.float32)

            for i in tl.static_range(0, 16):
                q0 = tl.load(q_base + (4 * i + 0)).to(tl.float32)
                q1 = tl.load(q_base + (4 * i + 1)).to(tl.float32)
                q2 = tl.load(q_base + (4 * i + 2)).to(tl.float32)
                q3 = tl.load(q_base + (4 * i + 3)).to(tl.float32)

                k0 = tl.load(kb_base + (4 * i + 0), mask=cmask, other=0.0).to(tl.float32)
                k1 = tl.load(kb_base + (4 * i + 1), mask=cmask, other=0.0).to(tl.float32)
                k2 = tl.load(kb_base + (4 * i + 2), mask=cmask, other=0.0).to(tl.float32)
                k3 = tl.load(kb_base + (4 * i + 3), mask=cmask, other=0.0).to(tl.float32)

                acc0 = tl.math.fma(q0, k0, acc0)
                acc1 = tl.math.fma(q1, k1, acc1)
                acc2 = tl.math.fma(q2, k2, acc2)
                acc3 = tl.math.fma(q3, k3, acc3)

            dot = (acc0 + acc2) + (acc1 + acc3)
            content = dot * sc

        q_pos = tl.load(POS)
        pos_f = q_pos.to(tl.float32)
        cen = (c.to(tl.float32) + 0.5) * CELL_SIZE
        ceil_v = tl.load(CEIL + b * scb + h * sch + c, mask=cmask, other=0.0).to(tl.float32)

        lam = tl.load(LAM + h).to(tl.float32)
        sig = tl.load(SIG + h).to(tl.float32)

        dist = tl.abs(pos_f - cen)
        erf_v = _libdevice.erf(dist / (2.0 * sig))
        score = content + (lam * ceil_v) * erf_v
        score = tl.where(cmask, score, -float("inf"))

        out_base = bh * (SPLIT_M * K_SELECT) + sm * K_SELECT
        sentinel = max_cells + BLOCK_C
        for slot in tl.static_range(0, K_SELECT):
            max_s = tl.max(score, axis=0)
            has_valid = max_s > -float("inf")
            is_max = (score == max_s) & cmask & has_valid
            best_c = tl.min(tl.where(is_max, c, sentinel), axis=0)
            valid = best_c < m_vis
            out_s = tl.where(valid, max_s, -float("inf"))
            out_c = tl.where(valid, best_c, -1)
            tl.store(PART_S + out_base + slot, out_s)
            tl.store(PART_C + out_base + slot, out_c)
            score = tl.where(c == best_c, -float("inf"), score)

    @triton.jit
    def _fs_dense_decode_readout_combine_kernel(
        Q, PART_S, PART_C, POS, M_VIS_PTR,
        K, V, SR, W_NORM, LRS, O,
        sqb, sqh,
        skb, skh, skn,
        svb, svh, svn,
        srb, srh,
        N_KV, eps,
        H: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
        CELL_SIZE: tl.constexpr, QUERY_BLOCK: tl.constexpr,
        K_SELECT: tl.constexpr, BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
        NORM_FP32: tl.constexpr, W_BF16: tl.constexpr, LRS_BF16: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        h = bh - b * H
        idx = tl.arange(0, BLOCK_P)
        score = tl.load(PART_S + bh * BLOCK_P + idx)
        cand = tl.load(PART_C + bh * BLOCK_P + idx)
        cmask = cand >= 0
        score = tl.where(cmask, score, -float("inf"))

        q_pos = tl.load(POS)
        m_vis = tl.load(M_VIS_PTR).to(tl.int32)
        q_base = Q + b * sqb + h * sqh
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        q = tl.load(q_base + offs_d, mask=d_mask, other=0.0)

        block_start = (q_pos // QUERY_BLOCK) * QUERY_BLOCK
        n_cells = (N_KV + CELL_SIZE - 1) // CELL_SIZE
        offs_s = tl.arange(0, BLOCK_S)

        m_i = tl.full((1,), -float("inf"), tl.float32)
        l_i = tl.zeros((1,), tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)

        for slot in tl.static_range(0, K_SELECT):
            max_s = tl.max(score, axis=0)
            has_valid = (max_s > -float("inf")) & (slot < m_vis)
            is_max = (score == max_s) & cmask & has_valid
            best_idx = tl.min(tl.where(is_max, idx, BLOCK_P), axis=0)
            picked_ok = best_idx < BLOCK_P
            picked_cell = tl.sum(tl.where(idx == best_idx, cand, 0), axis=0)
            cell = tl.where(picked_ok, picked_cell, -1)
            score = tl.where(idx == best_idx, -float("inf"), score)
            cmask = cmask & (idx != best_idx)

            cell_ok = (
                (cell >= 0)
                & (cell < m_vis)
                & (cell < n_cells)
                & ((cell + 1) * CELL_SIZE <= block_start)
            )
            safe_cell = tl.maximum(cell, 0)
            tok = safe_cell * CELL_SIZE + offs_s
            tok_ok = (
                cell_ok
                & (offs_s < CELL_SIZE)
                & (tok < N_KV)
                & (tok < q_pos)
            )

            k_ptrs = K + b * skb + h * skh + tok[:, None] * skn + offs_d[None, :]
            kval = tl.load(k_ptrs, mask=tok_ok[:, None] & d_mask[None, :], other=0.0)
            scores = tl.sum(kval.to(tl.float32) * q[None, :], axis=1) * SCALE
            scores = tl.where(tok_ok, scores, -float("inf"))

            has_token = tl.sum(tok_ok.to(tl.int32), axis=0) > 0
            tile_m = tl.max(scores, axis=0)
            m_new = tl.where(has_token, tl.maximum(m_i, tile_m), m_i)
            alpha = tl.where(has_token, tl.exp(m_i - m_new), 1.0)
            p = tl.where(tok_ok, tl.exp(scores - m_new), 0.0)

            v_ptrs = V + b * svb + h * svh + tok[:, None] * svn + offs_d[None, :]
            vval = tl.load(v_ptrs, mask=tok_ok[:, None] & d_mask[None, :], other=0.0)
            acc_new = acc * alpha + tl.sum(p[:, None].to(vval.dtype) * vval, axis=0).to(tl.float32)
            l_new = l_i * alpha + tl.sum(p, axis=0)

            acc = tl.where(has_token, acc_new, acc)
            l_i = tl.where(has_token, l_new, l_i)
            m_i = m_new

        row_ok = l_i > 0.0
        out = tl.where(row_ok, acc / l_i, 0.0).to(Q.dtype.element_ty)

        xf = out.to(tl.float32)
        wq = tl.load(W_NORM + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        sr = tl.load(SR + b * srb + h * srh + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        lrs = tl.load(LRS + h).to(tl.float32)
        if NORM_FP32:
            sq = xf * xf
            ms = tl.sum(sq, axis=0) * (1.0 / D)
            rms = tl.math.rsqrt(ms + eps)
            out_long = (xf * rms) * wq
            y = (sr + lrs * out_long).to(O.dtype.element_ty)
        else:
            sq = (xf * xf).to(Q.dtype.element_ty).to(tl.float32)
            ms = (tl.sum(sq, axis=0) * (1.0 / D)).to(Q.dtype.element_ty).to(tl.float32)
            rms = tl.math.rsqrt((ms + eps).to(Q.dtype.element_ty).to(tl.float32)).to(Q.dtype.element_ty).to(tl.float32)
            if W_BF16:
                out_long = ((xf * rms).to(Q.dtype.element_ty).to(tl.float32) * wq).to(Q.dtype.element_ty).to(tl.float32)
            else:
                out_long = (xf * rms).to(Q.dtype.element_ty).to(tl.float32) * wq
            if W_BF16 and LRS_BF16:
                scaled_lr = (lrs * out_long).to(Q.dtype.element_ty).to(tl.float32)
                y = (sr + scaled_lr).to(O.dtype.element_ty)
            else:
                scaled_lr = lrs * out_long
                y = (sr + scaled_lr).to(O.dtype.element_ty)

        tl.store(O + b * (H * D) + h * D + offs_d, y, mask=d_mask)


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def rope_ok(x: torch.Tensor, cos: torch.Tensor) -> bool:
    if compiling() or not available() or os.environ.get("FSL_ROPE_FUSED", "1") == "0":
        return False
    if not (x.is_cuda and x.dim() == 4 and cos.dim() == 2 and x.shape[2] == cos.shape[0] and x.shape[2] > 0):
        return False
    D = x.shape[-1]
    half = D // 2
    ct = torch.promote_types(x.dtype, cos.dtype)
    return (D % 2 == 0 and half >= 16 and (half & (half - 1)) == 0 and cos.shape[1] >= half and ct in _RND
            and x.stride(-1) == 1 and cos.stride(-1) == 1 and torch.is_tensor(cos) and not torch.is_grad_enabled())


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out_dtype=None) -> torch.Tensor:
    """_apply_rotary_emb(x, cos, sin) (x (B, H, N, D), any strides with a unit last stride; cos / sin (N, >= D/2)),
    bit for bit, written as a contiguous (B, H, N, D) tensor in out_dtype (default: torch's result dtype)."""
    B, H, N, D = x.shape
    ct = torch.promote_types(x.dtype, cos.dtype)
    out = torch.empty(B, H, N, D, device=x.device, dtype=out_dtype or ct)
    BN = 64
    grid = (triton.cdiv(N, BN), B * H)
    _rope_kernel[grid](x, cos, sin, out, N, H, x.stride(0), x.stride(1), x.stride(2), cos.stride(0), sin.stride(0),
                       out.stride(0), out.stride(1), out.stride(2), HALF=D // 2, BN=BN, RND=_RND[ct],
                       num_warps=4, enable_fp_fusion=False)
    return out


def autocast_dtype():
    if not torch.is_autocast_enabled():
        return None
    return torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else torch.get_autocast_gpu_dtype()


# Fewest rows (values of all but the last dim) that take the fused norm and silu * mul in eager mode. Single-token
# eager decode (1 row per sequence) keeps the eager ops: there the fusions' Triton launches cost more host time than
# the eager ops. Inside a CUDA-graph capture (torch.cuda.is_current_stream_capturing()), host launch overhead at replay
# is zero, so single-row norms and silu * mul take the one-pass kernels as well.
MIN_ROWS = 16
_GRAPH_WARMUP = False


def _rows(x: torch.Tensor) -> int:
    return x.numel() // x.shape[-1] if x.dim() > 0 and x.shape[-1] > 0 else 0


def _rows_ok(x: torch.Tensor) -> bool:
    r = _rows(x)
    return r >= MIN_ROWS or (r >= 1 and x.is_cuda and (_GRAPH_WARMUP or torch.cuda.is_current_stream_capturing()))


def rms_norm_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if (not _rows_ok(x) or compiling() or not available() or os.environ.get("FSL_TRUNK_FUSED", "1") == "0"
            or torch.is_grad_enabled()):
        return False
    dt = autocast_dtype() if x.is_cuda else None
    return (dt in (torch.bfloat16, torch.float16) and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and weight.dtype in (torch.float32, torch.bfloat16, torch.float16) and x.is_contiguous()
            and weight.is_contiguous() and x.shape[-1] == weight.shape[0] and x.shape[-1] <= 8192
            and x.numel() > 0)


# row widths of the one-pass norm (see the module docstring for the compile-time cap)
_ONEPASS_WIDTHS = (256, 512, 1024, 2048)
_ONEPASS_1ROW_WIDTHS = (128, 256, 512, 1024, 2048)


def rms_norm_cast(x: torch.Tensor, weight: torch.Tensor, eps: float, dtype: torch.dtype) -> torch.Tensor:
    """RMSNorm(x) * weight as model.RMSNorm computes it under autocast (fp32), cast to dtype, bit for bit."""
    D = x.shape[-1]
    n = x.numel()
    rows = n // D
    if os.environ.get("FSL_NORM_ONEPASS", "1") != "0":
        if D in _ONEPASS_WIDTHS and rows >= 16:
            out = torch.empty(x.shape, device=x.device, dtype=dtype)
            _rms_fused_kernel[(rows,)](x, weight, out, float(eps), J=D // 128, num_warps=1, enable_fp_fusion=False)
            return out
        if D in _ONEPASS_1ROW_WIDTHS and rows == 1:
            out = torch.empty(x.shape, device=x.device, dtype=dtype)
            V = 1 if D == 128 else 4
            _rms_1row_kernel[(1,)](x, weight, out, float(eps), D=D, T=D // V, V=V, num_warps=1, enable_fp_fusion=False)
            return out
    sq = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    BLK = 4096
    _square_kernel[(triton.cdiv(n, BLK),)](x, sq, n, BLOCK=BLK, num_warps=8, enable_fp_fusion=False)
    ms = sq.mean(-1, keepdim=True)
    del sq
    out = torch.empty(x.shape, device=x.device, dtype=dtype)
    _rms_out_kernel[(rows,)](x, ms, weight, out, float(eps), D, BLOCK=triton.next_power_of_2(D), num_warps=4,
                             enable_fp_fusion=False)
    return out


def silu_mul_ok(g: torch.Tensor, u: torch.Tensor) -> bool:
    return (_rows_ok(g) and not compiling() and available()
            and os.environ.get("FSL_TRUNK_FUSED", "1") != "0" and not torch.is_grad_enabled()
            and g.is_cuda and g.shape == u.shape and g.dtype == u.dtype and g.dtype in _RND
            and g.is_contiguous() and u.is_contiguous() and g.numel() > 0)


def silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """F.silu(g) * u, bit for bit, in one pass."""
    out = torch.empty_like(g)
    n = g.numel()
    BLK = 4096
    _silu_mul_kernel[(triton.cdiv(n, BLK),)](g, u, out, n, BLOCK=BLK, RND=_RND[g.dtype], num_warps=8,
                                            enable_fp_fusion=False)
    return out


def sr_decode_ok(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, kbuf: torch.Tensor, vbuf: torch.Tensor,
                 win_idx: torch.Tensor, win_pos: torch.Tensor, cos=None) -> bool:
    if (compiling() or not available() or os.environ.get("FSL_TRUNK_FUSED", "1") == "0"
            or os.environ.get("FSL_SR_DECODE_FUSED", "1") == "0"
            or torch.is_grad_enabled() or not Q.is_cuda):
        return False
    B, H, N, D = Q.shape
    Lw = win_idx.numel()
    half = D // 2
    if not (N == 1 and B * H > 0 and 1 <= Lw <= 1024 and D % 2 == 0 and half >= 16 and (half & (half - 1)) == 0):
        return False
    if not (Q.stride(-1) == 1 and K.stride(-1) == 1 and V.stride(-1) == 1
            and kbuf.stride(-1) == 1 and vbuf.stride(-1) == 1 and win_idx.is_contiguous() and win_pos.is_contiguous()):
        return False
    if cos is not None:
        ct = torch.promote_types(Q.dtype, cos.dtype)
        if not (cos.dim() == 2 and cos.shape[1] >= half and cos.stride(-1) == 1 and ct in _RND):
            return False
    return Q.dtype in _RND and kbuf.dtype == Q.dtype and vbuf.dtype == V.dtype


def sr_decode_fused(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, kbuf: torch.Tensor, vbuf: torch.Tensor,
                    pos_t: torch.Tensor, win_idx: torch.Tensor, win_pos: torch.Tensor, sigma: torch.Tensor,
                    scale: float, window_size: int, fs_uniform_sr: bool, use_rope: bool,
                    cos: torch.Tensor | None = None, sin: torch.Tensor | None = None) -> torch.Tensor:
    """Bit-exact fused sliding-window decode step for graph_decode_layer / graph_decode_layer_fsb."""
    B, H, _, D = Q.shape
    Lw = win_idx.shape[0]
    dt_mm = autocast_dtype() or (torch.promote_types(Q.dtype, cos.dtype) if (use_rope and cos is not None) else Q.dtype)
    ct = torch.promote_types(Q.dtype, cos.dtype) if (use_rope and cos is not None) else Q.dtype
    Qrot = torch.empty(B, H, D, device=Q.device, dtype=dt_mm)
    Kw = torch.empty(B, H, Lw, D, device=Q.device, dtype=dt_mm)
    Vw = torch.empty(B, H, Lw, D, device=Q.device, dtype=V.dtype)
    cos_t = cos if (use_rope and cos is not None) else Q
    sin_t = sin if (use_rope and sin is not None) else Q
    scn = cos_t.stride(0) if (use_rope and cos is not None) else 0
    ssn = sin_t.stride(0) if (use_rope and sin is not None) else 0
    BN = 64
    grid = (triton.cdiv(Lw, BN), B * H)
    _sr_kv_rope_kernel[grid](
        Q, K, V, kbuf, vbuf, cos_t, sin_t, pos_t, win_idx,
        Qrot, Kw, Vw,
        Lw, H=H, HALF=D // 2,
        sqb=Q.stride(0), sqh=Q.stride(1), skb=K.stride(0), skh=K.stride(1), svb=V.stride(0), svh=V.stride(1),
        skbb=kbuf.stride(0), skbh=kbuf.stride(1), skbn=kbuf.stride(2),
        svbb=vbuf.stride(0), svbh=vbuf.stride(1), svbn=vbuf.stride(2),
        scn=scn, ssn=ssn,
        sqrb=Qrot.stride(0), sqrh=Qrot.stride(1),
        skwb=Kw.stride(0), skwh=Kw.stride(1), skwl=Kw.stride(2),
        svwb=Vw.stride(0), svwh=Vw.stride(1), svwl=Vw.stride(2),
        BN=BN, RND=_RND[ct], USE_ROPE=bool(use_rope),
        num_warps=4, enable_fp_fusion=False,
    )
    scores = torch.einsum("bhd,bhld->bhl", Qrot, Kw)
    w = torch.empty(B, H, Lw, device=Q.device, dtype=Vw.dtype)
    log2_l = max(5, (Lw - 1).bit_length())
    _sr_softmax_cast_kernel[(B * H,)](
        scores, pos_t, win_pos, sigma, w,
        float(scale), int(window_size), Lw,
        H=H, UNIFORM=bool(fs_uniform_sr), LOG2_L=log2_l,
        num_warps=1, enable_fp_fusion=False,
    )
    return torch.einsum("bhl,bhld->bhd", w, Vw).unsqueeze(2)


def fsl_pool_refresh_ok(Kbar_all: torch.Tensor, ceil_all: torch.Tensor, M_vis: int, Pk: int, Wc: int, D: int) -> bool:
    if (compiling() or not available() or os.environ.get("FSL_TRUNK_FUSED", "1") == "0"
            or os.environ.get("FSL_POOL_REFRESH_FUSED", "1") == "0"
            or torch.is_grad_enabled() or not Kbar_all.is_cuda):
        return False
    C = Pk + Wc
    return (M_vis > 0 and 1 <= Pk <= M_vis and Wc >= 0 and 1 <= C <= 128 and 1 <= D <= 128
            and Kbar_all.dtype == torch.float32 and ceil_all.dtype == torch.float32
            and Kbar_all.stride(-1) == 1 and ceil_all.stride(-1) == 1)


def fsl_pool_refresh(topP: torch.Tensor, Kbar_all: torch.Tensor, ceil_all: torch.Tensor,
                     M_vis: int, blk_start: int, s: int, Pk: int, Wc: int,
                     out: dict | None = None) -> dict:
    """Bit-exact fused FS-L decode candidate pool assembly after topP = ceil_c.topk(Pk).indices."""
    B, H, _, D = Kbar_all.shape
    C = Pk + Wc
    dev = Kbar_all.device
    topP_c = topP.contiguous()
    if (isinstance(out, dict) and not out.get("empty", True) and out.get("C") == C
            and tuple(out["Kbar_cand"].shape) == (B, H, C, D) and out["Kbar_cand"].is_contiguous()
            and out["ceil_cand"].is_contiguous() and out["cen_cand"].is_contiguous()
            and out["ok_cell"].is_contiguous() and out["cand_c"].is_contiguous()):
        Kbar_out = out["Kbar_cand"]
        ceil_out = out["ceil_cand"]
        cen_out = out["cen_cand"]
        ok_out = out["ok_cell"]
        cand_out = out["cand_c"]
        out["M_vis"] = M_vis
        res = out
    else:
        Kbar_out = torch.empty((B, H, C, D), device=dev, dtype=torch.float32)
        ceil_out = torch.empty((B, H, C), device=dev, dtype=torch.float32)
        cen_out = torch.empty((B, H, C), device=dev, dtype=torch.float32)
        ok_out = torch.empty((B, H, C), device=dev, dtype=torch.bool)
        cand_out = torch.empty((B, H, C), device=dev, dtype=torch.int64)
        res = {
            "empty": False, "M_vis": M_vis, "C": C,
            "Kbar_cand": Kbar_out, "ceil_cand": ceil_out, "cen_cand": cen_out,
            "ok_cell": ok_out, "cand_c": cand_out,
        }
    hi = blk_start // s - 1
    block_c = max(16, triton.next_power_of_2(C))
    block_d = max(16, triton.next_power_of_2(D))
    _fsl_pool_refresh_kernel[(B * H,)](
        topP_c, Kbar_all, ceil_all,
        Kbar_out, ceil_out, cen_out, ok_out, cand_out,
        Kbar_all.stride(0), Kbar_all.stride(1), Kbar_all.stride(2),
        ceil_all.stride(0), ceil_all.stride(1),
        Kbar_out.stride(0), Kbar_out.stride(1), Kbar_out.stride(2),
        ceil_out.stride(0), ceil_out.stride(1),
        int(hi), float(blk_start), float(s), int(Pk), int(C),
        H=H, D=D, BLOCK_C=block_c, BLOCK_D=block_d,
        num_warps=4, enable_fp_fusion=False,
    )
    return res


def fsl_lr_decode_combine_ok(attn, q_t: torch.Tensor, kbuf: torch.Tensor, vbuf: torch.Tensor,
                             out_short: torch.Tensor, pool: dict, pos_t: torch.Tensor, sigma: torch.Tensor) -> bool:
    if (compiling() or not available() or os.environ.get("FSL_TRUNK_FUSED", "1") == "0"
            or os.environ.get("FSL_LR_DECODE_FUSED", "1") == "0"
            or torch.is_grad_enabled() or not q_t.is_cuda):
        return False
    if getattr(attn, "sel_content_only", False) or pool.get("empty", False):
        return False
    B, H, N, D = q_t.shape
    C = int(pool.get("C", 0))
    M_vis = int(pool.get("M_vis", 0))
    k_sel = min(int(attn.cell_topk), M_vis, C)
    s = int(attn.cell_stride)
    if pool.get("dense", False):
        m_vis_t = pool.get("m_vis_t")
        Kbar_all = pool.get("Kbar_all")
        ceil_all = pool.get("ceil_all")
        part_s = pool.get("part_s")
        part_c = pool.get("part_c")
        if not (N == 1 and B * H > 0 and D == 64 and C >= 1 and 1 <= k_sel <= 16
                and s in (16, 32, 64) and out_short.shape == q_t.shape
                and m_vis_t is not None and Kbar_all is not None and ceil_all is not None
                and part_s is not None and part_c is not None):
            return False
        if not (q_t.dtype in (torch.bfloat16, torch.float32)
                and kbuf.dtype == q_t.dtype and vbuf.dtype == q_t.dtype and out_short.dtype == q_t.dtype
                and Kbar_all.dtype == torch.float32 and ceil_all.dtype == torch.float32
                and m_vis_t.dtype == torch.int32 and part_s.dtype == torch.float32 and part_c.dtype == torch.int32):
            return False
        return (q_t.stride(-1) == 1 and kbuf.stride(-1) == 1 and vbuf.stride(-1) == 1 and out_short.stride(-1) == 1
                and Kbar_all.stride(-1) == 1 and ceil_all.stride(-1) == 1
                and part_s.is_contiguous() and part_c.is_contiguous()
                and attn.lr_norm.scale.is_contiguous() and attn.long_range_scale.is_contiguous()
                and attn.sel_scale.is_contiguous() and sigma.is_contiguous())
    cand_c = pool.get("cand_c")
    if not (N == 1 and B * H > 0 and D == 64 and 1 <= C <= 128 and 1 <= k_sel <= 16
            and s in (16, 32, 64) and cand_c is not None and out_short.shape == q_t.shape):
        return False
    Kbar_cand = pool["Kbar_cand"]
    ceil_cand = pool["ceil_cand"]
    cen_cand = pool["cen_cand"]
    ok_cell = pool["ok_cell"]
    if not (q_t.dtype in (torch.bfloat16, torch.float32)
            and kbuf.dtype == q_t.dtype and vbuf.dtype == q_t.dtype and out_short.dtype == q_t.dtype
            and Kbar_cand.dtype == torch.float32 and ceil_cand.dtype == torch.float32
            and cen_cand.dtype == torch.float32 and ok_cell.dtype == torch.bool
            and cand_c.dtype in (torch.int64, torch.int32)):
        return False
    return (q_t.stride(-1) == 1 and kbuf.stride(-1) == 1 and vbuf.stride(-1) == 1 and out_short.stride(-1) == 1
            and Kbar_cand.stride(-1) == 1 and ceil_cand.stride(-1) == 1 and cen_cand.stride(-1) == 1
            and ok_cell.stride(-1) == 1 and cand_c.stride(-1) == 1
            and attn.lr_norm.scale.is_contiguous() and attn.long_range_scale.is_contiguous()
            and attn.sel_scale.is_contiguous() and sigma.is_contiguous())


def fsl_lr_decode_combine(attn, q_t: torch.Tensor, kbuf: torch.Tensor, vbuf: torch.Tensor,
                          out_short: torch.Tensor, pool: dict, pos_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Bit-exact fused FS-L / Base FS candidate scoring + top-k cell selection + online softmax readout + lr_norm + SR combine."""
    B, H, _, D = q_t.shape
    C = int(pool["C"])
    M_vis = int(pool["M_vis"])
    s = int(attn.cell_stride)
    w_norm = attn.lr_norm.scale
    lrs = attn.long_range_scale.view(-1)
    w_bf16 = (q_t.dtype == torch.bfloat16) and (w_norm.dtype == torch.bfloat16)
    lrs_bf16 = (q_t.dtype == torch.bfloat16) and (lrs.dtype == torch.bfloat16)
    adt = autocast_dtype()
    autocast_bf16 = bool(adt == torch.bfloat16)
    norm_fp32 = bool(torch.is_autocast_enabled() or q_t.dtype == torch.float32)
    out_dt = adt or (
        torch.float32 if norm_fp32 else torch.promote_types(
            out_short.dtype, torch.promote_types(lrs.dtype, torch.promote_types(q_t.dtype, w_norm.dtype))
        )
    )
    out = torch.empty((B, 1, H * D), device=q_t.device, dtype=out_dt)
    block_s = max(16, triton.next_power_of_2(s))
    if pool.get("dense", False):
        k_sel = min(max(1, int(attn.cell_topk)), 16)
        split_m = 16
        block_c = max(16, triton.next_power_of_2(triton.cdiv(C, split_m)))
        block_p = max(16, triton.next_power_of_2(split_m * k_sel))
        Kbar_all = pool["Kbar_all"]
        ceil_all = pool["ceil_all"]
        m_vis_t = pool["m_vis_t"]
        part_s = pool["part_s"]
        part_c = pool["part_c"]
        _fs_dense_decode_topk_split_kernel[(B * H, split_m)](
            q_t[:, :, 0, :], Kbar_all, ceil_all, pos_t, m_vis_t, attn.sel_scale, sigma,
            part_s, part_c,
            q_t.stride(0), q_t.stride(1),
            Kbar_all.stride(0), Kbar_all.stride(1), Kbar_all.stride(2),
            ceil_all.stride(0), ceil_all.stride(1),
            float(attn.scale), C,
            H=H, D=D, CELL_SIZE=s, K_SELECT=k_sel, SPLIT_M=split_m, BLOCK_C=block_c,
            AUTOCAST_BF16=autocast_bf16,
            num_warps=2 if block_c <= 64 else 4, num_stages=2, enable_fp_fusion=False,
        )
        _fs_dense_decode_readout_combine_kernel[(B * H,)](
            q_t[:, :, 0, :], part_s, part_c, pos_t, m_vis_t,
            kbuf, vbuf, out_short[:, :, 0, :], w_norm, lrs, out,
            q_t.stride(0), q_t.stride(1),
            kbuf.stride(0), kbuf.stride(1), kbuf.stride(2),
            vbuf.stride(0), vbuf.stride(1), vbuf.stride(2),
            out_short.stride(0), out_short.stride(1),
            int(kbuf.shape[2]), float(attn.lr_norm.eps),
            H=H, D=D, SCALE=float(attn.scale), CELL_SIZE=s, QUERY_BLOCK=int(attn._cell_block),
            K_SELECT=k_sel, BLOCK_P=block_p, BLOCK_D=64, BLOCK_S=block_s,
            NORM_FP32=norm_fp32, W_BF16=w_bf16, LRS_BF16=lrs_bf16,
            num_warps=4, num_stages=2, enable_fp_fusion=False,
        )
        return out
    k_sel = min(int(attn.cell_topk), M_vis, C)
    Kbar_cand = pool["Kbar_cand"]
    ceil_cand = pool["ceil_cand"]
    cen_cand = pool["cen_cand"]
    ok_cell = pool["ok_cell"]
    cand_c = pool["cand_c"]
    block_c = max(16, triton.next_power_of_2(C))
    _fsl_lr_decode_combine_kernel[(B * H,)](
        q_t[:, :, 0, :], Kbar_cand, ceil_cand, cen_cand, ok_cell, cand_c, pos_t, attn.sel_scale, sigma,
        kbuf, vbuf, out_short[:, :, 0, :], w_norm, lrs, out,
        q_t.stride(0), q_t.stride(1),
        Kbar_cand.stride(0), Kbar_cand.stride(1), Kbar_cand.stride(2),
        ceil_cand.stride(0), ceil_cand.stride(1),
        kbuf.stride(0), kbuf.stride(1), kbuf.stride(2),
        vbuf.stride(0), vbuf.stride(1), vbuf.stride(2),
        out_short.stride(0), out_short.stride(1),
        int(kbuf.shape[2]), float(attn.scale), float(attn.lr_norm.eps), C,
        H=H, D=D, SCALE=float(attn.scale), CELL_SIZE=s, QUERY_BLOCK=int(attn._cell_block), K_SELECT=k_sel,
        BLOCK_C=block_c, BLOCK_D=64, BLOCK_S=block_s,
        NORM_FP32=norm_fp32, W_BF16=w_bf16, LRS_BF16=lrs_bf16,
        AUTOCAST_BF16=autocast_bf16,
        num_warps=4, num_stages=2, enable_fp_fusion=False,
    )
    return out

