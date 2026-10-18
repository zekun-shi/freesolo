"""Fused FS-B traversal kernels (on by default; FSB_FAST_BEAM=0 / FSB_FAST_RERANK=0 restore the reference).

FSB_FAST_BEAM runs the whole beam descent of _forest_beam (every level and the cells epilogue) in one
kernel, for decode and batched prefill queries: its CUDA builds where NVRTC is available (fsb_beam_cuda: one block
per prefill row, both beams of a stage in one launch via forest_beam_pair, else one warp per query; FSB_BEAM_ROWS=0
restores 9b4e9ff's warp-per-query kernel for every row), else _fsb_beam_kernel (FSB_BEAM_CUDA=0); FSB_FAST_RERANK
fuses the lse / lse_pos / max re-rank's token scores and stable top-K selection of rerank_top for single-query rows.
The union re-rank scores every candidate in a one-program-per-(row, candidate) kernel and selects with a second
small one; decode's readout is one Triton kernel when attn's fs_fused_attn / FS_FUSED_ATTN is on. Every re-rank
output is bit-exact with the reference path it replaces; the decode readout is F8-class (2e-3, torch's operation
order with different reduction trees) — tests F121-F126 and tests/test_fsb_fast_step.py.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice as _ld              # triton >= 3.1
    except Exception:
        from triton.language.extra.cuda import libdevice as _ld        # triton 3.0
    from .fsb_traverse_triton import _halve_rows
    _HAVE_TRITON = True
    _NEG = tl.constexpr(-1e9)
except Exception:
    _HAVE_TRITON = False


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


if _HAVE_TRITON:
    @triton.jit
    def _beam_level(ref, q, qn, qpos, lam, sig, budget, wbeam, t, tidx, CHUNK, SPINE, Ntot, Sp, scale, cell_size,
                    NC: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr, WM: tl.constexpr, BOUND: tl.constexpr):
        # one _level_kernel step on a register frontier. The node scores use _level_kernel's expressions and, like
        # it, compile without multiply-add contraction, so they do not depend on the launch width; the candidate
        # bookkeeping is cheaper but selects the same frontier: carried slots (score 1e30) keep their order ahead
        # of every child, children rank by (score desc, slot asc)
        w = tl.arange(0, WM)
        active = ref != -1
        is_c = ref >= 0
        is_s = ref <= -2
        pn = tl.where(is_c, CHUNK + (tidx * Ntot + tl.where(is_c, ref, 0)) * NC,
                      SPINE + (t * Sp + tl.where(is_s, -ref - 2, 0)) * NC)
        okn = is_c | is_s
        cnt = tl.load(pn + DH + 3, mask=okn, other=0.0)
        chl = tl.load(pn + DH + 4, mask=okn, other=-1.0)
        chr_ = tl.load(pn + DH + 5, mask=okn, other=-1.0)
        expand = active & (cnt > budget.to(tl.float32))
        carried = active & (cnt <= budget.to(tl.float32))
        chv = tl.interleave(tl.where(expand, chl, -1.0), tl.where(expand, chr_, -1.0)).to(tl.int64)
        c = tl.arange(0, 2 * WM)
        c_is_c = chv >= 0
        c_is_s = chv <= -2
        d = tl.arange(0, DHP)
        dmask = d < DH
        pc = tl.where(c_is_c, CHUNK + (tidx * Ntot + tl.where(c_is_c, chv, 0)) * NC,
                      SPINE + (t * Sp + tl.where(c_is_s, -chv - 2, 0)) * NC)
        okc = c_is_c | c_is_s
        mu = tl.load(pc[:, None] + d[None, :], mask=okc[:, None] & dmask[None, :], other=0.0)
        rad = tl.load(pc + DH, mask=okc, other=0.0)
        amax = tl.load(pc + DH + 1, mask=okc, other=0.0)
        pmin = tl.load(pc + DH + 2, mask=okc, other=0.0)
        prod = q[None, :] * mu
        content = _halve_rows(prod, DHP) * scale
        if BOUND:
            content = content + (qn * rad) * scale
        pmin_pos = (pmin + 0.5) * cell_size
        erf = tl.math.erf(tl.abs(qpos - pmin_pos) / (2.0 * sig))
        score = content + (lam * amax) * erf
        score = tl.where(chv == -1, _NEG, score)
        ncar_before = tl.sum(((w[None, :] < w[:, None]) & carried[None, :]).to(tl.int32), axis=1)
        ncar = tl.sum(carried.to(tl.int32), axis=0)
        gt = (score[None, :] > score[:, None]) | ((score[None, :] == score[:, None]) & (c[None, :] < c[:, None]))
        crank = ncar + tl.sum(gt.to(tl.int32), axis=1)
        f_ok = carried & (ncar_before < wbeam)
        c_ok = (score > _NEG / 2) & (crank < wbeam)
        hit_f = w[:, None] == tl.where(f_ok, ncar_before, WM + 1)[None, :]
        hit_c = w[:, None] == tl.where(c_ok, crank, WM + 1)[None, :]
        newref = tl.sum(tl.where(hit_f, ref[None, :], 0), axis=1) + tl.sum(tl.where(hit_c, chv[None, :], 0), axis=1)
        has = (tl.sum(hit_f.to(tl.int32), axis=1) + tl.sum(hit_c.to(tl.int32), axis=1)) > 0
        return tl.where(has, newref, -1), tl.sum((chv != -1).to(tl.float32), axis=0)

    @triton.jit
    def _fsb_beam_kernel(ROOT, CELLS, NVIS, Q, RC, rc_s, QPOS, LAM, SIG, WBEAM, BUDGET, TIDX,
                         CHUNK, SPINE, LEAF, ADM, Ntot, Sp, Ltot, Mm, Bq, scale, cell_size, KMAXP1,
                         NC: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr, WM: tl.constexpr,
                         R: tl.constexpr, BOUND: tl.constexpr, RC_T: tl.constexpr):
        # the whole beam descent for (t', b): the radius-scaled query norm of _forest_beam (_det_norm's halving
        # sum and IEEE sqrt), ROOT init, KMAXP1 _beam_level steps with the frontier in registers and the visit
        # count summed in level order, then the cells epilogue
        pid = tl.program_id(0)
        t = pid // Bq
        tidx = tl.load(TIDX + t)
        w = tl.arange(0, WM)
        ref = tl.where(w == 0, tl.load(ROOT + t), -1)
        d = tl.arange(0, DHP)
        q = tl.load(Q + pid * DH + d, mask=d < DH, other=0.0)          # (DHP,) fp32, zero-padded
        qn = tl.sqrt_rn(tl.sum(_halve_rows(tl.reshape(q * q, (1, DHP)), DHP), axis=0))
        if RC_T:
            qn = qn * tl.load(RC + t)
        else:
            qn = qn * rc_s
        qpos = tl.load(QPOS + pid)
        lam = tl.load(LAM + t)
        sig = tl.load(SIG + t)
        budget = tl.load(BUDGET + pid)
        wbeam = tl.load(WBEAM + pid)
        nvis = 0.0
        for _lev in range(KMAXP1):
            ref, nv = _beam_level(ref, q, qn, qpos, lam, sig, budget, wbeam, t, tidx, CHUNK, SPINE, Ntot, Sp,
                                  scale, cell_size, NC, DH, DHP, WM, BOUND)
            nvis += nv
        # cells epilogue (the _forest_beam torch tail): src/off/len of each frontier node, then the
        # cell ids from the leaf table (src == 1) or the admission table (src == 2)
        is_c = ref >= 0
        base_c = (tidx * Ntot + tl.where(is_c, ref, 0)) * NC
        base_s = (t * Sp + tl.where(ref <= -2, -ref - 2, 0)) * NC
        src = tl.where(is_c, tl.load(CHUNK + base_c + DH + 6, mask=is_c, other=0.0),
                       tl.load(SPINE + base_s + DH + 6, mask=(ref <= -2), other=0.0))
        off = tl.where(is_c, tl.load(CHUNK + base_c + DH + 7, mask=is_c, other=0.0),
                       tl.load(SPINE + base_s + DH + 7, mask=(ref <= -2), other=0.0))
        ln = tl.where(is_c, tl.load(CHUNK + base_c + DH + 8, mask=is_c, other=0.0),
                      tl.load(SPINE + base_s + DH + 8, mask=(ref <= -2), other=0.0))
        tr = tl.arange(0, R)
        pos = off.to(tl.int64)[:, None] + tr[None, :]
        valid = (src[:, None] > 0) & (tr[None, :] < ln[:, None]) & (ref[:, None] != -1)
        from_leaf = src == 1
        gl = tidx * Ltot + tl.minimum(tl.maximum(pos, 0), Ltot - 1)
        ga = tidx * Mm + tl.minimum(tl.maximum(pos, 0), Mm - 1)
        leaf_v = tl.load(LEAF + gl)
        adm_v = tl.load(ADM + ga)
        cells = tl.where(from_leaf[:, None], leaf_v, adm_v)
        cells = tl.where(valid, cells, -1)
        tl.store(CELLS + pid * WM * R + w[:, None] * R + tr[None, :], cells)
        tl.store(NVIS + pid, nvis)

    @triton.jit
    def _fsb_u_kernel(Q, SEL, TIDX, KEY, U, PT, C, Bq, Hh, sb, sh, sl, scale, S,
                      DH: tl.constexpr, DHP: tl.constexpr, SP: tl.constexpr,
                      LSE: tl.constexpr, POS: tl.constexpr):
        # u score of one (row, candidate): token lse / max over the cell's keys (+ the lse_pos position
        # term PT, computed by rerank_top's own torch ops); -inf on empty slots - rerank_top without the sort
        pid = tl.program_id(0)                                           # = (t' * Bq + b) * C + candidate
        r = pid // C
        cell = tl.load(SEL + pid)
        if cell >= 0:
            seq = tl.load(TIDX + r // Bq)
            bi = seq // Hh
            hi = seq - bi * Hh
            d = tl.arange(0, DHP)
            j = tl.arange(0, SP)
            q = tl.load(Q + r * DH + d, mask=d < DH, other=0.0)          # (DHP,) fp32, zero-padded
            tok = cell * S + j
            k = tl.load(KEY + bi * sb + hi * sh + tok[:, None] * sl + d[None, :],
                        mask=(j[:, None] < S) & (d[None, :] < DH), other=0.0).to(tl.float32)
            sc = _halve_rows(k * q[None, :], DHP) * scale                # (SP,) fp32 product, halving dot, * scale
            sc = tl.where(j < S, sc, float("-inf"))
            m = tl.max(sc, axis=0)
            if LSE:
                e = tl.where(j < S, _ld.exp(sc - m), 0.0)
                tot = tl.sum(_halve_rows(tl.reshape(e, (1, SP)), SP), axis=0)
                res = m + _ld.log(tot)
            else:
                res = m
            if POS:
                res = res + tl.load(PT + pid)
            tl.store(U + pid, res)
        else:
            tl.store(U + pid, float("-inf"))

    @triton.jit
    def _fsb_sel_kernel(U, SEL, OUT, C, K, CWP: tl.constexpr, KP2: tl.constexpr):
        # stable descending top-K of u per row with ties by candidate slot (torch.sort stable), then
        # rerank_top's -inf mask
        pid = tl.program_id(0)
        i = tl.arange(0, CWP)
        u = tl.load(U + pid * C + i, mask=i < C, other=float("-inf"))
        sel = tl.load(SEL + pid * C + i, mask=i < C, other=-1)
        gt = (u[None, :] > u[:, None]) | ((u[None, :] == u[:, None]) & (i[None, :] < i[:, None]))
        gt = gt & (i[None, :] < C)
        rank = tl.sum(gt.to(tl.int32), axis=1)
        k2 = tl.arange(0, KP2)
        eqr = (k2[:, None] == rank[None, :]) & (k2[:, None] < K)
        outs = tl.sum(tl.where(eqr, sel[None, :], 0), axis=1)
        has = tl.sum(eqr.to(tl.int32), axis=1) > 0
        outu = tl.sum(tl.where(eqr, u[None, :], 0.0), axis=1)
        res = tl.where(has & (outu > float("-inf")), outs, -1)
        tl.store(OUT + pid * K + k2, res, mask=k2 < K)

    import struct as _struct

    def _hf(u: int):
        return tl.constexpr(_struct.unpack("!f", _struct.pack("!I", u))[0])

    _ERF_C0A, _ERF_C0B = _hf(0x38EB4C3A), _hf(0x38B1E96A)
    _ERF_C1A, _ERF_C1B = _hf(0xBAAE005B), _hf(0xBA574D20)
    _ERF_C2A, _ERF_C2B = _hf(0x3C09919F), _hf(0x3BAAD5EA)
    _ERF_C3A, _ERF_C3B = _hf(0xBD24D99A), _hf(0xBCDC1BE7)
    _ERF_C4A, _ERF_C4B = _hf(0x3E235519), _hf(0x3DE718AF)
    _ERF_C5A, _ERF_C5B = _hf(0x3F69B4F9), _hf(0xBEC093AC)
    _ERF_C6A, _ERF_C6B = _hf(0x3F210A14), _hf(0x3E0375D3)
    _ERF_THR = _hf(0x3F8060FE)

    @triton.jit
    def _erf_torch_tl(x):
        # CUDA's built-in erff(x) (as called by torch.erf), instruction for instruction
        a = tl.inline_asm_elementwise("abs.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)
        big = a >= _ERF_THR
        u = tl.where(big, a, x * x)
        r = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                      [tl.where(big, _ERF_C0A, _ERF_C0B), u, tl.where(big, _ERF_C1A, _ERF_C1B)],
                                      dtype=tl.float32, is_pure=True, pack=1)
        r = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                      [r, u, tl.where(big, _ERF_C2A, _ERF_C2B)], dtype=tl.float32, is_pure=True, pack=1)
        r = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                      [r, u, tl.where(big, _ERF_C3A, _ERF_C3B)], dtype=tl.float32, is_pure=True, pack=1)
        r = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                      [r, u, tl.where(big, _ERF_C4A, _ERF_C4B)], dtype=tl.float32, is_pure=True, pack=1)
        r = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                      [r, u, tl.where(big, _ERF_C5A, _ERF_C5B)], dtype=tl.float32, is_pure=True, pack=1)
        r = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                      [r, u, tl.where(big, _ERF_C6A, _ERF_C6B)], dtype=tl.float32, is_pure=True, pack=1)
        s = tl.where(big, -a, x)
        r = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                      [r, s, s], dtype=tl.float32, is_pure=True, pack=1)
        ex = tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [r], dtype=tl.float32, is_pure=True, pack=1)
        rb = 1.0 - ex
        rb = ((x.to(tl.int32, bitcast=True) & (-0x80000000)) | rb.to(tl.int32, bitcast=True)).to(tl.float32, bitcast=True)
        return tl.where(big, rb, r)

    @triton.jit
    def _fsb_upar_kernel(Q, SELA, SELB, TIDX, KEY, QPOS, sq0, sq1, SIG, ss0, ss1, CELLA, ca0, ca1, LAM, ls0, ls1, U,
                         Ca, Cb, C, Bq, Hh, sb, sh, sl, scale, S, Mc,
                         CAP: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr, SP: tl.constexpr,
                         HAS_B: tl.constexpr, LSE: tl.constexpr, POS: tl.constexpr):
        # one u score per program (grid row * C + candidate), fusing union_candidates deduplication and torch.erf
        pid = tl.program_id(0)
        r = pid // C
        cand = pid - r * C
        if HAS_B:
            if cand < Ca:
                cell = tl.load(SELA + r * Ca + cand)
            else:
                cell = tl.load(SELB + r * Cb + (cand - Ca))
                if cell >= 0:
                    ia = tl.arange(0, CAP)
                    a_row = tl.load(SELA + r * Ca + ia, mask=ia < Ca, other=-2)
                    if tl.sum((a_row == cell).to(tl.int32), axis=0) > 0:
                        cell = cell - (cell + 1)
        else:
            cell = tl.load(SELA + pid)
        if cell >= 0:
            t = r // Bq
            b = r - t * Bq
            seq = tl.load(TIDX + t)
            bi = seq // Hh
            hi = seq - bi * Hh
            d = tl.arange(0, DHP)
            j = tl.arange(0, SP)
            q = tl.load(Q + r * DH + d, mask=d < DH, other=0.0)          # (DHP,) fp32, zero-padded
            tok = cell * S + j
            k = tl.load(KEY + bi * sb + hi * sh + tok[:, None] * sl + d[None, :],
                        mask=(j[:, None] < S) & (d[None, :] < DH), other=0.0).to(tl.float32)
            sc = _halve_rows(k * q[None, :], DHP) * scale                # (SP,) fp32 product, halving dot, * scale
            sc = tl.where(j < S, sc, float("-inf"))
            m = tl.max(sc, axis=0)
            if LSE:
                e = tl.where(j < S, _ld.exp(sc - m), 0.0)
                res = m + _ld.log(tl.sum(_halve_rows(tl.reshape(e, (1, SP)), SP), axis=0))
            else:
                res = m
            if POS:
                qp = tl.load(QPOS + t * sq0 + b * sq1)
                den = 2.0 * tl.load(SIG + t * ss0 + b * ss1)
                x = tl.math.div_rn(tl.abs(qp - (cell.to(tl.float32) + 0.5) * S), den)
                lam = tl.load(LAM + t * ls0 + b * ls1)
                a = tl.load(CELLA + seq * ca0 + tl.minimum(cell, Mc - 1) * ca1)
                res = res + (lam * a) * _erf_torch_tl(x)
            tl.store(U + pid, res)
        else:
            tl.store(U + pid, float("-inf"))

    @triton.jit
    def _fsb_selpv_kernel(U, SELA, SELB, OUT, PROV, C, K, Ca, Cb, PROV_BIT,
                          CWP: tl.constexpr, KP2: tl.constexpr, HAS_B: tl.constexpr, PRV: tl.constexpr):
        # the tail of _fsb_usel_kernel (stable descending top-K over the row's u scores, ties by slot, the -inf
        # mask and the utau provenance), reading u written by _fsb_upar_kernel: the same ranks and outputs
        r = tl.program_id(0)
        i = tl.arange(0, CWP)
        u = tl.load(U + r * C + i, mask=i < C, other=float("-inf"))
        if HAS_B:
            isa = i < Ca
            sa = tl.load(SELA + r * Ca + i, mask=isa, other=-1)
            sb = tl.load(SELB + r * Cb + (i - Ca), mask=(i >= Ca) & (i < C), other=-1)
            sel = tl.where(isa, sa, sb)
        else:
            sel = tl.load(SELA + r * Ca + i, mask=i < C, other=-1)
        gt = (u[None, :] > u[:, None]) | ((u[None, :] == u[:, None]) & (i[None, :] < i[:, None]))
        gt = gt & (i[None, :] < C)
        rank = tl.sum(gt.to(tl.int32), axis=1)
        k2 = tl.arange(0, KP2)
        eqr = (k2[:, None] == rank[None, :]) & (k2[:, None] < K)
        outs = tl.sum(tl.where(eqr, sel[None, :], 0), axis=1)
        has = tl.sum(eqr.to(tl.int32), axis=1) > 0
        outu = tl.sum(tl.where(eqr, u[None, :], 0.0), axis=1)
        res = tl.where(has & (outu > float("-inf")), outs, -1)
        tl.store(OUT + r * K + k2, res, mask=k2 < K)
        if PRV:
            src = tl.sum(tl.where(eqr, i[None, :], 0), axis=1)
            pv = tl.where(res < 0, -1, tl.where(src >= Ca, PROV_BIT, 0))
            tl.store(PROV + r * K + k2, pv.to(tl.int8), mask=k2 < K)

    @triton.jit
    def _fsb_decread_kernel(Q, SEL, KEY, VAL, POS, W, SR, LRS, OUT,
                            sqb, sqh, skb, skh, skn, svb, svh, svn, srb, srh, sob, soh,
                            scale, eps, Hh, K, S,
                            DH: tl.constexpr, DHP: tl.constexpr, SP: tl.constexpr, KK: tl.constexpr,
                            HAS_SR: tl.constexpr, LRS_BF16: tl.constexpr):
        # _fsb_decode_readout for one decode query per (batch, head) row: the K selected cells' token keys and values
        # gathered, q.k scores, masked bf16/fp32 softmax over the K * S tokens, the weighted value sum, the empty-row
        # zero and attn.lr_norm. torch's operation order and store roundings with different reduction trees (the two
        # dots halve-sum like the re-rank kernels instead of cublas's accumulation order): F8-class (2e-3).
        r = tl.program_id(0)                                           # = b * H + h (sel.view(B, H, K) rows)
        bi = r // Hh
        hi = r - bi * Hh
        d = tl.arange(0, DHP)
        kk = tl.arange(0, KK)
        j = tl.arange(0, SP)
        q = tl.load(Q + bi * sqb + hi * sqh + d, mask=d < DH, other=0.0) # fp32 view of the stored query
        sel = tl.load(SEL + r * K + kk, mask=kk < K, other=-1)           # (KK,)
        cellc = tl.where(sel >= 0, sel, 0).to(tl.int64)
        tok = cellc[:, None] * S + j[None, :].to(tl.int64)               # (KK, SP) clamped like the reference
        posv = tl.load(POS)
        ok = (sel[:, None] >= 0) & (j[None, :] < S) & (tok < posv)
        kt = tl.load(KEY + bi * skb + hi * skh + tok[:, :, None] * skn + d[None, None, :],
                     mask=ok[:, :, None] & (d[None, None, :] < DH), other=0.0)
        sc = tl.reshape(_halve_rows(tl.reshape(kt.to(tl.float32) * q.to(tl.float32)[None, None, :],
                                               (KK * SP, DHP)), DHP), (KK, SP))
        sc = sc.to(Q.dtype.element_ty).to(tl.float32) * scale      # the score einsum's store, then * scale
        sc = tl.where(ok, sc, -1e9)
        m = tl.max(tl.max(sc, axis=1), axis=0)
        e = _ld.exp(sc - m)
        den = tl.sum(tl.sum(e, axis=1), axis=0)
        w = (e / den).to(VAL.dtype.element_ty).to(tl.float32)            # softmax stored, cast to the value dtype
        vt = tl.load(VAL + bi * svb + hi * svh + tok[:, :, None] * svn + d[None, None, :],
                     mask=ok[:, :, None] & (d[None, None, :] < DH), other=0.0)
        acc = tl.sum(tl.sum(w[:, :, None] * vt.to(tl.float32), axis=0), axis=0)  # (DHP,)
        x = tl.where(tl.sum(ok.to(tl.int32)) > 0, acc, 0.0).to(Q.dtype.element_ty)   # the readout store; empty row 0
        xf = x.to(tl.float32)
        sq = (xf * xf).to(Q.dtype.element_ty).to(tl.float32)             # attn.lr_norm in torch's order and dtypes
        ms = (tl.sum(sq, axis=0) * (1.0 / DH)).to(Q.dtype.element_ty).to(tl.float32)
        rms = tl.math.rsqrt((ms + eps).to(Q.dtype.element_ty).to(tl.float32)).to(Q.dtype.element_ty).to(tl.float32)
        wq = tl.load(W + d, mask=d < DH, other=0.0).to(tl.float32)
        y = ((xf * rms).to(Q.dtype.element_ty).to(tl.float32) * wq).to(Q.dtype.element_ty)
        if HAS_SR:
            sr = tl.load(SR + bi * srb + hi * srh + d, mask=d < DH, other=0.0).to(tl.float32)
            lrs = tl.load(LRS + hi).to(tl.float32)
            yf = y.to(tl.float32)
            if LRS_BF16:
                scaled_lr = (lrs * yf).to(Q.dtype.element_ty).to(tl.float32)
            else:
                scaled_lr = lrs * yf
            y = (sr + scaled_lr).to(OUT.dtype.element_ty)
        tl.store(OUT + bi * sob + hi * soh + d, y, mask=d < DH)


def fast_beam_on() -> bool:
    return _HAVE_TRITON and os.environ.get("FSB_FAST_BEAM", "1") == "1"


def fast_rerank_on() -> bool:
    return _HAVE_TRITON and os.environ.get("FSB_FAST_RERANK", "1") == "1"


def fast_beam_ok(F: dict, SP: dict, q: torch.Tensor, W_max: int, delta_max: int) -> bool:
    if not (fast_beam_on() and q.is_cuda and q.dtype == torch.float32):
        return False
    Dh = q.shape[-1]
    return (Dh <= 256 and W_max >= 1 and (W_max & (W_max - 1)) == 0 and W_max * (1 << delta_max) <= 256
            and SP["root"].dtype == torch.int64 and SP["t_idx"].dtype == torch.int64
            and F["leaf"].dtype == torch.int64 and F["adm"].dtype == torch.int64
            and F["leaf"].is_contiguous() and F["adm"].is_contiguous()
            and F["node"].is_contiguous() and SP["node"].is_contiguous())


@torch.no_grad()
def forest_beam_fused(F: dict, SP: dict, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor,
                      sigv: torch.Tensor, scale: float, s: int, W: torch.Tensor, delta: torch.Tensor,
                      W_max: int, delta_max: int, node_score: str, radius_coef, bud=None):
    # bit-exact fused _forest_beam (Triton path): -> cells (T',Bq,W_max*2^delta_max), n_visit (T',Bq); bud (2^delta)
    # may be passed when the caller already has it. Every kernel on the path compiles without multiply-add
    # contraction, so the launch width is free: one warp per descent for prefill grids, eight for decode.
    Tp, Bq, Dh = q.shape
    R = 1 << delta_max
    chunk, spine = F["node"], SP["node"]
    rc_t = torch.is_tensor(radius_coef)
    rc = radius_coef.reshape(-1).contiguous() if rc_t else q
    rc_s = 1.0 if rc_t else float(radius_coef)
    if bud is None:
        bud = torch.bitwise_left_shift(torch.ones_like(delta), delta).contiguous()
    # the CUDA builds of this kernel (same cells and visits, fsb_beam_cuda) when NVRTC is present: the row-cooperative
    # kernel for prefill rows, else one warp per query
    from .fsb_beam_cuda import beam_cuda_ok, beam_rows_ok, forest_beam_cuda, forest_beam_rows_cuda
    if beam_cuda_ok(F, q, W_max, delta_max):
        if beam_rows_ok(F, q, W_max):
            out = forest_beam_rows_cuda([(F, SP, W, bud, W_max)], q, qpos, lam, sigv, scale, s, delta_max, node_score,
                                        radius_coef)
            if out is not None:
                return out[0]
        out = forest_beam_cuda(F, SP, q, qpos, lam, sigv, scale, s, W, bud, W_max, delta_max, node_score, radius_coef)
        if out is not None:
            return out
    cells = torch.empty((Tp, Bq, W_max * R), dtype=torch.long, device=q.device)
    nvis = torch.empty((Tp, Bq), dtype=torch.float32, device=q.device)
    _fsb_beam_kernel[(Tp * Bq,)](SP["root"].contiguous(), cells, nvis, q.contiguous(), rc, rc_s, qpos.contiguous(),
                                 lam.reshape(-1).contiguous(), sigv.reshape(-1).contiguous(),
                                 W.contiguous(), bud, SP["t_idx"].contiguous(), chunk, spine, F["leaf"], F["adm"],
                                 chunk.shape[1], spine.shape[1], F["Ltot"], F["M"], Bq, float(scale), float(s),
                                 F["Kmax"] + 1, NC=chunk.shape[-1], DH=Dh, DHP=_pow2(Dh), WM=W_max, R=R,
                                 BOUND=(node_score == "bound"), RC_T=rc_t,
                                 num_warps=8 if Tp * Bq < 2048 else 1, enable_fp_fusion=False)
    return cells, nvis


def forest_beam_pair_ok(F: dict, SP: dict, Fu: dict, SPu: dict, q: torch.Tensor, W_max: int, Wu_max: int,
                        delta_max: int) -> bool:
    # forest_beam_pair applies (host checks only: no GPU work, so a caller can keep its op order when it does not)
    if not (q.is_cuda and W_max == Wu_max and os.environ.get("FSB_TRAVERSE_TRITON", "1") == "1"):
        return False
    from .fsb_traverse_triton import _HAVE_TRITON as _have_level
    if not (_have_level and fast_beam_ok(F, SP, q, W_max, delta_max) and fast_beam_ok(Fu, SPu, q, Wu_max, delta_max)):
        return False
    from .fsb_beam_cuda import beam_cuda_ok, rows_flag
    return rows_flag() and beam_cuda_ok(F, q, W_max, delta_max) and beam_cuda_ok(Fu, q, Wu_max, delta_max)


@torch.no_grad()
def forest_beam_pair(F: dict, SP: dict, W: torch.Tensor, delta: torch.Tensor, W_max: int, Fu: dict, SPu: dict,
                     Wu: torch.Tensor, du: torch.Tensor, Wu_max: int, q: torch.Tensor, qpos: torch.Tensor,
                     lam: torch.Tensor, sigv: torch.Tensor, scale: float, s: int, delta_max: int, node_score: str,
                     radius_coef, bud=None, bud_u=None):
    # both beams of a stage (the main and the union beam over the same queries) in one CUDA launch ->
    # ((cells, n_visit), (cells_u, n_visit_u)), bit-identical to two _forest_beam calls; None when neither pair
    # kernel applies (the caller then runs the beams one by one). bud / bud_u (2^delta, 2^du) may be passed when
    # the caller already has them, as for forest_beam_fused (delta / du are then unused)
    if not forest_beam_pair_ok(F, SP, Fu, SPu, q, W_max, Wu_max, delta_max):
        return None
    from .fsb_beam_cuda import beam_rows_ok, forest_beam_pair_cuda, forest_beam_rows_cuda
    if bud is None:
        bud = torch.bitwise_left_shift(torch.ones_like(delta), delta)
    if bud_u is None:
        bud_u = torch.bitwise_left_shift(torch.ones_like(du), du)
    beams = [(F, SP, W, bud, W_max), (Fu, SPu, Wu, bud_u, Wu_max)]
    if beam_rows_ok(F, q, W_max) and beam_rows_ok(Fu, q, Wu_max):
        out = forest_beam_rows_cuda(beams, q, qpos, lam, sigv, scale, s, delta_max, node_score, radius_coef)
    else:
        out = forest_beam_pair_cuda(beams, q, qpos, lam, sigv, scale, s, delta_max, node_score, radius_coef)
    return None if out is None else (out[0], out[1])


def fast_rerank_ok(sel: torch.Tensor, q: torch.Tensor, tokK, s: int, K: int, score: str) -> bool:
    if not (fast_rerank_on() and q.is_cuda and q.dtype == torch.float32 and q.shape[1] == 1 and tokK is not None):
        return False
    C = sel.shape[-1]
    return (score in ("lse", "lse_pos", "max") and sel.dtype == torch.int64 and sel.is_cuda
            and sel.is_contiguous() and 1 <= K <= min(16, C) and C <= 512 and s <= 256
            and q.shape[-1] <= 256)


@torch.no_grad()
def rerank_scores_fast(sel: torch.Tensor, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor,
                       sigv: torch.Tensor, cellA: torch.Tensor, t_idx: torch.Tensor, scale: float, s: int,
                       score: str, tokK: torch.Tensor) -> torch.Tensor:
    # rerank_top's u scores (T',Bq,C), -inf on empty slots: token scores in one kernel, plus rerank_top's own
    # torch position term for lse_pos (torch's erf and division round differently from Triton's)
    from .fsb_rerank import _key4
    Tp, Bq, C = sel.shape
    Dh = q.shape[-1]
    K4 = _key4(tokK)
    u = torch.empty((Tp, Bq, C), dtype=torch.float32, device=q.device)
    pt = u
    if score == "lse_pos":
        Mc = cellA.shape[1]
        gid = t_idx.view(Tp, 1, 1) * Mc + sel.clamp(min=0, max=Mc - 1)
        a = cellA.reshape(-1)[gid.reshape(-1)].view(Tp, Bq, C)
        pos_c = (sel.float() + 0.5) * s
        pt = (lam * a * torch.erf((qpos[..., None] - pos_c).abs() / (2.0 * sigv))).contiguous()
    _fsb_u_kernel[(Tp * Bq * C,)](q.reshape(Tp * Bq, Dh).contiguous(), sel, t_idx.contiguous(), K4, u, pt,
                                  C, Bq, K4.shape[1], K4.stride(0), K4.stride(1), K4.stride(2), float(scale), s,
                                  DH=Dh, DHP=_pow2(Dh), SP=_pow2(s), LSE=(score != "max"), POS=(score == "lse_pos"),
                                  num_warps=4, enable_fp_fusion=False)
    return u


@torch.no_grad()
def rerank_top_fast(sel: torch.Tensor, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor,
                    sigv: torch.Tensor, cellA: torch.Tensor, t_idx: torch.Tensor, scale: float, s: int,
                    K: int, score: str, tokK: torch.Tensor) -> torch.Tensor:
    # bit-exact fused rerank_top (no rr_w / read margin / prior scale / prov): -> (T',Bq,K) int64
    Tp, Bq, C = sel.shape
    u = rerank_scores_fast(sel, q, qpos, lam, sigv, cellA, t_idx, scale, s, score, tokK)
    out = torch.empty((Tp, Bq, K), dtype=torch.long, device=q.device)
    _fsb_sel_kernel[(Tp * Bq,)](u, sel, out, C, K, CWP=_pow2(C), KP2=_pow2(K))
    return out


def fast_union_rerank_ok(Ca: int, Cb: int, q: torch.Tensor, tokK, s: int, K: int, score: str) -> bool:
    C = Ca + Cb
    return (fast_rerank_on() and q.is_cuda and q.dtype == torch.float32 and tokK is not None
            and score in ("lse", "lse_pos", "max") and 1 <= K <= min(16, C) and C <= 512 and s <= 256
            and q.shape[-1] <= 256)


def _row_view(x: torch.Tensor, Tp: int, Bq: int):
    # a per-row value of shape (T',1,1), (T',Bq,1), (T',1) or (T',Bq) -> (view, stride over t', stride over b)
    v = x.reshape(x.shape[0], -1).expand(Tp, Bq)
    return v, v.stride(0), v.stride(1)


@torch.no_grad()
def union_rerank_fast(sel_a: torch.Tensor, sel_b, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor,
                      sigv: torch.Tensor, cellA: torch.Tensor, t_idx: torch.Tensor, scale: float, s: int, K: int,
                      score: str, tokK: torch.Tensor, prov: bool = False):
    # bit-exact union_candidates + rerank_top (no rr_w / read margin / prior scale) + utau_prov_add: u scores (with
    # union deduplication and exact CUDA erff fused) one program per (row, candidate) and the stable top-K /
    # provenance in a second one -> (read cells (T',Bq,K) int64, provenance (T',Bq,K) int8 when prov else None)
    from .fsb_rerank import PROV_UTAU, _key4
    Tp, Bq, Ca = sel_a.shape
    Cb = 0 if sel_b is None else sel_b.shape[-1]
    C, rows, Dh, dev = Ca + Cb, Tp * Bq, q.shape[-1], q.device
    pos = score == "lse_pos"
    qv, sq0, sq1 = _row_view(qpos, Tp, Bq)
    sv, ss0, ss1 = _row_view(sigv, Tp, Bq)
    K4 = _key4(tokK)
    lv, ls0, ls1 = _row_view(lam, Tp, Bq)
    u = torch.empty((rows, C), dtype=torch.float32, device=dev)
    sb_arg = sel_b if Cb else sel_a
    _fsb_upar_kernel[(rows * C,)](q.reshape(rows, Dh).contiguous(), sel_a, sb_arg, t_idx.contiguous(), K4, qv, sq0, sq1,
                                  sv, ss0, ss1, cellA, cellA.stride(0), cellA.stride(1), lv, ls0, ls1, u,
                                  Ca, Cb, C, Bq, K4.shape[1], K4.stride(0), K4.stride(1), K4.stride(2), float(scale),
                                  s, cellA.shape[1], CAP=_pow2(Ca), DH=Dh, DHP=_pow2(Dh), SP=_pow2(s),
                                  HAS_B=Cb > 0, LSE=(score != "max"), POS=pos,
                                  num_warps=2, enable_fp_fusion=False)
    out = torch.empty((Tp, Bq, K), dtype=torch.long, device=dev)
    pv = torch.empty((Tp, Bq, K), dtype=torch.int8, device=dev) if prov else out
    cwp = _pow2(C)
    _fsb_selpv_kernel[(rows,)](u, sel_a, sb_arg, out, pv, C, K, Ca, Cb, PROV_UTAU, CWP=cwp, KP2=_pow2(K),
                               HAS_B=Cb > 0, PRV=prov, num_warps=1 if cwp <= 64 else 4, enable_fp_fusion=False)
    return out, (pv if prov else None)



def decode_readout_ok(attn, q_t: torch.Tensor, sel, pos_t, Kp, Vp) -> bool:
    # the fused decode readout's support envelope (callers reach it only while a decode graph is being
    # captured); anything outside it uses the reference eager readout
    if not (_HAVE_TRITON and bool(attn._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0)) and q_t.is_cuda
            and torch.cuda.is_current_stream_capturing()):
        return False
    if q_t.dim() != 4 or q_t.shape[2] != 1 or not torch.is_tensor(sel) or sel.dim() != 3:
        return False
    B, H, _, Dh = q_t.shape
    if (sel.shape[0] != B * H or sel.shape[1] != 1 or sel.numel() == 0
            or sel.dtype != torch.int64 or not torch.is_tensor(pos_t) or pos_t.dim() != 0):
        return False
    if Kp.dim() != 4 or Kp.shape != Vp.shape or Kp.shape[0] < B or Kp.shape[1] < H:
        return False
    K, s = sel.shape[-1], attn.cell_stride
    if K > 16 or s > 256 or Dh > 128:
        return False
    if Kp.stride(3) != 1 or Vp.stride(3) != 1 or q_t.stride(3) != 1:
        return False
    dt = q_t.dtype
    if dt not in (torch.bfloat16, torch.float32) or getattr(attn, "dtype", dt) != dt:
        return False
    if (Kp.dtype != dt or Vp.dtype != dt or attn.lr_norm.scale.dtype != dt):
        return False
    return _pow2(K) * _pow2(s) * _pow2(Dh) <= 16384


@torch.no_grad()
def decode_readout_fast(attn, cache, q_t: torch.Tensor, sel, pos_t, sr=None):
    # _fsb_decode_readout (full_buffer source selection) as one kernel per decode query; None when unsupported.
    # The capture check runs before anything touches the cache: on eager tokens the readout token buffers must
    # stay untouched for the reference chain to build them once.
    if not (_HAVE_TRITON and q_t.is_cuda and torch.cuda.is_current_stream_capturing()):
        return None
    Kp, Vp = ((cache._k_buf, cache._v_buf) if getattr(cache, "_k_buf", None) is not None
              else cache.lr_tokens())
    if not decode_readout_ok(attn, q_t, sel, pos_t, Kp, Vp):
        return None
    B, H, _, Dh = q_t.shape
    K, s = sel.shape[-1], attn.cell_stride
    q3 = q_t[:, :, 0, :]
    sel2 = sel.contiguous().view(B * H, K)
    lrs = attn.long_range_scale.view(-1)
    has_sr = (sr is not None and sr.shape == q_t.shape and sr.dtype == q_t.dtype
              and sr.stride(3) == 1 and lrs.is_contiguous() and lrs.dtype in (torch.bfloat16, torch.float32))
    if has_sr:
        sr3 = sr[:, :, 0, :]
        out = torch.empty(B, 1, H * Dh, device=q_t.device, dtype=q_t.dtype)
        sob, soh = H * Dh, Dh
        srb, srh = sr3.stride(0), sr3.stride(1)
    else:
        sr3 = q3
        out = torch.empty(B, H, 1, Dh, device=q_t.device, dtype=q_t.dtype)
        sob, soh = out.stride(0), out.stride(1)
        srb, srh = 0, 0
    lrs_bf16 = (q_t.dtype == torch.bfloat16) and (lrs.dtype == torch.bfloat16)
    _fsb_decread_kernel[(B * H,)](
        q3, sel2, Kp, Vp, pos_t, attn.lr_norm.scale, sr3, lrs, out,
        q3.stride(0), q3.stride(1), Kp.stride(0), Kp.stride(1), Kp.stride(2),
        Vp.stride(0), Vp.stride(1), Vp.stride(2), srb, srh, sob, soh,
        float(attn.scale), float(attn.lr_norm.eps), H, K, s,
        DH=Dh, DHP=_pow2(Dh), SP=_pow2(s), KK=_pow2(K),
        HAS_SR=has_sr, LRS_BF16=lrs_bf16,
        num_warps=4, enable_fp_fusion=False)
    return out
