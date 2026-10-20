"""Fused FS-B tree construction (on by default; FSB_FAST_BUILD=0 restores the reference).

build_trees_fast reproduces build_trees bit for bit: the same gathers and masks, pairwise-halving sums over the
node entries and the head dimension, IEEE division and square root, no multiply-add contraction, first-index
argmax, and torch's stable sort order. Nodes of up to 64 entries run one program per node and level; larger nodes
run a short kernel sequence whose split reuses torch.argsort, so its tie order is the reference order.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    from .fsb_traverse_triton import _halve_rows
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False

_CAP_SMALL = 64
_CAP_MAX = 16384
_TILE = 8192


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


if _HAVE_TRITON:
    @triton.jit
    def _halve_cols(x, P: tl.constexpr):
        # x: (P, C) -> (C,), pairwise halving over axis 0 (x[:h] + x[h:]), P a power of two <= 16384
        if P >= 16384:
            x = tl.sum(tl.reshape(x, (2, 8192, x.shape[1])), axis=0)
        if P >= 8192:
            x = tl.sum(tl.reshape(x, (2, 4096, x.shape[1])), axis=0)
        if P >= 4096:
            x = tl.sum(tl.reshape(x, (2, 2048, x.shape[1])), axis=0)
        if P >= 2048:
            x = tl.sum(tl.reshape(x, (2, 1024, x.shape[1])), axis=0)
        if P >= 1024:
            x = tl.sum(tl.reshape(x, (2, 512, x.shape[1])), axis=0)
        if P >= 512:
            x = tl.sum(tl.reshape(x, (2, 256, x.shape[1])), axis=0)
        if P >= 256:
            x = tl.sum(tl.reshape(x, (2, 128, x.shape[1])), axis=0)
        if P >= 128:
            x = tl.sum(tl.reshape(x, (2, 64, x.shape[1])), axis=0)
        if P >= 64:
            x = tl.sum(tl.reshape(x, (2, 32, x.shape[1])), axis=0)
        if P >= 32:
            x = tl.sum(tl.reshape(x, (2, 16, x.shape[1])), axis=0)
        if P >= 16:
            x = tl.sum(tl.reshape(x, (2, 8, x.shape[1])), axis=0)
        if P >= 8:
            x = tl.sum(tl.reshape(x, (2, 4, x.shape[1])), axis=0)
        if P >= 4:
            x = tl.sum(tl.reshape(x, (2, 2, x.shape[1])), axis=0)
        if P >= 2:
            x = tl.sum(tl.reshape(x, (2, 1, x.shape[1])), axis=0)
        return tl.sum(x, axis=0)

    @triton.jit
    def _unit(v, DHP: tl.constexpr):
        # v / max(sqrt(halving sum of v * v), 1e-6), correctly rounded
        nv = tl.sqrt_rn(tl.sum(_halve_rows(tl.reshape(v * v, (1, DHP)), DHP), axis=0))
        return tl.math.div_rn(v, tl.maximum(nv, 1e-6))

    @triton.jit
    def _lvl_small_kernel(SL, NSL, PK, KB, CE, SPR, MU, RAD, AMX, PMN, CNT,
                          nodes, M, cap0, pk_row, pk_off,
                          CAP: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr,
                          HAS_SP: tl.constexpr, SPLIT: tl.constexpr):
        # one (row, node) of one level: summaries, then the power-iteration split into the next level's slots
        pid = tl.program_id(0).to(tl.int64)
        r = pid // nodes
        n = pid - r * nodes
        j = tl.arange(0, CAP)
        d = tl.arange(0, DHP)
        dm = d < DH
        sb = r * cap0 + n * CAP
        S = tl.load(SL + sb + j)
        valid = S >= 0
        vf = valid.to(tl.float32)
        m = tl.sum(valid.to(tl.int64), axis=0)
        cell = r * M + tl.maximum(S, 0)
        X = tl.load(KB + cell[:, None] * DH + d[None, :], mask=dm[None, :], other=0.0) * vf[:, None]
        mu = tl.math.div_rn(_halve_cols(X, CAP), tl.maximum(m, 1).to(tl.float32))
        Xc = (X - mu[None, :]) * vf[:, None]
        d2 = _halve_rows(Xc * Xc, DHP)
        if HAS_SP:
            rad = tl.max(tl.where(valid, tl.sqrt_rn(d2) + tl.load(SPR + cell), 0.0), axis=0)
        else:
            rad = tl.sqrt_rn(tl.max(d2, axis=0))
        amax = tl.maximum(tl.max(tl.where(valid, tl.load(CE + cell), -1e9), axis=0), 0.0)
        pmin = tl.min(tl.where(valid, S, M), axis=0)
        ob = r * nodes + n
        tl.store(MU + ob * DH + d, mu, mask=dm)
        tl.store(RAD + ob, rad)
        tl.store(AMX + ob, amax)
        tl.store(PMN + ob, pmin)
        tl.store(CNT + ob, m)
        tl.store(PK + r * pk_row + pk_off + n * CAP + j, S)
        if SPLIT:
            far = tl.argmax(tl.where(valid, d2, -1.0), axis=0, tie_break_left=True)
            # exact row extraction: integer sum of the selected row's bit patterns
            v = tl.sum(tl.where((j == far)[:, None], Xc.to(tl.int32, bitcast=True), 0), axis=0)
            v = _unit(v.to(tl.float32, bitcast=True), DHP)
            pv = _halve_rows(Xc * v[None, :], DHP)
            v = _unit(_halve_cols(Xc * pv[:, None], CAP), DHP)
            pv = _halve_rows(Xc * v[None, :], DHP)
            v = _unit(_halve_cols(Xc * pv[:, None], CAP), DHP)
            proj = tl.where(valid, _halve_rows(Xc * v[None, :], DHP), float("inf"))
            # stable ascending rank; nodes of <= 128 entries sort by value comparison in torch
            lt = (proj[None, :] < proj[:, None]) | ((proj[None, :] == proj[:, None]) & (j[None, :] < j[:, None]))
            rank = tl.sum(lt.to(tl.int32), axis=1)
            nleft = (m + 1) // 2
            dest = tl.where(rank < nleft, rank, CAP // 2 + rank - nleft)
            hit = (dest[None, :] == j[:, None]) & valid[None, :]
            tl.store(NSL + sb + j, tl.sum(tl.where(hit, S[None, :] + 1, 0), axis=1) - 1)

    @triton.jit
    def _big_musum_kernel(SL, KB, MUS, CNT, nodes, M, cap0, CAP: tl.constexpr, CO: tl.constexpr, ND: tl.constexpr,
                          DH: tl.constexpr):
        # halving sum over a node's entries of the gathered, masked keys, CO head dimensions per program;
        # the first dimension block also stores the node's entry count
        pid = tl.program_id(0).to(tl.int64)
        rn = pid // ND
        db = pid - rn * ND
        r = rn // nodes
        n = rn - r * nodes
        j = tl.arange(0, CAP)
        dd = db * CO + tl.arange(0, CO)
        S = tl.load(SL + r * cap0 + n * CAP + j)
        valid = S >= 0
        cell = r * M + tl.maximum(S, 0)
        x = tl.load(KB + cell[:, None] * DH + dd[None, :], mask=dd[None, :] < DH, other=0.0)
        x = x * valid.to(tl.float32)[:, None]
        tl.store(MUS + rn * DH + dd, _halve_cols(x, CAP), mask=dd < DH)
        if db == 0:
            tl.store(CNT + rn, tl.sum(valid.to(tl.int64), axis=0))

    @triton.jit
    def _big_center_kernel(SL, KB, MUS, CNT, XC, D2, nodes, M, cap0,
                           CAP: tl.constexpr, EB: tl.constexpr, NB: tl.constexpr, DH: tl.constexpr,
                           DHP: tl.constexpr):
        # centered entries Xc = (X - mu) * valid (stored with row stride DHP) and their squared norms
        pid = tl.program_id(0).to(tl.int64)
        rn = pid // NB
        eb = pid - rn * NB
        r = rn // nodes
        n = rn - r * nodes
        e = r * cap0 + n * CAP + eb * EB + tl.arange(0, EB)
        d = tl.arange(0, DHP)
        dm = d < DH
        S = tl.load(SL + e)
        vf = (S >= 0).to(tl.float32)
        cell = r * M + tl.maximum(S, 0)
        X = tl.load(KB + cell[:, None] * DH + d[None, :], mask=dm[None, :], other=0.0) * vf[:, None]
        mf = tl.maximum(tl.load(CNT + rn), 1).to(tl.float32)
        mu = tl.math.div_rn(tl.load(MUS + rn * DH + d, mask=dm, other=0.0), mf)
        Xc = (X - mu[None, :]) * vf[:, None]
        tl.store(XC + e[:, None] * DHP + d[None, :], Xc)
        tl.store(D2 + e, _halve_rows(Xc * Xc, DHP))

    @triton.jit
    def _big_node_kernel(SL, CE, SPR, D2, XC, MUS, CNT, MU, RAD, AMX, PMN, PK, V,
                         nodes, M, cap0, pk_row, pk_off,
                         CAP: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr,
                         HAS_SP: tl.constexpr, SPLIT: tl.constexpr):
        # per-node summaries over the stored squared norms, and the split's unit start vector Xc[far]
        pid = tl.program_id(0).to(tl.int64)
        r = pid // nodes
        n = pid - r * nodes
        j = tl.arange(0, CAP)
        d = tl.arange(0, DHP)
        dm = d < DH
        base = r * cap0 + n * CAP
        S = tl.load(SL + base + j)
        valid = S >= 0
        cell = r * M + tl.maximum(S, 0)
        d2 = tl.load(D2 + base + j)
        if HAS_SP:
            rad = tl.max(tl.where(valid, tl.sqrt_rn(d2) + tl.load(SPR + cell), 0.0), axis=0)
        else:
            rad = tl.sqrt_rn(tl.max(d2, axis=0))
        amax = tl.maximum(tl.max(tl.where(valid, tl.load(CE + cell), -1e9), axis=0), 0.0)
        pmin = tl.min(tl.where(valid, S, M), axis=0)
        mf = tl.maximum(tl.load(CNT + pid), 1).to(tl.float32)
        tl.store(MU + pid * DH + d, tl.math.div_rn(tl.load(MUS + pid * DH + d, mask=dm, other=0.0), mf), mask=dm)
        tl.store(RAD + pid, rad)
        tl.store(AMX + pid, amax)
        tl.store(PMN + pid, pmin)
        tl.store(PK + r * pk_row + pk_off + n * CAP + j, S)
        if SPLIT:
            far = tl.argmax(tl.where(valid, d2, -1.0), axis=0, tie_break_left=True)
            tl.store(V + pid * DHP + d, _unit(tl.load(XC + (base + far) * DHP + d), DHP))

    @triton.jit
    def _big_dot_kernel(XC, V, OUT, SL, nodes, cap0, CAP: tl.constexpr, EB: tl.constexpr, NB: tl.constexpr,
                        DHP: tl.constexpr, NORM: tl.constexpr, PROJ: tl.constexpr):
        # per-entry halving dot Xc . v (v normalized here when NORM); PROJ masks empty slots to +inf
        pid = tl.program_id(0).to(tl.int64)
        rn = pid // NB
        eb = pid - rn * NB
        r = rn // nodes
        n = rn - r * nodes
        e = r * cap0 + n * CAP + eb * EB + tl.arange(0, EB)
        d = tl.arange(0, DHP)
        v = tl.load(V + rn * DHP + d)
        if NORM:
            v = _unit(v, DHP)
        p = _halve_rows(tl.load(XC + e[:, None] * DHP + d[None, :]) * v[None, :], DHP)
        if PROJ:
            p = tl.where(tl.load(SL + e) >= 0, p, float("inf"))
        tl.store(OUT + e, p)

    @triton.jit
    def _big_vsum_kernel(XC, PV, VS, nodes, cap0, CAP: tl.constexpr, CO: tl.constexpr, ND: tl.constexpr,
                         DHP: tl.constexpr):
        # v = halving sum over a node's entries of Xc * pv, CO head dimensions per program
        pid = tl.program_id(0).to(tl.int64)
        rn = pid // ND
        db = pid - rn * ND
        r = rn // nodes
        n = rn - r * nodes
        e = r * cap0 + n * CAP + tl.arange(0, CAP)
        dd = db * CO + tl.arange(0, CO)
        x = tl.load(XC + e[:, None] * DHP + dd[None, :]) * tl.load(PV + e)[:, None]
        tl.store(VS + rn * DHP + dd, _halve_cols(x, CAP))

    @triton.jit
    def _big_split_kernel(SL, ORD, CNT, NSL, nodes, cap0, CAP: tl.constexpr):
        # sorted entry p of a node goes to p (left half) or CAP/2 + p - nleft (right half); empty slots drop
        pid = tl.program_id(0).to(tl.int64)
        r = pid // nodes
        n = pid - r * nodes
        base = r * cap0 + n * CAP
        p = tl.arange(0, CAP)
        Ss = tl.load(SL + base + tl.load(ORD + base + p))
        nleft = (tl.load(CNT + pid) + 1) // 2
        dest = tl.where(p < nleft, p, CAP // 2 + p - nleft)
        tl.store(NSL + base + dest, Ss, mask=Ss >= 0)

    @triton.jit
    def _spine_kernel(MB, TIDX, NODE_MU, NODE_AX, BASE, NPR, HASK, JB, SPO, ROOT, HGT, Ntot, S, jb_row,
                      DH: tl.constexpr, DHP: tl.constexpr, NA: tl.constexpr, NC: tl.constexpr, HAS_JB: tl.constexpr,
                      PK: tl.constexpr):
        # one prefix row: merge the chunk roots selected by the bits of m_b into the spine, low bit first. PK=1:
        # the source is the packed fp32 node table (row stride NC, aux at DH..); PK=0: the split lean arena
        # (bf16 means at stride DH, fp32 aux at stride NA)
        p = tl.program_id(0).to(tl.int64)
        m = tl.load(MB + p)
        t = tl.load(TIDX + p)
        d = tl.arange(0, DHP)
        dm = d < DH
        r_mu = tl.zeros([DHP], dtype=tl.float32)
        r_rad = tl.zeros([], dtype=tl.float32)
        r_am = tl.zeros([], dtype=tl.float32)
        r_pm = tl.full([], 1073741824.0, tl.float32)
        r_cnt = tl.zeros([], dtype=tl.float32)
        r_ref = tl.full([], -1, tl.int64)
        pop = tl.zeros([], dtype=tl.int64)
        hb = tl.zeros([], dtype=tl.int64)
        for k in range(0, S):
            kk = k.to(tl.int64)
            off = m & ~((tl.full([], 2, tl.int64) << kk) - 1)
            j = off >> kk
            if HAS_JB:
                j = j - tl.load(JB + t * jb_row + kk)
            croot = tl.load(BASE + kk) + j * tl.load(NPR + kk)
            bit = ((m >> kk) & 1) != 0
            present = bit & (tl.load(HASK + kk) != 0)
            g = t * Ntot + tl.minimum(tl.maximum(croot, 0), Ntot - 1)
            if PK:
                c_mu = tl.load(NODE_MU + g * NC + d, mask=dm, other=0.0)
                c_rad = tl.load(NODE_MU + g * NC + DH)
                c_am = tl.load(NODE_MU + g * NC + DH + 1)
                c_pm = tl.load(NODE_MU + g * NC + DH + 2)
                c_cnt = tl.load(NODE_MU + g * NC + DH + 3)
            else:
                c_mu = tl.load(NODE_MU + g * DH + d, mask=dm, other=0.0).to(tl.float32)
                c_rad = tl.load(NODE_AX + g * NA)
                c_am = tl.load(NODE_AX + g * NA + 1)
                c_pm = tl.load(NODE_AX + g * NA + 2)
                c_cnt = tl.load(NODE_AX + g * NA + 3)
            mk = present & (r_cnt > 0)
            tot = tl.maximum(c_cnt + r_cnt, 1.0)
            n_mu = tl.math.div_rn(c_cnt * c_mu + r_cnt * r_mu, tot)
            a = c_mu - n_mu
            b = r_mu - n_mu
            nrm0 = tl.sqrt_rn(tl.sum(_halve_rows(tl.reshape(a * a, (1, DHP)), DHP), axis=0))
            nrm1 = tl.sqrt_rn(tl.sum(_halve_rows(tl.reshape(b * b, (1, DHP)), DHP), axis=0))
            n_rad = tl.maximum(c_rad + nrm0, r_rad + nrm1)
            n_am = tl.maximum(c_am, r_am)
            n_pm = tl.minimum(c_pm, r_pm)
            n_cnt = c_cnt + r_cnt
            row = SPO + (p * S + kk) * NC
            tl.store(row + d, tl.where(mk, n_mu, 0.0), mask=dm)
            tl.store(row + DH, tl.where(mk, n_rad, 0.0))
            tl.store(row + DH + 1, tl.where(mk, n_am, 0.0))
            tl.store(row + DH + 2, tl.where(mk, n_pm, 0.0))
            tl.store(row + DH + 3, tl.where(mk, n_cnt, 0.0))
            tl.store(row + DH + 4, tl.where(mk, croot.to(tl.float32), -1.0))
            tl.store(row + DH + 5, tl.where(mk, r_ref.to(tl.float32), -1.0))
            tl.store(row + DH + 6, tl.where(mk, 2.0, 0.0))
            tl.store(row + DH + 7, tl.where(mk, off.to(tl.float32), 0.0))
            tl.store(row + DH + 8, tl.where(mk, n_cnt, 0.0))
            r_mu = tl.where(mk, n_mu, tl.where(present, c_mu, r_mu))
            r_rad = tl.where(mk, n_rad, tl.where(present, c_rad, r_rad))
            r_am = tl.where(mk, n_am, tl.where(present, c_am, r_am))
            r_pm = tl.where(mk, n_pm, tl.where(present, c_pm, r_pm))
            r_cnt = tl.where(mk, n_cnt, tl.where(present, c_cnt, r_cnt))
            r_ref = tl.where(mk, -(kk + 2), tl.where(present, croot, r_ref))
            pop += present.to(tl.int64)
            hb = tl.where(bit, kk, hb)
        tl.store(ROOT + p, tl.where(m > 0, r_ref, -1))
        tl.store(HGT + p, tl.where(m > 1, hb + (pop >= 2).to(tl.int64), 0))

    @triton.jit
    def _append_adm_kernel(CE, TAU, ADM, MTOT, MC,
                           sB_e, sH_e, sM_e, adm_s0_e, H_e, T_e, n_seen_e, nn_e,
                           BLOCK_N: tl.constexpr):
        sB = (sB_e >> 2).to(tl.int64)
        sH = (sH_e >> 2).to(tl.int64)
        sM = (sM_e >> 2).to(tl.int64)
        adm_s0 = (adm_s0_e >> 2).to(tl.int64)
        H = (H_e >> 2).to(tl.int64)
        T = (T_e >> 2).to(tl.int64)
        n_seen = (n_seen_e >> 2).to(tl.int64)
        nn = (nn_e >> 2).to(tl.int64)
        t = tl.program_id(0).to(tl.int64)
        b = t // H
        h = t - b * H
        tau_t = tl.load(TAU + t)
        m0 = tl.load(MTOT + t)
        ce_row = CE + b * sB + h * sH
        adm_row = ADM + t * adm_s0
        j = tl.arange(0, BLOCK_N).to(tl.int64)
        tot_new = tl.zeros((), dtype=tl.int64)
        for off in range(0, nn, BLOCK_N):
            idx = off + j
            mask = idx < nn
            cid = n_seen + idx
            ce = tl.load(ce_row + cid * sM, mask=mask, other=-float("inf")).to(tl.float32)
            adm = (ce >= tau_t) & mask
            ai = adm.to(tl.int64)
            n_c = tl.sum(ai, axis=0)
            pref = tl.cumsum(ai, axis=0) - ai
            tl.store(adm_row + m0 + tot_new + pref, cid, mask=adm)
            tot_new += n_c
        for off in range(0, nn, BLOCK_N):
            idx = off + j
            tl.store(adm_row + m0 + tot_new + idx, -1, mask=idx < (nn - tot_new))
        m1 = m0 + tot_new
        tl.store(MTOT + t, m1)
        tl.store(MC + t, tot_new)
        tl.store(MC + T + t, m1)

    @triton.jit
    def _budgets_kernel(HM, HU, W, DL, BUD, UW, UD, BUDU, T, d, w, uw, UNION: tl.constexpr, MAIN_UW: tl.constexpr,
                        BLOCK: tl.constexpr):
        # the fixed policy's per-head beam budgets from the spines' heights h: W = min(w, 2^max(0, h - d)), delta =
        # h - max(0, h - d) = min(h, d), budget 2^delta; UNION: the union forest's (union_W, union_delta, its budget)
        # from its own heights; MAIN_UW: without one, the main forest's union width and budget
        t = tl.arange(0, BLOCK)
        m = t < T
        h = tl.load(HM + t, mask=m, other=0)
        D = tl.maximum(h - d, 0)
        one = tl.full([BLOCK], 1, tl.int64)
        tl.store(W + t, tl.minimum(one * w, one << D), mask=m)
        tl.store(DL + t, h - D, mask=m)
        tl.store(BUD + t, one << (h - D), mask=m)
        if UNION:
            hu = tl.load(HU + t, mask=m, other=0)
            Du = tl.maximum(hu - d, 0)
            tl.store(UD + t, hu - Du, mask=m)
            tl.store(UW + t, tl.minimum(one * uw, one << Du), mask=m)
            tl.store(BUDU + t, one << (hu - Du), mask=m)
        if MAIN_UW:
            tl.store(UW + t, tl.minimum(one * uw, one << D), mask=m)
            tl.store(BUDU + t, one << (h - D), mask=m)

    @triton.jit
    def _pack_lean_chunks_kernel(MTOT, JBASE, SRC_NODE, SRC_AX, SRC_LEAF, EMP, DST_MU, DST_AX, DST_LEAF,
                                 K_N, BASE_N, STR_N, U_N, DST_N,
                                 K_L, BASE_L, STR_L, U_L, DST_L,
                                 jb_s0_e, Kmax_e, Ntot_e, Ltot_e, ncap_e, lcap_e, U_tot_e, V_tot_e, n_leaf_blks_e,
                                 DH: tl.constexpr, DH3: tl.constexpr, DHP: tl.constexpr, NA: tl.constexpr,
                                 NC: tl.constexpr, PK: tl.constexpr):
        jb_s0 = (jb_s0_e >> 2).to(tl.int64)
        Kmax = (Kmax_e >> 2).to(tl.int64)
        Ntot = (Ntot_e >> 2).to(tl.int64)
        Ltot = (Ltot_e >> 2).to(tl.int64)
        ncap = (ncap_e >> 2).to(tl.int64)
        lcap = (lcap_e >> 2).to(tl.int64)
        U_tot = (U_tot_e >> 2).to(tl.int64)
        V_tot = (V_tot_e >> 2).to(tl.int64)
        n_leaf_blks = (n_leaf_blks_e >> 2).to(tl.int64)
        p = tl.program_id(0).to(tl.int64)
        t = tl.program_id(1).to(tl.int64)
        mi = tl.load(MTOT + t)
        if p == 0:
            kk = tl.arange(0, 32).to(tl.int64)
            km = kk <= Kmax
            bit = ((mi >> kk) & 1) != 0
            jb = tl.where(bit, (mi >> (kk + 1)) << 1, 0)
            tl.store(JBASE + t * jb_s0 + kk, jb, mask=km)
        if p < U_tot:
            k = tl.load(K_N + p).to(tl.int64)
            has = ((mi >> k) & 1) != 0
            j = tl.where(has, (mi >> (k + 1)) << 1, 0)
            sn = (tl.load(BASE_N + p).to(tl.int64)
                  + j * tl.load(STR_N + p).to(tl.int64)
                  + tl.load(U_N + p).to(tl.int64))
            dn = tl.load(DST_N + p).to(tl.int64)
            d = tl.arange(0, DHP).to(tl.int64)
            dm = d < DH3
            if PK:
                val = tl.load(SRC_NODE + (t * Ntot + sn) * NC + d, mask=dm & has, other=0.0)
            else:
                # the split table supplies its bf16 means already rounded: upcasting and re-rounding is the identity,
                # so the rounding slack this kernel would add is zero and the stored radius passes through
                val = tl.load(SRC_NODE + (t * Ntot + sn) * DH + d, mask=(d < DH) & has, other=0.0).to(tl.float32)
                av = tl.load(SRC_AX + (t * Ntot + sn) * NA + (d - DH), mask=dm & (d >= DH) & has, other=0.0)
                val = tl.where(d < DH, val, av)
            emp_v = tl.load(EMP + t * DH3 + d, mask=dm & (~has), other=0.0)
            row32 = tl.where(has, val, emp_v)
            mu_w = row32.to(tl.bfloat16)
            mu_up = mu_w.to(tl.float32)
            diff = tl.where(d < DH, row32 - mu_up, 0.0)
            eps = tl.sqrt_rn(tl.sum(diff * diff, axis=0))
            row_m = d < DH
            tl.store(DST_MU + (t * ncap + dn) * DH + d, mu_w, mask=row_m)
            rad = tl.sum(tl.where(d == DH, row32, 0.0), axis=0)
            am = tl.sum(tl.where(d == DH + 1, row32, 0.0), axis=0)
            pm = tl.sum(tl.where(d == DH + 2, row32, 0.0), axis=0)
            tl.store(DST_AX + (t * ncap + dn) * NA, rad + eps)
            tl.store(DST_AX + (t * ncap + dn) * NA + 1, am)
            tl.store(DST_AX + (t * ncap + dn) * NA + 2, pm)
        if p < n_leaf_blks:
            q = p * 64 + tl.arange(0, 64).to(tl.int64)
            qm = q < V_tot
            kl = tl.load(K_L + q, mask=qm, other=0).to(tl.int64)
            has_l = (((mi >> kl) & 1) != 0) & qm
            jl = (mi >> (kl + 1)) << 1
            sl = (tl.load(BASE_L + q, mask=qm, other=0).to(tl.int64)
                  + jl * tl.load(STR_L + q, mask=qm, other=0).to(tl.int64)
                  + tl.load(U_L + q, mask=qm, other=0).to(tl.int64))
            dl = tl.load(DST_L + q, mask=qm, other=0).to(tl.int64)
            lf = tl.load(SRC_LEAF + t * Ltot + sl, mask=has_l, other=-1)
            tl.store(DST_LEAF + t * lcap + dl, lf, mask=qm)


def fast_build_on() -> bool:
    return _HAVE_TRITON and os.environ.get("FSB_FAST_BUILD", "1") == "1"


def fast_build_ok(Kbar: torch.Tensor, L: int) -> bool:
    return (fast_build_on() and Kbar.is_cuda and Kbar.dtype == torch.float32 and Kbar.dim() == 3
            and 1 <= Kbar.shape[-1] <= 256 and (1 << L) <= _CAP_MAX)


def _warps(n: int) -> int:
    return 1 if n <= 256 else 2 if n <= 1024 else 4 if n <= 4096 else 8


@torch.no_grad()
def build_trees_fast(Kbar: torch.Tensor, ceil_c: torch.Tensor, admit: torch.Tensor, L: int, spread=None) -> dict:
    T, M, Dh = Kbar.shape
    dev = Kbar.device
    cap0 = 1 << L
    assert M <= cap0, (M, L)
    Kbar = Kbar.contiguous()
    ceil_c = ceil_c.float().contiguous()
    sp = spread.float().contiguous() if spread is not None else ceil_c
    ids = torch.arange(M, device=dev)
    m0 = admit.sum(-1)
    key = torch.where(admit, ids[None, :], ids[None, :] + M)
    order = key.argsort(dim=-1)
    packed0 = torch.where(ids[None, :] < m0[:, None], order, torch.full_like(order, -1))
    slots = torch.full((T, cap0), -1, dtype=torch.long, device=dev)
    slots[:, :M] = packed0
    packed = torch.empty((T, (L + 1) * cap0), dtype=torch.long, device=dev)
    DHP = _pow2(Dh)
    mu_l, rad_l, am_l, pm_l, cnt_l = [], [], [], [], []
    xc = d2 = vbuf = vs = pv = None
    for l in range(L + 1):
        nodes = 1 << l
        cap = cap0 >> l
        split = l < L
        R = T * nodes
        mu = torch.empty((T, nodes, Dh), dtype=torch.float32, device=dev)
        rad = torch.empty((T, nodes), dtype=torch.float32, device=dev)
        am = torch.empty((T, nodes), dtype=torch.float32, device=dev)
        pm = torch.empty((T, nodes), dtype=torch.long, device=dev)
        cnt = torch.empty((T, nodes), dtype=torch.long, device=dev)
        nsl = torch.full((T, cap0), -1, dtype=torch.long, device=dev) if split else slots
        if cap <= _CAP_SMALL:
            _lvl_small_kernel[(R,)](slots, nsl, packed, Kbar, ceil_c, sp, mu, rad, am, pm, cnt,
                                    nodes, M, cap0, packed.stride(0), l * cap0,
                                    CAP=cap, DH=Dh, DHP=DHP, HAS_SP=spread is not None, SPLIT=split,
                                    num_warps=_warps(cap * DHP), enable_fp_fusion=False)
        else:
            if xc is None:
                xc = torch.empty((T * cap0, DHP), dtype=torch.float32, device=dev)
                d2 = torch.empty((T * cap0,), dtype=torch.float32, device=dev)
                pv = torch.empty((T * cap0,), dtype=torch.float32, device=dev)
            mus = torch.empty((R, Dh), dtype=torch.float32, device=dev)
            vbuf = torch.empty((R, DHP), dtype=torch.float32, device=dev)
            vs = torch.empty((R, DHP), dtype=torch.float32, device=dev)
            co = max(1, min(_pow2(Dh), _TILE // cap))
            eb = 32
            nd = (Dh + co - 1) // co
            _big_musum_kernel[(R * nd,)](slots, Kbar, mus, cnt, nodes, M, cap0, CAP=cap, CO=co, ND=nd, DH=Dh,
                                         num_warps=_warps(cap * co), enable_fp_fusion=False)
            _big_center_kernel[(R * (cap // eb),)](slots, Kbar, mus, cnt, xc, d2, nodes, M, cap0,
                                                   CAP=cap, EB=eb, NB=cap // eb, DH=Dh, DHP=DHP,
                                                   num_warps=_warps(eb * DHP), enable_fp_fusion=False)
            _big_node_kernel[(R,)](slots, ceil_c, sp, d2, xc, mus, cnt, mu, rad, am, pm, packed, vbuf,
                                   nodes, M, cap0, packed.stride(0), l * cap0,
                                   CAP=cap, DH=Dh, DHP=DHP, HAS_SP=spread is not None, SPLIT=split,
                                   num_warps=_warps(cap), enable_fp_fusion=False)
            if split:
                cv = max(1, min(DHP, _TILE // cap))
                v_in, norm = vbuf, False
                for _ in range(2):
                    _big_dot_kernel[(R * (cap // eb),)](xc, v_in, pv, slots, nodes, cap0, CAP=cap, EB=eb,
                                                        NB=cap // eb, DHP=DHP, NORM=norm, PROJ=False,
                                                        num_warps=_warps(eb * DHP), enable_fp_fusion=False)
                    _big_vsum_kernel[(R * (DHP // cv),)](xc, pv, vs, nodes, cap0, CAP=cap, CO=cv, ND=DHP // cv,
                                                         DHP=DHP, num_warps=_warps(cap * cv), enable_fp_fusion=False)
                    v_in, norm = vs, True
                _big_dot_kernel[(R * (cap // eb),)](xc, vs, pv, slots, nodes, cap0, CAP=cap, EB=eb, NB=cap // eb,
                                                    DHP=DHP, NORM=True, PROJ=True, num_warps=_warps(eb * DHP),
                                                    enable_fp_fusion=False)
                order = pv.view(T, nodes, cap).argsort(dim=-1, stable=True).contiguous()
                _big_split_kernel[(R,)](slots, order, cnt, nsl, nodes, cap0, CAP=cap, num_warps=_warps(cap))
        mu_l.append(mu); rad_l.append(rad); am_l.append(am); pm_l.append(pm); cnt_l.append(cnt)
        if split:
            slots = nsl
    height = torch.log2(m0.clamp(min=1).float()).ceil().long()
    return {"L": L, "M": M, "mu": mu_l, "rad": rad_l, "amax": am_l, "pmin": pm_l, "cnt": cnt_l,
            "packed": packed, "m0": m0, "height": height}


def fast_spines_ok(F: dict, m_b: torch.Tensor) -> bool:
    # two exact sources, both read straight from their table so the result never depends on whether a beam call
    # left views cached on F: the packed fp32 node rows (build_forest / flattened caches) or the split lean arena
    if not (fast_build_on() and m_b.is_cuda and m_b.dtype == torch.int64 and 1 <= F.get("Dh", 0) <= 256):
        return False
    node = F.get("node")
    if node is not None:
        return node.is_cuda and node.dtype == torch.float32 and node.is_contiguous()
    mu = F.get("node_mu")
    return (mu is not None and mu.is_cuda and mu.dtype == torch.bfloat16 and mu.is_contiguous()
            and F["node_aux"].dtype == torch.float32 and F["node_aux"].is_contiguous())


@torch.no_grad()
def build_spines_fast(F: dict, m_b: torch.Tensor, t_idx: torch.Tensor, consts, out: dict | None = None) -> dict:
    # consts: (ks, bases, nodes_per, has_k[, has_k_i8]) tensors of length Kmax + 1, as cached by build_spines
    Tp = m_b.shape[0]
    Dh = F["Dh"]
    S = F["Kmax"] + 1
    node = F.get("node")
    pk = node is not None
    if pk:
        src_mu = src_ax = node
        NA = node.shape[-1] - Dh
    else:
        src_mu, src_ax = F["node_mu"], F["node_aux"]
        NA = src_ax.shape[-1]
    NC = Dh + NA
    _, base_t, npr_t, has_k = consts[:4]
    has_k_i8 = consts[4] if len(consts) > 4 else has_k.to(torch.int8)
    if out is not None and out["node"].shape == (Tp, S, NC):
        sp, root, height = out["node"], out["root"], out["height"]
    else:
        sp = torch.empty((Tp, S, NC), dtype=torch.float32, device=m_b.device)
        root = torch.empty((Tp,), dtype=torch.long, device=m_b.device)
        height = torch.empty((Tp,), dtype=torch.long, device=m_b.device)
    jb = F.get("jbase")
    if Tp > 0:
        _spine_kernel[(Tp,)](m_b.contiguous(), t_idx.contiguous(), src_mu, src_ax, base_t, npr_t,
                             has_k_i8, jb if jb is not None else m_b, sp, root, height, F["Ntot"], S,
                             jb.stride(0) if jb is not None else 0, DH=Dh, DHP=_pow2(Dh), NA=NA,
                             NC=NC, HAS_JB=jb is not None, PK=pk, num_warps=1, enable_fp_fusion=False)
    if out is not None and out["node"] is sp:
        out["m_b"] = m_b
        out["t_idx"] = t_idx
        return out
    return {"node": sp, "root": root, "height": height, "m_b": m_b, "t_idx": t_idx}


def _enc_i(x: int) -> int:
    return (int(x) << 2) | 3


@torch.no_grad()
def budgets_fast(h_main: torch.Tensor, h_union, outs: tuple, d: int, w: int, uw: int) -> bool:
    # _budgets_kernel for T heads: h_main / h_union the (T,) int64 spine heights (h_union None: no union forest), outs
    # = (W, delta, bud, union_W, union_delta, bud_u) contiguous int64 (T, 1) tensors (the union ones None where unused).
    # One launch, no host read; False where it does not apply (the caller computes them otherwise)
    W, dl, bud, u_w, u_d, bud_u = outs
    T = h_main.shape[0]
    main_uw = h_union is None and u_w is not None
    ts = [h_main, W, dl, bud] + ([h_union, u_w, u_d, bud_u] if h_union is not None else
                                 ([u_w, bud_u] if main_uw else []))
    if not (_HAVE_TRITON and 0 < T <= 1024 and all(x is not None and x.is_cuda and x.dtype == torch.int64
                                                    and x.is_contiguous() and x.numel() == T for x in ts)):
        return False
    _budgets_kernel[(1,)](h_main, h_union if h_union is not None else h_main, W, dl, bud,
                          u_w if u_w is not None else W, u_d if u_d is not None else dl,
                          bud_u if bud_u is not None else bud, T, d, w, uw,
                          UNION=h_union is not None, MAIN_UW=main_uw, BLOCK=_pow2(max(T, 16)), num_warps=1)
    return True


@torch.no_grad()
def append_admitted_fast(fc, ceil_all: torch.Tensor, M_vis: int, tau32: torch.Tensor) -> torch.Tensor | None:
    T = fc.T
    nn = M_vis - fc.n_seen
    if not (fast_build_on() and ceil_all.is_cuda and tau32.is_cuda and nn >= 1
            and ceil_all.dim() == 3 and ceil_all.shape[0] * ceil_all.shape[1] == T
            and ceil_all.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and tau32.numel() == T):
        return None
    cap = fc.adm.shape[1]
    if not (fc.fast and M_vis <= cap):
        capacity = max(M_vis, fc.m_cap, 2 * cap, 1) if fc.fast else cap + nn
        grown = torch.full((T, capacity), -1, dtype=torch.long, device=fc.dev)
        if cap > 0:
            grown[:, :cap] = fc.adm
        fc.adm = grown
    if fc._mc_buf is None or fc._mc_buf.shape != (2, T):
        fc._mc_buf = torch.empty((2, T), dtype=torch.long, device=fc.dev)
        fc._mc_nn = fc._mc_buf[0]
    H = ceil_all.shape[1]
    _append_adm_kernel[(T,)](ceil_all, tau32, fc.adm, fc.m_tot, fc._mc_buf,
                             _enc_i(ceil_all.stride(0)), _enc_i(ceil_all.stride(1)), _enc_i(ceil_all.stride(2)),
                             _enc_i(fc.adm.stride(0)), _enc_i(H), _enc_i(T), _enc_i(fc.n_seen), _enc_i(nn),
                             BLOCK_N=32, num_warps=1)
    fc.n_seen = M_vis
    return fc._mc_nn


@torch.no_grad()
def pack_lean_chunks_fast(fc, F_cuda: dict, emp: torch.Tensor, Kmax: int, idx_c: tuple) -> torch.Tensor:
    T, Dh, dev = fc.T, fc.Dh, fc.dev
    if fc._jbase_buf is None or fc._jbase_buf.shape != (T, Kmax + 1):
        fc._jbase_buf = torch.empty((T, Kmax + 1), dtype=torch.long, device=dev)
    jb = fc._jbase_buf
    k_n, base_n, str_n, u_n, dst_n, k_l, base_l, str_l, u_l, dst_l = idx_c
    U_tot = k_n.shape[0]
    V_tot = k_l.shape[0]
    n_leaf_blks = (V_tot + 63) // 64
    grid_p = max(U_tot, n_leaf_blks, 1)
    node_src = F_cuda.get("node")
    pk = node_src is not None
    if pk:
        ax_src = node_src
    else:
        node_src = F_cuda["node_mu"]
        ax_src = F_cuda["node_aux"]
    leaf_src = F_cuda["leaf"]
    NC = node_src.shape[-1]
    DH3 = Dh + 3
    DHP = _pow2(DH3)
    _pack_lean_chunks_kernel[(grid_p, T)](
        fc.m_tot, jb, node_src, ax_src, leaf_src, emp, fc._arena_mu, fc._arena_aux, fc._leaf_arena,
        k_n, base_n, str_n, u_n, dst_n,
        k_l, base_l, str_l, u_l, dst_l,
        _enc_i(jb.stride(0)), _enc_i(Kmax), _enc_i(F_cuda["Ntot"]), _enc_i(F_cuda["Ltot"]),
        _enc_i(fc._arena_mu.shape[1]), _enc_i(fc._leaf_arena.shape[1]),
        _enc_i(U_tot), _enc_i(V_tot), _enc_i(n_leaf_blks),
        DH=Dh, DH3=DH3, DHP=DHP, NA=9, NC=NC, PK=pk, num_warps=2)
    return jb

