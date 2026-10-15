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
    def _spine_kernel(MB, TIDX, NODE, BASE, NPR, HASK, JB, SPO, ROOT, HGT, Ntot, S, jb_row,
                      DH: tl.constexpr, DHP: tl.constexpr, NC: tl.constexpr, HAS_JB: tl.constexpr):
        # one prefix row: merge the chunk roots selected by the bits of m_b into the spine, low bit first
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
            g = (t * Ntot + tl.minimum(tl.maximum(croot, 0), Ntot - 1)) * NC
            c_mu = tl.load(NODE + g + d, mask=dm, other=0.0)
            c_rad = tl.load(NODE + g + DH)
            c_am = tl.load(NODE + g + DH + 1)
            c_pm = tl.load(NODE + g + DH + 2)
            c_cnt = tl.load(NODE + g + DH + 3)
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
    node = F["node"]
    return (fast_build_on() and m_b.is_cuda and node.is_cuda and node.dtype == torch.float32 and node.is_contiguous()
            and m_b.dtype == torch.int64 and 1 <= F["Dh"] <= 256)


@torch.no_grad()
def build_spines_fast(F: dict, m_b: torch.Tensor, t_idx: torch.Tensor, consts, out: dict | None = None) -> dict:
    # consts: (ks, bases, nodes_per, has_k[, has_k_i8]) tensors of length Kmax + 1, as cached by build_spines
    Tp = m_b.shape[0]
    Dh = F["Dh"]
    S = F["Kmax"] + 1
    NC = F["node"].shape[-1]
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
        _spine_kernel[(Tp,)](m_b.contiguous(), t_idx.contiguous(), F["node"], base_t, npr_t, has_k_i8,
                             jb if jb is not None else m_b, sp, root, height, F["Ntot"], S,
                             jb.stride(0) if jb is not None else 0, DH=Dh, DHP=_pow2(Dh), NC=NC,
                             HAS_JB=jb is not None, num_warps=1, enable_fp_fusion=False)
    if out is not None and out["node"] is sp:
        out["m_b"] = m_b
        out["t_idx"] = t_idx
        return out
    return {"node": sp, "root": root, "height": height, "m_b": m_b, "t_idx": t_idx}

