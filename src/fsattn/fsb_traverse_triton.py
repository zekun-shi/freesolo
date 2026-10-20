import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False

NEG = -1e9
BIG = 1e30

if _HAVE_TRITON:
    _NEG = tl.constexpr(-1e9)
    _BIG = tl.constexpr(1e30)

    @triton.jit
    def _halve_rows(x, D: tl.constexpr):
        # x: (C, D) -> (C,) by pairwise halving over D (D a power of two <= 256); each step one add
        if D >= 256:
            x = tl.sum(tl.reshape(x, (x.shape[0], 2, 128)), axis=1)
        if D >= 128:
            x = tl.sum(tl.reshape(x, (x.shape[0], 2, 64)), axis=1)
        if D >= 64:
            x = tl.sum(tl.reshape(x, (x.shape[0], 2, 32)), axis=1)
        if D >= 32:
            x = tl.sum(tl.reshape(x, (x.shape[0], 2, 16)), axis=1)
        if D >= 16:
            x = tl.sum(tl.reshape(x, (x.shape[0], 2, 8)), axis=1)
        if D >= 8:
            x = tl.sum(tl.reshape(x, (x.shape[0], 2, 4)), axis=1)
        if D >= 4:
            x = tl.sum(tl.reshape(x, (x.shape[0], 2, 2)), axis=1)
        if D >= 2:
            x = tl.sum(tl.reshape(x, (x.shape[0], 2, 1)), axis=1)
        return tl.sum(x, axis=1)

    @triton.jit
    def _level_kernel(REF, NEWREF, NVIS, Q, QN, QPOS, LAM, SIG, WBEAM, BUDGET, TIDX,
                      CHUNK_MU, CHUNK_AX, SPINE, Ntot, Sp, Bq, scale, cell_size, NC: tl.constexpr,
                      NA: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr, WM: tl.constexpr, CW: tl.constexpr,
                      CWP: tl.constexpr, BOUND: tl.constexpr):
        pid = tl.program_id(0)                                           # = t' * Bq + b
        t = pid // Bq
        b = pid - t * Bq
        tidx = tl.load(TIDX + t)
        w = tl.arange(0, WM)
        ref = tl.load(REF + pid * WM + w)                                # (WM,) int64
        active = ref != -1
        is_c = ref >= 0
        # node aux offset of each frontier ref (NA floats in the chunk slab, NC in a spine row)
        base_c = (tidx * Ntot + tl.where(is_c, ref, 0)) * NA
        base_s = (t * Sp + tl.where(ref <= -2, -ref - 2, 0)) * NC
        cnt = tl.where(is_c, tl.load(CHUNK_AX + base_c + 3, mask=is_c, other=0.0),
                       tl.load(SPINE + base_s + DH + 3, mask=(ref <= -2), other=0.0))
        chl = tl.where(is_c, tl.load(CHUNK_AX + base_c + 4, mask=is_c, other=-1.0),
                       tl.load(SPINE + base_s + DH + 4, mask=(ref <= -2), other=-1.0))
        chr_ = tl.where(is_c, tl.load(CHUNK_AX + base_c + 5, mask=is_c, other=-1.0),
                        tl.load(SPINE + base_s + DH + 5, mask=(ref <= -2), other=-1.0))
        budget = tl.load(BUDGET + pid)
        expand = active & (cnt > budget.to(tl.float32))
        carried = active & (cnt <= budget.to(tl.float32))
        # child refs (2*WM,) interleaved [l0, r0, l1, r1, ...] as torch's view of (WM, 2)
        c = tl.arange(0, 2 * WM)
        par = c // 2
        side = c - par * 2
        # gather parent's expand/child via one-hot reduction (WM small)
        eq = (par[:, None] == w[None, :])
        p_exp = tl.sum(tl.where(eq, expand[None, :].to(tl.int32), 0), axis=1) > 0
        p_chl = tl.sum(tl.where(eq, chl[None, :], 0.0), axis=1)
        p_chr = tl.sum(tl.where(eq, chr_[None, :], 0.0), axis=1)
        chv = tl.where(side == 0, p_chl, p_chr).to(tl.int64)
        chv = tl.where(p_exp, chv, -1)
        c_is_c = chv >= 0
        c_is_s = chv <= -2
        cb_c = (tidx * Ntot + tl.where(c_is_c, chv, 0)) * DH
        cb_s = (t * Sp + tl.where(c_is_s, -chv - 2, 0)) * NC
        ca_c = (tidx * Ntot + tl.where(c_is_c, chv, 0)) * NA
        d = tl.arange(0, DHP)
        dmask = d < DH
        mu_c = tl.load(CHUNK_MU + cb_c[:, None] + d[None, :],
                       mask=c_is_c[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        mu_s = tl.load(SPINE + cb_s[:, None] + d[None, :], mask=c_is_s[:, None] & dmask[None, :], other=0.0)
        mu = tl.where(c_is_c[:, None], mu_c, mu_s)                        # (2WM, DHP), zero beyond DH
        rad = tl.where(c_is_c, tl.load(CHUNK_AX + ca_c, mask=c_is_c, other=0.0), tl.load(SPINE + cb_s + DH, mask=c_is_s, other=0.0))
        amax = tl.where(c_is_c, tl.load(CHUNK_AX + ca_c + 1, mask=c_is_c, other=0.0), tl.load(SPINE + cb_s + DH + 1, mask=c_is_s, other=0.0))
        pmin = tl.where(c_is_c, tl.load(CHUNK_AX + ca_c + 2, mask=c_is_c, other=0.0), tl.load(SPINE + cb_s + DH + 2, mask=c_is_s, other=0.0))
        q = tl.load(Q + pid * DH + d, mask=dmask, other=0.0)             # (DHP,)
        prod = q[None, :] * mu                                           # fp32 product, zero-padded
        content = _halve_rows(prod, DHP) * scale
        if BOUND:
            qn = tl.load(QN + pid)
            content = content + (qn * rad) * scale
        qpos = tl.load(QPOS + pid)
        lam = tl.load(LAM + t)
        sig = tl.load(SIG + t)
        pmin_pos = (pmin + 0.5) * cell_size
        erf = tl.math.erf(tl.abs(qpos - pmin_pos) / (2.0 * sig))
        score = content + (lam * amax) * erf
        score = tl.where(chv == -1, _NEG, score)
        # candidates: [WM frontier slots | 2WM children], padded to CWP with _NEG/-1
        i = tl.arange(0, CWP)
        is_front = i < WM
        is_child = (i >= WM) & (i < CW)
        # frontier slot values
        eqf = (i[:, None] == w[None, :])
        f_car = tl.sum(tl.where(eqf, carried[None, :].to(tl.int32), 0), axis=1) > 0
        f_ref = tl.sum(tl.where(eqf, ref[None, :], 0), axis=1)
        eqc = ((i - WM)[:, None] == c[None, :])
        c_sc = tl.sum(tl.where(eqc, score[None, :], 0.0), axis=1)
        c_ref = tl.sum(tl.where(eqc, chv[None, :], 0), axis=1)
        cand_sc = tl.where(is_front, tl.where(f_car, _BIG, _NEG), tl.where(is_child, c_sc, _NEG))
        cand_ref = tl.where(is_front, tl.where(f_car, f_ref, -1), tl.where(is_child, c_ref, -1))
        cand_ref = tl.where(i < CW, cand_ref, -1)
        # stable descending rank: number of j with sc_j > sc_i or (sc_j == sc_i and j < i), padded slots last
        gt = (cand_sc[None, :] > cand_sc[:, None]) | ((cand_sc[None, :] == cand_sc[:, None]) & (i[None, :] < i[:, None]))
        gt = gt & (i[None, :] < CW)
        rank = tl.sum(gt.to(tl.int32), axis=1)                           # (CWP,)
        wbeam = tl.load(WBEAM + pid)
        # slot k <- candidate with rank k, if its value > _NEG/2 and k < wbeam
        sel_ok = (cand_sc > _NEG / 2) & (rank < wbeam) & (i < CW)
        ok_rank = tl.where(sel_ok, rank, CWP)
        eqr = (w[:, None] == ok_rank[None, :])                           # (WM, CWP)
        newref = tl.sum(tl.where(eqr, cand_ref[None, :], 0), axis=1)
        has = tl.sum(eqr.to(tl.int32), axis=1) > 0
        newref = tl.where(has, newref, -1)
        tl.store(NEWREF + pid * WM + w, newref)
        nvis = tl.sum((chv != -1).to(tl.float32), axis=0)
        tl.store(NVIS + pid, nvis)


def level_step(ref, q, qn, qpos, lam, sigv, W, budget, t_idx, chunk_mu, chunk_ax, spine, scale, s, W_max,
               node_score):
    # one traversal level for all (t', b): ref (T',Bq,W_max) int64 -> new ref, n_visit_add (T',Bq)
    Tp, Bq, WM = ref.shape
    Dh = q.shape[-1]
    NC = spine.shape[-1]
    DHP = 1
    while DHP < Dh:
        DHP <<= 1
    CW = 3 * WM
    CWP = 1
    while CWP < CW:
        CWP <<= 1
    newref = torch.empty_like(ref)
    nvis = torch.empty(Tp, Bq, dtype=torch.float32, device=ref.device)
    # no multiply-add contraction: with it, ptxas fuses different products for different thread layouts and the
    # node scores (and so the beams) would depend on the launch width; without it every FS-B kernel is IEEE
    # mul / add / div_full and any layout of the fused beam reproduces this kernel bit for bit
    _level_kernel[(Tp * Bq,)](ref, newref, nvis, q.contiguous(), qn.contiguous(), qpos.contiguous(),
                              lam.reshape(-1).contiguous(), sigv.reshape(-1).contiguous(),
                              W.contiguous(), budget.contiguous(), t_idx.contiguous(),
                              chunk_mu, chunk_ax, spine, chunk_mu.shape[1], spine.shape[1], Bq, float(scale),
                              float(s), NC=NC, NA=chunk_ax.shape[-1], DH=Dh, DHP=DHP, WM=WM, CW=CW, CWP=CWP,
                              BOUND=(node_score == "bound"), enable_fp_fusion=False)
    return newref, nvis
