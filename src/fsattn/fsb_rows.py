"""Block-shared FS-B prefill re-rank (on by default; FSB_FAST_ROWS=0 restores rerank_top).

The 128 queries of one (head, query block) row draw their re-rank candidates from a few dozen distinct cells (about 50
at 16k and 70 at 131k tokens on RULER prompts, out of 128 candidate slots per query). union_rerank_rows loads each
distinct cell's keys once per row and scores them against every query of the row on tensor cores, then picks each
query's top K by the lse_pos / lse / max score of rerank_top. Numerically equivalent to union_candidates +
rerank_top (+ utau_prov_add), not bit-identical: bf16 products are accumulated in fp32 in tensor-core order and
erf is libdevice's rather than torch's, so rare near-ties can order differently.

A stage takes three launches (FSB_ROWS_IMPL=cuda, the default): fsb_rows_cuda's row union (the distinct-cell list,
the cell -> list position map and the beam-b repeats, from the beams' own slots), the LSE kernel on a (rows,
FSB_ROWS_LSE_SPLIT) grid and a select kernel over each row's (query x slot) tile; tiles larger than ROWS_TILE_MAX
(wide candidate layouts) take the per-query select instead (one more launch). FSB_ROWS_IMPL=legacy keeps the
per-query union kernel, its torch glue and the per-query select; all give the same cells bit for bit. Under
torch.compile rows_impl, rows_loskip_on and rows_defer_on read as their legacy settings (the path as before them).
"""
from __future__ import annotations

import os

import torch

from ..utils.compile_guard import compiling

try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice as _ld              # triton >= 3.1
    except Exception:
        from triton.language.extra.cuda import libdevice as _ld        # triton 3.0
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False

UCAP = 512                       # distinct candidate cells per row handled on the fast path


def rows_on() -> bool:
    return _HAVE_TRITON and os.environ.get("FSB_FAST_ROWS", "1") == "1"


def rows_impl() -> str:
    # cuda (default): fsb_rows_cuda's row union, the LSE kernel on a (rows, FSB_ROWS_LSE_SPLIT) grid and the row-tile
    # select (3 launches per stage; the per-query select on tiles past rows_tile_ok); legacy: the per-query union
    # kernel, its torch glue and the per-query select
    if compiling():
        return "legacy"
    v = os.environ.get("FSB_ROWS_IMPL", "cuda")
    return "legacy" if v == "legacy" else "cuda"


def rows_lse_split() -> int:
    return max(1, int(os.environ.get("FSB_ROWS_LSE_SPLIT", "4")))


def rows_loskip_on() -> bool:
    # the LSE kernel skips the low-part products of bf16-exact queries (a per-row test); FSB_ROWS_LOSKIP=0 always
    # computes them
    return not compiling() and os.environ.get("FSB_ROWS_LOSKIP", "1") == "1"


def rows_defer_on() -> bool:
    # the row-capacity check (more than UCAP distinct candidates in a row): none where no row can overflow (a context
    # of at most UCAP cells), elsewhere a device flag that the caller reads once per layer; FSB_ROWS_DEFER=0 restores
    # the host check of every stage at every context length (one stream sync per stage)
    return not compiling() and os.environ.get("FSB_ROWS_DEFER", "1") == "1"


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


ROWS_TILE_MAX = 4096             # largest (query x slot) tile, BQ * pow2(Ca + Cb), of the row-tile select


def rows_tile_ok(Bq: int, C: int) -> bool:
    # the row-tile select (one 4-warp program per row) beats the per-query select on tiles of up to 4096 entries, e.g.
    # 128 queries x 32 slots (the compact 16 + 16 beams: 0.85-0.90 of its time on bench stages); larger tiles spill
    # registers (8192 entries: 1.06-3.0x slower, 16384: 2.4-6.7x, K 4-16), so FSB_ROWS_IMPL=cuda then keeps its row
    # union and LSE kernel and selects per query, which is 0.48-0.81 of the legacy path's time on every layout measured
    return max(16, _pow2(Bq)) * _pow2(C) <= ROWS_TILE_MAX


if _HAVE_TRITON:
    @triton.jit
    def _rows_union_kernel(SELA, SELB, SEL, Ca, Cb, C, CAP: tl.constexpr, CBP: tl.constexpr, HAS_B: tl.constexpr):
        # union_candidates for one query: beam-b cells that beam a also found become -1
        r = tl.program_id(0)
        ia = tl.arange(0, CAP)
        a = tl.load(SELA + r * Ca + ia, mask=ia < Ca, other=-1)
        tl.store(SEL + r * C + ia, a, mask=ia < Ca)
        if HAS_B:
            ib = tl.arange(0, CBP)
            b = tl.load(SELB + r * Cb + ib, mask=ib < Cb, other=-1)
            dup = tl.sum(((b[:, None] == a[None, :]) & (ia[None, :] < Ca)).to(tl.int32), axis=1) > 0
            tl.store(SEL + r * C + Ca + ib, tl.where(dup, -1, b), mask=ib < Cb)

    @triton.jit
    def _rows_cell_lse(qh, ql, kb, cell, j, d, S, sl, scale, DH: tl.constexpr, LSE_MODE: tl.constexpr,
                       LO: tl.constexpr):
        # the token logsumexp (or max) of the row's queries against one cell's S keys: bf16 tensor-core products of
        # the queries' high part (and, with LO, their low part) accumulated in fp32
        k = tl.load(kb + (cell * S + j)[:, None] * sl + d[None, :], mask=(j[:, None] < S) & (d[None, :] < DH),
                    other=0.0).to(tl.bfloat16)
        kt = tl.trans(k)
        if LO:
            sc = (tl.dot(qh, kt) + tl.dot(ql, kt)) * scale
        else:
            sc = tl.dot(qh, kt) * scale
        sc = tl.where(j[None, :] < S, sc, float("-inf"))
        m = tl.max(sc, axis=1)
        if LSE_MODE:
            m = m + _ld.log(tl.sum(_ld.exp(sc - m[:, None]), axis=1))
        return m

    @triton.jit
    def _rows_lse_loop(qh, ql, kb, CELLS, LSE, r, qi, j, d, U, u0, ustep, ucap, S, sl, scale, BQ: tl.constexpr,
                       DH: tl.constexpr, LSE_MODE: tl.constexpr, LO: tl.constexpr):
        for u in range(u0, U, ustep):
            cell = tl.load(CELLS + r * ucap + u).to(tl.int64)
            m = _rows_cell_lse(qh, ql, kb, cell, j, d, S, sl, scale, DH, LSE_MODE, LO)
            tl.store(LSE + (r.to(tl.int64) * ucap + u) * BQ + qi, m)

    @triton.jit
    def _rows_lse_kernel(Q, CELLS, UCNT, KEY, LSE, TIDX, Bq, ucap, Hh, sb, sh, sl, scale, S,
                         BQ: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr, SP: tl.constexpr,
                         LSE_MODE: tl.constexpr, LOSKIP: tl.constexpr = False, QBF16: tl.constexpr = False):
        # row r = (head, query block): the token logsumexp (or max) of every query of the row against every
        # distinct candidate cell of the row, keys loaded once per cell; LSE is (rows, ucap, BQ). Program (r, p) of a
        # (rows, P) grid takes the cells p, p + P, ... QBF16: the caller guarantees bf16-exact queries (no low part
        # at all); LOSKIP: each program tests its query tile instead
        r = tl.program_id(0)
        p = tl.program_id(1)
        nprog = tl.num_programs(1)
        seq = tl.load(TIDX + r)
        bi = seq // Hh
        hi = seq - bi * Hh
        qi = tl.arange(0, BQ)
        d = tl.arange(0, DHP)
        j = tl.arange(0, SP)
        qf = tl.load(Q + (r * Bq + qi[:, None]) * DH + d[None, :], mask=(qi[:, None] < Bq) & (d[None, :] < DH),
                     other=0.0)
        # fp32 query as two bf16 terms (exact for bf16 queries; about 2^-16 relative otherwise, e.g. routed queries)
        qh = qf.to(tl.bfloat16)
        if not QBF16:
            ql = (qf - qh.to(tl.float32)).to(tl.bfloat16)
        kb = KEY + bi * sb + hi * sh
        U = tl.minimum(tl.load(UCNT + r), ucap)          # an overflowing row (deferred check) stays in bounds
        if QBF16:
            _rows_lse_loop(qh, qh, kb, CELLS, LSE, r, qi, j, d, U, p, nprog, ucap, S, sl, scale, BQ, DH, LSE_MODE,
                           False)
        elif LOSKIP:
            # bf16-exact queries (the raw bf16 Q of eager prefill) have an all-zero low part, whose products are
            # exact zeros: skipping them changes no score (at most the sign of a zero score)
            if tl.max(tl.max(tl.abs(ql.to(tl.float32)), axis=1), axis=0) > 0:
                _rows_lse_loop(qh, ql, kb, CELLS, LSE, r, qi, j, d, U, p, nprog, ucap, S, sl, scale, BQ, DH, LSE_MODE,
                               True)
            else:
                _rows_lse_loop(qh, ql, kb, CELLS, LSE, r, qi, j, d, U, p, nprog, ucap, S, sl, scale, BQ, DH, LSE_MODE,
                               False)
        else:
            _rows_lse_loop(qh, ql, kb, CELLS, LSE, r, qi, j, d, U, p, nprog, ucap, S, sl, scale, BQ, DH, LSE_MODE,
                           True)

    @triton.jit
    def _rows_select_kernel(SEL, IDX, LSE, CELLA, ca0, ca1, TIDX, QPOS, sq0, sq1, LAM, ls0, SIG, ss0, OUT, PROV,
                            Bq, C, Ca, Mc, ucap, S, PROV_BIT, BQ: tl.constexpr, CP: tl.constexpr, KK: tl.constexpr,
                            KP2: tl.constexpr, POS: tl.constexpr, PRV: tl.constexpr):
        # one query: u = cell score (+ lam * a * erf(|qpos - (cell + 0.5) s| / (2 sigma)) for lse_pos), then the top
        # KK by repeated arg-max (first slot on ties, as torch's stable sort), -1 where the score is -inf
        pid = tl.program_id(0)
        r = pid // Bq
        b = pid - r * Bq
        c = tl.arange(0, CP)
        cell = tl.load(SEL + pid * C + c, mask=c < C, other=-1)
        ok = cell >= 0
        cz = tl.where(ok, cell, 0)
        pu = tl.minimum(tl.load(IDX + r * Mc + cz, mask=ok, other=0), ucap - 1)     # in bounds on overflowing rows
        u = tl.load(LSE + (r.to(tl.int64) * ucap + pu) * BQ + b, mask=ok, other=0.0)
        if POS:
            seq = tl.load(TIDX + r)
            a = tl.load(CELLA + seq * ca0 + cz * ca1, mask=ok, other=0.0)
            qp = tl.load(QPOS + r * sq0 + b * sq1)
            u = u + tl.load(LAM + r * ls0) * a * tl.math.erf(
                tl.math.div_rn(tl.abs(qp - (cz.to(tl.float32) + 0.5) * S), 2.0 * tl.load(SIG + r * ss0)))
        u = tl.where(ok, u, float("-inf"))
        k2 = tl.arange(0, KP2)
        out = tl.full((KP2,), -1, tl.int64)
        src = tl.full((KP2,), 0, tl.int32)
        for kk in tl.static_range(KK):
            mx = tl.max(u, axis=0)
            am = tl.argmax(u, axis=0, tie_break_left=True)
            hit = c == am
            cv = tl.sum(tl.where(hit, cell, 0), axis=0)
            out = tl.where(k2 == kk, tl.where(mx > float("-inf"), cv, -1), out)
            src = tl.where(k2 == kk, am, src)
            u = tl.where(hit, float("-inf"), u)
        tl.store(OUT + pid * KK + k2, out, mask=k2 < KK)
        if PRV:
            pv = tl.where(out < 0, -1, tl.where(src >= Ca, PROV_BIT, 0))
            tl.store(PROV + pid * KK + k2, pv.to(tl.int8), mask=k2 < KK)


if _HAVE_TRITON:
    @triton.jit
    def _rows_select_rows_kernel(SELA, SELB, DUP, POSM, LSE, CELLA, ca0, ca1, TIDX, QPOS, sq0, sq1, LAM, ls0, SIG, ss0,
                                 OUT, PROV, Bq, Ca, Cb, Mc, ucap, S, PROV_BIT, BQ: tl.constexpr, CP: tl.constexpr,
                                 KK: tl.constexpr, KP2: tl.constexpr, POS: tl.constexpr, PRV: tl.constexpr,
                                 HAS_B: tl.constexpr):
        # _rows_select_kernel for all queries of row r at once (query x slot tile), on the beams' own slots: slot
        # c < Ca is beam a's slot c, slot Ca + j is beam b's slot j unless the DUP bit j marks it as a repeat of one of
        # the query's beam-a cells (union_candidates' -1); positions POSM from fsb_rows_cuda's row union
        r = tl.program_id(0)
        qi = tl.arange(0, BQ)
        c = tl.arange(0, CP)
        qm = qi < Bq
        rq = r * Bq + qi
        isa = c < Ca
        cell = tl.load(SELA + rq[:, None] * Ca + c[None, :], mask=qm[:, None] & isa[None, :], other=-1)
        if HAS_B:
            jb = tl.where(isa, 0, c - Ca)
            isb = (c >= Ca) & (c < Ca + Cb)
            cb = tl.load(SELB + rq[:, None] * Cb + jb[None, :], mask=qm[:, None] & isb[None, :], other=-1)
            dup = tl.load(DUP + rq, mask=qm, other=0)
            dupb = ((dup[:, None] >> jb[None, :].to(tl.int64)) & 1) != 0
            cell = tl.where(isa[None, :], cell, tl.where(dupb, -1, cb))
        ok = cell >= 0
        cz = tl.where(ok, cell, 0)
        pu = tl.minimum(tl.load(POSM + r.to(tl.int64) * Mc + cz, mask=ok, other=0), ucap - 1)
        u = tl.load(LSE + (r.to(tl.int64) * ucap + pu) * BQ + qi[:, None], mask=ok, other=0.0)
        if POS:
            seq = tl.load(TIDX + r)
            a = tl.load(CELLA + seq * ca0 + cz * ca1, mask=ok, other=0.0)
            qp = tl.load(QPOS + r * sq0 + qi * sq1, mask=qm, other=0.0)
            u = u + tl.load(LAM + r * ls0) * a * tl.math.erf(
                tl.math.div_rn(tl.abs(qp[:, None] - (cz.to(tl.float32) + 0.5) * S), 2.0 * tl.load(SIG + r * ss0)))
        u = tl.where(ok, u, float("-inf"))
        k2 = tl.arange(0, KP2)
        out = tl.full((BQ, KP2), -1, tl.int64)
        src = tl.full((BQ, KP2), 0, tl.int32)
        for kk in tl.static_range(KK):
            mx = tl.max(u, axis=1)
            am = tl.argmax(u, axis=1, tie_break_left=True)
            hit = c[None, :] == am[:, None]
            cv = tl.sum(tl.where(hit, cell, 0), axis=1)
            out = tl.where(k2[None, :] == kk, tl.where(mx > float("-inf"), cv, -1)[:, None], out)
            src = tl.where(k2[None, :] == kk, am[:, None], src)
            u = tl.where(hit, float("-inf"), u)
        om = qm[:, None] & (k2[None, :] < KK)
        tl.store(OUT + rq[:, None] * KK + k2[None, :], out, mask=om)
        if PRV:
            pv = tl.where(out < 0, -1, tl.where(src >= Ca, PROV_BIT, 0))
            tl.store(PROV + rq[:, None] * KK + k2[None, :], pv.to(tl.int8), mask=om)


def _row_union(sel: torch.Tensor, Mc: int, ucap: int):
    # distinct valid cells of each row (T',Bq,C) -> cell list (T',ucap) int32, count (T',) int32, cell -> list
    # position (T',Mc) int32
    Tp = sel.shape[0]
    dev = sel.device
    flat = sel.reshape(Tp, -1)
    pres = torch.zeros(Tp, Mc + 1, dtype=torch.int32, device=dev)
    pres.scatter_(1, torch.where(flat >= 0, flat, Mc), 1)
    pres = pres[:, :Mc]
    pos = torch.cumsum(pres, 1, dtype=torch.int32)
    cnt = pos[:, -1].contiguous()
    pos = pos - 1
    cells = torch.full((Tp, ucap + 1), -1, dtype=torch.int32, device=dev)
    cells.scatter_(1, torch.where(pres > 0, pos, ucap).clamp_(max=ucap).long(),
                   torch.arange(Mc, dtype=torch.int32, device=dev).expand(Tp, Mc))
    return cells[:, :ucap].contiguous(), cnt, pos.contiguous()


def _row_view(x: torch.Tensor, Tp: int, Bq: int):
    v = x.reshape(x.shape[0], -1).expand(Tp, Bq)
    return v, v.stride(0), v.stride(1)


def rows_ok(q: torch.Tensor, tokK, s: int, K: int, score: str, *, allow_capture: bool = False) -> bool:
    # bf16 keys only (the cache's dtype in bf16 inference): they enter the tensor cores unrounded. Not while a CUDA
    # graph is being captured (FS-B training stages): the capacity check syncs with the host
    return (rows_on() and q.is_cuda and tokK is not None and tokK.is_cuda and tokK.dtype == torch.bfloat16
            and score in ("lse", "lse_pos", "max") and 1 <= K <= 16 and s <= 128 and q.shape[-1] <= 256
            and 1 < q.shape[1] <= 256 and (allow_capture or not torch.cuda.is_current_stream_capturing()))


@torch.no_grad()
def union_rerank_rows(sel_a: torch.Tensor, sel_b, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor,
                      sigv: torch.Tensor, cellA: torch.Tensor, t_idx: torch.Tensor, scale: float, s: int, K: int,
                      score: str, tokK: torch.Tensor, prov: bool = False, ovf_out=None, q_bf16: bool = False):
    # union_candidates + rerank_top (+ utau_prov_add) for batched prefill rows -> (cells (T',Bq,K) int64,
    # provenance (T',Bq,K) int8 or None); None when a row has more than UCAP distinct candidates. With a list ovf_out
    # (and FSB_ROWS_DEFER on) that check does not sync: a device flag (True = some row overflowed, the result is
    # invalid and the caller must redo the stage without the rows path) is appended to ovf_out instead. Rows cannot
    # overflow when the context has at most UCAP cells: then there is no check at all (FSB_ROWS_DEFER=0: the host
    # check of every stage, as before the flag). q_bf16: the caller guarantees that q holds bf16 values (Q.float() of
    # a bf16 Q), so the LSE kernel is built without the queries' low part
    from .fsb_rerank import PROV_UTAU, _key4
    Tp, Bq, Ca = sel_a.shape
    Cb = 0 if sel_b is None else sel_b.shape[-1]
    C, rows, Dh, dev = Ca + Cb, Tp * Bq, q.shape[-1], q.device
    Mc = cellA.shape[1]
    un = ovf = None
    if rows_impl() == "cuda":
        from .fsb_rows_cuda import rows_union_cuda, rows_union_ok
        if rows_union_ok(sel_a, sel_b, Mc):
            un = rows_union_cuda(sel_a, sel_b, Mc, UCAP)
    if un is not None:
        cells, cnt, pos, dup, ovf = un
    else:
        sel = torch.empty((Tp, Bq, C), dtype=torch.long, device=dev)
        _rows_union_kernel[(rows,)](sel_a.contiguous(), sel_b.contiguous() if Cb else sel_a, sel, Ca, Cb, C,
                                    CAP=_pow2(Ca), CBP=_pow2(max(Cb, 1)), HAS_B=Cb > 0)
        cells, cnt, pos = _row_union(sel, Mc, UCAP)
    capturing = torch.cuda.is_current_stream_capturing()
    if not rows_defer_on():
        if not capturing and int(cnt.max()) > UCAP:   # the host check of every stage (one stream sync)
            return None
    elif Mc > UCAP:
        if ovf is None:
            ovf = cnt.max() > UCAP
        if ovf_out is not None:
            ovf_out.append(ovf)
        elif not capturing and bool(ovf):
            return None
    K4 = _key4(tokK)
    BQ = max(16, _pow2(Bq))
    lse = torch.empty((Tp, UCAP, BQ), dtype=torch.float32, device=dev)
    _rows_lse_kernel[(Tp, rows_lse_split() if un is not None else 1)](
        q.reshape(rows, Dh).contiguous(), cells, cnt, K4, lse, t_idx.contiguous(), Bq, UCAP, K4.shape[1],
        K4.stride(0), K4.stride(1), K4.stride(2), float(scale), s, BQ=BQ, DH=Dh, DHP=max(16, _pow2(Dh)),
        SP=max(16, _pow2(s)), LSE_MODE=(score != "max"), LOSKIP=rows_loskip_on(), QBF16=q_bf16 and rows_loskip_on(),
        num_warps=4)
    qv, sq0, sq1 = _row_view(qpos, Tp, Bq)
    lv, ls0, _ = _row_view(lam, Tp, 1)
    sv, ss0, _ = _row_view(sigv, Tp, 1)
    out = torch.empty((Tp, Bq, K), dtype=torch.long, device=dev)
    pv = torch.empty((Tp, Bq, K), dtype=torch.int8, device=dev) if prov else out
    if un is not None and not rows_tile_ok(Bq, C):
        # a tile too large for one program: the per-query select on union_candidates' slots (the CUDA union's
        # positions equal _row_union's for every present cell, the only ones the select reads; same cells bitwise)
        sel = torch.empty((Tp, Bq, C), dtype=torch.long, device=dev)
        _rows_union_kernel[(rows,)](sel_a.contiguous(), sel_b.contiguous() if Cb else sel_a, sel, Ca, Cb, C,
                                    CAP=_pow2(Ca), CBP=_pow2(max(Cb, 1)), HAS_B=Cb > 0)
        un = None
    if un is not None:
        sa = sel_a.contiguous()
        _rows_select_rows_kernel[(Tp,)](sa, sel_b.contiguous() if Cb else sa, dup if Cb else cnt, pos, lse, cellA,
                                        cellA.stride(0), cellA.stride(1), t_idx.contiguous(), qv, sq0, sq1, lv, ls0,
                                        sv, ss0, out, pv, Bq, Ca, Cb, Mc, UCAP, s, PROV_UTAU, BQ=BQ, CP=_pow2(C),
                                        KK=K, KP2=_pow2(K), POS=(score == "lse_pos"), PRV=prov, HAS_B=Cb > 0,
                                        num_warps=4)
    else:
        _rows_select_kernel[(rows,)](sel, pos, lse, cellA, cellA.stride(0), cellA.stride(1), t_idx.contiguous(), qv,
                                     sq0, sq1, lv, ls0, sv, ss0, out, pv, Bq, C, Ca, Mc, UCAP, s, PROV_UTAU,
                                     BQ=BQ, CP=_pow2(C), KK=K, KP2=_pow2(K), POS=(score == "lse_pos"), PRV=prov,
                                     num_warps=1)
    return out, (pv if prov else None)

