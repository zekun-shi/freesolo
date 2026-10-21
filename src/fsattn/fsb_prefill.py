"""Eager FS-B inference prefill with batched stage plumbing (FSB_FAST_PREFILL, default 1; 0 or legacy restores the
stage loop of fsb_long_range).

fsb_long_range's eager loop rebuilds the per-query constants, the spines and both beams' widths and budgets for every
stage of fsb_prefill_blkchunk query blocks and computes training statistics that inference never reads. For the
fixed-policy forest serve config this path computes the same selection with:

- per-forward constants (block starts, query positions, stage-major (head, block) row tables), cached per shape and
  shared by every layer;
- per layer: the forests' CUDA build (FSB_BUILD_PREFILL_TRIM=1, the default: only the chunks the layer's spines can
  read, fsb_build_cuda.spine_chunks; FSB_GLUE_FORK=1: in a captured prefill, with the cell statistics on a side
  stream forked right after the layer's K projection, fork_layer_forests), one spine launch per forest over every
  (head, block) row and one launch that writes both beams' widths and budgets and each row's lambda and sigma;
- per stage (FSB_FAST_PREFILL_BLKCHUNK query blocks, default 128, fewer when the stage's scratch and the layer's
  tables would exceed FSB_FAST_PREFILL_SCRATCH_MB, default 2048, and at most fsb_prefill_blkchunk when that memory
  bound is given as a kwarg or FSB_PREFILL_BLKCHUNK): the query copy, both beams in one launch of
  the row-cooperative CUDA kernel (fsb_fast_step.forest_beam_pair; one forest_beam_fused per beam where it does not
  apply) on slices of the layer's tables, the block-shared re-rank (fsb_rows.union_rerank_rows) and one int32 store
  into the readout's cell table (FSB_PREFILL_NOCOPY=1, the default: the beams and the re-rank read the queries from Q
  in place and the re-rank writes the cell table directly, through head-major stage views). With two or more stages a
  stage's re-rank runs on a side stream under the next stage's beams (_stage_pipe, FSB_STAGE_PIPE default 1; the same
  op sequence per stage on disjoint stage slices, joined before the readout, so the outputs are unchanged;
  FSB_CHECK_COMPACT or 0 restores the serial loop). The beams write the stage loop's layout (fsb_tree.beam_delta_eff:
  2^fsb_delta cell slots per frontier node instead of 2^fsb_delta_max for the fixed policy, FSB_BEAM_COMPACT);
- the re-rank's row-capacity check of the stage loop (fsb_rows, FSB_ROWS_DEFER): none when no row can hold more
  distinct candidates than the capacity (M <= UCAP); otherwise one device flag per stage, read once per layer after
  the readout is queued. A flagged stage is redone as the stage loop's own stages that cover it (traverse_forest with
  the per-stage check, which falls back to rerank_top in exactly the overflowing ones), then the readout is redone;
- lazy statistics: admit_frac(_union) and m0_mean are computed on first access; has / cost / n_read / n_visit exist
  only with FSB_PREFILL_STATS=1;
- the cell statistics go to the KV cache (FSAttn._begin_lr's hand-off) as from the stage loop.

Every floating-point kernel is the stage loop's own, the stage partition changes no per-row result except under
overflow (redone as above), the compact layout keeps every candidate cell in the same order, and the integer tables
are the stage loop's formulas, so the selected cells, the layer output and the statistics are bit-identical to it
(tests/test_fsb_prefill.py).
"""
from __future__ import annotations

import math
import os
import struct
from collections import OrderedDict
from collections.abc import MutableMapping

import torch

from ..utils.compile_guard import compiling

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False

_CONSTS: "OrderedDict[tuple, dict]" = OrderedDict()
_CONSTS_MAX = 8
_STREAMS: dict = {}


def _stage_pipe(stages: tuple, chk: bool, dev):
    # the side stream a stage's re-rank runs on while the next stage's beams run on the caller's: the same op
    # sequence per stage on disjoint stage slices, so the outputs are identical; needs at least two stages and no
    # compact check (FSB_STAGE_PIPE=0: the serial stage loop)
    pipe = os.environ.get("FSB_STAGE_PIPE")
    if chk or len(stages) < 2 or dev.type != "cuda" or pipe == "0":
        return None
    key = str(dev)
    s1 = _STREAMS.get(key)
    if s1 is None:
        s1 = _STREAMS[key] = torch.cuda.Stream(device=dev)
    return s1


def fast_prefill_on() -> bool:
    return os.environ.get("FSB_FAST_PREFILL", "1") not in ("0", "legacy")


def prefill_stats_on() -> bool:
    return os.environ.get("FSB_PREFILL_STATS", "0") == "1"


def fast_status() -> dict:
    """What the FS-B CUDA paths did in this process, for run metadata (bench_efficiency rows): the build failure that
    turned one off for the rest of the process (None: none; the CUDA beam then runs the Triton kernel, the row beam the
    warp kernel, the lean row beam of FSB_BEAM_V2 fsb_beam_rows, the row union the Triton rows path, the re-rank of
    FSB_ROWS_V3 fsb_rows_fused2, the small-chunk build of FSB_BUILD_SMALL the build kernel, the uniform build of
    FSB_BUILD_UNIFORM the build module's kernels), and whether the fast prefill met a forest its fused beam does not take
    (it then ran the stage loop's stages)."""
    from . import fsb_beam_cuda as bc
    from . import fsb_build_cuda as fbc
    from . import fsb_rows_cuda as rc
    return {"beam_cuda_failure": bc._failed[0] if bc._failed else None,
            "beam_rows_failure": bc._rows_failed[0] if bc._rows_failed else None,
            "beam_rows2_failure": bc._rows2_failed[0] if bc._rows2_failed else None,
            "rows_union_failure": rc._failed[0] if rc._failed else None,
            "rows_v3_failure": rc._v3_failed[0] if rc._v3_failed else None,
            "build_small_failure": next(iter(fbc._small_failed.values()), None),
            "build_uniform_failure": next(iter(fbc._uniform_failed.values()), None),
            "fast_prefill_unfused": bool(_warned)}


if _HAVE_TRITON:
    @triton.jit
    def _row_tables_kernel(H1, H2, SELS, SIGMA, HEAD, W1, B1, W2, B2, LAM, SIG, n, Bq, delta, width, uwidth,
                           HAS_U: tl.constexpr, UTAU: tl.constexpr, BLOCK: tl.constexpr):
        # (row, query) entries of the fixed policy's beam tables, with fsb_long_range's integer formulas:
        # main beam D = max(h - delta, 0), W = min(width, 2^D), budget 2^max(h - D, 0); second beam on the filtered
        # forest (UTAU): D2 = max(h2 - delta, 0), du = h2 - D2, else du = the main beam's exponent; Du = max(hu - du, 0),
        # W = min(uwidth, 2^Du), budget 2^du. The first entry of each row also stores its head's lambda and sigma.
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        r = i // Bq
        one = tl.full([BLOCK], 1, tl.int64)
        h = tl.load(H1 + r, mask=m, other=0)
        D = tl.maximum(h - delta, 0)
        dl = tl.maximum(h - D, 0)
        tl.store(W1 + i, tl.minimum(one * width, one << D), mask=m)
        tl.store(B1 + i, one << dl, mask=m)
        if HAS_U:
            if UTAU:
                hu = tl.load(H2 + r, mask=m, other=0)
                du = hu - tl.maximum(hu - delta, 0)
            else:
                hu = h
                du = dl
            Du = tl.maximum(hu - du, 0)
            tl.store(W2 + i, tl.minimum(one * uwidth, one << Du), mask=m)
            tl.store(B2 + i, one << du, mask=m)
        first = m & ((i - r * Bq) == 0)
        hd = tl.load(HEAD + r, mask=first, other=0)
        tl.store(LAM + r, tl.load(SELS + hd, mask=first, other=0.0).to(tl.float32), mask=first)
        tl.store(SIG + r, tl.load(SIGMA + hd, mask=first, other=0.0).to(tl.float32), mask=first)


class LazyStats(MutableMapping):
    """attn._fsb_stats of the fast prefill: the reference's keys, in its order, and values; the entries given as
    functions are computed on first access."""

    def __init__(self, values=None, lazy=None):
        self._v = dict(values or {})
        self._lazy = dict(lazy or {})
        self._keys = list(self._v) + [k for k in self._lazy if k not in self._v]

    def __getitem__(self, key):
        if key in self._v:
            return self._v[key]
        fn = self._lazy.pop(key)
        v = self._v[key] = fn()
        return v

    def __setitem__(self, key, value):
        self._lazy.pop(key, None)
        if key not in self._v:
            self._keys.append(key)
        self._v[key] = value

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        self._lazy.pop(key, None)
        self._v.pop(key, None)
        self._keys.remove(key)

    def __iter__(self):
        return iter(list(self._keys))

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._v or key in self._lazy

    def __repr__(self):
        return f"LazyStats(computed={sorted(self._v)}, lazy={sorted(self._lazy)})"


def prefill_fast_ok(attn, Q: torch.Tensor, tokK, Bq: int) -> bool:
    # fsb_long_range takes this path for eager inference prefill (eval mode, no autograd) with the fixed policy on the
    # forest, both beams on the fused kernel, the block-shared re-rank (bf16 keys, lse / lse_pos / max), and none of
    # the options that change the per-stage selection (hybrid pool, beam / union quota, union prior, residual
    # re-ranker, read margin, routing adapter, learned radius) or read the stage internals (tau calibration,
    # selection override); never under torch.compile (the stage loop, as before)
    if (compiling() or not (_HAVE_TRITON and fast_prefill_on()) or attn.training or torch.is_grad_enabled()
            or not Q.is_cuda):
        return False
    if getattr(attn, "fsb_tree", "rebuild") != "forest" or getattr(attn, "fsb_policy_mode", "fixed") != "fixed":
        return False
    if os.environ.get("FSB_TRAIN_GRAPH", "1") == "1" and os.environ.get("FSB_EVAL_GRAPH", "0") == "1":
        return False
    if getattr(attn, "_fsb_calib", None) is not None or getattr(attn, "_fsb_sel_override", None) is not None:
        return False
    from .fsb_fast_step import fast_beam_on
    from .fsb_route import route_on
    from .fsb_rows import rows_defer_on, rows_ok
    from .fsb_traverse_triton import _HAVE_TRITON as _trav
    from .fsb_tree import beam_kw, hybrid_on, margin_kw, rr_kw, union_prior_on, union_quota_on
    if (hybrid_on(attn) or beam_kw(attn) or union_quota_on(attn) or union_prior_on(attn) or rr_kw(attn)
            or margin_kw(attn) or route_on(attn) or getattr(attn, "fsb_route_learn_c", False)):
        return False
    read_top = int(getattr(attn, "fsb_read_top", 0) or 0)
    score = getattr(attn, "fsb_rerank_score", "leaf")
    if read_top <= 0 or score not in ("lse", "lse_pos", "max") or attn.fsb_kk != read_top:
        return False
    if int(attn.fsb_delta) < 0:
        return False                          # budgets of 2^(negative): the stage loop's torch shifts decide them
    if not (_trav and fast_beam_on() and os.environ.get("FSB_TRAVERSE_TRITON", "1") == "1"):
        return False
    R = 1 << attn.fsb_delta_max
    for wm in (attn.fsb_W_max, int(getattr(attn, "fsb_union_W_max", 0) or 0)):
        if wm and ((wm & (wm - 1)) != 0 or wm * R > 256):
            return False
    if Q.shape[-1] > 256 or Q.shape[2] < Bq:
        return False
    return rows_ok(Q[0, :, :Bq], tokK, attn.cell_stride, read_top, score, allow_capture=rows_defer_on())


def _stage_bytes(T: int, Bq: int, Dh: int, Mc: int, Ca: int, Cb: int, K: int) -> int:
    # scratch of one query block of a stage (all T rows): fp32 queries, both beams' cells, the union candidates and
    # _row_union's temporaries, the re-rank's LSE table and the outputs
    from .fsb_rows import UCAP
    C = Ca + Cb
    bq = max(16, 1 << max(Bq - 1, 0).bit_length())
    per = (Bq * Dh * 4 + Bq * C * 8 + 8 * Bq + Bq * C * 8 * 2 + Bq * C + (Mc + 1) * 4 * 4 + Mc * 8
           + (UCAP + 1) * 4 + UCAP * bq * 4 + Bq * K * 9)
    return T * per


def _layer_bytes(T: int, nb: int, Bq: int, Dh: int, Mc: int, K: int, union: bool, utau: bool, prov: bool,
                 stats: bool) -> int:
    # the tables this path keeps for a whole layer: the spines of every (head, block) row (Kmax + 1 nodes of Dh + 9
    # fp32 each, per forest), both beams' width and budget tables, the query positions, the cell table, and the
    # provenance and visit tables when they are kept
    rows = T * nb
    S = max(Mc, 1).bit_length()                                     # floor(log2 Mc) + 1 spine nodes per row
    nbytes = rows * (S * (Dh + 9) * 4 + 32) * (2 if utau else 1)
    nbytes += rows * Bq * 8 * (4 if union else 2) + rows * Bq * 4 + rows * Bq * K * 4
    if prov:
        nbytes += rows * Bq * K
    if stats:
        nbytes += rows * Bq * 4
    return nbytes


def _stage_blocks(T, nb, Bq, Dh, Mc, Ca, Cb, K, layer_bytes: int = 0, cap: int = 0) -> int:
    # FSB_FAST_PREFILL_SCRATCH_MB bounds the layer's tables plus one stage's scratch (at least one block per stage);
    # cap > 0 (the module's fsb_prefill_blkchunk when given, _blkchunk_cap) bounds the blocks per stage
    want = int(os.environ.get("FSB_FAST_PREFILL_BLKCHUNK", "128"))
    want = nb if want <= 0 else min(want, nb)
    if cap > 0:
        want = min(want, cap)
    budget = int(float(os.environ.get("FSB_FAST_PREFILL_SCRATCH_MB", "2048")) * (1 << 20)) - int(layer_bytes)
    return max(1, min(want, budget // max(1, _stage_bytes(T, Bq, Dh, Mc, Ca, Cb, K))))


def _stages(attn, T: int, nb: int, Bq: int, Dh: int, M: int, d_eff, utau: bool) -> tuple:
    # the fast prefill's stages ((b0, b1) block spans) for a layer of these sizes
    Kk = attn.fsb_kk
    W_max, Wu_max = attn.fsb_W_max, int(getattr(attn, "fsb_union_W_max", 0) or 0)
    R = 1 << (attn.fsb_delta_max if d_eff is None else d_eff)
    union = int(getattr(attn, "fsb_union_width", 0) or 0) > 0
    acc = getattr(attn, "_fsb_read_acc", None)
    fixed = _layer_bytes(T, nb, Bq, Dh, M, Kk, union, utau, acc is not None and utau, prefill_stats_on())
    nbch = _stage_blocks(T, nb, Bq, Dh, M, W_max * R, Wu_max * R if union else 0, Kk, fixed, _blkchunk_cap(attn))
    return tuple((b0, min(b0 + nbch, nb)) for b0 in range(0, nb, nbch))


def _blkchunk_cap(attn) -> int:
    # fsb_prefill_blkchunk (eval_ruler / train_dist --fsb_prefill_blkchunk, FSB_PREFILL_BLKCHUNK: "query blocks per
    # prefill chunk, memory bound") caps the stage size when it was given; 0: no cap (FSB_FAST_PREFILL_BLKCHUNK)
    n = int(getattr(attn, "fsb_prefill_blkchunk", 0) or 0)
    return n if n > 0 and getattr(attn, "fsb_prefill_blkchunk_set", False) else 0


def _consts(dev, B: int, H: int, nb: int, Bq: int, stages: tuple) -> dict:
    # per-forward constants shared by every layer: block starts (fsb_long_range's), and for the stage-major rows
    # (stage (b0, b1): row T*b0 + t*nbc + j is head row t of block b0 + j, the order of fsb_long_range's stages)
    # the head row, head, block, index into m_b (T, nb) and the query positions
    key = (str(dev), B, H, nb, Bq, stages)
    hit = _CONSTS.get(key)
    if hit is not None:
        _CONSTS.move_to_end(key)
        return hit
    T = B * H
    blk_start = torch.arange(nb, device=dev, dtype=torch.float32) * Bq
    t_l, b_l = [], []
    for b0, b1 in stages:
        nbc = b1 - b0
        t_l.append(torch.arange(T, device=dev).repeat_interleave(nbc))
        b_l.append(torch.arange(b0, b1, device=dev).repeat(T))
    t_rows = torch.cat(t_l)
    blk_rows = torch.cat(b_l)
    qpos = blk_start[blk_rows][:, None] + torch.arange(Bq, device=dev, dtype=torch.float32)[None, :]
    hit = {"blk_start": blk_start, "t_rows": t_rows, "blk_rows": blk_rows, "head_rows": t_rows % H,
           "mb_index": t_rows * nb + blk_rows, "qpos": qpos.contiguous(), "blk_rows_cpu": None}
    _CONSTS[key] = hit
    while len(_CONSTS) > _CONSTS_MAX:
        _CONSTS.popitem(last=False)
    return hit


def _f32(x: float) -> float:
    # x rounded to fp32 (to nearest even)
    return struct.unpack("f", struct.pack("f", x))[0]


def _mb_max(nb: int, Bq: int, s: int, M: int) -> int:
    # the largest per-block prefix count m_b the forest build can return: m_b <= min(E[b], M) with E[b] =
    # floor(blk_start[b] / s) nondecreasing in b, so E[nb - 1] bounds every block of both forests. Computed on the host
    # without a tensor (no device or host sync inside a captured prefill) with the device's fp32 roundings: _consts'
    # blk_start[b] = fp32(b) * fp32(Bq), and _forest_build_stage's division of a CUDA tensor by a host scalar, which
    # torch runs as a multiply by the scalar's fp32 reciprocal. A double product or quotient of fp32 values rounded
    # once to fp32 is the fp32 operation's result (doubles carry more than 2 * 24 + 2 bits)
    blk = _f32(_f32(float(nb - 1)) * _f32(float(Bq)))
    return min(math.floor(_f32(blk * _f32(1.0 / _f32(float(s))))), M)


def _row_tables(attn, SP: dict, SP2, sel_scale, sigma, head_rows, Bq: int, union: bool):
    rows = SP["height"].shape[0]
    dev = SP["height"].device
    n = rows * Bq
    i64 = torch.int64
    W = torch.empty((rows, Bq), dtype=i64, device=dev)
    bud = torch.empty((rows, Bq), dtype=i64, device=dev)
    Wu = torch.empty((rows, Bq), dtype=i64, device=dev) if union else W
    budu = torch.empty((rows, Bq), dtype=i64, device=dev) if union else bud
    lam = torch.empty((rows,), dtype=torch.float32, device=dev)
    sig = torch.empty((rows,), dtype=torch.float32, device=dev)
    if rows:
        BLOCK = 1024
        h1 = SP["height"].contiguous()
        h2 = SP2["height"].contiguous() if SP2 is not None else h1
        # the kernel indexes every table with unit stride: e.g. a per-head sigma sliced out of a wider tensor
        sel_scale, sigma, head_rows = (t.reshape(-1).contiguous() for t in (sel_scale, sigma, head_rows))
        _row_tables_kernel[(triton.cdiv(n, BLOCK),)](
            h1, h2, sel_scale, sigma, head_rows, W, bud, Wu, budu, lam, sig, n, Bq,
            int(attn.fsb_delta), int(attn.fsb_width), int(getattr(attn, "fsb_union_width", 0) or 0),
            HAS_U=union, UTAU=SP2 is not None, BLOCK=BLOCK, num_warps=4)
    return W, bud, Wu, budu, lam, sig


def _sp_rows(SP: dict, r0: int, r1: int) -> dict:
    return {"node": SP["node"][r0:r1], "root": SP["root"][r0:r1], "height": SP["height"][r0:r1],
            "m_b": SP["m_b"][r0:r1], "t_idx": SP["t_idx"][r0:r1]}


def _reference_stage(attn, Q, F: dict, m_b_all, sigma, N: int, b0: int, b1: int, Bq: int, blk_start, tokK,
                     rows: bool):
    # one stage of fsb_long_range's eager loop (forest, fixed policy, the options prefill_fast_ok admits), verbatim:
    # used where this path's own stage does not apply. rows=True with the per-stage capacity check, as the loop
    # redoes a layer after an overflow: the rows re-rank, or rerank_top where a row of the stage overflows
    from .fsb_route import radius_rows
    from .fsb_tree import (_qf_rows, beam_delta_eff, build_spines, choose_actions, traverse_forest, union_kw,
                           union_tau_spines_kw)
    B, H, _, Dh = Q.shape
    dev = Q.device
    T = B * H
    nbc = b1 - b0
    Tc = T * nbc
    s = attn.cell_stride
    with torch.no_grad():
        if Q.dtype == torch.bfloat16:
            q0, q1 = b0 * Bq, b1 * Bq
            nq = min(q1, N) - q0
            q = torch.empty((Tc, Bq, Dh), dtype=torch.bfloat16, device=dev)
            qv = q.view(B, H, nbc * Bq, Dh)
            qv[:, :, :nq].copy_(Q[:, :, q0:q0 + nq])
            if nq < nbc * Bq:
                qv[:, :, nq:].zero_()
        else:
            qf_c = _qf_rows(Q, b0 * Bq, b1 * Bq, N)
            q = qf_c.reshape(B, H, nbc, Bq, Dh).reshape(Tc, Bq, Dh)
        qpos = (blk_start[b0:b1, None] + torch.arange(Bq, device=dev, dtype=torch.float32)[None, :])
        qpos = qpos[None, None].expand(B, H, nbc, Bq).reshape(Tc, Bq)
        head_id = torch.arange(H, device=dev)[None, :, None].expand(B, H, nbc).reshape(Tc)
        lam = attn.sel_scale.float()[head_id].view(Tc, 1, 1)
        sigv = sigma.float()[head_id].view(Tc, 1, 1)
        t_idx = torch.arange(T, device=dev).repeat_interleave(nbc)
        m_b = m_b_all[:, b0:b1].reshape(Tc)
        SP = build_spines(F, m_b, t_idx)
    W, D, _ = choose_actions(attn, q, head_id, m_b, SP["height"], 0, training=False)
    with torch.no_grad():
        rc = radius_rows(attn, head_id, None)
        delta = (SP["height"][:, None] - D).clamp(min=0)
        ukw = union_tau_spines_kw(attn, F, F["utau_mba"][:, b0:b1].reshape(Tc) if "utau_mba" in F else None, t_idx,
                                  Bq)
        pl = [] if ukw else None
        sel, n_visit, n_read = traverse_forest(F, SP, q, qpos, lam, sigv, attn.scale, s, W, delta,
                                               attn.fsb_W_max, attn.fsb_delta_max, attn.fsb_node_score,
                                               radius_coef=rc, read_top=attn.fsb_read_top,
                                               rerank_score=attn.fsb_rerank_score, tokK=tokK, rq=None,
                                               **union_kw(attn), **ukw, rows=rows, delta_eff=beam_delta_eff(attn),
                                               rows_q_bf16=Q.dtype == torch.bfloat16,
                                               **({"prov_out": pl} if pl is not None else {}))
    return sel, n_visit, n_read, (pl[0] if pl else None)


_warned: list = []


def fsb_prefill_fast(attn, Q, K, V, Kp, Vp, sigma, N: int, tokK):
    # fsb_long_range for the configurations of prefill_fast_ok; same output, selection and statistics
    from .fsb_beam_cuda import spines_tables_cuda
    from .fsb_fast_step import fast_beam_ok, forest_beam_fused, forest_beam_pair
    from .fsb_rows import UCAP, union_rerank_rows
    from .fsb_tree import (_check_compact, _forest_build_stage, beam_delta_eff, build_spines, fsb_read_acc,
                           fsb_utau_acc, masked_readout, readout_handoff, union_tau_on)
    B, H, _, Dh = Q.shape
    s = attn.cell_stride
    dev = Q.device
    M = (N + s - 1) // s
    pad = M * s - N
    Bq = min(attn._cell_block, N)
    nb = (N + Bq - 1) // Bq
    Npad = nb * Bq
    T = B * H
    Kk = attn.fsb_kk
    score = attn.fsb_rerank_score
    W_max, Wu_max = attn.fsb_W_max, int(getattr(attn, "fsb_union_W_max", 0) or 0)
    d_eff = beam_delta_eff(attn)
    dm = attn.fsb_delta_max if d_eff is None else d_eff        # the stage loop's beam layout (traverse_forest's dm_beam)
    union = int(getattr(attn, "fsb_union_width", 0) or 0) > 0
    utau = union_tau_on(attn)
    R = 1 << dm
    chk = dm != attn.fsb_delta_max and os.environ.get("FSB_CHECK_COMPACT", "0") == "1"
    stats = prefill_stats_on()
    acc = getattr(attn, "_fsb_read_acc", None)
    q_bf16 = Q.dtype == torch.bfloat16                          # the re-rank query is Q.float() (no routing here)
    stages = _stages(attn, T, nb, Bq, Dh, M, d_eff, utau)
    cst = _consts(dev, B, H, nb, Bq, stages)
    blk_start = cst["blk_start"]
    lb = _legacy_blocks(attn, nb)
    fk = join_layer_forests(attn)
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        if fk is not None and fk[0] is Kp and fk[1] == (N, blk_start.data_ptr()):
            Kbar, ceil_c, spread, F, m_b_all = fk[2]            # built by the fork after the K projection
        else:
            Kbar, ceil_c, spread, F, m_b_all = _layer_forests(attn, Kp, M, s, nb, Bq, cst)
        del fk
        if getattr(attn, "_cell_stats_out", None) is not None and not pad:
            attn._cell_stats_out.append((K, Kbar, ceil_c))      # FSKVCache.prefill: the same ops on the same view
        F["chunks"] = {}
        F2 = F.get("utau_F") if utau else None
        m_b2_all = F.get("utau_mba") if F2 is not None else None
        # one spine launch per forest for every (head, block) row, stage-major (one stage: m_b's own row order)
        rows_of = (lambda m: m.reshape(-1)) if len(stages) == 1 else (lambda m: m.reshape(-1)[cst["mb_index"]])
        mb_rows = rows_of(m_b_all)
        mb2_rows = rows_of(m_b2_all) if F2 is not None else None
        Fu = F2 if F2 is not None else F
        qprobe = torch.empty((0, Bq, Dh), dtype=torch.float32, device=dev)
        sp_probe = {"root": mb_rows[:0], "t_idx": cst["t_rows"][:0], "node": qprobe}
        fused = fast_beam_ok(F, sp_probe, qprobe, W_max, dm) and (
            not union or fast_beam_ok(Fu, sp_probe, qprobe, Wu_max, dm))
        del qprobe, sp_probe
        sel32 = torch.empty((B, H, Npad, Kk), dtype=torch.int32, device=dev)
        prov_full = torch.empty((B, H, Npad, Kk), dtype=torch.int8, device=dev) if (
            acc is not None and F2 is not None) else None
        nv_full = torch.empty((B, H, Npad), dtype=torch.float32, device=dev) if stats else None
        flags, flagged, redo = [], [], []
        q = pair = sel_a = sel_b = nv = nv_u = res = out = pv = None
        if fused:
            rc = attn.fsb_bound_radius_coef
            # FSB_PREFILL_NOCOPY=1: the stages' beams and re-rank read their queries from Q in place and the re-rank
            # stores its cells straight into sel32 (head-major stage views, fsb_beam_cuda.stage_view), for Q's
            # (B, N, H, D)-transposed layout as for a contiguous one; same values as the copies (pads read zeros)
            nocopy = (os.environ.get("FSB_PREFILL_NOCOPY", "1") == "1" and not chk and union
                      and Q.dtype in (torch.bfloat16, torch.float32) and Q.stride(-1) == 1
                      and (B == 1 or Q.stride(0) == H * Q.stride(1)))

            def _beam_stage(b0: int, b1: int):
                # a stage's queries (a view of Q, or the fp32 / bf16 copy) and its beams (one pair launch when the
                # union beam takes the tables)
                nbc = b1 - b0
                Tc, r0, r1, q0, q1 = T * nbc, T * b0, T * b1, b0 * Bq, b1 * Bq
                nq = min(q1, N) - q0

                def _q_copy():
                    if q_bf16 and len(stages) == 1 and nq == nbc * Bq and Q.is_contiguous() and Q.data_ptr() % 16 == 0:
                        return Q.view(Tc, Bq, Dh)                # the whole sequence in one stage: Q's own rows
                    qc = torch.empty((Tc, Bq, Dh), dtype=torch.bfloat16 if q_bf16 else torch.float32, device=dev)
                    qv = qc.view(B, H, nbc * Bq, Dh)
                    qv[:, :, :nq].copy_(Q[:, :, q0:q0 + nq])
                    if nq < nbc * Bq:
                        qv[:, :, nq:].zero_()
                    return qc

                # nocopy: head t's stage queries as one view row, its block j being row t * nbc + j (the pair kernel
                # reads them in place; should it not run, the stage takes the copy below)
                q = (Q.as_strided((T, nq, Dh), (Q.stride(1), Q.stride(2), 1), Q.storage_offset() + q0 * Q.stride(2))
                     if nocopy else _q_copy())
                # FSB_BEAM_V2: both forests' spines and the row tables from one launch (spines_tables_cuda, bitwise)
                got = spines_tables_cuda(F, mb_rows[r0:r1], F2, mb2_rows[r0:r1] if F2 is not None else None,
                                         cst["t_rows"][r0:r1], cst["head_rows"][r0:r1], attn.sel_scale, sigma, Bq,
                                         int(attn.fsb_delta), int(attn.fsb_width),
                                         int(getattr(attn, "fsb_union_width", 0) or 0), union)
                if got is not None:
                    SPc, SP2c, Wt, bt, Wut, but, lam_r, sig_r = got
                else:
                    SPc = build_spines(F, mb_rows[r0:r1], cst["t_rows"][r0:r1])
                    SP2c = build_spines(F2, mb2_rows[r0:r1], cst["t_rows"][r0:r1]) if F2 is not None else None
                    Wt, bt, Wut, but, lam_r, sig_r = _row_tables(attn, SPc, SP2c, attn.sel_scale, sigma,
                                                                 cst["head_rows"][r0:r1], Bq, union)
                SPuc = SP2c if SP2c is not None else SPc
                qpos = cst["qpos"][r0:r1]
                lam = lam_r.view(Tc, 1, 1)
                sigv = sig_r.view(Tc, 1, 1)
                pair = sel_b = nv_u = None
                if union:
                    # both beams in one launch (the stage loop's traverse_forest makes the same call)
                    pair = forest_beam_pair(F, SPc, Wt, None, W_max, Fu, SPuc, Wut, None, Wu_max, q,
                                            qpos, lam, sigv, attn.scale, s, dm, attn.fsb_node_score, rc,
                                            bud=bt, bud_u=but)
                    if pair is None and nocopy:
                        q = _q_copy()
                        pair = forest_beam_pair(F, SPc, Wt, None, W_max, Fu, SPuc, Wut, None, Wu_max, q,
                                                qpos, lam, sigv, attn.scale, s, dm, attn.fsb_node_score, rc,
                                                bud=bt, bud_u=but)
                if pair is not None:
                    (sel_a, nv), (sel_b, nv_u) = pair
                else:
                    sel_a, nv = forest_beam_fused(F, SPc, q, qpos, lam, sigv, attn.scale, s, Wt, None, W_max,
                                                  dm, attn.fsb_node_score, rc, bud=bt)
                    sel_b = None
                    if union:
                        sel_b, nv_u = forest_beam_fused(Fu, SPuc, q, qpos, lam, sigv, attn.scale, s, Wut, None,
                                                        Wu_max, dm, attn.fsb_node_score, rc, bud=but)
                if chk:
                    # FSB_CHECK_COMPACT=1: the full layout's beams, against which the compact output must check
                    full = forest_beam_fused(F, SPc, q, qpos, lam, sigv, attn.scale, s, Wt, None, W_max,
                                             attn.fsb_delta_max, attn.fsb_node_score, rc, bud=bt)
                    _check_compact(sel_a, nv, full, W_max, attn.fsb_delta_max, dm)
                    if union:
                        full = forest_beam_fused(Fu, SPuc, q, qpos, lam, sigv, attn.scale, s, Wut, None,
                                                 Wu_max, attn.fsb_delta_max, attn.fsb_node_score, rc, bud=but)
                        _check_compact(sel_b, nv_u, full, Wu_max, attn.fsb_delta_max, dm)
                    del full
                t_idx_c = SPc["t_idx"]
                del SPc, SP2c, SPuc, Wt, bt, Wut, but
                return {"t_idx": t_idx_c}, qpos, lam, sigv, q, nq, sel_a, sel_b, nv, nv_u

            def _rerank_stage(b0: int, b1: int, SPc: dict, qpos, lam, sigv, q, nq, sel_a, sel_b, nv, nv_u):
                # a stage's block-shared re-rank (fsb_rows.union_rerank_rows) and the int32 stores into the layer's
                # tables; appends the stage to redo and returns None when a row overflows without the deferred flag
                if stats and union:
                    nv = nv + nv_u
                nbc = b1 - b0
                q0, q1 = b0 * Bq, b1 * Bq
                Tc = T * nbc
                s_sl = sel32[:, :, q0:q1]
                p_sl = prov_full[:, :, q0:q1] if prov_full is not None else None
                direct = s_sl.is_contiguous()
                if direct:
                    out_buf = s_sl.view(Tc, Bq, Kk)
                    pv_buf = p_sl.view(Tc, Bq, Kk) if p_sl is not None else None
                elif nocopy:                                     # head-major stage views of the layer's tables
                    out_buf = s_sl.view(T, nbc * Bq, Kk)
                    pv_buf = p_sl.view(T, nbc * Bq, Kk) if p_sl is not None else None
                else:
                    out_buf = pv_buf = None
                nf = len(flags)
                res = union_rerank_rows(sel_a, sel_b, q, qpos, lam, sigv, F["cellA"], SPc["t_idx"], attn.scale, s,
                                        Kk, score, tokK, prov=prov_full is not None, ovf_out=flags, q_bf16=q_bf16,
                                        out=out_buf, pv_out=pv_buf, q_nq=nq)
                if len(flags) > nf:
                    flagged.append((b0, b1))                    # its device flag: read after the readout is queued
                if res is None:
                    redo.append((b0, b1))                       # an overflowing row (FSB_ROWS_DEFER=0 host check)
                    return None
                out, pv = res
                if out_buf is None:
                    s_sl.copy_(out.view(B, H, nbc * Bq, Kk))
                    if p_sl is not None:
                        p_sl.copy_(pv.view(B, H, nbc * Bq, Kk))
                if stats:
                    nv_full[:, :, q0:q1].copy_(nv.view(B, H, nbc * Bq))
                return res

            from . import fsb_rows as _rows_mod
            _prev_scr = _rows_mod._active_scratch
            _rows_mod._active_scratch = {} if len(stages) > 1 else None
            try:
                s1 = _stage_pipe(stages, chk, dev)
                if s1 is None:
                    for b0, b1 in stages:
                        out = _rerank_stage(b0, b1, *_beam_stage(b0, b1))
                        out = pv = None
                else:
                    s0 = torch.cuda.current_stream(dev)
                    ev_b, ev_r, ring = [torch.cuda.Event(), torch.cuda.Event()], [torch.cuda.Event() for _ in stages], \
                        [None, None]

                    def _beam_k(k: int):
                        # a stage's beams, enqueued one stage ahead of its re-rank so the re-rank's capacity check
                        # (FSB_ROWS_DEFER=0 syncs the host) does not leave the caller's stream with nothing to run:
                        # the slot's previous stage's re-rank must be done reading its buffers before this stage's
                        # writes reuse them (ev_r, and the ring holds them alive)
                        if k >= 2:
                            s0.wait_event(ev_r[k - 2])
                        ring[k % 2] = None
                        got = _beam_stage(*stages[k])
                        ring[k % 2] = got
                        ev_b[k % 2].record(s0)
                        return got

                    _beam_k(0)
                    for k, (b0, b1) in enumerate(stages):
                        if k + 1 < len(stages):
                            _beam_k(k + 1)
                        with torch.cuda.stream(s1):
                            s1.wait_event(ev_b[k % 2])
                            out = _rerank_stage(b0, b1, *ring[k % 2])
                            pv = out[1] if out is not None else None
                            ring[k % 2] = (ring[k % 2], out)
                            ev_r[k].record(s1)
                    s0.wait_event(ev_r[-1])
            finally:
                _rows_mod._active_scratch = _prev_scr
        else:
            if not _warned:
                _warned.append(1)
                import warnings
                warnings.warn("FS-B fast prefill: the fused beam does not take this forest's tables, running the "
                              "stage loop's stages (FSB_FAST_PREFILL=0 skips the unused spine work)")
            redo = [(0, nb)]
    del mb_rows, mb2_rows, q, pair, sel_a, sel_b, nv, nv_u, res, out, pv
    if redo:
        # (under the caller's autocast, as fsb_long_range runs its stages)
        _redo_stages(attn, Q, F, m_b_all, sigma, N, Bq, blk_start, tokK, lb, nb, redo, sel32, prov_full, nv_full)
    F = F2 = Fu = None
    pend = None
    if flags and not torch.cuda.is_current_stream_capturing():
        # device -> pinned host copy of the stage flags; the host waits for it only after the readout is queued
        fl = torch.stack(flags)
        pend = (torch.empty(fl.shape, dtype=torch.bool, pin_memory=True), torch.cuda.Event())
        pend[0].copy_(fl, non_blocking=True)
        pend[1].record(torch.cuda.current_stream(fl.device))
    fused_attn = bool(attn._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0))

    def readout():
        if fused_attn:
            return attn._cell_fused_lr(Q, K, V, sel32, s, Bq)
        return masked_readout(attn, Q, Kp, Vp, sel32, s, Bq, N)

    lr0 = readout_handoff(attn) if pend is not None else None
    y = readout()
    if pend is not None:
        pend[1].synchronize()
        bad = [st for st, f in zip(flagged, pend[0].tolist()) if f]
        if bad:
            # a row with more than UCAP distinct candidates (never seen on real prompts): the stage loop's stages that
            # cover the flagged ones, with the per-stage check, then the readout again with its hand-off restored
            with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
                F, _ = _forest_build_stage(attn, Kbar.reshape(T, M, Dh), ceil_c.reshape(T, M), blk_start, s,
                                           spread.reshape(T, M) if spread is not None else None)
            _redo_stages(attn, Q, F, m_b_all, sigma, N, Bq, blk_start, tokK, lb, nb, bad, sel32, prov_full, nv_full)
            F = None
            readout_handoff(attn, lr0)
            y = readout()
    del Kbar, ceil_c, spread
    attn._fsb_ppo_rec = []
    n_read = None
    if acc is not None or stats:
        n_read = (sel32 >= 0).sum(-1)
        if acc is not None:
            fsb_read_acc(attn, n_read[:, :, :N])
            if prov_full is not None:
                fsb_utau_acc(attn, prov_full[:, :, :N])
    attn._fsb_stats = _stats(attn, B, H, N, M, nb, Npad, Bq, s, dev, m_b_all, m_b2_all, blk_start,
                             (n_read, nv_full) if stats else None)
    if getattr(attn, "_fsb_rd", None) is not None:
        attn._fsb_rd = None
    return y


def _layer_forests(attn, Kp, M: int, s: int, nb: int, Bq: int, cst: dict):
    # a layer's cell statistics with fsb_long_range's own ops (the forests' inputs, also handed to the KV cache) and
    # its forests: (Kbar, ceil_c, spread, F, m_b_all)
    from .fsb_build_cuda import prefill_trim_on
    from .fsb_tree import _det_sum, _forest_build_stage
    from .fsl_select_cuda import cache_cell_stats
    B, H, _, Dh = Kp.shape
    T = B * H
    st_c = cache_cell_stats(Kp, M, s) if not attn.fsb_leaf_radius else None
    if st_c is not None:
        Kbar, ceil_c = st_c
    else:
        Kpv = Kp.view(B, H, M, s, Dh)
        Kbar = Kpv.mean(dim=3).float()
        ceil_c = Kpv.float().norm(dim=-1).amax(dim=-1)
    spread = None
    if attn.fsb_leaf_radius:
        dK = Kpv.float() - Kbar[..., None, :]
        spread = _det_sum(dK * dK, -1).amax(dim=-1).sqrt()
        del dK
    # FSB_BUILD_PREFILL_TRIM=1: every read of these forests below goes through the spines of the layer's m_b
    # (the beams descend from their chunk roots), so the CUDA build may skip the chunks no such spine reads
    F, m_b_all = _forest_build_stage(attn, Kbar.reshape(T, M, Dh), ceil_c.reshape(T, M), cst["blk_start"], s,
                                     spread.reshape(T, M) if spread is not None else None,
                                     mb_max=_mb_max(nb, Bq, s, M) if prefill_trim_on() else None)
    return Kbar, ceil_c, spread, F, m_b_all


_FORK_STREAMS: dict = {}


def glue_fork_on() -> bool:
    # FSB_GLUE_FORK=1 (the default; 0 keeps the fork off): fork_layer_forests
    return os.environ.get("FSB_GLUE_FORK", "1") == "1"


def _fork_now() -> bool:
    # the fork runs while a CUDA graph is captured, whose replays repeat it at no host cost; an eager prefill keeps
    # the build on the long-range branch: the fork's host work (its own gate, the stream switch and an event per
    # layer) would land on the short prompts, whose eager prefill is bound by the host
    return torch.cuda.is_current_stream_capturing()


def fork_layer_forests(attn, K: torch.Tensor, N: int) -> None:
    # FSB_GLUE_FORK=1: a standard FS-B prefill under CUDA-graph capture builds the layer's cell statistics and forests
    # (_layer_forests, which read only K) on a side stream forked right after the K projection, beside the V and Q
    # projections and the window branch, instead of at the head of the long-range branch; fsb_prefill_fast joins it
    # and takes them when they are its own inputs (the same K, length and constants), so the same kernels run on the
    # same inputs and give the same values. Only for the inputs the fast prefill takes (prefill_fast_ok with K for the
    # same-shaped Q) and no padded cells. The side stream allocates from its own pool after waiting for the caller's
    # stream, and the caller joins it before its first read of these tensors and keeps K until after the join
    # (join_layer_forests); the fork and the join put both streams in the captured graph. The stream is created on
    # the eager pass that precedes every capture (it compiles the kernels), outside the capture.
    join_layer_forests(attn)
    if not (glue_fork_on() and K.is_cuda and K.dim() == 4 and not torch.is_grad_enabled() and not compiling()):
        return
    dev = K.device
    side = _FORK_STREAMS.get(str(dev))
    if side is None:
        side = _FORK_STREAMS[str(dev)] = torch.cuda.Stream(device=dev)
    if not _fork_now():
        return
    from .fsb_tree import beam_delta_eff, union_tau_on
    B, H, _, Dh = K.shape
    s = attn.cell_stride
    M = (N + s - 1) // s
    Bq = min(attn._cell_block, N)
    if M * s != N or K.shape[2] != N or not prefill_fast_ok(attn, K, K, Bq):
        return
    nb = (N + Bq - 1) // Bq
    T = B * H
    stages = _stages(attn, T, nb, Bq, Dh, M, beam_delta_eff(attn), union_tau_on(attn))
    cst = _consts(dev, B, H, nb, Bq, stages)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side), torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        res = _layer_forests(attn, K, M, s, nb, Bq, cst)
    ev = torch.cuda.Event()
    ev.record(side)
    attn._fsb_fork = (K, (N, cst["blk_start"].data_ptr()), res, ev)


def join_layer_forests(attn):
    # the caller's stream waits for a pending fork_layer_forests and drops it -> (K, key, forests, event) or None
    fk = getattr(attn, "_fsb_fork", None)
    if fk is not None:
        attn._fsb_fork = None
        torch.cuda.current_stream(fk[0].device).wait_event(fk[3])
    return fk


def _redo_stages(attn, Q, F, m_b_all, sigma, N, Bq, blk_start, tokK, lb, nb, spans, sel32, prov_full, nv_full):
    # the stage loop's own stages of lb blocks that cover the block spans [b0, b1) (whole stages, so every redone
    # block gets exactly the stage loop's result), written into the layer's tables
    B, H = Q.shape[0], Q.shape[1]
    Kk = sel32.shape[-1]
    starts = sorted({r for b0, b1 in spans for r in range((b0 // lb) * lb, b1, lb)})
    for r0 in starts:
        r1 = min(r0 + lb, nb)
        q0, q1, nbc = r0 * Bq, r1 * Bq, r1 - r0
        sel, n_visit, _, pv = _reference_stage(attn, Q, F, m_b_all, sigma, N, r0, r1, Bq, blk_start, tokK, True)
        sel32[:, :, q0:q1].copy_(sel.view(B, H, nbc * Bq, Kk))
        if prov_full is not None and pv is not None:
            prov_full[:, :, q0:q1].copy_(pv.view(B, H, nbc * Bq, Kk))
        if nv_full is not None:
            nv_full[:, :, q0:q1].copy_(n_visit.view(B, H, nbc * Bq))


def _legacy_blocks(attn, nb: int) -> int:
    nbch = attn.fsb_prefill_blkchunk if attn.fsb_prefill_blkchunk > 0 else nb
    return min(nbch, nb)


def _stats(attn, B, H, N, M, nb, Npad, Bq, s, dev, m_b_all, m_b2_all, blk_start, eager):
    # fsb_long_range's statistics dict. n_adm, the admitted cells before each block's first query, is the integer
    # m_b of the forest build (both count the admitted cell ids below floor(block start / s)); admit_frac(_union)
    # and m0_mean are computed from it on first access with the reference's ops
    from .fsb_tree import _admit_frac
    n_adm = m_b_all.reshape(B, H, nb)
    n_adm_u = m_b2_all.reshape(B, H, nb) if m_b2_all is not None else None

    def eligible():
        cell_end = (torch.arange(M, device=dev, dtype=torch.float32) + 1.0) * s
        return cell_end[None, :] <= blk_start[:, None]

    lazy = {"admit_frac": lambda: _admit_frac(n_adm, eligible()), "m0_mean": lambda: n_adm.float().mean()}
    if n_adm_u is not None:
        lazy["admit_frac_union"] = lambda: _admit_frac(n_adm_u, eligible())
    values = {}
    if eager is not None:
        n_read, n_visit = eager
        real = (torch.arange(Npad, device=dev) < N).view(1, 1, Npad)
        has = (n_adm > 0)[..., None].expand(B, H, nb, Bq).reshape(B, H, Npad) & real
        cost = (n_read.float() * s + attn.fsb_route_cost * n_visit) / float(attn.cell_topk * s)
        values = {"has": has[:, :, :N], "cost": cost[:, :, :N], "n_read": n_read[:, :, :N].float(),
                  "n_visit": n_visit[:, :, :N]}
    order = ["has", "cost", "n_read", "n_visit", "admit_frac", "m0_mean", "admit_frac_union"]
    st = LazyStats({k: values[k] for k in order if k in values}, {k: lazy[k] for k in order if k in lazy})
    return st
