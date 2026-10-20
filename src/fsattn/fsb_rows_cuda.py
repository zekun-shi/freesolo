"""CUDA row union of the block-shared FS-B prefill re-rank (fsb_rows, FSB_ROWS_IMPL=cuda: the default).

One CTA per row (head, query block) replaces fsb_rows' per-query union kernel, the torch glue that turns its output
into the row's distinct-cell list and the cell -> list position map, and the host check of the list's capacity: the
row's beam slots are read once, every valid cell sets its bit in a shared-memory bitmap (the beam-b slots that repeat
a beam-a cell of the same query are flagged in a per-query bitmask instead, union_candidates' rule), and an exclusive
scan of the bitmap words' popcounts gives each present cell its rank in the ascending list. Integer logic only, so
the list, its count and the positions are the torch glue's, exactly. A row with more than `ud` distinct cells raises
a device flag that the caller reads when it chooses (fsb_rows defers it to once per layer). Compiled with NVRTC at
first use and launched through fsb_beam_cuda's driver-API helpers; without NVRTC fsb_rows keeps its Triton path.

rows_fused_cuda runs the whole re-rank of a stage in one launch: with FSB_ROWS_V3=1 (the default; contexts of up to
V3_MC_MAX cells) fsb_rows_v3, else fsb_rows_fused2. fsb_rows_v3 computes every (query, cell) score with fused2's
arithmetic in fused2's order, so both give the same cells and provenance bit for bit, at about 0.68 of fused2's time on
A100 prefill stages.
"""
from __future__ import annotations

import ctypes
import os

import torch

from pathlib import Path

_SRC = Path(__file__).with_name("fsb_rows.cu").read_text(encoding="utf-8")

MC_MAX = 1 << 16                                   # cells per context the shared-memory bitmap holds (8 KB)
_NT = 128
_fns: dict = {}
_failed: list = []


def rows_cuda_on() -> bool:
    return os.environ.get("FSB_ROWS_IMPL", "cuda") != "legacy" and not _failed


def rows_union_ok(sel_a: torch.Tensor, sel_b, Mc: int) -> bool:
    Ca = sel_a.shape[-1]
    Cb = 0 if sel_b is None else sel_b.shape[-1]
    return (rows_cuda_on() and sel_a.is_cuda and sel_a.dtype == torch.int64 and Mc <= MC_MAX
            and Ca % 2 == 0 and 2 <= Ca <= 64 and Cb % 2 == 0 and Cb <= 64
            and (sel_b is None or (sel_b.dtype == torch.int64 and sel_b.device == sel_a.device))
            and (not torch.cuda.is_current_stream_capturing() or (sel_a.device.index, Ca, Cb) in _fns))


def _source(CA: int, CB: int) -> str:
    defs = dict(CA=CA, CB=CB, NT=_NT, MCW=MC_MAX // 32)
    return "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _SRC


@torch.no_grad()
def rows_union_cuda(sel_a: torch.Tensor, sel_b, Mc: int, ud: int, scratch: dict = None):
    # the row union of the beams' cells (T',Bq,Ca) and (T',Bq,Cb) -> (cells (T',ud) int32 ascending, count (T',) int32,
    # position of each present cell (T',Mc) int32 (other entries unset), duplicate beam-b slots (T',Bq) int64 bitmask
    # or None, overflow flag () bool: some row has more than ud cells); None when NVRTC is unavailable
    from .fsb_beam_cuda import launch, load_function
    Tp, Bq, Ca = sel_a.shape
    Cb = 0 if sel_b is None else sel_b.shape[-1]
    dev = sel_a.device
    cfg = (dev.index, Ca, Cb)
    fn = _fns.get(cfg)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            fn = _fns[cfg] = load_function(dev, _source(Ca, Cb), "fsb_rows_union")
        except Exception as e:                                               # no NVRTC / driver: Triton path
            if not _failed:
                _failed.append(repr(e))
                import warnings
                warnings.warn(f"FS-B CUDA row union unavailable, using the Triton rows path: {e!r}")
            return None
    sa = sel_a.contiguous()
    sb = sel_b.contiguous() if Cb else sa
    if sa.data_ptr() % 16 or sb.data_ptr() % 16:
        sa, sb = sa.clone(), (sb.clone() if Cb else sa)
    if scratch is not None and Tp > 0:
        u_key = ("union", Tp, ud, Mc, Bq, bool(Cb))
        u_bufs = scratch.get(u_key)
        if u_bufs is None:
            cells = torch.empty((Tp, ud), dtype=torch.int32, device=dev)
            cnt = torch.empty((Tp,), dtype=torch.int32, device=dev)
            pos = torch.empty((Tp, Mc), dtype=torch.int32, device=dev)
            dup = torch.empty((Tp, Bq), dtype=torch.int64, device=dev) if Cb else cnt
            scratch[u_key] = (cells, cnt, pos, dup)
        else:
            cells, cnt, pos, dup = u_bufs
    else:
        cells = torch.empty((Tp, ud), dtype=torch.int32, device=dev)
        cnt = torch.empty((Tp,), dtype=torch.int32, device=dev)
        pos = torch.empty((Tp, Mc), dtype=torch.int32, device=dev)
        dup = torch.empty((Tp, Bq), dtype=torch.int64, device=dev) if Cb else cnt
    # rows cannot exceed ud cells when the context has at most ud: the kernel then never writes the flag
    ovf = (torch.zeros if Mc > ud else torch.empty)((), dtype=torch.bool, device=dev)
    if Tp == 0:
        return cells, cnt, pos, (dup if Cb else None), ovf
    P, I32 = ctypes.c_void_p, ctypes.c_int
    args = [P(sa.data_ptr()), P(sb.data_ptr()), I32(Bq), I32(Mc), I32(ud), P(cells.data_ptr()), P(cnt.data_ptr()),
            P(pos.data_ptr()), P(dup.data_ptr()), P(ovf.data_ptr())]
    launch(fn, Tp, _NT, dev, args)
    return cells, cnt, pos, (dup if Cb else None), ovf


_fused_fns: dict = {}
_fused_failed: list = []


def _fused_source(CA: int, CB: int, KK: int, lse_mode: bool, pos_mode: bool, prov_on: bool, prov_bit: int,
                  out_i32: bool) -> str:
    defs = dict(CA=CA, CB=CB, NT=_NT, MCW=MC_MAX // 32, KK=KK, LSE_MODE=int(lse_mode), POS_MODE=int(pos_mode),
                PROV_ON=int(prov_on), PROV_BIT=int(prov_bit), OUT_I32=int(out_i32), UTILE=32, MIN_BLOCKS=3)
    return "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _SRC


def rows_fused_ok(sel_a: torch.Tensor, sel_b, q: torch.Tensor, K4: torch.Tensor, Mc: int, s: int, K: int,
                  score: str, prov: bool, prov_bit: int, out: torch.Tensor) -> bool:
    if not (rows_cuda_on() and not _fused_failed and os.environ.get("FSB_ROWS_FUSED", "1") == "1"):
        return False
    Tp, Bq, Ca = sel_a.shape
    Cb = 0 if sel_b is None else sel_b.shape[-1]
    if not (Bq == 128 and q.shape[-1] == 64 and s == 32 and 1 <= K <= 16 and Mc <= MC_MAX
            and Ca % 2 == 0 and 2 <= Ca <= 32 and Cb % 2 == 0 and 0 <= Cb <= 32 and Ca + Cb <= 32
            and score in ("lse_pos", "lse", "max") and out.dtype in (torch.int64, torch.int32)
            and q.is_cuda and q.dtype == torch.bfloat16 and K4.is_cuda and K4.dtype == torch.bfloat16
            and K4.shape[-1] == 64 and K4.shape[1] >= 8 and K4.stride(-1) == 1
            and sel_a.is_cuda and sel_a.dtype == torch.int64
            and (sel_b is None or (sel_b.dtype == torch.int64 and sel_b.device == sel_a.device))):
        return False
    cfg = (sel_a.device.index, Ca, Cb, K, score != "max", score == "lse_pos", bool(prov), int(prov_bit),
           out.dtype == torch.int32)
    return (not torch.cuda.is_current_stream_capturing()
            or (cfg in _v3_fns if _v3_cfg_ok(Mc, Ca, Cb) else cfg in _fused_fns))


_v3_fns: dict = {}
_v3_failed: list = []
V3_MC_MAX = 1 << 14                                # cells per context fsb_rows_v3's presence bytes and rank map hold
# fsb_rows_v3's ring (NST stages of NC key cells: one 4 KB cell per barrier, double-buffered), score window (UTILE
# cells per top-k pass) and CTAs per SM: 38 KB of shared memory and 128 registers at 131k tokens, 4 CTAs per SM
_V3 = dict(NC=1, NST=2, UTILE=40, MINB=4)


def rows_v3_on() -> bool:
    # FSB_ROWS_V3=1 (the default): fsb_rows_v3 instead of fsb_rows_fused2 (the same cells and provenance bit for bit)
    return os.environ.get("FSB_ROWS_V3", "1") == "1" and not _v3_failed


def _v3_cfg_ok(Mc: int, Ca: int, Cb: int) -> bool:
    return rows_v3_on() and Mc <= V3_MC_MAX and Ca + Cb <= 32


def _v3_source(CA: int, CB: int, KK: int, lse_mode: bool, pos_mode: bool, prov_on: bool, prov_bit: int,
               out_i32: bool) -> str:
    defs = dict(CA=CA, CB=CB, NT=_NT, MCW=MC_MAX // 32, KK=KK, LSE_MODE=int(lse_mode), POS_MODE=int(pos_mode),
                PROV_ON=int(prov_on), PROV_BIT=int(prov_bit), OUT_I32=int(out_i32), V3=1, **_V3)
    return "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _SRC


def _v3_smem(C: int, Mc: int) -> int:
    # fsb_rows_v3's dynamic shared memory: cell list 1 KB, need bits 512 B, slot ranks (128 queries x an odd number of
    # words, 64-byte aligned), scan totals 64 B, then the larger of the union scratch (presence bytes and cell -> rank
    # map of Mcp cells, need bytes 4 KB) and the key ring (+ the cells' cellA values) with the score window
    Mcp = (Mc + 2047) // 2048 * 2048
    pu = (128 * ((C // 2) | 1) * 4 + 63) // 64 * 64
    ring = _V3["NST"] * _V3["NC"] * (4096 + 16) + _V3["UTILE"] * 512
    return 1024 + 512 + pu + 64 + max(3 * Mcp + 4096, ring)


def _v3_function(dev, cfg):
    # the compiled fsb_rows_v3 of a configuration (preferring the largest shared-memory carveout: 4 CTAs per SM take
    # about 140 KB), None when it cannot be built (the caller then runs fsb_rows_fused2)
    fn = _v3_fns.get(cfg)
    if fn is None and not torch.cuda.is_current_stream_capturing():
        from .fsb_beam_cuda import _libraries, load_function
        try:
            fn = load_function(dev, _v3_source(*cfg[1:]), "fsb_rows_v3")
            _libraries()[1].cuFuncSetAttribute(fn, 9, 100)       # CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            _v3_fns[cfg] = fn
        except Exception as e:
            if not _v3_failed:
                _v3_failed.append(repr(e))
                import warnings
                warnings.warn(f"FS-B fsb_rows_v3 unavailable, using fsb_rows_fused2: {e!r}")
            fn = None
    return fn


@torch.no_grad()
def rows_fused_cuda(sel_a: torch.Tensor, sel_b, q: torch.Tensor, K4: torch.Tensor, cellA: torch.Tensor,
                    t_idx: torch.Tensor, qv: torch.Tensor, sq0: int, sq1: int, lv: torch.Tensor, ls0: int,
                    sv: torch.Tensor, ss0: int, out: torch.Tensor, pv: torch.Tensor, Mc: int, ucap: int,
                    scale: float, K: int, score: str, prov: bool, prov_bit: int, q_nq: int = None):
    from .fsb_beam_cuda import _sm80, launch, load_function
    dev = sel_a.device
    if not _sm80(dev):
        return None
    Tp, Bq, Ca = sel_a.shape
    Cb = 0 if sel_b is None else sel_b.shape[-1]
    cfg = (dev.index, Ca, Cb, K, score != "max", score == "lse_pos", bool(prov), int(prov_bit),
           out.dtype == torch.int32)
    fn = _v3_function(dev, cfg) if _v3_cfg_ok(Mc, Ca, Cb) else None
    smem = _v3_smem(Ca + Cb, Mc) if fn is not None else 0
    if fn is None:
        fn = _fused_fns.get(cfg)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            fn = _fused_fns[cfg] = load_function(dev, _fused_source(*cfg[1:]), "fsb_rows_fused2")
        except Exception as e:
            if not _fused_failed:
                _fused_failed.append(repr(e))
            return None
    sa = sel_a.contiguous()
    sb = sel_b.contiguous() if Cb else sa
    if sa.data_ptr() % 16 or sb.data_ptr() % 16:
        sa, sb = sa.clone(), (sb.clone() if Cb else sa)
    from .fsb_beam_cuda import packed_queries, stage_view
    qsv = stage_view(q, Tp, Bq, q_nq)
    if qsv is not None:
        (rph, nq), qc, qst0, qst1 = qsv, q, q.stride(0), q.stride(1)   # read in place; the kernel clamps pads
    else:
        qc = packed_queries(q, Tp, Bq, q_nq) if q.dim() == 3 else q.reshape(Tp, Bq, 64).contiguous()
        if qc.data_ptr() % 16:
            qc = qc.clone()
        rph, nq, qst0, qst1 = 1, Bq, Bq * 64, 64
    # the outputs: packed (Tp, Bq, K) or a view (T'', n, K) in the stage-view row layout (view row v holds the cells
    # of rows v * RPH'' .. back to back, n >= RPH'' * Bq); provenance in the same layout. The kernel takes one RPH: a
    # packed side without pads is re-expressed in the other side's layout
    orph = Tp // out.shape[0] if (out.dim() == 3 and out.shape[0] > 0 and Tp % out.shape[0] == 0) else 0
    if not (orph and out.stride(2) == 1 and out.stride(1) == K and out.shape[1] >= orph * Bq
            and (not prov or pv.stride() == out.stride())):
        return None
    ost0 = out.stride(0)
    if orph != rph:
        if rph == 1 and nq == Bq and qst0 == Bq * qst1:
            rph, nq, qst0 = orph, orph * Bq, orph * Bq * qst1
        elif orph == 1 and ost0 == Bq * K:
            ost0 = rph * Bq * K
        else:
            return None
    tc = t_idx.contiguous()
    ovf = (torch.zeros if Mc > ucap else torch.empty)((), dtype=torch.bool, device=dev)
    if Tp == 0:
        return ovf
    P, I32, I64, F32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_float
    args = [P(sa.data_ptr()), P(sb.data_ptr()), P(qc.data_ptr()), P(K4.data_ptr()),
            P(cellA.data_ptr()), I64(cellA.stride(0)), I64(cellA.stride(1)),
            P(tc.data_ptr()), P(qv.data_ptr()), I64(sq0), I64(sq1),
            P(lv.data_ptr()), I64(ls0), P(sv.data_ptr()), I64(ss0),
            P(out.data_ptr()), P(pv.data_ptr()), P(ovf.data_ptr()),
            I32(Mc), I32(ucap), I32(K4.shape[1]), I64(K4.stride(0)), I64(K4.stride(1)), I64(K4.stride(2)),
            F32(float(scale)), I64(qst0), I64(qst1), I32(nq), I32(rph), I64(ost0)]
    launch(fn, Tp, _NT, dev, args, smem=smem)
    return ovf

