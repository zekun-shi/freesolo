"""CUDA row union of the block-shared FS-B prefill re-rank (fsb_rows, FSB_ROWS_IMPL=cuda: the default).

One CTA per row (head, query block) replaces fsb_rows' per-query union kernel, the torch glue that turns its output
into the row's distinct-cell list and the cell -> list position map, and the host check of the list's capacity: the
row's beam slots are read once, every valid cell sets its bit in a shared-memory bitmap (the beam-b slots that repeat
a beam-a cell of the same query are flagged in a per-query bitmask instead, union_candidates' rule), and an exclusive
scan of the bitmap words' popcounts gives each present cell its rank in the ascending list. Integer logic only, so
the list, its count and the positions are the torch glue's, exactly. A row with more than `ud` distinct cells raises
a device flag that the caller reads when it chooses (fsb_rows defers it to once per layer). Compiled with NVRTC at
first use and launched through fsb_beam_cuda's driver-API helpers; without NVRTC fsb_rows keeps its Triton path.
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
