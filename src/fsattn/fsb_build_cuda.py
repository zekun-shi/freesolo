"""CUDA construction of the FS-B prefill forests (FSB_BUILD_CUDA=1, the default, on top of the fused build
FSB_FAST_BUILD=1; FSB_BUILD_CUDA=0 keeps build_forest's chain, FSB_FAST_BUILD=0 the reference builder).

build_forests_cuda writes the flattened forests of one layer (node records, leaves, admission tables and the
per-block prefix counts m_b) directly, for both the main forest and the union-tau forest, bit-identical to
build_forest + searchsorted, in two launches. K0 (fsb_forest_prologue) builds the admission tables. K1
(fsb_forest_build) runs one 512-thread CTA per task of a static table: a batch of consecutive chunks of height
k <= 9 (512 slots, built top-down: nodes of 64-512 rows CTA-wide with the rows in shared memory, nodes of <= 32 rows
per warp with the rows in registers, <= 16-row nodes on 16-row half tiles), a node of >= 1024 rows of a taller chunk
(rows streamed from L2), or a 512-row subtree of such a chunk; the last two wait in-launch for their parent's
completion flag (blocks are assumed to start in index order, see wait_flag). Every floating-point step is the
instruction the Triton/torch builder uses (add/sub/mul .rn without ftz, div.rn.ftz and sqrt.rn.ftz as
tl.math.div_rn / tl.sqrt_rn compile, the mean as mul.rn.ftz by 2^-p, which is
bit-identical to div.rn.ftz by 2^p for every float), with the same pairwise-halving reduction trees, the padded
group build's signed zeros (folded once per sum), first-index argmax, and torch's stable sort order (value order with
-0 == +0, ties by slot). The source is compiled at first use with NVRTC through fsb_beam_cuda's loader; without
NVRTC, or for shapes it does not cover, the existing builder runs. With FSB_BUILD_SMALL=1 (the default; 0 keeps the
build kernel) a short or mid prefill's trimmed build runs its chunks (all at most 2^8 cells high) in
fsb_forest_build_small instead, with the same bits (small_on, _small_units).
"""
from __future__ import annotations

import ctypes
import functools
import heapq
import math
import os
import threading
import warnings
from collections import OrderedDict

import torch

from . import fsb_beam_cuda as _bc

_BUILD_CU = _bc._load_cu_sections("fsb_build.cu")
_COMMON = _BUILD_CU["_COMMON"]
_PROLOGUE = _BUILD_CU["_PROLOGUE"]
_CPHASE = _BUILD_CU["_CPHASE"]
_WPHASE = _BUILD_CU["_WPHASE"]
_GPHASE = _BUILD_CU["_GPHASE"]
_BHELPERS = _BUILD_CU["_BHELPERS"]
_BUILD = _BUILD_CU["_BUILD"]
_SMALL = _BUILD_CU["_SMALL"]

_NC_EXTRA = 9                                            # [mu | rad | amax | pmin | cnt | chl | chr | src | off | len]
_THREADS = 512
_lock = threading.Lock()
_fns: dict = {}
_TABLES_MAX = 8                                           # task tables kept for the most recently used shapes
_tables: OrderedDict = OrderedDict()                      # shape -> task table, least recently used first
_pinned: dict = {}                                        # shape -> task table a CUDA graph captured (kept for good)
_failed: dict = {}                                        # kernel variant -> its setup error (that variant falls back)
_small_fns: dict = {}                                     # kernel variant -> fsb_forest_build_small
_small_failed: dict = {}                                  # kernel variant -> its setup error (FSB_BUILD_SMALL off)
_warned: set = set()                                      # per-call failures already reported
_ONLY_K = None                                            # debug: restrict the task table to these heights


_SMALL_KMAX = 8                                          # tallest chunk fsb_forest_build_small builds (2^8 slots)
# share of the SMs a unit table that fits on the GPU spreads over: the rest stays free for the prefill's overlapped
# window branch (FSB_WINDOW_OVERLAP), which a build on every SM would hold back (each CTA fills an SM's registers)
_SMALL_SM_SHARE = 0.5


def small_on() -> bool:
    # FSB_BUILD_SMALL=1 (the default): a trimmed prefill build (mb_max) whose chunks are all at most _SMALL_KMAX high
    # runs them in fsb_forest_build_small (two warps per 32-slot tile, the CTAs spread over the GPU; the same records
    # bit for bit): the short and mid prefills, where the build kernel's 512-slot batches leave most of the GPU idle
    # and their long per-warp chains set the time. Taller builds are throughput-bound and stay in the build kernel, as
    # do decode builds. 0: the build kernel builds every chunk
    return os.environ.get("FSB_BUILD_SMALL", "1") == "1"


def _small_fits(M: int, mb_max) -> bool:
    # every chunk of the trimmed table is at most _SMALL_KMAX high (so the build kernel has nothing left)
    return mb_max is not None and all(k <= _SMALL_KMAX or spine_chunks(M, k, mb_max)[2] == 0
                                      for k in forest_layout(M)["ks"])


def prefill_trim_on() -> bool:
    # FSB_BUILD_PREFILL_TRIM=1 (the default): the inference prefill (fsb_prefill_fast) builds only the chunks its
    # spines can read (spine_chunks); 0 builds every chunk
    return os.environ.get("FSB_BUILD_PREFILL_TRIM", "1") == "1"


def spine_chunks(M: int, k: int, mb_max) -> tuple:
    # (first chunk, log2 stride, count) of the height-k chunks a whole-forest build writes: all M >> k of them, or
    # with mb_max (an upper bound on every m_b a spine row of the forest can have) the chunks a spine can read. The
    # spine of a prefix of m_b admitted cells (build_spines) reads chunk (k, j) only when bit k of m_b is set, with
    # j = (m_b >> (k + 1)) << 1 (even) and (j + 1) 2^k <= m_b; over every m_b <= mb_max that is each even j with
    # (j + 1) 2^k <= mb_max (m_b = (j + 1) 2^k reads it). Chunks outside that set are never read by a spine, by the
    # beams that descend from the spines' chunk roots, or by their cell epilogues (leaf rows of the chunk itself)
    if mb_max is None:
        return 0, 0, M >> k
    return 0, 1, ((min(int(mb_max), M) >> k) + 1) // 2


def build_cuda_on() -> bool:
    # FSB_BUILD_CUDA=1 replaces the fused (Triton) build it reproduces bit for bit: without that build (FSB_FAST_BUILD=0
    # restores the reference builder, or no Triton) it steps aside as well
    if os.environ.get("FSB_BUILD_CUDA", "1") != "1":
        return False
    from .fsb_build_fast import fast_build_on
    return fast_build_on()


def _variant(index, Dh: int) -> tuple:
    # one compiled variant of the kernels: device, head size, debug defines
    return (index, Dh, tuple(sorted(_DEBUG_DEFS.items())))


def _group() -> int:
    return int(os.environ.get("FSB_BUILD_GROUP", "4"))


def kpad_table(ks, g: int) -> dict:
    # Kpad(k) of _grouped_chunk_build_iter: the tallest height of k's build group
    if g <= 0:
        return {k: max(ks) for k in ks}
    groups: dict = {}
    for k in ks:
        groups.setdefault(k // g, []).append(k)
    return {k: max(v) for v in groups.values() for k in v}


def _layvec(lay: dict, g: int) -> list:
    # the layout vector the kernels index: per level the node and leaf bases, and the fold threshold at slot 32
    kp = kpad_table(lay["ks"], g)
    lv = [0] * 40
    for k in lay["ks"]:
        lv[k], lv[16 + k] = lay["bases"][k], lay["leafbases"][k]
    lv[32] = sum(1 << k for k in lay["ks"] if kp[k] > k)
    return lv


def _chunk_rows(rows: list, f: int, t: int, k: int, j: int, flag0: int) -> int:
    # append the task-table records of the single chunk (f, t, k, j) to rows: field-for-field the records the full
    # table holds for this chunk, parents before children (a task only ever waits for an earlier one), with the
    # chunk's completion flags at flag0.. (contiguous per chunk). Returns the number of flags used.
    if k <= 9:
        rows.append([0, f, t, k, j, 1, 0, 0, -1, -1])     # batch of one chunk
        return 0
    rows.append([1, f, t, k, j, 0, 0, 0, flag0, -1])
    for d in range(1, k - 9):
        for n in range(1 << d):
            rows.append([1, f, t, k, j, 0, d, n, flag0 + (1 << d) - 1 + n,
                         flag0 + (1 << (d - 1)) - 1 + (n >> 1)])
    for n in range(1 << (k - 9)):
        rows.append([2, f, t, k, j, 0, k - 9, n, -1, flag0 + (1 << (k - 10)) - 1 + (n >> 1)])
    return (1 << (k - 9)) - 1


@functools.lru_cache(maxsize=64)
def forest_layout(M: int) -> dict:
    # _flatten_forest's constants for M cells: chunk heights, node/leaf bases, nodes per chunk, chunk counts (one dict
    # per M shared by the callers, read only)
    Kmax = int(math.floor(math.log2(max(M, 1))))
    ks = [k for k in range(Kmax + 1) if (M >> k) > 0]
    bases, nodes_per, leafbases, Cmaxs = {}, {}, {}, {}
    nb = lb = 0
    for k in ks:
        C = M >> k
        bases[k], nodes_per[k], leafbases[k], Cmaxs[k] = nb, 2 * (1 << k) - 1, lb, C
        nb += C * nodes_per[k]
        lb += C * (1 << k)
    return {"Kmax": Kmax, "ks": ks, "bases": bases, "nodes_per": nodes_per, "leafbases": leafbases, "Cmaxs": Cmaxs,
            "Ntot": nb, "Ltot": lb}


_DEBUG_DEFS: dict = {}                                    # debug: extra #defines (e.g. DBG_TIME)


def _defines(Dh: int) -> str:
    defs = {"DH": Dh, "PER": Dh // 32, "NC": Dh + _NC_EXTRA, "NA": _NC_EXTRA, "XS": Dh + 4, **_DEBUG_DEFS}
    return "".join(f"#define {k} {v}\n" for k, v in defs.items())


def _source(Dh: int) -> str:
    return _defines(Dh) + _bc._HELPERS + _COMMON + _PROLOGUE + _WPHASE + _CPHASE + _GPHASE + _BHELPERS + _BUILD


def _small_source(Dh: int) -> str:
    # fsb_forest_build_small in a module of its own (compiled only when FSB_BUILD_SMALL is on)
    return _defines(Dh) + _bc._HELPERS + _COMMON + _WPHASE + _CPHASE + _BHELPERS + _SMALL


def _smem_bytes(Dh: int) -> int:
    # dynamic shared memory of a build CTA: the larger carve-up of the task kinds it may run. SMEM_FLOATS (carve:
    # batches and 512-row subtrees: rows, two key buffers, pad signs, perms, cells/ceil/spread, mu/v/v-sum, pv, warp
    # stats, far/pvp) and GSMEM_FLOATS (gcarve: nodes of >= 1024 rows: 8192 keys, 8192 cells, warp partials, mu, v,
    # warp stats, misc/pvp); the G nodes need more for Dh = 32 (101,056 vs 95,808 bytes)
    c = 512 * (Dh + 4) + 4 * 512 + 2 * 16 + 4 * 512 + 3 * 8 * Dh + 512 + 16 * 5 + 32
    g = 2 * 8192 + 8192 + 16 * Dh + 2 * Dh + 16 * 5 + 32
    return 4 * max(c, g)


def _small_smem_bytes(Dh: int) -> int:
    # SSMEM_FLOATS: eight 32-row tiles (rows, sort keys, hand-off lists, slot arrays), node slots, per-warp lists
    return 4 * (256 * (Dh + 4) + 6 * 256 + 16 + 4 * 256 + 12 * Dh + 48 + 2 * 512 + 4 * 256)


class _Unavailable(RuntimeError):
    """The kernels cannot be set up for this variant (reported once; the variant falls back from then on)."""


def _functions(dev: torch.device, Dh: int):
    # the prologue and build kernels of a variant, compiled at first use. A failure here (no NVRTC, a compile or load
    # error, less shared memory per block than the build needs) is permanent for the variant: it falls back to the
    # existing builder from then on, with one warning. Running out of memory is not such a failure.
    key = _variant(dev.index, Dh)
    fn = _fns.get(key)
    if fn is None:
        try:
            src = _source(Dh)
            pro = _bc.load_function(dev, src, "fsb_forest_prologue")
            bld = _bc.load_function(dev, src, "fsb_forest_build")
            bld_lean = _bc.load_function(dev, src, "fsb_forest_build_lean")
            prep_lean = _bc.load_function(dev, src, "fsb_prep_lean_tasks")
            sp_lean = _bc.load_function(dev, src, "fsb_spine_lean")
            bld_lean16 = _bc.load_function(dev, src, "fsb_forest_build_lean16")
            sp_lean16 = _bc.load_function(dev, src, "fsb_spine_lean16")
            bld16 = _bc.load_function(dev, src, "fsb_forest_build16")
            prep_dirty = _bc.load_function(dev, src, "fsb_prep_lean_dirty")
            _, cuda = _bc._libraries()
            cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            _bc._check(cuda.cuFuncSetAttribute(bld, 8, _smem_bytes(Dh)), "cuFuncSetAttribute(max dynamic smem)")
            _bc._check(cuda.cuFuncSetAttribute(bld_lean, 8, _smem_bytes(Dh)), "cuFuncSetAttribute(max dynamic smem)")
            _bc._check(cuda.cuFuncSetAttribute(bld_lean16, 8, _smem_bytes(Dh)), "cuFuncSetAttribute(max dynamic smem)")
            _bc._check(cuda.cuFuncSetAttribute(bld16, 8, _smem_bytes(Dh)), "cuFuncSetAttribute(max dynamic smem)")
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            if "CUDA error 2 (" in str(e):                  # CUDA_ERROR_OUT_OF_MEMORY from a driver call
                raise torch.cuda.OutOfMemoryError(f"fsb_build_cuda: {e}") from e
            _failed[key] = repr(e)
            warnings.warn(f"FS-B CUDA forest build unavailable on cuda:{dev.index} for head size {Dh}, using the "
                          f"Triton builder: {e!r}")
            raise _Unavailable(repr(e)) from e
        fn = _fns[key] = (pro, bld, bld_lean, prep_lean, sp_lean, bld_lean16, sp_lean16, bld16, prep_dirty)
    return fn[:2]



def _small_function(dev: torch.device, Dh: int):
    # fsb_forest_build_small of a variant, compiled at first use in a module of its own; a failure to set it up turns
    # FSB_BUILD_SMALL off for the variant (its builds run every chunk in the build kernel), with one warning
    key = _variant(dev.index, Dh)
    fn = _small_fns.get(key)
    if fn is None:
        try:
            fn = _bc.load_function(dev, _small_source(Dh), "fsb_forest_build_small")
            _, cuda = _bc._libraries()
            cuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            _bc._check(cuda.cuFuncSetAttribute(fn, 8, _small_smem_bytes(Dh)), "cuFuncSetAttribute(max dynamic smem)")
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            if "CUDA error 2 (" in str(e):
                raise torch.cuda.OutOfMemoryError(f"fsb_build_cuda: {e}") from e
            _small_failed[key] = repr(e)
            warnings.warn(f"FS-B small-chunk forest build unavailable on cuda:{dev.index} for head size {Dh}, "
                          f"using the build kernel for every chunk: {e!r}")
            return None
        _small_fns[key] = fn
    return fn


def _launch(fn, grid: int, block: int, smem: int, dev: torch.device, args) -> None:
    _, cuda = _bc._libraries()
    params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    stream = torch.cuda.current_stream(dev).cuda_stream
    res = cuda.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, smem, ctypes.c_void_p(stream), params, None)
    if res == 2:                                           # CUDA_ERROR_OUT_OF_MEMORY (e.g. local memory for the launch)
        raise torch.cuda.OutOfMemoryError("fsb_build_cuda: cuLaunchKernel: CUDA error 2 (out of memory)")
    _bc._check(res, "cuLaunchKernel")


def _task_cost(rec) -> float:
    # rough duration of a task (us on an A100) for placing dependent tasks: batch of height k <= 9 ~ 23 us per level,
    # G node ~0.087 us per row, 512-row subtree ~250 us
    typ, k, l = rec[0], rec[3], rec[6]
    if typ == 0:
        return 23.0 * (k + 1)
    if typ == 1:
        return 0.087 * (1 << (k - l))
    return 250.0


def _table_lookup(key) -> tuple:
    # (cached task table or None, whether a CUDA graph holds it). A capture pins the table (its replays read it); a
    # capture of a shape never built eagerly fails, like one of kernels never compiled
    capturing = torch.cuda.is_current_stream_capturing()
    with _lock:
        if key in _pinned:
            return _pinned[key], True
        hit = _tables.get(key)
        if hit is not None:
            if capturing:
                _pinned[key] = _tables.pop(key)
                return hit, True
            _tables.move_to_end(key)
            return hit, False
    if capturing:
        raise RuntimeError("fsb_build_cuda: the task table of a shape must be built before graph capture")
    return None, False


def _table_store(key, hit) -> None:
    evicted = False
    with _lock:
        _tables[key] = hit
        while len(_tables) > _TABLES_MAX:
            _tables.popitem(last=False)
            evicted = True
    if evicted and not torch.cuda.is_current_stream_capturing():
        import gc
        gc.collect()


def _table_key(T: int, M: int, nf: int, g: int, kmax_cuda: int, dev: torch.device, mb_max=None,
               kmin: int = -1) -> tuple:
    return (T, M, nf, g, kmax_cuda, dev.index, None if _ONLY_K is None else tuple(sorted(_ONLY_K)), mb_max) + (
        (kmin,) if kmin >= 0 else ())


def _tasks(T: int, M: int, nf: int, g: int, dev: torch.device, kmax_cuda: int, only_layv: bool = False,
           mb_max=None, kmin: int = -1):
    # static task table of one launch and the layout vector, cached per shape, grouping and spine bound: an LRU of the
    # _TABLES_MAX most recently used shapes, plus every table a CUDA graph captured (_table_lookup). Built outside
    # graph capture: the first call of a shape runs before any capture of it, as for the kernels.
    # Records (10 ints): type, f, t, k, j, nsub, l, n, flag, dep. type 0: batch of nsub chunks j + (i << l) of height
    # k <= 9, of the n heads t .. t + n - 1 (0: of head t); type 1: G node (level l, index n) of a chunk of height
    # >= 10, raising flag when its slot lists are written; type 2: the 512-cell subtree n (at level k - 9) of such a
    # chunk. dep: the flag a task waits for (its parent G node), -1 none. The chunks of each height are
    # spine_chunks(M, k, mb_max): every chunk, or with mb_max only those a prefill spine can read (every other chunk,
    # batched with l = 1; up to 16 heads per W batch when they fill it only partly). kmin >= 0: the heights above kmin
    # only (fsb_forest_build_small builds the others).
    # Order: G roots (tallest first), then the batches longest first; each dependent task is inserted once the
    # estimated dispatch time (sum of earlier task costs / SMs) passes its parent's estimated finish, so waits are
    # rare; a task only ever waits for an earlier one, which is resident or finished by then (blocks start in index
    # order: see wait_flag in the kernel source).
    key = _table_key(T, M, nf, g, kmax_cuda, dev, mb_max, kmin)
    hit, pinned = _table_lookup(key)
    if hit is None and only_layv:
        lay = forest_layout(M)
        hit = (None, torch.tensor(_layvec(lay, g), dtype=torch.int32, device=dev), [], 0, 0,
               {torch.cuda.current_stream(dev).cuda_stream})
        _table_store(key, hit)
    elif hit is None or (not only_layv and hit[0] is None):
        lay = forest_layout(M)
        chunks = {k: spine_chunks(M, k, mb_max) for k in lay["ks"]}
        use = [k for k in lay["ks"] if kmin < k <= kmax_cuda and (_ONLY_K is None or k in _ONLY_K) and chunks[k][2] > 0]
        big = sorted((k for k in use if k >= 10), reverse=True)
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        rows = []
        clock = [0.0]                                      # estimated dispatch time of the next appended task
        children: dict = {}                                # parent flag -> dependent records
        ready: list = []                                   # heap of (parent's estimated finish, seq, record)
        seq = [0]

        def push(rec):
            rows.append(rec)
            if rec[8] >= 0:
                done = clock[0] + _task_cost(rec)
                for ch in children.pop(rec[8], []):
                    heapq.heappush(ready, (done, seq[0], ch)); seq[0] += 1
            clock[0] += _task_cost(rec) / nsm

        def flush(force=False):
            while ready and (force or ready[0][0] <= clock[0]):
                push(heapq.heappop(ready)[2])

        flags: dict = {}
        roots = []
        for k in big:                                      # G nodes and subtrees of chunks of height >= 10
            ja, jsh, nj = chunks[k]
            for f in range(nf):
                for t in range(T):
                    for j in range(ja, ja + (nj << jsh), 1 << jsh):
                        flags[(f, t, k, j, 0, 0)] = len(flags)
                        roots.append([1, f, t, k, j, 0, 0, 0, flags[(f, t, k, j, 0, 0)], -1])
                        for d in range(1, k - 9):
                            for n in range(1 << d):
                                flags[(f, t, k, j, d, n)] = len(flags)
                                par = flags[(f, t, k, j, d - 1, n >> 1)]
                                children.setdefault(par, []).append([1, f, t, k, j, 0, d, n, flags[(f, t, k, j, d, n)], par])
                        for n in range(1 << (k - 9)):
                            par = flags[(f, t, k, j, k - 10, n >> 1)]
                            children.setdefault(par, []).append([2, f, t, k, j, 0, k - 9, n, -1, par])
        for rec in roots:                                  # tallest first (big is sorted descending)
            push(rec)
        for k in sorted((k for k in use if k <= 9), reverse=True):
            per = 512 >> k
            ja, jsh, nj = chunks[k]
            # the spine chunks of a short prefill fill a fraction of a block: its W batches (k <= 5, independent warps)
            # take the same chunks of several heads, 2^ceil(log2(warps per head)) warps each
            npk = max(1, 16 // (1 << (-(-(nj << k) // 32) - 1).bit_length())) if mb_max is not None and k <= 5 else 1
            for f in range(nf):
                for t in range(0, T, npk):
                    for i0 in range(0, nj, per):
                        flush()
                        push([0, f, t, k, ja + (i0 << jsh), min(per, nj - i0), jsh, min(npk, T - t) if npk > 1 else 0,
                              -1, -1])
        while ready:
            flush(force=True)
        assert not children
        layv = hit[1] if hit is not None else torch.tensor(_layvec(lay, g), dtype=torch.int32, device=dev)
        hit = (torch.tensor(rows if rows else [[0] * 8 + [-1, -1]], dtype=torch.int32, device=dev).reshape(-1, 10).contiguous(),
               layv, [(0, len(rows))] if rows else [], len(big), len(flags),
               {torch.cuda.current_stream(dev).cuda_stream})          # the streams that read the table
        _table_store(key, hit)
    if not pinned:
        # an evicted table's memory is reused at once on the stream that allocated it: have the allocator wait for any
        # other stream that read it as well
        s = torch.cuda.current_stream(dev)
        if s.cuda_stream not in hit[5]:
            if hit[0] is not None:
                hit[0].record_stream(s)
            hit[1].record_stream(s)
            hit[5].add(s.cuda_stream)
    return hit[:5]


def _tile_warp(s: int, role: int) -> int:
    # the warp of tile slot s's A (role 0) or B warp: A warps 0 .. 7, B warps 8 .. 15 rotated by two, so that the A
    # warps (alone in the loads, C levels and level-5 split) and the B warps of a CTA's first tiles sit on different
    # schedulers (warp w issues on scheduler w % 4)
    return s if role == 0 else 8 + ((s + 2) & 7)


# rough latency (thousands of cycles) of a fsb_forest_build_small unit by chunk height: the h-levels of a half tile from
# level min(k, 4), plus the level-5 split and the C levels of the taller chunks
_SMALL_COST = {0: 6, 1: 22, 2: 34, 3: 43, 4: 52, 5: 66, 6: 78, 7: 90, 8: 102}


def _small_units(T: int, M: int, nf: int, dev: torch.device, mb_max, kmax: int):
    # static unit table of fsb_forest_build_small for the chunks of height k <= kmax of spine_chunks(M, k, mb_max) (all
    # heads of both forests) and its CTA count, cached per shape like the task tables (and pinned by a capture). Units
    # (records: see the kernel source): one warp per 16 slots of chunks of height <= 4; per chunk of height 5 .. 8 one
    # 32-slot tile slot per 32 slots, two warps each (A splits, B builds half 1), the chunk's first tile slot a multiple
    # of its tile count (its named barriers: see the kernel). Each CTA holds units of one height only: every height runs
    # its own long straight-line code path, and warps of different paths on one SM thrash its instruction cache (about
    # twice as slow). A table that fits on _SMALL_SM_SHARE of the SMs at once (eight tile slots per SM) spreads each
    # height over that many CTAs in proportion to its estimated work (latency x warps), a larger one fills each CTA,
    # the longest heights first.
    key = ("small", T, M, nf, kmax, dev.index, None if _ONLY_K is None else tuple(sorted(_ONLY_K)), mb_max)
    hit, pinned = _table_lookup(key)
    if hit is None:
        lay = forest_layout(M)
        classes = {}                                       # height -> [(tiles (0: one warp of kind 1), record)]
        for k in lay["ks"]:
            if k > kmax or (_ONLY_K is not None and k not in _ONLY_K):
                continue
            ja, jsh, nj = spine_chunks(M, k, mb_max)
            for f in range(nf):
                for t in range(T):
                    if k <= 4:
                        per = 16 >> k
                        for i0 in range(0, nj, per):
                            classes.setdefault(k, []).append(
                                (0, [1, f, t, k, ja + (i0 << jsh), min(per, nj - i0), jsh, 0]))
                    else:
                        for i in range(nj):
                            classes.setdefault(k, []).append((1 << (k - 5), [2, f, t, k, ja + (i << jsh), 1, 0, 0]))
        nsm = int(torch.cuda.get_device_properties(dev).multi_processor_count * _SMALL_SM_SHARE)
        tiles = {k: sum(g for g, _ in us) + (sum(1 for g, _ in us if g == 0) + 1) // 2 for k, us in classes.items()}
        alloc = {k: -(-n // 8) for k, n in tiles.items()}  # CTAs per height, eight tile slots each at the least
        if sum(tiles.values()) <= 8 * nsm:
            work = {k: _SMALL_COST[k] * sum(2 * g or 1 for g, _ in us) for k, us in classes.items()}
            for _ in range(nsm - sum(alloc.values())):
                k = max((k for k in classes if alloc[k] < len(classes[k])), key=lambda k: work[k] / alloc[k],
                        default=None)
                if k is None:
                    break
                alloc[k] += 1
        recs = []
        for k in sorted(classes, key=lambda k: -_SMALL_COST[k]):
            n = alloc[k]
            crec = [[[0] * 8 for _ in range(16)] for _ in range(n)]
            free = [[2] * 8 for _ in range(n)]             # free warps of each tile slot
            for i, (g, rec) in enumerate(classes[k]):
                c = i % n
                if g == 0:
                    s0 = next((s for s in range(8) if free[c][s] == 1), None)
                    s0 = next(s for s in range(8) if free[c][s] == 2) if s0 is None else s0
                    crec[c][2 * s0 + 2 - free[c][s0]] = list(rec)
                    free[c][s0] -= 1
                else:
                    s0 = next(s0 for s0 in range(0, 8, g) if all(free[c][s] == 2 for s in range(s0, s0 + g)))
                    for ti in range(g):
                        free[c][s0 + ti] = 0
                        for role in range(2):
                            r = list(rec)
                            r[7] = role | (ti << 1) | ((s0 + ti) << 8) | (1 << 12)
                            crec[c][_tile_warp(s0 + ti, role)] = r
            recs += crec
        flat = [r for cta in recs for r in cta] or [[0] * 8]
        hit = (torch.tensor(flat, dtype=torch.int32, device=dev).reshape(-1, 8).contiguous(), len(recs),
               {torch.cuda.current_stream(dev).cuda_stream})
        _table_store(key, hit)
    if not pinned:
        s = torch.cuda.current_stream(dev)
        if s.cuda_stream not in hit[2]:
            hit[0].record_stream(s)
            hit[2].add(s.cuda_stream)
    return hit[0], hit[1]


def _aligned(x: torch.Tensor, n: int) -> bool:
    # the pointer the kernels get is n-byte aligned (a non-contiguous input is copied, into fresh aligned memory)
    return not x.is_contiguous() or x.data_ptr() % n == 0


def build_cuda_ok(Kbar: torch.Tensor, ceil_c: torch.Tensor, spread, taus, E: torch.Tensor) -> bool:
    # inputs the kernels cover: fp32 Kbar (T, M, 32 | 64) with 1 <= M < 16384 (T = 0: empty tables), 16-byte aligned
    # (rows are loaded as float4); fp32 ceilings, and spreads, of shape (T, M); int64 block starts E (1-D); 1 or 2
    # floating-point thresholds of one length H dividing T; all on one device, with a variant of the kernels that set up
    if not (build_cuda_on() and Kbar.is_cuda and Kbar.dim() == 3 and Kbar.dtype == torch.float32
            and ceil_c.dtype == torch.float32 and Kbar.shape[-1] in (32, 64) and 1 <= Kbar.shape[1] < 16384
            and tuple(ceil_c.shape) == tuple(Kbar.shape[:2]) and E.dim() == 1 and E.dtype == torch.int64
            and E.device == Kbar.device and ceil_c.device == Kbar.device
            and _aligned(Kbar, 16) and _aligned(ceil_c, 4) and _aligned(E, 8)
            and _variant(Kbar.device.index, Kbar.shape[-1]) not in _failed):
        return False
    if spread is not None and not (spread.dtype == torch.float32 and spread.device == Kbar.device
                                   and tuple(spread.shape) == tuple(Kbar.shape[:2]) and _aligned(spread, 4)):
        return False
    if not (1 <= len(taus) <= 2 and all(tau.dim() == 1 for tau in taus)):
        return False
    H = taus[0].shape[0]
    return H >= 1 and Kbar.shape[0] % H == 0 and all(
        tau.shape[0] == H and tau.device == Kbar.device and tau.is_floating_point()
        and (tau.dtype != torch.float32 or _aligned(tau, 4)) for tau in taus)


def _tau32(tau: torch.Tensor) -> torch.Tensor:
    # the fp32 threshold the prologue compares the fp32 ceilings with, admitting exactly what the reference does: torch
    # compares fp32 with fp64 in fp64, i.e. ce >= tau iff ce >= the smallest fp32 value >= tau (tau rounded up, which
    # keeps NaN, the infinities and the zeros); narrower floating types convert exactly
    t = tau.float()
    if tau.dtype == torch.float64:
        t = torch.where(t.double() < tau, torch.nextafter(t, torch.full_like(t, math.inf)), t)
    return t.contiguous()


def forests_cuda(Kbar: torch.Tensor, ceil_c: torch.Tensor, spread, taus, E: torch.Tensor, sel=None, lean=None,
                 mb_max=None):
    # build_forests_cuda when the kernels cover the inputs, else None (the caller runs the existing builder). Running out
    # of memory propagates (the existing builder needs more) and so does any failure inside a graph capture; a variant
    # that cannot be set up falls back from then on (_functions); any other failure falls back for this call only,
    # with a warning. sel = [(k, [j per head or None])] restricts the build to those dirty aligned chunks (a decode
    # ingest's); out-of-layout selections fall through to the whole-forest build inside build_forests_cuda. mb_max:
    # see build_forests_cuda.
    if not build_cuda_ok(Kbar, ceil_c, spread, taus, E):
        return None
    try:
        return build_forests_cuda(Kbar, ceil_c, spread, taus, E, sel=sel, lean=lean, mb_max=mb_max)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as e:
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:                                  # the context itself failed: report the original error
            capturing = True
        if capturing:
            raise
        wkey = (repr(e), sel is not None)
        if not isinstance(e, _Unavailable) and wkey not in _warned and len(_warned) < 64:
            _warned.add(wkey)
            warnings.warn(f"FS-B CUDA forest build failed, using the Triton builder for this call: {e!r}")
        return None


class _SelLayout(ValueError):
    # a chunk selection outside the M-cell layout: build_forests_cuda falls through to the whole-forest build
    pass


def _upload_tasks(rows: list, dev: torch.device) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32, device=dev).reshape(-1, 10).contiguous()


@functools.lru_cache(maxsize=64)
def _lean_layout_info(M: int, g: int) -> tuple:
    lay = forest_layout(M)
    kp = kpad_table(lay["ks"], g)
    fold_mask = sum(1 << k for k in lay["ks"] if kp[k] > k)
    nbig = sum(1 for k in lay["ks"] if k >= 10)
    return lay, fold_mask, nbig


def lean_fold_mask(M: int) -> int:
    # the fold flags (bit k: height k builds as part of a taller padded group) every CUDA build of a forest laid out
    # for M cells uses: the one input of a chunk's rows besides its cells, so rows built for different M may differ
    return _lean_layout_info(M, _group())[1]


def _lean_operands(Kbar: torch.Tensor, ceil_c: torch.Tensor, spread, lean):
    # the lean build's inputs as its kernels read them: row-strided (T, M_x, .) views where the layout allows (M_kb,
    # M_ce, M_spr, M_adm0 their per-head row strides), contiguous copies otherwise; the spread slot falls back to the
    # ceilings when there is no spread
    Dh = Kbar.shape[2]
    adm0, mt0 = lean[3], lean[4]
    if Kbar.stride(2) == 1 and Kbar.stride(1) == Dh and Kbar.stride(0) % Dh == 0 and Kbar.data_ptr() % 16 == 0:
        M_kb = Kbar.stride(0) // Dh
    else:
        Kbar = Kbar.contiguous()
        M_kb = Kbar.shape[1]
    if ceil_c.stride(1) == 1 and ceil_c.data_ptr() % 4 == 0:
        M_ce = ceil_c.stride(0)
    else:
        ceil_c = ceil_c.contiguous()
        M_ce = ceil_c.shape[1]
    if spread is not None:
        if spread.stride(1) == 1 and spread.data_ptr() % 4 == 0:
            sp = spread
            M_spr = sp.stride(0)
        else:
            sp = spread.contiguous()
            M_spr = sp.shape[1]
    else:
        sp = ceil_c
        M_spr = M_ce
    if adm0.stride(1) != 1 or adm0.data_ptr() % 8 != 0:
        adm0 = adm0.contiguous()
    M_adm0 = adm0.stride(0)
    mt0 = mt0 if (mt0.stride(0) == 1 and mt0.data_ptr() % 8 == 0) else mt0.contiguous()
    return Kbar, ceil_c, sp, adm0, mt0, M_kb, M_ce, M_spr, M_adm0


@torch.no_grad()
def _chunks_build(Kbar: torch.Tensor, ceil_c: torch.Tensor, spread, taus, E: torch.Tensor, sel: list,
                  kmax_cuda: int = 99, lean=None):
    # build_forests_cuda computing only the chunks of sel = [(k, js)] (js: the per-head chunk index of level k, or
    # None where the head has nothing at that level): each chunk writes the same node/leaf rows with the same bits
    # as the full build's records for it (record fields identical, scratch and flag addresses a pure function of
    # the record fields), parents before children inside one launch. node/leaf rows outside sel are not written.
    # The prologue (admission, m_tot, m_b, flag clearing) is unchanged, and the shape's task-table registry is the
    # full build's: one table per bucket in _tables (its cached layout vector is reused here). _SelLayout ->
    # out-of-layout selection.
    T, M_in, Dh = Kbar.shape
    M = lean[7] if lean is not None else M_in
    dev = Kbar.device
    nf = len(taus)
    H = taus[0].shape[0]
    g = _group()
    if lean is not None and nf == 1:
        lay, fold_mask, nbig = _lean_layout_info(M, g)
        Cmaxs = lay["Cmaxs"]
        fast_lean = 0 < len(sel) <= 16 and len(lean) >= 12 and lean[11] is not None
        for k, js in sel:
            Ck = Cmaxs.get(k)
            if Ck is None:
                raise _SelLayout(f"fsb_build_cuda: chunk level k={k} is not in the M={M} layout")
            if k > 9:
                fast_lean = False
            for j in js:
                if j is not None and (j < 0 or j >= Ck):
                    raise _SelLayout(f"fsb_build_cuda: chunk (k={k}, j={j}) is outside the M={M} layout")
        tkey = _table_key(T, M, nf, g, kmax_cuda, dev)
        if tkey not in _tables and tkey not in _pinned:
            _tasks(T, M, nf, g, dev, kmax_cuda, only_layv=True)
        _functions(dev, Dh)
        _, _, _, prep_lean, _, bld_lean16, sp_lean16 = _fns[_variant(dev.index, Dh)][:7]
        amu0, anx0, leaf0, _, _, jb0, Kmax_lean, _ = lean[:8]
        sp_node, sp_root, sp_hgt = lean[8:11] if len(lean) >= 11 else (None, None, None)
        task_buf = lean[11] if len(lean) >= 12 else None
        Kbar, ceil_c, sp, adm0, mt0, M_kb, M_ce, M_spr, M_adm0 = _lean_operands(Kbar, ceil_c, spread, lean)
        P, I32, U64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong
        jb_ptr = jb0.data_ptr() if jb0 is not None else 0
        if fast_lean:
            k_packed = 0
            for idx, (k, _) in enumerate(sel):
                k_packed |= (int(k) & 0xF) << (4 * idx)
            n_k = len(sel)
            nrec = n_k * T
            tasks_ptr = task_buf.data_ptr()
            lay_ptr = tasks_ptr + 40 * nrec
            _launch(prep_lean, 1, 256, 0, dev,
                    [P(tasks_ptr), P(lay_ptr), P(jb_ptr), P(mt0.data_ptr()),
                     U64(k_packed), I32(n_k), I32(T), I32(fold_mask), I32(int(Kmax_lean)),
                     I32(int(M_adm0)), I32(int(M_kb)), I32(int(M_ce)), I32(int(M_spr))])
            _launch(bld_lean16, nrec, _THREADS, _smem_bytes(Dh), dev,
                    [P(tasks_ptr), P(lay_ptr), P(Kbar.data_ptr()), P(ceil_c.data_ptr()),
                     P(sp.data_ptr()), P(adm0.data_ptr()), P(0), P(mt0.data_ptr()), P(mt0.data_ptr()),
                     P(amu0.data_ptr()), P(amu0.data_ptr()), P(anx0.data_ptr()), P(anx0.data_ptr()),
                     P(leaf0.data_ptr()), P(leaf0.data_ptr()), P(0),
                     P(0), I32(M), I32(amu0.shape[1]), I32(leaf0.shape[1]), I32(int(spread is not None)), I32(nbig),
                     I32(T), P(0), P(0)])
        else:
            rows: list = []
            nflags = 0
            for k, js in sel:
                for t in range(T):
                    j = js[t] if t < len(js) else None
                    if j is None:
                        rows.append([0 if k <= 9 else 1, 0, t, k, Cmaxs[k], 1, 0, 0, -1, -1])
                        continue
                    nflags += _chunk_rows(rows, 0, t, k, int(j), nflags)
            if not rows:
                rows = [[0] * 8 + [-1, -1]]
            lv = [2 * (1 << k) - k - 2 for k in range(16)] + [(1 << k) - 1 for k in range(16)] + [
                fold_mask, int(Kmax_lean), int(M_adm0), 0, int(M_kb), int(M_ce), int(M_spr), 0]
            nrec = len(rows)
            tasks_and_lay = _upload_tasks(rows + [lv[0:10], lv[10:20], lv[20:30], lv[30:40]], dev)
            tasks_ptr = tasks_and_lay.data_ptr()
            lay_ptr = tasks_ptr + 40 * nrec
            if nflags > 0:
                nctl = (nflags + 3) // 4 * 4
                work = torch.zeros((nctl + nbig * nf * T * M,), dtype=torch.int32, device=dev)
                ctl, scr = work.data_ptr(), work.data_ptr() + 4 * nctl
            else:
                ctl = scr = 0
            _launch(bld_lean16, nrec, _THREADS, _smem_bytes(Dh), dev,
                    [P(tasks_ptr), P(lay_ptr), P(Kbar.data_ptr()), P(ceil_c.data_ptr()),
                     P(sp.data_ptr()), P(adm0.data_ptr()), P(0), P(mt0.data_ptr()), P(mt0.data_ptr()),
                     P(amu0.data_ptr()), P(amu0.data_ptr()), P(anx0.data_ptr()), P(anx0.data_ptr()),
                     P(leaf0.data_ptr()), P(leaf0.data_ptr()), P(scr),
                     P(ctl), I32(M), I32(amu0.shape[1]), I32(leaf0.shape[1]), I32(int(spread is not None)), I32(nbig),
                     I32(T), P(jb_ptr), P(0)])
        if sp_node is not None and T > 0:
            _launch(sp_lean16, (T + 15) // 16, _THREADS, 0, dev,
                    [P(amu0.data_ptr()), P(anx0.data_ptr()), P(mt0.data_ptr()), P(sp_node.data_ptr()),
                     P(sp_root.data_ptr()), P(sp_hgt.data_ptr()), I32(T), I32(amu0.shape[1]), I32(int(Kmax_lean))])
            return [({"_lean_direct": True, "_sp_built": True, "jbase": jb0}, None)]
        return [({"_lean_direct": True, "_sp_built": False, "jbase": jb0}, None)]
    lay = forest_layout(M)
    rows = []
    nflags = 0
    for k, js in sel:
        if k not in lay["Cmaxs"]:
            raise _SelLayout(f"fsb_build_cuda: chunk level k={k} is not in the M={M} layout")
        for f in range(nf):
            for t in range(T):
                j = js[t] if t < len(js) else None
                if j is None:
                    continue
                if j < 0 or j >= lay["Cmaxs"][k]:
                    raise _SelLayout(f"fsb_build_cuda: chunk (k={k}, j={j}) is outside the M={M} layout")
                nflags += _chunk_rows(rows, f, t, k, int(j), nflags)
    if not rows:
        rows = [[0] * 8 + [-1, -1]]
    pro, bld = _functions(dev, Dh)
    layv = _tasks(T, M, nf, g, dev, kmax_cuda, only_layv=True)[1]     # the shape's registry table's layout vector
    tasks = _upload_tasks(rows, dev)
    Kbar = Kbar.contiguous(); ceil_c = ceil_c.contiguous()
    sp = spread.contiguous() if spread is not None else ceil_c
    E = E.contiguous()
    nb = E.shape[0]
    NC = Dh + _NC_EXTRA
    outs = []
    for f in range(2):
        if f < nf:
            outs.append((torch.empty((T, M), dtype=torch.long, device=dev), torch.empty((T,), dtype=torch.long, device=dev),
                         torch.empty((T, nb), dtype=torch.long, device=dev),
                         torch.empty((T, lay["Ntot"], NC), dtype=torch.float32, device=dev),
                         torch.empty((T, lay["Ltot"]), dtype=torch.long, device=dev)))
        else:
            outs.append(outs[0])
    t0, t1 = _tau32(taus[0]), _tau32(taus[-1])
    P, I32 = ctypes.c_void_p, ctypes.c_int
    (a0, m0, b0, n0, l0), (a1, m1, b1, n1, l1) = outs
    nbig = sum(1 for k in lay["ks"] if k >= 10)
    nctl = (nflags + 3) // 4 * 4
    work = torch.empty((nctl + nbig * nf * T * M,), dtype=torch.int32, device=dev)
    ctl, scr = work.data_ptr(), work.data_ptr() + 4 * nctl
    _launch(pro, nf * T, 1024, 0, dev, [P(ceil_c.data_ptr()), P(t0.data_ptr()), P(t1.data_ptr()), P(E.data_ptr()),
                                        P(a0.data_ptr()), P(a1.data_ptr()), P(m0.data_ptr()), P(m1.data_ptr()),
                                        P(b0.data_ptr()), P(b1.data_ptr()), P(ctl), I32(nctl), I32(T), I32(M), I32(H),
                                        I32(nb)])
    nrec = len(rows)
    _launch(bld, nrec, _THREADS, _smem_bytes(Dh), dev,
            [P(tasks.data_ptr()), P(layv.data_ptr()), P(Kbar.data_ptr()), P(ceil_c.data_ptr()),
             P(sp.data_ptr()), P(a0.data_ptr()), P(a1.data_ptr()), P(m0.data_ptr()), P(m1.data_ptr()),
             P(n0.data_ptr()), P(n1.data_ptr()), P(l0.data_ptr()), P(l1.data_ptr()), P(scr),
             P(ctl), I32(M), I32(lay["Ntot"]), I32(lay["Ltot"]), I32(int(spread is not None)), I32(nbig),
             I32(T)])
    res = []
    for f in range(nf):
        adm, mt, mb, node, leaf = outs[f]
        F = {"adm": adm, "m_tot": mt, "M": M, "T": T, "Dh": Dh, "ks": list(lay["ks"]), "chunks": {},
             "Kmax": lay["Kmax"], "node": node, "leaf": leaf, "Ntot": lay["Ntot"], "Ltot": lay["Ltot"],
             "bases": dict(lay["bases"]), "nodes_per": dict(lay["nodes_per"]), "leafbases": dict(lay["leafbases"]),
             "Cmaxs": dict(lay["Cmaxs"])}
        res.append((F, mb))
    return res


DIRTY_KMAX = 9                                            # tallest chunk lean_dirty_build rebuilds (one-CTA batches)


@torch.no_grad()
def lean_dirty_build(Kbar: torch.Tensor, ceil_c: torch.Tensor, spread, taus, E: torch.Tensor, nn: torch.Tensor,
                     lean, skip_mask: int = 0) -> bool:
    # _chunks_build's lean single-forest path for a decode ingest whose dirty chunk heights the device decides from the
    # append's admitted counts nn (fsb_prep_lean_dirty): records for every height up to min(Kmax, DIRTY_KMAX), the
    # clean ones empty, then the build and the spines; three launches and no host read, so a block change can be
    # enqueued while earlier decode steps still run. The rows it writes are those the host-decided path writes, with
    # the same bits; the caller guarantees that no taller chunk changes, and sets skip_mask's bit k only where the
    # arena's rows of height k were built with this layout's fold flag (lean_fold_mask of M = lean[7]): there a chunk
    # that stays in place keeps its rows, elsewhere it is rebuilt as the host-decided path rebuilds it. Kbar, ceil_c,
    # spread, taus, E and lean as forests_cuda gets them for that path (lean with its spine outputs and task buffer).
    # False (nothing launched) where the CUDA build does not cover the inputs.
    T, _, Dh = Kbar.shape
    M = lean[7]
    dev = Kbar.device
    amu0, anx0, leaf0, _, _, jb0, Kmax_lean, _, sp_node, sp_root, sp_hgt, task_buf = lean[:12]
    nk = min(int(Kmax_lean), DIRTY_KMAX) + 1
    if not (T > 0 and len(taus) == 1 and task_buf is not None and task_buf.dtype == torch.int32
            and nk * T * 10 + 40 <= task_buf.numel() and nn.dtype == torch.int64 and nn.is_contiguous()
            and nn.numel() == T and jb0 is not None and sp_node is not None
            and build_cuda_ok(Kbar, ceil_c, spread, taus, E)):
        return False
    _, fold_mask, nbig = _lean_layout_info(M, _group())
    _functions(dev, Dh)
    fns = _fns[_variant(dev.index, Dh)]
    bld_lean16, sp_lean16, prep_dirty = fns[5], fns[6], fns[8]
    Kbar, ceil_c, sp, adm0, mt0, M_kb, M_ce, M_spr, M_adm0 = _lean_operands(Kbar, ceil_c, spread, lean)
    P, I32 = ctypes.c_void_p, ctypes.c_int
    nrec = nk * T
    tasks_ptr = task_buf.data_ptr()
    lay_ptr = tasks_ptr + 40 * nrec
    _launch(prep_dirty, 1, 256, 0, dev,
            [P(tasks_ptr), P(lay_ptr), P(jb0.data_ptr()), P(mt0.data_ptr()), P(nn.data_ptr()), I32(nk), I32(T),
             I32(fold_mask), I32(int(Kmax_lean)), I32(int(M_adm0)), I32(int(M_kb)), I32(int(M_ce)), I32(int(M_spr)),
             I32(int(skip_mask))])
    _launch(bld_lean16, nrec, _THREADS, _smem_bytes(Dh), dev,
            [P(tasks_ptr), P(lay_ptr), P(Kbar.data_ptr()), P(ceil_c.data_ptr()),
             P(sp.data_ptr()), P(adm0.data_ptr()), P(0), P(mt0.data_ptr()), P(mt0.data_ptr()),
             P(amu0.data_ptr()), P(amu0.data_ptr()), P(anx0.data_ptr()), P(anx0.data_ptr()),
             P(leaf0.data_ptr()), P(leaf0.data_ptr()), P(0),
             P(0), I32(M), I32(amu0.shape[1]), I32(leaf0.shape[1]), I32(int(spread is not None)), I32(nbig),
             I32(T), P(0), P(0)])
    _launch(sp_lean16, (T + 15) // 16, _THREADS, 0, dev,
            [P(amu0.data_ptr()), P(anx0.data_ptr()), P(mt0.data_ptr()), P(sp_node.data_ptr()),
             P(sp_root.data_ptr()), P(sp_hgt.data_ptr()), I32(T), I32(amu0.shape[1]), I32(int(Kmax_lean))])
    return True


@torch.no_grad()
def build_forests_cuda(Kbar: torch.Tensor, ceil_c: torch.Tensor, spread, taus, E: torch.Tensor, sel=None,
                       _kmax_cuda: int = 99, lean=None, mb_max=None):
    # forests for admission thresholds taus (1 or 2 tensors of shape (H,), head t uses tau[t % H]); returns
    # [(F, m_b_all)] per tau: build_forest's prefill dict (chunks empty), except the node tables land split — means
    # in bf16 (node_mu, with each row's bf16 rounding error folded into the radius in node_aux[..., 0]) and the nine
    # aggregates in fp32 (node_aux) — and searchsorted(adm, E). T = 0 heads give empty tables of the same layout,
    # without a launch. sel = [(k, [j per head or None])] builds only those dirty aligned chunks (a decode ingest's;
    # node/leaf rows outside them are not written); an out-of-layout selection falls through to the whole-forest build.
    # mb_max (whole-forest builds only): a host bound on every m_b the caller's spines will use (each m_b is at most
    # min(max(E), M)); the build then writes only the chunks such spines can read (spine_chunks), each with the bits of
    # the full build, and leaves the other node/leaf rows unwritten.
    if sel is not None and Kbar.shape[0] > 0 and any(j is not None for _, js in sel for j in js):
        try:
            return _chunks_build(Kbar, ceil_c, spread, taus, E, sel, _kmax_cuda, lean=lean)
        except _SelLayout:
            if lean is not None and Kbar.shape[1] < lean[7]:
                T, M_vis, Dh = Kbar.shape
                M_pad = lean[7]
                Kp, cp = Kbar.new_zeros(T, M_pad, Dh), ceil_c.new_full((T, M_pad), float("-inf"))
                Kp[:, :M_vis] = Kbar; cp[:, :M_vis] = ceil_c
                Kbar, ceil_c = Kp, cp
                if spread is not None:
                    sp = spread.new_zeros(T, M_pad)
                    sp[:, :M_vis] = spread
                    spread = sp
    T, M, Dh = Kbar.shape

    dev = Kbar.device
    nf = len(taus)
    H = taus[0].shape[0]
    lay = forest_layout(M)
    small = None
    if T > 0:
        capturing = torch.cuda.is_current_stream_capturing()
        if small_on() and _small_fits(M, mb_max) and _variant(dev.index, Dh) not in _small_failed:
            if capturing and _variant(dev.index, Dh) not in _small_fns:
                raise RuntimeError("fsb_build_cuda: kernels must be compiled before graph capture")
            small = _small_function(dev, Dh)
        kmin = min(_SMALL_KMAX, _kmax_cuda) if small is not None else -1
        tasks, layv, bounds, nbig, nflags = _tasks(T, M, nf, _group(), dev, _kmax_cuda, mb_max=mb_max, kmin=kmin)
        units, ncta = _small_units(T, M, nf, dev, mb_max, kmin) if small is not None else (None, 0)
        if capturing and _variant(dev.index, Dh) not in _fns:
            raise RuntimeError("fsb_build_cuda: kernels must be compiled before graph capture")
        pro = _functions(dev, Dh)[0]
    Kbar = Kbar.contiguous(); ceil_c = ceil_c.contiguous()
    sp = spread.contiguous() if spread is not None else ceil_c
    E = E.contiguous()
    nb = E.shape[0]
    outs = []
    for f in range(2):
        if f < nf:
            outs.append((torch.empty((T, M), dtype=torch.long, device=dev), torch.empty((T,), dtype=torch.long, device=dev),
                         torch.empty((T, nb), dtype=torch.long, device=dev),
                         torch.empty((T, lay["Ntot"], Dh), dtype=torch.bfloat16, device=dev),
                         torch.empty((T, lay["Ntot"], _NC_EXTRA), dtype=torch.float32, device=dev),
                         torch.empty((T, lay["Ltot"]), dtype=torch.long, device=dev)))
        else:
            outs.append(outs[0])
    if T > 0:
        t0, t1 = _tau32(taus[0]), _tau32(taus[-1])
        P, I32 = ctypes.c_void_p, ctypes.c_int
        (a0, m0, b0, u0, x0, l0), (a1, m1, b1, u1, x1, l1) = outs
        bld16 = _fns[_variant(dev.index, Dh)][7]
        # this call's int32 work buffer, from the caching allocator like the outputs (a graph capture owns its own, and
        # calls on different streams never share one): the build kernel's control words (the completion flags of the G
        # nodes), cleared by the prologue, then the slot lists of the chunks of height >= 10
        nctl = (nflags + 3) // 4 * 4
        work = torch.empty((nctl + nbig * nf * T * M,), dtype=torch.int32, device=dev)
        ctl, scr = work.data_ptr(), work.data_ptr() + 4 * nctl
        _launch(pro, nf * T, 1024, 0, dev, [P(ceil_c.data_ptr()), P(t0.data_ptr()), P(t1.data_ptr()), P(E.data_ptr()),
                                            P(a0.data_ptr()), P(a1.data_ptr()), P(m0.data_ptr()), P(m1.data_ptr()),
                                            P(b0.data_ptr()), P(b1.data_ptr()), P(ctl), I32(nctl), I32(T), I32(M), I32(H),
                                            I32(nb)])
        if ncta > 0:
            _launch(small, ncta, _THREADS, _small_smem_bytes(Dh), dev,
                    [P(units.data_ptr()), P(layv.data_ptr()), P(Kbar.data_ptr()), P(ceil_c.data_ptr()),
                     P(sp.data_ptr()), P(a0.data_ptr()), P(a1.data_ptr()), P(m0.data_ptr()), P(m1.data_ptr()),
                     P(u0.data_ptr()), P(u1.data_ptr()), P(x0.data_ptr()), P(x1.data_ptr()),
                     P(l0.data_ptr()), P(l1.data_ptr()), I32(M), I32(lay["Ntot"]), I32(lay["Ltot"]),
                     I32(int(spread is not None)), I32(T)])
        for off, cnt in bounds:
            _launch(bld16, cnt, _THREADS, _smem_bytes(Dh), dev,
                    [P(tasks.data_ptr() + 40 * off), P(layv.data_ptr()), P(Kbar.data_ptr()), P(ceil_c.data_ptr()),
                     P(sp.data_ptr()), P(a0.data_ptr()), P(a1.data_ptr()), P(m0.data_ptr()), P(m1.data_ptr()),
                     P(u0.data_ptr()), P(u1.data_ptr()), P(x0.data_ptr()), P(x1.data_ptr()),
                     P(l0.data_ptr()), P(l1.data_ptr()), P(scr),
                     P(ctl), I32(M), I32(lay["Ntot"]), I32(lay["Ltot"]), I32(int(spread is not None)), I32(nbig),
                     I32(T)])
    res = []
    for f in range(nf):
        adm, mt, mb, mu, aux, leaf = outs[f]
        F = {"adm": adm, "m_tot": mt, "M": M, "T": T, "Dh": Dh, "ks": list(lay["ks"]), "chunks": {},
             "Kmax": lay["Kmax"], "node_mu": mu, "node_aux": aux, "leaf": leaf, "Ntot": lay["Ntot"], "Ltot": lay["Ltot"],
             "bases": dict(lay["bases"]), "nodes_per": dict(lay["nodes_per"]), "leafbases": dict(lay["leafbases"]),
             "Cmaxs": dict(lay["Cmaxs"])}
        res.append((F, mb))
    return res
