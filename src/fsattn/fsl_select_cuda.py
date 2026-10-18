"""FS-L prefill cell selection as a few CUDA kernels (FSL_FAST_SELECT=1, the default; 0 or 'legacy' keeps the torch
glue + Triton path in FSAttn._cell_score_On_chunked_strict).

Same selected cells, bit for bit, as the fused-select path (FS_FUSED_ATTN=1, FS_FUSED_SELECT=1) under bf16 autocast.
Per layer, for the bench config (32-token cells, head_dim 64, pool 64 + window 8):

* fsl_cellstats: each cell's bf16 mean and fp32 ceiling (max key norm) in one pass, in the summation orders torch's
  reduce kernels use for K.mean(dim=3) and K.float().norm(dim=-1).amax(dim=-1) on these views (fsl_ceil + torch's
  mean for other cell sizes). cache_cell_stats gives FSKVCache.prefill the same summaries.
* fsl_topp / fsl_topp_par: the top-P visible cells of every query block by ceiling, as torch.topk of the masked
  prefix returns them. topk gathers the picks above the k-th value, then those at it, each in index order, and sorts
  them: k > 32 with a stable radix sort (ties stay in index order), k <= 32 with an unstable bitonic network, which is
  replayed. Blocks that see fewer than P cells get the masked cells in index order after the visible ones, as topk of
  the -1e9 fill does. One warp per head walks the cells (short prompts) or one CTA per head splits them into segments.
* fsl_fused (the full pool): candidate slots (topP, then the Wc nearest cells; visibility; first occurrence), the
  content GEMM on tensor cores (four chained mma.m16n8k16 from zero: the bits of the cuBLAS kernels torch.matmul
  picks for this GEMM at batch counts >= 2 on an A100, and of tl.dot; a self-check against torch.matmul at each new
  batch count keeps the cuBLAS content wherever they differ), the score of the Triton select kernel instruction for
  instruction
  (fma(lam*ceil, erf(div.full(|qpos - cen|, 2 sig)), bf16(content * scale)) with libdevice's ftz erf, see
  fsb_beam_cuda.erf_tl) and its top-k order (score descending, slot ascending), written straight into sel_full.
  erf is skipped where every query of the block is far enough from the cell for it to be exactly 1.0f (tl.erf(x) ==
  1.0f for every float x >= 0x407ad8e6, checked exhaustively; the test keeps a 42-ulp margin over div.full's 2-ulp
  error); those slots score x + lam*ceil in a branch-free pass, the rest are inserted after it in (score, slot) order.
* otherwise (pool not filled yet, other shapes, FSL_FUSED_CONTENT=0): fsl_prep builds the slots, the bf16 candidate
  means and the block's Q rows, torch.matmul computes the content (the legacy cuBLAS call on the legacy operands) and
  fsl_select / fsl_select_g score and select.

The kernels are compiled with NVRTC at first use (fsb_beam_cuda's loader); without NVRTC the legacy path runs (and
the KV cache the torch ops), with a warning. Under CUDA-graph capture no kernel is compiled and no cached tensor is
handed to a kernel (a graph keeps only addresses): the calls run on the capture's own allocations. Under
torch.compile neither path is taken (each ctypes launch would be a graph break): the legacy path and the torch ops.
"""
from __future__ import annotations

import ctypes
import os
import struct
import threading

import torch

from ..utils.compile_guard import compiling
from .fsb_beam_cuda import _HELPERS, _check, _libraries, compile_image

_CMAX = 512                     # candidate slots per block (P + window) handled by the kernels
XS_SAFE_BITS = 0x407AD910       # erf-skip threshold of the kernels (XS_SAFE): tl.erf(x) == 1.0f for x >= 0x407ad8e6
_TILE_MAX = 48 * 1024           # dynamic shared memory per CTA without the opt-in attribute
from pathlib import Path

_SRC = Path(__file__).with_name("fsl_select.cu").read_text(encoding="utf-8")

_lock = threading.Lock()
_mods: dict = {}
# a failure to compile or load the kernels turns off only the fast path that hit it, for the rest of the process,
# with one warning: the selection (legacy path from then on) and the KV cache's cell statistics (torch ops)
_failed_select: list = []
_failed_stats: list = []
_NAMES = (
    "fsl_ceil", "fsl_cellstats", "fsl_topp", "fsl_topp_par",
    "fsl_prep", "fsl_select", "fsl_select_g", "fsl_dense_decode_topk",
)
_DEBUG: list | None = None      # tests: a list collects (b0, bf16 content, path) of every chunk


def fast_select_on() -> bool:
    return os.environ.get("FSL_FAST_SELECT", "1") not in ("0", "legacy") and not _failed_select


def fused_content_on() -> bool:
    return os.environ.get("FSL_FUSED_CONTENT", "1") not in ("0", "legacy")


def fast_cache_stats_on() -> bool:
    return os.environ.get("FSL_FAST_CACHE_STATS", "1") not in ("0", "legacy") and not _failed_stats


def _latch(failed: list, what: str, e: BaseException) -> None:
    if not failed:
        failed.append(repr(e))
        import warnings
        warnings.warn(f"{what}: {e!r}")


def status() -> dict:
    """What the FS-L fast paths can do in this process, for run metadata: the switches as resolved now, the failure
    that turned a path off (None: no failure) and the fused-content self-check results (False: that configuration
    takes the cuBLAS content path)."""
    return {"fast_select": fast_select_on(), "fused_content": fused_content_on(),
            "fast_cache_stats": fast_cache_stats_on(),
            "select_failure": _failed_select[0] if _failed_select else None,
            "stats_failure": _failed_stats[0] if _failed_stats else None,
            "fused_content_checks": [dict(zip(("device", "slots", "Bq", "blas", "batch"), k), exact=v)
                                     for k, v in _fused_checked.items()]}


def _source(per: int, kk: int, ntl: int) -> str:
    d = (f"#define PER {per}\n#define KK {kk}\n#define CMAX {_CMAX}\n#define XS_SAFE {XS_SAFE_BITS:#x}u\n"
         + (f"#define NTL {ntl}\n" if ntl else ""))
    return d + _HELPERS + _SRC


def _functions(device: torch.device, per: int, kk: int, ntl: int = 0):
    key = (device.index, per, kk, ntl)
    fns = _mods.get(key)
    if fns is None:
        with _lock, torch.cuda.device(device):
            fns = _mods.get(key)
            if fns is None:
                _, cuda = _libraries()
                torch.cuda.current_stream(device)
                ctx = ctypes.c_void_p()
                _check(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
                if not ctx.value:
                    raise RuntimeError("no current CUDA context")
                maj, mnr = torch.cuda.get_device_capability(device)
                image = compile_image(_source(per, kk, ntl), 10 * maj + mnr)
                mod = ctypes.c_void_p()
                _check(cuda.cuModuleLoadData(ctypes.byref(mod), image), "cuModuleLoadData")
                fs = {}
                for nm in _NAMES + (("fsl_fused", "fsl_dense_fused") if ntl else ()):
                    f = ctypes.c_void_p()
                    _check(cuda.cuModuleGetFunction(ctypes.byref(f), mod, nm.encode()), "cuModuleGetFunction")
                    fs[nm] = f
                fns = _mods[key] = (fs, mod, image)
    return fns[0]


_PARAM_BUF, _PARAM_SIZE, _PARAM_END = 1, 2, 0         # CU_LAUNCH_PARAM_BUFFER_POINTER / _SIZE / _END


def _launch(fn, grid: int, block: int, fmt: str, vals, smem: int = 0, dev: int | None = None) -> None:
    # kernel arguments packed in one buffer (struct codes P pointer, q int64, i int32, f float32; native alignment is
    # the kernel parameter layout), on the current torch stream of device dev (default: the current device)
    _, cuda = _libraries()
    buf = struct.pack("@" + fmt, *vals)
    cbuf = ctypes.create_string_buffer(buf, len(buf))
    size = ctypes.c_size_t(len(buf))
    extra = (ctypes.c_void_p * 5)(_PARAM_BUF, ctypes.addressof(cbuf), _PARAM_SIZE, ctypes.addressof(size), _PARAM_END)
    stream = torch._C._cuda_getCurrentRawStream(torch.cuda.current_device() if dev is None else dev)
    _check(cuda.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, smem, ctypes.c_void_p(stream), None,
                               ctypes.addressof(extra)), "cuLaunchKernel")


def _autocast_bf16() -> bool:
    try:
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
    except (TypeError, AttributeError):                                  # torch < 2.4
        return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16


def fast_select_ok(Q: torch.Tensor, K: torch.Tensor, Pk: int, Wc: int, kk: int, Bq: int, N: int) -> bool:
    if compiling() or not fast_select_on() or not (Q.is_cuda and K.is_cuda) or Q.device != K.device:
        return False
    if Q.dtype != torch.bfloat16 or K.dtype != torch.bfloat16 or Q.shape[-1] % 8 != 0:
        return False
    if Q.device.index != torch.cuda.current_device():
        return False
    if not _autocast_bf16():
        return False                          # the legacy content matmul runs in bf16 only under bf16 autocast
    C = Pk + Wc
    return 0 < Pk <= 128 and 0 <= Wc and C <= _CMAX and 1 <= kk <= 8 and kk <= C and Bq <= 128 and N < (1 << 24)


def dense_fast_select_ok(Q: torch.Tensor, K: torch.Tensor, kk: int, Bq: int, s: int, N: int) -> bool:
    if compiling() or not fast_select_on() or not fused_content_on():
        return False
    if not (Q.is_cuda and K.is_cuda) or Q.device != K.device:
        return False
    if Q.dtype != torch.bfloat16 or K.dtype != torch.bfloat16 or Q.shape[-1] != 64:
        return False
    if Q.device.index != torch.cuda.current_device() or not _rows16(Q) or not _rows16(K):
        return False
    if not _autocast_bf16():
        return False
    return 1 <= kk <= 8 and 1 <= Bq <= 128 and s > 0 and N < (1 << 24)


def prefix_topp(fs, ceil: torch.Tensor, BH: int, M: int, nb: int, Bq: int, s: int, Pk: int, par=None) -> torch.Tensor:
    # (BH, nb, Pk) int32: the top-Pk visible cells of every query block, as topk of the masked prefix returns them.
    # Small M: one warp per head walks the cells; else one CTA per head, warps on segments of >= 64 cells.
    topp = torch.empty(BH, nb, Pk, dtype=torch.int32, device=ceil.device)
    dev = ceil.device.index
    ps = 32 * ((Pk + 31) // 32)
    gmax = min(32, (_TILE_MAX // 4 - 2 * ps) // (4 * ps + 66))
    if par is None:
        par = M >= 256 and gmax >= 2
    if par:
        S = max(64, 32 * -(-M // (32 * gmax)))
        G = -(-M // S)
        _launch(fs["fsl_topp_par"], BH, 32 * G, "PPqqqqiq", (ceil.data_ptr(), topp.data_ptr(), M, nb, Bq, s, Pk, S),
                smem=(4 * G * ps + 2 * ps + 66 * G) * 4, dev=dev)
    else:
        _launch(fs["fsl_topp"], BH, 32, "PPqqqqi", (ceil.data_ptr(), topp.data_ptr(), M, nb, Bq, s, Pk), dev=dev)
    return topp


def cell_stats(fs, K: torch.Tensor, M: int, s: int):
    """(kbar, ceil) of the first M cells of K (B,H,>=M*s,D) bf16: kbar (B,H,M,D) bf16 = K-cells.mean(dim=3) and
    ceil (B,H,M) fp32 = K-cells.float().norm(dim=-1).amax(dim=-1), bit for bit; None when the kernel does not apply."""
    B, H, _, D = K.shape
    if s != 32 or D != 64 or K.dtype != torch.bfloat16 or not _rows16(K) or M <= 0:
        return None
    kbar = torch.empty(B, H, M, D, dtype=K.dtype, device=K.device)
    ceil = torch.empty(B, H, M, dtype=torch.float32, device=K.device)
    _launch(fs["fsl_cellstats"], (B * H * M * 32 + 127) // 128, 128, "PPPqqqqqq",
            (K.data_ptr(), ceil.data_ptr(), kbar.data_ptr(), H, M, K.stride(0), K.stride(1), K.stride(2), B * H * M),
            dev=K.device.index)
    return kbar, ceil


def cache_cell_stats(K: torch.Tensor, ncell: int, s: int):
    """FSKVCache.prefill's cell summaries of K (B,H,N,D) bf16, bit for bit: (Kc.mean(dim=3).float(),
    Kc.float().norm(dim=-1).amax(dim=-1)) for Kc = the first ncell complete cells; None when the kernel does not
    apply (the cache then runs the torch ops). FSL_FAST_CACHE_STATS=0 turns it off."""
    if (compiling() or not fast_cache_stats_on() or not K.is_cuda or s != 32 or K.shape[-1] != 64
            or K.dtype != torch.bfloat16 or K.device.index != torch.cuda.current_device()
            or torch.is_grad_enabled() and K.requires_grad):
        return None
    key = next((k for k in _mods if k[0] == K.device.index), None)
    try:
        if key is None and torch.cuda.is_current_stream_capturing():
            return None
        fs = _mods[key][0] if key is not None else _functions(K.device, 2, 4)
    except Exception as e:                    # no NVRTC / driver: torch ops
        _latch(_failed_stats, "FS-L fast KV-cache cell statistics unavailable, using the torch ops", e)
        return None
    st = cell_stats(fs, K, ncell, s)
    return None if st is None else (st[0].float(), st[1])


_hi_cache: dict = {}


def _near_hi(dev, nb: int, Bq: int, s: int, capturing: bool) -> torch.Tensor:
    # the last visible cell of every block as the legacy glue computes it, floor(blk_start / s) - 1 in fp32 (the same
    # torch ops, once per shape and stream instead of once per layer).
    # A CUDA graph keeps only the address a kernel reads, and a cached tensor can be evicted (the cache clears itself)
    # and its block reused while the graph is still replayed: a capture gets its own tensor, from the graph's memory
    # pool, which lives as long as the graph. The cache is per stream: a stream-ordered allocator frees and reuses a
    # block in the order of the stream that allocated it, and a kernel on another stream could read the tensor before
    # the stream computing it has written it.
    if capturing:
        return (torch.arange(nb, device=dev, dtype=torch.float32) * Bq / s).floor().long() - 1
    key = (dev.index, torch._C._cuda_getCurrentRawStream(dev.index), nb, Bq, s)
    hi = _hi_cache.get(key)
    if hi is None:
        if len(_hi_cache) >= 64:
            _hi_cache.clear()
        hi = _hi_cache[key] = (torch.arange(nb, device=dev, dtype=torch.float32) * Bq / s).floor().long() - 1
    return hi


def _rows16(t: torch.Tensor) -> bool:
    # (B,H,N,D) bf16 view whose rows are 16-byte aligned and contiguous: the kernels read them as uint4
    return (t.stride(-1) == 1 and t.data_ptr() % 16 == 0
            and all(t.stride(i) % 8 == 0 for i in range(t.dim() - 1)))


def _fused_smem(C: int) -> int:
    return (max((128 + C) * 72 * 2, 128 * (C + 2) * 2) + 15) // 16 * 16 + 24 * C + 16


# ---- runtime guard of the fused content: its bits equal those of the legacy torch.matmul only where the cuBLAS
# kernel accumulates the products as four chained mma.m16n8k16 do. cuBLAS picks the kernel by the GEMM's batch count
# (B*H*blocks of a chunk); on an A100 with the cu121 cuBLAS: 1 slice-K (different bits), 2-16 WMMA, 32-512 s16816,
# >= 1024 s1688 (same bits), and another GPU or cuBLAS may differ anywhere. So the first time a device meets a batch
# count (counts >= _CHECK_CAP share one check, at _CHECK_CAP: one regime on the A100 up to 16384), fsl_fused's content
# is compared bit for bit with torch.matmul's at that count (the cuBLAS call of the FSL_FUSED_CONTENT=0 path, same
# operands and layout), on inputs whose rounding depends on how the products are grouped; a call whose counts are not
# all exact takes the cuBLAS content path.
_fused_checked: dict = {}       # (device, C, Bq, cuBLAS settings, batch count) -> fsl_fused's content is torch.matmul's
_CHECK_CAP = 1024


def _blas_settings():
    # the switches that steer which cuBLAS kernel torch.matmul runs for the bf16 content GEMM
    try:
        return (bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
                str(torch.backends.cuda.preferred_blas_library()))
    except (AttributeError, RuntimeError):                               # older torch
        return None


def _fused_content_exact(fs, dev, C: int, Bq: int, Pk: int, Wc: int, kk: int, counts, capturing: bool) -> bool:
    blas = _blas_settings()
    for nbt in sorted({min(c, _CHECK_CAP) for c in counts}):
        key = (dev.index, C, Bq, blas, nbt)
        ok = _fused_checked.get(key)
        if ok is None:
            if capturing:
                return False                  # no check inside a graph capture: this call takes the cuBLAS content
            try:
                bad = _fused_content_mismatches(fs, dev, C, Bq, Pk, Wc, kk, (nbt,))
                why = f"{bad} of {nbt * Bq * C} content values differ"
            except Exception as e:           # e.g. out of memory: exactness first
                bad, why = -1, f"the check failed: {e!r}"
            ok = _fused_checked[key] = bad == 0
            if not ok:
                import warnings
                warnings.warn(f"FS-L fused content differs from torch.matmul on {torch.cuda.get_device_name(dev)} at "
                              f"GEMM batch count {nbt} ({why}): such calls take the cuBLAS content path (exact, "
                              f"slower)")
        if not ok:
            return False
    return True


@torch.no_grad()
def _fused_content_mismatches(fs, dev, C: int, Bq: int, Pk: int, Wc: int, kk: int, batches) -> int:
    """bf16 content values where fsl_fused (dbg output) and torch.matmul on fsl_prep's gathered candidate means differ,
    summed over GEMM batch counts `batches` (each: one head, that many query blocks of Bq rows, random candidates).
    Extra memory: ~74 KB per block of the largest count (76 MB at _CHECK_CAP)."""
    D, M, R = 64, 1024, 4096
    di = dev.index
    g = torch.Generator(device=dev)
    g.manual_seed(0x5E1EC7)                   # private generator: the caller's random stream is untouched

    def wide(rows: int, mirror: int) -> torch.Tensor:
        # per-element exponents over +-12 binades: a dot product's fp32 sum rounds at almost every addition, so a
        # different grouping of the products gives different bits; every 4th row has halves equal (mirror 1) or
        # opposite (mirror -1), so Q.K cancels to rounding residue that depends on the order alone
        x = torch.randn(rows, D, device=dev, generator=g)
        x = x * torch.exp2(torch.randint(-12, 13, (rows, D), device=dev, generator=g, dtype=torch.int32).float())
        x[::4, D // 2:] = x[::4, :D // 2] * mirror
        return x.to(torch.bfloat16)

    qpool = wide(R, 1)
    kbar = wide(M, -1).view(1, 1, M, D)
    ceil = torch.rand(1, 1, M, device=dev, generator=g)
    one = torch.ones(1, device=dev)
    bad = 0
    for nbt in batches:
        N = nbt * Bq
        Q = qpool.repeat((N + R - 1) // R, 1)[:N].view(1, 1, N, D)
        topp = torch.randint(0, M, (1, nbt, Pk), device=dev, generator=g, dtype=torch.int32)
        hi = torch.randint(max(Wc, 1) - 1, M, (nbt,), device=dev, generator=g)
        sel = torch.empty(1, 1, N, kk, dtype=torch.int32, device=dev)
        dbg = torch.empty(nbt, Bq, C, dtype=torch.bfloat16, device=dev)
        _launch(fs["fsl_fused"], nbt, 256, "PPPPPPPPPqqiqiqiiqqqqqfqii",
                (topp.data_ptr(), hi.data_ptr(), kbar.data_ptr(), Q.data_ptr(), ceil.data_ptr(), one.data_ptr(),
                 one.data_ptr(), sel.data_ptr(), dbg.data_ptr(), nbt, M, nbt, 0, Bq, 32, Pk, Wc, 1, N, Q.stride(0),
                 Q.stride(1), Q.stride(2), 1.0, N, 0, 0), smem=_fused_smem(C), dev=di)
        cmeta = torch.empty(1, nbt, C, dtype=torch.int32, device=dev)
        kc = torch.empty(1, 1, nbt, C, D, dtype=torch.bfloat16, device=dev)
        _launch(fs["fsl_prep"], nbt, 128, "PPPPPPPqqiqqqiiiqqqqq",
                (topp.data_ptr(), hi.data_ptr(), kbar.data_ptr(), 0, cmeta.data_ptr(), kc.data_ptr(), 0, nbt, M, nbt,
                 0, Bq, 32, Pk, Wc, D, 1, N, Q.stride(0), Q.stride(1), Q.stride(2)), dev=di)
        content = torch.matmul(Q.view(1, 1, nbt, Bq, D), kc.transpose(-1, -2)).reshape(nbt, Bq, C).view(torch.int16)
        if not torch.equal(content, dbg.view(torch.int16)):   # (a count would cast the mask to int64: 8 B a value)
            bad += sum(int(torch.ne(content[i:i + 64], dbg.view(torch.int16)[i:i + 64]).sum())
                       for i in range(0, nbt, 64))
        del Q, topp, hi, sel, dbg, cmeta, kc, content
    return bad


@torch.no_grad()
def fsl_fast_select(Q, Kp, *, N, M, s, Bq, nb, NBCH, Pk, Wc, kk, lam, sigv, scale, content_only, pool=None,
                    stats_out=None):
    """sel_full (B,H,nb*Bq,kk) int32 of the fused-select path. Q: (B,H,N,D) bf16, Kp: K zero-padded to M*s tokens;
    pool: the configured candidate pool P (the fused content kernel is compiled for C = P + Wc only); stats_out: a
    list that receives the cell means (bf16, as torch's mean returns them) and ceilings it computed."""
    B, H, _, D = Q.shape
    dev = Q.device
    C = Pk + Wc
    per = (Pk + 31) // 32
    qrows = _rows16(Q)
    smem_f = _fused_smem(C)
    fused = (fused_content_on() and D == 64 and qrows and C % 8 == 0 and smem_f <= _TILE_MAX and Bq <= 128
             and (pool is None or Pk == pool))
    ntl = C // 8 if fused else 0
    capturing = torch.cuda.is_current_stream_capturing()
    try:
        if capturing and (dev.index, per, kk, ntl) not in _mods:
            return None
        fs = _functions(dev, per, kk, ntl)
    except Exception as e:                    # no NVRTC / driver: legacy path
        _latch(_failed_select, "FS-L fast selection unavailable, using the legacy path", e)
        return None
    # the GEMM batch counts of the cuBLAS content path's chunks: full chunks and the last one
    counts = (B * H * min(NBCH, nb), B * H * (nb - (nb - 1) // NBCH * NBCH))
    if fused and not _fused_content_exact(fs, dev, C, Bq, Pk, Wc, kk, counts, capturing):
        fused = False                         # the cuBLAS content path (this module has its kernels too)
    di = dev.index
    # cell means (bf16, as torch's mean returns them; the legacy fp32 copy rounds back to them in the autocast cast)
    # and ceilings, both bit for bit as the legacy torch ops compute them
    Kpv = Kp.view(B, H, M, s, D)
    _MCH = 16384
    st = cell_stats(fs, Kp, M, s) if s == 32 else None
    if st is not None:
        kbar, ceil = st
    else:
        if M <= _MCH:
            kbar = Kpv.mean(dim=3)
        else:
            kbar = Kp.new_empty(B, H, M, D)
            for m0 in range(0, M, _MCH):
                kbar[:, :, m0:min(m0 + _MCH, M)] = Kpv[:, :, m0:min(m0 + _MCH, M)].mean(dim=3)
        kbar = kbar.contiguous()
    if st is not None:
        pass
    elif D == 64 and _rows16(Kp):
        ceil = torch.empty(B, H, M, dtype=torch.float32, device=dev)
        _launch(fs["fsl_ceil"], (B * H * M * 32 + 127) // 128, 128, "PPqqqqqqq",
                (Kp.data_ptr(), ceil.data_ptr(), H, M, s, Kp.stride(0), Kp.stride(1), Kp.stride(2), B * H * M), dev=di)
    else:
        ceil = torch.empty(B, H, M, dtype=torch.float32, device=dev)
        for m0 in range(0, M, _MCH):
            m1 = min(m0 + _MCH, M)
            ceil[:, :, m0:m1] = Kpv[:, :, m0:m1].float().norm(dim=-1).amax(dim=-1)
    if stats_out is not None:
        stats_out.append((kbar, ceil))
    topp = prefix_topp(fs, ceil, B * H, M, nb, Bq, s, Pk)
    hi = _near_hi(dev, nb, Bq, s, capturing)
    # per-head sel_scale and sigma as given (fp32 or bf16, converted exactly in the kernels)
    lam_f, sig_f = (t.reshape(-1) if t.dtype in (torch.float32, torch.bfloat16) else t.reshape(-1).float()
                    for t in (lam, sigv))
    lam_f, sig_f = lam_f.contiguous(), sig_f.contiguous()
    pflags = (lam_f.dtype == torch.bfloat16) | 2 * (sig_f.dtype == torch.bfloat16)
    Npad = nb * Bq
    sel_full = torch.empty(B, H, Npad, kk, dtype=torch.int32, device=dev)
    if fused:
        for b0 in range(0, nb, NBCH):
            nbc = min(b0 + NBCH, nb) - b0
            dbg = torch.empty(B * H * nbc, Bq, C, dtype=Q.dtype, device=dev) if _DEBUG is not None else None
            _launch(fs["fsl_fused"], B * H * nbc, 256, "PPPPPPPPPqqiqiqiiqqqqqfqii",
                    (topp.data_ptr(), hi.data_ptr(), kbar.data_ptr(), Q.data_ptr(), ceil.data_ptr(), lam_f.data_ptr(),
                     sig_f.data_ptr(), sel_full.data_ptr(), dbg.data_ptr() if dbg is not None else 0, nb, M, nbc, b0,
                     Bq, s, Pk, Wc, H, N, Q.stride(0), Q.stride(1), Q.stride(2), float(scale), Npad,
                     1 if content_only else 0, pflags), smem=smem_f, dev=di)
            if dbg is not None:
                _DEBUG.append((b0, dbg.view(B, H, nbc, Bq, C), "fused"))
        return sel_full
    ru4 = (C // 8) | 1 if C % 8 == 0 else 0                    # odd uint4 row stride: conflict-free 16-byte reads
    tile = ru4 and (Bq * ru4 * 16 + 16 * C) <= _TILE_MAX
    smem = (Bq * ru4 * 16 if tile else 0) + 16 * C
    for b0 in range(0, nb, NBCH):
        b1 = min(b0 + NBCH, nb)
        nbc = b1 - b0
        q16 = torch.empty(B, H, nbc * Bq, D, dtype=Q.dtype, device=dev)
        if not qrows:
            n_real = max(0, min(b1 * Bq, N) - b0 * Bq)
            q16.zero_()
            if n_real > 0:
                q16[:, :, :n_real, :] = Q[:, :, b0 * Bq:b0 * Bq + n_real, :]
        cmeta = torch.empty(B * H, nbc, C, dtype=torch.int32, device=dev)
        kc = torch.empty(B, H, nbc, C, D, dtype=Q.dtype, device=dev)
        _launch(fs["fsl_prep"], B * H * nbc, 128, "PPPPPPPqqiqqqiiiqqqqq",
                (topp.data_ptr(), hi.data_ptr(), kbar.data_ptr(), Q.data_ptr() if qrows else 0, cmeta.data_ptr(),
                 kc.data_ptr(), q16.data_ptr(), nb, M, nbc, b0, Bq, s, Pk, Wc, D, H, N, Q.stride(0), Q.stride(1),
                 Q.stride(2)), dev=di)
        # the legacy content GEMM: same bf16 operands, shapes and strides (so the same cuBLAS kernel and bits)
        content = torch.matmul(q16.view(B, H, nbc, Bq, D), kc.transpose(-1, -2))
        if content.dtype != torch.bfloat16 or not content.is_contiguous():
            raise RuntimeError("FS-L fast selection: unexpected content layout")
        if _DEBUG is not None:
            _DEBUG.append((b0, content, "cublas"))
        _launch(fs["fsl_select" if tile else "fsl_select_g"], B * H * nbc, 128, "PPPPPPiqiiiiqqfqii",
                (content.data_ptr(), cmeta.data_ptr(), ceil.data_ptr(), lam_f.data_ptr(), sig_f.data_ptr(),
                 sel_full.data_ptr(), nbc, b0, Bq, C, ru4 if tile else 0, H, M, s, float(scale), Npad,
                 1 if content_only else 0, pflags), smem=smem, dev=di)
    return sel_full


def _dense_fused_smem() -> int:
    return ((64 * 72 + 128 * 72) * 2 + 15) // 16 * 16 + (8 * 64 + 2) * 4 + 16


@torch.no_grad()
def fsl_dense_fast_select(Q, Kp, *, N, M, s, Bq, nb, kk, lam, sigv, scale, content_only, stats_out=None):
    """Dense exact cell selection (no candidate pool pruning) using tensor-core tiled 64-cell passes per query block."""
    B, H, _, D = Q.shape
    dev = Q.device
    per, ntl = 2, 9
    capturing = torch.cuda.is_current_stream_capturing()
    try:
        if capturing and (dev.index, per, kk, ntl) not in _mods:
            return None
        fs = _functions(dev, per, kk, ntl)
    except Exception as e:
        _latch(_failed_select, "FS dense fast selection unavailable, using the fallback path", e)
        return None
    di = dev.index
    Kpv = Kp.view(B, H, M, s, D)
    _MCH = 16384
    st = cell_stats(fs, Kp, M, s) if s == 32 else None
    if st is not None:
        kbar, ceil = st
    else:
        if M <= _MCH:
            kbar = Kpv.mean(dim=3)
        else:
            kbar = Kp.new_empty(B, H, M, D)
            for m0 in range(0, M, _MCH):
                kbar[:, :, m0:min(m0 + _MCH, M)] = Kpv[:, :, m0:min(m0 + _MCH, M)].mean(dim=3)
        kbar = kbar.contiguous()
        if D == 64 and _rows16(Kp):
            ceil = torch.empty(B, H, M, dtype=torch.float32, device=dev)
            _launch(fs["fsl_ceil"], (B * H * M * 32 + 127) // 128, 128, "PPqqqqqqq",
                    (Kp.data_ptr(), ceil.data_ptr(), H, M, s, Kp.stride(0), Kp.stride(1), Kp.stride(2), B * H * M),
                    dev=di)
        else:
            ceil = torch.empty(B, H, M, dtype=torch.float32, device=dev)
            for m0 in range(0, M, _MCH):
                m1 = min(m0 + _MCH, M)
                ceil[:, :, m0:m1] = Kpv[:, :, m0:m1].float().norm(dim=-1).amax(dim=-1)
    if stats_out is not None:
        stats_out.append((kbar, ceil))
    lam_f, sig_f = (t.reshape(-1) if t.dtype in (torch.float32, torch.bfloat16) else t.reshape(-1).float()
                    for t in (lam, sigv))
    lam_f, sig_f = lam_f.contiguous(), sig_f.contiguous()
    pflags = (lam_f.dtype == torch.bfloat16) | 2 * (sig_f.dtype == torch.bfloat16)
    Npad = nb * Bq
    sel_full = torch.empty(B, H, Npad, kk, dtype=torch.int32, device=dev)
    _launch(fs["fsl_dense_fused"], B * H * nb, 256, "PPPPPPqqiqqqqqqfqii",
            (kbar.data_ptr(), Q.data_ptr(), ceil.data_ptr(), lam_f.data_ptr(), sig_f.data_ptr(),
             sel_full.data_ptr(), nb, M, Bq, s, H, N, Q.stride(0), Q.stride(1), Q.stride(2),
             float(scale), Npad, 1 if content_only else 0, pflags),
            smem=_dense_fused_smem(), dev=di)
    return sel_full


@torch.no_grad()
def fsl_dense_decode_topk(
    q_t: torch.Tensor, Kbar_all: torch.Tensor, ceil_all: torch.Tensor,
    pos_t: torch.Tensor, m_vis_t: torch.Tensor, lam: torch.Tensor, sigv: torch.Tensor,
    part_s: torch.Tensor, part_c: torch.Tensor,
    scale: float, max_cells: int, cell_size: int, k_sel: int, split_m: int, block_c: int,
) -> bool:
    """Coalesced CUDA split-topk kernel for dense FS decode (AUTOCAST_BF16=1, D=64)."""
    if (compiling() or not fast_select_on() or os.environ.get("FSL_DENSE_DECODE_CUDA", "1") == "0"
            or not (1 <= block_c <= 1024 and 1 <= k_sel <= 16)):
        return False
    B, H, _, D = q_t.shape
    dev = q_t.device
    if (D != 64 or dev.index != torch.cuda.current_device()
            or Kbar_all.stride(0) != H * max_cells * 64 or Kbar_all.stride(1) != max_cells * 64
            or Kbar_all.stride(2) != 64 or Kbar_all.stride(3) != 1 or Kbar_all.data_ptr() % 16 != 0
            or ceil_all.stride(0) != H * max_cells or ceil_all.stride(1) != max_cells or ceil_all.stride(2) != 1
            or pos_t.dtype != torch.int64 or m_vis_t.dtype != torch.int32
            or lam.dtype not in (torch.float32, torch.bfloat16)
            or sigv.dtype not in (torch.float32, torch.bfloat16)):
        return False
    key = next((k for k in _mods if k[0] == dev.index), None)
    try:
        if key is None and torch.cuda.is_current_stream_capturing():
            return False
        fs = _mods[key][0] if key is not None else _functions(dev, 2, 4)
    except Exception as e:
        _latch(_failed_select, "FS dense decode CUDA topk unavailable, using Triton fallback", e)
        return False
    pflags = (
        (1 if lam.dtype == torch.bfloat16 else 0)
        | (2 if sigv.dtype == torch.bfloat16 else 0)
        | (4 if q_t.dtype == torch.bfloat16 else 0)
    )
    _launch(
        fs["fsl_dense_decode_topk"], B * H * split_m, 128, "PPPPPPPPPqqqfiiiiii",
        (q_t.data_ptr(), Kbar_all.data_ptr(), ceil_all.data_ptr(),
         pos_t.data_ptr(), m_vis_t.data_ptr(), lam.data_ptr(), sigv.data_ptr(),
         part_s.data_ptr(), part_c.data_ptr(),
         q_t.stride(0), q_t.stride(1), H,
         float(scale), int(max_cells), int(cell_size),
         int(k_sel), int(split_m), int(block_c), int(pflags)),
        dev=dev.index,
    )
    return True


