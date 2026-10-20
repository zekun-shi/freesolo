"""CUDA builds of the fused FS-B beam descent (FSB_BEAM_CUDA=1, the default; 0 keeps the Triton kernel).

fsb_beam_rows (prefill rows; FSB_BEAM_ROWS=1, the default): one block per 64 queries of a (beam, row) that share a
forest, both beams of a stage in one launch (forest_beam_rows_cuda). The block's queries descend together, level by
level: the block forms the union of the children of the nodes its queries expand (their beams overlap, so the union is
about one to two times a query's own 32 children), stages those nodes in shared memory once, and two lanes per query
(each holding half of the query's columns) score the query's children from there. A level whose union outgrows the
shared tables hands the block's queries to the warp-per-query descent, which continues each from that level.

fsb_beam_rows2 (FSB_BEAM_V2=1, default 0): the bf16 tensor-core configuration of fsb_beam_rows (128 queries per block,
width 16) as a lean kernel: per-lane sorting networks on packed (score, child) keys instead of rank counting, register
frontiers, two block barriers per level and warp-local dots. Its selections match fsb_beam_rows except where two
children's scores agree in their top 27 bits (they are then ordered by child index); its 64-bit-key test build
(key64) matches it bitwise.

fsb_beam_cuda (decode, small rows): one warp per query, lane l holds child slot l (frontier slot l // 2); each level
loads the children's node fields once, the dot products of the valid children run as a shuffle reduce-scatter, and
the descent stops at the frontier's fixed point. FSB_BEAM_ROWS=0 uses the same warp-per-query kernel for every row,
one launch per beam (no row-cooperative kernel).

The forest's node tables are split (F["node_mu"] bf16, F["node_aux"] fp32): the kernels read the means straight from
the bf16 slab (an exact upcast: every stored mean is the bf16 value the beams agree on). Both keep _halve_rows'
pairwise order for every dot product, rank children by (score desc, slot asc) and use exactly the instructions
_fsb_beam_kernel compiles to without multiply-add contraction (add.rn / mul.rn / div.full / sqrt.rn.ftz and
libdevice's erff with its ftz FMAs, as inline PTX), so cells and visit counts are bit-identical to the Triton kernel
(tests/test_fsb_beam_cuda.py). The source is compiled at first use with NVRTC, which ships with PyTorch's CUDA
wheels, and launched through the driver API; without NVRTC the Triton kernel runs. A row kernel that fails to build
hands its rows to the warp kernel; a warp kernel that fails to build turns every CUDA beam off.
"""
from __future__ import annotations

import ctypes
import os
import threading

import torch

from ..utils.compile_guard import compiling

from pathlib import Path


def _load_cu_sections(filename: str) -> dict[str, str]:
    txt = Path(__file__).with_name(filename).read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for part in txt.split("// >>> SECTION: ")[1:]:
        header, body = part.split(" <<<\n", 1)
        out[header] = body
    return out


_BEAM_CU = _load_cu_sections("fsb_beam.cu")
_HELPERS = _BEAM_CU["_HELPERS"]
_KERNEL = _BEAM_CU["_KERNEL"]
_WARP_KERNEL = _BEAM_CU["_WARP_KERNEL"]
_ROWS_KERNEL = _BEAM_CU["_ROWS_KERNEL"]
_ROWS2_KERNEL = _BEAM_CU["_ROWS2_KERNEL"]

_NWARP = 4                                               # queries (warps) per block
_lock = threading.Lock()
_funcs: dict = {}
_mods: dict = {}
_beam_fns: dict = {}
_libs: dict = {}
_failed: list = []                                       # the warp kernel failed to build: every CUDA beam is off
_rows_failed: list = []                                  # the row kernel failed to build: its rows use the warp kernel
_rows2_failed: list = []                                 # the lean row kernel failed to build: fsb_beam_rows runs


def beam_cuda_on() -> bool:
    return os.environ.get("FSB_BEAM_CUDA", "1") == "1" and not _failed


def rows_flag() -> bool:
    # FSB_BEAM_ROWS (default 1): the row-cooperative prefill kernel, and the current warp-per-query kernel elsewhere;
    # 0 (and torch.compile) uses the warp-per-query kernel for every row, one launch per beam
    return not compiling() and os.environ.get("FSB_BEAM_ROWS", "1") == "1"


def _nvrtc_paths():
    paths = []
    try:
        import nvidia.cuda_nvrtc as m                                   # PyTorch's pip CUDA wheels
        for d in m.__path__:
            lib = os.path.join(d, "lib")
            if os.path.isdir(lib):
                paths += [os.path.join(lib, f) for f in sorted(os.listdir(lib)) if f.startswith("libnvrtc")]
    except Exception:
        pass
    major = (torch.version.cuda or "12").split(".")[0]
    return paths + [f"libnvrtc.so.{major}", "libnvrtc.so"]


def _libraries():
    if "cuda" not in _libs:
        nvrtc = None
        for p in _nvrtc_paths():
            try:
                lib = ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            if "builtins" not in os.path.basename(p):
                nvrtc = lib
                break
        if nvrtc is None:
            raise OSError("libnvrtc not found")
        cuda = ctypes.CDLL("libcuda.so.1")
        vp, sz, ci = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        nvrtc.nvrtcCreateProgram.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p, ctypes.c_char_p, ci, vp, vp]
        nvrtc.nvrtcCompileProgram.argtypes = [vp, ci, ctypes.POINTER(ctypes.c_char_p)]
        nvrtc.nvrtcGetProgramLogSize.argtypes = [vp, ctypes.POINTER(sz)]
        nvrtc.nvrtcGetProgramLog.argtypes = [vp, ctypes.c_char_p]
        nvrtc.nvrtcGetCUBINSize.argtypes = [vp, ctypes.POINTER(sz)]
        nvrtc.nvrtcGetCUBIN.argtypes = [vp, ctypes.c_char_p]
        nvrtc.nvrtcGetPTXSize.argtypes = [vp, ctypes.POINTER(sz)]
        nvrtc.nvrtcGetPTX.argtypes = [vp, ctypes.c_char_p]
        nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(vp)]
        nvrtc.nvrtcGetNumSupportedArchs.argtypes = [ctypes.POINTER(ci)]
        nvrtc.nvrtcGetSupportedArchs.argtypes = [ctypes.POINTER(ci)]
        cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(vp)]
        cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p]
        cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
        cuda.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ci), ci, vp]
        cuda.cuFuncSetAttribute.argtypes = [vp, ci, ci]
        cuda.cuLaunchKernel.argtypes = [vp, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                        ctypes.c_uint, ctypes.c_uint, vp, ctypes.POINTER(vp), vp]
        cuda.cuGetErrorString.argtypes = [ci, ctypes.POINTER(ctypes.c_char_p)]
        _libs["nvrtc"], _libs["cuda"] = nvrtc, cuda
    return _libs["nvrtc"], _libs["cuda"]


def _check(res, what):
    if res != 0:
        msg = ctypes.c_char_p()
        _libs["cuda"].cuGetErrorString(res, ctypes.byref(msg))
        raise RuntimeError(f"{what}: CUDA error {res} ({(msg.value or b'').decode()})")


def compile_image(src: str, arch: int) -> bytes:
    # NVRTC: a cubin for the device when NVRTC knows its architecture, else PTX for the newest one it knows
    nvrtc, _ = _libraries()
    na = ctypes.c_int()
    nvrtc.nvrtcGetNumSupportedArchs(ctypes.byref(na))
    archs = (ctypes.c_int * na.value)()
    nvrtc.nvrtcGetSupportedArchs(archs)
    cubin = arch in list(archs)
    target = f"sm_{arch}" if cubin else f"compute_{max(a for a in archs if a <= arch)}"
    prog = ctypes.c_void_p()
    if nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"fsb_beam.cu", 0, None, None) != 0:
        raise RuntimeError("nvrtcCreateProgram failed")
    try:
        opts = [f"--gpu-architecture={target}".encode(), b"--fmad=false", b"-default-device"]
        res = nvrtc.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
        if res != 0:
            n = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            log = ctypes.create_string_buffer(n.value)
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("NVRTC compile failed:\n" + log.value.decode(errors="replace"))
        n = ctypes.c_size_t()
        if cubin:
            nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n))
            buf = ctypes.create_string_buffer(n.value)
            nvrtc.nvrtcGetCUBIN(prog, buf)
        else:
            nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(n))
            buf = ctypes.create_string_buffer(n.value)
            nvrtc.nvrtcGetPTX(prog, buf)
        return buf.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))


def load_function(device: torch.device, src: str, name: str):
    # module of `src` in the device's primary context (cached per device and source)
    key = (device.index, src, name)
    fn = _funcs.get(key)
    if fn is None:
        with _lock, torch.cuda.device(device):
            fn = _funcs.get(key)
            if fn is None:
                _, cuda = _libraries()
                torch.cuda.current_stream(device)                          # primary context current on this thread
                ctx = ctypes.c_void_p()
                _check(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
                if not ctx.value:
                    raise RuntimeError("no current CUDA context")
                mkey = (device.index, src)
                m = _mods.get(mkey)
                if m is None:
                    maj, mnr = torch.cuda.get_device_capability(device)
                    image = compile_image(src, 10 * maj + mnr)
                    mod = ctypes.c_void_p()
                    _check(cuda.cuModuleLoadData(ctypes.byref(mod), image), "cuModuleLoadData")
                    m = _mods[mkey] = (mod, image)
                mod, image = m
                f = ctypes.c_void_p()
                _check(cuda.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), "cuModuleGetFunction")
                fn = _funcs[key] = (f, mod, image)
    return fn[0]


_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
_smem_set: dict = {}                                     # function handle -> dynamic shared memory limit set on it


def _dynamic_smem(fn, smem: int) -> None:
    # opt the function in to `smem` bytes of dynamic shared memory (needed above the default 48 KB), once per handle
    key = fn.value if isinstance(fn, ctypes.c_void_p) else fn
    if _smem_set.get(key, 48 * 1024) >= smem:
        return
    _, cuda = _libraries()
    _check(cuda.cuFuncSetAttribute(fn, _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(smem)),
           f"cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={smem})")
    _smem_set[key] = int(smem)


def launch(fn, grid, block: int, device: torch.device, args, smem: int = 0) -> None:
    # on the device's current torch stream (so CUDA-graph captures record it); the device's context must be current.
    # grid: an int or a (x, y, z) tuple; smem: dynamic shared memory bytes (extern __shared__), set up once per function
    _, cuda = _libraries()
    if device.index != torch.cuda.current_device():
        with torch.cuda.device(device):
            return launch(fn, grid, block, device, args, smem)
    gx, gy, gz = (tuple(grid) + (1, 1))[:3] if isinstance(grid, (tuple, list)) else (int(grid), 1, 1)
    if smem > 48 * 1024:
        _dynamic_smem(fn, smem)
    params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    stream = torch.cuda.current_stream(device).cuda_stream
    _check(cuda.cuLaunchKernel(fn, gx, gy, gz, block, 1, 1, int(smem), ctypes.c_void_p(stream), params, None),
           "cuLaunchKernel")


def beam_source(NC: int, DH: int, DHP: int, WM: int, R: int, BOUND: bool, RC_T: bool, NWARP: int) -> str:
    defs = dict(NC=NC, NA=NC - DH, DH=DH, DHP=DHP, WM=WM, R=R, BOUND=int(BOUND), RC_T=int(RC_T), NWARP=NWARP)
    return "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _HELPERS + _KERNEL + _WARP_KERNEL


def beam_cuda_ok(F: dict, q: torch.Tensor, W_max: int, delta_max: int) -> bool:
    from .fsb_tree import _beam_views
    mu, aux = _beam_views(F)
    Dh = q.shape[-1]
    DHP = 1 << max(Dh - 1, 0).bit_length()
    return (beam_cuda_on() and 32 <= DHP <= 256 and W_max <= 16 and mu.shape[-1] == Dh and aux.shape[-1] == 9
            and mu.dtype == torch.bfloat16 and aux.dtype == torch.float32
            and mu.is_contiguous() and aux.is_contiguous())


@torch.no_grad()
def forest_beam_cuda(F: dict, SP: dict, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor, sigv: torch.Tensor,
                     scale: float, s: int, W: torch.Tensor, bud: torch.Tensor, W_max: int, delta_max: int,
                     node_score: str, radius_coef, legacy: bool = None):
    # forest_beam_fused's contract (cells (T',Bq,W_max*2^delta_max), n_visit (T',Bq)); None when NVRTC or the driver
    # API is unavailable (the caller then runs the Triton kernel). legacy: an inert remnant (the pre-split kernel)
    Tp, Bq, Dh = q.shape
    R = 1 << delta_max
    DHP = 1 << max(Dh - 1, 0).bit_length()
    rc_t = torch.is_tensor(radius_coef)
    f32, bf16, i64, dev = torch.float32, torch.bfloat16, torch.int64, q.device
    from .fsb_tree import _beam_views
    _beam_views(F)     # this wrapper may see a packed F when it is called directly (no host gate ran first)
    if not (q.dtype in (f32, bf16) and W.dtype == bud.dtype == i64 and qpos.dtype == lam.dtype == sigv.dtype == f32
            and F["node_mu"].dtype == bf16 and F["node_aux"].dtype == SP["node"].dtype == f32
            and (not rc_t or radius_coef.is_floating_point())
            and all(t.device == dev for t in (W, bud, qpos, lam, sigv, F["node_mu"], F["node_aux"], SP["node"],
                                              F["leaf"], F["adm"], SP["root"], SP["t_idx"]))):
        return None                                                          # the kernel reads these types as is
    if Tp * Bq == 0:
        return (torch.empty((Tp, Bq, W_max * R), dtype=torch.long, device=q.device),
                torch.empty((Tp, Bq), dtype=torch.float32, device=q.device))
    nw = _NWARP
    NC = SP["node"].shape[-1]
    cfg = (q.device.index, NC, Dh, DHP, W_max, R, node_score == "bound", rc_t, nw)
    fn = _beam_fns.get(cfg)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return None                                                      # never compile inside a graph capture
        try:
            fn = _beam_fns[cfg] = load_function(q.device, beam_source(*cfg[1:]), "fsb_beam_cuda")
        except Exception as e:                                               # no NVRTC / driver: Triton path
            if not _failed:
                _failed.append(repr(e))
                import warnings
                warnings.warn(f"FS-B CUDA beam unavailable, using the Triton kernel: {e!r}")
            return None
    chunk_mu, chunk_ax, spine = F["node_mu"], F["node_aux"], SP["node"]
    cells = torch.empty((Tp, Bq, W_max * R), dtype=torch.long, device=q.device)
    nvis = torch.empty((Tp, Bq), dtype=torch.float32, device=q.device)
    qin = q
    keep = [SP["root"].contiguous(), qin.contiguous(), qpos.contiguous(), lam.reshape(-1).contiguous(),
            sigv.reshape(-1).contiguous(), W.contiguous(), bud.contiguous(), SP["t_idx"].contiguous(),
            radius_coef.reshape(-1).float().contiguous() if rc_t else qin]
    root, qc, qp, lm, sg, wb, bd, ti, rc = keep
    P, I64, F32, I32 = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_float, ctypes.c_int
    nq = Tp * Bq
    args = [P(root.data_ptr()), P(cells.data_ptr()), P(nvis.data_ptr()), P(qc.data_ptr()), P(rc.data_ptr()),
            F32(1.0 if rc_t else float(radius_coef)), P(qp.data_ptr()), P(lm.data_ptr()), P(sg.data_ptr()),
            P(wb.data_ptr()), P(bd.data_ptr()), P(ti.data_ptr()), P(chunk_mu.data_ptr()), P(chunk_ax.data_ptr()),
            P(spine.data_ptr()),
            P(F["leaf"].data_ptr()), P(F["adm"].data_ptr()), I64(chunk_mu.shape[1]), I64(spine.shape[1]),
            I64(F["Ltot"]), I64(F["M"]), I64(Bq), I64(nq), F32(float(scale)), F32(float(s)), I32(F["Kmax"] + 1),
            I32(1 if qin.dtype == bf16 else 0)]
    launch(fn, (nq + nw - 1) // nw, 32 * nw, q.device, args)
    return cells, nvis


class _BeamP(ctypes.Structure):
    # struct BeamP of _KERNEL (one beam's tables), passed by value
    _fields_ = ([(k, ctypes.c_void_p) for k in ("ROOT", "CELLS", "NVIS", "WBEAM", "BUDGET", "TIDX", "CHUNK",
                                                "CHUNK_AX", "SPINE", "LEAF", "ADM")]
                + [(k, ctypes.c_longlong) for k in ("Ntot", "Sp", "Ltot", "Mm", "rows")]
                + [("KMAXP1", ctypes.c_int), ("pad_", ctypes.c_int)])


@torch.no_grad()
def forest_beam_pair_cuda(beams, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor, sigv: torch.Tensor,
                          scale: float, s: int, delta_max: int, node_score: str, radius_coef):
    # both beams (main and union) over the same queries in one launch of the warp-per-query CUDA kernel ->
    # [(cells, n_visit), (cells_u, n_visit_u)], bit-identical to two forest_beam_cuda calls
    if len(beams) != 2 or not rows_flag() or not beam_cuda_on():
        return None
    Tp, Bq, Dh = q.shape
    R = 1 << delta_max
    WM = beams[0][4]
    DHP = 1 << max(Dh - 1, 0).bit_length()
    NC = beams[0][1]["node"].shape[-1]
    rc_t = torch.is_tensor(radius_coef)
    f32, bf16, i64, dev = torch.float32, torch.bfloat16, torch.int64, q.device
    from .fsb_tree import _beam_views
    for F, SP, W, bud, W_max in beams:
        _beam_views(F)     # a direct call (no host gate) may hand in packed forests
    if not (32 <= DHP <= 256 and 1 <= WM <= 16 and NC == Dh + 9
            and q.dtype in (f32, bf16) and qpos.dtype == lam.dtype == sigv.dtype == f32
            and (not rc_t or radius_coef.is_floating_point())
            and all(W_max == WM and W.dtype == bud.dtype == i64 and F["node_mu"].dtype == bf16
                    and F["node_aux"].dtype == SP["node"].dtype == f32
                    and F["node_mu"].shape[-1] == Dh and F["node_aux"].shape[-1] == NC - Dh
                    and F["node_mu"].is_contiguous() and F["node_aux"].is_contiguous()
                    and W.shape == bud.shape == (Tp, Bq)
                    and all(t.device == dev for t in (W, bud, F["node_mu"], F["node_aux"], SP["node"], F["leaf"],
                                                      F["adm"], SP["root"], SP["t_idx"]))
                    for F, SP, W, bud, W_max in beams)
            and all(t.device == dev for t in (qpos, lam, sigv))):
        return None
    if Tp * Bq == 0:
        return [(torch.empty((Tp, Bq, WM * R), dtype=i64, device=dev), torch.empty((Tp, Bq), dtype=f32, device=dev))
                for _ in beams]
    cfg = (dev.index, NC, Dh, DHP, WM, R, node_score == "bound", rc_t, _NWARP, False)
    p_cfg = ("pair", *cfg)
    fn = _beam_fns.get(p_cfg)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            fn = _beam_fns[p_cfg] = load_function(dev, beam_source(NC, Dh, DHP, WM, R, node_score == "bound", rc_t,
                                                                   _NWARP), "fsb_beam_pair_cuda")
        except Exception as e:
            if not _failed:
                _failed.append(repr(e))
                import warnings
                warnings.warn(f"FS-B CUDA beam unavailable, using the Triton kernel: {e!r}")
            return None
    qc = q.contiguous()
    if qpos.is_contiguous():
        qp, qp_s = qpos, 1
    elif qpos.stride(0) == 0 and (Bq == 1 or qpos.stride(-1) == 0):
        qp, qp_s = qpos, 0
    else:
        qp, qp_s = qpos.contiguous(), 1
    lm, sg = lam.reshape(-1).contiguous(), sigv.reshape(-1).contiguous()
    rc = radius_coef.reshape(-1).float().contiguous() if rc_t else qc
    keep, outs, ps = [qc, qp, lm, sg, rc], [], []
    for F, SP, W, bud, _ in beams:
        cells = torch.empty((Tp, Bq, WM * R), dtype=i64, device=dev)
        nvis = torch.empty((Tp, Bq), dtype=f32, device=dev)
        root, wb, bd, ti = SP["root"].contiguous(), W.contiguous(), bud.contiguous(), SP["t_idx"].contiguous()
        keep += [root, wb, bd, ti]
        ps.append(_BeamP(root.data_ptr(), cells.data_ptr(), nvis.data_ptr(), wb.data_ptr(), bd.data_ptr(),
                         ti.data_ptr(), F["node_mu"].data_ptr(), F["node_aux"].data_ptr(), SP["node"].data_ptr(),
                         F["leaf"].data_ptr(),
                         F["adm"].data_ptr(), F["node_mu"].shape[1], SP["node"].shape[1], F["Ltot"], F["M"], Tp,
                         F["Kmax"] + 1, 0))
        outs.append((cells, nvis))
    P, I64, F32, I32 = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_float, ctypes.c_int
    nq = Tp * Bq
    args = [ps[0], ps[1], P(qc.data_ptr()), P(rc.data_ptr()), F32(1.0 if rc_t else float(radius_coef)),
            P(qp.data_ptr()), P(lm.data_ptr()), P(sg.data_ptr()), I64(Bq), I64(nq), F32(float(scale)), F32(float(s)),
            I64(qp_s), I32(1 if qc.dtype == bf16 else 0)]
    launch(fn, (nq, 2), 32, dev, args)
    return outs


# ---- row-cooperative prefill descent (FSB_BEAM_ROWS=1, the default; 0 keeps the warp-per-query kernel) ------------

_ROWS_CAP, _ROWS_UMAX = 64, 128        # union nodes per shared-memory tile; child slots of a level's union it tracks
_ROWS_QB = 64                          # queries per block (a row of Bq queries takes ceil(Bq / QB) blocks): four
                                       # 40 KB blocks per SM hide each other's latency-bound phases better than two
_ROWS_MIN_BQ = 32                      # rows with fewer queries (decode) keep the warp-per-query kernel
_WARPS_BYTES = 1536                    # >= sizeof(WarpS), the fallback's per-warp scratch (static_assert in the kernel)
_cap_cache: dict = {}


def beam_rows_on() -> bool:
    return rows_flag() and beam_cuda_on() and not _rows_failed


def _rows_layout(CAP: int, UMAX: int, QB: int, q_bf16: bool = False):
    # byte offsets of the rows kernel's dynamic shared memory regions (16-byte aligned) and the total
    tc = q_bf16 and QB == 128 and CAP == 64 and UMAX == 128
    mu_bytes = max(CAP * 256, CAP * 128 + QB * CAP * 2, QB // 16 * _WARPS_BYTES) if tc else max(CAP * 256, QB // 16 * _WARPS_BYTES)
    sc_bytes = 0 if tc else (QB * 36 * 4)
    sizes = (("MU", mu_bytes), ("UM", 32 * UMAX), ("UM2", 16 * UMAX),
             ("MARK", 2 * UMAX), ("RANK", 2 * UMAX), ("FRONT", QB * 64), ("SC", sc_bytes), ("CU", QB * 34 * 2))
    off, o = {}, 0
    for name, size in sizes:
        off[name] = o
        o += -(-size // 16) * 16
    return off, o


def rows_source(NC: int, DH: int, WM: int, R: int, BOUND: bool, RC_T: bool, CAP: int = _ROWS_CAP,
                UMAX: int = _ROWS_UMAX, QB: int = _ROWS_QB, q_bf16: bool = False):
    # -> (source, dynamic shared memory bytes)
    off, total = _rows_layout(CAP, UMAX, QB, q_bf16)
    tc = q_bf16 and QB == 128 and CAP == 64 and UMAX == 128
    defs = dict(NC=NC, NA=NC - DH, DH=DH, DHP=64, WM=WM, R=R, BOUND=int(BOUND), RC_T=int(RC_T), CAP=CAP, UMAX=UMAX, QB=QB,
                MINB=3 if tc else max(1, 256 // QB), OSTAGE=int(QB * WM * R * 8 <= total), Q_BF16=int(q_bf16),
                **{f"OFF_{k}": v for k, v in off.items()})
    return "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _HELPERS + _KERNEL + _ROWS_KERNEL, total


# ---- lean row-cooperative prefill descent (FSB_BEAM_V2=1, default 0): fsb_beam_rows2 takes the bf16 tensor-core
# configuration of fsb_beam_rows (128 queries per block, head dim 64, frontier width 16, up to 4 cells per frontier
# node) with sorting-network selection on packed keys, register frontiers and two block barriers per level. Its one
# semantic deviation: children whose scores agree in the top 27 bits are ordered by child index. Other shapes, the
# test hooks (cap / umax / qb) and FSB_BEAM_V2=0 keep fsb_beam_rows.

_ROWS2_UC, _ROWS2_DCH = 128, 64        # union slots per level (a larger union falls back); union slots per dot chunk


def beam_v2_on() -> bool:
    return os.environ.get("FSB_BEAM_V2", "0") == "1" and not _rows2_failed


def _rows2_layout(uc: int = _ROWS2_UC):
    # byte offsets of fsb_beam_rows2's dynamic shared memory regions (16-byte aligned) and the total: the union tile
    # (also the fallback's per-warp scratch), the warps' dot chunks, two levels of union node fields, the union's score
    # terms, the carried prefixes, the child slot lists, the two mark buffers, the parents' child slots and two levels
    # of node codes
    sizes = (("MU", max(uc * 128, 8 * _WARPS_BYTES)), ("DOT", 8 * 16 * _ROWS2_DCH * 2), ("UM", 2 * uc * 16),
             ("UX", uc * 8), ("CAR", 128 * 16 * 4), ("CS", 128 * 16), ("MK", 2 * (uc + 16)), ("RANK", uc),
             ("CODE", 2 * uc))
    off, o = {}, 0
    for name, size in sizes:
        off[name] = o
        o += -(-size // 16) * 16
    return off, o


def rows2_source(NC: int, DH: int, WM: int, R: int, BOUND: bool, RC_T: bool, key64: bool = False,
                 uc: int = _ROWS2_UC):
    # -> (source, dynamic shared memory bytes); fsb_beam_rows' sections in their tensor-core configuration supply the
    # shared helpers and the fallback descent. key64 / uc: test builds (64-bit selection keys; a smaller union table)
    src, _ = rows_source(NC, DH, WM, R, BOUND, RC_T, _ROWS_CAP, _ROWS_UMAX, 128, True)
    off, total = _rows2_layout(uc)
    defs = dict(MINB2=3, KEY64=int(key64), UC2=uc, DCH=_ROWS2_DCH,
                **{f"OFF2_{k}": v for k, v in off.items()})
    return "".join(f"#define {k} {v}\n" for k, v in defs.items()) + src + _ROWS2_KERNEL, total


def _sm80(device: torch.device) -> bool:
    ok = _cap_cache.get(device.index)
    if ok is None:
        ok = _cap_cache[device.index] = torch.cuda.get_device_capability(device) >= (8, 0)
    return ok


def beam_rows_ok(F: dict, q: torch.Tensor, W_max: int) -> bool:
    # prefill rows: head dim 64, at least 32 queries per row, beam width <= 16, sm_80+ (cp.async, 40 KB of shared memory
    # per 64-query block). The row length first: decode beams (one query per row) then skip the switch lookups, which
    # cost host time on every decode step
    if not (q.dim() == 3 and q.shape[1] >= _ROWS_MIN_BQ and q.is_cuda and beam_rows_on()):
        return False
    Tp, Bq, Dh = q.shape
    from .fsb_tree import _beam_views
    mu = _beam_views(F)[0]
    return (Dh == 64 and 1 <= W_max <= 16 and mu.shape[-1] == Dh
            and mu.dtype == torch.bfloat16 and F["node_aux"].dtype == torch.float32
            and mu.is_contiguous() and F["node_aux"].is_contiguous() and _sm80(q.device))


@torch.no_grad()
def forest_beam_rows_cuda(beams, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor, sigv: torch.Tensor,
                          scale: float, s: int, delta_max: int, node_score: str, radius_coef, cap: int = None,
                          umax: int = None, qb: int = None, fallback: torch.Tensor = None, q_nq: int = None,
                          key64: bool = False, uc2: int = None):
    # beams: one or two (F, SP, W, bud, W_max) sharing the queries (same W_max) -> [(cells (T',Bq,W_max*2^delta_max)
    # int64, n_visit (T',Bq) fp32)] per beam, forest_beam_fused's contract, from one launch; None when the kernel is
    # unavailable (the caller then runs the warp-per-query kernel). cap / umax / qb: test hooks for the tile and union
    # sizes and the queries per block. fallback: test / diagnostic hook, 65 int32 counters on the device to which the
    # launch adds the number of its blocks that left the cooperative descent for the warp-per-query fallback ([0]) and
    # the level at which they did ([1 + level]). FSB_BEAM_V2=1 runs fsb_beam_rows2 where it applies; key64: its test
    # build with 64-bit selection keys (fsb_beam_rows' exact child order), uc2: its union table size (32, 64 or 128).
    v2 = (cap is None and umax is None and qb is None and q.dtype == torch.bfloat16 and delta_max <= 2
          and beams[0][4] == 16 and beam_v2_on())
    cap, umax, qb = cap or _ROWS_CAP, umax or _ROWS_UMAX, qb or (128 if q.dtype == torch.bfloat16 else _ROWS_QB)
    Tp, Bq, Dh = q.shape
    R = 1 << delta_max
    WM = beams[0][4]
    rc_t = torch.is_tensor(radius_coef)
    f32, bf16, i64, dev = torch.float32, torch.bfloat16, torch.int64, q.device
    from .fsb_tree import _beam_views
    for F, SP, W, bud, W_max in beams:
        _beam_views(F)     # a direct call (no host gate) may hand in packed forests
    if not (q.dtype in (f32, bf16) and qpos.dtype == lam.dtype == sigv.dtype == f32
            and (not rc_t or radius_coef.is_floating_point())
            and all(W_max == WM and W.dtype == bud.dtype == i64 and F["node_mu"].dtype == bf16
                    and F["node_aux"].dtype == SP["node"].dtype == f32
                    and F["node_mu"].is_contiguous() and F["node_aux"].is_contiguous()
                    and W.shape == bud.shape == (Tp, Bq) and SP["root"].shape[0] >= Tp and SP["t_idx"].shape[0] >= Tp
                    and SP["node"].shape[0] >= Tp
                    and all(x.device == dev for x in (W, bud, F["node_mu"], F["node_aux"], SP["node"], F["leaf"],
                                                      F["adm"], SP["root"], SP["t_idx"]))
                    for F, SP, W, bud, W_max in beams)
            and qpos.device == lam.device == sigv.device == dev
            and (fallback is None or (fallback.dtype == torch.int32 and fallback.device == dev
                                      and fallback.numel() >= 65))):
        return None
    if Tp * Bq == 0:
        return [(torch.empty((Tp, Bq, WM * R), dtype=i64, device=dev), torch.empty((Tp, Bq), dtype=f32, device=dev))
                for _ in beams]
    q_bf16 = q.dtype == bf16
    ent = None
    if v2:
        cfg2 = ("rows2", dev.index, Dh + 9, Dh, WM, R, node_score == "bound", rc_t, key64, uc2 or _ROWS2_UC)
        ent = _beam_fns.get(cfg2)
        if ent is None and not torch.cuda.is_current_stream_capturing():
            try:
                src, smem = rows2_source(Dh + 9, Dh, WM, R, node_score == "bound", rc_t, key64, uc2 or _ROWS2_UC)
                ent = _beam_fns[cfg2] = (load_function(dev, src, "fsb_beam_rows2"), smem)
            except Exception as e:                                           # its own latch: fsb_beam_rows runs
                import warnings
                if not _rows2_failed:
                    _rows2_failed.append(repr(e))
                    warnings.warn(f"FS-B lean row beam (FSB_BEAM_V2) unavailable, using fsb_beam_rows: {e!r}")
                ent = None
    cfg = ("rows", dev.index, Dh + 9, Dh, WM, R, node_score == "bound", rc_t, cap, umax, qb, q_bf16)
    if ent is None:
        ent = _beam_fns.get(cfg)
    if ent is None:
        if torch.cuda.is_current_stream_capturing():
            return None                                                      # never compile inside a graph capture
        try:
            src, smem = rows_source(Dh + 9, Dh, WM, R, node_score == "bound", rc_t, cap, umax, qb, q_bf16)
            ent = _beam_fns[cfg] = (load_function(dev, src, "fsb_beam_rows"), smem)
        except Exception as e:
            import warnings
            if isinstance(e, OSError):                                       # no NVRTC / driver: no CUDA beam builds
                if not _failed:
                    _failed.append(repr(e))
                    warnings.warn(f"FS-B CUDA beam unavailable, using the Triton kernel: {e!r}")
            elif not _rows_failed:                                           # its own latch: the warp kernel stays on
                _rows_failed.append(repr(e))
                warnings.warn(f"FS-B row-cooperative CUDA beam unavailable, using the warp-per-query kernel: {e!r}")
            return None
    fn, smem = ent
    nq = Bq if q_nq is None else int(q_nq)
    esz = q.element_size()
    if (q.stride(2) == 1 and q.stride(1) >= Dh and q.stride(1) * esz % 16 == 0 and q.stride(0) * esz % 16 == 0
            and q.data_ptr() % 16 == 0 and 0 < nq <= Bq):
        qc, qst_t, qst_b = q, q.stride(0), q.stride(1)      # a strided stage view: kernel guards rows >= nq
    else:
        qc = q.contiguous()
        if qc.data_ptr() % 16:
            qc = qc.clone()                                                  # the kernel reads query rows as float4s/uint4s
        qst_t, qst_b, nq = Bq * Dh, Dh, Bq
    qp = qpos.contiguous()
    lm, sg = lam.reshape(-1).contiguous(), sigv.reshape(-1).contiguous()
    rc = radius_coef.reshape(-1).float().contiguous() if rc_t else qc
    keep, outs, ps = [qc, qp, lm, sg, rc], [], []
    for F, SP, W, bud, _ in beams:
        cells = torch.empty((Tp, Bq, WM * R), dtype=i64, device=dev)
        nvis = torch.empty((Tp, Bq), dtype=f32, device=dev)
        root, wb, bd, ti = SP["root"].contiguous(), W.contiguous(), bud.contiguous(), SP["t_idx"].contiguous()
        keep += [root, wb, bd, ti]
        ps.append(_BeamP(root.data_ptr(), cells.data_ptr(), nvis.data_ptr(), wb.data_ptr(), bd.data_ptr(),
                         ti.data_ptr(), F["node_mu"].data_ptr(), F["node_aux"].data_ptr(), SP["node"].data_ptr(),
                         F["leaf"].data_ptr(),
                         F["adm"].data_ptr(), F["node_mu"].shape[1], SP["node"].shape[1], F["Ltot"], F["M"], Tp,
                         F["Kmax"] + 1, 0))
        outs.append((cells, nvis))
    if len(ps) == 1:
        ps.append(_BeamP())                                                  # rows = 0: every block runs beam 0
    P, I64, F32, I32 = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_float, ctypes.c_int
    args = [ps[0], ps[1], P(qc.data_ptr()), P(qp.data_ptr()), P(rc.data_ptr()), F32(1.0 if rc_t else float(radius_coef)),
            P(lm.data_ptr()), P(sg.data_ptr()), I32(Bq), F32(float(scale)), F32(float(s)),
            I64(qst_t), I64(qst_b), I32(nq), P(fallback.data_ptr() if fallback is not None else 0)]
    launch(fn, Tp * len(beams) * (-(-Bq // qb)), 2 * qb, dev, args, smem=smem)
    return outs
