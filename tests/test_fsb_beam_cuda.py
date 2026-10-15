import ctypes
import hashlib

import pytest
import torch

from src.fsattn import fsb_beam_cuda as bc
from src.fsattn import fsb_fast_step as fast
from src.fsattn import fsb_tree


def _nvrtc_ok():
    try:
        bc._libraries()
        return True
    except Exception:
        return False


gpu = pytest.mark.skipif(not (torch.cuda.is_available() and fast._HAVE_TRITON and _nvrtc_ok()),
                         reason="needs CUDA, Triton and NVRTC")

if fast._HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _erf_kernel(X, O, N, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        tl.store(O + i, tl.math.erf(tl.load(X + i, mask=m, other=0.0)), mask=m)


_ERF_SRC = "#define DHP 64\n" + bc._HELPERS + r'''
extern "C" __global__ void erf_rows(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = erf_tl(x[i]);
}'''


@gpu
def test_erf_matches_triton_bitwise():
    # every 251st fp32 bit pattern plus zeros, subnormals, infinities and the polynomial switch at 1.00296
    x = (torch.arange(0, 1 << 32, 251, dtype=torch.int64) - (1 << 31)).to(torch.int32)
    edge = torch.tensor([0, 1, 2, 0x007FFFFF, 0x00800000, 0x3F8060FD, 0x3F8060FE, 0x3F8060FF, 0x7F800000, 0x7FC00000],
                        dtype=torch.int64)
    x = torch.cat([x, edge.to(torch.int32), (edge | (1 << 31)).sub(1 << 32).to(torch.int32)]).cuda().view(torch.float32)
    ref = torch.empty_like(x)
    _erf_kernel[(triton.cdiv(x.numel(), 1024),)](x, ref, x.numel(), BLOCK=1024, enable_fp_fusion=False)
    got = torch.empty_like(x)
    fn = bc.load_function(x.device, _ERF_SRC, "erf_rows")
    bc.launch(fn, (x.numel() + 255) // 256, 256, x.device,
              [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(got.data_ptr()), ctypes.c_int(x.numel())])
    same = (ref.view(torch.int32) == got.view(torch.int32)) | (torch.isnan(ref) & torch.isnan(got))
    assert bool(same.all())


_SMEM_SRC = r'''
extern "C" __global__ void smem_flip(const int* x, int* out, int n) {
    extern __shared__ int buf[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] = 3 * x[i] + 1;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = buf[n - 1 - i];
}'''


@gpu
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
                    reason="needs more than 48 KB of shared memory per block (sm_80+)")
def test_launch_dynamic_shared_memory():
    # launch(..., smem=) above the 48 KB default (cuFuncSetAttribute opt-in), and a clear error past the device limit
    n = 96 * 1024 // 4
    x = torch.arange(n, dtype=torch.int32, device="cuda")
    out = torch.full_like(x, -7)
    fn = bc.load_function(x.device, _SMEM_SRC, "smem_flip")
    args = [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int(n)]
    bc.launch(fn, 1, 256, x.device, args, smem=4 * n)
    bc.launch(fn, (1, 1, 1), 256, x.device, args, smem=4 * n)                     # tuple grid, attribute already set
    torch.cuda.synchronize()
    assert torch.equal(out, (3 * x + 1).flip(0))
    small = torch.arange(1000, dtype=torch.int32, device="cuda")
    so = torch.empty_like(small)
    bc.launch(fn, 1, 128, small.device, [ctypes.c_void_p(small.data_ptr()), ctypes.c_void_p(so.data_ptr()),
                                         ctypes.c_int(1000)], smem=4000)
    torch.cuda.synchronize()
    assert torch.equal(so, (3 * small + 1).flip(0))
    with pytest.raises(RuntimeError, match="cuFuncSetAttribute"):
        bc.launch(fn, 1, 256, x.device, args, smem=1 << 20)                         # 1 MB: past every GPU's limit


def test_legacy_source_is_9b4e9ff():
    # FSB_BEAM_ROWS=0 must restore the CUDA beam of 9b4e9ff exactly: its kernel source, byte for byte (the sha256 of
    # 9b4e9ff's beam_source for these configurations); the current warp kernel's source differs
    golden = {(73, 64, 64, 16, 4, True, False, 4): "bc050b301c19451a368e01bd486a9c32587befb03b8500524684542a1814f3b8",
              (73, 64, 64, 16, 1, True, True, 4): "9448e4479315581d655be1e31038fff6b6920e7fe6b0d2e1140863260dae9cda",
              (41, 32, 32, 8, 4, False, False, 4): "c69e85a23903529c63179f49b21d5e00ecbc5d98aff921fca384cb43e0aa3e12"}
    for cfg, h in golden.items():
        assert hashlib.sha256(bc.beam_source(*cfg, legacy=True).encode()).hexdigest() == h
        assert bc.beam_source(*cfg) != bc.beam_source(*cfg, legacy=True)


def _forest_stage(seed, M, H=16, nbc=32, Bq=128, D=64, s=32, admit_all=True):
    # one prefill stage of a 16k-token forest (65536 descents at Bq=128) with clustered keys and queries
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(H, 64, D, generator=g)
    keys = (centers[:, torch.randint(0, 64, (M * s,), generator=g)] + 0.6 * torch.randn(H, M * s, D, generator=g)) * 0.5
    keys = keys.to("cuda", torch.bfloat16).float().view(H, M, s, D)
    kbar, ceil = keys.mean(2), keys.norm(dim=-1).amax(-1)
    admit = torch.ones(H, M, dtype=torch.bool, device="cuda") if admit_all else ceil > ceil.median()
    F = fsb_tree.build_forest(kbar, ceil, admit)
    nb = M * s // Bq
    blk = torch.arange(nb, device="cuda", dtype=torch.float32)[nb - nbc:] * Bq
    m_b = torch.searchsorted(F["adm"].masked_fill(F["adm"] < 0, M),
                             (blk / s).floor().long()[None].expand(H, nbc).contiguous()).reshape(-1)
    SP = fsb_tree.build_spines(F, m_b, torch.arange(H, device="cuda").repeat_interleave(nbc))
    rows = H * nbc
    qc = torch.randint(0, 64, (rows, Bq), generator=g)
    q = ((centers[torch.arange(rows)[:, None] // nbc, qc] + 0.8 * torch.randn(rows, Bq, D, generator=g)) * 0.5).cuda()
    qpos = (blk[:, None] + torch.arange(Bq, device="cuda", dtype=torch.float32))[None].expand(H, nbc, Bq)
    lam = (torch.rand(rows, 1, 1, generator=g) * 3).cuda()
    sigv = (torch.rand(rows, 1, 1, generator=g) * 1000 + 50).cuda()
    return F, SP, q, qpos.reshape(rows, Bq).contiguous(), lam, sigv


@gpu
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("width,delta,admit_all,node_score,rc_tensor", [
    (16, 0, True, "bound", False), (4, 2, True, "bound", False), (16, 2, False, "bound", True),
    (8, 0, False, "mean", False), (1, 0, True, "bound", False)])
def test_cuda_beam_matches_level_kernel(monkeypatch, width, delta, admit_all, node_score, rc_tensor, legacy):
    # warp-per-query CUDA descent (the current kernel and 9b4e9ff's) == the per-level Triton reference: cells and
    # visit counts, bitwise
    F, SP, q, qpos, lam, sigv = _forest_stage(9 + width + delta, 512, admit_all=admit_all)
    s = 32
    rows, Bq = q.shape[:2]
    depth = (SP["height"] - delta).clamp(min=0)[:, None].expand(rows, Bq)
    W = torch.minimum(torch.full_like(depth, width), torch.bitwise_left_shift(torch.ones_like(depth), depth))
    dl = (SP["height"][:, None] - depth).clamp(min=0)
    rc = torch.rand(rows, generator=torch.Generator().manual_seed(1)).cuda() * 8 + 1 if rc_tensor else 8.0
    monkeypatch.setenv("FSB_FAST_BEAM", "0")
    ref = fsb_tree._forest_beam(F, SP, q, qpos, lam, sigv, 0.125, s, W, dl, width, 2, node_score, rc, None)
    bud = torch.bitwise_left_shift(torch.ones_like(dl), dl)
    got = bc.forest_beam_cuda(F, SP, q, qpos, lam, sigv, 0.125, s, W, bud, width, 2, node_score, rc, legacy=legacy)
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


@gpu
@pytest.mark.parametrize("rows", ["1", "0"])
@pytest.mark.parametrize("Bq", [1, 3])
def test_cuda_beam_decode_rows(monkeypatch, Bq, rows):
    # decode-shaped grids (one row per head, a few queries) through forest_beam_fused's dispatch: the current warp
    # kernel, or 9b4e9ff's with FSB_BEAM_ROWS=0
    monkeypatch.setenv("FSB_BEAM_ROWS", rows)
    F, SP, q, qpos, lam, sigv = _forest_stage(21, 512, nbc=1)
    q, qpos = q[:, :Bq].contiguous(), qpos[:, :Bq].contiguous()
    depth = SP["height"][:, None].expand(q.shape[0], Bq)
    W = torch.minimum(torch.full_like(depth, 16), torch.bitwise_left_shift(torch.ones_like(depth), depth))
    dl = (SP["height"][:, None] - depth).clamp(min=0)
    monkeypatch.setenv("FSB_FAST_BEAM", "0")
    ref = fsb_tree._forest_beam(F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, 16, 2, "bound", 8.0, None)
    monkeypatch.setenv("FSB_FAST_BEAM", "1")
    ran = []
    real = bc.forest_beam_cuda
    monkeypatch.setattr(bc, "forest_beam_cuda", lambda *a: ran.append(real(*a)) or ran[-1])
    got = fast.forest_beam_fused(F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, 16, 2, "bound", 8.0)
    assert len(ran) == 1 and ran[0] is not None                                  # the CUDA build produced it
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


@gpu
def test_beam_cuda_flag_and_fallback(monkeypatch):
    # FSB_BEAM_CUDA=0 keeps the Triton kernel; a missing NVRTC warns once and falls back to it as well
    F, SP, q, qpos, lam, sigv = _forest_stage(5, 128, nbc=2)
    depth = SP["height"][:, None].expand(q.shape[:2])
    W = torch.minimum(torch.full_like(depth, 16), torch.bitwise_left_shift(torch.ones_like(depth), depth))
    dl = (SP["height"][:, None] - depth).clamp(min=0)
    args = (F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, 16, 2, "bound", 8.0)
    default = fast.forest_beam_fused(*args)
    monkeypatch.setenv("FSB_BEAM_CUDA", "0")
    monkeypatch.setattr(bc, "forest_beam_cuda", lambda *a: pytest.fail("CUDA beam ran with FSB_BEAM_CUDA=0"))
    off = fast.forest_beam_fused(*args)
    assert torch.equal(default[0], off[0]) and torch.equal(default[1], off[1])
    monkeypatch.undo()

    def no_nvrtc(*a):
        raise OSError("libnvrtc not found")

    monkeypatch.setattr(bc, "_failed", [])
    monkeypatch.setattr(bc, "_rows_failed", [])
    monkeypatch.setattr(bc, "_beam_fns", {})
    monkeypatch.setattr(bc, "load_function", no_nvrtc)
    with pytest.warns(UserWarning, match="CUDA beam unavailable") as rec:
        fb = fast.forest_beam_fused(*args)                                   # prefill rows: the row kernel is first
    assert torch.equal(default[0], fb[0]) and torch.equal(default[1], fb[1])
    assert not bc.beam_cuda_on() and not bc.beam_rows_on()
    beam_warns = [str(w.message) for w in rec if "CUDA beam unavailable" in str(w.message)]
    assert len(beam_warns) == 1 and "Triton" in beam_warns[0]                # one warning for both kernels


@gpu
def test_rows_build_failure_keeps_warp_kernel(monkeypatch):
    # a row kernel that fails to build warns once and hands its rows to the warp kernel; the warp kernel's latch is
    # separate, so it stays on (decode keeps the CUDA beam) and the row kernel is not retried
    import warnings
    F, SP, q, qpos, lam, sigv = _forest_stage(5, 128, nbc=2)
    W, dl, _ = _rows_args(F, SP, q, qpos, lam, sigv, 16, 0)
    args = (F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, 16, 2, "bound", 8.0)
    default = fast.forest_beam_fused(*args)
    real_load = bc.load_function

    def no_rows(dev, src, name):
        if name == "fsb_beam_rows":
            raise RuntimeError("NVRTC compile failed: (simulated)")
        return real_load(dev, src, name)
    monkeypatch.setattr(bc, "_failed", [])
    monkeypatch.setattr(bc, "_rows_failed", [])
    monkeypatch.setattr(bc, "_beam_fns", {k: v for k, v in bc._beam_fns.items() if k[0] != "rows"})
    monkeypatch.setattr(bc, "load_function", no_rows)
    ran = []
    real_warp = bc.forest_beam_cuda
    monkeypatch.setattr(bc, "forest_beam_cuda", lambda *a, **k: ran.append(1) or real_warp(*a, **k))
    with pytest.warns(UserWarning, match="row-cooperative CUDA beam unavailable"):
        got = fast.forest_beam_fused(*args)
    assert ran == [1] and torch.equal(default[0], got[0]) and torch.equal(default[1], got[1])
    assert bc.beam_cuda_on() and not bc.beam_rows_on() and len(bc._rows_failed) == 1 and not bc._failed
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # no second warning, no retry
        again = fast.forest_beam_fused(*args)
    assert ran == [1, 1] and torch.equal(default[0], again[0]) and torch.equal(default[1], again[1])


@gpu
@pytest.mark.parametrize("legacy", [False, True])
def test_cuda_beam_edge_rows(monkeypatch, legacy):
    # rows of beam width 0 (the reference drops even a carried root), roots carried at once (large budgets), and an
    # empty grid
    F, SP, q, qpos, lam, sigv = _forest_stage(13, 128, nbc=2)
    rows, Bq = q.shape[:2]
    g = torch.Generator().manual_seed(2)
    W = torch.randint(0, 3, (rows, Bq), generator=g).cuda()
    dl = torch.randint(0, 9, (rows, Bq), generator=g).cuda()
    monkeypatch.setenv("FSB_FAST_BEAM", "0")
    ref = fsb_tree._forest_beam(F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, 2, 2, "bound", 8.0, None)
    bud = torch.bitwise_left_shift(torch.ones_like(dl), dl)
    got = bc.forest_beam_cuda(F, SP, q, qpos, lam, sigv, 0.125, 32, W, bud, 2, 2, "bound", 8.0, legacy=legacy)
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
    assert bool((W == 0).any()) and bool((ref[0][W == 0] == -1).all())
    cells, nvis = bc.forest_beam_cuda(F, SP, q[:0], qpos[:0], lam[:0], sigv[:0], 0.125, 32, W[:0], bud[:0], 2, 2,
                                      "bound", 8.0, legacy=legacy)
    assert cells.shape == (0, Bq, 8) and nvis.shape == (0, Bq)


def _rows_args(F, SP, q, qpos, lam, sigv, width, delta, node_score="bound", rc=8.0, W=None, dl=None):
    rows, Bq = q.shape[:2]
    if W is None:
        depth = (SP["height"] - delta).clamp(min=0)[:, None].expand(rows, Bq)
        W = torch.minimum(torch.full_like(depth, width), torch.bitwise_left_shift(torch.ones_like(depth), depth))
        dl = (SP["height"][:, None] - depth).clamp(min=0)
    bud = torch.bitwise_left_shift(torch.ones_like(dl), dl)
    return W, dl, bud


def _ref_beam(monkeypatch, F, SP, q, qpos, lam, sigv, W, dl, W_max, dmax, ns, rc):
    monkeypatch.setenv("FSB_FAST_BEAM", "0")
    out = fsb_tree._forest_beam(F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, W_max, dmax, ns, rc, None)
    monkeypatch.setenv("FSB_FAST_BEAM", "1")
    return out


@gpu
@pytest.mark.parametrize("qb,cap,umax", [(64, 64, 128), (128, 128, 256), (64, 2, 64), (128, 4, 32)])
@pytest.mark.parametrize("width,delta,admit_all,node_score,rc_tensor", [
    (16, 0, True, "bound", False), (4, 2, True, "bound", False), (16, 2, False, "bound", True),
    (8, 0, False, "mean", False), (1, 0, True, "bound", False)])
def test_rows_beam_matches_level_kernel(monkeypatch, width, delta, admit_all, node_score, rc_tensor, qb, cap, umax):
    # row-cooperative descent == the per-level reference, bitwise, with 64 and 128 queries per block; a 2-node tile
    # stages every level's union in several passes, 32 union slots overflow so rows fall back to one warp per query
    F, SP, q, qpos, lam, sigv = _forest_stage(9 + width + delta, 512, admit_all=admit_all)
    rows = q.shape[0]
    rc = torch.rand(rows, generator=torch.Generator().manual_seed(1)).cuda() * 8 + 1 if rc_tensor else 8.0
    W, dl, bud = _rows_args(F, SP, q, qpos, lam, sigv, width, delta)
    ref = _ref_beam(monkeypatch, F, SP, q, qpos, lam, sigv, W, dl, width, 2, node_score, rc)
    got = bc.forest_beam_rows_cuda([(F, SP, W, bud, width)], q, qpos, lam, sigv, 0.125, 32, 2, node_score, rc,
                                   cap=cap, umax=umax, qb=qb)
    assert got is not None
    assert torch.equal(ref[0], got[0][0]) and torch.equal(ref[1], got[0][1])


@gpu
def test_rows_pair_one_launch(monkeypatch):
    # two beams over the same queries (different forests, widths and budgets) in one launch == two reference calls;
    # forest_beam_pair dispatches it, and traverse_forest takes it for the union beam
    F, SP, q, qpos, lam, sigv = _forest_stage(31, 512)
    Fu, SPu, *_ = _forest_stage(31, 512, admit_all=False)
    W, dl, bud = _rows_args(F, SP, q, qpos, lam, sigv, 16, 0)
    Wu, du, budu = _rows_args(Fu, SPu, q, qpos, lam, sigv, 8, 1)
    ref_a = _ref_beam(monkeypatch, F, SP, q, qpos, lam, sigv, W, dl, 16, 2, "bound", 8.0)
    ref_u = _ref_beam(monkeypatch, Fu, SPu, q, qpos, lam, sigv, Wu, du, 16, 2, "bound", 8.0)
    got = bc.forest_beam_rows_cuda([(F, SP, W, bud, 16), (Fu, SPu, Wu, budu, 16)], q, qpos, lam, sigv, 0.125, 32, 2,
                                   "bound", 8.0)
    assert torch.equal(ref_a[0], got[0][0]) and torch.equal(ref_a[1], got[0][1])
    assert torch.equal(ref_u[0], got[1][0]) and torch.equal(ref_u[1], got[1][1])
    pair = fast.forest_beam_pair(F, SP, W, dl, 16, Fu, SPu, Wu, du, 16, q, qpos, lam, sigv, 0.125, 32, 2, "bound", 8.0)
    assert pair is not None
    assert all(torch.equal(a, b) for a, b in zip(pair[0] + pair[1], got[0] + got[1]))
    calls = []
    real = fast.forest_beam_pair
    monkeypatch.setattr(fast, "forest_beam_pair", lambda *a: calls.append(1) or real(*a))
    sel, nv, _ = fsb_tree.traverse_forest(F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, 16, 2, "bound", 8.0,
                                          union_width=16, union_W_max=16, union_F=Fu, union_SP=SPu, union_delta=du,
                                          union_W=Wu)
    assert calls == [1]
    assert torch.equal(sel, fsb_tree.union_candidates(ref_a[0], ref_u[0])) and torch.equal(nv, ref_a[1] + ref_u[1])
    monkeypatch.setenv("FSB_BEAM_ROWS", "0")                                    # warp kernel, two launches
    assert fast.forest_beam_pair(F, SP, W, dl, 16, Fu, SPu, Wu, du, 16, q, qpos, lam, sigv, 0.125, 32, 2, "bound",
                                 8.0) is None
    sel0, nv0, _ = fsb_tree.traverse_forest(F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, 16, 2, "bound", 8.0,
                                            union_width=16, union_W_max=16, union_F=Fu, union_SP=SPu, union_delta=du,
                                            union_W=Wu)
    assert torch.equal(sel0, sel) and torch.equal(nv0, nv)


@gpu
@pytest.mark.parametrize("umax", [None, 32])
@pytest.mark.parametrize("Bq", [128, 77, 32, 200])
def test_rows_edge_rows(monkeypatch, Bq, umax):
    # beam widths 0..16 and budgets 2^0..2^8 per query, roots carried at once, rows that fill a block partly or need
    # several blocks, an empty grid; with 32 union slots, blocks hand their queries to the warp descent mid-way, some
    # of them already at their fixed point
    F, SP, q, qpos, lam, sigv = _forest_stage(13, 512, nbc=4, Bq=max(Bq, 128))
    q, qpos = q[:, :Bq].contiguous(), qpos[:, :Bq].contiguous()
    rows = q.shape[0]
    g = torch.Generator().manual_seed(Bq)
    W = torch.randint(0, 17, (rows, Bq), generator=g).cuda()
    dl = torch.randint(0, 9, (rows, Bq), generator=g).cuda()
    bud = torch.bitwise_left_shift(torch.ones_like(dl), dl)
    ref = _ref_beam(monkeypatch, F, SP, q, qpos, lam, sigv, W, dl, 16, 2, "bound", 8.0)
    got = bc.forest_beam_rows_cuda([(F, SP, W, bud, 16)], q, qpos, lam, sigv, 0.125, 32, 2, "bound", 8.0, umax=umax)
    assert torch.equal(ref[0], got[0][0]) and torch.equal(ref[1], got[0][1])
    assert bool((W == 0).any()) and bool((ref[0][W == 0] == -1).all())
    e = bc.forest_beam_rows_cuda([(F, SP, W[:0], bud[:0], 16)], q[:0], qpos[:0], lam[:0], sigv[:0], 0.125, 32, 2,
                                 "bound", 8.0, umax=umax)
    assert e[0][0].shape == (0, Bq, 64) and e[0][1].shape == (0, Bq)


@gpu
def test_rows_ties_and_nonfinite(monkeypatch):
    # exact score ties (cells with duplicated keys, no position term: siblings tie and the slot order decides), and
    # infinite query entries (inf / NaN scores) against the warp kernels (9b4e9ff's and the current one), which share
    # the fused kernels' convention
    g = torch.Generator().manual_seed(7)
    H, M, s, D, nbc, Bq = 16, 512, 32, 64, 8, 128
    base = (torch.randn(H, M // 2, s, D, generator=g) * 0.5).repeat_interleave(2, 1).to("cuda", torch.bfloat16).float()
    kbar, ceil = base.mean(2), base.norm(dim=-1).amax(-1)
    F = fsb_tree.build_forest(kbar, ceil, torch.ones(H, M, dtype=torch.bool, device="cuda"))
    nb = M * s // Bq
    blk = torch.arange(nb, device="cuda", dtype=torch.float32)[nb - nbc:] * Bq
    m_b = torch.searchsorted(F["adm"].masked_fill(F["adm"] < 0, M),
                             (blk / s).floor().long()[None].expand(H, nbc).contiguous()).reshape(-1)
    SP = fsb_tree.build_spines(F, m_b, torch.arange(H, device="cuda").repeat_interleave(nbc))
    rows = H * nbc
    q = (torch.randn(rows, Bq, D, generator=g) * 0.5).cuda()
    qpos = (blk[:, None] + torch.arange(Bq, device="cuda", dtype=torch.float32))[None].expand(H, nbc, Bq)
    qpos = qpos.reshape(rows, Bq).contiguous()
    lam = torch.zeros(rows, 1, 1, device="cuda")
    sigv = torch.full((rows, 1, 1), 300.0, device="cuda")
    W, dl, bud = _rows_args(F, SP, q, qpos, lam, sigv, 16, 0)
    ref = _ref_beam(monkeypatch, F, SP, q, qpos, lam, sigv, W, dl, 16, 2, "bound", 8.0)
    got = bc.forest_beam_rows_cuda([(F, SP, W, bud, 16)], q, qpos, lam, sigv, 0.125, 32, 2, "bound", 8.0)
    assert torch.equal(ref[0], got[0][0]) and torch.equal(ref[1], got[0][1])
    qi = q.clone()
    qi[::3, ::5, 0] = float("inf")
    qi[1::3, ::7, 3] = -float("inf")
    got = bc.forest_beam_rows_cuda([(F, SP, W, bud, 16)], qi, qpos, lam, sigv, 0.125, 32, 2, "bound", 8.0)
    for legacy in (True, False):
        warp = bc.forest_beam_cuda(F, SP, qi, qpos, lam, sigv, 0.125, 32, W, bud, 16, 2, "bound", 8.0, legacy=legacy)
        assert torch.equal(warp[0], got[0][0]) and torch.equal(warp[1], got[0][1])


@gpu
def test_rows_compact_cells(monkeypatch):
    # fixed policy, fsb_delta 0: a frontier node holds at most one cell, so delta_max = 0 (one cell per slot) is the
    # delta_max = 2 layout without its always-empty slots (the compact output of the prefill plan, item 1.2)
    F, SP, q, qpos, lam, sigv = _forest_stage(17, 512)
    Fu, SPu, *_ = _forest_stage(17, 512, admit_all=False)
    W, dl, bud = _rows_args(F, SP, q, qpos, lam, sigv, 16, 0)
    Wu, du, budu = _rows_args(Fu, SPu, q, qpos, lam, sigv, 16, 0)
    beams = [(F, SP, W, bud, 16), (Fu, SPu, Wu, budu, 16)]
    full = bc.forest_beam_rows_cuda(beams, q, qpos, lam, sigv, 0.125, 32, 2, "bound", 8.0)
    comp = bc.forest_beam_rows_cuda(beams, q, qpos, lam, sigv, 0.125, 32, 0, "bound", 8.0)
    for (cf, nf), (cc, nc) in zip(full, comp):
        assert cc.shape[-1] == 16 and torch.equal(cf[..., ::4], cc) and torch.equal(nf, nc)
        assert bool((cf.view(*cf.shape[:2], 16, 4)[..., 1:] == -1).all())


@gpu
def test_rows_flag_and_dispatch(monkeypatch):
    # prefill rows go to the row-cooperative kernel; FSB_BEAM_ROWS=0 and decode-sized rows keep the warp kernel
    F, SP, q, qpos, lam, sigv = _forest_stage(23, 512, nbc=2)
    W, dl, bud = _rows_args(F, SP, q, qpos, lam, sigv, 16, 0)
    args = (F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, 16, 2, "bound", 8.0)
    ran = []
    real_rows, real_warp = bc.forest_beam_rows_cuda, bc.forest_beam_cuda
    monkeypatch.setattr(bc, "forest_beam_rows_cuda", lambda *a, **k: ran.append("rows") or real_rows(*a, **k))
    monkeypatch.setattr(bc, "forest_beam_cuda", lambda *a, **k: ran.append("warp") or real_warp(*a, **k))
    on = fast.forest_beam_fused(*args)
    monkeypatch.setenv("FSB_BEAM_ROWS", "0")
    off = fast.forest_beam_fused(*args)
    assert ran == ["rows", "warp"]
    assert torch.equal(on[0], off[0]) and torch.equal(on[1], off[1])
    monkeypatch.setenv("FSB_BEAM_ROWS", "1")
    ran.clear()
    fast.forest_beam_fused(F, SP, q[:, :4].contiguous(), qpos[:, :4].contiguous(), lam, sigv, 0.125, 32,
                           W[:, :4].contiguous(), dl[:, :4].contiguous(), 16, 2, "bound", 8.0)
    assert ran == ["warp"]


def _launch_spy(monkeypatch):
    # -> a function returning the sources of the kernels launched since its last call (bc.launch is the one entry)
    seen = []
    real = bc.launch
    monkeypatch.setattr(bc, "launch", lambda fn, *a, **k: seen.append(fn.value) or real(fn, *a, **k))

    def sources():
        by_fn = {f.value: key[1] for key, (f, _, _) in bc._funcs.items()}
        out = [by_fn[v] for v in seen]
        seen.clear()
        return out
    return sources


@gpu
def test_rows_flag_restores_9b4e9ff_path(monkeypatch):
    # FSB_BEAM_ROWS=0 launches 9b4e9ff's warp kernel (its exact source) for prefill and decode rows, one launch per
    # beam, and traverse_forest does not try the paired launch; the default runs the row kernel (one launch for both
    # beams) and the current warp kernel for decode rows. Same cells, visit counts and selections either way.
    F, SP, q, qpos, lam, sigv = _forest_stage(23, 512, nbc=2)
    Fu, SPu, *_ = _forest_stage(23, 512, nbc=2, admit_all=False)
    W, dl, _ = _rows_args(F, SP, q, qpos, lam, sigv, 16, 0)
    Wu, du, _ = _rows_args(Fu, SPu, q, qpos, lam, sigv, 16, 0)
    pre = (F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, 16, 2, "bound", 8.0)
    dec = (F, SP, q[:, :1].contiguous(), qpos[:, :1].contiguous(), lam, sigv, 0.125, 32, W[:, :1].contiguous(),
           dl[:, :1].contiguous(), 16, 2, "bound", 8.0)

    def trav():
        return fsb_tree.traverse_forest(F, SP, q, qpos, lam, sigv, 0.125, 32, W, dl, 16, 2, "bound", 8.0,
                                        union_width=16, union_W_max=16, union_F=Fu, union_SP=SPu, union_delta=du)
    warp = {lg: bc.beam_source(73, 64, 64, 16, 4, True, False, bc._NWARP, legacy=lg) for lg in (False, True)}
    rows_src = bc.rows_source(73, 64, 16, 4, True, False)[0]
    pairs = []
    real_pair = fast.forest_beam_pair
    monkeypatch.setattr(fast, "forest_beam_pair", lambda *a: pairs.append(1) or real_pair(*a))
    sources = _launch_spy(monkeypatch)
    got = {}
    for flag, want in (("0", ([warp[True]], [warp[True]], [warp[True]] * 2)),
                       ("1", ([rows_src], [warp[False]], [rows_src]))):
        monkeypatch.setenv("FSB_BEAM_ROWS", flag)
        got[flag] = []
        for run, w in zip((lambda: fast.forest_beam_fused(*pre), lambda: fast.forest_beam_fused(*dec), trav), want):
            got[flag].append(run())
            assert sources() == w, flag
        assert pairs == ([] if flag == "0" else [1])
    for a, b in zip(got["0"], got["1"]):
        assert all(torch.equal(x, y) for x, y in zip(a, b))


@gpu
def test_rows_cooperative_path_runs(monkeypatch):
    # the row kernel's blocks descend cooperatively on the test forests (one tile pass, and several with an 8-node tile):
    # none falls back to the warp-per-query descent, so a change that sent blocks there would fail here even though the
    # cells stay exact; 32- and 64-slot union tables do send blocks to the fallback, which continues their queries from
    # the level where the union overflowed (several levels here, never the root's) and stays exact
    F, SP, q, qpos, lam, sigv = _forest_stage(31, 512)
    Fu, SPu, *_ = _forest_stage(31, 512, admit_all=False)
    W, dl, bud = _rows_args(F, SP, q, qpos, lam, sigv, 16, 0)
    Wu, du, budu = _rows_args(Fu, SPu, q, qpos, lam, sigv, 16, 0)
    ref = [_ref_beam(monkeypatch, F, SP, q, qpos, lam, sigv, W, dl, 16, 2, "bound", 8.0),
           _ref_beam(monkeypatch, Fu, SPu, q, qpos, lam, sigv, Wu, du, 16, 2, "bound", 8.0)]
    beams = [(F, SP, W, bud, 16), (Fu, SPu, Wu, budu, 16)]
    nblk = 2 * q.shape[0] * -(-q.shape[1] // bc._ROWS_QB)
    levels = set()
    for kw, fell in ((dict(), False), (dict(cap=8), False), (dict(umax=32), True), (dict(umax=64), True)):
        fb = torch.zeros(65, dtype=torch.int32, device="cuda")
        got = bc.forest_beam_rows_cuda(beams, q, qpos, lam, sigv, 0.125, 32, 2, "bound", 8.0, fallback=fb, **kw)
        assert all(torch.equal(r[0], g[0]) and torch.equal(r[1], g[1]) for r, g in zip(ref, got)), kw
        n, by_level = int(fb[0]), fb[1:].tolist()
        assert sum(by_level) == n and by_level[0] == 0 and ((0 < n <= nblk) if fell else n == 0), (kw, n)
        levels |= {lv for lv, c in enumerate(by_level) if c}
    assert len(levels) >= 2, levels


@gpu
@pytest.mark.parametrize("rows", ["1", "0"])
def test_beam_cuda_graph_capture(monkeypatch, rows):
    # the CUDA beams inside a CUDA graph at head dim 64 (the graphs of tests/test_fsb_tree.py use head dim 16, which no
    # CUDA beam takes): traverse_forest with a union beam (one paired row-kernel launch, or two warp launches with
    # FSB_BEAM_ROWS=0) and decode-sized rows (the warp kernel) replay equal to eager, also after the captured inputs
    # change in place; a capture that meets an unbuilt row kernel does not build it and falls back exactly
    monkeypatch.setenv("FSB_BEAM_ROWS", rows)
    F, SP, q, qpos, lam, sigv = _forest_stage(31, 512, nbc=4)
    Fu, SPu, *_ = _forest_stage(31, 512, nbc=4, admit_all=False)
    W, dl, _ = _rows_args(F, SP, q, qpos, lam, sigv, 16, 0)
    _, du, _ = _rows_args(Fu, SPu, q, qpos, lam, sigv, 16, 0)
    qs, ps = q.clone(), qpos.clone()
    qd, pd = q[:, :3].clone(), qpos[:, :3].clone()
    Wd, dld = W[:, :3].contiguous(), dl[:, :3].contiguous()

    def run():
        sel, nv, _ = fsb_tree.traverse_forest(F, SP, qs, ps, lam, sigv, 0.125, 32, W, dl, 16, 2, "bound", 8.0,
                                              union_width=16, union_W_max=16, union_F=Fu, union_SP=SPu, union_delta=du)
        return (sel, nv) + tuple(fast.forest_beam_fused(F, SP, qd, pd, lam, sigv, 0.125, 32, Wd, dld, 16, 2, "bound",
                                                        8.0))

    def capture(warm=True):
        if warm:                                                             # torch.cuda.graph's side-stream warm-up
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                run()
            torch.cuda.current_stream().wait_stream(st)
        sources()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = run()
        return g, out, sources()

    eager = [t.clone() for t in run()]                                       # builds the kernels
    sources = _launch_spy(monkeypatch)
    warp = bc.beam_source(73, 64, 64, 16, 4, True, False, bc._NWARP, legacy=rows == "0")
    g, out, launched = capture()
    assert launched == ([bc.rows_source(73, 64, 16, 4, True, False)[0], warp] if rows == "1" else [warp] * 3)
    g.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(out, eager))
    gen = torch.Generator().manual_seed(77)
    for t in (qs, qd):
        t.add_(0.3 * torch.randn(t.shape, generator=gen).cuda())
    ps.add_(5.0)
    pd.add_(5.0)
    g.replay()
    torch.cuda.synchronize()
    eager2 = run()
    assert not torch.equal(eager2[0], eager[0]) and not torch.equal(eager2[2], eager[2])     # the selections moved
    assert all(torch.equal(a, b) for a, b in zip(out, eager2))
    if rows == "1":
        monkeypatch.setattr(bc, "_beam_fns", {k: v for k, v in bc._beam_fns.items() if k[0] != "rows"})
        g2, out2, launched = capture(warm=False)
        assert launched == [warp] * 3 and not any(k[0] == "rows" for k in bc._beam_fns)
        g2.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(out2, eager2))


@gpu
def test_nan_counts_dropped_by_every_cuda_beam(monkeypatch):
    # a node whose subtree count is NaN (the forest builders never write one) is neither expanded nor carried: the row
    # kernel drops it like both warp kernels and the per-level reference (the PyTorch loop, FSB_TRAVERSE_TRITON=0,
    # carries it, as on 9b4e9ff), also through several tile passes and the warp fallback, and for a NaN root
    F, SP, q, qpos, lam, sigv = _forest_stage(41, 512, nbc=4)
    W, dl, bud = _rows_args(F, SP, q, qpos, lam, sigv, 16, 0)
    D = q.shape[-1]
    clean = _ref_beam(monkeypatch, F, SP, q, qpos, lam, sigv, W, dl, 16, 2, "bound", 8.0)
    Fn, SPn = dict(F, node=F["node"].clone()), dict(SP, node=SP["node"].clone())
    Fn["node"][:, 3::7, D + 3] = float("nan")                                # every 7th chunk node
    SPn["node"][::5, :, D + 3] = float("nan")                                # every spine node (and root) of some rows
    ref = _ref_beam(monkeypatch, Fn, SPn, q, qpos, lam, sigv, W, dl, 16, 2, "bound", 8.0)
    assert not torch.equal(ref[0], clean[0]) and bool((ref[0][::5] == -1).all())
    beams = [(Fn, SPn, W, bud, 16)]
    got = {"rows": bc.forest_beam_rows_cuda(beams, q, qpos, lam, sigv, 0.125, 32, 2, "bound", 8.0)[0],
           "rows_2node_tile": bc.forest_beam_rows_cuda(beams, q, qpos, lam, sigv, 0.125, 32, 2, "bound", 8.0,
                                                       cap=2)[0],
           "rows_fallback": bc.forest_beam_rows_cuda(beams, q, qpos, lam, sigv, 0.125, 32, 2, "bound", 8.0,
                                                     umax=32)[0]}
    for lg in (False, True):
        got[f"warp_legacy={lg}"] = bc.forest_beam_cuda(Fn, SPn, q, qpos, lam, sigv, 0.125, 32, W, bud, 16, 2, "bound",
                                                       8.0, legacy=lg)
    for k, v in got.items():
        assert torch.equal(ref[0], v[0]) and torch.equal(ref[1], v[1]), k


@gpu
@pytest.mark.parametrize("Bq", [1, 3])
@pytest.mark.parametrize("dmax", [0, 2])
@pytest.mark.parametrize("rc_tensor", [False, True])
def test_decode_pair_one_launch(monkeypatch, Bq, dmax, rc_tensor):
    # decode and small-row beam pairs (Bq < 32): both beams run in one launch of fsb_beam_pair_cuda and match two
    # reference calls bitwise; FSB_BEAM_ROWS=0 falls back to two single-beam legacy launches
    F, SP, q, qpos, lam, sigv = _forest_stage(37 + Bq + dmax, 512, nbc=1)
    Fu, SPu, *_ = _forest_stage(37 + Bq + dmax, 512, nbc=1, admit_all=False)
    q, qpos = q[:, :Bq].contiguous(), qpos[:, :Bq].contiguous()
    rows = q.shape[0]
    rc = torch.rand(rows, generator=torch.Generator().manual_seed(3)).cuda() * 8 + 1 if rc_tensor else 8.0
    W, dl, bud = _rows_args(F, SP, q, qpos, lam, sigv, 16, 0)
    Wu, du, budu = _rows_args(Fu, SPu, q, qpos, lam, sigv, 8, min(1, dmax))
    ref_a = _ref_beam(monkeypatch, F, SP, q, qpos, lam, sigv, W, dl, 16, dmax, "bound", rc)
    ref_u = _ref_beam(monkeypatch, Fu, SPu, q, qpos, lam, sigv, Wu, du, 16, dmax, "bound", rc)
    sources = _launch_spy(monkeypatch)
    pair = fast.forest_beam_pair(F, SP, W, dl, 16, Fu, SPu, Wu, du, 16, q, qpos, lam, sigv, 0.125, 32, dmax, "bound",
                                 rc, bud=bud, bud_u=budu)
    assert pair is not None and len(sources()) == 1
    assert torch.equal(ref_a[0], pair[0][0]) and torch.equal(ref_a[1], pair[0][1])
    assert torch.equal(ref_u[0], pair[1][0]) and torch.equal(ref_u[1], pair[1][1])
    monkeypatch.setenv("FSB_BEAM_ROWS", "0")
    assert fast.forest_beam_pair(F, SP, W, dl, 16, Fu, SPu, Wu, du, 16, q, qpos, lam, sigv, 0.125, 32, dmax, "bound",
                                 rc, bud=bud, bud_u=budu) is None

