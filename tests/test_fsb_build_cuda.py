import ctypes
import math
import os
import time

import pytest
import torch

from src.fsattn import fsb_tree as ft
from src.fsattn.fsb_build_fast import _HAVE_TRITON

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and _HAVE_TRITON),
                                reason="the CUDA forest build is compared against the Triton builder on CUDA")
DEV = "cuda"

if _HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _div_sqrt_ref(X, Y, Q, S, N: tl.constexpr):
        # the builder's tl.math.div_rn / tl.sqrt_rn, compiled the way fsb_build_fast compiles them
        i = tl.program_id(0) * N + tl.arange(0, N)
        x = tl.load(X + i)
        y = tl.load(Y + i)
        tl.store(Q + i, tl.math.div_rn(x, y))
        tl.store(S + i, tl.sqrt_rn(tl.abs(x)))


def _bc():
    from src.fsattn import fsb_build_cuda as bc
    return bc


def _bits(t):
    t = t.contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _same(a, b):
    if torch.is_tensor(a):
        return a.shape == b.shape and a.dtype == b.dtype and a.is_contiguous() == b.is_contiguous() \
            and torch.equal(_bits(a), _bits(b))
    return a == b


def _keys(kind, T, M, Dh, g):
    K = torch.randn(T, M, Dh, generator=g)
    if kind == "ties":
        K = (K * 2).round() / 2
    elif kind == "zeros":
        K = torch.zeros(T, M, Dh)
    elif kind == "negzero":
        K = torch.where(torch.rand(T, M, Dh, generator=g) < 0.5, -torch.zeros(T, M, Dh), K)
    elif kind == "dups":
        K = K[:, torch.randint(0, max(1, M // 3), (M,), generator=g)]
    elif kind == "subnormal":
        K = K * 1e-39
    return K


def _taus(kind, ce, H, g):
    T = ce.shape[0]
    if kind == "zero":
        return torch.zeros(H)
    if kind == "q50":                                     # the union-tau forest: about half the cells admitted
        return ce.view(T // H, H, -1).transpose(0, 1).reshape(H, -1).float().quantile(0.5, dim=-1).cpu()
    if kind == "rand":
        return torch.rand(H, generator=g)
    if kind == "none":
        return torch.full((H,), 2.0)
    return torch.full((H,), float(ce.min()) + 1e-7)       # "allbut1": every cell but the smallest ceiling


def _reference(K, ce, tau, H, E, sp):
    T, M, _ = K.shape
    admit = ce >= tau.view(1, H, 1).expand(T // H, H, 1).reshape(T, 1)
    F = ft.build_forest(K, ce, admit, spread=sp)
    mb = torch.searchsorted(F["adm"].masked_fill(F["adm"] < 0, M), E[None, :].expand(T, -1).contiguous())
    return F, mb


def _check(F, mb, G, gb):
    bad = [k for k in ("adm", "m_tot", "node", "leaf", "Ntot", "Ltot", "bases", "nodes_per", "leafbases", "Cmaxs",
                       "ks", "Kmax", "M", "T", "Dh") if not _same(F[k], G[k])]
    if not _same(mb, gb):
        bad.append("m_b")
    return bad


SHAPES = [(16, 512, 64), (16, 4096, 64), (16, 2048, 64), (16, 1000, 64), (4, 7, 64), (2, 8192, 64), (8, 300, 32),
          (2, 1024, 32), (1, 8192, 32)]


@pytest.mark.parametrize("T,M,Dh", SHAPES)
@pytest.mark.parametrize("kind", ["randn", "ties", "zeros", "negzero", "dups", "subnormal"])
def test_forests_cuda_match_build_forest(monkeypatch, T, M, Dh, kind):
    # both forests of one call (as prefill_forests builds them) against build_forest + searchsorted, bit for bit
    bc = _bc()
    monkeypatch.setenv("FSB_BUILD_GROUP", "4")
    g = torch.Generator().manual_seed(31 * M + T + Dh)
    K = _keys(kind, T, M, Dh, g).to(DEV)
    ce = torch.rand(T, M, generator=g).to(DEV)
    spread = torch.rand(T, M, generator=g).to(DEV)
    H = 4 if T % 4 == 0 else T
    E = torch.randint(0, M + 1, (37,), generator=g).sort().values.to(DEV)
    for pair in (("zero", "q50"), ("rand", "none"), ("allbut1", "zero")):
        taus = [_taus(p, ce, H, g).float().to(DEV) for p in pair]
        for sp in (None, spread):
            got = bc.forests_cuda(K, ce, sp, taus, E)
            assert got is not None
            for tau, (G, gb) in zip(taus, got):
                F, mb = _reference(K, ce, tau, H, E, sp)
                assert _check(F, mb, G, gb) == [], (pair, sp is None)


@pytest.mark.parametrize("group", ["1", "3", "0"])
@pytest.mark.parametrize("T,M,Dh", [(16, 512, 64), (16, 1000, 64), (4, 7, 64), (8, 300, 32), (2, 2048, 64)])
def test_forests_cuda_groupings(monkeypatch, group, T, M, Dh):
    # the padded group build's signed zeros depend on FSB_BUILD_GROUP; the kernels take its Kpad table
    bc = _bc()
    monkeypatch.setenv("FSB_BUILD_GROUP", group)
    for kind in ("randn", "negzero", "zeros"):
        g = torch.Generator().manual_seed(7 * M + int(group))
        K = _keys(kind, T, M, Dh, g).to(DEV)
        ce = torch.rand(T, M, generator=g).to(DEV)
        H = 4 if T % 4 == 0 else T
        E = torch.randint(0, M + 1, (17,), generator=g).sort().values.to(DEV)
        taus = [_taus("zero", ce, H, g).to(DEV), _taus("q50", ce, H, g).float().to(DEV)]
        for tau, (G, gb) in zip(taus, bc.forests_cuda(K, ce, None, taus, E)):
            F, mb = _reference(K, ce, tau, H, E, None)
            assert _check(F, mb, G, gb) == [], kind


@pytest.mark.parametrize("T,M,Dh", [(4, 7, 64), (16, 100, 64), (8, 300, 32), (4, 1000, 64)])
def test_forests_cuda_match_torch_reference(monkeypatch, T, M, Dh):
    # against the torch builder (FSB_FAST_BUILD=0) too, not only the Triton one
    bc = _bc()
    g = torch.Generator().manual_seed(M + Dh)
    for kind in ("randn", "negzero", "dups"):
        K = _keys(kind, T, M, Dh, g).to(DEV)
        ce = torch.rand(T, M, generator=g).to(DEV)
        H = 4 if T % 4 == 0 else T
        E = torch.randint(0, M + 1, (9,), generator=g).sort().values.to(DEV)
        tau = _taus("rand", ce, H, g).to(DEV)
        (G, gb), = bc.forests_cuda(K, ce, None, [tau], E)
        monkeypatch.setenv("FSB_FAST_BUILD", "0")
        F, mb = _reference(K, ce, tau, H, E, None)
        monkeypatch.setenv("FSB_FAST_BUILD", "1")
        assert _check(F, mb, G, gb) == [], kind


class _Attn:
    def __init__(self, H, tau, tau_u):
        self.fsb_tau = tau
        self.fsb_tau_union = tau_u
        self.fsb_union_tau_quantile = 0.5 if tau_u is not None else 0.0
        self.fsb_read_top = 4
        self.fsb_hybrid = "off"


@pytest.mark.parametrize("union", [False, True])
def test_prefill_forests_flag(monkeypatch, union):
    # prefill_forests with FSB_BUILD_CUDA=1 equals the FSB_BUILD_CUDA=0 chain (build_forest, searchsorted, utau_build)
    T, M, Dh, H = 16, 1500, 64, 16
    g = torch.Generator().manual_seed(5)
    K = torch.randn(T, M, Dh, generator=g).to(DEV)
    ce = (torch.rand(T, M, generator=g) + 0.5).to(DEV)
    tau_u = ce.view(1, H, M)[0].quantile(0.5, dim=-1).contiguous() if union else None
    attn = _Attn(H, torch.zeros(H, device=DEV), tau_u if union else torch.zeros(H, device=DEV))
    if not union:
        attn.fsb_union_tau_quantile = 0.0
    E = (torch.arange(M * 32 // 128, device=DEV) * 4).long()
    calls = []
    build_forest = ft.build_forest
    monkeypatch.setattr(ft, "build_forest", lambda *a, **k: calls.append(1) or build_forest(*a, **k))
    out, n_ref = {}, {}
    for flag in ("0", "1"):
        monkeypatch.setenv("FSB_BUILD_CUDA", flag)
        calls.clear()
        out[flag] = ft.prefill_forests(attn, K, ce, E)
        n_ref[flag] = len(calls)
    assert n_ref == {"0": 2 if union else 1, "1": 0}       # the CUDA build ran (and replaced every build_forest)
    (F0, mb0), (F1, mb1) = out["0"], out["1"]
    assert _check(F0, mb0, F1, mb1) == []
    assert F0["chunks"] == {} and F1["chunks"] == {}
    assert ("utau_F" in F0) == ("utau_F" in F1) == union
    if union:
        assert _check(F0["utau_F"], F0["utau_mba"], F1["utau_F"], F1["utau_mba"]) == []
        monkeypatch.setenv("FSB_BUILD_CUDA", "1")
        F2, m2, _ = ft.utau_build(attn, K, ce, E)
        assert _check(F0["utau_F"], F0["utau_mba"], F2, m2) == []


def test_forests_cuda_graph_capture():
    # the build launches no host syncs: a captured build replays bit-identically on new inputs
    bc = _bc()
    T, M, Dh, H = 16, 2500, 64, 16
    g = torch.Generator().manual_seed(11)
    K = torch.randn(T, M, Dh, generator=g).to(DEV)
    ce = torch.rand(T, M, generator=g).to(DEV)
    taus = [torch.zeros(H, device=DEV), torch.full((H,), 0.5, device=DEV)]
    E = (torch.arange(M // 4, device=DEV) * 4).long()
    bc.forests_cuda(K, ce, None, taus, E)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = bc.forests_cuda(K, ce, None, taus, E)
    K.copy_(torch.randn(T, M, Dh, generator=g).to(DEV))
    ce.copy_(torch.rand(T, M, generator=g).to(DEV))
    graph.replay()
    torch.cuda.synchronize()
    for tau, (G, gb) in zip(taus, out):
        F, mb = _reference(K, ce, tau, H, E, None)
        assert _check(F, mb, G, gb) == []


def test_instruction_flavours():
    # tl.math.div_rn / tl.sqrt_rn compile to div.rn.ftz / sqrt.rn.ftz; the kernels' mean multiplies by 2^-p
    # with mul.rn.ftz, bit-identical to div.rn.ftz by 2^p (random, subnormal-boundary and signed-zero inputs)
    import ctypes
    from src.fsattn import fsb_beam_cuda as fb
    bc = _bc()
    _ref = _div_sqrt_ref
    src = bc._source(64) + r'''
extern "C" __global__ void probe(const float* X, const float* Y, float* Q, float* S, float* P, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Q[i] = f_div_ftz(X[i], Y[i]);
    S[i] = f_sqrt_ftz(fabsf(X[i]));
    const int p = i % 14;
    P[i] = f_mul_ftz(X[i], f_pow2neg(p));
}
'''
    dev = torch.device(DEV, torch.cuda.current_device())
    fn = fb.load_function(dev, src, "probe")
    g = torch.Generator().manual_seed(0)
    n = 1 << 16
    edge = torch.tensor([0.0, -0.0, 1e-38, -1e-38, 1.1754942e-38, 1.1754944e-38, 1e-45, -1e-45, 3e-39, 1e-6,
                         2.0 ** -126, -(2.0 ** -126), 2.0 ** -127, 1.0, -1.0, 3.4e38])
    X = torch.cat([edge, torch.randn(n - 2 * edge.numel(), generator=g) * torch.exp2(torch.randint(-140, 20, (n - 2 * edge.numel(),), generator=g).float()), edge])
    Y = torch.cat([torch.full((edge.numel(),), 1e-6), (torch.rand(n - 2 * edge.numel(), generator=g) * 4 + 1e-6), edge.abs() + 1e-6])
    X, Y = X.to(DEV).contiguous(), Y.to(DEV).contiguous()
    Qr, Sr = torch.empty_like(X), torch.empty_like(X)
    _ref[(n // 1024,)](X, Y, Qr, Sr, N=1024, enable_fp_fusion=False)
    Q, S, P = torch.empty_like(X), torch.empty_like(X), torch.empty_like(X)
    args = [ctypes.c_void_p(t.data_ptr()) for t in (X, Y, Q, S, P)] + [ctypes.c_int(n)]
    fb.launch(fn, n // 256, 256, dev, args)
    p = torch.arange(n, device=DEV) % 14
    Pr = torch.empty_like(X)
    _ref[(n // 1024,)](X, torch.exp2(p.float()), Pr, Sr.clone(), N=1024, enable_fp_fusion=False)
    torch.cuda.synchronize()
    assert torch.equal(_bits(Q), _bits(Qr))
    assert torch.equal(_bits(S), _bits(Sr))
    assert torch.equal(_bits(P), _bits(Pr))


# ---- fixes from the review of the CUDA build: memory safety, fallbacks, input coverage ----

def _inputs(T, M, Dh, seed, H=None):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(T, M, Dh, generator=g).to(DEV)
    ce = torch.rand(T, M, generator=g).to(DEV)
    H = H or (4 if T % 4 == 0 else T)
    E = torch.randint(0, M + 1, (33,), generator=g).sort().values.to(DEV)
    taus = [torch.zeros(H, device=DEV), _taus("q50", ce, H, g).float().to(DEV)]
    return K, ce, taus, E, H


def _exact(K, ce, sp, taus, E, H, got):
    assert got is not None
    assert len(got) == len(taus)
    for tau, (G, gb) in zip(taus, got):
        assert _check(*_reference(K, ce, tau, H, E, sp), G, gb) == []


@pytest.mark.parametrize("Dh", [32, 64])
def test_smem_covers_every_task_kind(Dh):
    # the build kernel's dynamic shared memory covers the carve-up of every task kind (batches and 512-row subtrees:
    # carve; nodes of >= 1024 rows: gcarve), measured on the device code
    from src.fsattn import fsb_beam_cuda as fb
    bc = _bc()
    src = bc._source(Dh) + r'''
extern "C" __global__ void carve_ends(int* out) {
    extern __shared__ float smem[];
    out[0] = (int)((char*)carve(smem).end - (char*)smem);
    out[1] = (int)((char*)gcarve(smem).end - (char*)smem);
}
'''
    dev = torch.device(DEV, torch.cuda.current_device())
    out = torch.zeros(2, dtype=torch.int32, device=DEV)
    fb.launch(fb.load_function(dev, src, "carve_ends"), 1, 1, dev, [ctypes.c_void_p(out.data_ptr())])
    torch.cuda.synchronize()
    c_end, g_end = out.tolist()
    assert 0 < c_end <= bc._smem_bytes(Dh) and 0 < g_end <= bc._smem_bytes(Dh), (c_end, g_end, bc._smem_bytes(Dh))


def test_forests_cuda_graph_replay_after_larger_build():
    # a captured build owns its control words: a later eager build with more G-node flags than any earlier call (the
    # old per-device flag buffer was reallocated, and the captured one freed, then) neither breaks the graph's replays
    # nor lets them write into memory handed out since
    bc = _bc()
    K1, c1, taus1, E1, H1 = _inputs(2, 2048, 64, seed=12, H=2)
    bc.forests_cuda(K1, c1, None, taus1, E1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out1 = bc.forests_cuda(K1, c1, None, taus1, E1)
    K2, c2, taus2, E2, _ = _inputs(42, 8192, 32, seed=13, H=2)          # 42 heads x 49 G nodes x 2 forests = 4116 flags
    assert bc.forests_cuda(K2, c2, None, taus2, E2) is not None
    del K2, c2
    torch.cuda.synchronize()
    victims = [torch.full((4096,), 7, dtype=torch.int32, device=DEV) for _ in range(1024)]
    g = torch.Generator().manual_seed(14)
    for _ in range(3):
        K1.copy_(torch.randn(K1.shape, generator=g).to(DEV))
        c1.copy_(torch.rand(c1.shape, generator=g).to(DEV))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(torch.stack(victims).eq(7).all())
        _exact(K1, c1, None, taus1, E1, H1, out1)


def test_forests_cuda_concurrent_streams():
    # builds on two streams released by one event, so both prologues run before either build kernel: each call has its
    # own control words, so neither build takes the other's completion flags for its own. The second build is the
    # larger, its slot-list scratch starts as junk (a task that skipped its wait would read it)
    bc = _bc()
    cases = [_inputs(16, 1024, 64, seed=20), _inputs(16, 16383, 64, seed=21)]
    for K, ce, taus, E, H in cases:
        assert bc.forests_cuda(K, ce, None, taus, E) is not None             # kernels and task tables ready
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in cases]
    gate = torch.cuda.Stream()
    for _ in range(4):
        with torch.cuda.stream(gate):
            torch.cuda._sleep(50_000_000)
            go = torch.cuda.Event()
            go.record()
        outs, done = [], []
        for st, (K, ce, taus, E, H) in zip(streams, cases):
            st.wait_event(go)
            with torch.cuda.stream(st):
                T, M = K.shape[:2]
                nbig = sum(1 for k in range(10, 14) if M >> k)
                junk = torch.full((nbig * 2 * T * M,), -7, dtype=torch.int32, device=DEV)
                del junk
                outs.append(bc.forests_cuda(K, ce, None, taus, E))
                done.append(torch.cuda.Event())
                done[-1].record()
        t0 = time.time()
        while not all(ev.query() for ev in done):
            assert time.time() - t0 < 120, "builds on two streams did not finish"
            time.sleep(0.01)
        for st in streams:
            torch.cuda.current_stream().wait_stream(st)
        for (K, ce, taus, E, H), got in zip(cases, outs):
            _exact(K, ce, None, taus, E, H, got)


def test_forests_cuda_oom_is_raised_and_keeps_the_path():
    # an out-of-memory error inside the build reaches the caller (as the old builder's would); with memory available
    # again the next call still runs the CUDA build
    bc = _bc()
    K, ce, taus, E, H = _inputs(16, 8192, 64, seed=30)
    assert bc.forests_cuda(K, ce, None, taus, E) is not None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    hog = torch.empty(torch.cuda.mem_get_info()[0] - (512 << 20), dtype=torch.uint8, device=DEV)
    try:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            bc.forests_cuda(K, ce, None, taus, E)
    finally:
        del hog
        torch.cuda.empty_cache()
    _exact(K, ce, None, taus, E, H, bc.forests_cuda(K, ce, None, taus, E))


def test_forests_cuda_transient_failure_does_not_latch(monkeypatch):
    # a failure of one call (here a launch error) falls back for that call only
    bc = _bc()
    K, ce, taus, E, H = _inputs(4, 1000, 64, seed=31)
    launch = bc._launch

    def failing(*a, **k):
        raise RuntimeError("cuLaunchKernel: CUDA error 999 (unknown error, test)")

    monkeypatch.setattr(bc, "_launch", failing)
    with pytest.warns(UserWarning):
        assert bc.forests_cuda(K, ce, None, taus, E) is None
    monkeypatch.setattr(bc, "_launch", launch)
    _exact(K, ce, None, taus, E, H, bc.forests_cuda(K, ce, None, taus, E))


def test_forests_cuda_setup_failure_disables_that_variant_only(monkeypatch):
    # a kernel that cannot be compiled or loaded falls back from then on, with one warning, for that device and head
    # size (here a variant of this test's own) only
    from src.fsattn import fsb_beam_cuda as fb
    bc = _bc()
    if isinstance(getattr(bc, "_failed", None), dict):
        monkeypatch.setattr(bc, "_failed", {})
    K, ce, taus, E, H = _inputs(4, 100, 64, seed=32)
    tries = []

    def broken(*a, **k):
        tries.append(1)
        raise RuntimeError("NVRTC compile failed: (test)")

    with monkeypatch.context() as m:
        m.setitem(bc._DEBUG_DEFS, "FSB_TEST_VARIANT", 1)
        m.setattr(fb, "load_function", broken)
        with pytest.warns(UserWarning):
            assert bc.forests_cuda(K, ce, None, taus, E) is None
        assert bc.forests_cuda(K, ce, None, taus, E) is None and len(tries) == 1
    _exact(K, ce, None, taus, E, H, bc.forests_cuda(K, ce, None, taus, E))


def test_fast_build_off_selects_the_reference_builder(monkeypatch):
    # FSB_FAST_BUILD=0 restores the reference builder (README), so the CUDA build steps aside as well
    bc = _bc()
    monkeypatch.setenv("FSB_FAST_BUILD", "0")
    K, ce, taus, E, H = _inputs(16, 600, 64, seed=40, H=16)
    assert bc.forests_cuda(K, ce, None, taus, E) is None
    calls = []
    build_forest = ft.build_forest
    monkeypatch.setattr(ft, "build_forest", lambda *a, **k: calls.append(1) or build_forest(*a, **k))
    F, mb = ft.prefill_forests(_Attn(H, taus[0], None), K, ce, E)
    assert len(calls) == 1


def test_forests_cuda_input_guards():
    # inputs the kernels do not cover are refused (the caller runs the old builder); covered ones build exactly
    bc = _bc()
    K, ce, taus, E, H = _inputs(4, 3000, 64, seed=50)
    # Kbar 4 bytes past a 16-byte boundary (rows are loaded as float4): refused, or built exactly
    Km = torch.empty(K.numel() + 1, device=DEV)[1:].view(K.shape)
    Km.copy_(K)
    got = bc.forests_cuda(Km, ce, None, taus, E)
    torch.cuda.synchronize()
    if got is not None:
        _exact(K, ce, None, taus, E, H, got)
    # a spread of another shape, thresholds of different lengths: refused
    assert bc.forests_cuda(K, ce, torch.rand(4, 2 * 3000, device=DEV), taus, E) is None
    assert bc.forests_cuda(K, ce, torch.rand(4, 3000, 1, device=DEV), taus, E) is None
    assert bc.forests_cuda(K, ce, None, [taus[0], taus[1][:2]], E) is None
    _exact(K, ce, None, taus, E, H, bc.forests_cuda(K, ce, None, taus, E))


def test_forests_cuda_fp64_tau_matches_reference():
    # the reference compares the fp32 ceilings with an fp64 threshold in fp64: thresholds strictly between two fp32
    # values, fp64 subnormals and values beyond the fp32 range admit the same cells in the CUDA build
    bc = _bc()
    K, ce, _, E, H = _inputs(4, 1500, 64, seed=51)
    ce[:, :3] = 0.0
    c = ce[:H, 7].double()
    for tau in (c + 2.0 ** -40, c - 2.0 ** -40, torch.full((H,), 1e-310, dtype=torch.float64, device=DEV),
                torch.tensor([1e39, -1e39, math.inf, -1e-310], dtype=torch.float64, device=DEV)):
        _exact(K, ce, None, [tau], E, H, bc.forests_cuda(K, ce, None, [tau], E))


def test_forests_cuda_empty_batch():
    # T = 0 heads: empty tables of the usual layout, and the CUDA build stays on
    bc = _bc()
    M, Dh, H = 100, 64, 4
    E = torch.arange(10, device=DEV).long()
    taus = [torch.zeros(H, device=DEV), torch.full((H,), 0.5, device=DEV)]
    got = bc.forests_cuda(torch.zeros(0, M, Dh, device=DEV), torch.zeros(0, M, device=DEV), None, taus, E)
    assert got is not None and len(got) == 2
    lay = bc.forest_layout(M)
    for F, mb in got:
        assert F["T"] == 0 and tuple(F["adm"].shape) == (0, M) and tuple(F["m_tot"].shape) == (0,)
        assert tuple(F["node"].shape) == (0, lay["Ntot"], Dh + 9) and tuple(F["leaf"].shape) == (0, lay["Ltot"])
        assert tuple(mb.shape) == (0, 10) and mb.dtype == torch.long
    K, ce, taus, E, H = _inputs(4, M, Dh, seed=52)
    _exact(K, ce, None, taus, E, H, bc.forests_cuda(K, ce, None, taus, E))


@pytest.mark.parametrize("T,M,Dh", [(4, 700, 64), (8, 300, 32), (16, 1500, 64)])
def test_forests_cuda_padded_M_matches_unpadded(monkeypatch, T, M, Dh):
    # the decode adapter pads (T, M) with never-admitted cells to a power of two before building: the padded build
    # admits the same cells (padded adm is -1 past them), and every complete chunk matches the unpadded build's
    # (node rows up to the chunk's base offset, leaf rows as they are)
    bc = _bc()
    g = torch.Generator().manual_seed(91 * M + T + Dh)
    K = torch.randn(T, M, Dh, generator=g).to(DEV)
    ce = torch.rand(T, M, generator=g).to(DEV)
    sp = torch.rand(T, M, generator=g).to(DEV)
    H = 4 if T % 4 == 0 else T
    taus = [_taus("q50", ce, H, g).float().to(DEV)]
    E = torch.randint(0, M + 1, (13,), generator=g).sort().values.to(DEV)
    M_pad = 1 << (M - 1).bit_length()
    Kp = torch.zeros(T, M_pad, Dh, device=DEV); Kp[:, :M] = K
    cep = torch.full((T, M_pad), float("-inf"), device=DEV); cep[:, :M] = ce
    spp = torch.zeros(T, M_pad, device=DEV); spp[:, :M] = sp

    def rebased(F, t, k, j):
        size = 1 << k; nodes = 2 * size - 1
        nb, lb = F["bases"][k] + j * nodes, F["leafbases"][k] + j * size
        row = F["node"][t, nb:nb + nodes].clone()
        cc = row[:, Dh + 4:Dh + 6]
        row[:, Dh + 4:Dh + 6] = torch.where(cc >= 0, cc - float(nb), cc)
        row[:, Dh + 7] -= float(lb)
        return row, F["leaf"][t, lb:lb + size]

    for sp0, sp1 in ((None, None), (sp, spp)):
        (F0, mb0), = bc.forests_cuda(K, ce, sp0, taus, E)
        (F1, mb1), = bc.forests_cuda(Kp, cep, sp1, taus, E)
        assert torch.equal(F0["m_tot"], F1["m_tot"])
        assert torch.equal(F0["adm"], F1["adm"][:, :M])
        assert (F1["adm"][:, M:] == -1).all()
        assert torch.equal(mb0, mb1)
        for t in range(T):
            mt = int(F0["m_tot"][t])
            for k in F0["ks"]:
                for j in range(mt >> k):
                    r0, l0 = rebased(F0, t, k, j)
                    r1, l1 = rebased(F1, t, k, j)
                    assert torch.equal(_bits(r0), _bits(r1)) and torch.equal(l0, l1), (t, k, j, sp0 is None)


def test_decode_tables_bounded_over_padded_growth(monkeypatch):
    # decode ingests at growing lengths: the power-of-two M buckets keep the live task tables at one per bucket
    bc = _bc()
    monkeypatch.setattr(bc, "_tables", type(bc._tables)())
    monkeypatch.setenv("FSB_BUILD_CUDA", "1")
    monkeypatch.setenv("FSB_FAST_CACHE", "1")
    torch.manual_seed(133)
    rows, cells, dim = 4, 2200, 64
    keys = torch.randn(1, rows, cells, dim, device=DEV)
    ceilings = torch.rand(1, rows, cells, device=DEV)
    tau = torch.zeros(rows, device=DEV)
    cache = ft.ForestCache(rows, dim, DEV, lean=True)
    for visible in (400, 700, 1500, 2200):                 # buckets 512, 1024, 2048, 4096
        cache.ingest(keys, ceilings, visible, tau)
    assert {k[1] for k in bc._tables} == {512, 1024, 2048, 4096}
    assert len(bc._tables) <= 4 <= bc._TABLES_MAX


def test_task_tables_are_bounded():
    # the per-shape task tables form a small LRU: builds at many prompt lengths do not accumulate GPU memory
    bc = _bc()
    H = 16
    taus = [torch.zeros(H, device=DEV), torch.full((H,), 0.5, device=DEV)]

    def build(M):
        K = torch.randn(16, M, 64, device=DEV)
        ce = torch.rand(16, M, device=DEV)
        assert bc.forests_cuda(K, ce, None, taus, (torch.arange(M // 4, device=DEV) * 4).long()) is not None
        torch.cuda.synchronize()

    build(9000)
    base = torch.cuda.memory_allocated()
    build(9001)
    one = torch.cuda.memory_allocated() - base                             # one more shape's task table
    for M in range(9002, 9042):
        build(M)
    assert 0 < one and torch.cuda.memory_allocated() - base <= 12 * one, (one, torch.cuda.memory_allocated() - base)


def test_captured_build_survives_many_later_shapes():
    # a graph keeps using the task table it captured: tables of captured shapes stay allocated however many other
    # shapes are built afterwards
    bc = _bc()
    K1, c1, taus1, E1, H1 = _inputs(4, 3000, 64, seed=60)
    bc.forests_cuda(K1, c1, None, taus1, E1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out1 = bc.forests_cuda(K1, c1, None, taus1, E1)
    for M in range(2000, 2040):
        K, ce, taus, E, H = _inputs(2, M, 64, seed=M, H=2)
        assert bc.forests_cuda(K, ce, None, taus, E) is not None
    torch.cuda.synchronize()
    junk = [torch.full((n,), -1, dtype=torch.int32, device=DEV) for n in (16, 64, 256, 1024, 4096, 16384) for _ in range(64)]
    graph.replay()
    done = torch.cuda.Event()
    done.record()
    t0 = time.time()
    while not done.query():                                                # a freed table would hang or fault the replay
        assert time.time() - t0 < 60, "the replay did not finish: the captured task table was freed"
        time.sleep(0.01)
    del junk
    _exact(K1, c1, None, taus1, E1, H1, out1)
