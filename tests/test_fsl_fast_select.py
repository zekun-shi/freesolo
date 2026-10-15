"""FS-L fast cell selection (src/fsattn/fsl_select_cuda.py) against the legacy torch glue + Triton select: the selected
cells (sel_full) must be identical, bit for bit, including ties, padded queries and chunked blocks."""
import math
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import triton
import triton.language as tl

from src.fsattn.attention import FSAttn
from src.fsattn import fsl_select_cuda as fsc

DEV = "cuda"


@triton.jit
def _erf_k(OUT, base, step, n, BLOCK: tl.constexpr):
    # tl.erf of the floats with bit patterns base + i * step (the erf the legacy Triton select kernel uses)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = (base + i.to(tl.int64) * step).to(tl.int32).to(tl.float32, bitcast=True)
    tl.store(OUT + i, tl.erf(x).to(tl.int32, bitcast=True), mask=i < n)


def _attn(H, D, *, content_only=False, chunk_gib=None, dtype=torch.bfloat16, seed=0):
    a = FSAttn(d_model=H * D, n_heads=H, window_size=512, causal=True, long_range_variant="cell_topk", cell_stride=32,
               cell_topk=4, cell_select="score", sel_strict_causal=1, use_rope=False, fs_uniform_sr=True,
               rope_long_range=False, cell_prune_kernel=1, cell_prune_pool=64, cell_prune_window=8,
               cell_prune_blkchunk=16, fs_fused_attn=1, cell_fused_select=1, sel_content_only=content_only)
    if chunk_gib is not None:
        a.cell_fused_chunk_gib = chunk_gib
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        a.sel_scale.copy_(0.5 + torch.rand(H, generator=g) * 2)
        a.log_sigma.copy_(torch.log(1.0 + torch.rand(H, generator=g) * 40))
    return a.to(DEV, dtype).eval()


def _qk(B, H, N, D, kind, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    if kind == "randn":
        x = torch.randn(B, N, 2, H, D, device=DEV, generator=g)
    elif kind == "vocab":          # tokens from a small vocabulary: equal K rows, tied cell ceilings
        voc = torch.randn(40, 2, H, D, device=DEV, generator=g)
        x = voc[torch.randint(0, 40, (B, N), device=DEV, generator=g)]
    elif kind == "periodic":       # the same 32-token cell over and over: equal cell means, ceilings and far scores
        voc = torch.randn(32, 2, H, D, device=DEV, generator=g)
        x = voc[(torch.arange(N, device=DEV) % 32).expand(B, N)]
    elif kind == "lowprec":        # few distinct bf16 values: many exactly tied content scores
        x = torch.randint(-2, 3, (B, N, 2, H, D), device=DEV, generator=g).float() * 0.25
    else:
        raise ValueError(kind)
    x = x.to(torch.bfloat16)
    # the model's layout: (B,N,H,D) projections viewed as (B,H,N,D)
    Q = x[:, :, 0].transpose(1, 2)
    K = x[:, :, 1].transpose(1, 2)
    return Q, K


def _sel(a, Q, K, fast, fused="1", sigma=None):
    # sel_full of _cell_score_On_chunked_strict; sigma None: a.sigma, evaluated under autocast (fp32) as the model does
    out = {}

    def grab(Qx, Kx, Vx, sel, s, Bq, bias=None, q_pos=0):
        out["sel"] = sel.clone()
        return None

    a._cell_fused_lr = grab
    old = os.environ.get("FSL_FAST_SELECT")
    oldf = os.environ.get("FSL_FUSED_CONTENT")
    os.environ["FSL_FAST_SELECT"] = "1" if fast else "0"
    os.environ["FSL_FUSED_CONTENT"] = fused
    calls = []
    real = fsc.fsl_fast_select

    def spy(*args, **kw):
        r = real(*args, **kw)
        calls.append(r is not None)
        if sigma is not None:
            assert kw["sigv"] is sigma                       # handed to the kernels as given (no cast)
        return r

    fsc.fsl_fast_select = spy
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            a._cell_score_On_chunked_strict(Q, K, K, a.sigma if sigma is None else sigma, Q.shape[2])
    finally:
        fsc.fsl_fast_select = real
        for k, v in (("FSL_FAST_SELECT", old), ("FSL_FUSED_CONTENT", oldf)):
            if v is None:
                os.environ.pop(k)
            else:
                os.environ[k] = v
    assert calls == ([True] if fast else []), calls          # the fast path ran (no silent fallback) only when on
    return out["sel"]


def _check(a, Q, K, sigma=None):
    ref = _sel(a, Q, K, fast=False, sigma=sigma)
    for fused in ("1", "0"):           # tensor-core content kernel, and the cuBLAS content GEMM
        new = _sel(a, Q, K, fast=True, fused=fused, sigma=sigma)
        assert new.dtype == ref.dtype and new.shape == ref.shape
        bad = (new != ref).any(-1)
        assert not bad.any(), (f"fused={fused}: {int(bad.sum())} of {bad.numel()} query rows differ; "
                               f"first {bad.nonzero()[:4].tolist()}")


@pytest.mark.parametrize("N", [33, 100, 128, 300, 1000, 2048, 4096, 5000])
@pytest.mark.parametrize("kind", ["randn", "vocab", "periodic", "lowprec"])
def test_fast_select_matches_legacy(N, kind):
    a = _attn(4, 64, seed=N)
    Q, K = _qk(2, 4, N, 64, kind, seed=7 * N + len(kind))
    _check(a, Q, K)


@pytest.mark.parametrize("N", [3000, 16384])
def test_fast_select_matches_legacy_16_heads(N):
    a = _attn(16, 64, seed=1)
    for kind in ("randn", "vocab"):
        Q, K = _qk(1, 16, N, 64, kind, seed=N)
        _check(a, Q, K)


def test_fast_select_chunked_blocks_and_content_only():
    # FS_FUSED_CHUNK_GIB 0 -> 16 query blocks per chunk (several chunks, a padded last one)
    for co in (False, True):
        a = _attn(4, 64, content_only=co, chunk_gib=0, seed=3)
        for kind in ("randn", "vocab"):
            Q, K = _qk(1, 4, 7000, 64, kind, seed=11)
            _check(a, Q, K)


def test_fast_select_fp32_params():
    # RULER runs fp32 parameters under bf16 autocast
    a = _attn(4, 64, dtype=torch.float32, seed=5)
    Q, K = _qk(1, 4, 3000, 64, "vocab", seed=5)
    _check(a, Q, K)


def test_fast_select_bf16_sigma():
    # sigma handed over in bf16 (computed outside autocast): read and widened in the kernels; _sel's spy checks that
    # the fast path ran (both content paths) and got this bf16 tensor
    a = _attn(4, 64, seed=6)
    Q, K = _qk(1, 4, 4096, 64, "randn", seed=6)
    sig16 = a.log_sigma.exp().to(torch.bfloat16)
    assert sig16.dtype == torch.bfloat16
    _check(a, Q, K, sigma=sig16)


def test_fast_select_mbtopk_path():
    # > 4000 slices of >= 400 cells: torch.topk takes its multi-block radix path
    a = _attn(16, 64, seed=9)
    Q, K = _qk(1, 16, 65536, 64, "vocab", seed=9)
    _check(a, Q, K)


@pytest.mark.parametrize("M,Bq,s,P", [(1, 1, 32, 64), (5, 64, 32, 64), (64, 128, 32, 64), (300, 128, 32, 64),
                                      (1000, 128, 32, 37), (777, 96, 32, 100), (2048, 128, 32, 128), (513, 128, 16, 64),
                                      (300, 128, 32, 20), (4096, 128, 32, 64), (5000, 64, 32, 31), (3000, 128, 32, 9)])
@pytest.mark.parametrize("ties", [0, 3, 1000, -1])
@pytest.mark.parametrize("par", [False, True])
def test_prefix_topp_matches_topk(M, Bq, s, P, ties, par):
    # fsl_topp against torch.topk of the masked prefix rows, as the legacy path builds them
    BH = 6
    nb = max(1, (M * s + Bq - 1) // Bq)
    g = torch.Generator(device=DEV).manual_seed(M + ties)
    if ties == -1:                # rising ceilings: every cell enters the list
        ceil = torch.arange(M, device=DEV).float().expand(BH, M).contiguous() * 0.25
    elif ties:
        ceil = torch.randint(0, ties, (BH, M), device=DEV, generator=g).float() * 0.5
    else:
        ceil = torch.rand(BH, M, device=DEV, generator=g) * 10
    Pk = min(P, M)
    cell_end = (torch.arange(M, device=DEV, dtype=torch.float32) + 1.0) * s
    blk_start = torch.arange(nb, device=DEV, dtype=torch.float32) * Bq
    vis = cell_end[None, :] <= blk_start[:, None]
    pref = ceil[:, None, :].expand(BH, nb, M).masked_fill(~vis[None], -1e9)
    ref = pref.topk(Pk, dim=-1).indices.to(torch.int32)
    fs = fsc._functions(torch.device(DEV), (Pk + 31) // 32, 4)
    out = fsc.prefix_topp(fs, ceil, BH, M, nb, Bq, s, Pk, par=par)
    torch.cuda.synchronize()
    assert torch.equal(out, ref), (out != ref).nonzero()[:5].tolist()


def test_legacy_switch_and_eligibility():
    Q = torch.zeros(1, 2, 256, 64, device=DEV, dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert fsc.fast_select_ok(Q, Q, 8, 8, 4, 128, 256)
        os.environ["FSL_FAST_SELECT"] = "legacy"
        try:
            assert not fsc.fast_select_ok(Q, Q, 8, 8, 4, 128, 256)
        finally:
            os.environ.pop("FSL_FAST_SELECT")
    assert not fsc.fast_select_ok(Q, Q, 8, 8, 4, 128, 256)          # no autocast: legacy (fp32 content matmul)


@pytest.mark.parametrize("s", [1, 16, 32, 48])
@pytest.mark.parametrize("kind", ["randn", "wide", "zeros"])
@pytest.mark.parametrize("contig", [False, True])
def test_ceil_matches_torch_norm(s, kind, contig):
    # fsl_ceil against K.float().norm(dim=-1).amax(dim=-1) on the cell view; s=1 compares every token's norm
    B, H, D = 2, 4, 64
    M = 4096 // s + 3
    N = M * s
    g = torch.Generator(device=DEV).manual_seed(s * 31 + len(kind))
    x = torch.randn(B, N, H, D, device=DEV, generator=g)
    if kind == "wide":            # exponents over +-30 binades: the fp32 sum of squares rounds at every step
        x = x * torch.exp2(torch.randint(-30, 31, (B, N, H, 1), device=DEV, generator=g).float())
    elif kind == "zeros":
        x = x * (torch.rand(B, N, H, 1, device=DEV, generator=g) < 0.5)
    K = x.to(torch.bfloat16).transpose(1, 2)
    if contig:
        K = K.contiguous()
    ref = K.reshape(B, H, M, s, D).float().norm(dim=-1).amax(dim=-1) if contig else \
        K.view(B, H, M, s, D).float().norm(dim=-1).amax(dim=-1)
    fs = fsc._functions(torch.device(DEV), 2, 4)
    out = torch.empty(B, H, M, dtype=torch.float32, device=DEV)
    fsc._launch(fs["fsl_ceil"], (B * H * M * 32 + 127) // 128, 128, "PPqqqqqqq",
                (K.data_ptr(), out.data_ptr(), H, M, s, K.stride(0), K.stride(1), K.stride(2), B * H * M))
    torch.cuda.synchronize()
    bad = (out.view(torch.int32) != ref.view(torch.int32))
    assert not bad.any(), f"{int(bad.sum())} of {bad.numel()} cells differ"


@pytest.mark.parametrize("kind", ["randn", "wide", "cancel"])
def test_fused_content_matches_cublas(kind):
    # the tensor-core content of the fused kernel against torch.matmul's (cuBLAS) bf16 content, bit for bit
    H, N = 16, 16384
    a = _attn(H, 64, seed=2)
    g = torch.Generator(device=DEV).manual_seed(len(kind))
    x = torch.randn(1, N, 2, H, 64, device=DEV, generator=g)
    if kind == "wide":            # per-element exponents: the terms of one dot product span ~24 binades
        x = x * torch.exp2(torch.randint(-12, 13, x.shape, device=DEV, generator=g).float())
    elif kind == "cancel":        # large terms that cancel within a dot product
        x = x * torch.exp2(torch.randint(0, 2, x.shape, device=DEV, generator=g).float() * 10)
        x[..., 1, :, 32:] = -x[..., 1, :, :32] + 1e-3 * x[..., 1, :, 32:]
    x = x.to(torch.bfloat16)
    Q, K = x[:, :, 0].transpose(1, 2), x[:, :, 1].transpose(1, 2)
    got = {}
    for fused in ("0", "1"):
        fsc._DEBUG = []
        try:
            got[fused] = _sel(a, Q, K, fast=True, fused=fused)
            got["c" + fused] = fsc._DEBUG
        finally:
            fsc._DEBUG = None
    assert len(got["c0"]) == len(got["c1"]) > 0
    for (b0a, ca, pa), (b0b, cb, pb) in zip(got["c0"], got["c1"]):
        assert (pa, pb) == ("cublas", "fused") and b0a == b0b and ca.shape == cb.shape
        bad = ca.view(torch.int16) != cb.view(torch.int16)
        assert not bad.any(), f"{int(bad.sum())} of {bad.numel()} content values differ"
    assert torch.equal(got["0"], got["1"])


@pytest.mark.parametrize("kind", ["randn", "wide", "zeros", "negzero"])
@pytest.mark.parametrize("contig", [False, True])
def test_cell_stats_match_torch(kind, contig):
    # fsl_cellstats against K.mean(dim=3) (bf16) and K.float().norm(dim=-1).amax(dim=-1) on the cell view
    B, H, D, s = 2, 4, 64, 32
    M = 2051
    g = torch.Generator(device=DEV).manual_seed(len(kind) + 7 * contig)
    x = torch.randn(B, M * s, H, D, device=DEV, generator=g)
    if kind == "wide":            # +-30 binades: the fp32 sums round at every step
        x = x * torch.exp2(torch.randint(-30, 31, x.shape, device=DEV, generator=g).float())
    elif kind == "zeros":
        x = x * (torch.rand(B, M * s, H, 1, device=DEV, generator=g) < 0.5)
    elif kind == "negzero":       # signed zeros: torch's sum starts from +0
        x = torch.where(torch.rand(x.shape, device=DEV, generator=g) < 0.7, torch.full_like(x, -0.0), x)
    K = x.to(torch.bfloat16).transpose(1, 2)
    if contig:
        K = K.contiguous()
    Kv = K.view(B, H, M, s, D) if not contig else K.reshape(B, H, M, s, D)
    ref_m = Kv.mean(dim=3)
    ref_c = Kv.float().norm(dim=-1).amax(dim=-1)
    fs = fsc._functions(torch.device(DEV), 2, 4)
    kbar, ceil = fsc.cell_stats(fs, K, M, s)
    torch.cuda.synchronize()
    assert torch.equal(kbar.view(torch.int16), ref_m.view(torch.int16)), int((kbar.view(torch.int16) != ref_m.view(torch.int16)).sum())
    assert torch.equal(ceil.view(torch.int32), ref_c.view(torch.int32)), int((ceil.view(torch.int32) != ref_c.view(torch.int32)).sum())


def test_cache_prefill_stats_match_torch(monkeypatch):
    # FSKVCache.prefill's cell summaries from the CUDA pass against the torch ops (FSL_FAST_CACHE_STATS=0); a spy on
    # the kernel's host function makes sure the CUDA pass ran in mode 1 and not in mode 0
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "scaling"))
    from kv_cache import FSKVCache
    assert not fsc._failed_stats, fsc._failed_stats
    passes = []
    real = fsc.cell_stats

    def spy(*args, **kw):
        r = real(*args, **kw)
        passes.append(r is not None)
        return r

    monkeypatch.setattr(fsc, "cell_stats", spy)
    a = _attn(4, 64, seed=4)
    for N in (31, 32, 1000, 4096, 5000):
        Q, K = _qk(1, 4, N, 64, "vocab", seed=N)
        got = {}
        for mode in ("1", "0"):
            monkeypatch.setenv("FSL_FAST_CACHE_STATS", mode)
            passes.clear()
            c = FSKVCache(a)
            c.prefill(K, K)
            got[mode] = (c._kbar, c._ceil)
            assert passes == ([True] if mode == "1" and N >= 32 else []), (N, mode, passes)
        if N < 32:
            assert got["1"][0] is None and got["0"][0] is None
            continue
        for x, y in zip(got["1"], got["0"]):
            assert x.dtype == y.dtype and x.shape == y.shape
            assert torch.equal(x.view(torch.int32), y.view(torch.int32))


@pytest.mark.parametrize("N", [4096, 5000])
def test_fast_select_hands_cell_stats_to_cache(monkeypatch, N):
    # the prefill's long-range branch hands its cell statistics to FSKVCache.prefill (FSAttn._begin_lr,
    # FSL_PREFILL_CELL_REUSE): the fast selection hands over what the legacy path's own hand-off holds (nothing for a
    # padded length), and the cache then keeps exactly its torch ops' values
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "scaling"))
    from kv_cache import FSKVCache
    a = _attn(4, 64, seed=6)
    Q, K = _qk(1, 4, N, 64, "randn", seed=N)
    a._cell_fused_lr = lambda *args, **kw: None
    got = {}
    try:
        for fast in ("1", "0"):
            monkeypatch.setenv("FSL_FAST_SELECT", fast)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                a._begin_lr(None, want_cell_stats=True)
                try:
                    a._cell_score_On_chunked_strict(Q, K, K, a.sigma, N)
                finally:
                    got[fast] = a._end_lr()[1]
    finally:
        del a._cell_fused_lr
    if N % 32:
        assert got["1"] is None and got["0"] is None
        return
    assert got["1"][0] is K and got["0"][0] is K
    for x, y in zip(got["1"][1:], got["0"][1:]):
        assert x.dtype == y.dtype == torch.float32 and x.shape == y.shape
        assert torch.equal(x.view(torch.int32), y.view(torch.int32))
    monkeypatch.setenv("FSL_FAST_CACHE_STATS", "0")
    c = FSKVCache(a)
    c.prefill(K, K)
    d = FSKVCache(a)
    d._prefill_cell_stats = got["1"]
    d.prefill(K, K)
    assert d._kbar is got["1"][1] and torch.equal(c._kbar.view(torch.int32), d._kbar.view(torch.int32))
    assert torch.equal(c._ceil.view(torch.int32), d._ceil.view(torch.int32))


def test_erf_is_one_past_the_skip_threshold():
    # fsl_fused / fsl_select skip erf where its argument is >= the XS_SAFE threshold: tl.erf (the Triton select
    # kernel's erf) must be exactly 1.0f for every float from 0x407ad8e6 to +inf, and the CUDA erf_tl must equal
    # tl.erf (checked here on every 64th float of [0, 8])
    import ctypes
    from src.fsattn import fsb_beam_cuda as fb

    one, lo, hi = 0x3F800000, 0x407AD8E6, 0x7F800000
    ch = 1 << 26
    for base in range(lo, hi + 1, ch):
        n = min(ch, hi + 1 - base)
        out = torch.empty(n, dtype=torch.int32, device=DEV)
        _erf_k[(triton.cdiv(n, 1024),)](out, base, 1, n, BLOCK=1024)
        assert bool((out == one).all()), hex(base + int((out != one).nonzero()[0]))
    assert fsc.XS_SAFE_BITS >= lo + 32                     # the kernels' margin over div.full's 2-ulp error
    src = fb._HELPERS + r"""
    extern "C" __global__ void erf_cu(int* out, long long base, long long step, long long n) {
        long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = __float_as_int(erf_tl(__int_as_float((int)(base + i * step))));
    }"""
    fn = fb.load_function(torch.device(DEV, torch.cuda.current_device()), src, "erf_cu")
    n = (0x41000000 >> 6) + 1                              # [0, 8], every 64th bit pattern
    ref = torch.empty(n, dtype=torch.int32, device=DEV)
    got = torch.empty(n, dtype=torch.int32, device=DEV)
    _erf_k[(triton.cdiv(n, 1024),)](ref, 0, 64, n, BLOCK=1024)
    fb.launch(fn, (n + 255) // 256, 256, torch.device(DEV, torch.cuda.current_device()),
              [ctypes.c_void_p(got.data_ptr()), ctypes.c_longlong(0), ctypes.c_longlong(64), ctypes.c_longlong(n)])
    torch.cuda.synchronize()
    assert torch.equal(got, ref)


def _direct(B, H, N, P, Wc, kk, seed, s=32, Bq=128):
    # fsl_fast_select called as attention.py does, on random inputs (one chunk of all blocks)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(B, N, 2, H, 64, device=DEV, generator=g).to(torch.bfloat16)
    Q, K = x[:, :, 0].transpose(1, 2), x[:, :, 1].transpose(1, 2)
    lam = torch.rand(H, device=DEV, generator=g) + 0.5
    sig = torch.rand(H, device=DEV, generator=g) * 20 + 1
    M, nb = N // s, N // Bq
    return lambda: fsc.fsl_fast_select(Q, K, N=N, M=M, s=s, Bq=Bq, nb=nb, NBCH=nb, Pk=P, Wc=Wc, kk=kk, lam=lam,
                                       sigv=sig, scale=0.125, content_only=False, pool=P)


def test_graph_replay_does_not_read_cached_tensors():
    # a CUDA graph keeps only the addresses its kernels read: a capture must not hand a kernel _near_hi's cached
    # tensor, which the cache can evict and the allocator reuse while the graph is still replayed
    go = _direct(1, 4, 8192, 64, 8, 4, seed=12)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = go().clone()                         # eager: compiles the kernels, caches hi
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            go()
        torch.cuda.current_stream().wait_stream(side)
        n_cached = len(fsc._hi_cache)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = go()
        assert len(fsc._hi_cache) == n_cached     # nothing cached from the graph's pool either
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
        # evict the cache (as it does itself past 64 shapes) and allocate same-size blocks over the freed ones
        nb = 8192 // 128
        freed = {t.data_ptr() for t in fsc._hi_cache.values() if t.numel() == nb}
        assert freed
        fsc._hi_cache.clear()
        junk = []
        for _ in range(20000):
            junk.append(torch.full((nb,), 7, dtype=torch.int64, device=DEV))
            if junk[-1].data_ptr() in freed:
                break
        assert junk[-1].data_ptr() in freed       # the scenario happened: a freed cached block holds other data
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref), f"{int((out != ref).any(-1).sum())} rows differ after the eviction"


def test_hi_cache_is_per_stream():
    # a cached hi is only handed to kernels on the stream that computed it (stream-ordered writes and frees)
    dev = torch.device(DEV, torch.cuda.current_device())
    a = fsc._near_hi(dev, 77, 128, 32, False)
    assert fsc._near_hi(dev, 77, 128, 32, False) is a
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        b = fsc._near_hi(dev, 77, 128, 32, False)
    torch.cuda.current_stream().wait_stream(side)
    assert b is not a and torch.equal(a, b)
    assert fsc._near_hi(dev, 77, 128, 32, True) is not a       # under capture: never the cached tensor


def test_fused_content_self_check():
    # the self-check finds no difference at the batch counts of every cuBLAS kernel regime >= 2 on this GPU (it runs
    # before fsl_fused is first used at a count), and one flipped bit of torch.matmul's content shows up in it
    dev = torch.device(DEV, torch.cuda.current_device())
    fs = fsc._functions(dev, 2, 4, 9)
    assert fsc._fused_content_mismatches(fs, dev, 72, 128, 64, 8, 4, (2, 16, 32, 512, 1024)) == 0
    real = torch.matmul

    def flipped(a, b):
        out = real(a, b)
        out.view(torch.int16).view(-1)[123] ^= 1
        return out

    torch.matmul = flipped
    try:
        assert fsc._fused_content_mismatches(fs, dev, 72, 128, 64, 8, 4, (16, 256)) == 2
    finally:
        torch.matmul = real


def test_fused_content_falls_back_when_the_check_fails(monkeypatch):
    # a batch count where the check finds a difference: the call takes the cuBLAS content path, with a warning, and
    # the selections stay the legacy ones
    monkeypatch.setattr(fsc, "_fused_checked", {})
    monkeypatch.setattr(fsc, "_fused_content_mismatches", lambda *args, **kw: 5)
    a = _attn(4, 64, seed=13)
    Q, K = _qk(1, 4, 4096, 64, "randn", seed=13)
    ref = _sel(a, Q, K, fast=False)
    fsc._DEBUG = []
    try:
        with pytest.warns(UserWarning, match="differs from torch.matmul"):
            new = _sel(a, Q, K, fast=True, fused="1")
        paths = {p for _, _, p in fsc._DEBUG}
    finally:
        fsc._DEBUG = None
    assert paths == {"cublas"}
    assert torch.equal(new, ref)
    checks = fsc.status()["fused_content_checks"]
    assert checks and not any(c["exact"] for c in checks)


def test_fast_select_one_block_chunks_single_head():
    # B*H = 1 with a last chunk of one query block: the legacy content GEMM then has batch count 1, where cuBLAS (A100,
    # cu121) runs a slice-K kernel whose bits differ from fsl_fused's on wide-exponent inputs (the self-check finds
    # them); a call with such a chunk takes the cuBLAS content path, and the selections are the legacy ones
    a = _attn(1, 64, chunk_gib=0, seed=21)        # chunks of 16 blocks; 17 blocks -> chunks of 16 and 1
    dev = torch.cuda.current_device()
    for kind in ("randn", "vocab"):
        Q, K = _qk(1, 1, 17 * 128, 64, kind, seed=21)
        _check(a, Q, K)
        fsc._DEBUG = []
        try:
            _sel(a, Q, K, fast=True, fused="1")
            paths = [p for _, _, p in fsc._DEBUG]
        finally:
            fsc._DEBUG = None
        exact = all(fsc._fused_checked[(dev, 72, 128, fsc._blas_settings(), n)] for n in (1, 16))
        assert paths == ["fused" if exact else "cublas"] * 2, paths


def test_failure_latches_are_separate(monkeypatch):
    # a compile failure of the selection kernels turns off the fast selection only (legacy path, one warning); one of
    # the cache-statistics kernels turns off the statistics only (torch ops, one warning)
    monkeypatch.setattr(fsc, "_failed_select", [])
    monkeypatch.setattr(fsc, "_failed_stats", [])
    dev = torch.device(DEV, torch.cuda.current_device())
    fsc._functions(dev, 2, 4)                    # a compiled module (any one serves the statistics)

    def broken(*args, **kw):
        raise RuntimeError("simulated compile failure")

    a = _attn(4, 64, seed=8)
    Q, K = _qk(1, 4, 3000, 64, "randn", seed=8)
    grabbed = []
    a._cell_fused_lr = lambda Qx, Kx, Vx, sel, s, Bq, bias=None, q_pos=0: grabbed.append(sel.clone())
    ncell = 3000 // 32
    with monkeypatch.context() as m:
        m.setattr(fsc, "_functions", broken)
        with pytest.warns(UserWarning, match="fast selection unavailable"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                a._cell_score_On_chunked_strict(Q, K, K, a.sigma, Q.shape[2])
        assert len(grabbed) == 1                 # the legacy path selected
        assert fsc._failed_select and not fsc._failed_stats
        assert not fsc.fast_select_on() and fsc.fast_cache_stats_on()
        assert fsc.cache_cell_stats(K, ncell, 32) is not None
        st = fsc.status()
        assert "simulated" in st["select_failure"] and st["stats_failure"] is None
    monkeypatch.setattr(fsc, "_failed_select", [])
    with monkeypatch.context() as m:
        m.setattr(fsc, "_functions", broken)
        m.setattr(fsc, "_mods", {})               # nothing compiled: the statistics path has to compile
        with pytest.warns(UserWarning, match="cell statistics unavailable"):
            assert fsc.cache_cell_stats(K, ncell, 32) is None
        assert fsc._failed_stats and not fsc._failed_select
        assert fsc.fast_select_on() and not fsc.fast_cache_stats_on()
    ref = _sel(a, Q, K, fast=False)
    assert torch.equal(_sel(a, Q, K, fast=True), ref)          # the selection still takes the fast path


@pytest.mark.parametrize("N", [512, 1000, 4096])
@pytest.mark.parametrize("content_only", [False, True])
def test_dense_fast_select_matches_reference(monkeypatch, N, content_only):
    a = FSAttn(d_model=4 * 64, n_heads=4, window_size=128, causal=True, long_range_variant="cell_topk",
               use_rope=True, rope_long_range=False, fs_uniform_sr=True, cell_stride=32, cell_topk=4,
               cell_select="score", sel_strict_causal=1, cell_prune_kernel=0, fs_fused_attn=1,
               cell_fused_select=1, sel_content_only=content_only).to(DEV).to(torch.bfloat16).eval()
    g = torch.Generator(device=DEV).manual_seed(91 + N + int(content_only))
    x = torch.randn(1, N, 4 * 64, device=DEV, generator=g, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        monkeypatch.setenv("FSL_FAST_SELECT", "1")
        y_fast = a(x)
        monkeypatch.setenv("FSL_FAST_SELECT", "0")
        y_ref = a(x)
    assert torch.equal(y_fast, y_ref)

