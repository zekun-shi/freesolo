"""Fused inference glue (src/utils/trunk_triton.py) against the eager torch ops it replaces, bit for bit."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "scaling"))

from src.fsattn.attention import _apply_rotary_emb
from src.utils import trunk_triton as tt

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _qk_view(N, H=16, D=64, seed=0, dtype=torch.bfloat16):
    # W_q-style view: (B, N, H*D) linear output viewed as (B, H, N, D)
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(1, N, H * D, device="cuda", generator=g) * 3).to(dtype).view(1, N, H, D).transpose(1, 2)


def _cos_sin(N, D=64, dtype=torch.float32):
    from experiments.scaling.yarn_rope import build_rope
    r = build_rope(D, N + 8).cuda()
    return r.cos_cached[3:N + 3].to(dtype), r.sin_cached[3:N + 3].to(dtype)


@gpu
@pytest.mark.parametrize("N", [2, 5, 64, 513, 700, 5121, 16384])
@pytest.mark.parametrize("cos_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_rope_matches_apply_rotary_emb(N, cos_dtype):
    x = _qk_view(N, seed=N)
    cos, sin = _cos_sin(N, dtype=cos_dtype)
    ref = _apply_rotary_emb(x, cos, sin)
    assert tt.rope_ok(x, cos)
    got = tt.rope(x, cos, sin)
    assert got.dtype == ref.dtype and got.shape == ref.shape and got.is_contiguous()
    assert torch.equal(got, ref)
    if cos_dtype == torch.float32:
        # RULER regime: fp32 result that SDPA's autocast casts to bf16
        assert torch.equal(tt.rope(x, cos, sin, torch.bfloat16), ref.to(torch.bfloat16))


@gpu
@torch.no_grad()
def test_rope_contiguous_and_fp32_inputs():
    N = 777
    cos, sin = _cos_sin(N)
    x = torch.randn(2, 4, N, 64, device="cuda")
    assert torch.equal(tt.rope(x, cos, sin), _apply_rotary_emb(x, cos, sin))
    xb = x.to(torch.bfloat16)
    assert torch.equal(tt.rope(xb, cos.to(torch.bfloat16), sin.to(torch.bfloat16)),
                       _apply_rotary_emb(xb, cos.to(torch.bfloat16), sin.to(torch.bfloat16)))


@gpu
@pytest.mark.parametrize("N", [1, 7, 4096, 16384])
@pytest.mark.parametrize("x_dtype,w_dtype", [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32),
                                             (torch.bfloat16, torch.float32)])
@torch.no_grad()
def test_rms_norm_cast_matches_autocast_rmsnorm(N, x_dtype, w_dtype):
    from experiments.scaling.model import RMSNorm
    g = torch.Generator(device="cuda").manual_seed(N)
    norm = RMSNorm(1024).cuda()
    norm.weight.copy_(1 + 0.3 * torch.randn(1024, device="cuda", generator=g))
    norm = norm.to(w_dtype)
    x = (torch.randn(1, N, 1024, device="cuda", generator=g) * torch.rand(1, N, 1, device="cuda", generator=g) * 5)
    x = x.to(x_dtype)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = norm(x).to(torch.bfloat16)          # the cast autocast applies at the next linear
        assert tt.rms_norm_ok(x, norm.weight) == (N >= tt.MIN_ROWS)    # fewer rows (decode) keep the eager norm
        got = tt.rms_norm_cast(x, norm.weight, norm.eps, torch.bfloat16)
    assert got.dtype == torch.bfloat16 and torch.equal(got, ref)


@gpu
@pytest.mark.parametrize("D", [256, 1024, 2048])
@pytest.mark.parametrize("N", [16, 17, 100, 511, 513, 3000])
@pytest.mark.parametrize("onepass", ["1", "0"])
@torch.no_grad()
def test_rms_norm_cast_shapes(monkeypatch, D, N, onepass):
    # the one-pass kernel reproduces torch's mean order for these shapes; the three-pass path is the fallback
    from experiments.scaling.model import RMSNorm
    monkeypatch.setenv("FSL_NORM_ONEPASS", onepass)
    g = torch.Generator(device="cuda").manual_seed(D + N)
    norm = RMSNorm(D).cuda()
    norm.weight.copy_(1 + 0.3 * torch.randn(D, device="cuda", generator=g))
    norm = norm.to(torch.bfloat16)
    x = (torch.randn(2, N, D, device="cuda", generator=g) * torch.rand(2, N, 1, device="cuda", generator=g) * 9)
    x = x.to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = norm(x).to(torch.bfloat16)
        got = tt.rms_norm_cast(x, norm.weight, norm.eps, torch.bfloat16)
    assert torch.equal(got, ref)


@gpu
@pytest.mark.parametrize("onepass", ["1", "0"])
@torch.no_grad()
def test_rms_norm_cast_wide_rows_take_the_fallback(monkeypatch, onepass):
    # rows wider than 2048 (no config has them; 4096 did not finish compiling as one pass in 13 minutes) take the
    # three-pass path, which is also exact
    from experiments.scaling.model import RMSNorm
    monkeypatch.setenv("FSL_NORM_ONEPASS", onepass)
    D = 4096
    assert D not in tt._ONEPASS_WIDTHS
    g = torch.Generator(device="cuda").manual_seed(9)
    norm = RMSNorm(D).cuda()
    norm.weight.copy_(1 + 0.3 * torch.randn(D, device="cuda", generator=g))
    norm = norm.to(torch.bfloat16)
    x = (torch.randn(1, 300, D, device="cuda", generator=g) * torch.rand(1, 300, 1, device="cuda", generator=g) * 9)
    x = x.to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert tt.rms_norm_ok(x, norm.weight)
        ref = norm(x).to(torch.bfloat16)
        got = tt.rms_norm_cast(x, norm.weight, norm.eps, torch.bfloat16)
    assert torch.equal(got, ref)


@gpu
@torch.no_grad()
def test_rms_norm_fused_only_without_grad():
    x = torch.randn(1, 16, 1024, device="cuda")
    w = torch.ones(1024, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert tt.rms_norm_ok(x, w)
        with torch.enable_grad():
            assert not tt.rms_norm_ok(x, w)
    assert not tt.rms_norm_ok(x, w)            # no autocast: the norm output is not cast at the next linear


@gpu
@torch.no_grad()
def test_cache_reuses_identical_cell_stats():
    # FSKVCache.prefill gets the long-range branch's cell means / ceilings only when they are the same ops on the same
    # K view; check they equal what the cache computes itself
    from kv_cache import FSKVCache

    class _Attn:
        cell_stride = 32
        window_size = 512
        n_heads = 16
        head_dim = 64
        fsb_leaf_radius = False

        def _assert_cacheable(self, phase):
            pass

    N = 4096
    K = _qk_view(N, seed=1)
    V = _qk_view(N, seed=2)
    Kpv = K.view(1, 16, N // 32, 32, 64)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        kbar = Kpv.mean(dim=3).float()
        ceil = Kpv.float().norm(dim=-1).amax(dim=-1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = FSKVCache(_Attn()); a.prefill(K, V)
        b = FSKVCache(_Attn()); b.prefill(K, V, cell_stats=(K, kbar, ceil))
        c = FSKVCache(_Attn()); c.prefill(K, V, cell_stats=(K.clone(), kbar, ceil))   # another tensor: recomputed
        d = FSKVCache(_Attn()); d._prefill_cell_stats = (K, kbar, ceil); d.prefill(K, V)   # the attention's hand-off
    assert b._kbar is kbar and c._kbar is not kbar and d._kbar is kbar and d._prefill_cell_stats is None
    assert torch.equal(a._kbar, b._kbar) and torch.equal(a._ceil, b._ceil)
    assert torch.equal(a._kbar, c._kbar) and torch.equal(a._ceil, c._ceil)


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.no_grad()
def test_silu_mul_matches_eager(dtype):
    import torch.nn.functional as F
    if dtype == torch.bfloat16:
        # every bf16 value as the gate, against a few multipliers each
        g = torch.arange(1 << 16, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        g = g.repeat(4).view(-1, 1024)
        u = torch.randn(g.shape, device="cuda").to(torch.bfloat16)
    else:
        gen = torch.Generator(device="cuda").manual_seed(3)
        g = (torch.randn(1024, 1024, device="cuda", generator=gen) * 8).to(dtype)
        u = torch.randn(1024, 1024, device="cuda", generator=gen).to(dtype)
    assert tt.silu_mul_ok(g, u)
    ref = F.silu(g) * u
    got = tt.silu_mul(g, u)
    same = (got == ref) | (got.isnan() & ref.isnan())
    assert bool(same.all()), (g[~same][:8], got[~same][:8], ref[~same][:8])


@gpu
@torch.no_grad()
def test_swiglu_fused_matches_eager(monkeypatch):
    from experiments.scaling.model import SwiGLU
    torch.manual_seed(0)
    ffn = SwiGLU(1024, 2816).cuda().to(torch.bfloat16)
    x = torch.randn(1, 3000, 1024, device="cuda").to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = ffn(x)
        monkeypatch.setenv("FSL_TRUNK_FUSED", "0")
        ref = ffn(x)
    assert torch.equal(got, ref)


@gpu
@torch.no_grad()
def test_few_rows_keep_the_eager_ops(monkeypatch):
    # single-token decode (1 row per sequence) keeps the eager norm and silu * mul, whose host time is lower than the
    # Triton launches'; from MIN_ROWS rows on (a prefill, or a decode batch of 16) the fusions run. Same bits either way.
    from experiments.scaling.model import FreeSoloLM
    from kv_cache import _trunk_cached, make_caches
    w = torch.ones(1024, device="cuda")
    for rows in (1, 2, 15, 16, 64):
        g = torch.randn(rows, 2816, device="cuda").to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert tt.rms_norm_ok(torch.randn(1, rows, 1024, device="cuda"), w) == (rows >= tt.MIN_ROWS)
            assert tt.rms_norm_ok(torch.randn(rows, 1, 1024, device="cuda"), w) == (rows >= tt.MIN_ROWS)
        assert tt.silu_mul_ok(g, g) == (rows >= tt.MIN_ROWS)
    calls = []
    for name in ("rms_norm_cast", "silu_mul"):
        real = getattr(tt, name)
        monkeypatch.setattr(tt, name, lambda *a, _r=real, _n=name, **k: (calls.append(_n), _r(*a, **k))[1])
    torch.manual_seed(3)
    m = FreeSoloLM(vocab_size=512, d_model=256, n_layers=2, n_heads=4, d_ff=704, max_seq_len=1024,
                   attn_type="fsattn", window_size=64, long_range_variant="cell_topk", cell_stride=32, cell_topk=4,
                   cell_select="score", sel_strict_causal=1, use_rope=True, rope_long_range=False,
                   fs_uniform_sr=True).cuda().to(torch.bfloat16).eval()
    prompt = torch.randint(0, 512, (1, 300), device="cuda", generator=torch.Generator(device="cuda").manual_seed(4))
    outs = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("FSL_TRUNK_FUSED", flag)
        cache = make_caches(m)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            calls.clear()
            h = [_trunk_cached(m, prompt, cache)]
            n_prefill = len(calls)
            for t in range(3):
                calls.clear()
                h.append(_trunk_cached(m, prompt[:, t:t + 1], cache))
                assert not calls, calls                                  # decode: the eager ops
        outs[flag] = h
        assert n_prefill == (6 if flag == "1" else 0), n_prefill        # 2 layers x (2 norms + silu * mul)
    assert all(torch.equal(a, b) for a, b in zip(outs["1"], outs["0"]))


@gpu
@pytest.mark.parametrize("D", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("rows", [1, 4, 15])
@pytest.mark.parametrize("out_dt", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_one_row_rms_norm_cast_matches_eager(monkeypatch, D, rows, out_dt):
    from experiments.scaling.model import RMSNorm
    torch.manual_seed(D + rows)
    ln = RMSNorm(D).cuda().to(torch.bfloat16)
    ln.weight.copy_(torch.randn(D, device="cuda").to(torch.bfloat16))
    x = (torch.randn(1, rows, D, device="cuda") * 3.0).to(torch.bfloat16)
    monkeypatch.setattr(tt, "_GRAPH_WARMUP", True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert tt.rms_norm_ok(x, ln.weight)
        got = tt.rms_norm_cast(x, ln.weight, ln.eps, out_dt)
        ref = ln(x).to(out_dt)
    assert torch.equal(got, ref)


@gpu
@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("Lw", [65, 513])
@torch.no_grad()
def test_sr_decode_fused_matches_eager(monkeypatch, use_rope, uniform, Lw):
    from src.fsattn.attention import FSAttn
    torch.manual_seed(17 + Lw + int(use_rope) * 3 + int(uniform) * 7)
    H, D = 8, 64
    W = Lw - 1
    attn = FSAttn(d_model=H * D, n_heads=H, window_size=W, causal=True, use_rope=use_rope,
                  rope_long_range=False, fs_uniform_sr=uniform, max_seq_len=2048,
                  cell_stride=32, cell_topk=4, sel_strict_causal=1).cuda().to(torch.bfloat16).eval()
    Lmax = 1024
    pos = 600
    kbuf_ref = torch.randn(1, H, Lmax, D, device="cuda", dtype=torch.bfloat16)
    vbuf_ref = torch.randn(1, H, Lmax, D, device="cuda", dtype=torch.bfloat16)
    kbuf_got = kbuf_ref.clone()
    vbuf_got = vbuf_ref.clone()
    Q = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
    pos_t = torch.tensor(pos, device="cuda", dtype=torch.int64)
    win_idx = torch.arange(pos - W, pos + 1, device="cuda", dtype=torch.int64)
    win_pos = win_idx.to(torch.float32)
    buf = {"pos_t": pos_t, "win_idx": win_idx, "win_pos": win_pos}
    sigma = attn.sigma
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = attn._sr_kv_and_decode(Q, K, V, kbuf_got, vbuf_got, buf, sigma)
        monkeypatch.setenv("FSL_SR_DECODE_FUSED", "0")
        ref = attn._sr_kv_and_decode(Q, K, V, kbuf_ref, vbuf_ref, buf, sigma)
    assert torch.equal(kbuf_got, kbuf_ref)
    assert torch.equal(vbuf_got, vbuf_ref)
    assert torch.equal(got, ref)


@gpu
@pytest.mark.parametrize("N", [256, 1280, 4096])
@torch.no_grad()
def test_fsl_pool_refresh_matches_eager(monkeypatch, N):
    from kv_cache import FSKVCache
    from src.fsattn.attention import FSAttn
    torch.manual_seed(31 + N)
    H, D, s = 8, 64, 32
    attn = FSAttn(d_model=H * D, n_heads=H, window_size=64, causal=True, use_rope=True,
                  rope_long_range=False, fs_uniform_sr=True, max_seq_len=N + 128,
                  cell_stride=s, cell_topk=4, sel_strict_causal=1).cuda().to(torch.bfloat16).eval()
    K = _qk_view(N, H=H, D=D, seed=N + 1)
    V = _qk_view(N, H=H, D=D, seed=N + 2)
    cache = FSKVCache(attn)
    cache.prefill(K, V)
    blk_start = (N // 128) * 128
    got = attn._build_decode_pool(cache, blk_start, s, K.device, need_tokens=False)
    monkeypatch.setenv("FSL_POOL_REFRESH_FUSED", "0")
    ref = attn._build_decode_pool(cache, blk_start, s, K.device, need_tokens=False)
    for k in ("Kbar_cand", "ceil_cand", "cen_cand", "ok_cell", "cand_c"):
        assert torch.equal(got[k], ref[k]), k
    # also test in-place refresh into an existing pool dict
    monkeypatch.setenv("FSL_POOL_REFRESH_FUSED", "1")
    ptrs = {k: got[k].data_ptr() for k in ("Kbar_cand", "ceil_cand", "cen_cand", "ok_cell", "cand_c")}
    got2 = attn._build_decode_pool(cache, blk_start, s, K.device, need_tokens=False, out=got)
    assert got2 is got
    for k in ("Kbar_cand", "ceil_cand", "cen_cand", "ok_cell", "cand_c"):
        assert got2[k].data_ptr() == ptrs[k]
        assert torch.equal(got2[k], ref[k]), k


@gpu
@pytest.mark.parametrize("use_autocast", [False, True])
@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_rope", [False, True])
@torch.no_grad()
def test_fsl_lr_decode_combine_matches_eager(monkeypatch, use_autocast, w_dtype, use_rope):
    if not use_autocast and w_dtype == torch.float32:
        pytest.skip("non-autocast fp32 weights require fp32 activations")
    from contextlib import nullcontext
    from kv_cache import FSKVCache
    from src.fsattn.attention import FSAttn
    torch.manual_seed(73 + int(use_rope) * 5 + (1 if w_dtype == torch.bfloat16 else 2) + int(use_autocast) * 11)
    H, D, W, s = 8, 64, 64, 32
    N = 1024
    Lmax = 1280
    pos = 1030
    attn = FSAttn(d_model=H * D, n_heads=H, window_size=W, causal=True, use_rope=use_rope,
                  rope_long_range=False, fs_uniform_sr=False, max_seq_len=Lmax + 16,
                  cell_stride=s, cell_topk=4, sel_strict_causal=1).cuda().to(w_dtype).eval()
    attn.lr_norm.scale.copy_(0.8 + 0.4 * torch.randn(D, device="cuda", dtype=w_dtype))
    attn.long_range_scale.copy_(0.2 * torch.randn(H, 1, 1, device="cuda", dtype=w_dtype))
    attn.sel_scale.copy_(0.5 + torch.randn(H, device="cuda", dtype=w_dtype))
    K = _qk_view(N, H=H, D=D, seed=11)
    V = _qk_view(N, H=H, D=D, seed=12)
    cache = FSKVCache(attn)
    cache.reserve = Lmax - N
    cache.preallocate(1, N, torch.bfloat16, "cuda")
    cache.prefill(K, V)
    blk_start = (pos // 128) * 128
    pool = attn._build_decode_pool(cache, blk_start, s, K.device, need_tokens=False)
    kbuf_got = cache._k_buf.clone()
    vbuf_got = cache._v_buf.clone()
    kbuf_ref = cache._k_buf.clone()
    vbuf_ref = cache._v_buf.clone()
    pos_t = torch.tensor(pos, device="cuda", dtype=torch.int64)
    win_idx = torch.arange(pos - W, pos + 1, device="cuda", dtype=torch.int64)
    win_pos = win_idx.to(torch.float32)
    x = torch.randn(1, 1, H * D, device="cuda", dtype=torch.bfloat16)
    buf_got = {"kbuf": kbuf_got, "vbuf": vbuf_got, "pos_t": pos_t, "win_idx": win_idx, "win_pos": win_pos, "pool": pool}
    buf_ref = {"kbuf": kbuf_ref, "vbuf": vbuf_ref, "pos_t": pos_t, "win_idx": win_idx, "win_pos": win_pos, "pool": pool}
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
    with ctx:
        got = attn.graph_decode_layer(x, buf_got)
        monkeypatch.setenv("FSL_LR_DECODE_FUSED", "0")
        ref = attn.graph_decode_layer(x, buf_ref)
    assert torch.equal(got, ref)


@gpu
@torch.no_grad()
def test_fsl_graphed_decoder_matches_cached_generate(monkeypatch):
    from decode_graph import GraphedDecoder
    from experiments.scaling.model import FreeSoloLM
    from kv_cache import cached_generate
    # Bit identity with the eager cached_generate loop holds only on identical KV storage: pin the bf16
    # store (FS_KV_INT8=0); the int8 store has its own parity coverage in tests/test_kv_int8.py
    monkeypatch.setenv("FS_KV_INT8", "0")
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_STRICT_CAUSAL", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "1")
    torch.manual_seed(99)
    m = FreeSoloLM(vocab_size=512, d_model=512, n_layers=2, n_heads=8, d_ff=1408, max_seq_len=1024,
                   attn_type="fsattn", window_size=64, long_range_variant="cell_topk", cell_stride=32, cell_topk=4,
                   cell_select="score", sel_strict_causal=1, use_rope=True, rope_long_range=False,
                   fs_uniform_sr=True).cuda().to(torch.bfloat16).eval()
    prompt = torch.randint(0, 512, (1, 370), device="cuda", generator=torch.Generator(device="cuda").manual_seed(42))
    steps = 25
    ref_seq = cached_generate(m, prompt, steps, device="cuda", dtype="bfloat16")
    gd = GraphedDecoder(m, prompt, Lmax=370 + steps + 16, device="cuda", dtype=torch.bfloat16)
    gd.prefill()
    toks = [gd.next_tok.clone()]
    for _ in range(steps - 1):
        toks.append(gd.step().clone())
    got_seq = torch.cat([prompt, torch.stack(toks, dim=1)], dim=1)
    assert torch.equal(got_seq, ref_seq)
    logits_fused = gd.static_logits.clone()
    monkeypatch.setenv("FSL_LR_DECODE_FUSED", "0")
    monkeypatch.setenv("FSL_POOL_REFRESH_FUSED", "0")
    gd_ref = GraphedDecoder(m, prompt, Lmax=370 + steps + 16, device="cuda", dtype=torch.bfloat16)
    gd_ref.prefill()
    for _ in range(steps - 1):
        gd_ref.step()
    assert torch.equal(logits_fused, gd_ref.static_logits)


@gpu
@torch.no_grad()
def test_dense_graphed_decoder_matches_cached_generate(monkeypatch):
    from decode_graph import GraphedDecoder
    from experiments.scaling.model import FreeSoloLM
    from kv_cache import cached_generate
    # Bit identity with the eager cached_generate loop holds only on identical KV storage: pin the bf16
    # store (FS_KV_INT8=0); the int8 store has its own parity coverage in tests/test_kv_int8.py
    monkeypatch.setenv("FS_KV_INT8", "0")
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_FUSED_SELECT", "1")
    monkeypatch.setenv("FS_STRICT_CAUSAL", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "0")
    torch.manual_seed(109)
    m = FreeSoloLM(vocab_size=512, d_model=512, n_layers=2, n_heads=8, d_ff=1408, max_seq_len=1024,
                   attn_type="fsattn", window_size=64, long_range_variant="cell_topk", cell_stride=32, cell_topk=4,
                   cell_select="score", sel_strict_causal=1, cell_prune_kernel=0, use_rope=True,
                   rope_long_range=False, fs_uniform_sr=True).cuda().to(torch.bfloat16).eval()
    prompt = torch.randint(0, 512, (1, 370), device="cuda", generator=torch.Generator(device="cuda").manual_seed(43))
    steps = 30
    ref_seq = cached_generate(m, prompt, steps, device="cuda", dtype="bfloat16")
    gd = GraphedDecoder(m, prompt, Lmax=370 + steps + 16, device="cuda", dtype=torch.bfloat16)
    gd.prefill()
    toks = [gd.next_tok.clone()]
    for _ in range(steps - 1):
        toks.append(gd.step().clone())
    got_seq = torch.cat([prompt, torch.stack(toks, dim=1)], dim=1)
    assert torch.equal(got_seq, ref_seq)
    logits_fused = gd.static_logits.clone()
    monkeypatch.setenv("FSL_LR_DECODE_FUSED", "0")
    gd_ref = GraphedDecoder(m, prompt, Lmax=370 + steps + 16, device="cuda", dtype=torch.bfloat16)
    gd_ref.prefill()
    for _ in range(steps - 1):
        gd_ref.step()
    assert torch.equal(logits_fused, gd_ref.static_logits)



