import pytest
import torch

from src.fsattn import kv_int8


gpu = pytest.mark.skipif(not kv_int8._HAVE_TRITON or not torch.cuda.is_available(),
                         reason="needs CUDA and Triton")


def _ref_q(x, s=32):
    return kv_int8.quantize_cells(x, s)


@pytest.mark.parametrize("shape", [(1, 2, 96, 8), (2, 3, 64, 32)])
def test_quantize_torch_reference(shape):
    torch.manual_seed(7)
    B, H, N, D = shape
    s = 32
    x = (torch.randn(B, H, N + 5, D) * 3).bfloat16()
    q, sc = _ref_q(x, s)
    nc = N // s
    assert q.shape == (B, H, nc * s, D) and q.dtype == torch.int8
    assert sc.shape == (B, H, nc) and sc.dtype == torch.bfloat16
    xc = x[:, :, :nc * s].reshape(B, H, nc, s, D).float()
    expect_sc = xc.abs().amax(dim=(3, 4)) / 127.0
    expect_sc = torch.where(expect_sc > 0, expect_sc, torch.ones_like(expect_sc)).bfloat16()
    assert torch.equal(sc, expect_sc)
    deq = q.float().reshape(B, H, nc, s, D) * sc[:, :, :, None, None].float()
    # symmetric round-to-nearest: |q * sc - x| <= sc / 2 elementwise
    assert ((deq - xc).abs() <= sc[:, :, :, None, None].float() * 0.5 + 1e-6).all()


def test_quantize_zero_cell():
    x = torch.zeros(1, 2, 64, 16).bfloat16()
    q, sc = _ref_q(x, 32)
    assert (q == 0).all()
    assert (sc == 1.0).all()                      # empty cells keep x == 0 exactly


@gpu
@pytest.mark.parametrize("shape", [(1, 4, 128, 64), (2, 8, 96, 64), (1, 16, 256, 64)])
def test_quantize_fast_matches_torch(shape):
    torch.manual_seed(11)
    B, H, N, D = shape
    s = 32
    x = (torch.randn(B, H, N + 7, D, device="cuda") * 5).bfloat16()
    out = kv_int8.quantize_cells_fast(x, s)
    assert out is not None
    q, sc = out
    qr, scr = _ref_q(x, s)
    assert torch.equal(q, qr)
    assert torch.equal(sc, scr)


@gpu
def test_quantize_fast_into_preallocated_slices():
    # the cache's usage: quantize into slices of larger preallocated buffers
    torch.manual_seed(13)
    B, H, D, s = 1, 4, 64, 32
    cap, nq = 192, 96
    x = (torch.randn(B, H, nq, D, device="cuda") * 2).bfloat16()
    qbuf = torch.zeros(B, H, cap, D, dtype=torch.int8, device="cuda")
    sbuf = torch.full((B, H, cap // s), float("nan"), dtype=torch.bfloat16, device="cuda")
    kv_int8.quantize_cells_fast(x, s, qbuf[:, :, :nq], sbuf[:, :, :nq // s])
    qr, scr = _ref_q(x, s)
    assert torch.equal(qbuf[:, :, :nq], qr)
    assert torch.equal(sbuf[:, :, :nq // s], scr)
    assert (qbuf[:, :, nq:] == 0).all()           # nothing outside the target slices is touched
    assert torch.isnan(sbuf[:, :, nq // s:]).all()


@gpu
def test_dequant_roundtrip():
    torch.manual_seed(17)
    q, sc = kv_int8.quantize_cells((torch.randn(1, 4, 64, 64, device="cuda") * 3).bfloat16(), 32)
    x = kv_int8.dequant_cells(q, sc, 32, dtype=torch.float32)
    assert x.shape == q.shape
    expect = q.float().reshape(1, 4, 2, 32, 64) * sc.float()[:, :, :, None, None]
    assert torch.equal(x.reshape(1, 4, 2, 32, 64), expect)
    # sc may carry extra trailing cells: they are ignored
    out = kv_int8.dequant_cells(q, torch.cat([sc, sc], dim=2), 32, dtype=torch.float32)
    assert torch.equal(out, x)


@gpu
@torch.no_grad()
def test_cache_int8_matches_bf16_read(monkeypatch):
    # cache-level int8: the materialized read matches the bf16 cache within the cell quantization
    # (hot open-cell rows exact), and complete-cell statistics are taken pre-quantization
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "scaling"))
    from src.fsattn.attention import FSAttn
    from experiments.scaling.kv_cache import FSKVCache

    torch.manual_seed(23)
    attn = FSAttn(d_model=128, n_heads=4, window_size=8, causal=True, use_rope=False,
                  lr_scale_mode="one", cell_stride=32, cell_topk=2, sel_strict_causal=1).to(torch.bfloat16).cuda().eval()
    x = torch.randn(1, 96, 128, device="cuda")
    K = V = None

    caches = {}
    for i8 in (False, True):
        monkeypatch.setenv("FS_KV_INT8", "1" if i8 else "0")
        c = FSKVCache(attn)
        if i8:
            c.enable_int8()
        c.preallocate(1, 96, torch.bfloat16, torch.device("cuda"), keep=True)
        attn(x.bfloat16(), cache=c)
        caches[i8] = c
    assert caches[True].use_int8 and not caches[False].use_int8
    Kb, Vb = caches[False].read_full_kv()
    Ki, Vi = caches[True].read_full_kv()
    Kb, Vb, Ki, Vi = (t[:, :, :96] for t in (Kb, Vb, Ki, Vi))
    s = 32
    n_q = caches[True]._n_q
    assert n_q == 96                                        # 96 rows = 3 full cells
    # quantized rows: dequantization error within sc / 2 (+ bf16 rounding)
    D = Kb.shape[-1]
    sc = caches[True]._ks_buf[:, :, :n_q // s].float().reshape(1, 4, n_q // s, 1, 1)
    err = (Ki[:, :, :n_q].float().reshape(1, 4, n_q // s, s, D)
           - Kb[:, :, :n_q].float().reshape(1, 4, n_q // s, s, D)).abs()
    # sc/2 from the per-cell round-to-nearest plus the bf16 rounding of the dequantized output row
    assert (err <= sc * 0.5 + 0.02).all()
    assert err.max() > 0, "int8 storage must actually quantize (differ from bf16 somewhere)"

    # complete-cell statistics identical (computed from pre-quantization bf16 in both modes)
    assert torch.equal(caches[True]._kbar, caches[False]._kbar)
    assert torch.equal(caches[True]._ceil, caches[False]._ceil)


@gpu
@torch.no_grad()
def test_cache_finish_cell_quantizes_open_cell(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "scaling"))
    from src.fsattn.attention import FSAttn
    from experiments.scaling.kv_cache import FSKVCache

    torch.manual_seed(29)
    attn = FSAttn(d_model=128, n_heads=4, window_size=8, causal=True, use_rope=False,
                  lr_scale_mode="one", cell_stride=32, cell_topk=2, sel_strict_causal=1).to(torch.bfloat16).cuda().eval()
    x = torch.randn(1, 96 + 32 + 13, 128, device="cuda")
    monkeypatch.setenv("FS_KV_INT8", "1")
    c = FSKVCache(attn)
    c.enable_int8()
    c.reserve = x.shape[1] - 96
    c.preallocate(1, 96, torch.bfloat16, torch.device("cuda"), keep=True)
    attn(x[:, :96].bfloat16(), cache=c)
    assert c._n_q == 96
    # one full open cell + a partial one through single-token decode: rows 96..127 hot, then quantized
    pos_row = 96
    for t in range(32):
        attn(x[:, pos_row:pos_row + 1].bfloat16(), cache=c)
        pos_row += 1
    assert c._n_q == 128, "a full hot cell must be quantized at the cell boundary"
    assert int(c._nq_t) == 128
    hot_left = c._len - c._n_q
    assert hot_left == 0
    for t in range(13):
        attn(x[:, pos_row:pos_row + 1].bfloat16(), cache=c)
        pos_row += 1
    assert c._n_q == 128 and c._len - c._n_q == 13
    # hot rows read back exactly bf16; the cell just closed matches its quantization of the prefill write
    Ki, Vi = c.read_full_kv()
    assert torch.equal(Ki[:, :, 128:141], c._k_hot[:, :, :13])
    Kb = None
    c2 = FSKVCache(attn)
    c2.reserve = x.shape[1] - 96
    c2.preallocate(1, 96, torch.bfloat16, torch.device("cuda"), keep=True)
    attn(x[:, :96].bfloat16(), cache=c2)
    p2 = 96
    for t in range(45):
        attn(x[:, p2:p2 + 1].bfloat16(), cache=c2)
        p2 += 1
    Kb2, _ = c2.read_full_kv()
    D = Kb2.shape[-1]
    sc = c._ks_buf[:, :, 3:4].float()[:, :, :, None, None]
    err = (Ki[:, :, 96:128].float().reshape(1, 4, 1, 32, D)
           - Kb2[:, :, 96:128].float().reshape(1, 4, 1, 32, D)).abs()
    assert (err <= sc * 0.5 + 0.02).all()


def _fsb_attn_and_int8_cache(monkeypatch, n=128, d_model=128, n_heads=4, int8=True):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "scaling"))
    from src.fsattn.attention import FSAttn
    from experiments.scaling.kv_cache import FSKVCache

    attn = FSAttn(d_model=d_model, n_heads=n_heads, window_size=8, causal=True, use_rope=False,
                  lr_scale_mode="one", cell_stride=32, cell_topk=2, sel_strict_causal=1).to(torch.bfloat16).cuda().eval()
    monkeypatch.setenv("FS_KV_INT8", "1" if int8 else "0")
    c = FSKVCache(attn)
    if int8:
        c.enable_int8()
    c.reserve = 96
    c.preallocate(1, n, torch.bfloat16, torch.device("cuda"), keep=True)
    x = torch.randn(1, n, d_model, device="cuda", generator=torch.Generator(device="cuda").manual_seed(31))
    attn(x.bfloat16(), cache=c)
    assert c.use_int8 == int8 and (not int8 or c._n_q == (n // 32) * 32)
    return attn, c


@gpu
@torch.no_grad()
def test_decread_int8_fp32_lr_norm_matches_reference(monkeypatch):
    # an fp32 lr_norm.scale rejected the fused int8 decode readout; the torch fallback then gathered
    # from a length-sliced materialization, which cannot live inside a decode graph
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    attn, c = _fsb_attn_and_int8_cache(monkeypatch)
    # the released checkpoints (and therefore every eval loader) carry the norms in fp32; the readout's
    # norm promotion is independent of the quantized cache contents, so flip the dtype after prefill
    attn.lr_norm.scale.data = attn.lr_norm.scale.data.float()
    n = c._len
    B, H, Dh = 1, 4, attn.head_dim
    q_t = torch.randn(B, H, 1, Dh, device="cuda", dtype=torch.bfloat16,
                      generator=torch.Generator(device="cuda").manual_seed(37))
    # forest selections contain completed cells only (the reference gathers whole 32-row cells
    # before masking): two closed int8 cells, like a real re-ranked selection
    sel = torch.tensor([[[1, 2]]] * (B * H), device="cuda", dtype=torch.int64)
    pos_t = torch.full((), n, dtype=torch.int64, device="cuda")
    from src.fsattn.fsb_fast_step import decode_readout_fast
    from src.fsattn.fsb_tree import _fsb_decode_readout
    out = decode_readout_fast(attn, c, q_t, sel, pos_t)
    assert out is not None, "the fused int8 readout must accept an fp32 lr_norm.scale"
    ref = _fsb_decode_readout(attn, q_t, c, sel, pos_t, full_buffer=True)
    # both read the same quantized cells; only the fp32-norm promotion order can differ
    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@gpu
@torch.no_grad()
@pytest.mark.parametrize("int8,ln_f32", [(True, True), (True, False), (False, False)])
def test_decread_production_shape_matches_reference(monkeypatch, int8, ln_f32):
    # the fused decode readouts at the released models' shape (16 heads of 64, four read cells) against the torch
    # reference: nested reductions of the (cells, tokens, Dh) tile once miscompiled silently at this register
    # layout while the toy shapes above stayed exact, so the shape itself is pinned here
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    from src.fsattn import fsb_fast_step as fst
    from src.fsattn.fsb_tree import _fsb_decode_readout
    attn, c = _fsb_attn_and_int8_cache(monkeypatch, n=1024, d_model=1024, n_heads=16, int8=int8)
    if ln_f32:
        attn.lr_norm.scale.data = attn.lr_norm.scale.data.float()
    B, H, Dh, K = 1, 16, attn.head_dim, 4
    assert Dh == 64
    q_t = torch.randn(B, H, 1, Dh, device="cuda", dtype=torch.bfloat16,
                      generator=torch.Generator(device="cuda").manual_seed(37))
    g = torch.Generator(device="cuda").manual_seed(5)
    sel = torch.stack([torch.randperm(c._len // 32, device="cuda", generator=g)[:K] for _ in range(B * H)])
    sel = sel.view(B * H, 1, K)
    pos_t = torch.full((), c._len, dtype=torch.int64, device="cuda")
    real_ok = fst.decode_readout_ok
    # the bf16 readout is fused only inside a capture; run the same launch eagerly
    monkeypatch.setattr(fst, "decode_readout_ok", lambda *a, **k: real_ok(*a, **{**k, "must_capture": False}))
    out = fst.decode_readout_fast(attn, c, q_t, sel, pos_t)
    assert out is not None
    ref = _fsb_decode_readout(attn, q_t, c, sel, pos_t, full_buffer=True)
    assert out.shape == ref.shape
    o, r = out.float().reshape(-1), ref.float().reshape(-1)
    assert ((o - r).norm() / r.norm()).item() < 1e-2


@gpu
@torch.no_grad()
def test_decread_int8_replay_across_cell_close(monkeypatch):
    # a captured fused int8 readout must stay valid when later tokens close more cells: no baked length
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    attn, c = _fsb_attn_and_int8_cache(monkeypatch)
    B, H, Dh = 1, 4, attn.head_dim
    q_t = torch.randn(B, H, 1, Dh, device="cuda", dtype=torch.bfloat16,
                      generator=torch.Generator(device="cuda").manual_seed(41))
    sel = torch.tensor([[[1, 3]]] * (B * H), device="cuda", dtype=torch.int64)
    pos_t = torch.full((), c._len, dtype=torch.int64, device="cuda")
    from src.fsattn.fsb_fast_step import decode_readout_fast
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        decode_readout_fast(attn, c, q_t, sel, pos_t)  # warm-up: triton autotune/compilation off the capture
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        out_static = decode_readout_fast(attn, c, q_t, sel, pos_t)
    assert out_static is not None
    x = torch.randn(1, 40, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(43))
    for t in range(40):                     # 128 -> 168: closes cell 4 and opens cell 5
        attn(x[:, t:t + 1].bfloat16(), cache=c)
    n2 = c._len
    assert c._n_q == 160, "a new cell must have closed after the capture"
    sel.fill_(2).view(B * H, 1, 2)[:, :, 1] = 5
    pos_t.fill_(n2)
    g.replay()
    got = out_static.clone()
    want = decode_readout_fast(attn, c, q_t, sel, pos_t)
    assert torch.equal(got, want)


@gpu
@torch.no_grad()
def test_fsb_reference_readout_capture_guard(monkeypatch):
    # the reference readout bakes a gather over the length-sliced materialization: refuse capture loudly
    attn, c = _fsb_attn_and_int8_cache(monkeypatch)
    B, H, Dh = 1, 4, attn.head_dim
    q_t = torch.randn(B, H, 1, Dh, device="cuda", dtype=torch.bfloat16)
    sel = torch.tensor([[[1, 2]]] * (B * H), device="cuda", dtype=torch.int64)
    pos_t = torch.full((), c._len, dtype=torch.int64, device="cuda")
    from src.fsattn import fsb_tree
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="must not be captured"):
        fsb_tree._fsb_decode_readout(attn, q_t, c, sel, pos_t, full_buffer=True)


@gpu
@torch.no_grad()
def test_fsb_int8_graphed_decode_fp32_lr_norm_across_blocks(monkeypatch):
    # the RULER fingerprint: fp32 norms + int8 KV + decode graph replayed past the captured length.
    # Before the fix this crashed at the first replay whose selection crossed the captured n.
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "scaling"))
    import bench_efficiency as be
    from decode_graph import FSBGraphedDecoder
    from experiments.scaling.model import FreeSoloLM
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_KV_INT8", "1")
    kw = be.serve_kwargs("fsb", 0.5)
    torch.manual_seed(41)
    L = 1024
    m = FreeSoloLM(vocab_size=512, d_model=256, n_layers=2, n_heads=4, d_ff=512, max_seq_len=L + 512,
                   attn_type="fsattn", **kw).cuda().to(torch.bfloat16).eval()
    for blk in m.blocks:
        blk.attn.fsattn.lr_norm.scale.data = blk.attn.fsattn.lr_norm.scale.data.float()
    prompt = torch.randint(0, 512, (1, L), generator=torch.Generator().manual_seed(43)).cuda()
    forced = torch.randint(0, 512, (300,), generator=torch.Generator().manual_seed(47)).cuda()

    runs = {}
    for graph in (0, 1):
        monkeypatch.setenv("FSB_DECODE_GRAPH", str(graph))
        gd = FSBGraphedDecoder(m, prompt, 300, device="cuda", dtype=torch.bfloat16)
        gd.prefill()
        for blk, c in zip(m.blocks, gd.cache):
            f = blk.attn.fsattn
            f.fsb_tau.fill_(0.0)
            f.fsb_tau_union.copy_(c._ceil[0].float().median(dim=-1).values)
            f.fsb_tau_union[0] = c._ceil[0, 0].float().kthvalue(2).values
        assert all(getattr(c, "use_int8", False) for c in gd.cache)
        outs = []
        for t in forced:
            gd.next_tok = t.view(1)           # teacher-forced: both runs see the same stream
            outs.append(gd.step().clone())
        runs[graph] = torch.cat(outs)
    agree = (runs[0] == runs[1]).float().mean().item()
    assert agree >= 0.99, agree


@gpu
@torch.no_grad()
@pytest.mark.parametrize("serve", ["sparse", "dense"])
def test_fsl_fs_int8_graphed_decode_across_blocks(monkeypatch, serve):
    # the FS-L / base-FS long-range decode on the int8 store: the fused lr_comb_int8 kernel must match
    # the bf16 graph decode over teacher-forced steps that cross a query-block boundary
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "scaling"))
    import bench_efficiency as be
    from decode_graph import GraphedDecoder
    from experiments.scaling.model import FreeSoloLM
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    kw = be.serve_kwargs(serve, 0.5)
    torch.manual_seed(52)
    L = 1024
    m = FreeSoloLM(vocab_size=512, d_model=256, n_layers=2, n_heads=4, d_ff=512, max_seq_len=L + 512,
                   attn_type="fsattn", **kw).cuda().to(torch.bfloat16).eval()
    for blk in m.blocks:
        blk.attn.fsattn.lr_norm.scale.data = blk.attn.fsattn.lr_norm.scale.data.float()
    prompt = torch.randint(0, 512, (1, L), generator=torch.Generator().manual_seed(53)).cuda()
    forced = torch.randint(0, 512, (300,), generator=torch.Generator().manual_seed(57)).cuda()

    runs = []
    for i8 in (False, True, True):
        monkeypatch.setenv("FS_KV_INT8", "1" if i8 else "0")
        gd = GraphedDecoder(m, prompt, Lmax=L + 300 + 16, device="cuda", dtype=torch.bfloat16)
        gd.prefill()
        assert all(getattr(c, "use_int8", False) == i8 for c in gd.cache)
        outs = []
        for t in forced:
            gd.next_tok = t.view(1)           # teacher-forced: both runs see the same stream
            outs.append(gd.step().clone())
        runs.append(torch.cat(outs))
    assert (runs[1] == runs[2]).all(), "the int8 decode graph must replay deterministically"
    agree = (runs[0] == runs[1]).float().mean().item()
    # cross-mode agreement within the cell quantization noise (sc/2 per row); a kernel bug instead
    # shows up clustered (boundary steps) or as a tail of flips, which the 0.97 floor rejects
    assert agree >= 0.97, agree


@gpu
@torch.no_grad()
@pytest.mark.parametrize("serve", ["sparse", "dense"])
def test_lr_comb_int8_matches_bf16(monkeypatch, serve):
    # kernel-level parity: lr_comb_int8 vs fsl_lr_decode_combine on the materialized (dequantized) cache;
    # the dequantization rounds back to bf16, so only the int8 cell error may show, not kernel drift
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "scaling"))
    import bench_efficiency as be
    from decode_graph import GraphedDecoder
    from experiments.scaling.model import FreeSoloLM
    from src.utils import trunk_triton as _tt
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    kw = be.serve_kwargs(serve, 0.5)
    torch.manual_seed(61)
    L = 1024
    m = FreeSoloLM(vocab_size=512, d_model=256, n_layers=2, n_heads=4, d_ff=512, max_seq_len=L + 512,
                   attn_type="fsattn", **kw).cuda().to(torch.bfloat16).eval()
    prompt = torch.randint(0, 512, (1, L), generator=torch.Generator().manual_seed(59)).cuda()
    monkeypatch.setenv("FS_KV_INT8", "1")
    # the full int8 store, whose history and buffers this test reads (the bounded store, FSL_KV_BOUNDED=1, keeps
    # neither: tests/test_kv_bounded.py checks its reads against this store)
    monkeypatch.setenv("FSL_KV_BOUNDED", "0")
    gd = GraphedDecoder(m, prompt, Lmax=L + 64, device="cuda", dtype=torch.bfloat16)
    gd.prefill()
    for c in gd.cache:
        assert c.use_int8
    gd._refresh_block_pool()                         # pool tables over the pre-quant cell statistics
    attn = gd.attn[0]
    c = gd.cache[0]
    q_t = torch.randn(1, 4, 1, 64, device="cuda", dtype=torch.bfloat16,
                      generator=torch.Generator(device="cuda").manual_seed(63))
    pos_t = gd.pos_t
    Kp, Vp = c.read_full_kv()
    B, H, _, D = q_t.shape
    sr = torch.zeros_like(q_t)
    pool = gd.layer_bufs[0]["pool"]
    st = (c._qk_buf, c._qv_buf, c._ks_buf, c._vs_buf, c._k_hot, c._v_hot, c._nq_t, c.s)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = _tt.lr_comb_int8(attn, q_t, sr, pool, pos_t, attn.sigma, st)
        assert out is not None, "lr_comb_int8 must cover the bench configuration"
        ref = _tt.fsl_lr_decode_combine(attn, q_t, Kp, Vp, sr, pool, pos_t, attn.sigma)
    sr_out = out.float().view(B, 1, H, D).transpose(1, 2)
    sr_ref = ref.float().view(B, 1, H, D).transpose(1, 2)
    assert (sr_out - sr_ref).abs().max().item() < 0.05
