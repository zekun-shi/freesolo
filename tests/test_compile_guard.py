"""torch.compile and the hand-written inference paths (src/utils/compile_guard.py): while dynamo traces, every gate of
a Triton / ctypes CUDA fast path reads False (with no graph break of its own), so a compiled model traces the eager
reference ops, as before these paths existed. Without the guard dynamo traced the Triton launches (torch 2.5 passes
enable_fp_fusion as a kernel argument): NaN validation loss, wrong compiled logits, and hundreds of graph breaks."""
import os
import sys

import pytest
import torch

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "scaling")]

from src.utils.compile_guard import compiling  # noqa: E402

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# every inference switch of the trunk, window, readout and FS-L selection ("0": the eager path)
FSL_SWITCHES = ("FSL_WINDOW_BATCHED", "FSL_TRUNK_FUSED", "FSL_NORM_ONEPASS", "FSL_ROPE_FUSED", "FSL_PREFILL_CELL_REUSE",
                "FSL_READOUT_FUSED", "FSL_FAST_SELECT", "FSL_FUSED_CONTENT", "FSL_FAST_CACHE_STATS")


def _traced(fn, *args):
    # fn(*args) as dynamo traces it (eager backend, one graph: a graph break inside fn raises)
    torch._dynamo.reset()
    out = torch.compile(lambda *a: (a[0] * 1, fn(*a)), backend="eager", fullgraph=True)(*args)[1]
    torch._dynamo.reset()
    return out


def test_compiling_folds_to_true_under_dynamo():
    assert compiling() is False
    assert _traced(lambda x: compiling(), torch.ones(3)) is True


@gpu
@torch.no_grad()
def test_trunk_and_readout_gates_off_under_compile(monkeypatch):
    from src.fsattn.readout_triton import readout_ok
    from src.utils import trunk_triton as tt
    for name in FSL_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    g = torch.Generator(device="cuda").manual_seed(0)
    x4 = torch.randn(1, 4, 256, 64, device="cuda", generator=g).to(torch.bfloat16)
    cos = torch.randn(256, 32, device="cuda", generator=g)
    x = torch.randn(1, 64, 256, device="cuda", generator=g)
    w = torch.ones(256, device="cuda")
    gu = torch.randn(64, 704, device="cuda", generator=g).to(torch.bfloat16)
    N, H, D = 512, 2, 64
    Q = torch.randn(1, H, N, D, device="cuda", generator=g).to(torch.bfloat16)
    sel = torch.zeros(1, H, N, 4, device="cuda", dtype=torch.int64)
    gates = {"rope_ok": (lambda a, c: tt.rope_ok(a, c), (x4, cos)),
             "silu_mul_ok": (lambda a: tt.silu_mul_ok(a, a), (gu,)),
             "readout_ok": (lambda q, s: readout_ok(q, q, q, s, q, cell_size=32, query_block=128), (Q, sel))}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gates["rms_norm_ok"] = (lambda a, b: tt.rms_norm_ok(a, b), (x, w))
        for name, (fn, args) in gates.items():
            assert fn(*args) is True, name
            assert _traced(fn, *args) is False, name


@gpu
@torch.no_grad()
def test_window_batched_off_under_compile(monkeypatch):
    import src.utils.local_attention as la
    monkeypatch.delenv("FSL_WINDOW_BATCHED", raising=False)
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
    calls = []
    real = la._batched_sliding_window
    monkeypatch.setattr(la, "_batched_sliding_window", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    g = torch.Generator(device="cuda").manual_seed(1)
    Q, K, V = (torch.randn(1, 4, 1000, 64, device="cuda", generator=g).to(torch.bfloat16) for _ in range(3))

    def win(q, k, v):
        return la.local_attention(q, k, v, 128, scale=0.125, causal=True, sigma=None, fast_sliding=True)

    ref = win(Q, K, V)
    assert calls == [1]
    torch._dynamo.reset()
    got = torch.compile(win, backend="eager")(Q, K, V)
    torch._dynamo.reset()
    assert calls == [1] and torch.equal(got, ref)        # the chunked path: the same bits, nothing batched


@gpu
@torch.no_grad()
def test_fsl_fast_paths_off_under_compile(monkeypatch):
    from src.fsattn import fsl_select_cuda as fsc
    for name in ("FSL_FAST_SELECT", "FSL_FAST_CACHE_STATS"):
        monkeypatch.delenv(name, raising=False)
    Q = torch.zeros(1, 2, 256, 64, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(1, 2, 256, 64, device="cuda").to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert fsc.fast_select_ok(Q, Q, 8, 8, 4, 128, 256)
        assert _traced(lambda q: fsc.fast_select_ok(q, q, 8, 8, 4, 128, 256), Q) is False
    if fsc.cache_cell_stats(K, 8, 32) is None:
        pytest.skip("no NVRTC: the cache statistics kernel is off anyway")
    assert _traced(lambda k: fsc.cache_cell_stats(k, 8, 32) is None, K) is True


@gpu
@torch.no_grad()
def test_fsb_fast_paths_off_under_compile(monkeypatch):
    # the fast prefill (track E), the CUDA rows path, its deferred check and low-part skip (A), the compact beam
    # layout (A / D) and the row-cooperative beam with the current warp kernel (D) read as their legacy settings
    from src.fsattn import fsb_beam_cuda as fbc
    from src.fsattn import fsb_prefill as fp
    from src.fsattn import fsb_rows as fr
    from src.fsattn import fsb_tree as ft
    from src.fsattn.attention import FSAttn
    for name in ("FSB_FAST_PREFILL", "FSB_ROWS_IMPL", "FSB_ROWS_DEFER", "FSB_ROWS_LOSKIP", "FSB_BEAM_COMPACT",
                 "FSB_BEAM_ROWS", "FSB_TRAIN_GRAPH", "FSB_EVAL_GRAPH"):
        monkeypatch.delenv(name, raising=False)
    attn = FSAttn(d_model=256, n_heads=4, causal=True, lr_scale_mode="one", window_size=512,
                  long_range_variant="cell_topk", cell_stride=32, cell_topk=4, cell_select="score", sel_strict_causal=1,
                  use_rope=False, fs_uniform_sr=True, rope_long_range=False, cell_prune_kernel=0, fsb_enabled=True,
                  fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_node_score="bound", fsb_bound_radius_coef=8.0,
                  fsb_tree="forest", fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw",
                  fsb_union_width=16).cuda().to(torch.bfloat16).eval()
    Q = torch.randn(1, 4, 600, 64, device="cuda").to(torch.bfloat16)
    tokK = torch.randn(1, 4, 640, 64, device="cuda").to(torch.bfloat16)
    eager = (fp.prefill_fast_ok(attn, Q, tokK, 128), fr.rows_impl(), fr.rows_defer_on(), fr.rows_loskip_on(),
             ft.beam_delta_eff(attn), fbc.rows_flag())
    assert eager == (True, "cuda", True, True, 0, True), eager
    traced = _traced(lambda q, k: (fp.prefill_fast_ok(attn, q, k, 128), fr.rows_impl(), fr.rows_defer_on(),
                                   fr.rows_loskip_on(), ft.beam_delta_eff(attn), fbc.rows_flag()), Q, tokK)
    assert traced == (False, "legacy", False, False, None, False), traced


def _small_lm(serve):
    import bench_efficiency as be
    from experiments.scaling.model import FreeSoloLM
    kw = be.serve_kwargs(serve)
    kw["window_size"] = 64                       # batched-window eligible (64-key tiles), several windows per prompt
    torch.manual_seed(7)
    return FreeSoloLM(vocab_size=512, d_model=256, n_layers=2, n_heads=4, d_ff=704, max_seq_len=1600,
                      attn_type="fsattn", **kw).cuda().eval()


def _env(monkeypatch, serve, off=False):
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_STRICT_CAUSAL", "1")
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "1" if serve == "sparse" else "0")
    if serve == "sparse":
        monkeypatch.setenv("FS_FUSED_SELECT", "1")
    for name in FSL_SWITCHES:
        if off:
            monkeypatch.setenv(name, "0")
        else:
            monkeypatch.delenv(name, raising=False)


@gpu
@pytest.mark.parametrize("serve", ["sparse", "dense"])
def test_compiled_prefill_breaks_only_at_the_readout(monkeypatch, serve):
    # FSAttn runs the long-range branch without the readout hand-off and outside a try block while dynamo traces
    # (_lr_handoff_off): a graph break inside a try block cannot be resumed, so dynamo would run the whole attention
    # forward uncompiled ('step_unsupported'). The only breaks left are the readout's, as before the hand-off.
    from kv_cache import _trunk_cached, make_caches
    from src.fsattn.attention import FSAttn
    assert FSAttn._lr_handoff_off() is False and _traced(lambda x: FSAttn._lr_handoff_off(), torch.ones(2)) is True
    _env(monkeypatch, serve)
    m = _small_lm(serve)
    x = torch.randint(0, 512, (1, 1500), device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
    for fn, args in ((m, (x,)), (_trunk_cached, (m, x, make_caches(m)))):
        torch._dynamo.reset()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ex = torch._dynamo.explain(fn)(*args)
        torch._dynamo.reset()
        reasons = [br.reason for br in ex.break_reasons]
        assert reasons and all("indexed_cell_attention" in r for r in reasons), reasons


@gpu
@pytest.mark.parametrize("serve", ["sparse", "dense"])
def test_compiled_model_traces_the_eager_paths(monkeypatch, serve):
    # train_dist's compiled validation (model(x) in eval, no_grad, bf16 autocast, FS_FUSED_ATTN=1) and
    # bench_efficiency --compile 1's compiled cached prefill (_trunk_cached), FS-L and FS: with the default switches
    # the compiled outputs equal those with every switch off bit for bit (the same traced graph), are finite and close
    # to eager, and the only graph breaks are the readout's (indexed_cell_attention runs outside the graph, as before)
    import torch.nn.functional as F
    from kv_cache import _trunk_cached, make_caches
    from torch._dynamo.utils import counters
    _env(monkeypatch, serve)
    m = _small_lm(serve)
    x = torch.randint(0, 512, (2, 1500), device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))

    def loss(lg):
        return F.cross_entropy(lg[:, :-1].float().reshape(-1, 512), x[:, 1:].reshape(-1)).item()

    def compiled(fn, args, off):
        _env(monkeypatch, serve, off)
        torch._dynamo.reset()
        counters.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = torch.compile(fn)(*args())
        breaks = dict(counters.get("graph_break", {}))
        torch._dynamo.reset()
        return out, breaks

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager = m(x)
    lg_def, gb = compiled(m, lambda: (x,), off=False)
    lg_off, _ = compiled(m, lambda: (x,), off=True)
    assert torch.isfinite(lg_def.float()).all() and torch.equal(lg_def, lg_off)
    assert abs(loss(lg_def) - loss(eager)) < 1e-3, (loss(lg_def), loss(eager))
    assert gb and all("indexed_cell_attention" in str(k) for k in gb), gb
    h_def, gb = compiled(_trunk_cached, lambda: (m, x[:1], make_caches(m)), off=False)
    h_off, _ = compiled(_trunk_cached, lambda: (m, x[:1], make_caches(m)), off=True)
    assert torch.isfinite(h_def.float()).all() and torch.equal(h_def, h_off)
    assert gb and all("indexed_cell_attention" in str(k) for k in gb), gb
