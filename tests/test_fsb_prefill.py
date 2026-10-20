"""The batched FS-B prefill stage plumbing (src/fsattn/fsb_prefill.py, FSB_FAST_PREFILL) against fsb_long_range's own
stage loop (FSB_FAST_PREFILL=0): output, selected cells, statistics and read accumulators bit for bit, across padded
lengths, batch, both forests or one, re-rank overflow, stage sizes and both readouts; no host sync when no row can
overflow; the per-layer launch count; the shared pieces it must use (fsb_rows.union_rerank_rows with the deferred
flags and bf16 queries, both beams in one forest_beam_pair launch in the compact layout, the cell statistics handed to
the KV cache, the fused readout's hand-off across an overflow redo); the window branch forked next to the long-range
branch (FSB_WINDOW_OVERLAP) against the serial prefill, eager and CUDA-graph captured; the trimmed forest build, the
in-place stage views and the faster re-rank and beam kernels (FSB_BUILD_PREFILL_TRIM, FSB_PREFILL_NOCOPY, FSB_ROWS_V3,
FSB_BEAM_V2) in every combination at the released shape."""
import os
import sys

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_REPO, "experiments", "scaling"), _REPO]

from src.fsattn import fsb_prefill as fp
from src.fsattn import fsb_rows as rows
from src.fsattn import fsb_tree as ft
from src.fsattn.attention import FSAttn

gpu = pytest.mark.skipif(not (torch.cuda.is_available() and fp._HAVE_TRITON), reason="needs CUDA and Triton")

# the paper's FS-B serve config (experiments/scaling/bench_efficiency.py serve_kwargs("fsb"), union tau 0.5)
SERVE = dict(window_size=512, long_range_variant="cell_topk", cell_stride=32, cell_topk=4, cell_select="score",
             sel_strict_causal=1, use_rope=False, fs_uniform_sr=True, rope_long_range=False, cell_prune_kernel=0,
             fsb_enabled=True, fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_node_score="bound",
             fsb_bound_radius_coef=8.0, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score="lse_pos",
             fsb_rerank_query="raw", fsb_union_width=16, fsb_union_tau_quantile=0.5)


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    v = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}.get(t.dtype)
    return t.view(v) if v is not None else t


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _attn(H=4, D=64, blkchunk=32, cap=False, **over):
    kw = dict(SERVE)
    kw.update(over)
    if kw.get("fsb_union_width", 0) == 0 or kw.get("fsb_union_tau_quantile", 0) == 0:
        kw.pop("fsb_union_tau_quantile", None)
    m = FSAttn(d_model=H * D, n_heads=H, causal=True, lr_scale_mode="one", fsb_prefill_blkchunk=blkchunk, **kw)
    # blkchunk sizes the reference loop's stages; as a given memory bound it also caps the fast path's stages, which
    # the tests apply only with cap=True (elsewhere the fast path keeps its own stage size, as by default)
    m.fsb_prefill_blkchunk_set = cap
    m = m.cuda().to(torch.bfloat16).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        m.sel_scale.copy_(torch.rand(H, generator=g) * 2 + 0.5)
    return m


def _inputs(seed, B, H, N, D=64, clustered=True):
    # (B, H, N, D) bf16 views of (B, N, H, D) projections, as FSAttn passes them; clustered keys and queries so that a
    # block's queries share candidate cells as on real prompts
    g = torch.Generator().manual_seed(seed)
    if clustered:
        centers = torch.randn(H, 24, D, generator=g)
        pick = torch.randint(0, 24, (B, N, H), generator=g)
        base = centers[torch.arange(H)[None, None, :], pick]
        K = (base + 0.6 * torch.randn(B, N, H, D, generator=g)) * 0.5
        Q = (centers[torch.arange(H)[None, None, :], torch.randint(0, 24, (B, N, H), generator=g)]
             + 0.8 * torch.randn(B, N, H, D, generator=g)) * 0.5
    else:
        K = torch.randn(B, N, H, D, generator=g)
        Q = torch.randn(B, N, H, D, generator=g)
    V = torch.randn(B, N, H, D, generator=g)
    sigma = torch.rand(H, generator=g) * 600 + 40
    f = lambda x: x.to("cuda", torch.bfloat16).transpose(1, 2)
    return f(Q), f(K), f(V), sigma.cuda()


def _set_tau(attn, K, tau_q=None, union_q=0.5):
    # fsb_tau at a per-head ceil quantile (None: 0, as the bench), fsb_tau_union at the per-head union quantile
    B, H, N, D = K.shape
    s = attn.cell_stride
    M = (N + s - 1) // s
    Kp = torch.cat([K, K.new_zeros(B, H, M * s - N, D)], 2) if M * s > N else K
    ceil = Kp.view(B, H, M, s, D).float().norm(dim=-1).amax(-1).transpose(0, 1).reshape(H, -1)
    with torch.no_grad():
        attn.fsb_tau.copy_(torch.zeros(H) if tau_q is None else ceil.quantile(tau_q, dim=1))
        attn.fsb_tau_union.copy_(ceil.quantile(union_q, dim=1))


def _run(monkeypatch, attn, Q, K, V, sigma, fast, acc=False, stats=False, fused=True):
    monkeypatch.setenv("FSB_FAST_PREFILL", "1" if fast else "0")
    monkeypatch.setenv("FSB_PREFILL_STATS", "1" if stats else "0")
    monkeypatch.setenv("FS_FUSED_ATTN", "1" if fused else "0")
    attn._fsb_read_acc = torch.zeros(5, device="cuda") if acc else None
    rec = []
    orig = FSAttn._cell_fused_lr

    def _hook(Q_, K_, V_, sel, s, Bq, **kw):
        rec.append(sel.clone())
        return orig(attn, Q_, K_, V_, sel, s, Bq, **kw)

    attn._cell_fused_lr = _hook
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = ft.fsb_long_range(attn, Q, K, V, sigma, Q.shape[2])
    finally:
        del attn._cell_fused_lr
    st = attn._fsb_stats
    assert isinstance(st, fp.LazyStats) == fast
    # the last readout: after an overflow both paths discard the first one they queued and redo it
    return {"out": out, "sel": rec[-1] if rec else None, "stats": {k: st[k] for k in st},
            "acc": None if attn._fsb_read_acc is None else attn._fsb_read_acc.clone()}


def _assert_same(got, ref):
    # got: the fast path; ref: the reference loop. Every statistic the fast path keeps equals the reference's (it
    # drops has / cost / n_read / n_visit unless FSB_PREFILL_STATS=1); the reference's key order
    assert _same(got["out"], ref["out"])
    if got["sel"] is not None or ref["sel"] is not None:
        assert _same(got["sel"], ref["sel"])
    assert list(got["stats"]) == [k for k in ref["stats"] if k in got["stats"]]
    assert {"admit_frac", "m0_mean"} <= set(got["stats"])
    for k in got["stats"]:
        assert _same(got["stats"][k], ref["stats"][k]), k
    if got["acc"] is not None or ref["acc"] is not None:
        assert _same(got["acc"], ref["acc"])


CASES = {
    "utau": dict(),
    "union_no_tau": dict(fsb_union_tau_quantile=0.0),
    "no_union": dict(fsb_union_width=0),
    "lse": dict(fsb_rerank_score="lse"),
    "max": dict(fsb_rerank_score="max"),
    "delta1": dict(fsb_delta=1),
    "narrow": dict(fsb_width=8, fsb_union_width=4),
    "centroid": dict(fsb_node_score="centroid"),
    "leaf_radius": dict(fsb_leaf_radius=1),
}


@gpu
@pytest.mark.parametrize("N", [100, 300, 1100, 4096])
@pytest.mark.parametrize("case", list(CASES))
def test_fast_prefill_matches_reference(monkeypatch, N, case):
    attn = _attn(**CASES[case])
    Q, K, V, sigma = _inputs(10 + N, 1, 4, N)
    _set_tau(attn, K, tau_q=0.3 if case == "utau" else None)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True)
    got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True)
    _assert_same(got, ref)


@gpu
@pytest.mark.parametrize("compact", ["0", "1"])
@pytest.mark.parametrize("case", ["utau", "delta1"])
def test_compact_layout_switch(monkeypatch, compact, case):
    monkeypatch.setenv("FSB_BEAM_COMPACT", compact)                # one switch for both paths
    attn = _attn(**CASES[case])
    Q, K, V, sigma = _inputs(55, 1, 4, 3000)
    _set_tau(attn, K, tau_q=0.25)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True)
    got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True)
    _assert_same(got, ref)


@gpu
@pytest.mark.parametrize("delta", [0, 1, 2])
@pytest.mark.parametrize("admit_all", [True, False])
def test_compact_beam_output(delta, admit_all):
    # with the fixed policy every final frontier node holds at most 2^delta cells: the beam's compact output (2^delta
    # slots per node) is the reference's first 2^delta slots of every node, and the reference's other slots are -1
    from src.fsattn.fsb_fast_step import forest_beam_fused
    g = torch.Generator().manual_seed(17 + delta)
    H, M, D, Bq, nbc, s = 4, 512, 64, 128, 8, 32
    keys = torch.randn(H, M * s, D, generator=g).to("cuda", torch.bfloat16)
    kv = keys.float().view(H, M, s, D)
    kbar, ceil = kv.mean(2), kv.norm(dim=-1).amax(-1)
    admit = torch.ones_like(ceil, dtype=torch.bool) if admit_all else ceil > ceil.median()
    F = ft.build_forest(kbar, ceil, admit)
    blk = torch.arange(M * s // Bq, device="cuda", dtype=torch.float32)[-nbc:] * Bq
    m_b = torch.searchsorted(F["adm"].masked_fill(F["adm"] < 0, M),
                             (blk / s).floor().long()[None].expand(H, nbc).contiguous()).reshape(-1)
    SP = ft.build_spines(F, m_b, torch.arange(H, device="cuda").repeat_interleave(nbc))
    n = H * nbc
    q = torch.randn(n, Bq, D, generator=g).to(torch.bfloat16).float().cuda()
    qpos = (blk[:, None] + torch.arange(Bq, device="cuda", dtype=torch.float32))[None].expand(H, nbc, Bq)
    qpos = qpos.reshape(n, Bq).contiguous()
    lam = (torch.rand(n, 1, 1, generator=g) * 2 + 0.5).cuda()
    sigv = (torch.rand(n, 1, 1, generator=g) * 1000 + 50).cuda()
    h = SP["height"]
    Dn = (h - delta).clamp(min=0)
    W = torch.minimum(torch.full_like(Dn, 16), torch.bitwise_left_shift(torch.ones_like(Dn), Dn))[:, None].expand(n, Bq)
    dl = (h[:, None] - Dn[:, None]).clamp(min=0).expand(n, Bq).contiguous()
    ref, nv = forest_beam_fused(F, SP, q, qpos, lam, sigv, 0.125, s, W, dl, 16, 2, "bound", 8.0)
    got, nv2 = forest_beam_fused(F, SP, q, qpos, lam, sigv, 0.125, s, W, dl, 16, delta, "bound", 8.0)
    Re = 1 << delta
    r = ref.view(n, Bq, 16, 4)
    assert torch.equal(got.view(n, Bq, 16, Re), r[..., :Re]) and bool((r[..., Re:] == -1).all())
    assert torch.equal(nv, nv2) and bool((got >= 0).any())


@gpu
@pytest.mark.parametrize("N,B,clustered", [(16384, 1, True), (20000, 1, False), (2500, 2, True)])
def test_fast_prefill_long_and_batched(monkeypatch, N, B, clustered):
    # 16384: M = UCAP (no capacity check); 20000: M > UCAP (the deferred check runs); B = 2: rows of two sequences
    attn = _attn()
    Q, K, V, sigma = _inputs(3 + N, B, 4, N, clustered=clustered)
    _set_tau(attn, K)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True)
    got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True)
    _assert_same(got, ref)


@gpu
def test_unfused_forest_runs_the_reference_stages(monkeypatch):
    # a forest the fused beam does not take (fast_beam_ok False): the fast path runs the reference's own stages
    from src.fsattn import fsb_fast_step
    monkeypatch.setattr(fsb_fast_step, "fast_beam_ok", lambda *a, **k: False)
    attn = _attn()
    Q, K, V, sigma = _inputs(23, 1, 4, 700)
    _set_tau(attn, K)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True, stats=True)
    got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True, stats=True)
    _assert_same(got, ref)


@gpu
def test_stats_and_masked_readout(monkeypatch):
    attn = _attn()
    Q, K, V, sigma = _inputs(77, 1, 4, 1500)
    _set_tau(attn, K, tau_q=0.2)
    for fused in (True, False):
        ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, stats=True, fused=fused)
        got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, stats=True, fused=fused)
        assert {"has", "cost", "n_read", "n_visit", "admit_frac", "m0_mean", "admit_frac_union"} <= set(got["stats"])
        _assert_same(got, ref)
    # without FSB_PREFILL_STATS the eval statistics stay, computed on first access; the per-query ones are absent
    lean = _run(monkeypatch, attn, Q, K, V, sigma, fast=True)
    assert list(lean["stats"]) == ["admit_frac", "m0_mean", "admit_frac_union"]
    for k in lean["stats"]:
        assert _same(lean["stats"][k], ref["stats"][k]), k
    st = attn._fsb_stats
    assert "admit_frac" in st and "has" not in st and st.get("has") is None and bool(st)


@gpu
@pytest.mark.parametrize("fast_blk", ["1", "3", "0"])
def test_overflow_stages_match_reference(monkeypatch, fast_blk):
    # a small re-rank capacity makes some rows overflow: the reference recomputes those of its 2-block stages with
    # rerank_top; the fast path must recompute exactly those stages, whatever its own stage size
    N = 4096
    attn = _attn(blkchunk=2)
    Q, K, V, sigma = _inputs(41, 1, 4, N)
    _set_tau(attn, K)
    seen = []
    real = rows.union_rerank_rows

    def spy(*a, **k):
        r = real(*a, **k)
        seen.append(r is None)
        return r

    monkeypatch.setattr(rows, "union_rerank_rows", spy)
    monkeypatch.setattr(rows, "UCAP", 40)
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", fast_blk)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True, stats=True)
    assert any(seen) and not all(seen), seen                      # mixed: some stages overflow, some do not
    got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True, stats=True)
    assert list(got["stats"]) == list(ref["stats"])
    _assert_same(got, ref)


def _host_syncs(fn) -> int:
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    # (the profiler's own stop records a cudaDeviceSynchronize: not counted)
    names = ("cudaStreamSynchronize", "cudaEventSynchronize", "aten::_local_scalar_dense")
    return sum(e.count for e in prof.key_averages() if e.key in names)


@gpu
@pytest.mark.parametrize("trim", ["0", "1"])
def test_no_host_sync(monkeypatch, trim):
    # M = 256 cells <= UCAP: no row can overflow the re-rank, so neither path waits for the GPU (FSB_ROWS_DEFER);
    # the trimmed build's spine bound is host arithmetic
    attn = _attn(H=8)
    Q, K, V, sigma = _inputs(9, 1, 8, 8192)
    _set_tau(attn, K)
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FSB_BUILD_PREFILL_TRIM", trim)
    attn._fsb_read_acc = None

    def one():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return ft.fsb_long_range(attn, Q, K, V, sigma, 8192)

    syncs = {}
    for fast in ("0", "1"):
        monkeypatch.setenv("FSB_FAST_PREFILL", fast)
        one()                                                              # compile and cache
        syncs[fast] = _host_syncs(one)
    print(f"host syncs per layer at M = 256 cells: stage loop {syncs['0']}, fast path {syncs['1']}")
    assert syncs["1"] == 0, syncs
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = one()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.isfinite(out.float()).all()


def _kernels(fn) -> int:
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


@gpu
def test_launch_budget(monkeypatch):
    # one bench layer (H = 16, 16384 tokens): kernels of the long-range branch outside build_forest (whose two calls,
    # main and filtered forest, are replayed from a cache in both paths). The fast path's 39 (58 before it shared
    # fsb_rows' three-launch re-rank and the paired beam launch): cell statistics, the rest of the build stage
    # (admission, m_b), spines, beam tables, query copy, both beams in one launch, the re-rank's row union, LSE and
    # select, the cell table, readout and lr_norm; the stage loop's 299 repeat the per-stage plumbing every 32 blocks
    # and add the statistics
    attn = _attn(H=16)
    Q, K, V, sigma = _inputs(12, 1, 16, 16384)
    _set_tau(attn, K)
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    attn._fsb_read_acc = None
    built, calls = [], [0]
    real = ft.build_forest

    def cached(*a, **k):
        i = calls[0] % 2
        calls[0] += 1
        if len(built) <= i:
            built.append(real(*a, **k))
        return dict(built[i])

    monkeypatch.setattr(ft, "build_forest", cached)
    counts = {}
    for fast in ("0", "1"):
        monkeypatch.setenv("FSB_FAST_PREFILL", fast)

        def one():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ft.fsb_long_range(attn, Q, K, V, sigma, 16384)

        one()
        counts[fast] = _kernels(one)
    print(f"long-range kernels per layer outside build_forest at 16k: reference {counts['0']}, fast {counts['1']}")
    assert counts["1"] <= 45 and counts["0"] >= 5 * counts["1"], counts


@gpu
@torch.no_grad()
def test_gate(monkeypatch):
    attn = _attn()
    Q, K, V, sigma = _inputs(1, 1, 4, 600)
    tokK = K.detach()
    monkeypatch.setenv("FSB_FAST_PREFILL", "1")
    assert fp.prefill_fast_ok(attn, Q, tokK, 128)
    with torch.enable_grad():
        assert not fp.prefill_fast_ok(attn, Q, tokK, 128)               # an eval-mode forward under autograd
    for v in ("0", "legacy"):
        monkeypatch.setenv("FSB_FAST_PREFILL", v)
        assert not fp.prefill_fast_ok(attn, Q, tokK, 128)
    monkeypatch.setenv("FSB_FAST_PREFILL", "1")
    assert not fp.prefill_fast_ok(attn, Q, tokK.float(), 128)          # fp32 keys: rerank_top path
    attn.train()
    assert not fp.prefill_fast_ok(attn, Q, tokK, 128)
    attn.eval()
    monkeypatch.setenv("FSB_TRAIN_GRAPH", "1")
    monkeypatch.setenv("FSB_EVAL_GRAPH", "1")
    assert not fp.prefill_fast_ok(attn, Q, tokK, 128)                  # graphed eval stages
    monkeypatch.setenv("FSB_EVAL_GRAPH", "0")
    attn._fsb_calib = []
    assert not fp.prefill_fast_ok(attn, Q, tokK, 128)
    attn._fsb_calib = None
    assert not fp.prefill_fast_ok(_attn(fsb_read_margin=0.5), Q, tokK, 128)
    assert not fp.prefill_fast_ok(_attn(fsb_beam_quota=2), Q, tokK, 128)


@gpu
@pytest.mark.parametrize("utau", [True, False])
def test_row_tables_match_reference_formulas(utau):
    g = torch.Generator().manual_seed(3)
    rows_n, Bq, H = 333, 128, 4
    h = torch.randint(0, 14, (rows_n,), generator=g).cuda()
    h2 = torch.randint(0, 14, (rows_n,), generator=g).cuda()
    head = (torch.arange(rows_n) % H).cuda()
    sel_scale = torch.rand(H, generator=g).to("cuda", torch.bfloat16)
    sigma = (torch.rand(H, generator=g) * 100).cuda()
    attn = type("A", (), {"fsb_delta": 1, "fsb_width": 16, "fsb_union_width": 8})()
    SP = {"height": h}
    SP2 = {"height": h2} if utau else None
    W, bud, Wu, budu, lam, sig = fp._row_tables(attn, SP, SP2, sel_scale, sigma, head, Bq, True)
    D = (h - 1).clamp(min=0)
    W_ref = torch.minimum(torch.full_like(D, 16), torch.bitwise_left_shift(torch.ones_like(D), D))[:, None].expand(-1, Bq)
    delta = (h[:, None] - D[:, None].expand(-1, Bq)).clamp(min=0)
    if utau:
        D2 = (h2 - 1).clamp(min=0)
        du = (h2 - D2)[:, None].expand(-1, Bq)
        hu = h2
    else:
        du, hu = delta, h
    Du = (hu[:, None] - du).clamp(min=0)
    Wu_ref = torch.minimum(torch.full_like(Du, 8), torch.bitwise_left_shift(torch.ones_like(Du), Du))
    assert torch.equal(W, W_ref) and torch.equal(bud, torch.bitwise_left_shift(torch.ones_like(delta), delta))
    assert torch.equal(Wu, Wu_ref) and torch.equal(budu, torch.bitwise_left_shift(torch.ones_like(du), du))
    assert _same(lam, sel_scale.float()[head]) and _same(sig, sigma.float()[head])


@gpu
def test_cached_prefill_end_to_end(monkeypatch):
    # a 2-layer model's cached prefill (the eval drivers' path): the same hidden state with and without the fast path
    from experiments.scaling.model import FreeSoloLM
    from kv_cache import make_caches, _trunk_cached
    kw = dict(SERVE)
    kw.update(window_size=64, fsb_width=4, fsb_union_width=4)
    torch.manual_seed(0)
    m = FreeSoloLM(vocab_size=256, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_seq_len=2048,
                   attn_type="fsattn", **kw).cuda().to(torch.bfloat16).eval()
    layers = [mod for mod in m.modules() if getattr(mod, "fsb_enabled", False)]
    prompt = torch.randint(0, 256, (1, 1700), generator=torch.Generator().manual_seed(1)).cuda()
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FSB_FAST_PREFILL", "0")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        caches = make_caches(m)
        _trunk_cached(m, prompt, caches)
        for mod, c in zip(layers, caches):
            mod.fsb_tau.fill_(0.0)
            mod.fsb_tau_union.copy_(c._ceil[0].float().median(dim=-1).values)
    hs = {}
    for fast in ("0", "1"):
        monkeypatch.setenv("FSB_FAST_PREFILL", fast)
        ft.fsb_read_stats_reset(m)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hs[fast] = (_trunk_cached(m, prompt, make_caches(m)), torch.stack([mod._fsb_read_acc for mod in layers]))
    ft.fsb_read_stats_reset(m, on=False)
    assert _same(hs["0"][0], hs["1"][0]) and _same(hs["0"][1], hs["1"][1])


@gpu
def test_fast_prefill_uses_shared_rows_and_pair(monkeypatch):
    # the fast path runs fsb_rows' re-rank (deferred flags, bf16 queries) on both beams from one forest_beam_pair
    # launch in the compact layout (2^fsb_delta slots per frontier node), as the stage loop does
    from src.fsattn import fsb_fast_step
    calls = {"rows": [], "pair": []}
    real_rows, real_pair = rows.union_rerank_rows, fsb_fast_step.forest_beam_pair

    def spy_rows(*a, **k):
        calls["rows"].append((a[0].shape[-1], None if a[1] is None else a[1].shape[-1], k.get("q_bf16"),
                              k.get("ovf_out")))
        return real_rows(*a, **k)

    def spy_pair(*a, **k):
        r = real_pair(*a, **k)
        calls["pair"].append(r is not None)
        return r

    monkeypatch.setattr(rows, "union_rerank_rows", spy_rows)
    monkeypatch.setattr(fsb_fast_step, "forest_beam_pair", spy_pair)
    attn = _attn()
    N = 20000                                              # M = 625 > UCAP: every stage keeps a deferred flag
    Q, K, V, sigma = _inputs(31, 1, 4, N)
    _set_tau(attn, K)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True)
    assert calls["pair"] and all(calls["pair"]) and calls["rows"][0][2] is True
    calls["rows"].clear()
    calls["pair"].clear()
    got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True)
    _assert_same(got, ref)
    d = min(attn.fsb_delta, attn.fsb_delta_max)
    assert calls["pair"] and all(calls["pair"]) and len(calls["rows"]) == len(calls["pair"]), calls
    for ca, cb, qb, ovf in calls["rows"]:
        assert (ca, cb) == (attn.fsb_W_max << d, attn.fsb_union_W_max << d) and qb is True and isinstance(ovf, list)


@gpu
@pytest.mark.parametrize("N", [4096, 4100])
def test_fast_prefill_hands_cell_stats_to_cache(monkeypatch, N):
    # FSAttn._begin_lr asks the long-range branch for its cell statistics (FSL_PREFILL_CELL_REUSE): the fast path hands
    # over what the stage loop does, the same values for this K (nothing for a padded length)
    attn = _attn()
    Q, K, V, sigma = _inputs(8, 1, 4, N)
    _set_tau(attn, K)
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    got = {}
    for fast in ("0", "1"):
        monkeypatch.setenv("FSB_FAST_PREFILL", fast)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            attn._begin_lr(None, want_cell_stats=True)
            try:
                ft.fsb_long_range(attn, Q, K, V, sigma, N)
            finally:
                _, got[fast] = attn._end_lr()
    if N % attn.cell_stride:
        assert got["0"] is None and got["1"] is None
    else:
        assert got["1"] is not None and got["1"][0] is K and got["0"][0] is K
        assert _same(got["1"][1], got["0"][1]) and _same(got["1"][2], got["0"][2])


@gpu
@pytest.mark.parametrize("fast", ["0", "1"])
def test_overflow_redo_keeps_fused_combine(monkeypatch, fast):
    # the fused readout returns W_o's input with the window output combined (FSAttn._begin_lr). An overflow found
    # after that readout is queued (FSB_ROWS_DEFER=1) redoes it: the redo must combine again and equal the single
    # readout of the per-stage check (FSB_ROWS_DEFER=0), and the cache gets one copy of the cell statistics
    attn = _attn(blkchunk=2)
    N = 4096
    Q, K, V, sigma = _inputs(41, 1, 4, N)
    _set_tau(attn, K)
    sr = torch.randn(1, 4, N, 64, generator=torch.Generator().manual_seed(2)).to("cuda", torch.bfloat16)
    monkeypatch.setattr(rows, "UCAP", 40)
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FSL_READOUT_FUSED", "1")
    monkeypatch.setenv("FSB_FAST_PREFILL", fast)
    seen = []
    real = rows.union_rerank_rows

    def spy(*a, **k):
        r = real(*a, **k)
        seen.append(r is None)
        return r

    monkeypatch.setattr(rows, "union_rerank_rows", spy)
    res = {}
    for defer in ("1", "0"):
        monkeypatch.setenv("FSB_ROWS_DEFER", defer)
        n_stats = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            attn._begin_lr(sr, want_cell_stats=True)               # (as the prefill calls it: without autograd)
            try:
                y = ft.fsb_long_range(attn, Q, K, V, sigma, N)
                n_stats = len(attn._cell_stats_out)
            finally:
                fused, st = attn._end_lr()
        res[defer] = (y, fused, st, n_stats)
    assert any(seen), "no stage overflowed"
    for defer, (y, fused, st, n_stats) in res.items():
        assert fused and tuple(y.shape) == (1, N, 4 * 64) and st is not None and n_stats == 1, (defer, y.shape)
    assert _same(res["1"][0], res["0"][0])



@gpu
@torch.no_grad()
def test_overflow_redo_with_in_place_window_output(monkeypatch):
    # with the window output handed over in place (the chunked-RoPE prefill), the first fused readout overwrites it
    # with W_o's input; a deferred overflow redo must combine from the window output itself, not from that sum.
    # A flag forced on the first stage (no row really overflows, so the redo repeats the same selection) must
    # leave the layer output unchanged
    attn = _attn(blkchunk=2)
    N = 16384 + 1024                                      # M = 544 > UCAP: the stages raise deferred flags
    Q, K, V, sigma = _inputs(43, 1, 4, N)
    _set_tau(attn, K)
    assert rows.UCAP >= 512                               # the in-place hand-off's gate (FSAttn._begin_lr)
    for k, v in (("FS_FUSED_ATTN", "1"), ("FSL_READOUT_FUSED", "1"), ("FSB_FAST_PREFILL", "1"),
                 ("FSB_ROWS_DEFER", "1")):
        monkeypatch.setenv(k, v)
    real = rows.union_rerank_rows
    forced = []

    def force(*a, **k):
        r = real(*a, **k)
        flags = k.get("ovf_out")
        if forced and flags is not None and not forced[0]:
            flags.append(torch.ones((), dtype=torch.bool, device="cuda"))
            forced[0] = True
        return r

    monkeypatch.setattr(rows, "union_rerank_rows", force)
    sr0 = torch.randn(1, N, 4, 64, generator=torch.Generator().manual_seed(3)).to("cuda", torch.bfloat16)
    out = {}
    for f in (False, True):
        forced[:] = [False] if f else []
        sr = torch.empty(1, N, 4, 64, device="cuda", dtype=torch.bfloat16)
        sr.copy_(sr0)
        sr = sr.transpose(1, 2)                           # the window kernel's (B, H, N, D) view of (B, N, H, D)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            attn._begin_lr(sr, want_cell_stats=True, inplace_sr=True)
            assert attn._lr_inplace_sr
            try:
                y = ft.fsb_long_range(attn, Q, K, V, sigma, N)
            finally:
                fused, _ = attn._end_lr()
        assert fused
        out[f] = y.clone()
    assert forced[0], "no deferred flag was forced"
    assert _same(out[True], out[False])


@gpu
@torch.no_grad()
@pytest.mark.parametrize("fast", ["0", "1"])
def test_overflow_redo_with_forked_window_output(monkeypatch, fast):
    # FSB_WINDOW_OVERLAP: the window branch writes its output on the side stream while the long-range branch runs, and
    # hands it over in place; the overflow redo's snapshot of that output must wait for the branch's event. The branch
    # is held back (a sleep before its write) until well after the snapshot is queued, and a flag forced on the first
    # stage (the redo repeats the same selection) must leave the layer output unchanged, in the fast path and in the
    # stage loop (which snapshots before its stages)
    attn = _attn(blkchunk=2)
    N = 16384 + 1024                                      # M = 544 > UCAP: the stages raise deferred flags
    Q, K, V, sigma = _inputs(43, 1, 4, N)
    _set_tau(attn, K)
    for k, v in (("FS_FUSED_ATTN", "1"), ("FSL_READOUT_FUSED", "1"), ("FSB_FAST_PREFILL", fast),
                 ("FSB_ROWS_DEFER", "1")):
        monkeypatch.setenv(k, v)
    real = rows.union_rerank_rows
    forced = []

    def force(*a, **k):
        r = real(*a, **k)
        flags = k.get("ovf_out")
        if forced and flags is not None and not forced[0]:
            flags.append(torch.ones((), dtype=torch.bool, device="cuda"))
            forced[0] = True
        return r

    monkeypatch.setattr(rows, "union_rerank_rows", force)
    sr0 = torch.randn(1, N, 4, 64, generator=torch.Generator().manual_seed(3)).to("cuda", torch.bfloat16)
    out = {}
    for f in (False, True):                               # (the first run also compiles the kernels)
        forced[:] = [False] if f else []
        buf = torch.zeros(1, N, 4, 64, device="cuda", dtype=torch.bfloat16)

        def window():
            torch.cuda._sleep(1 << 30)                    # about 0.7 s on an A100
            buf.copy_(sr0)
            return buf.transpose(1, 2)                    # the window kernel's (B, H, N, D) view of (B, N, H, D)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            sr, ev = FSAttn._window_fork(window, Q.device)
            attn._begin_lr(sr, want_cell_stats=True, sr_event=ev, inplace_sr=True)
            assert attn._lr_inplace_sr
            try:
                y = ft.fsb_long_range(attn, Q, K, V, sigma, N)
            finally:
                fused, _ = attn._end_lr()
        assert fused
        out[f] = y.clone()
    assert forced[0], "no deferred flag was forced"
    assert _same(out[True], out[False])


@gpu
def test_log_stats_after_lean_eval_prefill(monkeypatch):
    # fsb_log_stats after an eval prefill on the fast path: without FSB_PREFILL_STATS the per-query statistics are not
    # kept (NaN in the log) and the others equal the stage loop's; with it, every value does
    import math
    attn = _attn()
    model = torch.nn.ModuleList([attn])
    Q, K, V, sigma = _inputs(19, 1, 4, 3000)
    _set_tau(attn, K, tau_q=0.2)
    logs = {}
    for fast, st in ((False, False), (True, False), (True, True)):
        _run(monkeypatch, attn, Q, K, V, sigma, fast=fast, stats=st)
        logs[(fast, st)] = ft.fsb_log_stats(model)
    ref, lean, full = logs[(False, False)], logs[(True, False)], logs[(True, True)]
    assert set(ref) == set(lean) == set(full)
    for k, v in ref.items():
        assert (math.isnan(v) and math.isnan(full[k])) or v == full[k], k
        if k in ("n_read", "n_visit", "cost"):
            assert math.isnan(lean[k]) and not math.isnan(v), k
        else:
            assert (math.isnan(v) and math.isnan(lean[k])) or v == lean[k], k


@gpu
def test_strided_sigma_matches_reference(monkeypatch):
    # per-head sigma and sel_scale views with a non-unit stride (a column of a wider tensor): the row tables must read
    # them by element
    attn = _attn()
    Q, K, V, sigma = _inputs(61, 1, 4, 3000)
    _set_tau(attn, K)
    sig2 = torch.stack([sigma, 2 * sigma], 1)[:, 0]
    assert not sig2.is_contiguous()
    ref = _run(monkeypatch, attn, Q, K, V, sig2, fast=False, acc=True)
    got = _run(monkeypatch, attn, Q, K, V, sig2, fast=True, acc=True)
    _assert_same(got, ref)
    head = (torch.arange(64) % 4).cuda()
    SP = {"height": torch.randint(0, 12, (64,), generator=torch.Generator().manual_seed(1)).cuda()}
    sel = torch.stack([attn.sel_scale.detach(), attn.sel_scale.detach() * 3], 1)[:, 0]
    a = fp._row_tables(attn, SP, None, sel, sig2, head, 128, True)
    b = fp._row_tables(attn, SP, None, sel.contiguous(), sig2.contiguous(), head, 128, True)
    assert all(_same(x, y) for x, y in zip(a, b))


@gpu
@torch.no_grad()
def test_negative_delta_keeps_the_stage_loop():
    # budgets 2^(negative): the fast path leaves them to the stage loop
    attn = _attn(fsb_delta=-1)
    Q, K, V, sigma = _inputs(4, 1, 4, 1000)
    assert not fp.prefill_fast_ok(attn, Q, K.detach(), 128)
    assert fp.prefill_fast_ok(_attn(), Q, K.detach(), 128)


def test_prefill_blkchunk_is_a_cap_only_when_given(monkeypatch):
    # fsb_prefill_blkchunk (eval_ruler / train_dist --fsb_prefill_blkchunk, FSB_PREFILL_BLKCHUNK) caps the fast path's
    # stage size when it is given; unset, the stage loop keeps 32 blocks and the fast path FSB_FAST_PREFILL_BLKCHUNK
    monkeypatch.delenv("FSB_PREFILL_BLKCHUNK", raising=False)

    def mk(**k):
        return FSAttn(d_model=256, n_heads=4, causal=True, lr_scale_mode="one", **dict(SERVE, **k))

    a, b, c = mk(), mk(fsb_prefill_blkchunk=5), mk(fsb_prefill_blkchunk=0)
    assert (a.fsb_prefill_blkchunk, a.fsb_prefill_blkchunk_set, fp._blkchunk_cap(a)) == (32, False, 0)
    assert (b.fsb_prefill_blkchunk, b.fsb_prefill_blkchunk_set, fp._blkchunk_cap(b)) == (5, True, 5)
    assert fp._blkchunk_cap(c) == 0                                   # <= 0: all blocks at once, no cap
    monkeypatch.setenv("FSB_PREFILL_BLKCHUNK", "7")
    d = mk()
    assert (d.fsb_prefill_blkchunk, d.fsb_prefill_blkchunk_set, fp._blkchunk_cap(d)) == (7, True, 7)
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "128")
    monkeypatch.setenv("FSB_FAST_PREFILL_SCRATCH_MB", "2048")
    assert fp._stage_blocks(4, 47, 128, 64, 188, 16, 16, 4) == 47
    assert fp._stage_blocks(4, 47, 128, 64, 188, 16, 16, 4, cap=5) == 5
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "3")
    assert fp._stage_blocks(4, 47, 128, 64, 188, 16, 16, 4, cap=5) == 3


@gpu
def test_prefill_blkchunk_bounds_fast_stages(monkeypatch):
    # the cap's stages (same cells as the stage loop) and the memory it bounds: at 16384 tokens the 128-block stage's
    # rows scratch (the LSE table: 512 cells x 128 queries per row) dominates the fast path's peak
    sizes = []
    real = fp._stage_blocks

    def spy(*a, **k):
        n = real(*a, **k)
        sizes.append(n)
        return n

    monkeypatch.setattr(fp, "_stage_blocks", spy)
    N = 6000                                                            # 47 query blocks
    Q, K, V, sigma = _inputs(29, 1, 4, N)
    for cap, want in ((False, 47), (True, 5)):
        attn = _attn(blkchunk=5, cap=cap)
        _set_tau(attn, K)
        ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True)
        sizes.clear()
        got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True)
        assert sizes == [want], (cap, sizes)
        _assert_same(got, ref)
    N = 16384
    Q, K, V, sigma = _inputs(30, 1, 4, N)
    peak = {}
    for cap in (False, True):
        attn = _attn(blkchunk=4, cap=cap)
        _set_tau(attn, K)
        attn._fsb_read_acc = None
        monkeypatch.setenv("FSB_FAST_PREFILL", "1")
        monkeypatch.setenv("FS_FUSED_ATTN", "1")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ft.fsb_long_range(attn, Q, K, V, sigma, N)                  # compile and cache
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ft.fsb_long_range(attn, Q, K, V, sigma, N)
            torch.cuda.synchronize()
        peak[cap] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
    assert peak[True] < peak[False] - 64, peak


def test_bench_rows_record_fsb_switches_and_latches(monkeypatch):
    # bench_efficiency rows of FS-B record the resolved FS-B switches and the fallback latches
    import bench_efficiency as be
    from src.fsattn import fsb_beam_cuda, fsb_build_cuda, fsb_rows_cuda
    for name in be.FSB_FLAGS:
        monkeypatch.delenv(name, raising=False)
    assert be.fsb_flags() == be.FSB_FLAGS
    monkeypatch.setenv("FSB_FAST_PREFILL", "0")
    assert be.fsb_flags() == dict(be.FSB_FLAGS, FSB_FAST_PREFILL="0")
    monkeypatch.setattr(fsb_beam_cuda, "_failed", [])
    monkeypatch.setattr(fsb_beam_cuda, "_rows_failed", ["OSError('simulated')"])
    monkeypatch.setattr(fsb_beam_cuda, "_rows2_failed", [])
    monkeypatch.setattr(fsb_rows_cuda, "_failed", [])
    monkeypatch.setattr(fsb_rows_cuda, "_v3_failed", ["RuntimeError('simulated')"])
    monkeypatch.setattr(fsb_build_cuda, "_small_failed", {(0, 64, ()): "RuntimeError('small')"})
    monkeypatch.setattr(fp, "_warned", [1])
    assert be.fsb_fast_status() == {"beam_cuda_failure": None, "beam_rows_failure": "OSError('simulated')",
                                    "beam_rows2_failure": None, "rows_union_failure": None,
                                    "rows_v3_failure": "RuntimeError('simulated')",
                                    "build_small_failure": "RuntimeError('small')", "fast_prefill_unfused": True}


def test_scratch_budget_counts_the_layer_tables(monkeypatch):
    # FSB_FAST_PREFILL_SCRATCH_MB bounds the layer's tables plus one stage's scratch, at least one block per stage
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "128")
    monkeypatch.setenv("FSB_FAST_PREFILL_SCRATCH_MB", "2048")
    T, nb, Bq, Dh, M, K = 64, 256, 128, 64, 1024, 4
    per = fp._stage_bytes(T, Bq, Dh, M, 16, 16, K)
    lay = fp._layer_bytes(T, nb, Bq, Dh, M, K, True, True, True, False)
    n0 = fp._stage_blocks(T, nb, Bq, Dh, M, 16, 16, K)
    n1 = fp._stage_blocks(T, nb, Bq, Dh, M, 16, 16, K, lay)
    assert n1 == (2048 * 2 ** 20 - lay) // per and n1 < n0 and lay + n1 * per <= 2048 * 2 ** 20
    monkeypatch.setenv("FSB_FAST_PREFILL_SCRATCH_MB", "1")
    assert fp._stage_blocks(T, nb, Bq, Dh, M, 16, 16, K, lay) == 1


@gpu
def test_one_block_stages_match_reference(monkeypatch):
    # a scratch budget smaller than the layer's tables: one block per stage, same cells
    monkeypatch.setenv("FSB_FAST_PREFILL_SCRATCH_MB", "1")
    attn = _attn()
    Q, K, V, sigma = _inputs(71, 1, 4, 2000)
    _set_tau(attn, K)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True)
    got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True)
    _assert_same(got, ref)


@gpu
def test_check_compact_runs_on_the_fast_path(monkeypatch):
    # FSB_CHECK_COMPACT=1 checks both beams' compact output against the full layout on this path too
    n = [0]
    real = ft._check_compact

    def spy(*a, **k):
        n[0] += 1
        return real(*a, **k)

    monkeypatch.setattr(ft, "_check_compact", spy)
    monkeypatch.setenv("FSB_CHECK_COMPACT", "1")
    attn = _attn()
    Q, K, V, sigma = _inputs(13, 1, 4, 3000)
    _set_tau(attn, K)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True)
    n_ref, n[0] = n[0], 0
    got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True)
    _assert_same(got, ref)
    assert n_ref > 0 and n[0] == 2, (n_ref, n[0])           # one stage here, two beams


@gpu
@pytest.mark.parametrize("serve", ["dense", "sparse", "fsb"])
def test_graphed_prefiller_matches_eager(monkeypatch, serve):
    import bench_efficiency as be
    from decode_graph import GraphedPrefiller
    from experiments.scaling.model import FreeSoloLM
    from kv_cache import _trunk_cached, make_caches
    from longctx_infer import _lm_head_fp32
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "1" if serve == "sparse" else "0")
    monkeypatch.setenv("FS_FUSED_SELECT", "1" if serve == "sparse" else "0")
    kw = be.serve_kwargs(serve, 0.5 if serve == "fsb" else 0.0)
    torch.manual_seed(11)
    L = 1024
    m = FreeSoloLM(vocab_size=512, d_model=256, n_layers=2, n_heads=4, d_ff=512, max_seq_len=L + 64,
                   attn_type="fsattn", **kw).cuda().to(torch.bfloat16).eval()
    g = torch.Generator().manual_seed(21)
    x1 = torch.randint(0, 512, (1, L), generator=g).cuda()
    x2 = torch.randint(0, 512, (1, L), generator=g).cuda()
    if serve == "fsb":
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cs = make_caches(m)
            _trunk_cached(m, x1, cs)
            for blk, c in zip(m.blocks, cs):
                blk.attn.fsattn.fsb_tau.fill_(0.0)
                blk.attn.fsattn.fsb_tau_union.copy_(c._ceil[0].float().median(dim=-1).values)
    gp = GraphedPrefiller(m, L, B=1, device="cuda", dtype=torch.bfloat16, tail_tokens=64)
    gp.capture(x1)
    for x in (x1, x2):
        got_last = gp.run(x).clone()
        got_tail = gp.tail_logits().clone()
        cs = make_caches(m)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = _trunk_cached(m, x, cs)
            ref_last = _lm_head_fp32(m, h[:, -1:, :], 4096)[:, 0, :]
            ref_tail = _lm_head_fp32(m, h[:, -64:, :], 4096)
        assert _same(got_last, ref_last), serve
        assert _same(got_tail, ref_tail), serve
        for cg, ce in zip(gp.cache, cs):
            assert _same(cg._k_hist, ce._k_hist) and _same(cg._v_hist, ce._v_hist)
            assert _same(cg._kbar, ce._kbar) and _same(cg._ceil, ce._ceil)


@gpu
@pytest.mark.parametrize("pipe", ["0", "1"])
def test_window_overlap_matches_serial(monkeypatch, pipe):
    # FSB_WINDOW_OVERLAP=1 forks every layer's window branch onto a side stream next to the long-range branch
    # (FSAttn._window_fork), with the stage pipe's own side stream on top when a layer has several stages: the eager
    # prefill and the CUDA-graph prefill (per-layer KV and the bench's shared K / V buffers, which the next layer
    # overwrites) equal the serial path's bit for bit, and no record_stream is issued (under capture it would keep each
    # layer's window output out of the graph pool)
    import bench_efficiency as be
    from decode_graph import GraphedPrefiller
    from experiments.scaling.model import FreeSoloLM
    from kv_cache import _trunk_cached, make_caches
    from longctx_infer import _lm_head_fp32, extend_rope_cache
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "0")
    monkeypatch.setenv("FSB_STAGE_PIPE", pipe)
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "2")      # 8 query blocks of 128: four stages per layer
    kw = be.serve_kwargs("fsb", 0.5)
    torch.manual_seed(13)
    L, n_layers = 1024, 3
    m = FreeSoloLM(vocab_size=512, d_model=256, n_layers=n_layers, n_heads=4, d_ff=512, max_seq_len=L + 64,
                   attn_type="fsattn", **kw).cuda().to(torch.bfloat16).eval()
    extend_rope_cache(m, L + 256)                             # the prefiller's rope tables, for the eager runs too
    g = torch.Generator().manual_seed(23)
    x1 = torch.randint(0, 512, (1, L), generator=g).cuda()
    x2 = torch.randint(0, 512, (1, L), generator=g).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        cs = make_caches(m)
        _trunk_cached(m, x1, cs)
        for blk, c in zip(m.blocks, cs):
            blk.attn.fsattn.fsb_tau.fill_(0.0)
            blk.attn.fsattn.fsb_tau_union.copy_(c._ceil[0].float().median(dim=-1).values)
    forks, rs = [0], [0]
    real_fork, real_rs = FSAttn._window_fork, torch.Tensor.record_stream

    def spy_fork(fn, dev):
        forks[0] += 1
        return real_fork(fn, dev)

    def spy_rs(t, s):
        # only the window branch's calls (FSAttn's hand-off) count: the forest build's task tables record every new
        # stream that reads them (fsb_build_cuda._tasks), as the first capture of a shape in a process does on either
        # path
        if sys._getframe(1).f_globals.get("__name__", "").endswith(".attention"):
            rs[0] += 1
        return real_rs(t, s)

    monkeypatch.setattr(FSAttn, "_window_fork", staticmethod(spy_fork))
    monkeypatch.setattr(torch.Tensor, "record_stream", spy_rs)
    res = {}
    for ov in ("0", "1"):
        monkeypatch.setenv("FSB_WINDOW_OVERLAP", ov)
        forks[0] = 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = _trunk_cached(m, x2, make_caches(m))
            out = {"eager_h": h, "eager_tail": _lm_head_fp32(m, h[:, -64:, :], 4096)}
        for retain in (True, False):
            gp = GraphedPrefiller(m, L, B=1, device="cuda", dtype=torch.bfloat16, tail_tokens=64, retain_kv=retain)
            gp.capture(x1)
            for i, x in enumerate((x1, x2)):
                out[f"last{i}_{retain}"] = gp.run(x).clone()
                out[f"tail{i}_{retain}"] = gp.tail_logits().clone()
                if retain:
                    for j, c in enumerate(gp.cache):
                        for name in ("_k_hist", "_v_hist", "_kbar", "_ceil"):
                            out[f"{name}{i}_{j}"] = getattr(c, name).clone()
            del gp
        # one fork per layer and forward: the eager prefill, then each prefiller's warm-up pass and its capture
        assert forks[0] == (5 * n_layers if ov == "1" else 0), (ov, forks[0])
        res[ov] = out
    assert rs[0] == 0, rs[0]
    assert res["0"].keys() == res["1"].keys()
    for k in res["0"]:
        assert _same(res["0"][k], res["1"][k]), k
    # the replays agree with the eager prefill of the same prompt
    assert torch.isfinite(res["1"]["eager_tail"]).all()
    for retain in (True, False):
        assert _same(res["1"][f"tail1_{retain}"], res["1"]["eager_tail"]), retain


@gpu
def test_fsb_graphed_decoder_single_capture_across_blocks(monkeypatch):
    import bench_efficiency as be
    from decode_graph import FSBGraphedDecoder
    from experiments.scaling.model import FreeSoloLM
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    kw = be.serve_kwargs("fsb", 0.5)
    torch.manual_seed(19)
    L = 1024
    m = FreeSoloLM(vocab_size=512, d_model=256, n_layers=2, n_heads=4, d_ff=512, max_seq_len=L + 512,
                   attn_type="fsattn", **kw).cuda().to(torch.bfloat16).eval()
    prompt = torch.randint(0, 512, (1, L), generator=torch.Generator().manual_seed(29)).cuda()
    gd = FSBGraphedDecoder(m, prompt, 280, device="cuda", dtype=torch.bfloat16)
    gd.prefill()
    for blk, c in zip(m.blocks, gd.cache):
        blk.attn.fsattn.fsb_tau.fill_(0.0)
        blk.attn.fsattn.fsb_tau_union.copy_(c._ceil[0].float().median(dim=-1).values)
        blk.attn.fsattn.fsb_tau_union[0] = c._ceil[0, 0].float().kthvalue(2).values
    captures = [0]
    real_cap = gd._capture

    def spy_cap():
        captures[0] += 1
        return real_cap()

    gd._capture = spy_cap
    for _ in range(260):
        gd.step()
    assert captures[0] == 1, captures[0]


@gpu
@pytest.mark.parametrize("N,tau_q,case", [(4096, None, "utau"), (3000, 0.3, "utau"), (20000, None, "utau"),
                                          (1100, 0.25, "no_union"), (2500, None, "union_no_tau"),
                                          (700, None, "leaf_radius")])
def test_prefill_trim_same_selection(monkeypatch, N, tau_q, case):
    # FSB_BUILD_PREFILL_TRIM=1 builds only the forest chunks the layer's spines read: output, cells, statistics and
    # read accumulators bit for bit as with the full build, over several stages (and the stage pipe)
    from src.fsattn import fsb_build_cuda as bc
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "8")
    seen = []
    real = bc._tasks
    monkeypatch.setattr(bc, "_tasks", lambda *a, **k: seen.append(k.get("mb_max")) or real(*a, **k))
    attn = _attn(**CASES[case])
    Q, K, V, sigma = _inputs(90 + N, 1, 4, N)
    _set_tau(attn, K, tau_q=tau_q)
    out = {}
    for trim in ("0", "1"):
        monkeypatch.setenv("FSB_BUILD_PREFILL_TRIM", trim)
        seen.clear()
        out[trim] = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True, stats=True)
        nb = -(-N // 128)
        assert seen == [((nb - 1) * 4 if trim == "1" else None)], seen    # cell_stride 32, 128-query blocks
    _assert_same(out["1"], out["0"])


@gpu
def test_mb_max_is_the_device_bound():
    # the host's _mb_max(nb, Bq, s, M) equals min(E[nb - 1], M) for the device's E = floor(blk_start / s) of
    # _consts and _forest_build_stage, for every nb up to 3000 and some past fp32's exact integers, also where the fp32
    # reciprocal of s rounds (s not a power of two)
    from src.fsattn.fsb_prefill import _mb_max
    for Bq in (1, 3, 32, 100, 128, 129):
        for s in (1, 3, 7, 10, 24, 32, 33, 100, 127):
            E = (torch.arange(3000, device="cuda", dtype=torch.float32) * Bq / s).floor().long().tolist()
            assert [_mb_max(nb, Bq, s, 1 << 40) for nb in range(1, 3001)] == E, (Bq, s)
            assert _mb_max(3000, Bq, s, 5) == min(E[-1], 5)
    for nb in ((1 << 24) + 3, 40000017):
        for Bq, s in ((1, 3), (3, 7), (128, 24)):
            e = int((torch.arange(nb, device="cuda", dtype=torch.float32) * Bq / s).floor()[-1])
            assert _mb_max(nb, Bq, s, 1 << 40) == e, (nb, Bq, s)


PREFILL_SWITCHES = ("FSB_BUILD_PREFILL_TRIM", "FSB_PREFILL_NOCOPY", "FSB_WINDOW_OVERLAP", "FSB_ROWS_V3", "FSB_BEAM_V2",
                    "FSB_BEAM_V2_EXACT")


@gpu
@pytest.mark.parametrize("switches,H", [(PREFILL_SWITCHES[:2], 4), (PREFILL_SWITCHES, 16)], ids=["trim_nocopy", "all"])
def test_graphed_prefill_switches_match_eager(monkeypatch, switches, H):
    # the trimmed build and the in-place stage views inside a captured prefill (several stages on the stage pipe), and
    # with 16 heads (the fused re-rank's shape) every exact prefill switch at once: replays equal the eager prefill
    # with the switches off (the full build, the copies, the serial window, fsb_rows_fused2, fsb_beam_rows) bit for bit
    import bench_efficiency as be
    from decode_graph import GraphedPrefiller
    from experiments.scaling.model import FreeSoloLM
    from kv_cache import _trunk_cached, make_caches
    from longctx_infer import _lm_head_fp32
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "0")
    monkeypatch.setenv("FS_FUSED_SELECT", "0")
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "4")
    kw = be.serve_kwargs("fsb", 0.5)
    torch.manual_seed(13)
    L = 3000
    m = FreeSoloLM(vocab_size=512, d_model=64 * H, n_layers=2, n_heads=H, d_ff=512, max_seq_len=L + 64,
                   attn_type="fsattn", **kw).cuda().to(torch.bfloat16).eval()
    g = torch.Generator().manual_seed(27)
    x1 = torch.randint(0, 512, (1, L), generator=g).cuda()
    x2 = torch.randint(0, 512, (1, L), generator=g).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        cs = make_caches(m)
        _trunk_cached(m, x1, cs)
        for blk, c in zip(m.blocks, cs):
            blk.attn.fsattn.fsb_tau.fill_(0.0)
            blk.attn.fsattn.fsb_tau_union.copy_(c._ceil[0].float().median(dim=-1).values)
    for name in PREFILL_SWITCHES:
        monkeypatch.setenv(name, "1" if name in switches else "0")
    gp = GraphedPrefiller(m, L, B=1, device="cuda", dtype=torch.bfloat16, tail_tokens=64)
    gp.capture(x1)                                         # its eager warm-up builds the trimmed task tables
    for name in PREFILL_SWITCHES:
        monkeypatch.setenv(name, "0")
    for x in (x1, x2):
        gp.run(x)
        got_tail = gp.tail_logits().clone()
        cs = make_caches(m)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = _trunk_cached(m, x, cs)
            ref_tail = _lm_head_fp32(m, h[:, -64:, :], 4096)
        assert _same(got_tail, ref_tail)


def _copies(fn) -> int:
    # direct-copy kernels (tensor copies) of one call
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events()
               if e.device_type == torch.autograd.DeviceType.CUDA and "direct_copy" in e.name)


@gpu
@pytest.mark.parametrize("N,B,H,blk,contig", [(4096, 1, 8, "8", False), (3000, 1, 8, "8", False),
                                              (3000, 1, 8, "3", True), (2500, 2, 8, "4", False),
                                              (2500, 2, 8, "4", True), (700, 1, 8, "0", False),
                                              (20000, 1, 8, "0", False), (3000, 1, 4, "8", False)])
def test_prefill_nocopy_matches_reference(monkeypatch, N, B, H, blk, contig):
    # FSB_PREFILL_NOCOPY=1: the beams and the re-rank read the stage queries from Q in place and the re-rank writes the
    # cell table directly; output, cells, statistics and read accumulators bit for bit as the copies and the stage
    # loop, for Q's transposed layout and a contiguous Q (both flags), batched rows, partial last stages (H = 8: the
    # fused re-rank kernel; H = 4: its Triton path, which packs the views)
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", blk)
    attn = _attn(H=H)
    Q, K, V, sigma = _inputs(60 + N, B, H, N)
    if contig:
        Q = Q.contiguous()
    _set_tau(attn, K, tau_q=0.2 if N == 3000 else None)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True, stats=True)
    for nocopy in ("0", "1"):
        monkeypatch.setenv("FSB_PREFILL_NOCOPY", nocopy)
        got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True, stats=True)
        _assert_same(got, ref)


@gpu
def test_prefill_nocopy_removes_the_stage_copies(monkeypatch):
    # with Q in the (B, N, H, D)-transposed layout every stage copies its queries and, with two or more stages, its
    # int64 cells into the int32 table; FSB_PREFILL_NOCOPY=1 drops both
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "8")
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    attn = _attn(H=8)
    N = 4096                                                     # 32 blocks: 4 stages
    Q, K, V, sigma = _inputs(5, 1, 8, N)
    _set_tau(attn, K)
    attn._fsb_read_acc = None

    def one():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ft.fsb_long_range(attn, Q, K, V, sigma, N)

    n = {}
    for nocopy in ("0", "1"):
        monkeypatch.setenv("FSB_PREFILL_NOCOPY", nocopy)
        one()
        n[nocopy] = _copies(one)
    assert n["0"] - n["1"] == 2 * 4, n


@gpu
@pytest.mark.parametrize("which", ["beam", "rows", "both"])
def test_prefill_nocopy_fallbacks(monkeypatch, which):
    # where the in-place kernels do not run, the stage takes the copies (the beams) or the re-rank packs the views
    # itself (its Triton path): same values
    from src.fsattn import fsb_fast_step
    from src.fsattn import fsb_rows_cuda
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "3")
    attn = _attn(H=8)
    Q, K, V, sigma = _inputs(8, 1, 8, 2900)
    _set_tau(attn, K)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True)
    if which in ("beam", "both"):
        real = fsb_fast_step.forest_beam_pair

        def pair(F, SP, W, d, Wm, Fu, SPu, Wu, du, Wum, q, *a, **k):    # views: as if the pair kernel did not run
            return None if q.shape[0] != W.shape[0] else real(F, SP, W, d, Wm, Fu, SPu, Wu, du, Wum, q, *a, **k)

        monkeypatch.setattr(fsb_fast_step, "forest_beam_pair", pair)
    if which in ("rows", "both"):
        monkeypatch.setattr(fsb_rows_cuda, "rows_fused_ok", lambda *a, **k: False)
    monkeypatch.setenv("FSB_PREFILL_NOCOPY", "1")
    got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True)
    _assert_same(got, ref)


@gpu
def test_prefill_nocopy_overflow_redo(monkeypatch):
    # rows over a small re-rank capacity: the in-place stages' cells of the flagged stages are redone by the stage
    # loop's stages exactly as with the copies (deferred flags, trimmed build)
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "3")
    monkeypatch.setenv("FSB_BUILD_PREFILL_TRIM", "1")
    N = 4096
    attn = _attn(H=8, blkchunk=2)
    Q, K, V, sigma = _inputs(41, 1, 8, N)
    _set_tau(attn, K)
    seen = []
    real = rows.union_rerank_rows

    def spy(*a, **k):
        r = real(*a, **k)
        seen.append(r is None)
        return r

    monkeypatch.setattr(rows, "union_rerank_rows", spy)
    monkeypatch.setattr(rows, "UCAP", 40)
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True, stats=True)
    assert any(seen) and not all(seen), seen
    monkeypatch.setenv("FSB_PREFILL_NOCOPY", "1")
    got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True, stats=True)
    _assert_same(got, ref)


def _poison_trimmed(monkeypatch):
    # a trimmed build (mb_max) leaves the chunks no spine reads unwritten: fill their node rows with NaNs and their
    # leaves with -7 after each such build, so a kernel that reads one of them changes its result
    from src.fsattn import fsb_build_cuda as bc
    real = bc.forests_cuda

    def poisoned(*a, mb_max=None, **k):
        got = real(*a, mb_max=mb_max, **k)
        if got is not None and mb_max is not None:
            for F, _ in got:
                nodes, leaves = [], []
                for kk in F["ks"]:
                    ja, jsh, nj = bc.spine_chunks(F["M"], kk, mb_max)
                    keep = set(range(ja, ja + (nj << jsh), 1 << jsh))
                    npr, nb0, lb0 = F["nodes_per"][kk], F["bases"][kk], F["leafbases"][kk]
                    for j in range(F["Cmaxs"][kk]):
                        if j not in keep:
                            nodes.append(torch.arange(nb0 + j * npr, nb0 + (j + 1) * npr))
                            leaves.append(torch.arange(lb0 + (j << kk), lb0 + ((j + 1) << kk)))
                if nodes:
                    ni = torch.cat(nodes).cuda()
                    F["node_mu"].index_fill_(1, ni, float("nan"))
                    F["node_aux"].index_fill_(1, ni, float("nan"))
                    F["leaf"].index_fill_(1, torch.cat(leaves).cuda(), -7)
        return got

    monkeypatch.setattr(bc, "forests_cuda", poisoned)


@gpu
@pytest.mark.parametrize("N,blk", [(4096, "0"), (3000, "8"), (9000, "16"), (17000, "32")])
def test_prefill_switches_at_released_shape(monkeypatch, N, blk):
    # the released shape (16 heads of 64, top 4 cells): every combination of FSB_BUILD_PREFILL_TRIM,
    # FSB_PREFILL_NOCOPY, FSB_ROWS_V3 and FSB_BEAM_V2 (its exact build) gives the stage loop's output, cells,
    # statistics and read accumulators bit for bit, over one stage or several with a partial last one (the stage pipe;
    # 17000 tokens: more cells than the re-rank capacity, deferred flags), and each switch runs its kernels. The
    # trimmed builds' unwritten chunks are poisoned, so none of the kernels may read them
    from src.fsattn import fsb_beam_cuda as bc
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", blk)
    monkeypatch.setenv("FSB_BEAM_V2_EXACT", "1")
    seen = []
    real_launch = bc.launch
    monkeypatch.setattr(bc, "launch", lambda fn, *a, **k: seen.append(fn.value) or real_launch(fn, *a, **k))
    _poison_trimmed(monkeypatch)
    attn = _attn(H=16)
    Q, K, V, sigma = _inputs(70 + N, 1, 16, N)
    _set_tau(attn, K)
    flags = ("FSB_BUILD_PREFILL_TRIM", "FSB_PREFILL_NOCOPY", "FSB_ROWS_V3", "FSB_BEAM_V2")
    for f in flags:
        monkeypatch.setenv(f, "0")
    ref = _run(monkeypatch, attn, Q, K, V, sigma, fast=False, acc=True, stats=True)
    for bits in range(1 << len(flags)):
        on = {f: (bits >> i) & 1 for i, f in enumerate(flags)}
        for f, v in on.items():
            monkeypatch.setenv(f, str(v))
        seen.clear()
        got = _run(monkeypatch, attn, Q, K, V, sigma, fast=True, acc=True, stats=True)
        _assert_same(got, ref)
        by_fn = {f.value: key[2] for key, (f, _, _) in bc._funcs.items()}
        ran = {by_fn.get(v) for v in seen}
        assert ("fsb_beam_rows2" in ran) == ("fsb_spines2" in ran) == bool(on["FSB_BEAM_V2"]), (on, ran)
        assert ("fsb_beam_rows" in ran) != bool(on["FSB_BEAM_V2"]), (on, ran)
        assert ("fsb_rows_v3" in ran) == bool(on["FSB_ROWS_V3"]) != ("fsb_rows_fused2" in ran), (on, ran)
