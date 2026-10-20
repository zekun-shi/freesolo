import math
import warnings
from pathlib import Path

import pytest
import torch

from src.fsattn.fsb_tree import ForestCache, build_forest, build_spines, traverse_forest


def make_cache(monkeypatch, rows, dim, lean, fast):
    monkeypatch.setenv("FSB_FAST_CACHE", "1" if fast else "0")
    return ForestCache(rows, dim, "cpu", lean=lean)


def selections(forest, q, pos):
    rows, queries, dim = q.shape
    dev = q.device
    sp = build_spines(forest, forest["m_tot"], torch.arange(rows, device=dev))
    delta = torch.minimum(sp["height"][:, None], torch.full((rows, queries), 2, device=dev))
    width = torch.full_like(delta, 4)
    return traverse_forest(
        forest, sp, q, torch.full((rows, queries), float(pos), device=dev),
        torch.ones(rows, 1, 1, device=dev), torch.full((rows, 1, 1), 4.0, device=dev),
        1 / math.sqrt(dim), 16, width, delta, 4, 2, radius_coef=8.0,
    )


@pytest.mark.parametrize("lean", [False, True])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("spread", [False, True])
def test_streamed_cache_matches_reference(monkeypatch, lean, ties, spread):
    torch.manual_seed(126)
    rows, cells, dim = 4, 69, 8
    keys = torch.randn(1, rows, cells, dim)
    if ties:
        keys.zero_()
    ceilings = torch.rand(1, rows, cells)
    tau = torch.tensor([float("inf"), 0.0, 0.4, 0.7])
    ceilings[:, 2:, :9] = 0
    spread_all = torch.rand(1, rows, cells) if spread else None
    ref = make_cache(monkeypatch, rows, dim, lean, False)
    fast = make_cache(monkeypatch, rows, dim, lean, True)
    q = torch.randn(rows, 3, dim)
    for visible in (1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 63, 64, 69):
        for cache in (ref, fast):
            cache.ingest(keys, ceilings, visible, tau, spread_all=spread_all, keep_cells=True)
        assert torch.equal(ref.m_tot, fast.m_tot)
        assert torch.equal(ref.adm, fast.adm[:, :ref.adm.shape[1]])
        assert (fast.adm[:, visible:] == -1).all()
        assert fast.adm.is_contiguous()
        assert torch.equal(ref.F["cellK"], fast.F["cellK"])
        assert torch.equal(ref.F["cellA"], fast.F["cellA"])
        expected = selections(ref.F, q, visible * 16)
        actual = selections(fast.F, q, visible * 16)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual)), visible
        rebuilt = build_forest(
            keys.reshape(rows, cells, dim)[:, :visible],
            ceilings.reshape(rows, cells)[:, :visible],
            ceilings.reshape(rows, cells)[:, :visible] >= tau[:, None],
            spread=spread_all.reshape(rows, cells)[:, :visible] if spread else None,
        )
        batch = selections(rebuilt, q, visible * 16)
        assert all(torch.equal(a, b) for a, b in zip(batch, actual)), visible


def test_admission_capacity_and_order(monkeypatch):
    rows, cells = 3, 65
    ceilings = torch.ones(1, rows, cells)
    ceilings[:, 1, ::2] = 0
    tau = torch.tensor([0.5, 0.5, float("inf")])
    fast = make_cache(monkeypatch, rows, 8, True, True)
    allocations = 0
    for visible in range(1, cells + 1):
        old = fast.adm
        fast._append_admitted(ceilings, visible, tau)
        allocations += fast.adm is not old
        assert visible <= fast.adm.shape[1] < 2 * visible
        for row in range(rows):
            expected = torch.where(ceilings[0, row, :visible] >= tau[row])[0]
            count = expected.numel()
            assert fast.m_tot[row] == count
            assert torch.equal(fast.adm[row, :count], expected)
            assert (fast.adm[row, count:] == -1).all()
    assert allocations == (cells - 1).bit_length() + 1


def test_lean_arena_reuses_unchanged_chunks(monkeypatch):
    torch.manual_seed(127)
    rows, cells, dim = 2, 17, 8
    keys = torch.randn(1, rows, cells, dim)
    ceilings = torch.ones(1, rows, cells)
    tau = torch.zeros(rows)
    fast = make_cache(monkeypatch, rows, dim, True, True)
    fast.ingest(keys, ceilings, 12, tau)
    old_mu, old_aux, old_leaf = fast.F["node_mu"], fast.F["node_aux"], fast.F["leaf"]
    base, leafbase = fast.F["bases"][3], fast.F["leafbases"][3]
    unchanged_mu = old_mu[:, base:base + 15].clone()
    unchanged_aux = old_aux[:, base:base + 15].clone()
    unchanged_leaf = old_leaf[:, leafbase:leafbase + 8].clone()
    fast.ingest(keys, ceilings, 13, tau)
    assert fast.F["node_mu"] is old_mu
    assert fast.F["node_aux"] is old_aux
    assert fast.F["leaf"] is old_leaf
    assert fast.F["bases"][3] == base
    assert fast.F["leafbases"][3] == leafbase
    assert torch.equal(old_mu[:, base:base + 15], unchanged_mu)
    assert torch.equal(old_aux[:, base:base + 15], unchanged_aux)
    assert torch.equal(old_leaf[:, leafbase:leafbase + 8], unchanged_leaf)
    fast.ingest(keys, ceilings, 17, tau)
    assert fast.F["node_mu"] is not old_mu
    assert fast.F["node_aux"] is not old_aux
    assert fast.F["leaf"] is not old_leaf
    assert torch.equal(fast.F["node_mu"][:, base:base + 15], unchanged_mu)
    assert torch.equal(fast.F["node_aux"][:, base:base + 15], unchanged_aux)
    assert torch.equal(fast.F["leaf"][:, leafbase:leafbase + 8], unchanged_leaf)


@pytest.mark.parametrize("lean", [False, True])
def test_no_admissions_and_no_new_cells(monkeypatch, lean):
    keys = torch.ones(1, 2, 17, 8)
    ceilings = torch.zeros(1, 2, 17)
    fast = make_cache(monkeypatch, 2, 8, lean, True)
    tau = torch.ones(2)
    fast.ingest(keys, ceilings, 8, tau)
    initial = fast.F
    fast.ingest(keys, ceilings, 17, tau)
    fast.ingest(keys, ceilings, 17, tau)
    assert fast.F is initial
    assert (fast.m_tot == 0).all()
    out, visits, reads = selections(fast.F, torch.zeros(2, 1, 8), 512)
    assert (out == -1).all()
    assert (visits == 0).all()
    assert (reads == 0).all()


@pytest.mark.parametrize("lean", [False, True])
@torch.no_grad()
def test_two_forest_decode_outputs(monkeypatch, lean):
    from src.fsattn.attention import FSAttn

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "scaling"))
    from experiments.scaling.kv_cache import FSKVCache

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    monkeypatch.setenv("FSB_DECODE_LEAN", "1" if lean else "0")
    torch.manual_seed(128)
    attn = FSAttn(
        d_model=128, n_heads=2, window_size=8, causal=True, use_rope=False,
        lr_scale_mode="one", cell_stride=4, cell_topk=4, sel_strict_causal=1,
        fsb_enabled=True, fsb_tree="forest", fsb_width=4, fsb_read_top=4,
        fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_union_width=4,
        fsb_union_tau_quantile=0.5,
    ).to(dev).eval()
    attn._cell_block = 16
    x = torch.randn(1, 145, 128, device=dev)
    with torch.no_grad():
        c0 = FSKVCache(attn, lean_prefill=True)
        attn(x[:, :64], cache=c0)
        ce0 = c0._ceil[0]                                    # (H, ncell)
        attn.fsb_tau_union.copy_(torch.stack([ce0[h].float().quantile(q) for h, q in enumerate((0.5, 0.8))]))

    def run(fast):
        monkeypatch.setenv("FSB_FAST_CACHE", "1" if fast else "0")
        cache = FSKVCache(attn, lean_prefill=True)
        cache.reserve = x.shape[1] - 64
        outputs = [attn(x[:, :64], cache=cache)]
        selected = []
        for pos in range(64, x.shape[1]):
            outputs.append(attn(x[:, pos:pos + 1], cache=cache))
            selected.append(cache._fsb_forest._last_sel.clone())
        assert cache._fsb_forest.utau is not None
        assert cache._fsb_forest.utau.m_tot.sum() > 0
        assert cache._fsb_forest.utau.m_tot.sum() < cache._fsb_forest.m_tot.sum()
        return torch.cat(outputs, dim=1), selected

    arms = (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")) if dev == "cuda" else (("0", "0"),)
    outs = {}
    for bc, dg in arms:                                    # FSB_BUILD_CUDA x FSB_DECODE_GRAPH
        monkeypatch.setenv("FSB_BUILD_CUDA", bc)
        monkeypatch.setenv("FSB_DECODE_GRAPH", dg)
        reference, ref_sel = run(False)
        actual, actual_sel = run(True)
        assert torch.equal(reference, actual), ("fast", bc, dg)
        assert all(torch.equal(a, b) for a, b in zip(ref_sel, actual_sel)), ("fast sel", bc, dg)
        outs[(bc, dg)] = (reference, ref_sel)
    reference, ref_sel = outs[arms[0]]
    for arm, (other, other_sel) in outs.items():
        assert torch.equal(reference, other), arm
        assert all(torch.equal(a, b) for a, b in zip(ref_sel, other_sel)), arm


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the CUDA decode forest build runs on CUDA")


@_CUDA
@pytest.mark.parametrize("spread", [False, True])
@pytest.mark.parametrize("tau_mode", ["zero", "quantile"])
def test_decode_ingest_cuda_matches_lean_reference(monkeypatch, tau_mode, spread):
    # the decode-time whole-forest CUDA build (FSB_BUILD_CUDA=1) packs exactly the lean reference's forest
    from src.fsattn import fsb_build_cuda as bc

    torch.manual_seed(129)
    rows, cells, dim = 4, 96, 64
    keys = torch.randn(1, rows, cells, dim, device="cuda")
    ceilings = torch.rand(1, rows, cells, device="cuda")
    if tau_mode == "zero":
        tau = torch.zeros(rows, device="cuda")
    else:
        tau = torch.stack([ceilings[0, h].quantile(q) for h, q in enumerate((0.2, 0.4, 0.6, 0.8))])
    spread_all = torch.rand(1, rows, cells, device="cuda") if spread else None
    q = torch.randn(rows, 3, dim, device="cuda")
    steps = (8, 16, 17, 24, 32, 33, 48, 64, 65, 80, 96)
    calls = []
    real = bc.forests_cuda
    monkeypatch.setattr(bc, "forests_cuda", lambda *a, **k: calls.append(1) or real(*a, **k))
    runs = {}

    for flag in ("0", "1"):
        monkeypatch.setenv("FSB_BUILD_CUDA", flag)
        monkeypatch.setenv("FSB_FAST_CACHE", "1")
        cache = ForestCache(rows, dim, "cuda", lean=True)
        calls.clear()
        snap = []
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for visible in steps:
                cache.ingest(keys, ceilings, visible, tau, spread_all=spread_all)
                F = cache.F
                snap.append((F["node_mu"].clone(), F["node_aux"].clone(), F["leaf"].clone(), dict(F["bases"]),
                             dict(F["leafbases"]), F["jbase"].clone(), cache.m_tot.clone(), cache.adm.clone(),
                             selections(F, q, visible * 16)))
        runs[flag] = snap
        if flag == "1":
            assert len(calls) == len(steps)                # every step built through the CUDA forest
            assert not [w for w in rec if "forest build failed" in str(w.message)]
    for step, (ref, got) in enumerate(zip(runs["0"], runs["1"])):
        for name, a, b in zip(("node_mu", "leaf", "bases", "leafbases", "jbase", "m_tot", "adm"),
                              ref[:1] + ref[2:8], got[:1] + got[2:8]):
            assert (a == b) if isinstance(a, dict) else torch.equal(a, b), (name, steps[step])
        # both writers store radius + |mu32 - bf16(mu32)| as the bound; the two eps terms round in different
        # reduction orders, so cols 1..8 and the leaves match bit for bit while col 0 agrees up to a bf16 slack
        for name, i in (("aux1+", 1),):
            assert torch.equal(ref[1][..., i:], got[1][..., i:]), (name, steps[step])
        slack = ref[0].float().norm(dim=-1) * (2.0**-8) + 1e-6
        assert bool(((ref[1][..., 0] - got[1][..., 0]).abs() <= slack).all()), ("aux0", steps[step])
        assert all(torch.equal(a, b) for a, b in zip(ref[8], got[8])), ("selections", steps[step])


@_CUDA
def test_decode_ingest_cuda_transient_failure_falls_back(monkeypatch):
    # a transient failure of the CUDA build falls back for that call; later ingests use it again (bytes unchanged)
    from src.fsattn import fsb_build_cuda as bc

    torch.manual_seed(130)
    rows, cells, dim = 4, 80, 64
    keys = torch.randn(1, rows, cells, dim, device="cuda")
    ceilings = torch.rand(1, rows, cells, device="cuda")
    tau = torch.stack([ceilings[0, h].quantile(q) for h, q in enumerate((0.1, 0.3, 0.5, 0.7))])
    q = torch.randn(rows, 3, dim, device="cuda")
    steps = (16, 24, 32, 48, 64, 80)
    monkeypatch.setenv("FSB_FAST_CACHE", "1")

    def run():
        cache = ForestCache(rows, dim, "cuda", lean=True)
        snap = []
        for visible in steps:
            cache.ingest(keys, ceilings, visible, tau)
            F = cache.F
            snap.append((F["node_mu"].clone(), F["node_aux"].clone(), F["leaf"].clone(), F["jbase"].clone(),
                         cache.m_tot.clone(), cache.adm.clone(), selections(F, q, visible * 16)))
        return snap

    monkeypatch.setenv("FSB_BUILD_CUDA", "0")
    ref = run()
    monkeypatch.setenv("FSB_BUILD_CUDA", "1")
    calls = []
    real = bc.build_forests_cuda

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("cuLaunchKernel: CUDA error 999 (unknown error, test)")
        return real(*a, **k)

    monkeypatch.setattr(bc, "build_forests_cuda", flaky)
    got = run()
    assert len(calls) == len(steps)                       # attempted every step, failed only the first
    for step, (a, b) in enumerate(zip(ref, got)):
        assert all(torch.equal(x, y) for x, y in zip(a[0:1] + a[2:6], b[0:1] + b[2:6])), steps[step]
        # means and the non-radius aux fields match bit for bit; each writer's radius carries its own bf16
        # mean-rounding slack, so col 0 agrees only up to that scale
        assert torch.equal(a[1][..., 1:], b[1][..., 1:]), ("aux1+", steps[step])
        slack = a[0].float().norm(dim=-1) * (2.0**-8) + 1e-6
        assert bool(((a[1][..., 0] - b[1][..., 0]).abs() <= slack).all()), ("aux0", steps[step])
        assert all(torch.equal(x, y) for x, y in zip(a[6], b[6])), ("selections", steps[step])
