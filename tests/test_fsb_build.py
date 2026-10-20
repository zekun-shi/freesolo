import os

import pytest
import torch

from src.fsattn import fsb_tree as ft
from src.fsattn.fsb_build_fast import _HAVE_TRITON, build_trees_fast

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and _HAVE_TRITON),
                                reason="fused FS-B construction needs CUDA and Triton")
DEV = "cuda"


def _bits(t):
    t = t.contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _same(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))
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
    return K


@pytest.mark.parametrize("T,M,L,Dh", [(3, 1, 0, 16), (4, 5, 3, 16), (16, 64, 6, 64), (16, 100, 7, 64),
                                      (8, 300, 9, 64), (4, 1000, 10, 32), (2, 200, 8, 96), (2, 4096, 12, 64)])
@pytest.mark.parametrize("kind", ["randn", "ties", "zeros", "negzero", "dups"])
def test_build_trees_fast_matches_reference(monkeypatch, T, M, L, Dh, kind):
    g = torch.Generator().manual_seed(1000 * M + L)
    K = _keys(kind, T, M, Dh, g).to(DEV)
    ce = torch.rand(T, M, generator=g).to(DEV)
    spread = torch.rand(T, M, generator=g).to(DEV)
    admits = [torch.ones(T, M, dtype=torch.bool), torch.rand(T, M, generator=g) < 0.5,
              torch.zeros(T, M, dtype=torch.bool), torch.arange(M)[None] < torch.randint(0, M + 1, (T, 1), generator=g)]
    monkeypatch.setenv("FSB_FAST_BUILD", "0")
    for adm in admits:
        adm = adm.to(DEV)
        for sp in (None, spread):
            ref = ft.build_trees(K, ce, adm, L, spread=sp)
            got = build_trees_fast(K, ce, adm, L, spread=sp)
            assert [k for k in ref if not _same(ref[k], got[k])] == []


@pytest.mark.parametrize("T,M,Dh", [(4, 7, 16), (16, 512, 64), (8, 1000, 32)])
@pytest.mark.parametrize("kind", ["randn", "zeros"])
def test_forest_and_spines_fast_match_reference(monkeypatch, T, M, Dh, kind):
    g = torch.Generator().manual_seed(7 * M + Dh)
    K = _keys(kind, T, M, Dh, g).to(DEV)
    ce = torch.rand(T, M, generator=g).to(DEV)
    adm = (torch.rand(T, M, generator=g) < 0.6).to(DEV)
    monkeypatch.setenv("FSB_FAST_BUILD", "0")
    ref = ft.build_forest(K, ce, adm)
    monkeypatch.setenv("FSB_FAST_BUILD", "1")
    got = ft.build_forest(K, ce, adm)
    for k in ("adm", "m_tot", "node", "leaf"):
        assert _same(ref[k], got[k]), k
    mt = ref["m_tot"]
    mb = torch.cat([torch.randint(0, int(mt.max()) + 1, (32,), generator=g).to(DEV).clamp(max=mt[t]) for t in range(T)])
    ti = torch.arange(T, device=DEV).repeat_interleave(32)
    monkeypatch.setenv("FSB_FAST_BUILD", "0")
    a = ft.build_spines(ref, mb, ti)
    monkeypatch.setenv("FSB_FAST_BUILD", "1")
    b = ft.build_spines(ref, mb, ti)
    for k in ("node", "root", "height"):
        assert _same(a[k], b[k]), k


@pytest.mark.parametrize("fast_cache", ["0", "1"])
def test_lean_stream_spines_match_reference(monkeypatch, fast_cache):
    monkeypatch.setenv("FSB_FAST_CACHE", fast_cache)
    T, M, Dh = 16, 400, 64
    g = torch.Generator().manual_seed(400)
    K = torch.randn(1, T, M, Dh, generator=g).to(DEV)
    ce = torch.rand(1, T, M, generator=g).to(DEV)
    tau = (torch.rand(T, generator=g) * 0.3).to(DEV)
    caches = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("FSB_FAST_BUILD", flag)
        caches[flag] = ft.ForestCache(T, Dh, DEV, lean=True)
    for Mv in list(range(1, 24)) + list(range(24, M, 17)):
        out = {}
        for flag, fc in caches.items():
            monkeypatch.setenv("FSB_FAST_BUILD", flag)
            fc.ingest(K, ce, Mv, tau, keep_cells=True)
            out[flag] = ft.build_spines(fc.F, fc.m_tot, torch.arange(T, device=DEV))
        F0, F1 = caches["0"].F, caches["1"].F
        if F0.get("node_mu") is None:
            assert _same(F0["node"], F1["node"]), Mv            # the packed flatten keeps exact fp32 rows
        else:
            assert _same(F0["node_mu"], F1["node_mu"]), Mv
            assert _same(F0["node_aux"][..., 1:], F1["node_aux"][..., 1:]), Mv
            # both writers store radius + |mu32 - bf16(mu32)| as the bound; the two eps terms round in different
            # reduction orders, so the radii agree only up to a bf16-scale slack on the row's mean norm
            slack = F0["node_mu"].float().norm(dim=-1) * (2.0**-8) + 1e-6
            d0 = (F0["node_aux"][..., 0] - F1["node_aux"][..., 0]).abs()
            assert bool((d0 <= slack).all()), Mv
        assert _same(F0["leaf"], F1["leaf"]), Mv
        for k in ("node", "root", "height"):
            a, b = out["0"][k], out["1"][k]
            if F0.get("node_mu") is None:
                assert _same(a, b), (Mv, k)
            elif k == "node":
                Dh2 = F0["Dh"]
                assert _same(torch.cat([a[..., :Dh2], a[..., Dh2 + 1:]], -1),
                             torch.cat([b[..., :Dh2], b[..., Dh2 + 1:]], -1)), (Mv, k)
                slack = a[..., :Dh2].float().norm(dim=-1) * (2.0**-8) + 1e-6
                assert bool(((a[..., Dh2] - b[..., Dh2]).abs() <= slack).all())
            else:
                assert _same(a, b), (Mv, k)
