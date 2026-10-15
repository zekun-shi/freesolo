from types import SimpleNamespace

import pytest
import torch

from src.fsattn import fsb_fast_step as fast
from src.fsattn import fsb_traverse_triton, fsb_tree


@pytest.mark.parametrize("queries", [1, 2, 128])
def test_fusion_dispatch(monkeypatch, queries):
    monkeypatch.setattr(fast, "_HAVE_TRITON", True)
    monkeypatch.setenv("FSB_FAST_BEAM", "1")
    monkeypatch.setenv("FSB_FAST_RERANK", "1")
    q = SimpleNamespace(is_cuda=True, dtype=torch.float32, shape=(2, queries, 8))
    f = {
        "node": torch.zeros(2, 1, 17),
        "leaf": torch.zeros(2, 1, dtype=torch.long),
        "adm": torch.zeros(2, 1, dtype=torch.long),
    }
    sp = {
        "node": torch.zeros(2, 1, 17),
        "root": torch.zeros(2, dtype=torch.long),
        "t_idx": torch.arange(2),
    }
    sel = SimpleNamespace(
        is_cuda=True,
        dtype=torch.int64,
        shape=(2, queries, 16),
        is_contiguous=lambda: True,
    )
    assert fast.fast_beam_ok(f, sp, q, 16, 0)
    assert fast.fast_rerank_ok(sel, q, object(), 16, 4, "lse_pos") == (queries == 1)
    monkeypatch.setenv("FSB_FAST_BEAM", "0")
    monkeypatch.setenv("FSB_FAST_RERANK", "0")
    assert not fast.fast_beam_ok(f, sp, q, 16, 0)
    assert not fast.fast_rerank_ok(sel, q, object(), 16, 4, "lse_pos")


def test_fused_beam_skips_reference_allocations(monkeypatch):
    monkeypatch.setenv("FSB_TRAVERSE_TRITON", "1")
    monkeypatch.setattr(fsb_traverse_triton, "_HAVE_TRITON", True)
    monkeypatch.setattr(fast, "fast_beam_ok", lambda *args: True)
    expected = object()
    monkeypatch.setattr(fast, "forest_beam_fused", lambda *args: expected)
    delta = torch.zeros(2, 1, dtype=torch.long)

    def unused(*args, **kwargs):
        pytest.fail("reference beam setup ran before fused dispatch")

    monkeypatch.setattr(torch, "full", unused)
    q = SimpleNamespace(is_cuda=True, shape=(2, 1, 8), device=torch.device("cpu"))
    result = fsb_tree._forest_beam(
        {}, {}, q, None, None, None, 1.0, 16, None, delta, 16, 0, "bound", 1.0, None
    )
    assert result is expected


@pytest.mark.parametrize("layout", ["transpose", "slice"])
def test_rerank_position_term_reads_logical_cell_table(monkeypatch, layout):
    class Kernel:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append(args)

            return launch

    score, select = Kernel(), Kernel()
    monkeypatch.setattr(fast, "_fsb_u_kernel", score, raising=False)
    monkeypatch.setattr(fast, "_fsb_sel_kernel", select, raising=False)
    rows, cells, dim, size = 3, 9, 8, 16
    cell_a = (
        torch.rand(cells, rows).t()
        if layout == "transpose"
        else torch.rand(rows, 2 * cells + 1)[:, :cells]
    )
    assert not cell_a.is_contiguous()
    sel = torch.arange(-1, cells - 1).view(1, 1, cells).expand(rows, 1, cells).contiguous()
    qpos = torch.full((rows, 1), 100.0)
    lam = torch.rand(rows, 1, 1)
    sigv = torch.rand(rows, 1, 1) * 7 + 3
    t_idx = torch.arange(rows)
    fast.rerank_top_fast(sel, torch.zeros(rows, 1, dim), qpos, lam, sigv, cell_a, t_idx, 1.0, size, 4, "lse_pos",
                         torch.zeros(rows, cells * size, dim))
    assert len(score.calls) == len(select.calls) == 1
    a = cell_a.reshape(-1)[(t_idx.view(rows, 1, 1) * cells + sel.clamp(min=0)).reshape(-1)].view(sel.shape)
    expected = lam * a * torch.erf((qpos[..., None] - (sel.float() + 0.5) * size).abs() / (2.0 * sigv))
    assert torch.equal(score.calls[0][5], expected)


gpu = pytest.mark.skipif(not (torch.cuda.is_available() and fast._HAVE_TRITON), reason="needs CUDA and Triton")


def _bits(t):
    return t.contiguous().view(torch.int32)


@gpu
def test_fused_rerank_scores_match_reference_bitwise():
    # learned-sigma heads: 2 sigma is not a power of two and queries sit near the candidates, so the
    # position term exercises the division and erf rounding
    g = torch.Generator().manual_seed(5)
    rows, C, D, s, M, H = 1024, 64, 64, 32, 512, 16
    tokK = (torch.randn(1, H, M * s, D, generator=g) * 0.5).to("cuda", torch.bfloat16)
    t_idx = torch.arange(rows, device="cuda") % H
    sel = torch.randint(-1, M, (rows, 1, C), generator=g).cuda()
    q = (torch.randn(rows, 1, D, generator=g) * 0.5).cuda()
    qpos = (torch.rand(rows, 1, generator=g) * M * s).cuda()
    lam = (torch.rand(rows, 1, 1, generator=g) * 3).cuda()
    sigv = (torch.rand(rows, 1, 1, generator=g) * 1000 + 50).cuda()
    cell_a = torch.rand(H, M, generator=g).cuda()
    from src.fsattn.fsb_rerank import cell_token_scores
    for score in ("lse_pos", "lse", "max"):
        content = cell_token_scores(q, sel, tokK, t_idx, 0.125, s, score)
        if score == "lse_pos":
            a = cell_a.reshape(-1)[(t_idx.view(-1, 1, 1) * M + sel.clamp(min=0, max=M - 1)).reshape(-1)].view(sel.shape)
            content = content + lam * a * torch.erf((qpos[..., None] - (sel.float() + 0.5) * s).abs() / (2.0 * sigv))
        ref_u = content.masked_fill(sel < 0, float("-inf"))
        got_u = fast.rerank_scores_fast(sel, q, qpos, lam, sigv, cell_a, t_idx, 0.125, s, score, tokK)
        assert torch.equal(_bits(ref_u), _bits(got_u)), score
        ref = fsb_tree.rerank_top(sel, q, qpos, lam, sigv, torch.zeros(H, M, D, device="cuda"), cell_a, t_idx,
                                  0.125, s, 4, score=score, tokK=tokK)
        got = fast.rerank_top_fast(sel, q, qpos, lam, sigv, cell_a, t_idx, 0.125, s, 4, score, tokK)
        assert torch.equal(ref, got), score


@gpu
@pytest.mark.parametrize("cuda", ["rows", "warp", "0"])
@pytest.mark.parametrize("width,delta", [(16, 0), (4, 2)])
def test_fused_beam_matches_level_kernel(monkeypatch, width, delta, cuda):
    # 65536 descents through a 16k-token forest: fused descent (its CUDA builds, row-cooperative or warp per query
    # with FSB_BEAM_ROWS=0, or the Triton kernel with FSB_BEAM_CUDA=0) == the per-level Triton reference
    g = torch.Generator().manual_seed(9)
    H, D, s, Bq, nbc, M = 16, 64, 32, 128, 32, 512
    keys = (torch.randn(H, M * s, D, generator=g) * 0.5).to("cuda", torch.bfloat16).float().view(H, M, s, D)
    kbar, ceil = keys.mean(2), keys.norm(dim=-1).amax(-1)
    F = fsb_tree.build_forest(kbar, ceil, torch.ones(H, M, dtype=torch.bool, device="cuda"))
    blk = torch.arange(M * s // Bq, device="cuda", dtype=torch.float32)[-nbc:] * Bq
    m_b = torch.searchsorted(F["adm"].masked_fill(F["adm"] < 0, M),
                             (blk / s).floor().long()[None].expand(H, nbc).contiguous()).reshape(-1)
    t_idx = torch.arange(H, device="cuda").repeat_interleave(nbc)
    SP = fsb_tree.build_spines(F, m_b, t_idx)
    rows = H * nbc
    q = (torch.randn(rows, Bq, D, generator=g) * 0.5).cuda()
    qpos = (blk[:, None] + torch.arange(Bq, device="cuda", dtype=torch.float32))[None].expand(H, nbc, Bq)
    qpos = qpos.reshape(rows, Bq).contiguous()
    lam = torch.full((rows, 1, 1), 2.0, device="cuda")
    sigv = (torch.rand(rows, 1, 1, generator=g) * 1000 + 50).cuda()
    depth = (SP["height"] - delta).clamp(min=0)[:, None].expand(rows, Bq)
    W = torch.minimum(torch.full_like(depth, width), torch.bitwise_left_shift(torch.ones_like(depth), depth))
    dl = (SP["height"][:, None] - depth).clamp(min=0)
    monkeypatch.setenv("FSB_FAST_BEAM", "0")
    ref = fsb_tree._forest_beam(F, SP, q, qpos, lam, sigv, 0.125, s, W, dl, 16, 2, "bound", 8.0, None)
    monkeypatch.setenv("FSB_BEAM_CUDA", "0" if cuda == "0" else "1")
    monkeypatch.setenv("FSB_BEAM_ROWS", "1" if cuda == "rows" else "0")
    from src.fsattn import fsb_beam_cuda
    ran = []
    name = "forest_beam_rows_cuda" if cuda == "rows" else "forest_beam_cuda"
    real = getattr(fsb_beam_cuda, name)
    monkeypatch.setattr(fsb_beam_cuda, name, lambda *a: ran.append(real(*a)) or ran[-1])
    got = fast.forest_beam_fused(F, SP, q, qpos, lam, sigv, 0.125, s, W, dl, 16, 2, "bound", 8.0)
    if cuda != "0" and ran == [None]:
        pytest.skip("NVRTC unavailable: the CUDA build fell back to the Triton kernel")
    assert len(ran) == (cuda != "0") and all(r is not None for r in ran)         # the build under test produced it
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


@gpu
@pytest.mark.parametrize("Bq", [1, 128])
@pytest.mark.parametrize("layout", ["contiguous", "transpose"])
def test_fused_union_rerank_matches_reference_bitwise(Bq, layout):
    # union_candidates + rerank_top + utau_prov_add == union_rerank_fast (read cells and provenance): -1 slots,
    # cells found by both beams, cells with identical keys (tied scores), queries next to their candidates
    g = torch.Generator().manual_seed(17 + Bq)
    rows, Ca, Cb, D, s, M, H, K = 2048 // Bq, 16, 16, 64, 32, 512, 16, 4
    keys = torch.randn(1, H, M, s, D, generator=g) * 0.5
    keys[:, :, 1:24] = keys[:, :, :1]
    tokK = keys.reshape(1, H, M * s, D).to("cuda", torch.bfloat16)
    t_idx = torch.arange(rows, device="cuda") % H
    sel_a = torch.randint(-1, M, (rows, Bq, Ca), generator=g)
    sel_b = torch.where(torch.rand(rows, Bq, Cb, generator=g) < 0.3, sel_a[..., torch.randperm(Ca, generator=g)[:Cb]],
                        torch.randint(-1, M, (rows, Bq, Cb), generator=g))
    sel_a, sel_b = sel_a.cuda(), sel_b.cuda()
    q = (torch.randn(rows, Bq, D, generator=g) * 0.5).cuda()
    qpos = ((torch.rand(rows, 1, generator=g) * (M - 4) * s).floor() + torch.arange(Bq)).cuda()
    lam = (torch.rand(rows, 1, 1, generator=g) * 3).cuda()
    sigv = (torch.rand(rows, 1, 1, generator=g) * 1000 + 50).cuda()
    cell_a = torch.rand(H, M, generator=g).cuda() if layout == "contiguous" else torch.rand(M, H, generator=g).cuda().t()
    cell_k = torch.zeros(H, M, D, device="cuda")
    for score in ("lse_pos", "lse", "max"):
        for second in (True, False):
            sel = fsb_tree.union_candidates(sel_a, sel_b) if second else sel_a
            ref = fsb_tree.rerank_top(sel, q, qpos, lam, sigv, cell_k, cell_a, t_idx, 0.125, s, K, score=score,
                                      tokK=tokK)
            got, pv = fast.union_rerank_fast(sel_a, sel_b if second else None, q, qpos, lam, sigv, cell_a, t_idx,
                                             0.125, s, K, score, tokK, prov=second)
            assert torch.equal(ref, got), (score, second)
            if second:
                prov = []
                fsb_tree.utau_prov_add(prov, ref, sel[..., Ca:])
                assert torch.equal(prov[0], pv), score
