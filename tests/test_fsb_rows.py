import os
import sys

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# kv_cache.py and decode_graph.py import their siblings by bare name (as the eval drivers do)
sys.path[:0] = [os.path.join(_REPO, "experiments", "scaling"), _REPO]

from src.fsattn import fsb_fast_step as fast
from src.fsattn import fsb_rows as rows
from src.fsattn import fsb_tree as ft
from src.fsattn.fsb_rerank import cell_token_scores

gpu = pytest.mark.skipif(not (torch.cuda.is_available() and rows._HAVE_TRITON), reason="needs CUDA and Triton")


def _stage(seed, M, H=4, nbc=8, Bq=128, D=64, s=32):
    # one prefill traversal stage with clustered keys and queries, so that the queries of a block share candidate
    # cells as on real prompts; bf16 keys, two forests, both beams from the fused kernel
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(H, 32, D, generator=g)
    keys = (centers[:, torch.randint(0, 32, (M * s,), generator=g)] + 0.6 * torch.randn(H, M * s, D, generator=g)) * 0.5
    keys = keys.to("cuda", torch.bfloat16)
    kv = keys.float().view(H, M, s, D)
    kbar, ceil = kv.mean(2), kv.norm(dim=-1).amax(-1)
    blk = torch.arange(M * s // Bq, device="cuda", dtype=torch.float32)[-nbc:] * Bq
    n = H * nbc
    qc = torch.randint(0, 32, (n, Bq), generator=g)
    q = (centers[torch.arange(n)[:, None] // nbc, qc] + 0.8 * torch.randn(n, Bq, D, generator=g)) * 0.5
    q = q.to(torch.bfloat16).float().cuda()
    qpos = (blk[:, None] + torch.arange(Bq, device="cuda", dtype=torch.float32))[None].expand(H, nbc, Bq)
    qpos = qpos.reshape(n, Bq).contiguous()
    lam = (torch.rand(n, 1, 1, generator=g) * 2 + 0.5).cuda()
    sigv = (torch.rand(n, 1, 1, generator=g) * 1000 + 50).cuda()
    sels = []
    for admit in (torch.ones(H, M, dtype=torch.bool, device="cuda"), ceil > ceil.median()):
        F = ft.build_forest(kbar, ceil, admit)
        m_b = torch.searchsorted(F["adm"].masked_fill(F["adm"] < 0, M),
                                 (blk / s).floor().long()[None].expand(H, nbc).contiguous()).reshape(-1)
        SP = ft.build_spines(F, m_b, torch.arange(H, device="cuda").repeat_interleave(nbc))
        depth = SP["height"][:, None].expand(n, Bq)
        W = torch.minimum(torch.full_like(depth, 16), torch.bitwise_left_shift(torch.ones_like(depth), depth))
        dl = (SP["height"][:, None] - depth).clamp(min=0)
        sels.append(fast.forest_beam_fused(F, SP, q, qpos, lam, sigv, 0.125, s, W, dl, 16, 2, "bound", 8.0)[0])
    return dict(sel_a=sels[0], sel_b=sels[1], q=q, qpos=qpos, lam=lam, sigv=sigv, cellK=kbar, cellA=ceil.contiguous(),
                t_idx=SP["t_idx"], tokK=keys.unsqueeze(0), s=s)


@gpu
@pytest.mark.parametrize("score", ["lse_pos", "lse", "max"])
@pytest.mark.parametrize("M", [512, 2048])
def test_block_shared_rerank_matches_rerank_top(score, M):
    st = _stage(7 + M, M)
    Ca = st["sel_a"].shape[-1]
    sel = ft.union_candidates(st["sel_a"], st["sel_b"])
    ref = ft.rerank_top(sel, st["q"], st["qpos"], st["lam"], st["sigv"], st["cellK"], st["cellA"], st["t_idx"], 0.125,
                        st["s"], 4, score=score, tokK=st["tokK"])
    prov = []
    ft.utau_prov_add(prov, ref, sel[..., Ca:])
    got, pv = rows.union_rerank_rows(st["sel_a"], st["sel_b"], st["q"], st["qpos"], st["lam"], st["sigv"], st["cellA"],
                                     st["t_idx"], 0.125, st["s"], 4, score, st["tokK"], prov=True)
    same = (ref.sort(-1).values == got.sort(-1).values).all(-1)
    assert same.float().mean() > 0.999
    if not same.all():
        # every disagreement is a near-tie of the reference scores: 4th and 5th best within 1e-4 relative
        u = cell_token_scores(st["q"], sel, st["tokK"], st["t_idx"], 0.125, st["s"], score)
        if score == "lse_pos":
            a = st["cellA"].reshape(-1)[(st["t_idx"].view(-1, 1, 1) * M + sel.clamp(min=0)).reshape(-1)].view(sel.shape)
            u = u + st["lam"] * a * torch.erf((st["qpos"][..., None] - (sel.float() + 0.5) * st["s"]).abs()
                                              / (2.0 * st["sigv"]))
        top = u.masked_fill(sel < 0, float("-inf")).topk(5, dim=-1).values
        gap = (top[..., 3] - top[..., 4]) / top[..., 0].abs().clamp(min=1.0)
        assert bool((gap[~same] < 1e-4).all()), gap[~same].max()
    order = same & (ref == got).all(-1)
    assert torch.equal(prov[0][order], pv[order])


@gpu
def test_block_shared_rerank_fallbacks(monkeypatch):
    monkeypatch.setenv("FSB_FAST_ROWS", "1")
    q = torch.zeros(4, 128, 64, device="cuda")
    keys = torch.zeros(1, 4, 1024, 64, device="cuda", dtype=torch.bfloat16)
    assert rows.rows_ok(q, keys, 32, 4, "lse_pos")
    assert not rows.rows_ok(q, keys.float(), 32, 4, "lse_pos")            # fp32 keys stay on rerank_top
    assert not rows.rows_ok(q[:, :1], keys, 32, 4, "lse_pos")             # decode rows use the per-query kernels
    assert not rows.rows_ok(q, keys, 32, 4, "leaf")
    monkeypatch.setenv("FSB_FAST_ROWS", "0")
    assert not rows.rows_ok(q, keys, 32, 4, "lse_pos")
    monkeypatch.setenv("FSB_FAST_ROWS", "1")
    seen, x, g = [], torch.zeros(1, device="cuda"), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):                                              # captured stages: no host sync allowed
        seen.append(rows.rows_ok(q, keys, 32, 4, "lse_pos"))
        x.add_(1)
    assert seen == [False]


@gpu
def test_rows_path_is_opt_in(monkeypatch):
    # traverse_forest takes the block-shared re-rank only when the caller passes rows=True (eager inference prefill);
    # graphed stages, training and traced re-traversals keep rerank_top
    monkeypatch.setenv("FSB_FAST_ROWS", "1")
    st = _stage(11, 512)
    F = ft.build_forest(st["cellK"], st["cellA"], torch.ones_like(st["cellA"], dtype=torch.bool))
    F["cellK"], F["cellA"] = st["cellK"], st["cellA"]
    H, nbc, Bq = 4, 8, 128
    M = st["cellA"].shape[1]
    blk = torch.arange(M * 32 // Bq, device="cuda", dtype=torch.float32)[-nbc:] * Bq
    m_b = torch.searchsorted(F["adm"].masked_fill(F["adm"] < 0, M),
                             (blk / 32).floor().long()[None].expand(H, nbc).contiguous()).reshape(-1)
    SP = ft.build_spines(F, m_b, torch.arange(H, device="cuda").repeat_interleave(nbc))
    n = H * nbc
    depth = SP["height"][:, None].expand(n, Bq)
    W = torch.minimum(torch.full_like(depth, 16), torch.bitwise_left_shift(torch.ones_like(depth), depth))
    dl = (SP["height"][:, None] - depth).clamp(min=0)
    calls = []
    real = rows.union_rerank_rows
    monkeypatch.setattr(rows, "union_rerank_rows", lambda *a, **k: calls.append(1) or real(*a, **k))
    args = (F, SP, st["q"], st["qpos"], st["lam"], st["sigv"], 0.125, 32, W, dl, 16, 2, "bound")
    kw = dict(radius_coef=8.0, read_top=4, rerank_score="lse_pos", tokK=st["tokK"], union_width=16, union_W_max=16)
    ref = ft.traverse_forest(*args, **kw)
    assert calls == []
    got = ft.traverse_forest(*args, rows=True, **kw)
    assert calls == [1]
    assert (ref[0].sort(-1).values == got[0].sort(-1).values).all(-1).float().mean() > 0.999


@gpu
def test_whole_step_decode_matches_cached_generate(monkeypatch):
    # the paper's FS-B serve config on a tiny random model, with 4-wide beams (narrower than the ~20 cells of the
    # context, so the filtered second forest - each head's threshold at the median cell ceiling of the prompt - adds
    # reads of its own): whole-step graph decode across query-block boundaries and past the model's initial RoPE
    # table gives the tokens and read statistics of the per-layer decode loop.
    # Bit identity between the graph and the loop holds only on identical KV storage: pin the
    # bf16 store (FS_KV_INT8=0); the int8 store has its own parity coverage in tests/test_kv_int8.py
    from experiments.scaling.model import FreeSoloLM
    from kv_cache import cached_generate, make_caches, _trunk_cached
    kw = dict(window_size=64, long_range_variant="cell_topk", cell_stride=32, cell_topk=4, cell_select="score",
              sel_strict_causal=1, use_rope=True, fs_uniform_sr=True, rope_long_range=False, cell_prune_kernel=0,
              fsb_enabled=True, fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_node_score="bound",
              fsb_bound_radius_coef=8.0, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score="lse_pos",
              fsb_rerank_query="raw", fsb_union_width=4, fsb_union_tau_quantile=0.5)
    torch.manual_seed(0)
    m = FreeSoloLM(vocab_size=256, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_seq_len=512,
                   attn_type="fsattn", **kw).cuda().eval()
    layers = [mod for mod in m.modules() if getattr(mod, "fsb_enabled", False)]
    prompt = torch.randint(0, 256, (1, 300), generator=torch.Generator().manual_seed(1))
    monkeypatch.setenv("FS_KV_INT8", "0")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        caches = make_caches(m)
        _trunk_cached(m, prompt.cuda(), caches)
        for mod, c in zip(layers, caches):
            mod.fsb_tau.fill_(0.0)
            mod.fsb_tau_union.copy_(c._ceil[0].float().median(dim=-1).values)
    out, acc = {}, {}
    for whole in ("0", "1"):
        monkeypatch.setenv("FSB_DECODE_WHOLE", whole)
        ft.fsb_read_stats_reset(m)
        out[whole] = cached_generate(m, prompt, 300, device="cuda", dtype="bfloat16")
        acc[whole] = torch.stack([mod._fsb_read_acc for mod in layers])
    ft.fsb_read_stats_reset(m, on=False)
    assert out["1"].shape[1] == 600 > 512
    assert torch.equal(out["0"], out["1"])
    assert torch.equal(acc["0"], acc["1"])
    assert float(acc["1"][:, 3].sum()) > 0                       # reads found only by the second beam


def _tiny_fsb(N, layers=2, **over):
    # the paper's FS-B serve config (fixed policy, 16-wide beams, union beam on the median-ceiling forest, top-4 lse_pos
    # re-rank) on a tiny random model, two query blocks per prefill stage; thresholds set from one prefill
    from experiments.scaling.model import FreeSoloLM
    from kv_cache import make_caches, _trunk_cached
    kw = dict(window_size=64, long_range_variant="cell_topk", cell_stride=32, cell_topk=4, cell_select="score",
              sel_strict_causal=1, use_rope=True, fs_uniform_sr=True, rope_long_range=False, cell_prune_kernel=0,
              fsb_enabled=True, fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_node_score="bound",
              fsb_bound_radius_coef=8.0, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score="lse_pos",
              fsb_rerank_query="raw", fsb_union_width=16, fsb_union_tau_quantile=0.5, fsb_prefill_blkchunk=2)
    kw.update(over)
    torch.manual_seed(0)
    m = FreeSoloLM(vocab_size=256, d_model=128, n_layers=layers, n_heads=4, d_ff=256, max_seq_len=N + 64,
                   attn_type="fsattn", **kw).cuda().eval()
    x = torch.randint(0, 256, (1, N), generator=torch.Generator().manual_seed(N)).cuda()
    layers_ = [mod for mod in m.modules() if getattr(mod, "fsb_enabled", False)]
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        caches = make_caches(m)
        _trunk_cached(m, x, caches)
        for mod, c in zip(layers_, caches):
            mod.fsb_tau.fill_(0.0)
            mod.fsb_tau_union.copy_(c._ceil[0].float().median(dim=-1).values)
    return m, x


def _prefill_sel(m, x, monkeypatch):
    # every layer's selected cells (sel_full of the layer's last readout: a recomputed layer reads out twice) and
    # the last hidden state of one cached prefill
    from kv_cache import make_caches, _trunk_cached
    monkeypatch.setenv("FS_FUSED_ATTN", "0")
    rec, real = {}, ft.masked_readout

    def hook(attn, Q, Kp, Vp, sel_full, *a):
        rec[id(attn)] = sel_full.clone()
        return real(attn, Q, Kp, Vp, sel_full, *a)
    monkeypatch.setattr(ft, "masked_readout", hook)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        h = _trunk_cached(m, x, make_caches(m))
    monkeypatch.setattr(ft, "masked_readout", real)
    return [rec[id(mod)] for mod in m.modules() if id(mod) in rec], h.float()


@gpu
@pytest.mark.parametrize("compact", ["1", "0"])
def test_deferred_capacity_check_matches_stage_check(monkeypatch, compact):
    # rows overflow UCAP (8 here) in some prefill stages: the deferred flags, read after each layer's readout is
    # queued, make the layer recompute with the per-stage check, so every layer's selection, the hidden states and
    # the read statistics equal the per-stage host check's; with compact (16 + 16 slots: row-tile select) and full
    # (64 + 64: per-query select) beams
    monkeypatch.setenv("FSB_BEAM_COMPACT", compact)
    monkeypatch.setattr(rows, "UCAP", 8)
    m, x = _tiny_fsb(2048)
    seen, real = [], rows.union_rerank_rows

    def spy(*a, **k):
        res = real(*a, **k)
        o = k.get("ovf_out")
        seen.append(res is None if not o else bool(o[-1]))
        return res
    monkeypatch.setattr(rows, "union_rerank_rows", spy)
    out, acc = {}, {}
    layers = [mod for mod in m.modules() if getattr(mod, "fsb_enabled", False)]
    for defer in ("0", "1"):
        monkeypatch.setenv("FSB_ROWS_DEFER", defer)
        seen.clear()
        ft.fsb_read_stats_reset(m)
        out[defer] = _prefill_sel(m, x, monkeypatch)
        acc[defer] = torch.stack([mod._fsb_read_acc for mod in layers])
        assert any(seen) and not all(seen), seen                      # overflowing and in-capacity stages
    ft.fsb_read_stats_reset(m, on=False)
    assert len(out["0"][0]) == len(out["1"][0]) == 2
    assert all(torch.equal(a, b) for a, b in zip(out["0"][0], out["1"][0]))
    assert torch.equal(out["0"][1], out["1"][1])
    assert torch.equal(acc["0"], acc["1"]) and float(acc["1"][:, 0].sum()) > 0


@gpu
def test_rows_capacity_check_does_not_sync(monkeypatch):
    # contexts of at most UCAP cells need no check; longer ones defer it to a device flag (no stream sync either way)
    monkeypatch.setenv("FSB_ROWS_DEFER", "1")
    for M in (512, 2048):
        st = _stage(3 + M, M)
        args = (st["sel_a"], st["sel_b"], st["q"], st["qpos"], st["lam"], st["sigv"], st["cellA"], st["t_idx"], 0.125,
                st["s"], 4, "lse_pos", st["tokK"])
        ref = rows.union_rerank_rows(*args, prov=True)
        flags = []
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            got = rows.union_rerank_rows(*args, prov=True, ovf_out=flags)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        assert len(flags) == (1 if M > rows.UCAP else 0)
        assert not any(bool(f) for f in flags)
        assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


@gpu
@pytest.mark.parametrize("impl", ["cuda", "legacy"])
def test_rows_defer_off_checks_every_stage(monkeypatch, impl):
    # FSB_ROWS_DEFER=0 restores the host capacity check of every stage (one stream sync), also in contexts of at most
    # UCAP cells, where no row can overflow and the default skips it; same read cells either way
    monkeypatch.setenv("FSB_ROWS_IMPL", impl)
    for M in (512, 2048):
        st = _stage(3 + M, M)
        monkeypatch.setenv("FSB_ROWS_DEFER", "1")
        ref = _rows_call(st)
        monkeypatch.setenv("FSB_ROWS_DEFER", "0")
        flags = []
        assert _same(ref, _rows_call(st, ovf_out=flags)) and flags == []
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with pytest.raises(RuntimeError, match="synchroniz"):
                _rows_call(st, ovf_out=flags)
        finally:
            torch.cuda.set_sync_debug_mode(0)


def _fixed_beam_inputs(seed, M, fsb_delta, admit_all=True, H=4, nbc=8, Bq=128, D=64, s=32, width=16):
    # the fixed policy's beam inputs for one prefill stage: D = max(height - fsb_delta, 0), W = min(width, 2^D),
    # delta = height - D (= min(height, fsb_delta))
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(H, 32, D, generator=g)
    keys = (centers[:, torch.randint(0, 32, (M * s,), generator=g)] + 0.6 * torch.randn(H, M * s, D, generator=g)) * 0.5
    kv = keys.to("cuda", torch.bfloat16).float().view(H, M, s, D)
    kbar, ceil = kv.mean(2), kv.norm(dim=-1).amax(-1)
    admit = torch.ones(H, M, dtype=torch.bool, device="cuda") if admit_all else ceil > ceil.median()
    F = ft.build_forest(kbar, ceil, admit)
    blk = torch.arange(M * s // Bq, device="cuda", dtype=torch.float32)[-nbc:] * Bq
    n = H * nbc
    m_b = torch.searchsorted(F["adm"].masked_fill(F["adm"] < 0, M),
                             (blk / s).floor().long()[None].expand(H, nbc).contiguous()).reshape(-1)
    SP = ft.build_spines(F, m_b, torch.arange(H, device="cuda").repeat_interleave(nbc))
    qc = torch.randint(0, 32, (n, Bq), generator=g)
    q = ((centers[torch.arange(n)[:, None] // nbc, qc] + 0.8 * torch.randn(n, Bq, D, generator=g)) * 0.5)
    q = q.to(torch.bfloat16).float().cuda()
    qpos = (blk[:, None] + torch.arange(Bq, device="cuda", dtype=torch.float32))[None].expand(H, nbc, Bq)
    lam = (torch.rand(n, 1, 1, generator=g) * 2 + 0.5).cuda()
    sigv = (torch.rand(n, 1, 1, generator=g) * 1000 + 50).cuda()
    h = SP["height"]
    Dd = (h - fsb_delta).clamp(min=0)
    W = torch.minimum(torch.full_like(Dd, width), torch.bitwise_left_shift(torch.ones_like(Dd), Dd))[:, None].expand(n, Bq)
    delta = (h - Dd)[:, None].expand(n, Bq)
    return dict(F=F, SP=SP, q=q, qpos=qpos.reshape(n, Bq).contiguous(), lam=lam, sigv=sigv, W=W, delta=delta,
                kbar=kbar, ceil=ceil, keys=keys.to("cuda", torch.bfloat16), blk=blk, m_b=m_b)


@gpu
@pytest.mark.parametrize("fsb_delta,admit_all,impl", [
    (0, True, "cuda"), (0, False, "cuda"), (1, True, "cuda"), (1, False, "cuda"), (0, False, "triton"),
    (1, True, "triton"), (0, True, "level"), (1, False, "torch")])
def test_compact_beam_layout(monkeypatch, fsb_delta, admit_all, impl):
    # fixed policy: the beam with 2^fsb_delta cell slots per frontier slot returns the full layout's first 2^fsb_delta
    # slots of every frontier slot, the full layout's other slots are all empty, and the visit counts are equal; for
    # every beam implementation (CUDA, Triton fused, per-level Triton, torch)
    env = {"cuda": {}, "triton": {"FSB_BEAM_CUDA": "0"}, "level": {"FSB_FAST_BEAM": "0"},
           "torch": {"FSB_FAST_BEAM": "0", "FSB_TRAVERSE_TRITON": "0"}}[impl]
    for k_, v_ in env.items():
        monkeypatch.setenv(k_, v_)
    st = _fixed_beam_inputs(31 + 2 * fsb_delta + admit_all, 2048 if impl != "torch" else 512, fsb_delta, admit_all)
    args = (st["F"], st["SP"], st["q"], st["qpos"], st["lam"], st["sigv"], 0.125, 32, st["W"], st["delta"], 16)
    full = ft._forest_beam(*args, 2, "bound", 8.0, None)
    comp = ft._forest_beam(*args, fsb_delta, "bound", 8.0, None)
    Re = 1 << fsb_delta
    f = full[0].view(*full[0].shape[:2], 16, 4)
    assert comp[0].shape[-1] == 16 * Re
    assert torch.equal(f[..., :Re].reshape(comp[0].shape), comp[0])
    assert bool((f[..., Re:] == -1).all())
    assert torch.equal(full[1], comp[1])
    assert int((comp[0] >= 0).sum()) > 0


@gpu
@pytest.mark.parametrize("use_rows", [True, False])
def test_traverse_compact_matches_full(monkeypatch, use_rows):
    # traverse_forest with delta_eff (compact beams) == without it: read cells, provenance, visits and read counts,
    # through the rows re-rank and through union_candidates + rerank_top
    monkeypatch.setenv("FSB_FAST_ROWS", "1")
    st = _fixed_beam_inputs(17, 2048, 0)
    F, SP = st["F"], st["SP"]
    M = st["ceil"].shape[1]
    F["cellK"], F["cellA"] = st["kbar"], st["ceil"].contiguous()
    F2 = ft.build_forest(st["kbar"], st["ceil"], st["ceil"] > st["ceil"].median())
    m_b2 = torch.searchsorted(F2["adm"].masked_fill(F2["adm"] < 0, M),
                              (st["blk"] / 32).floor().long()[None].expand(4, 8).contiguous()).reshape(-1)
    SP2 = ft.build_spines(F2, m_b2, SP["t_idx"])
    h2 = SP2["height"]
    du = (h2 - (h2 - 0).clamp(min=0))[:, None].expand(st["q"].shape[:2])
    out = {}
    for d_eff in (None, 0):
        monkeypatch.setenv("FSB_CHECK_COMPACT", "1")
        pl = []
        res = ft.traverse_forest(F, SP, st["q"], st["qpos"], st["lam"], st["sigv"], 0.125, 32, st["W"], st["delta"],
                                 16, 2, "bound", radius_coef=8.0, read_top=4, rerank_score="lse_pos",
                                 tokK=st["keys"].unsqueeze(0), union_width=16, union_W_max=16, union_F=F2,
                                 union_SP=SP2, union_delta=du, prov_out=pl, rows=use_rows, delta_eff=d_eff)
        out[d_eff] = (*res, pl[0])
    for a, b in zip(out[None], out[0]):
        assert torch.equal(a, b)
    assert int((out[0][0] >= 0).sum()) > 0


@gpu
def test_compact_beams_whole_prefill(monkeypatch):
    # the FS-B serve config end to end: compact beams (checked against the full layout at every stage) give every
    # layer's selection and the hidden states of the full layout
    m, x = _tiny_fsb(2048)
    out = {}
    for compact in ("0", "1"):
        monkeypatch.setenv("FSB_BEAM_COMPACT", compact)
        monkeypatch.setenv("FSB_CHECK_COMPACT", compact)
        out[compact] = _prefill_sel(m, x, monkeypatch)
    assert all(torch.equal(a, b) for a, b in zip(out["0"][0], out["1"][0]))
    assert torch.equal(out["0"][1], out["1"][1])


# fsb_init only guarantees fsb_read_top <= W_max * 2^Delta_max. The first five configurations would leave the compact
# beams fewer slots, (W_max + union W_max) * 2^fsb_delta, than fsb_read_top: rerank_top then returned fewer columns
# than the read and the read margin's trim failed, so they keep the full layout. The last two hold exactly
# fsb_read_top compact slots (with a union beam; with fsb_delta 1) and compact.
_FEW_SLOTS = {
    "w4_rt8_leaf": (False, dict(fsb_width=4, fsb_read_top=8, fsb_rerank_score="leaf")),
    "w4_rt8_lsepos": (False, dict(fsb_width=4, fsb_read_top=8, fsb_rerank_score="lse_pos")),
    "w4_rt8_lsepos_margin": (False, dict(fsb_width=4, fsb_read_top=8, fsb_rerank_score="lse_pos",
                                         fsb_read_margin=0.5)),
    "w8_rt16_lse": (False, dict(fsb_width=8, fsb_read_top=16, fsb_rerank_score="lse")),
    "w4_u2_rt8_lsepos": (False, dict(fsb_width=4, fsb_read_top=8, fsb_rerank_score="lse_pos", fsb_union_width=2)),
    "w4_u4_rt8_leaf": (True, dict(fsb_width=4, fsb_read_top=8, fsb_rerank_score="leaf", fsb_union_width=4)),
    "w4_d1_rt8_margin": (True, dict(fsb_width=4, fsb_delta=1, fsb_read_top=8, fsb_rerank_score="lse_pos",
                                    fsb_read_margin=0.5)),
}


def _tiny_cfg(N, dev, **over):
    # a tiny random fixed-policy forest model with the given beam and read options, admission threshold 0
    from experiments.scaling.model import FreeSoloLM
    kw = dict(window_size=64, long_range_variant="cell_topk", cell_stride=32, cell_topk=4, cell_select="score",
              sel_strict_causal=1, use_rope=True, fs_uniform_sr=True, rope_long_range=False, cell_prune_kernel=0,
              fsb_enabled=True, fsb_policy="fixed", fsb_delta=0, fsb_node_score="bound", fsb_bound_radius_coef=8.0,
              fsb_tree="forest", fsb_rerank_query="raw", fsb_prefill_blkchunk=2)
    kw.update(over)
    torch.manual_seed(0)
    m = FreeSoloLM(vocab_size=256, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_seq_len=N + 64,
                   attn_type="fsattn", **kw).to(dev).eval()
    for mod in m.modules():
        if getattr(mod, "fsb_enabled", False):
            mod.fsb_tau.fill_(0.0)
    return m, torch.randint(0, 256, (1, N), generator=torch.Generator().manual_seed(N)).to(dev)


@pytest.mark.parametrize("setting", ["gpu", "gpu_fast_rows_0", "gpu_overflow", "cpu"])
@pytest.mark.parametrize("name", list(_FEW_SLOTS))
def test_compact_beams_keep_read_top_slots(monkeypatch, name, setting):
    # compact beams only where they hold fsb_read_top slots: every configuration runs and gives the full layout's
    # selections and hidden states, through the rows re-rank, rerank_top (FSB_FAST_ROWS=0, the leaf score, the read
    # margin), the deferred-overflow recompute (UCAP 2: every rows stage overflows) and on the CPU
    if setting != "cpu" and not (torch.cuda.is_available() and rows._HAVE_TRITON):
        pytest.skip("needs CUDA and Triton")
    compacts, over = _FEW_SLOTS[name]
    if setting == "gpu_fast_rows_0":
        monkeypatch.setenv("FSB_FAST_ROWS", "0")
    elif setting == "gpu_overflow":
        monkeypatch.setattr(rows, "UCAP", 2)
    m, x = _tiny_cfg(1024, "cpu" if setting == "cpu" else "cuda", **over)
    layer = next(mod for mod in m.modules() if getattr(mod, "fsb_enabled", False))
    out = {}
    for compact in ("0", "1"):
        monkeypatch.setenv("FSB_BEAM_COMPACT", compact)
        monkeypatch.setenv("FSB_CHECK_COMPACT", compact)
        assert (ft.beam_delta_eff(layer) is not None) == (compacts and compact == "1")
        out[compact] = _prefill_sel(m, x, monkeypatch)
    assert len(out["1"][0]) == 2 and all(int((s_ >= 0).sum()) > 0 for s_ in out["1"][0])
    assert all(torch.equal(a, b) for a, b in zip(out["0"][0], out["1"][0]))
    assert torch.equal(out["0"][1], out["1"][1])


@gpu
def test_deferred_flags_event_on_flag_stream(monkeypatch):
    # the host waits for the deferred capacity flags' pinned copy on an event recorded on the stream that runs the
    # copy: the current stream of the flags' (the model's) device, which need not be the current device. The stage
    # pipeline's fork/join may record first on its side stream; the flags' copy event is the last record of a
    # prefill, on the model stream
    monkeypatch.setattr(rows, "UCAP", 8)
    m, x = _tiny_fsb(2048)
    seen = []

    class _Event(torch.cuda.Event):
        def record(self, stream=None):
            seen.append(stream)
            return super().record(stream)
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    _prefill_sel(m, x, monkeypatch)
    assert len(seen) >= 2 and all(s_ is not None for s_ in seen)
    assert seen[-1] == torch.cuda.current_stream(x.device)


def _rows_call(st, score="lse_pos", prov=True, **kw):
    return rows.union_rerank_rows(st["sel_a"], st["sel_b"], st["q"], st["qpos"], st["lam"], st["sigv"], st["cellA"],
                                  st["t_idx"], st.get("scale", 0.125), st["s"], st.get("K", 4), score, st["tokK"],
                                  prov=prov, **kw)


def _lse_table(st, score, qbf16=False):
    # the rows LSE kernel's (row, cell, query) table of a stage, with the legacy union and list
    sel = ft.union_candidates(st["sel_a"], st["sel_b"])
    Tp, Bq, _ = sel.shape
    cells, cnt, _ = rows._row_union(sel, st["cellA"].shape[1], rows.UCAP)
    K4 = st["tokK"] if st["tokK"].dim() == 4 else st["tokK"].unsqueeze(0)
    lse = torch.full((Tp, rows.UCAP, 128), float("nan"), device="cuda")
    rows._rows_lse_kernel[(Tp,)](st["q"].reshape(Tp * Bq, -1).contiguous(), cells, cnt, K4, lse,
                                 st["t_idx"].contiguous(), Bq, rows.UCAP, K4.shape[1], K4.stride(0), K4.stride(1),
                                 K4.stride(2), float(st.get("scale", 0.125)), st["s"], BQ=128, DH=st["q"].shape[-1],
                                 DHP=64, SP=32, LSE_MODE=(score != "max"), LOSKIP=rows.rows_loskip_on(), QBF16=qbf16,
                                 num_warps=4)
    return lse, cnt


@gpu
@pytest.mark.parametrize("score", ["lse_pos", "max"])
@pytest.mark.parametrize("qkind", ["bf16", "fp32"])
def test_rows_loskip_matches(monkeypatch, score, qkind):
    # the LSE kernel without the low-part products of bf16-exact queries gives the same scores (up to the sign of a
    # zero) and the same read cells and provenance; non-bf16 queries keep the low part
    st = _stage(5, 2048)
    if qkind == "fp32":
        st["q"] = st["q"] + 1e-3 * torch.randn(st["q"].shape, generator=torch.Generator().manual_seed(3)).cuda()
    out, lse = {}, {}
    for v in ("0", "1"):
        monkeypatch.setenv("FSB_ROWS_LOSKIP", v)
        out[v] = _rows_call(st, score)
        lse[v], cnt = _lse_table(st, score)

    assert torch.equal(out["0"][0], out["1"][0]) and torch.equal(out["0"][1], out["1"][1])
    filled = torch.arange(rows.UCAP, device="cuda")[None, :] < cnt[:, None].long()
    a, b = lse["0"][filled], lse["1"][filled]
    assert bool((a == b).all())
    if qkind == "fp32":
        assert torch.equal(a, b)
    if qkind == "bf16":                     # the caller's bf16 guarantee: the kernel built without a low part
        assert _same(out["0"], _rows_call(st, score, q_bf16=True))
        q16, _ = _lse_table(st, score, qbf16=True)
        assert torch.equal(lse["1"][filled].view(torch.int32), q16[filled].view(torch.int32))


def _captures():
    d = os.environ.get("FSB_ROWS_CAPTURES", "")
    if not d or not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.startswith("stage_"))


def _load_capture(name):
    d = os.environ["FSB_ROWS_CAPTURES"]
    st = torch.load(os.path.join(d, name), weights_only=True)
    st = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in st.items()}
    st["tokK"] = torch.load(os.path.join(d, st.pop("tokK_file")), weights_only=True).cuda()
    return st


@gpu
@pytest.mark.skipif(not _captures(), reason="set FSB_ROWS_CAPTURES to a directory of captured prefill stages")
@pytest.mark.parametrize("name", _captures())
def test_rows_loskip_on_captured_stages(monkeypatch, name):
    # captured FS-B bench-config prefill stages (random tokens and fineweb text, 16k-131k): identical read cells and
    # provenance with and without the low-part products
    st = _load_capture(name)
    out = {}
    for v in ("0", "1"):
        monkeypatch.setenv("FSB_ROWS_LOSKIP", v)
        out[v] = _rows_call(st, st["score"], prov=st["prov"])
    assert torch.equal(out["0"][0], out["1"][0])
    assert (out["0"][1] is None and out["1"][1] is None) or torch.equal(out["0"][1], out["1"][1])


def _full_layout(sel, R=4):
    # the compact beam layout (one cell slot per frontier slot) -> the 2^delta_max layout (R slots, the rest -1)
    full = torch.full((*sel.shape, R), -1, dtype=sel.dtype, device=sel.device)
    full[..., 0] = sel
    return full.reshape(*sel.shape[:-1], -1)


def _impls(monkeypatch, st, score="lse_pos", prov=True, splits=(1, 2, 3), **kw):
    # legacy rows path, then the CUDA union + row select path at several LSE splits: every result must equal legacy's
    monkeypatch.setenv("FSB_ROWS_IMPL", "legacy")
    ref = _rows_call(st, score, prov=prov, **kw)
    monkeypatch.setenv("FSB_ROWS_IMPL", "cuda")
    outs = []
    for p in splits:
        monkeypatch.setenv("FSB_ROWS_LSE_SPLIT", str(p))
        outs.append(_rows_call(st, score, prov=prov, **kw))
    return ref, outs


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return torch.equal(a[0], b[0]) and ((a[1] is None and b[1] is None) or torch.equal(a[1], b[1]))


def _layout(st, layout):
    # the stage's 64 + 64 beam slots as the compact (16 + 16) or full layout, with or without the second beam
    if layout.startswith("compact"):
        st["sel_a"] = st["sel_a"].view(*st["sel_a"].shape[:2], 16, 4)[..., 0].contiguous()
        st["sel_b"] = st["sel_b"].view(*st["sel_b"].shape[:2], 16, 4)[..., 0].contiguous()
    if layout.endswith("a_only"):
        st["sel_b"] = None
    return st


def _nvrtc_ok(M):
    from src.fsattn import fsb_rows_cuda as rc
    return rc.rows_union_ok(torch.zeros(1, 1, 2, dtype=torch.long, device="cuda"), None, M)


@gpu
@pytest.mark.parametrize("score", ["lse_pos", "lse", "max"])
@pytest.mark.parametrize("M,layout", [(512, "compact"), (2048, "compact"), (2048, "full"), (512, "a_only"),
                                      (2048, "compact_a_only")])
def test_rows_cuda_matches_legacy(monkeypatch, score, M, layout):
    # FSB_ROWS_IMPL=cuda == the legacy rows path, bitwise (read cells and provenance), on compact (16 + 16 slots: the
    # row-tile select) and full (64 + 64: the per-query select) beam layouts and without a second beam
    if not _nvrtc_ok(M):
        pytest.skip("NVRTC unavailable")
    st = _layout(_stage(23 + M, M), layout)
    ref, outs = _impls(monkeypatch, st, score, prov=not layout.endswith("a_only"))
    assert int((ref[0] >= 0).sum()) > 0
    for o in outs:
        assert _same(ref, o)


class _LaunchSpy:
    # stands in for a Triton kernel: records kernel[grid] launches, then launches the kernel
    def __init__(self, kern, tag, seen):
        self.kern, self.tag, self.seen = kern, tag, seen

    def __getitem__(self, grid):
        self.seen.append(self.tag)
        return self.kern[grid]


def test_rows_tile_gate():
    # the row-tile select only on (query x slot) tiles of up to 4096 entries, where it measured faster than the
    # per-query select: 128 queries x 32 slots (the compact 16 + 16 beams), not 128 x 64 or 128 x 128 (the full 64 + 64
    # layout); short blocks round up to 16 queries
    assert rows.rows_tile_ok(128, 32) and rows.rows_tile_ok(100, 32) and rows.rows_tile_ok(64, 64)
    assert rows.rows_tile_ok(256, 16) and rows.rows_tile_ok(2, 128) and rows.rows_tile_ok(128, 17)
    assert not rows.rows_tile_ok(128, 64) and not rows.rows_tile_ok(128, 128) and not rows.rows_tile_ok(256, 32)
    assert not rows.rows_tile_ok(64, 80) and not rows.rows_tile_ok(128, 33)


@gpu
@pytest.mark.parametrize("impl,layout,select", [("cuda", "compact", "tile"), ("cuda", "compact_a_only", "tile"),
                                                ("cuda", "full", "query"), ("cuda", "a_only", "query"),
                                                ("legacy", "compact", "query"), ("legacy", "full", "query")])
def test_rows_select_path(monkeypatch, impl, layout, select):
    # which select kernel a stage launches: the row-tile select for the compact layout, the per-query select for
    # the full layout (after the CUDA row union and LSE kernel) and always on the legacy path
    if impl == "cuda" and not _nvrtc_ok(2048):
        pytest.skip("NVRTC unavailable")
    monkeypatch.setenv("FSB_ROWS_IMPL", impl)
    st = _layout(_stage(29, 2048), layout)
    seen = []
    monkeypatch.setattr(rows, "_rows_select_rows_kernel", _LaunchSpy(rows._rows_select_rows_kernel, "tile", seen))
    monkeypatch.setattr(rows, "_rows_select_kernel", _LaunchSpy(rows._rows_select_kernel, "query", seen))
    assert _rows_call(st, prov=not layout.endswith("a_only")) is not None
    assert seen == [select]


@gpu
@pytest.mark.parametrize("select", ["tile", "query"])
@pytest.mark.parametrize("layout", ["compact", "full", "a_only", "compact_a_only"])
def test_rows_cuda_select_paths_match_legacy(monkeypatch, select, layout):
    # either select after the CUDA row union and split LSE kernel, forced on every layout, == the legacy rows path
    # bitwise, for every score and with overflowing rows refused alike
    if not _nvrtc_ok(2048):
        pytest.skip("NVRTC unavailable")
    monkeypatch.setattr(rows, "rows_tile_ok", lambda Bq, C: select == "tile")
    st = _layout(_stage(37, 2048), layout)
    for score in ("lse_pos", "lse", "max"):
        ref, outs = _impls(monkeypatch, st, score, prov=not layout.endswith("a_only"), splits=(1, 4))
        assert int((ref[0] >= 0).sum()) > 0
        for o in outs:
            assert _same(ref, o)
    monkeypatch.setattr(rows, "UCAP", 8)
    ref, outs = _impls(monkeypatch, st, splits=(4,), prov=not layout.endswith("a_only"))
    assert ref is None and outs == [None]


@gpu
@pytest.mark.parametrize("score", ["lse_pos", "lse", "max"])
def test_rows_fused_cuda_strided_stage_view(monkeypatch, score):
    # the fused union+rerank kernel on the prefill stage loop's strided query view (q_nq real rows in a wider
    # buffer) == itself on the zero-padded packed copy, bitwise: rows at and beyond q_nq must read as a zero query.
    # The pad region is poisoned with NaNs so an unguarded read changes that row's scores
    from src.fsattn import fsb_rows_cuda
    st = _stage(17, 512, H=8)
    st["sel_a"] = st["sel_a"].view(*st["sel_a"].shape[:2], 16, 4)[..., 0].contiguous()
    st["sel_b"] = st["sel_b"].view(*st["sel_b"].shape[:2], 16, 4)[..., 0].contiguous()
    Tp, Bq, Dh = st["q"].shape
    nq = 100
    qbig = torch.randn(Tp, 3 * Bq, Dh, generator=torch.Generator().manual_seed(5)).to("cuda", torch.bfloat16)
    qbig[:, nq:Bq].fill_(float("nan"))
    qref = torch.zeros(Tp, Bq, Dh, device="cuda", dtype=torch.bfloat16)
    qref[:, :nq] = qbig[:, :nq]
    calls = []
    real = fsb_rows_cuda.rows_fused_cuda
    monkeypatch.setattr(fsb_rows_cuda, "rows_fused_cuda", lambda *a, **k: calls.append(1) or real(*a, **k))
    ref = rows.union_rerank_rows(st["sel_a"], st["sel_b"], qref, st["qpos"], st["lam"], st["sigv"], st["cellA"],
                                 st["t_idx"], 0.125, st["s"], 4, score, st["tokK"], prov=True)
    got = rows.union_rerank_rows(st["sel_a"], st["sel_b"], qbig[:, :Bq], st["qpos"], st["lam"], st["sigv"],
                                 st["cellA"], st["t_idx"], 0.125, st["s"], 4, score, st["tokK"], prov=True, q_nq=nq)
    assert ref is not None and got is not None and len(calls) == 2
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


@gpu
def test_rows_cuda_ties_follow_slot_order(monkeypatch):
    # cells with identical keys and no position term tie exactly: both paths keep the first slot's cell, so they
    # agree bitwise and the tie-break is exercised (some query's 4th and 5th scores are equal)
    st = _stage(41, 512)
    st["sel_a"] = st["sel_a"].view(*st["sel_a"].shape[:2], 16, 4)[..., 0].contiguous()
    st["sel_b"] = st["sel_b"].view(*st["sel_b"].shape[:2], 16, 4)[..., 0].contiguous()
    tk = st["tokK"].clone()
    H, s = tk.shape[1], st["s"]
    kv = tk.view(1, H, -1, s, tk.shape[-1])
    kv[:, :, 1::2] = kv[:, :, 0:-1:2]                        # cell 2i + 1 has cell 2i's keys
    st["tokK"] = tk
    ref, outs = _impls(monkeypatch, st, "lse")
    for o in outs:
        assert _same(ref, o)
    sel = ft.union_candidates(st["sel_a"], st["sel_b"])
    from src.fsattn.fsb_rerank import cell_token_scores
    u = cell_token_scores(st["q"], sel, st["tokK"], st["t_idx"], 0.125, s, "lse").masked_fill(sel < 0, float("-inf"))
    top = u.topk(5, dim=-1).values
    assert bool((top[..., 3] == top[..., 4]).any())


@gpu
@pytest.mark.parametrize("ucap", [8, 40, 512])
def test_rows_cuda_capacity_flag(monkeypatch, ucap):
    # rows over capacity: both paths refuse the stage (None) without a flag list, and raise the same deferred flag
    # with one; within capacity they agree bitwise
    monkeypatch.setattr(rows, "UCAP", ucap)
    st = _stage(43, 2048)
    st["sel_a"] = st["sel_a"].view(*st["sel_a"].shape[:2], 16, 4)[..., 0].contiguous()
    st["sel_b"] = st["sel_b"].view(*st["sel_b"].shape[:2], 16, 4)[..., 0].contiguous()
    monkeypatch.setenv("FSB_ROWS_DEFER", "1")
    ref, outs = _impls(monkeypatch, st, splits=(2,))
    fl = {}
    for impl in ("legacy", "cuda"):
        monkeypatch.setenv("FSB_ROWS_IMPL", impl)
        f = []
        r_ = _rows_call(st, ovf_out=f)
        fl[impl] = bool(f[0])
        if not fl[impl]:
            assert _same(ref, r_)
    assert fl["legacy"] == fl["cuda"] == (ref is None)
    assert _same(ref, outs[0])
    if ucap == 8:
        assert ref is None


@gpu
def test_rows_union_cuda_tables(monkeypatch):
    # the CUDA row union's list, count, positions and duplicate bits == the legacy union kernel + torch glue
    from src.fsattn import fsb_rows_cuda as rc
    st = _stage(47, 2048)
    a = st["sel_a"].view(*st["sel_a"].shape[:2], 16, 4)[..., 0].contiguous()
    b = st["sel_b"].view(*st["sel_b"].shape[:2], 16, 4)[..., 0].contiguous()
    b[:, :, 3] = a[:, :, 0]                                   # forced repeats of beam a's cells
    Mc = st["cellA"].shape[1]
    un = rc.rows_union_cuda(a, b, Mc, rows.UCAP)
    if un is None:
        pytest.skip("NVRTC unavailable")
    cells, cnt, pos, dup, ovf = un
    sel = ft.union_candidates(a, b)
    rcells, rcnt, rpos = rows._row_union(sel, Mc, rows.UCAP)
    assert torch.equal(cnt, rcnt) and not bool(ovf)
    filled = torch.arange(rows.UCAP, device="cuda")[None, :] < cnt[:, None]
    assert torch.equal(cells[filled], rcells[filled])
    present = torch.zeros(sel.shape[0], Mc, dtype=torch.bool, device="cuda")
    present.scatter_(1, sel.reshape(sel.shape[0], -1).clamp(min=0), sel.reshape(sel.shape[0], -1) >= 0)
    assert torch.equal(pos[present], rpos[present])
    bits = (dup[..., None] >> torch.arange(16, device="cuda")) & 1
    assert torch.equal(bits.bool(), (sel[..., 16:] < 0) & (b >= 0))


@gpu
@pytest.mark.skipif(not _captures(), reason="set FSB_ROWS_CAPTURES to a directory of captured prefill stages")
@pytest.mark.parametrize("name", _captures())
def test_rows_cuda_on_captured_stages(monkeypatch, name):
    # captured FS-B bench-config prefill stages (random tokens and fineweb text, 16k-131k): the CUDA union + row
    # select give legacy's read cells and provenance, bitwise; and no beam repeats a cell within a query's slots
    st = _load_capture(name)
    for x in (st["sel_a"], st["sel_b"]):
        srt = x.sort(-1).values
        assert not bool(((srt[..., 1:] == srt[..., :-1]) & (srt[..., 1:] >= 0)).any())
    ref, outs = _impls(monkeypatch, st, st["score"], prov=st["prov"], splits=(1, 2))
    for o in outs:
        assert _same(ref, o)
    assert bool((st["q"] == st["q"].bfloat16().float()).all())          # eager prefill's re-rank query: bf16 values
    assert _same(ref, _rows_call(st, st["score"], prov=st["prov"], q_bf16=True))


def _v3_ok():
    from src.fsattn.fsb_beam_cuda import _sm80
    return _nvrtc_ok(2048) and _sm80(torch.device("cuda", torch.cuda.current_device()))


def _bf16_stage(st):
    # the fused re-rank kernels take bf16 queries (the prefill stage loop's Q; _stage's queries are bf16-exact fp32) and
    # at least 8 heads (the stages below use H=8)
    st["q"] = st["q"].to(torch.bfloat16)
    return st


def _rows_v3_pair(monkeypatch, st, score="lse_pos", prov=True, **kw):
    # union_rerank_rows on the fused path with fsb_rows_fused2 (FSB_ROWS_V3=0) and with fsb_rows_v3 (=1); asserts that
    # each ran the kernel it should
    from src.fsattn import fsb_rows_cuda as rc
    seen = []
    real = rc._v3_function
    monkeypatch.setattr(rc, "_v3_function", lambda *a: seen.append(1) or real(*a))
    out = {}
    for v in ("0", "1"):
        monkeypatch.setenv("FSB_ROWS_V3", v)
        seen.clear()
        kv = {k: (x.clone() if torch.is_tensor(x) else x) for k, x in kw.items()}       # fresh output buffers
        res = _rows_call(st, score, prov=prov, **kv)
        out[v] = None if res is None else tuple(None if x is None else x.clone() for x in res)
        assert seen == ([1] if v == "1" else [])
    assert rc._v3_fns and not rc._v3_failed
    return out["0"], out["1"]


@gpu
@pytest.mark.parametrize("score", ["lse_pos", "lse", "max"])
@pytest.mark.parametrize("M,layout", [(512, "compact"), (2048, "compact"), (2048, "compact_a_only")])
@pytest.mark.parametrize("prov", [True, False])
def test_rows_v3_matches_fused2(monkeypatch, score, M, layout, prov):
    # FSB_ROWS_V3=1 (fsb_rows_v3) == fsb_rows_fused2 bit for bit: read cells and provenance, every score, the compact
    # 16 + 16 layout and beam a alone, provenance on and off
    if not _v3_ok():
        pytest.skip("NVRTC unavailable")
    st = _bf16_stage(_layout(_stage(53 + M, M, H=8), layout))
    ref, got = _rows_v3_pair(monkeypatch, st, score, prov=prov and not layout.endswith("a_only"))
    assert ref is not None and int((ref[0] >= 0).sum()) > 0
    assert _same(ref, got)


@gpu
@pytest.mark.parametrize("K", [1, 3, 8])
@pytest.mark.parametrize("width", [4, 10])
def test_rows_v3_slot_counts(monkeypatch, K, width):
    # other read-top counts and beam widths (narrower beams leave fewer slots per query): v3 == fused2, int32 and int64
    # outputs
    if not _v3_ok():
        pytest.skip("NVRTC unavailable")
    st = _bf16_stage(_layout(_stage(61 + width, 2048, H=8), "compact"))
    st["sel_a"] = st["sel_a"][..., :width].contiguous()
    st["sel_b"] = st["sel_b"][..., :width].contiguous()
    st["K"] = K
    for dt in (torch.int64, torch.int32):
        out = torch.empty(st["sel_a"].shape[:2] + (K,), dtype=dt, device="cuda")
        pv = torch.empty(st["sel_a"].shape[:2] + (K,), dtype=torch.int8, device="cuda")
        ref, got = _rows_v3_pair(monkeypatch, st, prov=True, out=out.clone(), pv_out=pv.clone())
        assert _same(ref, got)


@gpu
@pytest.mark.parametrize("score", ["lse_pos", "lse"])
def test_rows_v3_strided_stage_view(monkeypatch, score):
    # fsb_rows_v3 on the prefill stage loop's strided query view (q_nq real rows in a wider buffer, the pad rows
    # poisoned with NaNs) == fsb_rows_fused2 on the zero-padded packed copy, bitwise
    if not _v3_ok():
        pytest.skip("NVRTC unavailable")
    st = _layout(_stage(17, 512, H=8), "compact")
    Tp, Bq, Dh = st["q"].shape
    nq = 100
    qbig = torch.randn(Tp, 3 * Bq, Dh, generator=torch.Generator().manual_seed(5)).to("cuda", torch.bfloat16)
    qbig[:, nq:Bq].fill_(float("nan"))
    qref = torch.zeros(Tp, Bq, Dh, device="cuda", dtype=torch.bfloat16)
    qref[:, :nq] = qbig[:, :nq]
    args = (st["sel_a"], st["sel_b"])
    rest = (st["qpos"], st["lam"], st["sigv"], st["cellA"], st["t_idx"], 0.125, st["s"], 4, score, st["tokK"])
    monkeypatch.setenv("FSB_ROWS_V3", "0")
    ref = rows.union_rerank_rows(*args, qref, *rest, prov=True)
    monkeypatch.setenv("FSB_ROWS_V3", "1")
    got = rows.union_rerank_rows(*args, qbig[:, :Bq], *rest, prov=True, q_nq=nq)
    assert ref is not None and got is not None
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


@gpu
@pytest.mark.parametrize("score", ["lse_pos", "lse"])
@pytest.mark.parametrize("M,pad", [(512, 0), (512, 45), (2048, 0)])
def test_rows_v3_head_major_stage_views(monkeypatch, score, M, pad):
    # FSB_PREFILL_NOCOPY's views at the released shape (16 heads of 64, top 4 cells, beams of 16 + 16 slots): row
    # t * nbc + j (block j of head t) reads its queries in place from Q's (B, N, H, D)-transposed layout, positions past
    # the stage's last real query (pad of them, poisoned with NaNs beyond the view) read zeros, and the int32 cells and
    # provenance land straight in the layer's (B, H, Npad, K) tables. fsb_rows_v3 == fsb_rows_fused2 on these views ==
    # both on the packed copies, bitwise, and neither touches the tables outside the stage
    from src.fsattn import fsb_beam_cuda as bc
    from src.fsattn import fsb_rows_cuda as rc
    if not _v3_ok():
        pytest.skip("NVRTC unavailable")
    H, nbc, K = 16, 4, 4
    st = _bf16_stage(_layout(_stage(83 + M + pad, M, H=H, nbc=nbc), "compact"))
    Tp, Bq, Dh = st["q"].shape
    nq, q0, Npad = nbc * Bq - pad, 2 * Bq, 8 * Bq
    heads = st["q"].view(H, nbc * Bq, Dh)                         # head t's stage queries, block j at j * Bq
    base = torch.full((1, Npad, H, Dh), float("nan"), device="cuda", dtype=torch.bfloat16)
    base[0, q0:q0 + nq] = heads[:, :nq].transpose(0, 1)
    Q = base.transpose(1, 2)                                      # (B, H, N, D) view of (B, N, H, D)
    qview = Q.as_strided((H, nq, Dh), (Q.stride(1), Q.stride(2), 1), Q.storage_offset() + q0 * Q.stride(2))
    qpack = torch.zeros(H, nbc * Bq, Dh, device="cuda", dtype=torch.bfloat16)
    qpack[:, :nq] = heads[:, :nq]
    qpack = qpack.view(Tp, Bq, Dh)
    packs = []
    real_pack = bc.packed_queries
    monkeypatch.setattr(bc, "packed_queries", lambda *a, **k: packs.append(1) or real_pack(*a, **k))
    rest = (st["qpos"], st["lam"], st["sigv"], st["cellA"], st["t_idx"], 0.125, st["s"], K, score, st["tokK"])
    res = {}
    for v3 in ("0", "1"):
        monkeypatch.setenv("FSB_ROWS_V3", v3)
        sel32 = torch.full((1, H, Npad, K), -9, dtype=torch.int32, device="cuda")
        prov = torch.full((1, H, Npad, K), -9, dtype=torch.int8, device="cuda")
        out_v = sel32[:, :, q0:q0 + nbc * Bq].view(H, nbc * Bq, K)
        pv_v = prov[:, :, q0:q0 + nbc * Bq].view(H, nbc * Bq, K)
        packs.clear()
        got = rows.union_rerank_rows(st["sel_a"], st["sel_b"], qview, *rest, prov=True, q_bf16=True, out=out_v,
                                     pv_out=pv_v, q_nq=nq)
        assert got is not None and got[0] is out_v and not packs     # the fused kernel read and wrote the views
        assert bool((sel32[:, :, :q0] == -9).all()) and bool((sel32[:, :, q0 + nbc * Bq:] == -9).all())
        assert bool((prov[:, :, :q0] == -9).all()) and bool((prov[:, :, q0 + nbc * Bq:] == -9).all())
        pk = rows.union_rerank_rows(st["sel_a"], st["sel_b"], qpack, *rest, prov=True, q_bf16=True,
                                    out=torch.empty((Tp, Bq, K), dtype=torch.int32, device="cuda"))
        assert torch.equal(out_v.reshape(Tp, Bq, K), pk[0]) and torch.equal(pv_v.reshape(Tp, Bq, K), pk[1])
        res[v3] = (sel32, prov)
    assert rc._v3_fns and not rc._v3_failed
    assert int((res["1"][0] >= 0).sum()) > 0
    assert torch.equal(res["0"][0], res["1"][0]) and torch.equal(res["0"][1], res["1"][1])


@gpu
def test_rows_v3_ties_follow_slot_order(monkeypatch):
    # cells with identical keys tie exactly ("lse": no position term); v3 keeps fused2's first-slot winner
    if not _v3_ok():
        pytest.skip("NVRTC unavailable")
    st = _bf16_stage(_layout(_stage(41, 512, H=8), "compact"))
    tk = st["tokK"].clone()
    kv = tk.view(1, tk.shape[1], -1, st["s"], tk.shape[-1])
    kv[:, :, 1::2] = kv[:, :, 0:-1:2]
    st["tokK"] = tk
    ref, got = _rows_v3_pair(monkeypatch, st, "lse")
    sel = ft.union_candidates(st["sel_a"], st["sel_b"])
    u = cell_token_scores(st["q"].float(), sel, st["tokK"], st["t_idx"], 0.125, st["s"], "lse").masked_fill(sel < 0,
                                                                                                         float("-inf"))
    top = u.topk(5, dim=-1).values
    assert bool((top[..., 3] == top[..., 4]).any())                    # the tie-break is exercised
    assert _same(ref, got)


@gpu
@pytest.mark.parametrize("ucap", [8, 40, 512])
def test_rows_v3_capacity_flag(monkeypatch, ucap):
    # rows over capacity raise the deferred flag exactly as fused2 does (the stage is then redone off the fast path);
    # within capacity the cells agree bitwise
    if not _v3_ok():
        pytest.skip("NVRTC unavailable")
    from src.fsattn import fsb_rows_cuda as rc
    monkeypatch.setattr(rows, "UCAP", ucap)
    monkeypatch.setenv("FSB_ROWS_DEFER", "1")
    st = _bf16_stage(_layout(_stage(43, 2048, H=8), "compact"))
    seen = []
    real = rc.rows_fused_cuda
    monkeypatch.setattr(rc, "rows_fused_cuda", lambda *a, **k: seen.append(1) or real(*a, **k))
    res, fl = {}, {}
    for v in ("0", "1"):
        monkeypatch.setenv("FSB_ROWS_V3", v)
        f = []
        res[v] = _rows_call(st, ovf_out=f)
        fl[v] = bool(f[0])
    assert seen == [1, 1]
    assert fl["0"] == fl["1"]
    if not fl["0"]:
        assert _same(res["0"], res["1"])
    if ucap == 8:
        assert fl["1"]


@gpu
def test_rows_v3_graph_capture(monkeypatch):
    # a CUDA-graph capture of the stage's re-rank replays fsb_rows_v3 (compiled before the capture) with the eager
    # result
    if not _v3_ok():
        pytest.skip("NVRTC unavailable")
    from src.fsattn import fsb_rows_cuda as rc
    monkeypatch.setenv("FSB_ROWS_V3", "1")
    st = _bf16_stage(_layout(_stage(71, 2048, H=8), "compact"))
    Tp, Bq = st["sel_a"].shape[:2]
    eager = _rows_call(st, out=torch.empty((Tp, Bq, 4), dtype=torch.int32, device="cuda"),
                       pv_out=torch.empty((Tp, Bq, 4), dtype=torch.int8, device="cuda"))   # compiles the int32 config
    assert rc._v3_fns
    seen = []
    real = rc._v3_function
    monkeypatch.setattr(rc, "_v3_function", lambda *a: seen.append(1) or real(*a))
    out = torch.full((Tp, Bq, 4), -7, dtype=torch.int32, device="cuda")
    pv = torch.full((Tp, Bq, 4), -7, dtype=torch.int8, device="cuda")
    flags = []
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        res = rows.union_rerank_rows(st["sel_a"], st["sel_b"], st["q"], st["qpos"], st["lam"], st["sigv"],
                                     st["cellA"], st["t_idx"], 0.125, st["s"], 4, "lse_pos", st["tokK"],
                                     prov=True, out=out, pv_out=pv, ovf_out=flags)
    assert res is not None and seen == [1]
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.long(), eager[0].long()) and torch.equal(pv, eager[1])


@gpu
@pytest.mark.skipif(not _captures(), reason="set FSB_ROWS_CAPTURES to a directory of captured prefill stages")
@pytest.mark.parametrize("name", _captures())
def test_rows_v3_on_captured_stages(monkeypatch, name):
    # captured FS-B bench-config prefill stages: fsb_rows_v3 == fsb_rows_fused2, bitwise
    if not _v3_ok():
        pytest.skip("NVRTC unavailable")
    st = _load_capture(name)
    st["sel_a"], st["sel_b"] = st["sel_a"].contiguous(), st["sel_b"].contiguous()
    if st["sel_a"].shape[-1] + st["sel_b"].shape[-1] > 32:
        st = _layout(st, "compact")
    ref, got = _rows_v3_pair(monkeypatch, st, st["score"], prov=st["prov"])
    assert ref is not None and _same(ref, got)
