from pathlib import Path

import pytest
import torch

from src.fsattn import fsb_beam_cuda as bc
from src.fsattn import fsb_decode_cuda as dcu
from src.fsattn import fsb_fast_step as fast


def _nvrtc_ok():
    try:
        bc._libraries()
        return True
    except Exception:
        return False


gpu = pytest.mark.skipif(not (torch.cuda.is_available() and fast._HAVE_TRITON and _nvrtc_ok()),
                         reason="needs CUDA, Triton and NVRTC")


def _model(monkeypatch, L, n_layers=1, seed=41):
    # the released attention shape (16 heads of 64, cells of 32, beams of 16, four read cells, int8 long-range KV) in
    # a small random model with the released checkpoints' fp32 norms
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "scaling"))
    import bench_efficiency as be
    from experiments.scaling.model import FreeSoloLM
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_KV_INT8", "1")
    monkeypatch.setenv("FSB_DECODE_FORK", "0")         # the serial step; test_forked_read_matches_serial_step forks it
    torch.manual_seed(seed)
    m = FreeSoloLM(vocab_size=512, d_model=1024, n_layers=n_layers, n_heads=16, d_ff=1024, max_seq_len=L + 1024,
                   attn_type="fsattn", **be.serve_kwargs("fsb", 0.5)).cuda().to(torch.bfloat16).eval()
    for blk in m.blocks:
        blk.attn.fsattn.lr_norm.scale.data = blk.attn.fsattn.lr_norm.scale.data.float()
    return m


def _decoder(m, prompt, n_new):
    from decode_graph import FSBGraphedDecoder
    gd = FSBGraphedDecoder(m, prompt, n_new + 8, device="cuda", dtype=torch.bfloat16)
    gd.prefill()
    for blk, c in zip(m.blocks, gd.cache):
        f = blk.attn.fsattn
        f.fsb_tau.fill_(0.0)
        f.fsb_tau_union.copy_(c._ceil[0].float().median(dim=-1).values)       # the second forest: half the cells
    return gd


def _taus(m):
    # the FS-B thresholds _decoder sets after its prefill: restoring them before the next run keeps that run's prefill
    # (whose layers above the first read through the union forest) the same as the first run's
    saved = [(b.attn.fsattn.fsb_tau.clone(), b.attn.fsattn.fsb_tau_union.clone()) for b in m.blocks]

    def restore():
        for b, (t, tu) in zip(m.blocks, saved):
            b.attn.fsattn.fsb_tau.copy_(t)
            b.attn.fsattn.fsb_tau_union.copy_(tu)
    return restore


def _spy(monkeypatch):
    ran = []
    real = dcu.decode_read_fused
    monkeypatch.setattr(dcu, "decode_read_fused", lambda *a, **k: ran.append(real(*a, **k)) or ran[-1])
    return ran


@gpu
@torch.no_grad()
@pytest.mark.parametrize("L", [1024, 4096])
def test_fused_decode_matches_chain(monkeypatch, L):
    # teacher-forced whole-step graph decode across query blocks (forest updates, in-place slot refreshes): the fused
    # read reads exactly the chain's cells with the same provenance at every token; its readout differs from the chain's
    # only by reduction order
    m = _model(monkeypatch, L)
    prompt = torch.randint(0, 512, (1, L), generator=torch.Generator().manual_seed(43)).cuda()
    forced = torch.randint(0, 512, (300,), generator=torch.Generator().manual_seed(47)).cuda()
    runs = {}
    for v2 in ("0", "1"):
        monkeypatch.setenv("FSB_DECODE_V2", v2)
        ran = _spy(monkeypatch)
        gd = _decoder(m, prompt, len(forced))
        logits, sels, provs = [], [], []
        for t in forced:
            gd.next_tok = t.view(1)
            gd.step()
            fc = gd.cache[0]._fsb_forest
            logits.append(gd.static_logits.float().clone())
            sels.append(fc._last_sel.clone())
            provs.append(fc._last_prov.clone())
        assert (len(ran) > 0 and all(r is not None for r in ran)) == (v2 == "1"), (v2, len(ran))
        runs[v2] = (torch.stack(logits), torch.stack(sels), torch.stack(provs))
    (l0, s0, p0), (l1, s1, p1) = runs["0"], runs["1"]
    assert torch.equal(s0, s1)
    assert torch.equal(p0, p1)
    assert (s0 >= 0).float().mean() > 0.9 and bool((p0 > 0).any())
    rel = ((l1 - l0).norm(dim=-1) / l0.norm(dim=-1)).max().item()
    assert rel < 1e-2, rel
    assert (l1.argmax(-1) == l0.argmax(-1)).float().mean().item() >= 0.99


@gpu
@torch.no_grad()
def test_fused_readout_matches_reference_released_shape(monkeypatch):
    # the fused readout (without the window combine) at the released shape against the torch reference readout of the
    # same cells and against the chain's fused readout, for queries spread over the scale of the real ones
    from src.fsattn.fsb_tree import _fsb_decode_readout, fsb_decode_token
    m = _model(monkeypatch, 4096)
    prompt = torch.randint(0, 512, (1, 4096), generator=torch.Generator().manual_seed(53)).cuda()
    monkeypatch.setenv("FSB_DECODE_V2", "1")
    gd = _decoder(m, prompt, 16)
    for _ in range(3):
        gd.step()
    attn, cache = gd.attn[0], gd.cache[0]
    fc = cache._fsb_forest
    pos = gd.pos
    pos_t = torch.full((), pos, dtype=torch.int64, device="cuda")
    pos_f = torch.full((1,), float(pos), device="cuda")
    g = torch.Generator(device="cuda").manual_seed(59)
    ran = _spy(monkeypatch)
    for scale in (0.5, 2.0, 8.0):
        q_t = (torch.randn(1, 16, 1, 64, device="cuda", generator=g) * scale).to(torch.bfloat16)
        sig = fc._tok["sigma"]
        monkeypatch.setenv("FSB_DECODE_V2", "1")
        out = fsb_decode_token(attn, q_t, cache, sig, pos_t, pos_f=pos_f)
        sel, prov = fc._last_sel.clone(), fc._last_prov.clone()
        monkeypatch.setenv("FSB_DECODE_V2", "0")
        chain = fsb_decode_token(attn, q_t, cache, sig, pos_t, pos_f=pos_f)
        assert torch.equal(sel, fc._last_sel) and torch.equal(prov, fc._last_prov)
        ref = _fsb_decode_readout(attn, q_t, cache, sel, pos_t, full_buffer=True)
        assert out.shape == ref.shape == chain.shape
        o, r, c = out.float().reshape(-1), ref.float().reshape(-1), chain.float().reshape(-1)
        assert ((o - r).norm() / r.norm()).item() < 1e-2
        assert ((o - c).norm() / c.norm()).item() < 4e-3
    assert len(ran) == 3 and all(x is not None for x in ran)


@gpu
@torch.no_grad()
def test_fused_decode_flag_and_envelope(monkeypatch):
    # FSB_DECODE_V2=0 keeps the chain and unset (the default) runs the fused read; configurations outside the
    # kernel's envelope (here the bf16 KV store) fall back to the chain without a launch
    m = _model(monkeypatch, 1024)
    prompt = torch.randint(0, 512, (1, 1024), generator=torch.Generator().manual_seed(61)).cuda()
    monkeypatch.setenv("FSB_DECODE_V2", "0")
    ran = _spy(monkeypatch)
    gd = _decoder(m, prompt, 4)
    gd.step()
    assert ran == []
    monkeypatch.delenv("FSB_DECODE_V2")
    gd = _decoder(m, prompt, 4)
    gd.step()
    assert ran and all(r is not None for r in ran)
    ran.clear()
    monkeypatch.setenv("FSB_DECODE_V2", "1")
    monkeypatch.setenv("FS_KV_INT8", "0")
    gd = _decoder(m, prompt, 4)
    gd.step()
    gd.step()
    assert ran and all(r is None for r in ran)


@gpu
@torch.no_grad()
@pytest.mark.parametrize("v2", ["0", "1"])
def test_forked_read_matches_serial_step(monkeypatch, v2):
    # FSB_DECODE_FORK: each layer's long-range read on its own stream beside the window, captured as two parallel
    # branches of the step graph: the same read cells as the serial step at every token; the combine runs as torch's
    # two ops instead of inside the readout, so the logits agree to that rounding
    m = _model(monkeypatch, 1024)
    prompt = torch.randint(0, 512, (1, 1024), generator=torch.Generator().manual_seed(67)).cuda()
    forced = torch.randint(0, 512, (200,), generator=torch.Generator().manual_seed(71)).cuda()
    monkeypatch.setenv("FSB_DECODE_V2", v2)
    runs = {}
    for fork in ("0", "1"):
        monkeypatch.setenv("FSB_DECODE_FORK", fork)
        gd = _decoder(m, prompt, len(forced))
        logits, sels = [], []
        for t in forced:
            gd.next_tok = t.view(1)
            gd.step()
            logits.append(gd.static_logits.float().clone())
            sels.append(gd.cache[0]._fsb_forest._last_sel.clone())
        runs[fork] = (torch.stack(logits), torch.stack(sels))
    (l0, s0), (l1, s1) = runs["0"], runs["1"]
    assert torch.equal(s0, s1)
    rel = ((l1 - l0).norm(dim=-1) / l0.norm(dim=-1)).max().item()
    assert rel < 1e-2, rel
    assert (l1.argmax(-1) == l0.argmax(-1)).float().mean().item() >= 0.99


@gpu
@torch.no_grad()
@pytest.mark.parametrize("L", [1024, 4096])
def test_fused_beams_match_pair_kernel(monkeypatch, L):
    # the fused kernel's two beam descents (its test hook writes both beams' cells) against forest_beam_pair (the
    # warp-per-query pair kernel, itself bitwise with the per-level reference) on the same decode state and queries
    from src.fsattn.fsb_fast_step import forest_beam_pair
    from src.fsattn.fsb_route import radius_rows, route_radius
    from src.fsattn.fsb_tree import beam_delta_eff, fsb_decode_token
    m = _model(monkeypatch, L)
    prompt = torch.randint(0, 512, (1, L), generator=torch.Generator().manual_seed(73)).cuda()
    monkeypatch.setenv("FSB_DECODE_V2", "1")
    gd = _decoder(m, prompt, 16)
    for _ in range(3):
        gd.step()
    attn, cache = gd.attn[0], gd.cache[0]
    fc = cache._fsb_forest
    tk, T, s = fc._tok, 16, attn.cell_stride
    ukw = tk["ukw"]
    pos_t = torch.full((), gd.pos, dtype=torch.int64, device="cuda")
    pos_f = torch.full((1,), float(gd.pos), device="cuda")
    d_max = beam_delta_eff(attn)
    rc = radius_rows(attn, tk["head_id"], route_radius(attn))
    cells = torch.full((T, 2 * attn.fsb_W_max << d_max), -7, dtype=torch.int64, device="cuda")
    monkeypatch.setattr(dcu, "_CELLS", cells)
    g = torch.Generator(device="cuda").manual_seed(79)
    for scale in (0.25, 1.0, 4.0, 16.0):
        q_t = (torch.randn(1, T, 1, 64, device="cuda", generator=g) * scale).to(torch.bfloat16)
        fsb_decode_token(attn, q_t, cache, tk["sigma"], pos_t, pos_f=pos_f)
        pair = forest_beam_pair(fc.F, fc.SP, tk["W"], tk["delta"], attn.fsb_W_max, ukw["union_F"], ukw["union_SP"],
                                ukw["union_W"], ukw["union_delta"], attn.fsb_union_W_max, q_t.view(T, 1, 64),
                                pos_f.view(1, 1).expand(T, 1), tk["lam"], tk["sigv"], attn.scale, s, d_max,
                                attn.fsb_node_score, rc, bud=tk["bud"], bud_u=tk["bud_u"])
        (ca, _), (cb, _) = pair
        want = torch.cat([ca.reshape(T, -1), cb.reshape(T, -1)], dim=1)
        assert torch.equal(cells, want), scale
        assert (want >= 0).any()


@gpu
@torch.no_grad()
@pytest.mark.parametrize("dts", [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.bfloat16),
                                 (torch.bfloat16, torch.float32), (torch.float32, torch.float32)])
def test_scale_combine_matches_torch_ops(dts):
    # lr_scale_combine (FSB_DECODE_JOIN's join) against torch's out_short + lrs * out_long and the transpose into W_o's
    # input, bit for bit and in the expression's dtype, for bf16 and fp32 scales and read outputs at the released shape
    from src.utils import trunk_triton as tt
    lrs_dt, lr_dt = dts
    g = torch.Generator(device="cuda").manual_seed(83)
    for scale in (0.05, 1.0, 30.0):
        sr = (torch.randn(1, 16, 1, 64, device="cuda", generator=g) * scale).to(torch.bfloat16)
        out_long = (torch.randn(1, 16, 1, 64, device="cuda", generator=g) * scale).to(lr_dt)
        lrs = (0.2 * torch.randn(16, 1, 1, device="cuda", generator=g)).to(lrs_dt)
        ref = (sr + lrs.unsqueeze(0) * out_long).transpose(1, 2).contiguous().view(1, 1, 16 * 64)
        assert tt.lr_scale_combine_ok(sr, lrs, out_long)
        got = tt.lr_scale_combine(sr, lrs, out_long)
        assert got.dtype == ref.dtype and got.shape == ref.shape
        assert torch.equal(got, ref), scale


@gpu
@torch.no_grad()
@pytest.mark.parametrize("v2", ["0", "1"])
def test_forked_join_matches_torch_combine(monkeypatch, v2):
    # FSB_DECODE_JOIN=1 (with FSB_DECODE_FORK=1): every forked layer joins with the one-kernel combine; teacher-forced
    # steps across query blocks give the logits and the read cells of the torch combine bit for bit
    from src.utils import trunk_triton as tt
    m = _model(monkeypatch, 1024, n_layers=2)
    prompt = torch.randint(0, 512, (1, 1024), generator=torch.Generator().manual_seed(89)).cuda()
    forced = torch.randint(0, 512, (200,), generator=torch.Generator().manual_seed(97)).cuda()
    monkeypatch.setenv("FSB_DECODE_V2", v2)
    monkeypatch.setenv("FSB_DECODE_FORK", "1")
    joins, real = [], tt.lr_scale_combine
    monkeypatch.setattr(tt, "lr_scale_combine", lambda *a: joins.append(1) or real(*a))
    restore = _taus(m)
    runs = {}
    for join in ("0", "1"):
        monkeypatch.setenv("FSB_DECODE_JOIN", join)
        joins.clear()
        restore()
        gd = _decoder(m, prompt, len(forced))
        logits, sels = [], []
        for t in forced:
            gd.next_tok = t.view(1)
            gd.step()
            logits.append(gd.static_logits.float().clone())
            sels.append(torch.stack([c._fsb_forest._last_sel.clone() for c in gd.cache]))
        runs[join] = (torch.stack(logits), torch.stack(sels))
        assert (len(joins) >= len(m.blocks)) == (join == "1"), (join, len(joins))
    (l0, s0), (l1, s1) = runs["0"], runs["1"]
    assert torch.isfinite(l0).all() and (s0 >= 0).float().mean() > 0.9
    assert torch.equal(s0, s1)
    assert torch.equal(l0, l1)


def _forest_tables(fc):
    # every table of a decode forest that a read or a later ingest uses, cloned (the arena, leaves, admission table,
    # counts, spines, chunk index table)
    out = [fc._arena_mu, fc._arena_aux, fc._leaf_arena, fc.adm, fc.m_tot, fc.SP["node"], fc.SP["root"],
           fc.SP["height"], fc._jbase_buf]
    return [t.clone() for t in out]


def _budget_tables(fc):
    # a layer's per-head beam budgets as the captured decode step reads them
    tk = fc._tok
    ukw = tk.get("ukw") or {}
    return [t.clone() for t in (tk["W"], tk["delta"], tk["bud"], tk["bud_u"], ukw.get("union_W"),
                                ukw.get("union_delta")) if t is not None]


@gpu
@torch.no_grad()
@pytest.mark.parametrize("L,uq", [(1024, 0.5), (4096, 0.5), (1024, 0.97)])
def test_async_block_change_matches_host_decided(monkeypatch, L, uq):
    # FSB_DECODE_ASYNC=1: a block change whose counts the host provably does not need appends, decides the dirty chunk
    # heights on the device and rebuilds them with no synchronizing call (checked with torch's sync debug mode); across
    # many teacher-forced block changes every forest table, the beam budgets, the read cells and the logits equal the
    # host-decided path's bit for bit, and the host state that follows one block late (counts, chunk metadata, height
    # list) agrees. uq: the union forest's threshold quantile (0.97: a few cells per head, so the beam widths grow
    # during the decode and are recomputed on the device)
    from src.fsattn import fsb_build_cuda as fbc
    m = _model(monkeypatch, L, n_layers=2)
    prompt = torch.randint(0, 512, (1, L), generator=torch.Generator().manual_seed(101)).cuda()
    forced = torch.randint(0, 512, (700,), generator=torch.Generator().manual_seed(103)).cuda()
    monkeypatch.setenv("FSB_DECODE_V2", "1")
    monkeypatch.setenv("FSB_DECODE_FORK", "1")
    ran, real = [], fbc.lean_dirty_build
    monkeypatch.setattr(fbc, "lean_dirty_build", lambda *a: ran.append(real(*a)) or ran[-1])
    restore = _taus(m)
    runs = {}
    for asy in ("0", "1"):
        monkeypatch.setenv("FSB_DECODE_ASYNC", asy)
        ran.clear()
        restore()
        gd = _decoder(m, prompt, len(forced))
        if uq != 0.5:
            for blk, c in zip(m.blocks, gd.cache):
                blk.attn.fsattn.fsb_tau_union.copy_(c._ceil[0].float().quantile(uq, dim=-1))
        logits, sels, tabs = [], [], []
        for i, t in enumerate(forced):
            gd.next_tok = t.view(1)
            if asy == "1" and i > 0 and gd.pos % gd.Bq == 0:
                torch.cuda.set_sync_debug_mode("error")
                try:
                    gd.step()
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            else:
                gd.step()
            logits.append(gd.static_logits.float().clone())
            sels.append(torch.stack([c._fsb_forest._last_sel.clone() for c in gd.cache]))
            if (gd.pos - 1) % gd.Bq == 0:
                torch.cuda.synchronize()
                tabs.append([_forest_tables(f) for c in gd.cache for f in (c._fsb_forest, c._fsb_forest.utau)]
                            + [_budget_tables(c._fsb_forest) for c in gd.cache])
        host = []
        for c in gd.cache:
            for f in (c._fsb_forest, c._fsb_forest.utau):
                if f._rb is not None:
                    f._settle()
                host.append((list(f._m_known), f._lean_meta, list(f.F["ks"]), f._height_h, f._m_max_h))
        runs[asy] = (torch.stack(logits), torch.stack(sels), tabs, host)
        assert (len(ran) > 0 and all(ran)) == (asy == "1"), (asy, ran)
    (l0, s0, t0, h0), (l1, s1, t1, h1) = runs["0"], runs["1"]
    assert len(t0) == len(t1) >= 5
    for a, b in zip(t0, t1):
        for fa, fb in zip(a, b):
            assert all(torch.equal(x, y) for x, y in zip(fa, fb))
    assert h0 == h1
    assert torch.equal(s0, s1) and (s0 >= 0).float().mean() > 0.9
    assert torch.equal(l0, l1)


@gpu
@torch.no_grad()
@pytest.mark.parametrize("L,tau_q,periodic", [(640, 0.0, False), (1408, 0.0, False), (2944, 0.3, False),
                                              (4096, 0.0, False), (4096, 0.3, True), (2944, 0.0, True)])
def test_prefix_beams_match_pair_kernel(monkeypatch, L, tau_q, periodic):
    # FSB_DECODE_V3=1: both descents with their keep-all top levels batched (team_prefix) against forest_beam_pair,
    # on decode states whose forests are a lone chunk or a spine of chunks (every cell admitted, or 70% per head; the
    # union forest about half), with tie-heavy cells too (a periodic prompt: identical cells, whose scores differ only
    # through a position prior that saturates far away); the hook reports how many levels each prefix covered
    from src.fsattn.fsb_fast_step import forest_beam_pair
    from src.fsattn.fsb_route import radius_rows, route_radius
    from src.fsattn.fsb_tree import beam_delta_eff, fsb_decode_token
    m = _model(monkeypatch, L)
    gen = torch.Generator().manual_seed(107)
    if periodic:
        prompt = torch.randint(0, 512, (1, 32), generator=gen).repeat(1, L // 32).cuda()
    else:
        prompt = torch.randint(0, 512, (1, L), generator=gen).cuda()
    monkeypatch.setenv("FSB_DECODE_V2", "1")
    monkeypatch.setenv("FSB_DECODE_V3", "1")
    gd = _decoder(m, prompt, 16)
    if tau_q > 0:
        f, c = gd.attn[0], gd.cache[0]
        f.fsb_tau.copy_(c._ceil[0].float().quantile(tau_q, dim=-1))
    for _ in range(3):
        gd.step()
    attn, cache = gd.attn[0], gd.cache[0]
    fc = cache._fsb_forest
    tk, T, s = fc._tok, 16, attn.cell_stride
    ukw = tk["ukw"]
    pos_t = torch.full((), gd.pos, dtype=torch.int64, device="cuda")
    pos_f = torch.full((1,), float(gd.pos), device="cuda")
    d_max = beam_delta_eff(attn)
    rc = radius_rows(attn, tk["head_id"], route_radius(attn))
    cells = torch.full((T, 2 * attn.fsb_W_max << d_max), -7, dtype=torch.int64, device="cuda")
    lev0 = torch.full((T, 2), -1, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(dcu, "_CELLS", cells)
    monkeypatch.setattr(dcu, "_LEV0", lev0)
    g = torch.Generator(device="cuda").manual_seed(109)
    covered = []
    for scale in (0.25, 1.0, 4.0, 16.0):
        q_t = (torch.randn(1, T, 1, 64, device="cuda", generator=g) * scale).to(torch.bfloat16)
        fsb_decode_token(attn, q_t, cache, tk["sigma"], pos_t, pos_f=pos_f)
        pair = forest_beam_pair(fc.F, fc.SP, tk["W"], tk["delta"], attn.fsb_W_max, ukw["union_F"], ukw["union_SP"],
                                ukw["union_W"], ukw["union_delta"], attn.fsb_union_W_max, q_t.view(T, 1, 64),
                                pos_f.view(1, 1).expand(T, 1), tk["lam"], tk["sigv"], attn.scale, s, d_max,
                                attn.fsb_node_score, rc, bud=tk["bud"], bud_u=tk["bud_u"])
        (ca, _), (cb, _) = pair
        want = torch.cat([ca.reshape(T, -1), cb.reshape(T, -1)], dim=1)
        assert torch.equal(cells, want), scale
        assert (want >= 0).any()
        covered.append(lev0.clone())
    # every descent of a forest with at least 16 admitted cells batched its top levels (an empty forest has none)
    big = torch.stack([fc.m_tot, fc.utau.m_tot], dim=1) >= 16
    assert big.any()
    for lv in covered:
        assert bool((lv[big] >= 2).all()), (lv.tolist(), big.tolist())


@gpu
@torch.no_grad()
def test_prefix_decode_matches_level_loop(monkeypatch):
    # FSB_DECODE_V3=1 in the whole-step decode (with the fork, the one-kernel join and the device-decided block
    # changes): teacher-forced steps across query blocks give the read cells and the logits of the level-by-level
    # descent bit for bit; both arms read through the fused kernel at every call (a silent fallback to the kernel
    # chain would compare the chain with itself)
    m = _model(monkeypatch, 2944, n_layers=2)
    prompt = torch.randint(0, 512, (1, 2944), generator=torch.Generator().manual_seed(113)).cuda()
    forced = torch.randint(0, 512, (400,), generator=torch.Generator().manual_seed(127)).cuda()
    for k, v in (("FSB_DECODE_V2", "1"), ("FSB_DECODE_FORK", "1"), ("FSB_DECODE_JOIN", "1"), ("FSB_DECODE_ASYNC", "1")):
        monkeypatch.setenv(k, v)
    restore = _taus(m)
    ran = _spy(monkeypatch)
    runs = {}
    for pre in ("0", "1"):
        monkeypatch.setenv("FSB_DECODE_V3", pre)
        restore()
        ran.clear()
        gd = _decoder(m, prompt, len(forced))
        logits, sels = [], []
        for t in forced:
            gd.next_tok = t.view(1)
            gd.step()
            logits.append(gd.static_logits.float().clone())
            sels.append(torch.stack([c._fsb_forest._last_sel.clone() for c in gd.cache]))
        runs[pre] = (torch.stack(logits), torch.stack(sels))
        assert ran and all(r is not None for r in ran), (pre, len(ran))
    assert not dcu._failed
    (l0, s0), (l1, s1) = runs["0"], runs["1"]
    assert (s0 >= 0).float().mean() > 0.9
    assert torch.equal(s0, s1)
    assert torch.equal(l0, l1)
