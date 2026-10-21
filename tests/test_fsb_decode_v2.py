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
    monkeypatch.setenv("FSB_DECODE_JOIN", "0")         # torch's combine; test_forked_join_matches_torch_combine joins
    monkeypatch.setenv("FSB_DECODE_V3", "0")           # the second version; the prefix tests build the third
    monkeypatch.setenv("FSB_DECODE_ASYNC", "0")        # host-decided block changes; the async tests decide on the device
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


def _bits(t):
    # a tensor's bit pattern (NaN-safe, signed zeros distinct) for bit-for-bit comparisons
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def _budget_tables(fc):
    # a layer's per-head beam budgets as the captured decode step reads them
    tk = fc._tok
    ukw = tk.get("ukw") or {}
    return [t.clone() for t in (tk["W"], tk["delta"], tk["bud"], tk["bud_u"], ukw.get("union_W"),
                                ukw.get("union_delta")) if t is not None]


@gpu
@torch.no_grad()
@pytest.mark.parametrize("L,uq,dev_budgets,busy,mix", [(1024, 0.5, True, True, None), (4096, 0.5, True, True, None),
                                                       (1024, 0.97, True, True, None), (1024, 0.97, False, True, None),
                                                       (1024, 0.5, True, False, None),
                                                       (1024, 0.5, True, True, "decline"),
                                                       (1024, 0.97, True, True, "kmax3"),
                                                       (4096, 0.5, True, True, "kmax3")])
def test_async_block_change_matches_host_decided(monkeypatch, L, uq, dev_budgets, busy, mix):
    # FSB_DECODE_ASYNC=1: a block change whose counts the host provably does not need appends, decides the dirty chunk
    # heights on the device and rebuilds them with no synchronizing call (checked with torch's sync debug mode, with
    # earlier work still queued on the stream: busy); across many teacher-forced block changes every forest table, the
    # beam budgets, the read cells and the logits equal the host-decided path's bit for bit, and the host state that
    # follows one block late (counts, chunk metadata, height list) agrees. uq: the union forest's threshold quantile
    # (0.97: a few cells per head, so the beam widths grow during the decode and are recomputed on the device).
    # dev_budgets False: the device cannot write the budgets in place (_budgets_on_device declines) while the host's
    # heights are a block old, so they are recomputed from the spines' heights on the device, never from the stale host
    # heights. busy False: the stream is idle at every block change (a loop that synchronises every token). The
    # device-decided ingests of a block run as one batch (lean_dirty_build_many) where they can. mix "decline": every
    # other forest stays out of the batch and every other such forest's own device-decided ingest declines (it reads
    # its counts and takes the host-decided path), so blocks mix the batch, per-forest device ingests and host-decided
    # ones (no sync guard then); "kmax3": chunks taller than 8 cells count as tall, so heads cross their boundaries
    # every few blocks and the device completes those chunks from the host's proposals (their forests outside the
    # batch)
    from src.fsattn import fsb_build_cuda as fbc
    from src.fsattn import fsb_tree as ft
    m = _model(monkeypatch, L, n_layers=2)
    prompt = torch.randint(0, 512, (1, L), generator=torch.Generator().manual_seed(101)).cuda()
    forced = torch.randint(0, 512, (700,), generator=torch.Generator().manual_seed(103)).cuda()
    monkeypatch.setenv("FSB_DECODE_V2", "1")
    monkeypatch.setenv("FSB_DECODE_FORK", "1")
    ran, talls, nok = [], [], [0]
    real, real_many, real_ok = fbc.lean_dirty_build, fbc.lean_dirty_build_many, fbc.lean_many_ok

    def spy_build(*a, **k):                             # a forest's own device-decided ingest
        talls.append(k.get("tall") is not None and len(k["tall"][0]) > 0)
        ran.append(False if mix == "decline" and len(ran) % 2 else real(*a, **k))
        return ran[-1]

    def spy_many(entries, *a, **k):                     # the batch of a block's device-decided ingests
        ran.extend([True] * len(entries))
        return real_many(entries, *a, **k)

    def spy_ok(e):                                      # "decline": every other forest left out of the batch
        nok[0] += 1
        return real_ok(e) and not (mix == "decline" and nok[0] % 2)
    monkeypatch.setattr(fbc, "lean_dirty_build", spy_build)
    monkeypatch.setattr(fbc, "lean_dirty_build_many", spy_many)
    monkeypatch.setattr(fbc, "lean_many_ok", spy_ok)
    if mix == "kmax3":
        monkeypatch.setattr(fbc, "DIRTY_KMAX", 3)
    opened, real_bud = [], ft._forest_budgets

    def spy_budgets(attn, fc, *a):
        u = getattr(fc, "utau", None)
        opened.append(bool(fc._eff_open or (u is not None and u._eff_open)))
        return real_bud(attn, fc, *a)
    monkeypatch.setattr(ft, "_forest_budgets", spy_budgets)
    if not dev_budgets:
        monkeypatch.setattr(ft, "_budgets_on_device", lambda *a, **k: False)
    restore = _taus(m)
    runs = {}
    for asy in ("0", "1"):
        monkeypatch.setenv("FSB_DECODE_ASYNC", asy)
        ran.clear()
        talls.clear()
        opened.clear()
        nok[0] = 0
        restore()
        gd = _decoder(m, prompt, len(forced))
        if uq != 0.5:
            for blk, c in zip(m.blocks, gd.cache):
                blk.attn.fsattn.fsb_tau_union.copy_(c._ceil[0].float().quantile(uq, dim=-1))
        logits, sels, tabs = [], [], []
        for i, t in enumerate(forced):
            gd.next_tok = t.view(1)
            if asy == "1" and i > 0 and gd.pos % gd.Bq == 0 and busy:
                torch.cuda._sleep(20_000_000)                # earlier work still queued when the block changes
                torch.cuda.set_sync_debug_mode("error" if mix != "decline" else "default")
                try:
                    gd.step()
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            else:
                if gd.pos % gd.Bq == 0:
                    torch.cuda.synchronize()
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
                if f._meta_m is not None:
                    f._meta_now()                            # the chunk metadata the next host-decided ingest compares
                host.append((list(f._m_known), f._lean_meta, f._height_h, f._m_max_h, dict(f._fold_rows)))
        runs[asy] = (torch.stack(logits), torch.stack(sels), tabs, host)
        if mix == "decline":
            assert (asy == "1") == (any(ran) and not all(ran)), (asy, ran)
        else:
            assert (len(ran) > 0 and all(ran)) == (asy == "1"), (asy, ran)
        if mix == "kmax3":
            assert (sum(talls) > 0) == (asy == "1"), (asy, talls)
        assert any(opened) if (asy == "1" and uq == 0.97) else (asy == "1" or not any(opened)), opened
    (l0, s0, t0, h0), (l1, s1, t1, h1) = runs["0"], runs["1"]
    assert len(t0) == len(t1) >= 5
    for a, b in zip(t0, t1):
        for fa, fb in zip(a, b):
            assert all(torch.equal(_bits(x), _bits(y)) for x, y in zip(fa, fb))
    assert h0 == h1
    assert torch.equal(s0, s1) and (s0 >= 0).float().mean() > 0.9
    assert torch.equal(l0, l1)


def _ingests(Kbar, ceil, tau, m_cap, seq, asy):
    # one decode forest through the ingests of seq (visible cell counts, the first one host-decided): FSB_DECODE_ASYNC's
    # device-decided ingest wherever it qualifies (asy), else the host-decided one; every table after each ingest, the
    # settled host state at the end, and the visible counts of the ingests that went to the device
    from src.fsattn.fsb_tree import ForestCache, _async_readback
    T, Dh = Kbar.shape[1], Kbar.shape[3]
    fc = ForestCache(T, Dh, Kbar.device, lean=True)
    fc.m_cap, fc.k_cap = m_cap, m_cap.bit_length() - 1
    tabs, on_dev = [], []
    for M_vis in seq:
        if asy and fc.F is not None:
            if fc._rb is not None:
                fc._settle()
            if fc._async_ready(M_vis, need_live=True):
                assert fc._append_admitted(ceil, M_vis, tau) is fc._mc_nn
                fc._async_next = True
                on_dev.append(M_vis)
        fc.ingest(Kbar, ceil, M_vis, tau)
        fc._stable_spines()
        if asy and fc._m_known is None:
            _async_readback([fc])
        torch.cuda.synchronize()
        tabs.append(_forest_tables(fc))
    if fc._rb is not None:
        fc._settle()
    if fc._meta_m is not None:
        fc._meta_now()
    return tabs, (list(fc._m_known), fc._lean_meta, fc._height_h, dict(fc._fold_rows)), on_dev


@gpu
@torch.no_grad()
@pytest.mark.parametrize("zeros", [False, True])
def test_async_ingest_rebuilds_chunks_under_a_new_fold_flag(monkeypatch, zeros):
    # FSB_DECODE_ASYNC's device-decided ingest against the host-decided one when the layout's fold flag of a dirty
    # height changes under a chunk that stays in place: the visible cells pass 512 (M_pad 512 -> 1024 turns height 9's
    # flag on) while head 0 keeps its 512-cell chunk and head 1 completes one, so height 9 is dirty with head 0's chunk
    # unmoved. The host path rebuilds that chunk with the new flag, so the device path must too (it keeps unmoved chunks
    # only where their rows carry the launch's flag): every forest table, the spines, the chunk index table and the
    # settled host state agree bit for bit at every ingest. zeros: head 0's first dimension is -0 in every cell but the
    # chunk's first (+0), so each node without that cell sums to -0 there and the folded build's pad zero (the chunk's
    # first cell's sign) makes its mean +0: the flag changes those rows, and rows kept from the old flag would differ
    from src.fsattn import fsb_build_cuda as fbc
    T, Dh, M_cap, dev = 16, 64, 544, torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(7)
    Kbar = torch.randn(1, T, M_cap, Dh, device=dev, generator=g)
    if zeros:
        Kbar[0, 0, :, 0] = -0.0
        Kbar[0, 0, 0, 0] = 0.0
    ceil = torch.rand(1, T, M_cap, device=dev, generator=g) + 0.5
    ceil[0, 1, 100] = -1.0                                  # head 1 admits 511 of the first 512 cells
    tau = torch.zeros(T, device=dev)
    skips, real = [], fbc.lean_dirty_build
    monkeypatch.setattr(fbc, "lean_dirty_build", lambda *a, **k: skips.append(k.get("skip_mask")) or real(*a, **k))
    seq = (512, 516, 520, 524, 528)
    t0, h0, _ = _ingests(Kbar, ceil, tau, M_cap, seq, False)
    t1, h1, on_dev = _ingests(Kbar, ceil, tau, M_cap, seq, True)
    assert on_dev == list(seq[1:]) and len(skips) == 4, (on_dev, skips)
    assert not (skips[0] >> 9) & 1 and all((sk >> 9) & 1 for sk in skips[1:]), skips
    for a, b in zip(t0, t1):
        assert all(torch.equal(_bits(x), _bits(y)) for x, y in zip(a, b))
    assert h0 == h1 and h1[3][9] == 1
    m = t0[1][4]
    assert m[0].item() == 516 and m[1].item() == 515      # head 0's chunk of height 9 stays, head 1's appears
    if zeros:
        # the flag matters for this input: the host rebuild at the new flag changed head 0's unmoved chunk rows
        nb, nodes = 2 * (1 << 9) - 9 - 2, 2 * (1 << 9) - 1
        assert not all(torch.equal(_bits(t0[0][i][0, nb:nb + nodes]), _bits(t0[1][i][0, nb:nb + nodes]))
                       for i in (0, 1))


@gpu
@torch.no_grad()
def test_async_ingest_needs_the_cuda_build(monkeypatch):
    # FSB_DECODE_ASYNC: only the CUDA build rebuilds chunks on the device, so with FSB_BUILD_CUDA=0, or with its kernel
    # variant failed to set up, a forest takes the host-decided ingest from the start (its counts in the block's one
    # batched read) instead of a device-decided one that falls back to a read of its own
    from src.fsattn import fsb_build_cuda as fbc
    from src.fsattn.fsb_tree import ForestCache
    T, Dh, M_cap, dev = 16, 64, 544, torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(5)
    Kbar = torch.randn(1, T, M_cap, Dh, device=dev, generator=g)
    ceil = torch.rand(1, T, M_cap, device=dev, generator=g) + 0.5
    fc = ForestCache(T, Dh, dev, lean=True)
    fc.m_cap, fc.k_cap = M_cap, M_cap.bit_length() - 1
    fc.ingest(Kbar, ceil, 512, torch.zeros(T, device=dev))
    fc._stable_spines()
    assert fc._async_ready(516, need_live=True)
    monkeypatch.setenv("FSB_BUILD_CUDA", "0")
    assert not fc._async_ready(516, need_live=True)
    monkeypatch.setenv("FSB_BUILD_CUDA", "1")
    monkeypatch.setitem(fbc._failed, fbc._variant(torch.cuda.current_device(), Dh), "setup failed")
    assert not fc._async_ready(516, need_live=True)


@gpu
@torch.no_grad()
def test_static_tables_from_a_side_stream_reach_the_others_in_order(monkeypatch):
    # while a block change's side streams are in flight (FSB_DECODE_ASYNC), a shared static table that one of them made
    # (its kernels possibly still writing it) reaches another stream only behind an event, and the cache does not clear
    # under them; the block change makes the tables it can foresee before the fork, so this is the fallback
    from src.fsattn import fsb_tree as ft
    monkeypatch.setattr(ft, "_STATIC_CACHE", {})
    monkeypatch.setattr(ft, "_STATIC_EVENTS", {})
    monkeypatch.setattr(ft, "_SIDE_PHASE", [True])
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    t = torch.zeros(1 << 20, device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.stream(a):
        torch.cuda._sleep(50_000_000)                   # the maker's kernels run late
        t.fill_(1.0)
        ft._static_put(("t",), t, t.device)
    with torch.cuda.stream(b):
        s = ft._static_get(("t",)).sum()
    torch.cuda.synchronize()
    assert s.item() == float(1 << 20)
    for i in range(600):
        ft._static_put(("x", i), i)
    assert ("t",) in ft._STATIC_CACHE
    ft._SIDE_PHASE[0] = False
    ft._static_put(("y",), 0)
    assert ("t",) not in ft._STATIC_CACHE and not ft._STATIC_EVENTS


@gpu
@torch.no_grad()
@pytest.mark.parametrize("base,zeros", [(1016, False), (2040, False), (1016, True)])
def test_async_ingest_completes_tall_chunks_on_the_device(monkeypatch, base, zeros):
    # FSB_DECODE_ASYNC: an ingest during which a head's admitted count may cross a multiple of 1024 decides it on the
    # device too: the host proposes the one chunk each such head completes if it gets there (height 10 when the count
    # passes 1024 or 3072, 11 at 2048, ...), fsb_prep_lean_tall keeps the completed ones and the empty chunks of the
    # heights they leave, and every table equals the host-decided ingest's bit for bit, including heads that come within
    # reach of the boundary without crossing it and heads that cross a block later. base: the visible cells of the first
    # (host-decided) ingest, below 1024 or 2048. From 1016, the ingest at 1028 cells lays the forest out for 2048
    # (height 10's fold flag turns on) while heads that crossed earlier hold height-10 chunks built without it: the host
    # proposes those for rebuilding too, which happens where height 10 is dirty, as the host-decided ingest does.
    # zeros: head 0's first dimension is -0 in every cell but the first (+0), as in
    # test_async_ingest_rebuilds_chunks_under_a_new_fold_flag, so the flag changes the rows of the height-10 chunk head 0
    # completes at 1024 cells and keeps at 1028: rows kept from the old flag would differ from the host's rebuild
    from src.fsattn import fsb_build_cuda as fbc
    T, Dh, dev = 16, 64, torch.device("cuda")
    M_cap = base + 64
    g = torch.Generator(device="cuda").manual_seed(11)
    Kbar = torch.randn(1, T, M_cap, Dh, device=dev, generator=g)
    if zeros:
        Kbar[0, 0, :, 0] = -0.0
        Kbar[0, 0, 0, 0] = 0.0
    ceil = torch.rand(1, T, M_cap, device=dev, generator=g) + 0.5
    for t in range(1, T):                                   # head t drops t cells: counts spread around the boundary
        ceil[0, t, torch.randperm(base, generator=torch.Generator().manual_seed(t))[:t].to(dev)] = -1.0
    ceil[0, 3, base + 7:] = -1.0                            # head 3 stops 4 short of the boundary: in reach, never over
    tau = torch.zeros(T, device=dev)
    talls, real = [], fbc.lean_dirty_build
    monkeypatch.setattr(fbc, "lean_dirty_build", lambda *a, **k: talls.append(k.get("tall")) or real(*a, **k))
    seq = tuple(base + 4 * i for i in range(10))
    t0, h0, _ = _ingests(Kbar, ceil, tau, M_cap, seq, False)
    t1, h1, on_dev = _ingests(Kbar, ceil, tau, M_cap, seq, True)
    for a, b in zip(t0, t1):
        assert all(torch.equal(_bits(x), _bits(y)) for x, y in zip(a, b))
    assert h0 == h1
    bound = (base + 8) // 1024 * 1024
    m0, m1 = t0[0][4], t0[-1][4]
    crossed = (m0 < bound) & (m1 >= bound)
    assert int(crossed.sum()) >= 3 and bool((m0 < bound).any() & (m1 < bound).any()), (m0.tolist(), m1.tolist())
    # every ingest after the first decides on the device, several of them with tall chunks proposed
    assert on_dev == list(seq[1:]) and len(talls) == len(on_dev), (on_dev, len(talls))
    assert sum(t is not None and len(t[0]) > 0 for t in talls) >= 3
    assert (base == 1016) == any(t is not None and t[1] > 0 for t in talls), [t and t[1] for t in talls]
    if zeros:
        # the flag matters for this input: the host rebuild at 1028 cells changed head 0's unmoved height-10 rows
        i0, i1 = seq.index(1024), seq.index(1028)
        assert t0[i1][4][0].item() == 1028 and t0[i1][4][1].item() == 1027, t0[i1][4][:2].tolist()
        nb, nodes = 2 * (1 << 10) - 10 - 2, 2 * (1 << 10) - 1
        assert not all(torch.equal(_bits(t0[i0][i][0, nb:nb + nodes]), _bits(t0[i1][i][0, nb:nb + nodes]))
                       for i in (0, 1))


@gpu
@torch.no_grad()
@pytest.mark.parametrize("zeros,cap", [(False, None), (True, None), (True, 2)])
def test_batched_async_ingests_match_host_decided(monkeypatch, zeros, cap):
    # FSB_DECODE_ASYNC's batched block change outside a decode graph (_ingest_async_many -> lean_dirty_build_many: one
    # append, record, build and spine launch for all forests) against each forest's own host-decided ingests: three
    # forests with different thresholds (different counts per head, heights and Kmax) through ingests that pass 512
    # visible cells, where height 9's fold flag turns on while forest 0's head 0 keeps its 512-cell chunk and head 1
    # completes one; every table after each ingest and the settled host state equal bit for bit, every ingest after
    # the first in one batch of all three forests. zeros: head 0's first dimension is -0 in every cell but the first
    # (+0), as in test_async_ingest_rebuilds_chunks_under_a_new_fold_flag, so the flag changes the rows of forest 0's
    # unmoved chunk: a batch that kept them would differ from the host's rebuild. cap: the most forests a batch holds
    # (fsb_build_cuda.MANY_FORESTS, which its work list's forest numbers bound) lowered to 2, so the third forest takes
    # its own device-decided ingest
    from src.fsattn import fsb_build_cuda as fbc
    from src.fsattn.fsb_tree import ForestCache, _async_readback, _ingest_async_many
    if cap is not None:
        monkeypatch.setattr(fbc, "MANY_FORESTS", cap)
    T, Dh, M_cap, dev = 16, 64, 544, torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(13)
    Kbar = torch.randn(1, T, M_cap, Dh, device=dev, generator=g)
    if zeros:
        Kbar[0, 0, :, 0] = -0.0
        Kbar[0, 0, 0, 0] = 0.0
    ceil = torch.rand(1, T, M_cap, device=dev, generator=g)
    ceil[0, 1, 100] = -1.0                                  # forest 0's head 1 admits 511 of the first 512 cells
    taus = [torch.full((T,), q, device=dev) for q in (0.0, 0.25, 0.5)]
    seq = (508, 512, 516, 520, 524)
    batches, single = [], []
    real_many, real_one = fbc.lean_dirty_build_many, fbc.lean_dirty_build
    monkeypatch.setattr(fbc, "lean_dirty_build_many",
                        lambda e, *a, **k: batches.append(len(e)) or real_many(e, *a, **k))
    monkeypatch.setattr(fbc, "lean_dirty_build", lambda *a, **k: single.append(1) or real_one(*a, **k))

    def run(asy):
        fcs = []
        for _ in taus:
            fc = ForestCache(T, Dh, dev, lean=True)
            fc.m_cap, fc.k_cap = M_cap, M_cap.bit_length() - 1
            fcs.append(fc)
        tabs = []
        for M_vis in seq:
            items = []
            if asy and fcs[0].F is not None:
                for fc, tau in zip(fcs, taus):
                    if fc._rb is not None:
                        fc._settle()
                    assert fc._async_ready(M_vis, need_live=True)
                    items.append((fc, Kbar, ceil, None, M_vis, tau))
                rest = _ingest_async_many(items)
                assert len(rest) == len(fcs) - nb_f and all(it[0] is fc for it, fc in zip(rest, fcs[nb_f:]))
                for fc, _, _, _, _, tau in rest:            # the forests past the batch: their own device ingest
                    assert fc._append_admitted(ceil, M_vis, tau) is fc._mc_nn
                    fc._async_next = True
            for fc, tau in zip(fcs, taus):
                fc.ingest(Kbar, ceil, M_vis, tau)
                fc._stable_spines()
            if items:
                _async_readback(fcs)
            torch.cuda.synchronize()
            tabs.append([_forest_tables(fc) for fc in fcs])
        host = []
        for fc in fcs:
            if fc._rb is not None:
                fc._settle()
            if fc._meta_m is not None:
                fc._meta_now()
            host.append((list(fc._m_known), fc._lean_meta, fc._height_h, dict(fc._fold_rows)))
        return tabs, host

    nb_f = cap or len(taus)                                 # the forests one batch takes
    t0, h0 = run(False)
    t1, h1 = run(True)
    n_own = (len(taus) - nb_f) * (len(seq) - 1)
    assert batches == [nb_f] * (len(seq) - 1) and single == [1] * n_own, (batches, single)
    for a, b in zip(t0, t1):
        for fa, fb in zip(a, b):
            assert all(torch.equal(_bits(x), _bits(y)) for x, y in zip(fa, fb))
    assert h0 == h1 and h1[0][3][9] == 1
    assert len({h[2] and max(h[2]) for h in h1}) > 1          # the forests' heights differ
    if zeros:
        # the flag matters for this input: the host rebuild at 516 cells changed forest 0's unmoved height-9 rows
        assert t0[2][0][4][0].item() == 516 and t0[2][0][4][1].item() == 515, t0[2][0][4][:2].tolist()
        nb, nodes = 2 * (1 << 9) - 9 - 2, 2 * (1 << 9) - 1
        assert not all(torch.equal(_bits(t0[1][0][i][0, nb:nb + nodes]), _bits(t0[2][0][i][0, nb:nb + nodes]))
                       for i in (0, 1))


@gpu
@torch.no_grad()
def test_graphed_decode_refuses_steps_past_its_reserve(monkeypatch):
    # FSBGraphedDecoder's K / V store, forest cell capacity and RoPE table end at prompt + max_new_tokens + 8 positions:
    # every step before that runs, the next one raises before it writes anything
    m = _model(monkeypatch, 1024)
    prompt = torch.randint(0, 512, (1, 1024), generator=torch.Generator().manual_seed(61)).cuda()
    gd = _decoder(m, prompt, 40)
    assert gd.end == 1024 + 40 + 16
    while gd.pos < gd.end:
        gd.step()
    torch.cuda.synchronize()
    nq = [c._n_q for c in gd.cache]
    with pytest.raises(RuntimeError, match="past the decoder's K / V reserve"):
        gd.step()
    assert gd.pos == gd.end and [c._n_q for c in gd.cache] == nq


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


@gpu
@torch.no_grad()
@pytest.mark.parametrize("L,v3,periodic", [(1024, "0", False), (4096, "0", False), (1408, "1", False),
                                           (4096, "1", False), (2944, "1", True)])
def test_shared_code_read_matches_two_copies(monkeypatch, L, v3, periodic):
    # FSB_DECODE_SHARE=1: the build whose two descents and two re-rank passes run one copy of their code gives, at the
    # released shape, both beams' cells, the levels each V3 prefix covered, the read cells, their provenance and the
    # output (with and without the window combine) of the build with two copies, bit for bit, for queries over the
    # scale of the real ones and tie-heavy cells (a periodic prompt), with the V3 prefix plans walked and cached
    from src.fsattn.fsb_tree import beam_delta_eff, fsb_decode_token
    m = _model(monkeypatch, L)
    gen = torch.Generator().manual_seed(131)
    if periodic:
        prompt = torch.randint(0, 512, (1, 32), generator=gen).repeat(1, L // 32).cuda()
    else:
        prompt = torch.randint(0, 512, (1, L), generator=gen).cuda()
    monkeypatch.setenv("FSB_DECODE_V2", "1")
    monkeypatch.setenv("FSB_DECODE_V3", v3)
    gd = _decoder(m, prompt, 16)
    for _ in range(3):
        gd.step()
    attn, cache = gd.attn[0], gd.cache[0]
    fc = cache._fsb_forest
    tk, T = fc._tok, 16
    pos_t = torch.full((), gd.pos, dtype=torch.int64, device="cuda")
    pos_f = torch.full((1,), float(gd.pos), device="cuda")
    cells = torch.full((T, 2 * attn.fsb_W_max << beam_delta_eff(attn)), -7, dtype=torch.int64, device="cuda")
    lev0 = torch.full((T, 2), -1, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(dcu, "_CELLS", cells)
    monkeypatch.setattr(dcu, "_LEV0", lev0)
    ran = _spy(monkeypatch)
    g = torch.Generator(device="cuda").manual_seed(137)
    plans = [p for p in (getattr(fc, "_plan", None), getattr(fc.utau, "_plan", None)) if p is not None]
    for scale in (0.25, 1.0, 4.0, 16.0):
        q_t = (torch.randn(1, T, 1, 64, device="cuda", generator=g) * scale).to(torch.bfloat16)
        sr = (torch.randn(1, T, 1, 64, device="cuda", generator=g) * scale).to(torch.bfloat16)
        got = {}
        for walk in (True, False):              # the prefix plans walked (cleared caches), then cached
            for share in ("0", "1"):
                monkeypatch.setenv("FSB_DECODE_SHARE", share)
                for p in plans if walk else ():
                    p.zero_()
                outs = [fsb_decode_token(attn, q_t, cache, tk["sigma"], pos_t, pos_f=pos_f, sr=x) for x in (None, sr)]
                got[(walk, share)] = (outs, cells.clone(), lev0.clone(), fc._last_sel.clone(), fc._last_prov.clone())
            (o0, c0, v0, s0, p0), (o1, c1, v1, s1, p1) = got[(walk, "0")], got[(walk, "1")]
            assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(o0, o1)), (scale, walk)
            assert torch.equal(c0, c1) and torch.equal(v0, v1), (scale, walk)
            assert torch.equal(s0, s1) and torch.equal(p0, p1), (scale, walk)
            assert (c0 >= 0).any() and (s0 >= 0).any()
    assert len(ran) == 32 and all(r is not None for r in ran)
    assert not dcu._failed


@gpu
@torch.no_grad()
def test_shared_code_decode_matches_two_copies(monkeypatch):
    # FSB_DECODE_SHARE=1 in the whole-step decode (V3, the fork, the one-kernel join and the device-decided block
    # changes): teacher-forced steps across query blocks give the read cells and the logits of the build with two
    # copies of the code bit for bit, every read through the fused kernel
    m = _model(monkeypatch, 2944, n_layers=2)
    prompt = torch.randint(0, 512, (1, 2944), generator=torch.Generator().manual_seed(139)).cuda()
    forced = torch.randint(0, 512, (400,), generator=torch.Generator().manual_seed(149)).cuda()
    for k, v in (("FSB_DECODE_V2", "1"), ("FSB_DECODE_V3", "1"), ("FSB_DECODE_FORK", "1"), ("FSB_DECODE_JOIN", "1"),
                 ("FSB_DECODE_ASYNC", "1")):
        monkeypatch.setenv(k, v)
    restore = _taus(m)
    ran = _spy(monkeypatch)
    runs = {}
    for share in ("0", "1"):
        monkeypatch.setenv("FSB_DECODE_SHARE", share)
        restore()
        ran.clear()
        gd = _decoder(m, prompt, len(forced))
        logits, sels = [], []
        for t in forced:
            gd.next_tok = t.view(1)
            gd.step()
            logits.append(gd.static_logits.float().clone())
            sels.append(torch.stack([c._fsb_forest._last_sel.clone() for c in gd.cache]))
        runs[share] = (torch.stack(logits), torch.stack(sels))
        assert ran and all(r is not None for r in ran), (share, len(ran))
    assert not dcu._failed
    (l0, s0), (l1, s1) = runs["0"], runs["1"]
    assert (s0 >= 0).float().mean() > 0.9
    assert torch.equal(s0, s1)
    assert torch.equal(l0, l1)
