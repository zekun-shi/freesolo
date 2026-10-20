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
