"""FS (dense) and FS-L (sparse) under the scheduling the FS-B paths use: each prefill layer's window branch forked next
to its long-range branch (FS_WINDOW_OVERLAP) and each graphed decode layer's long-range read forked beside its window
(FS_DECODE_FORK), against the serial paths bit for bit: the eager and the CUDA-graph prefill, teacher-forced
GraphedDecoder steps across query blocks on the int8 and the bf16 KV store, and the fused decode reads stopped before
their combine (lr_decode_read + lr_combine) in every rounding branch of that last step."""
import os
import sys
from contextlib import nullcontext

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "scaling"))

import bench_efficiency as be  # noqa: E402
from experiments.scaling.model import FreeSoloLM  # noqa: E402
from src.fsattn.attention import FSAttn  # noqa: E402
from src.utils import trunk_triton as tt  # noqa: E402


def _serve(monkeypatch, serve):
    # the bench's switches for its FS / FS-L serve configs (bench_efficiency.load and main)
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "1" if serve == "sparse" else "0")
    monkeypatch.setenv("FS_FUSED_SELECT", "1")
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")


def _model(serve, L, n_layers, seed):
    torch.manual_seed(seed)
    return FreeSoloLM(vocab_size=512, d_model=256, n_layers=n_layers, n_heads=4, d_ff=512, max_seq_len=L + 512,
                      attn_type="fsattn", **be.serve_kwargs(serve)).cuda().to(torch.bfloat16).eval()


@pytest.mark.parametrize("serve", ["dense", "sparse"])
def test_window_overlap_matches_serial(monkeypatch, serve):
    # FS_WINDOW_OVERLAP=1 forks every FS / FS-L layer's window branch onto a side stream next to the long-range branch
    # (FSAttn._window_fork, as FSB_WINDOW_OVERLAP does for FS-B layers): the eager prefill and the CUDA-graph prefill
    # (per-layer KV and the bench's shared K / V buffers, which the next layer overwrites) equal the serial path's bit
    # for bit, and the window hand-off issues no record_stream (under capture it would keep each layer's window output
    # out of the graph pool)
    from decode_graph import GraphedPrefiller
    from kv_cache import _trunk_cached, make_caches
    from longctx_infer import _lm_head_fp32, extend_rope_cache
    _serve(monkeypatch, serve)
    L, n_layers = 4096, 3                                     # 128 cells: more than the FS-L candidate pool
    m = _model(serve, L, n_layers, 13)
    extend_rope_cache(m, L + 256)
    g = torch.Generator().manual_seed(23)
    x1 = torch.randint(0, 512, (1, L), generator=g).cuda()
    x2 = torch.randint(0, 512, (1, L), generator=g).cuda()
    forks, rs = [0], [0]
    real_fork, real_rs = FSAttn._window_fork, torch.Tensor.record_stream

    def spy_fork(fn, dev):
        forks[0] += 1
        return real_fork(fn, dev)

    def spy_rs(t, s):
        if sys._getframe(1).f_globals.get("__name__", "").endswith(".attention"):
            rs[0] += 1
        return real_rs(t, s)

    monkeypatch.setattr(FSAttn, "_window_fork", staticmethod(spy_fork))
    monkeypatch.setattr(torch.Tensor, "record_stream", spy_rs)
    res = {}
    for ov in ("0", "1"):
        monkeypatch.setenv("FS_WINDOW_OVERLAP", ov)
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
        assert torch.equal(res["0"][k], res["1"][k]), k
    assert torch.isfinite(res["1"]["eager_tail"]).all()
    for retain in (True, False):                              # the replays agree with the eager prefill
        assert torch.equal(res["1"][f"tail1_{retain}"], res["1"]["eager_tail"]), retain


@pytest.mark.parametrize("serve", ["dense", "sparse"])
@pytest.mark.parametrize("int8", ["1", "0"])
@torch.no_grad()
def test_decode_fork_matches_serial(monkeypatch, serve, int8):
    # FS_DECODE_FORK=1: every graphed decode layer's long-range read runs on its own stream beside the window (two
    # parallel branches of the step graph) and stops before its combine, which lr_combine applies after the join:
    # teacher-forced steps that cross query-block boundaries (pool refreshes and re-captures) and cell closes (int8
    # quantization) give the serial step's logits bit for bit, and every layer of every traced step forked a read
    from decode_graph import GraphedDecoder
    _serve(monkeypatch, serve)
    monkeypatch.setenv("FS_KV_INT8", int8)
    L = 1024
    m = _model(serve, L, 2, 52)
    prompt = torch.randint(0, 512, (1, L), generator=torch.Generator().manual_seed(53)).cuda()
    forced = torch.randint(0, 512, (300,), generator=torch.Generator().manual_seed(57)).cuda()
    reads, real = [], tt.lr_decode_read

    def spy(*a, **k):
        r = real(*a, **k)
        reads.append(r is not None)
        return r

    monkeypatch.setattr(tt, "lr_decode_read", spy)
    runs = {}
    for fork in ("0", "1"):
        monkeypatch.setenv("FS_DECODE_FORK", fork)
        reads.clear()
        gd = GraphedDecoder(m, prompt, Lmax=L + len(forced) + 16, device="cuda", dtype=torch.bfloat16)
        gd.prefill()
        assert all(getattr(c, "use_int8", False) == (int8 == "1") for c in gd.cache)
        logits = []
        for t in forced:
            gd.next_tok = t.view(1)                           # teacher-forced: both runs see the same tokens
            gd.step()
            logits.append(gd.static_logits.clone())
        runs[fork] = torch.stack(logits)
        assert (len(reads) > 0 and all(reads)) if fork == "1" else not reads, (fork, reads)
    assert torch.isfinite(runs["0"]).all()
    assert torch.equal(runs["0"], runs["1"])


@pytest.mark.parametrize("serve", ["dense", "sparse"])
@pytest.mark.parametrize("int8", [True, False])
@pytest.mark.parametrize("autocast,w32", [(True, False), (False, False), (False, True)])
@torch.no_grad()
def test_read_then_combine_matches_fused_read(monkeypatch, serve, int8, autocast, w32):
    # lr_decode_read + lr_combine against the fused read that combines in its kernel (fsl_lr_decode_combine on the
    # bf16 history, lr_comb_int8 on the int8 state), bit for bit and in the same dtype, in each rounding branch of the
    # last step: bf16 autocast (fp32 norm), no autocast with bf16 norm weights and long-range scales (bf16 roundings),
    # no autocast with fp32 ones
    from decode_graph import GraphedDecoder
    _serve(monkeypatch, serve)
    monkeypatch.setenv("FS_KV_INT8", "1" if int8 else "0")
    L = 1024
    m = _model(serve, L, 1, 61)
    attn = m.blocks[0].attn.fsattn
    g = torch.Generator(device="cuda").manual_seed(67)
    attn.lr_norm.scale.copy_(0.8 + 0.4 * torch.randn(64, device="cuda", generator=g))
    attn.long_range_scale.copy_(0.2 * torch.randn(4, 1, 1, device="cuda", generator=g))
    if w32:
        attn.lr_norm.scale.data = attn.lr_norm.scale.data.float()
        attn.long_range_scale.data = attn.long_range_scale.data.float()
    prompt = torch.randint(0, 512, (1, L), generator=torch.Generator().manual_seed(59)).cuda()
    gd = GraphedDecoder(m, prompt, Lmax=L + 64, device="cuda", dtype=torch.bfloat16)
    gd.prefill()
    buf, c = gd.layer_bufs[0], gd.cache[0]
    pool = buf["pool"]
    assert not pool.get("empty", False) and bool(pool.get("dense", False)) == (serve == "dense")
    st = (c._qk_buf, c._qv_buf, c._ks_buf, c._vs_buf, c._k_hot, c._v_hot, c._nq_t, c.s) if int8 else None
    kbuf, vbuf = buf.get("kbuf"), buf.get("vbuf")
    q_t, sr = (torch.randn(1, 4, 1, 64, device="cuda", generator=g).to(torch.bfloat16) for _ in range(2))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16) if autocast else nullcontext():
        if int8:
            ref = tt.lr_comb_int8(attn, q_t, sr, pool, gd.pos_t, attn.sigma, st)
        else:
            assert tt.fsl_lr_decode_combine_ok(attn, q_t, kbuf, vbuf, sr, pool, gd.pos_t, attn.sigma)
            ref = tt.fsl_lr_decode_combine(attn, q_t, kbuf, vbuf, sr, pool, gd.pos_t, attn.sigma)
        read = tt.lr_decode_read(attn, q_t, kbuf, vbuf, pool, gd.pos_t, attn.sigma, st)
        assert ref is not None and read is not None and tt.lr_combine_ok(sr, q_t)
        got = tt.lr_combine(sr, *read)
    assert read[0].dtype == torch.float32 and got.dtype == ref.dtype and got.shape == ref.shape
    assert torch.equal(got, ref)
    assert not torch.equal(ref, tt.lr_combine(torch.zeros_like(sr), *read))     # the window output is in it
