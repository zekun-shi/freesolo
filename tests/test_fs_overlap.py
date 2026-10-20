"""FS (dense) and FS-L (sparse) under the scheduling the FS-B prefill uses: each layer's window branch forked next to
its long-range branch (FS_WINDOW_OVERLAP), against the serial prefill bit for bit, eager and CUDA-graph captured."""
import os
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "scaling"))

import bench_efficiency as be  # noqa: E402
from experiments.scaling.model import FreeSoloLM  # noqa: E402
from src.fsattn.attention import FSAttn  # noqa: E402


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
