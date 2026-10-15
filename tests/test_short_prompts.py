"""Prompts of 2-129 tokens through FS, FS-L and FS-B: the bench serve configs (bench_efficiency.serve_kwargs, fused
cell readout on) on a small random bf16 model under bf16 autocast without autograd, uncached (model(x), as
eval_downstream runs) and through a KV-cache prefill. With the default switches every length runs; FS (dense) and
FS-L and FS-B all route their prefill readout through the fused kernel where it applies. Up to 128 tokens no cell is
complete before any query's block, so every position equals the eager readout (FSL_READOUT_FUSED=0) bit for bit; at
129 tokens only the last position may differ (it reads cells 0-3), within the fused readout's tolerance."""
import os
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "scaling"))

import bench_efficiency as be  # noqa: E402
from experiments.scaling.model import FreeSoloLM  # noqa: E402
from kv_cache import _trunk_cached, make_caches  # noqa: E402
from src.fsattn.attention import FSAttn  # noqa: E402


def _model(serve):
    kw = be.serve_kwargs(serve)
    kw["window_size"] = 64            # the batched window also runs on these lengths (64-key tiles, W = 64)
    torch.manual_seed(5)
    m = FreeSoloLM(vocab_size=512, d_model=256, n_layers=2, n_heads=4, d_ff=704, max_seq_len=1024,
                   attn_type="fsattn", **kw).cuda()
    for mod in m.modules():
        if getattr(mod, "fsb_enabled", False):
            mod.fsb_tau.fill_(0.0)    # bench_efficiency.apply_fsb_tau: admit every eligible cell
    return m.to(torch.bfloat16).eval()


@pytest.mark.parametrize("serve", ["dense", "sparse", "fsb"])
def test_short_prompts_run_and_match_eager_readout(monkeypatch, serve):
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "1" if serve == "sparse" else "0")
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
    monkeypatch.setenv("FSB_TRAIN_GRAPH", "0")
    if serve == "sparse":
        monkeypatch.setenv("FS_FUSED_SELECT", "1")
    for name in ("FSL_WINDOW_BATCHED", "FSL_TRUNK_FUSED", "FSL_NORM_ONEPASS", "FSL_ROPE_FUSED",
                 "FSL_PREFILL_CELL_REUSE", "FSL_READOUT_FUSED", "FSL_WINDOW_STREAM"):
        monkeypatch.delenv(name, raising=False)
    m = _model(serve)
    fused_calls = []
    orig = FSAttn._cell_fused_lr

    def spy(self, Q, *a, **k):
        out = orig(self, Q, *a, **k)
        fused_calls.append(bool(getattr(self, "_lr_combined", False)))
        return out
    monkeypatch.setattr(FSAttn, "_cell_fused_lr", spy)

    def run(x, readout_fused):
        # (uncached logits, cached-prefill hidden states) and whether the fused readout ran in each
        monkeypatch.setenv("FSL_READOUT_FUSED", readout_fused)
        outs, ran = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for fn in (lambda: m(x), lambda: _trunk_cached(m, x, make_caches(m))):
                fused_calls.clear()
                outs.append(fn())
                ran.append(any(fused_calls))
        return outs, ran

    fused_lengths = {0: set(), 1: set()}
    for N in range(2, 130):
        x = torch.randint(0, 512, (1, N), device="cuda", generator=torch.Generator(device="cuda").manual_seed(N))
        got, ran = run(x, "1")
        ref, ran_ref = run(x, "0")
        assert not any(ran_ref), (serve, N)
        for i, (a, b) in enumerate(zip(got, ref)):
            if ran[i]:
                fused_lengths[i].add(N)
            assert a.shape[1] == N and bool(torch.isfinite(a).all()), (serve, N, i)
            if N <= 128:
                assert torch.equal(a, b), (serve, N, i)
            else:
                assert torch.equal(a[:, :128], b[:, :128]), (serve, N, i)
                assert torch.allclose(a[:, 128:].float(), b[:, 128:].float(), rtol=0.05, atol=0.05), (serve, N, i)
    # the fused kernel ran exactly where its query group is valid, through both entry points
    for i in (0, 1):
        assert sorted(fused_lengths[i]) == [16, 32, 64, 128, 129], (serve, i, sorted(fused_lengths[i]))
