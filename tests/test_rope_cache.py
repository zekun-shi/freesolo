"""RoPE table growth (experiments/scaling/yarn_rope.py) after the model is cast to bf16, as bench_efficiency casts it
before its graphed decoders extend the table past the loaded length."""
import pytest
import torch

from experiments.scaling.yarn_rope import build_rope

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_growth_after_bf16_cast_matches_a_fresh_build(device):
    # the entries the table had stay bit for bit; the new positions are the fresh fp32 build's values cast to bf16
    # (within one bf16 rounding step: the device's cos / sin may differ from the CPU's in the last fp32 bit), so
    # neighbouring positions stay distinct (bf16 positions put 4093-4097 all at 4096)
    n0, n1 = 4096, 4096 + 793
    r = build_rope(64, n0).to(device).to(torch.bfloat16)
    cos0, sin0 = r.cos_cached.clone(), r.sin_cached.clone()
    r._build_cache(n1)
    ref = build_rope(64, n1).to(device).to(torch.bfloat16)
    assert r.cos_cached.dtype == torch.bfloat16 and r.cos_cached.shape == ref.cos_cached.shape
    assert torch.equal(r.cos_cached[:n0], cos0) and torch.equal(r.sin_cached[:n0], sin0)
    for got, want in ((r.cos_cached, ref.cos_cached), (r.sin_cached, ref.sin_cached)):
        assert (got[n0:].float() - want[n0:].float()).abs().max().item() <= 2.0 ** -8
    assert not torch.equal(r.cos_cached[n0 + 1], r.cos_cached[n0 + 2])


@torch.no_grad()
def test_growth_in_fp32_rebuilds_the_whole_table():
    r = build_rope(64, 1024)
    r._build_cache(4096)
    t = torch.arange(4096, dtype=torch.float32)
    freqs = torch.outer(t, r.inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    assert torch.equal(r.cos_cached, emb.cos() * r.mscale) and torch.equal(r.sin_cached, emb.sin() * r.mscale)
