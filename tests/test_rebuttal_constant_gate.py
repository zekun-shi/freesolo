"""Independent constant-bonus reference for the trained no-erf control."""

import copy
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from src.fsattn.attention import FSAttn


def constant_reference(attn, q, k, v):
    """Select by content + norm bonus, then explicitly mask token attention."""
    b, h, n, d = q.shape
    s, block = attn.cell_stride, min(attn._cell_block, n)
    cells = (n + s - 1) // s
    keys = F.pad(k, (0, 0, 0, cells * s - n))
    summaries = keys.reshape(b, h, cells, s, d).mean(-2).float()
    bonus = keys.float().norm(dim=-1).reshape(b, h, cells, s).amax(-1)
    score = torch.matmul(q.float(), summaries.transpose(-2, -1)) * attn.scale
    score = score + attn.sel_scale.float()[None, :, None, None] * bonus[:, :, None]
    positions = torch.arange(n, device=q.device)
    ends = torch.arange(1, cells + 1, device=q.device) * s
    visible = ends[None] <= (positions // block * block)[:, None]
    indices = score.masked_fill(~visible, -1e9).topk(min(attn.cell_topk, cells), dim=-1).indices
    valid = visible[None, None].expand(b, h, n, cells).gather(-1, indices)
    selected = torch.zeros_like(score, dtype=torch.bool).scatter(-1, indices, valid)
    allowed = selected.repeat_interleave(s, dim=-1)[..., :n]
    logits = torch.matmul(q, k.transpose(-2, -1)).float() * attn.scale
    weights = logits.masked_fill(~allowed, -1e9).softmax(-1).to(v.dtype)
    out = torch.matmul(weights, v) * allowed.any(-1, keepdim=True).to(v.dtype)
    return attn.lr_norm(out.to(q.dtype)), indices.masked_fill(~valid, -1)


def make_attention(device):
    return FSAttn(d_model=128, n_heads=2, window_size=512, sigma_init=0.01,
                  causal=True, use_rope=True, rope_long_range=False, max_seq_len=1024,
                  fs_uniform_sr=True, cell_stride=32, cell_topk=4,
                  sel_strict_causal=1, cell_prune_kernel=0, fs_fused_attn=1,
                  cell_fused_select=0, bias=False).to(device)


def explicit_model(attn):
    ref = copy.deepcopy(attn)
    def long_range(self, q, k, v, sigma, n):
        return constant_reference(self, q, k, v)[0]
    ref._long_range = types.MethodType(long_range, ref)
    return ref


def test_fixed_sigma_saturates_every_eligible_integer_distance_and_restores(tmp_path):
    model = make_attention("cpu")
    # Completed 32-token cells end before the 128-token query block. Their nearest
    # center is at least 16 positions away. This covers every possible distance
    # through the optional 256k evaluation, not just a sampled prompt.
    distance = torch.arange(16, 262145, dtype=torch.float32)
    gate = torch.erf(distance[None] / (2 * model.sigma[:, None]))
    torch.testing.assert_close(gate, torch.ones_like(gate), atol=0, rtol=0)
    assert "log_sigma" not in dict(model.named_parameters())
    assert not model.sel_content_only
    checkpoint = tmp_path / "constant.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = make_attention("cpu")
    restored.log_sigma.fill_(0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored.sigma, model.sigma, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_training_indices_outputs_and_gradients_against_constant_reference(monkeypatch):
    torch.manual_seed(620)
    attn = make_attention("cuda").train()
    expected = explicit_model(attn).train()
    captured = []
    production = attn._cell_fused_lr
    def capture(q, k, v, indices, *args, **kwargs):
        captured.append((q.detach(), k.detach(), v.detach(), indices.detach()))
        return production(q, k, v, indices, *args, **kwargs)
    monkeypatch.setattr(attn, "_cell_fused_lr", capture)
    x = torch.randn(1, 417, 128, device="cuda", requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    got, want = attn(x), expected(z)
    torch.testing.assert_close(got, want, atol=2e-5, rtol=3e-4)
    direction = torch.randn_like(got)
    (got * direction).sum().backward()
    (want * direction).sum().backward()
    torch.testing.assert_close(x.grad, z.grad, atol=4e-5, rtol=4e-4)
    for (name, parameter), (other, target) in zip(attn.named_parameters(), expected.named_parameters()):
        assert name == other
        if parameter.grad is None or target.grad is None:
            assert parameter.grad is None and target.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, target.grad, atol=3e-4, rtol=5e-4, msg=name)
    assert captured
    for q, k, v, indices in captured:
        _, want_indices = constant_reference(attn, q, k, v)
        torch.testing.assert_close(indices[:, :, :q.shape[2]].long().sort(-1).values,
                                   want_indices.sort(-1).values, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("lean", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_prefill_and_decode_match_constant_reference_across_boundaries(lean, dtype):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/scaling"))
    from kv_cache import FSKVCache

    torch.manual_seed(221)
    attn = make_attention("cuda").eval()
    expected = explicit_model(attn).eval()
    x = torch.randn(1, 289, 128, device="cuda")
    cache = FSKVCache(attn, lean_prefill=lean)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        want = expected(x)
        outputs = [attn(x[:, :255], cache=cache)]
        outputs += [attn(x[:, i:i + 1], cache=cache) for i in range(255, x.shape[1])]
        got = torch.cat(outputs, 1)
    tolerance = 0.008 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(got.float(), want.float(), atol=tolerance, rtol=tolerance)


def test_norm_bonus_is_retained_and_local_branch_is_unchanged():
    model = make_attention("cpu").eval()
    q = torch.zeros(1, 2, 384, 64)
    k = torch.zeros_like(q)
    v = torch.randn_like(q)
    k[:, :, :32] = 5
    k[:, :, 32:64] = 4
    k[:, :, 64:96] = 3
    k[:, :, 96:128] = 2
    _, indices = constant_reference(model, q, k, v)
    assert set(indices[0, 0, 383].tolist()) == {0, 1, 2, 3}
    ordinary = copy.deepcopy(model)
    ordinary.log_sigma.fill_(torch.log(torch.tensor(4.0)))
    torch.testing.assert_close(model._short_range(q, k, v, model.sigma),
                               ordinary._short_range(q, k, v, ordinary.sigma), atol=0, rtol=0)
