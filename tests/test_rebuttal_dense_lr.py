"""The dense LR control must equal FS reading every eligible cell."""

import copy
from pathlib import Path
import sys

import pytest
import torch

from src.fsattn.attention import FSAttn


def make_attention(device, dense=False, **overrides):
    kwargs = dict(d_model=128, n_heads=2, window_size=512, causal=True,
                  use_rope=True, rope_long_range=False, max_seq_len=1024,
                  fs_uniform_sr=True, cell_stride=32, cell_topk=1024,
                  sel_strict_causal=1, cell_prune_kernel=0,
                  fs_fused_attn=int(device == "cuda"), cell_fused_select=0,
                  lr_dense_control=dense, bias=False)
    kwargs.update(overrides)
    model = FSAttn(**kwargs).to(device)
    with torch.no_grad():
        model.long_range_scale.fill_(1)
    return model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("length", [96, 257, 417])
def test_dense_lr_matches_all_cell_fs_outputs_and_gradients(device, length):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(51)
    all_cells = make_attention(device).train()
    dense = make_attention(device, dense=True).train()
    dense.load_state_dict(all_cells.state_dict())
    x = torch.randn(1, length, 128, device=device, requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    a, b = all_cells(x), dense(z)
    torch.testing.assert_close(a, b, atol=2e-5, rtol=3e-4)
    target = torch.randn_like(a)
    (a * target).sum().backward()
    (b * target).sum().backward()
    torch.testing.assert_close(x.grad, z.grad, atol=3e-5, rtol=5e-4)
    for (name, p), (other, ref) in zip(all_cells.named_parameters(), dense.named_parameters()):
        assert name == other
        if p.grad is None or ref.grad is None:
            assert p.grad is None and ref.grad is None, name
        else:
            torch.testing.assert_close(p.grad, ref.grad, atol=4e-4, rtol=8e-4, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("lean", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dense_cached_prefill_and_decode_match_all_cell_fs(lean, dtype):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/scaling"))
    from kv_cache import FSKVCache

    torch.manual_seed(17)
    all_cells = make_attention("cuda").eval()
    dense = make_attention("cuda", dense=True).eval()
    dense.load_state_dict(all_cells.state_dict())
    tokens = torch.randn(1, 289, 128, device="cuda")
    cache = FSKVCache(dense, lean_prefill=lean)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        expected = all_cells(tokens)
        whole = dense(tokens)
        got = [dense(tokens[:, :255], cache=cache)]
        got += [dense(tokens[:, i:i + 1], cache=cache) for i in range(255, 289)]
    tolerance = 0.015 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(whole.float(), expected.float(), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(torch.cat(got, 1).float(), expected.float(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("override", [dict(cell_prune_kernel=1), dict(rope_long_range=True),
                                      dict(sel_strict_causal=0), dict(fsb_enabled=True)])
def test_dense_control_rejects_incompatible_routing(override):
    with pytest.raises(ValueError, match="Dense LR control requires"):
        make_attention("cpu", dense=True, **override)


def test_dense_control_cannot_use_truncated_kv_history():
    dense = make_attention("cpu", dense=True)
    q = torch.zeros(1, 2, 1, 64)
    with pytest.raises(ValueError, match="complete eligible KV history"):
        dense._lr_all_eligible(q, q, q, q_offset=256)
