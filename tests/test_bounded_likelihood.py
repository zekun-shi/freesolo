"""Completion boundaries, eviction, and candidate isolation for bounded scoring."""
import pytest
import torch

from experiments.rebuttal import bounded_likelihood as adapter
from experiments.scaling.model import FreeSoloLM
from src.fsattn.linear_cache import LinearFSCache


def tiny_model():
    torch.manual_seed(41)
    model = FreeSoloLM(vocab_size=101, d_model=64, n_layers=2, n_heads=2,
        d_ff=128, max_seq_len=512, window_size=16, cell_stride=8,
        cell_topk=2, cell_prune_kernel=1, cell_prune_pool=2, cell_prune_window=1,
        cell_prune_blkchunk=32, sel_strict_causal=1, use_rope=True,
        rope_long_range=False, fs_uniform_sr=True, fs_fused_attn=0,
        cell_fused_select=0).eval()
    return model


@torch.no_grad()
def tokenwise_reference(model, ids, n):
    caches = [LinearFSCache(block.attn.fsattn) for block in model.blocks]
    logs = []
    for i in range(len(ids) - 1):
        logits = model(ids[i:i + 1][None], cache=caches)[0, 0].float()
        if i >= len(ids) - 1 - n:
            logs.append(logits[ids[i + 1]] - torch.logsumexp(logits, dim=-1))
    return torch.stack(logs), caches


@pytest.mark.parametrize('boundary', [127, 128, 129, 383])
@pytest.mark.parametrize('chunk', [1, 17, 64])
def test_completion_likelihood_matches_tokenwise_across_cells_and_eviction(boundary, chunk):
    model = tiny_model()
    ids = torch.randint(0, 101, (boundary + 19,))
    expected, caches = tokenwise_reference(model, ids, 19)
    got = adapter.completion_logprobs(model, ids, 19, chunk_size=chunk, dtype=torch.float32)
    torch.testing.assert_close(got, expected, atol=3e-5, rtol=1e-5)
    assert all(c.pos == len(ids) - 1 and c._top['ids'].shape[-1] <= 2 for c in caches)
    if boundary == 383:
        assert all(c.eligible > c.P for c in caches)


def test_candidates_are_independent_and_no_context_token_is_scored():
    model = tiny_model()
    prefix = torch.randint(0, 101, (129,))
    one = torch.cat([prefix, torch.tensor([4, 5, 6])])
    two = torch.cat([prefix, torch.tensor([7, 8, 9, 10])])
    a = adapter.completion_logprobs(model, one, 3, dtype=torch.float32)
    adapter.completion_logprobs(model, two, 4, dtype=torch.float32)
    b = adapter.completion_logprobs(model, one, 3, dtype=torch.float32)
    assert len(a) == 3 and torch.equal(a, b)
    reference, _ = tokenwise_reference(model, one, 3)
    torch.testing.assert_close(a, reference, atol=3e-5, rtol=1e-5)


def test_refuses_an_unpruned_model():
    model = tiny_model()
    model.blocks[0].attn.fsattn.cell_prune_kernel = 0
    with pytest.raises(ValueError, match='native FS-L'):
        adapter.completion_logprobs(model, [1, 2, 3], 1, dtype=torch.float32)
