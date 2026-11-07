"""Independent scalar, mask and actual-attention checks for control costs."""
from collections import Counter
import math

import pytest

from experiments.rebuttal.control_cost import (
    Geometry, attention_work, model_cost, parameter_count, prefix_pairs,
)
from experiments.rebuttal.baseline_cost import dense_attention
from test_custom_cost_formulas import Scalar, matrix_dot, reduction


def test_pairs_match_explicit_token_eligibility_across_boundaries():
    for n in [1, 7, 8, 9, 17, 31, 32, 33, 127, 128, 129, 511, 512, 513, 514]:
        masks = {
            'global': lambda q, k: k <= q,
            'local': lambda q, k: 0 <= q-k <= 4,
            'dense_lr': lambda q, k: (k//2+1)*2 <= (q//8)*8,
        }
        for mode, allowed in masks.items():
            explicit = sum(allowed(q, k) for q in range(n) for k in range(n))
            assert prefix_pairs(n, mode, window=4, query_block=8) == explicit
        for steps in [1, 7, 9]:
            for mode, allowed in masks.items():
                explicit = sum(allowed(q, k) for q in range(n, n+steps) for k in range(n+steps))
                count = attention_work(mode, n, steps=steps,
                    geometry=Geometry(hidden=8, heads=2, window=4, query_block=8, cell_stride=2))
                assert count['valid_token_pairs'] == 2*explicit


def test_attention_arithmetic_matches_scalar_qk_softmax_pv():
    g = Geometry(hidden=8, heads=2, window=4, query_block=8, cell_stride=2)
    for mode in ['global', 'local', 'dense_lr']:
        Scalar.counts.clear()
        for q in range(17):
            keys = q+1 if mode == 'global' else min(q+1, 5) if mode == 'local' else q//8*8
            if not keys:
                continue
            for _ in range(g.heads):
                query = [Scalar(.1)]*g.width
                logits = [matrix_dot(query, [Scalar(.2)]*g.width)*.5 for _ in range(keys)]
                maximum = max(logits)
                weights = [(x-maximum).special('exp', math.exp) for x in logits]
                denominator = reduction(weights)
                weights = [x/denominator for x in weights]
                output = [matrix_dot(weights, [Scalar(.3)]*keys) for _ in range(g.width)]
                assert len(output) == g.width
        estimate = attention_work(mode, 17, geometry=g)
        assert dict(Scalar.counts) == {k: estimate[k] for k in ['flops','exp','fp_comparisons']}


def test_global_formula_agrees_with_retained_baseline_convention():
    for n in [4096, 8192, 131072]:
        for steps in [None, 257]:
            old = dense_attention(16, 64, n, steps)
            new = attention_work('global', n, steps=steps)
            assert all(new[k] == old[k] for k in ['flops', 'exp', 'fp_comparisons'])
            assert new['valid_token_pairs'] == old['causal_token_pairs']


def test_control_layers_parameters_and_component_totals():
    assert parameter_count('nope') == parameter_count('swan') == 341099520
    assert parameter_count('dense_lr') == 341101824
    big = Geometry(hidden=1536, ffn=4224)
    assert parameter_count('swan', big) == 742860288
    for arm in ['nope', 'swan', 'dense_lr']:
        for steps in [None, 257]:
            result = model_cost(arm, 4096, steps=steps)
            counted = Counter()
            for values in result['components'].values():
                counted.update(values)
            assert dict(counted) == result['total']
    swan = model_cost('swan', 4096)
    global_one = attention_work('global', 4096)
    local_one = attention_work('local', 4096)
    assert swan['components']['global_attention']['valid_token_pairs'] == 6*global_one['valid_token_pairs']
    assert swan['components']['local_attention']['valid_token_pairs'] == 18*local_one['valid_token_pairs']
    dense = model_cost('dense_lr', 4096, steps=257)
    # Eight completed cells over decode; do not count existing cells again.
    assert dense['components']['cache_cell_statistics']['sqrt'] == 24*16*8*32


def test_pair_counts_match_executed_control_attention_calls(monkeypatch):
    torch = pytest.importorskip('torch')
    from experiments.scaling.dense_attention import DenseAttention
    from src.fsattn.attention import FSAttn

    recorded = []
    def observe(q, k, v, attn_mask=None, is_causal=False, **kwargs):
        nq, nk, heads = q.shape[-2], k.shape[-2], q.shape[1]
        if attn_mask is not None:
            mask = attn_mask if attn_mask.dtype == torch.bool else torch.isfinite(attn_mask)
            mask = mask.expand(1, heads, nq, nk)
            pairs = int(mask.sum())
        elif is_causal:
            pairs = heads*sum(min(i+1, nk) for i in range(nq))
        else:
            pairs = heads*nq*nk
        recorded.append(pairs)
        return torch.zeros_like(q)
    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', observe)
    g = Geometry(hidden=8, heads=2, window=4, query_block=8, cell_stride=2)
    x = torch.ones(1, 17, 8)
    for window, mode in [(None, 'global'), (4, 'local')]:
        recorded.clear()
        attn = DenseAttention(8, 2, 32, use_rope=window is not None, window_size=window).eval()
        with torch.no_grad():
            attn(x)
        assert sum(recorded) == attention_work(mode, 17, geometry=g)['valid_token_pairs']
    dense = FSAttn(8, 2, window_size=4, use_rope=True, rope_long_range=False,
                  max_seq_len=32, causal=True, fs_uniform_sr=True, cell_stride=2,
                  sel_strict_causal=1, cell_prune_kernel=0, lr_dense_control=True)
    dense._cell_block = 8
    for offset, count in [(0, 17), (8, 1), (9, 7)]:
        recorded.clear()
        q = torch.ones(1, 2, count, 4)
        kv = torch.ones(1, 2, offset+count, 4)
        dense._lr_all_eligible(q, kv, kv, q_offset=offset)
        expected = 2*(prefix_pairs(offset+count, 'dense_lr', query_block=8)
                      - prefix_pairs(offset, 'dense_lr', query_block=8))
        assert sum(recorded) == expected
