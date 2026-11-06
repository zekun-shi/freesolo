import pytest
import torch

from experiments.rebuttal.multifact_oracle import replace_required, source_spans


def test_fixed_budget_preserves_targets_and_unaffected_rows():
    selected = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [-1, 2, -1, -1], [1, 2, 3, 4]])
    required = torch.tensor([[2, 7], [-1, -1], [6, 2], [4, 3]])
    result, changed = replace_required(selected, required)
    assert result.tolist() == [[1, 2, 3, 7], [1, 2, 3, 4], [-1, 2, -1, 6], [1, 2, 3, 4]]
    assert changed.tolist() == [1, 0, 1, 0]
    assert selected.tolist()[0] == [1, 2, 3, 4]


def test_four_sources_keep_every_target_and_causal_padding():
    result, changed = replace_required(torch.tensor([[2, 3, 4, 5], [2, 3, 4, 5]]),
                                      torch.tensor([[6, 7, 4, 2], [6, -1, -1, -1]]))
    assert set(result[0].tolist()) == {2, 4, 6, 7}
    assert result[1].tolist() == [2, 3, 4, 6]
    assert changed.tolist() == [2, 1]


def test_source_key_and_value_cross_cell_boundary():
    text = "x" * 29 + "key 12345" + "x" * 30
    offsets = [(0, 0)] + [(i, i + 1) for i in range(len(text))]
    found = source_spans(text, offsets, {"key_desc": "key", "value": "12345"})
    assert found["key_cells"] == [0, 1]
    assert found["value_cells"] == [1]
    assert found["cells"] == [0, 1]
    assert found["value_tokens"] == [34, 35, 36, 37, 38]


def test_reject_missing_ambiguous_or_over_budget():
    with pytest.raises(ValueError, match="one source"):
        source_spans("key key 123", [(i, i + 1) for i in range(11)], {"key_desc": "key", "value": "123"})
    with pytest.raises(ValueError, match="budget"):
        replace_required(torch.zeros(1, 4, dtype=torch.long), torch.arange(5)[None])
