"""Released tied-weight checkpoints load completely; missing independent weights fail."""
import pytest
import torch
from safetensors.torch import save_file

from experiments.scaling.longctx_infer import load_model
from experiments.scaling.model import FreeSoloLM, MODEL_CONFIGS


@pytest.fixture
def released_checkpoint(tmp_path, monkeypatch):
    config = dict(d_model=32, n_layers=1, n_heads=4, d_ff=64,
                  max_seq_len=32, vocab_size=64)
    monkeypatch.setitem(MODEL_CONFIGS, 'checkpoint_alias_test', config)
    model = FreeSoloLM(attn_type='standard', use_rope=False, **config)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()
             if key != 'lm_head.weight'}
    save_file(state, str(tmp_path/'model.safetensors'))
    return tmp_path, model, state


def test_released_tied_head_alias_is_loaded(released_checkpoint):
    path, original, _ = released_checkpoint
    loaded = load_model(str(path), config='checkpoint_alias_test', attn_type='standard',
                        use_rope=False, device='cpu', strict=False, verbose=False)
    assert loaded.lm_head.weight is loaded.tok_emb.weight
    torch.testing.assert_close(loaded.lm_head.weight, original.lm_head.weight, rtol=0, atol=0)
    for name, parameter in loaded.named_parameters():
        torch.testing.assert_close(parameter, dict(original.named_parameters())[name], rtol=0, atol=0)


def test_missing_independent_weight_still_fails(released_checkpoint):
    path, original, state = released_checkpoint
    independent = next(name for name, _ in original.named_parameters()
                       if name.startswith('blocks.') and name.endswith('.weight'))
    state.pop(independent)
    save_file(state, str(path/'model.safetensors'))
    with pytest.raises(RuntimeError, match='parameters were not found in the checkpoint'):
        load_model(str(path), config='checkpoint_alias_test', attn_type='standard',
                   use_rope=False, device='cpu', strict=False, verbose=False)
