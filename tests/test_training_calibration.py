import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.scaling.config import build_attn_kwargs
from experiments.scaling.fsb_tau_calib import tau_sha256
from experiments.scaling.training_calibration import (
    load_union_training, save_union_training, validate_union_training,
)


def args(**updates):
    values = dict(fsb=True, fsb_tree="forest", fsb_policy="fixed", fsb_union_width=16,
                  fsb_read_top=4, fsb_union_tau_quantile=0.5, fsb_union_tau_file="tau.json",
                  init_from="checkpoint", resume=None)
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("changes", [dict(fsb=False), dict(fsb_tree="rebuild"),
    dict(fsb_policy="learned"), dict(fsb_union_width=0), dict(fsb_read_top=0),
    dict(fsb_union_tau_file=None), dict(fsb_union_tau_quantile=0),
    dict(fsb_union_tau_quantile=None), dict(init_from=None), dict(resume="checkpoint")])
def test_reject_invalid_training_configuration(changes):
    with pytest.raises(ValueError):
        validate_union_training(args(**changes))


def test_default_is_unchanged():
    a = SimpleNamespace()
    validate_union_training(a)
    assert load_union_training(None, a) is None


def test_filtered_quantile_reaches_model():
    a = args(attn_type="fsattn", window_size=512, long_range_variant="cell_topk", sigma_init=1)
    assert build_attn_kwargs(a)[1]["fsb_union_tau_quantile"] == 0.5


def test_load_save_and_frozen_threshold_check(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint identity")
    layer = torch.nn.Module()
    layer.fsb_enabled = True
    layer.register_buffer("fsb_tau_union", torch.zeros(2), persistent=False)
    layer.register_buffer("fsb_tau", torch.zeros(2))
    model = torch.nn.ModuleList([layer])
    tau = torch.tensor([[0.25, 0.75]])
    payload = dict(schema="freesolo_union_calibration/1",
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   quantile=0.5, tau_sha256=tau_sha256(tau), tau=tau.tolist(), calibration={})
    path = tmp_path / "tau.json"
    path.write_text(json.dumps(payload))
    a = args(init_from=str(tmp_path), fsb_union_tau_file=str(path))
    loaded = load_union_training(model, a)
    assert torch.equal(layer.fsb_tau_union, tau[0])
    assert layer.fsb_tau.count_nonzero() == 0
    save_union_training(model, loaded, tmp_path)
    assert json.loads((tmp_path / "training_union_calibration.json").read_text()) == payload
    layer.fsb_tau_union.add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        save_union_training(model, loaded, tmp_path)
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="different checkpoint"):
        load_union_training(model, a)
