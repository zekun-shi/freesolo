import json

import pytest
import torch
from torch import nn

from experiments.scaling.fsb_release import export_calibration, load_union_calibration
from experiments.scaling.fsb_tau_calib import tau_sha256


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fsb_enabled = True
        self.register_buffer("fsb_tau", torch.zeros(3))
        self.register_buffer("fsb_tau_union", torch.zeros(3), persistent=False)


@pytest.fixture
def calibration(tmp_path):
    tau = torch.tensor([[0.1, 1.125, 17.25], [0.0003, 2.5, 3.75]])
    meta = {"ckpt_sha256": "a" * 64, "quantile": 0.5, "tau_sha256": tau_sha256(tau),
            "n_batches": 2, "length": 4096, "seed": 1234, "train_frac": 0.98, "chunk_ids": [3, 7],
            "data": str(tmp_path / "private-corpus.bin"), "cache_file": str(tmp_path / "cache.pt")}
    cache = tmp_path / "cache.pt"
    torch.save({"tau": tau, "meta": meta}, cache)
    path = tmp_path / "release" / "union_calibration.json"
    export_calibration(cache, path)
    return path, tau, nn.Sequential(Layer(), Layer())


def test_portable_calibration_preserves_float32_values(calibration):
    path, tau, model = calibration
    payload = load_union_calibration(model, path, checkpoint_sha256="a" * 64, quantile=0.5)
    assert torch.equal(torch.stack([layer.fsb_tau_union for layer in model]), tau)
    assert torch.equal(torch.stack([layer.fsb_tau for layer in model]), torch.zeros_like(tau))
    assert payload["tau_sha256"] == tau_sha256(tau)
    text = path.read_text()
    assert "private-corpus" not in text and str(path.parent.parent) not in text
    assert "cache_file" not in text


@pytest.mark.parametrize("change", ["checkpoint", "quantile", "value", "shape", "nan", "extra"])
def test_invalid_calibration_never_changes_model(calibration, change):
    path, _, model = calibration
    payload = json.loads(path.read_text())
    if change == "checkpoint":
        payload["checkpoint_sha256"] = "b" * 64
    elif change == "quantile":
        payload["quantile"] = 0.25
    elif change == "value":
        payload["tau"][0][0] += 1
    elif change == "shape":
        payload["tau"].pop()
    elif change == "nan":
        payload["tau"][0][0] = float("nan")
    else:
        payload["local_path"] = "private"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_union_calibration(model, path, checkpoint_sha256="a" * 64, quantile=0.5)
    assert not any(layer.fsb_tau_union.count_nonzero() for layer in model)


def test_export_does_not_overwrite(calibration):
    path, _, _ = calibration
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        export_calibration(path.parent.parent / "cache.pt", path)
    assert path.read_bytes() == before


def test_benchmark_loads_portable_thresholds_without_a_corpus(calibration, monkeypatch):
    from types import SimpleNamespace
    from experiments.scaling import bench_efficiency

    path, tau, model = calibration
    monkeypatch.setattr(bench_efficiency, "_file_sha256", lambda _: "a" * 64)
    (path.parent / "model.safetensors").write_bytes(b"test checkpoint")
    args = SimpleNamespace(checkpoint=str(path.parent), fsb_union_tau_quantile=0.5,
                           fsb_union_tau_file=str(path), fsb_tau_calib_data=None)
    result = bench_efficiency.apply_fsb_tau(model, args, str(path.parent / "bench.json"))
    assert result["tau_sha256"] == tau_sha256(tau)
    assert torch.equal(torch.stack([layer.fsb_tau_union for layer in model]), tau)
