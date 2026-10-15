import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.scaling.hub import configure, download_model


@pytest.fixture
def hub_snapshot(tmp_path, monkeypatch):
    import huggingface_hub
    (tmp_path / "model.safetensors").write_bytes(b"checkpoint bytes")
    digest = hashlib.sha256(b"checkpoint bytes").hexdigest()
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "freesolo", "variant": "fs-b", "weights_sha256": digest,
    }))
    (tmp_path / "union_calibration.json").write_text("{}")
    info = SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(
        rfilename="model.safetensors", lfs=SimpleNamespace(sha256=digest))])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=lambda *a, **kw: info))
    calls = []
    def snapshot(*args, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    return tmp_path, calls


def test_hub_pins_download_to_resolved_revision(hub_snapshot):
    root, calls = hub_snapshot
    folder, config, provenance = download_model("owner/checkpoint")
    assert folder == root and config["variant"] == "fs-b"
    assert calls[0]["revision"] == provenance["revision"] == "a" * 40


def test_hub_rejects_changed_weights(hub_snapshot):
    root, _ = hub_snapshot
    (root / "model.safetensors").write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="SHA256"):
        download_model("owner/checkpoint")


def test_hub_requires_fsb_calibration(hub_snapshot):
    root, _ = hub_snapshot
    (root / "union_calibration.json").unlink()
    with pytest.raises(ValueError, match="union_calibration"):
        download_model("owner/checkpoint")


def test_masked_and_bounded_fsl_are_explicit(monkeypatch):
    import os
    for key in ("FS_FUSED_ATTN", "FS_STRICT_CAUSAL", "FS_PRUNE_KERNEL", "FSL_LINEAR_CACHE",
                "FS_PRUNE_POOL", "FS_PRUNE_WINDOW", "FS_PRUNE_BLKCHUNK", "TOKENIZERS_PARALLELISM",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        monkeypatch.setenv(key, "unused")
    configure("fs-l")
    assert os.environ["FSL_LINEAR_CACHE"] == "0"
    configure("fs-l", bounded_cache=True)
    assert os.environ["FSL_LINEAR_CACHE"] == "1"
    configure("fs-l", device="cpu")
    assert os.environ["FS_FUSED_ATTN"] == "0"
    with pytest.raises(ValueError, match="FS-L"):
        configure("fs-b", bounded_cache=True)
