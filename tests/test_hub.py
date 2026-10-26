import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from experiments.scaling import hub
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


@pytest.mark.parametrize("fallback", [False, True, None])
def test_benchmark_rejects_rows_without_a_graph_decode(tmp_path, monkeypatch, fallback):
    # freesolo-benchmark accepts an FS-B row only when its graph decode is recorded as a CUDA-graph run: a row whose
    # decoder fell back to its eager step after a failed capture (decode_graph_fallback true), or that does not say,
    # does not reproduce the graph speed
    folder = tmp_path / "snapshot"
    folder.mkdir()
    config = {"variant": "fs-b", "freesolo_config": "340m"}
    provenance = {"model": "owner/checkpoint", "revision": "a" * 40, "weights_sha256": "0" * 64}
    output = tmp_path / "bench.json"
    row = {"L": 65536, "prefill_graph_ktoks": 100.0, "decode_graph_tps": 50.0}
    if fallback is not None:
        row["decode_graph_fallback"] = fallback
    monkeypatch.setattr(hub, "download_model", lambda *a, **k: (folder, config, provenance))
    monkeypatch.setattr(hub, "configure", lambda *a, **k: None)
    monkeypatch.setattr(hub, "environment", lambda: {})
    monkeypatch.setattr(hub, "subprocess",
                        SimpleNamespace(run=lambda command, check: output.write_text(json.dumps([row]))))
    monkeypatch.setattr(sys, "argv", ["freesolo-benchmark", "--lengths", "65536", "--output", str(output)])
    if fallback is False:
        hub.benchmark_main()
    else:
        with pytest.raises(SystemExit, match="decode_graph_fallback"):
            hub.benchmark_main()


def test_fsb_configure_sets_the_recorded_train_graph_default(monkeypatch):
    # configure("fs-b") sets FSB_TRAIN_GRAPH=0 as the eval drivers do, the value environment() records from the bench
    # switch table (the modules' own default is 1), and keeps a value the caller exported; FS-L leaves it unset
    import os
    for key in ("FS_FUSED_ATTN", "FS_STRICT_CAUSAL", "FS_PRUNE_KERNEL", "FSL_LINEAR_CACHE", "FS_PRUNE_POOL",
                "FS_PRUNE_WINDOW", "FS_PRUNE_BLKCHUNK", "TOKENIZERS_PARALLELISM", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        monkeypatch.setenv(key, "unused")
    monkeypatch.setenv("FSB_TRAIN_GRAPH", "unused")
    monkeypatch.delenv("FSB_TRAIN_GRAPH")        # set first, then delete: teardown removes what configure sets
    configure("fs-l")
    assert "FSB_TRAIN_GRAPH" not in os.environ
    configure("fs-b")
    assert os.environ["FSB_TRAIN_GRAPH"] == "0"
    monkeypatch.setenv("FSB_TRAIN_GRAPH", "1")
    configure("fs-b")
    assert os.environ["FSB_TRAIN_GRAPH"] == "1"
