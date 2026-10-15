import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from experiments.scaling.dense_attention import DenseAttention
from experiments.scaling.scaling_data import ScalingTokenDataset
from experiments.scaling.scaling_schedule import branch_steps, validate_resume, wsd_lr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scaling_launcher", ROOT / "scripts/train_scaling_law.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize("size", launcher.BUDGETS)
@pytest.mark.parametrize("world", [1, 4, 8, 16, 32])
def test_launch_grid_keeps_batch_and_annealing_budgets(size, world, tmp_path):
    args = launcher.parser().parse_args([
        "--arm", "fs", "--size", size, "--data", str(tmp_path / "data with spaces.bin"),
        "--gpus-per-node", str(world), "--dry-run"])
    plan = launcher.build_plan(args)
    assert world * plan["grad_accum_steps"] * 4096 == 524288
    for branch in plan["grid"]:
        assert 0 <= branch["actual_tokens"] - branch["requested_tokens"] < 524288
        assert branch["cooldown_start"] >= 2000
        assert abs(branch["cooldown_steps"] / branch["steps"] - 0.2) < 1 / branch["steps"]
    args.phase, args.budget = "cooldown", launcher.BUDGETS[size][0]
    branch = launcher.build_plan(args)
    assert branch["resume"].endswith(f"_step{branch['endpoint']['cooldown_start']}")
    assert branch["command"][branch["command"].index("--pretokenized") + 1].endswith("data with spaces.bin")


def test_invalid_topology_and_largest_cooldown_refused():
    args = launcher.parser().parse_args(["--arm", "tfpp", "--size", "340m", "--data", "/data",
                                        "--nnodes", "1", "--node-rank", "0", "--master-addr", "127.0.0.1"])
    args.gpus_per_node = 3
    with pytest.raises(ValueError, match="fixed"):
        launcher.build_plan(args)
    args.gpus_per_node, args.phase, args.budget = 8, "cooldown", "30"
    with pytest.raises(ValueError, match="already annealed"):
        launcher.build_plan(args)
    args.phase, args.budget, args.nnodes = "trunk", None, 2
    with pytest.raises(ValueError, match="reachable"):
        launcher.build_plan(args)


def test_wsd_anneals_only_after_saved_stable_weights():
    values = [wsd_lr(s, 2, 10, 1.0, 0.1, 8) for s in range(10)]
    assert values == pytest.approx([0.5, 1, 1, 1, 1, 1, 1, 1, 0.55, 0.1])
    assert branch_steps(15_000_000_000)["steps"] == 28611
    with pytest.raises(ValueError):
        wsd_lr(0, 3, 10, 1, 0.1, 2)


def test_resume_rejects_wrong_data_layout_and_annealed_parent():
    contract = {"grad_accum_steps": 2, "seed": 42, "data_sha256": "a" * 64, "world_size": 2}
    saved = {"schema": "freesolo_wsd/1", "step": 4, "next_microbatch": 8, "contract": contract,
                 "schedule": {"max_steps": 10, "cooldown_start": 8}}
    validate_resume(saved, contract, {"max_steps": 6, "cooldown_start": 4}, 4)
    validate_resume(saved, contract, saved["schedule"], 4)
    for key, wrong in [("world_size", 1), ("seed", 43), ("data_sha256", "b" * 64)]:
        with pytest.raises(ValueError, match="contract"):
            validate_resume(saved, dict(contract, **{key: wrong}), saved["schedule"], 4)
    bad = copy.deepcopy(saved)
    bad["schedule"]["cooldown_start"] = 3
    with pytest.raises(ValueError, match="stable"):
        validate_resume(bad, contract, {"max_steps": 6, "cooldown_start": 4}, 4)


@pytest.mark.parametrize("workers", [0, 2])
def test_global_data_order_disjoint_and_resume_exact(tmp_path, workers):
    data = tmp_path / "tokens.bin"
    np.repeat(np.arange(1000, dtype="<u2"), 9).tofile(data)
    full = []
    for rank in range(2):
        dataset = ScalingTokenDataset(data, 8, rank, 2, 2, seed=42)
        batches = list(DataLoader(dataset, batch_size=2, num_workers=workers))
        full.append([x[:, 0].tolist() for x, _ in batches])
        dataset.start_microbatch = 7
        resumed = [x[:, 0].tolist() for x, _ in DataLoader(dataset, batch_size=2, num_workers=workers)]
        assert resumed == full[rank][7:]
    all_ids = [cid for per_rank in full for batch in per_rank for cid in batch]
    assert len(set(all_ids)) == len(all_ids) and max(all_ids) < 990
    order = np.arange(990)
    np.random.default_rng(42).shuffle(order)
    assert [cid for batch in zip(*full) for rank_batch in batch for cid in rank_batch] == order[:988].tolist()
    val = ScalingTokenDataset(data, 8, 0, 2, 2, validation_chunks=8)
    assert [x[0].item() for x, _ in val] == [990, 991, 994, 995]


def test_data_verification_receipt_invalidated_on_change(tmp_path, monkeypatch):
    data, receipt = tmp_path / "data", tmp_path / "verified.json"
    data.write_bytes(b"original token bytes")
    monkeypatch.setattr(launcher, "DATA_BYTES", data.stat().st_size)
    monkeypatch.setattr(launcher, "DATA_SHA256", hashlib.sha256(data.read_bytes()).hexdigest())
    assert launcher.verify_data(data, receipt) == launcher.verify_data(data, receipt)
    data.write_bytes(b"tampered token bytes")
    # Simulate a filesystem whose identity fields did not change between writes.
    saved = json.loads(receipt.read_text())
    stat = data.stat()
    saved["identity"].update(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                             ctime_ns=stat.st_ctime_ns, inode=stat.st_ino, device=stat.st_dev)
    receipt.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="checksum mismatch"):
        launcher.verify_data(data, receipt)


def test_archive_source_identity_detects_code_changes(tmp_path):
    from experiments.scaling.provenance import source_revision
    source = tmp_path / "experiments" / "model.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")
    before = source_revision(tmp_path)
    assert before.startswith("archive-sha256:") and source_revision(tmp_path) == before
    source.write_text("value = 2\n")
    assert source_revision(tmp_path) != before


def test_dense_attention_causality_and_softmax_reference():
    torch.manual_seed(42)
    layer = DenseAttention(32, 4, 32).double()
    x = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)
    actual = layer(x)
    perturbed = x.detach().clone()
    perturbed[:, 4:] += 20
    torch.testing.assert_close(actual[:, :4], layer(perturbed)[:, :4], rtol=0, atol=0)
    actual[:, :4].sum().backward()
    assert torch.count_nonzero(x.grad[:, 4:]) == 0
    q, k, v = [p(x).reshape(2, 7, 4, 8).transpose(1, 2) for p in (layer.W_q, layer.W_k, layer.W_v)]
    angle = torch.arange(7).float()[:, None] / (10000 ** (torch.arange(4).float() / 4))
    cos, sin = angle.cos(), angle.sin()
    def rotate(t):
        a, b = t.chunk(2, -1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)
    scores = (rotate(q) @ rotate(k).transpose(-1, -2)) / 8**0.5
    scores.masked_fill_(torch.ones(7, 7, dtype=torch.bool).triu(1), -torch.inf)
    reference = layer.W_o((scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 32))
    torch.testing.assert_close(actual, reference, rtol=1e-7, atol=1e-7)


def test_model_configuration_counts_without_allocating_weights():
    from experiments.scaling.model import MODEL_CONFIGS, FreeSoloLM
    with torch.device("meta"):
        for size in ("150m", "340m", "800m", "1.3b_fs"):
            cfg = dict(MODEL_CONFIGS[size], vocab_size=32000)
            dense = FreeSoloLM(attn_type="standard", **cfg)
            width, layers, ffn = cfg["d_model"], cfg["n_layers"], cfg["d_ff"]
            expected = 32000 * width + layers * (4 * width**2 + 3 * width * ffn + 2 * width) + width
            assert dense.count_params() == expected
            fs = FreeSoloLM(attn_type="fsattn", fs_uniform_sr=True, rope_long_range=False,
                            cell_stride=32, cell_topk=4, window_size=512, sel_strict_causal=1, **cfg)
            assert fs.count_params() == expected + layers * (2 * cfg["n_heads"] + width // cfg["n_heads"])
            if size == "150m":
                assert 149_000_000 <= expected < 151_000_000


def test_full_optimizer_resume_matches_uninterrupted_cooldown(tmp_path):
    pytest.importorskip("accelerate")
    from safetensors.torch import load_file
    data = tmp_path / "tiny.bin"
    np.random.default_rng(5).integers(0, 64, 9000, dtype="<u2").tofile(data)
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    common = [sys.executable, "-m", "experiments.scaling.train_dist", "--config", "debug",
              "--attn_type", "standard", "--n_layers", "1", "--vocab", "64", "--max_seq_len", "8",
              "--lr_schedule", "wsd", "--pretokenized", str(data), "--data_sha256", digest,
              "--per_gpu_batch_size", "1", "--grad_accum_steps", "2", "--num_workers", "0",
              "--warmup_steps", "2", "--lr", "0.001", "--min_lr", "0.0001", "--log_every", "1",
              "--save_every", "99", "--eval_every", "99", "--validation_chunks", "4", "--seed", "42"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FS_", "FSB_"))}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", WANDB_MODE="disabled")
    def run(name, steps, cooldown, extra=()):
        result = subprocess.run(common + ["--run_name", name, "--output_dir", str(tmp_path / name),
                                          "--max_steps", str(steps), "--cooldown_start_step", str(cooldown),
                                          *extra], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90,
                                check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        return tmp_path / name / f"{name}_step{steps}"
    uninterrupted = run("continuous", 6, 4)
    run("trunk", 8, 6, ["--checkpoint_steps", "4"])
    branch = run("branch", 6, 4, ["--resume", str(tmp_path / "trunk/trunk_step4")])
    a, b = load_file(str(uninterrupted / "model.safetensors")), load_file(str(branch / "model.safetensors"))
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    state = json.loads((branch / "scaling_state.json").read_text())
    assert state["step"] == 6 and state["next_microbatch"] == 12 and state["tokens"] == 96
    assert state["validation_loss"] is not None
