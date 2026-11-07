import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.rebuttal.resume_contract import stopping_step, validate_fixed_resume


def checkpoint(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"files": {"train": {"sha256": "frozen"}}}))
    args = dict(max_steps=28000, lr=3e-4, min_lr=3e-5, warmup_steps=1000,
                lr_schedule="cosine", seed=42, weight_decay=.1, grad_clip=1,
                optimizer="adamw", fused_adam=False, num_workers=2,
                data_sha256="source", disable_cudnn_sdpa=False,
                gradient_checkpointing=False, rebuttal_data_plan=str(plan),
                per_gpu_batch_size=2, grad_accum_steps=32, attn_type="fsattn")
    config = dict(max_seq_len=4096, dropout=0)
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "step.txt").write_text("1000")
    (folder / "run_config.json").write_text(json.dumps(dict(step=1000, config=config,
        attn_type="fsattn", attn_kwargs={"lr_dense_control": True}, args=args,
        world_size=2, tokens=1000 * 128 * 4096)))
    args["grad_accum_steps"] = 8
    return folder, SimpleNamespace(**args), config


def test_topology_change_preserves_budget(tmp_path):
    folder, args, cfg = checkpoint(tmp_path)
    result = validate_fixed_resume(folder, args, cfg, {"lr_dense_control": True}, 8)
    assert result["sequences_per_update"] == 128
    assert result["previous_world"] == 2
    assert result["rank_rng_preserved"] is False
    assert stopping_step(args.max_steps, 1002, result["step"]) == 1002


@pytest.mark.parametrize("key,value", [("max_steps", 1002), ("lr", 1e-4),
                                      ("seed", 43), ("data_sha256", "wrong")])
def test_changed_recipe_is_rejected(tmp_path, key, value):
    folder, args, cfg = checkpoint(tmp_path)
    setattr(args, key, value)
    with pytest.raises(ValueError, match="recipe differs"):
        validate_fixed_resume(folder, args, cfg, {"lr_dense_control": True}, 8)


def test_changed_batches_and_attention_are_rejected(tmp_path):
    folder, args, cfg = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="attention"):
        validate_fixed_resume(folder, args, cfg, {}, 8)
    args.grad_accum_steps = 4
    with pytest.raises(ValueError, match="global batch"):
        validate_fixed_resume(folder, args, cfg, {"lr_dense_control": True}, 8)
    args.grad_accum_steps = 8
    different = tmp_path / "different.json"
    different.write_text("{}")
    args.rebuttal_data_plan = str(different)
    with pytest.raises(ValueError, match="batch plan"):
        validate_fixed_resume(folder, args, cfg, {"lr_dense_control": True}, 8)


@pytest.mark.parametrize("start,end", [(1000, 1000), (1000, 999), (0, 28001), (0, 0)])
def test_invalid_stopping_step(start, end):
    with pytest.raises(ValueError, match="schedule horizon"):
        stopping_step(28000, end, start)


def legacy_checkpoint(tmp_path):
    folder, args, cfg = checkpoint(tmp_path)
    saved = json.loads((folder / "run_config.json").read_text())
    saved["world_size"] = 8
    saved["args"].update(rebuttal_data_plan=None, grad_accum_steps=8)
    saved["args"].pop("disable_cudnn_sdpa")
    (folder / "run_config.json").write_text(json.dumps(saved))
    plan = dict(schema="freesolo_matched_batches/1", seq_len=4096,
                corpus_sha256="source", seed=42, steps=28000,
                reference_world=8, reference_batch=2, reference_workers=2,
                reference_accumulation=8)
    Path(args.rebuttal_data_plan).write_text(json.dumps(plan))
    args.grad_accum_steps = 4
    return folder, args, cfg, plan


def test_legacy_eight_gpu_stream_can_resume_on_sixteen(tmp_path):
    folder, args, cfg, _ = legacy_checkpoint(tmp_path)
    result = validate_fixed_resume(folder, args, cfg, {"lr_dense_control": True}, 16)
    assert result["legacy_stream_bridge"] is True
    assert result["sequences_per_update"] == 128
    assert result["previous_world"] == 8 and result["world"] == 16


@pytest.mark.parametrize("key,value", [("seed", 43), ("corpus_sha256", "wrong"),
    ("reference_world", 16), ("reference_batch", 1), ("reference_workers", 1),
    ("reference_accumulation", 4), ("seq_len", 2048), ("steps", 1002)])
def test_legacy_bridge_rejects_a_different_stream(tmp_path, key, value):
    folder, args, cfg, plan = legacy_checkpoint(tmp_path)
    plan[key] = value
    Path(args.rebuttal_data_plan).write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        validate_fixed_resume(folder, args, cfg, {"lr_dense_control": True}, 16)
