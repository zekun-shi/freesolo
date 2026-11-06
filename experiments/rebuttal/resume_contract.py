"""Checks for finite handoffs of the fixed-batch rebuttal training runs."""

import json
from pathlib import Path


def stopping_step(max_steps, stop_after_step=None, start_step=0):
    end = max_steps if stop_after_step is None else stop_after_step
    if not 0 <= start_step < end <= max_steps:
        raise ValueError("Require 0 <= restored step < stopping step <= schedule horizon")
    return end


def validate_fixed_resume(folder, args, config, attn_kwargs, world):
    """Allow topology changes while keeping the optimizer recipe and batch stream."""
    folder = Path(folder)
    saved = json.loads((folder / "run_config.json").read_text())
    step = int((folder / "step.txt").read_text())
    if saved["step"] != step or saved["config"] != config:
        raise ValueError("Resume checkpoint step or model configuration differs")
    if saved["attn_type"] != args.attn_type or saved["attn_kwargs"] != attn_kwargs:
        raise ValueError("Resume attention configuration differs")
    old = saved["args"]
    keys = ("max_steps", "lr", "min_lr", "warmup_steps", "lr_schedule", "seed",
            "weight_decay", "grad_clip", "optimizer", "fused_adam", "num_workers",
            "data_sha256", "disable_cudnn_sdpa", "gradient_checkpointing")
    changed = [key for key in keys if old.get(key) != getattr(args, key)]
    if changed:
        raise ValueError("Resume training recipe differs: " + ", ".join(changed))
    previous = json.loads(Path(old["rebuttal_data_plan"]).read_text())
    current = json.loads(Path(args.rebuttal_data_plan).read_text())
    if previous != current:
        raise ValueError("Resume global batch plan differs")
    old_batch = saved["world_size"] * old["per_gpu_batch_size"] * old["grad_accum_steps"]
    new_batch = world * args.per_gpu_batch_size * args.grad_accum_steps
    if old_batch != new_batch or saved["tokens"] != step * new_batch * config["max_seq_len"]:
        raise ValueError("Resume global batch or cumulative token count differs")
    if world != saved["world_size"] and config.get("dropout", 0) != 0:
        raise ValueError("Topology handoff requires zero dropout")
    return dict(checkpoint=str(folder), step=step, previous_world=saved["world_size"],
                world=world, sequences_per_update=new_batch,
                rank_rng_preserved=world == saved["world_size"],
                data_order="fixed global batch plan")
