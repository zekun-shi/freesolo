"""Warmup/stable/decay schedules and branch-checkpoint contracts."""

import math


def wsd_lr(step, warmup, max_steps, max_lr, min_lr, cooldown_start):
    """LR for a zero-based update; the last cooldown update uses min_lr."""
    if not 0 <= warmup <= cooldown_start < max_steps:
        raise ValueError("WSD requires 0 <= warmup <= cooldown_start < max_steps")
    if not 0 <= min_lr <= max_lr or max_lr <= 0:
        raise ValueError("WSD requires 0 <= min_lr <= max_lr and max_lr > 0")
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step < cooldown_start:
        return max_lr
    remaining = max_steps - cooldown_start
    fraction = min(1.0, (step - cooldown_start + 1) / remaining)
    return max_lr + fraction * (min_lr - max_lr)


def branch_steps(tokens, tokens_per_step=524288):
    """Round a requested token budget up to updates and cool down for 20%."""
    if tokens <= 0 or tokens_per_step <= 0:
        raise ValueError("Token budgets and update sizes must be positive")
    steps = math.ceil(tokens / tokens_per_step)
    start = steps * 4 // 5
    return {"requested_tokens": tokens, "steps": steps, "cooldown_start": start,
                "cooldown_steps": steps - start, "actual_tokens": steps * tokens_per_step}


def validate_resume(saved, contract, schedule, step):
    """Only identical resumes or cooldowns branched from stable weights are valid."""
    if saved.get("schema") != "freesolo_wsd/1":
        raise ValueError("Resume requires a complete WSD checkpoint with scaling_state.json")
    if saved["step"] != step:
        raise ValueError("Checkpoint step.txt and scaling_state.json disagree")
    if saved["contract"] != contract:
        changed = sorted(k for k in set(saved["contract"]) | set(contract)
                         if saved["contract"].get(k) != contract.get(k))
        raise ValueError(f"Resume changed the training/data contract: {changed}")
    if step >= schedule["max_steps"]:
        raise ValueError("Resume checkpoint has already reached the requested budget")
    old = saved["schedule"]
    if old != schedule and not (step <= old["cooldown_start"] and schedule["cooldown_start"] == step):
        raise ValueError("A new cooldown must start at a stable checkpoint's exact step")
    expected = step * contract["grad_accum_steps"]
    if saved["next_microbatch"] != expected:
        raise ValueError("Checkpoint data cursor does not match its update count")
