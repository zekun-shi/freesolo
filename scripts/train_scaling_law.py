#!/usr/bin/env python3
"""Launch a matched FS/TF++ WSD trunk or an annealed branch."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.scaling.scaling_schedule import branch_steps
from experiments.scaling.provenance import source_revision

DATA_BYTES = 221647106208
DATA_SHA256 = "a22c79baec4f8b53217fde674d78f6e4f01cbaa32120c51f6936ace58ee0c5c0"
TOKENS_PER_UPDATE = 524288
BUDGETS = {
    "150m": ["1.9", "3.75", "7.5", "15"],
    "340m": ["3.75", "7.5", "15", "30"],
    "800m": ["7.5", "15", "30", "60"],
    "1.3b_fs": ["12.5", "25", "50", "100"],
}


def verify_data(path, receipt):
    """Verify content on every launch; receipts record evidence, not a hash cache."""
    path = Path(path).resolve(strict=True)
    before = path.stat()
    if before.st_size != DATA_BYTES:
        raise ValueError(f"Expected the 100BT Llama-32k corpus ({DATA_BYTES} bytes), got {before.st_size}")
    identity = {"path": str(path), "size": before.st_size, "mtime_ns": before.st_mtime_ns,
                    "ctime_ns": before.st_ctime_ns, "inode": before.st_ino, "device": before.st_dev}
    digest = hashlib.sha256()
    print(f"Verifying the full training corpus: {path}", flush=True)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_dev) != (
            before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino, before.st_dev):
        raise ValueError("Training data changed during verification")
    if digest.hexdigest() != DATA_SHA256:
        raise ValueError(f"Training corpus checksum mismatch: {digest.hexdigest()}")
    result = {"identity": identity, "sha256": digest.hexdigest()}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(receipt.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(receipt)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["fs", "tfpp"], required=True)
    p.add_argument("--size", choices=BUDGETS, required=True)
    p.add_argument("--phase", choices=["trunk", "cooldown"], default="trunk")
    p.add_argument("--budget", help="cooldown endpoint in billions of tokens, from the size's grid")
    p.add_argument("--data", default=os.environ.get("DATA"), help="verified fineweb_edu_llama32k_100bt.bin")
    p.add_argument("--output-root", default=os.environ.get("RUNS", "./runs/scaling"))
    p.add_argument("--nnodes", type=int, default=int(os.environ.get("NNODES", "1")))
    p.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("GPUS_PER_NODE", "8")))
    p.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    p.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    p.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29531")))
    p.add_argument("--per-gpu-batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr-multiplier", type=float, choices=[0.5, 1.0, 2.0], default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--resume", help="full checkpoint for recovery; cooldown otherwise finds its trunk checkpoint")
    p.add_argument("--dry-run", action="store_true", help="print exact budgets, environment and command; no I/O or launch")
    return p


def build_plan(args):
    if not args.data:
        raise ValueError("Provide --data or DATA with the tokenized 100BT corpus path")
    if min(args.nnodes, args.gpus_per_node, args.per_gpu_batch_size, args.save_every, args.eval_every) < 1:
        raise ValueError("Node, GPU, batch and checkpoint/evaluation intervals must be positive")
    if args.num_workers < 0 or not 0 <= args.node_rank < args.nnodes:
        raise ValueError("Invalid worker count or node rank")
    world = args.nnodes * args.gpus_per_node
    micro_tokens = world * args.per_gpu_batch_size * 4096
    if TOKENS_PER_UPDATE % micro_tokens:
        raise ValueError("GPU count and batch size must divide the fixed 524288-token update")
    if args.nnodes > 1 and args.master_addr in ("127.0.0.1", "localhost"):
        raise ValueError("Multi-node training requires the rank-0 node's reachable --master-addr")
    accum = TOKENS_PER_UPDATE // micro_tokens
    grid = [dict(label=b, **branch_steps(int(Decimal(b) * 10**9))) for b in BUDGETS[args.size]]
    run = f"scaling_{args.arm}_{args.size}_lr{args.lr_multiplier:g}_seed{args.seed}"
    output_root = Path(args.output_root).expanduser().resolve()
    trunk_output = output_root / run
    if args.phase == "cooldown":
        if args.budget is None:
            raise ValueError("A cooldown requires --budget, in billions of tokens")
        selected = [b for b in grid if Decimal(b["label"]) == Decimal(args.budget)]
        if not selected:
            raise ValueError(f"Choose a budget from {BUDGETS[args.size]}")
        endpoint = selected[0]
        if endpoint == grid[-1]:
            raise ValueError("The largest budget is already annealed at the end of the trunk")
        resume = args.resume or str(trunk_output / f"{run}_step{endpoint['cooldown_start']}")
        run += "_cooldown_" + endpoint["label"].replace(".", "p") + "b"
        saves = [endpoint["steps"]]
    else:
        if args.budget:
            raise ValueError("--budget is used only with --phase cooldown")
        endpoint = grid[-1]
        resume = args.resume
        saves = [b["cooldown_start"] for b in grid]
    output = output_root / run
    lr = (3e-4 if args.size in ("150m", "340m") else 4e-4) * args.lr_multiplier
    env = {"FS_FUSED_ATTN": "1", "FS_STRICT_CAUSAL": "1", "FS_PRUNE_KERNEL": "0",
               "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled"}
    command = [sys.executable, "-m", "accelerate.commands.launch"]
    if world > 1:
        command += ["--multi_gpu"]
    command += ["--num_machines", str(args.nnodes), "--num_processes", str(world),
                "--machine_rank", str(args.node_rank), "--main_process_ip", args.master_addr,
                "--main_process_port", str(args.master_port), "--same_network", "--mixed_precision", "bf16",
                "--module", "experiments.scaling.train_dist",
                "--config", args.size, "--attn_type", "fsattn" if args.arm == "fs" else "standard",
                "--vocab", "32000", "--max_seq_len", "4096",
                "--per_gpu_batch_size", str(args.per_gpu_batch_size), "--grad_accum_steps", str(accum),
                "--lr_schedule", "wsd", "--max_steps", str(endpoint["steps"]),
                "--cooldown_start_step", str(endpoint["cooldown_start"]),
                "--warmup_steps", "2000", "--lr", str(lr), "--min_lr", str(lr / 10),
                "--checkpoint_steps", ",".join(map(str, saves)), "--save_every", str(args.save_every),
                "--eval_every", str(args.eval_every), "--validation_chunks", "2560",
                "--weight_decay", "0.1", "--grad_clip", "1.0", "--seed", str(args.seed),
                "--pretokenized", str(Path(args.data).expanduser().resolve()), "--data_sha256", DATA_SHA256,
                "--num_workers", str(args.num_workers), "--run_name", run, "--output_dir", str(output)]
    if args.arm == "fs":
        command += ["--long_range_variant", "cell_topk", "--cell_select", "score", "--cell_topk", "4",
                    "--cell_stride", "32", "--sel_strict_causal", "1", "--fs_nope_longrange",
                    "--fs_uniform_sr", "--window_size", "512", "--require_freesolo"]
    for flag in ("compile", "gradient_checkpointing", "fused_adam"):
        if getattr(args, flag):
            command.append("--" + flag)
    if resume:
        command += ["--resume", str(Path(resume).expanduser().resolve())]
    return {"grid": grid, "endpoint": endpoint, "run_name": run, "output_dir": str(output),
                "resume": resume, "tokens_per_update": TOKENS_PER_UPDATE, "world_size": world,
                "grad_accum_steps": accum, "environment": env, "command": command, "cwd": str(ROOT)}


def main():
    p = parser()
    args = p.parse_args()
    try:
        plan = build_plan(args)
        print(json.dumps(plan, indent=2), flush=True)
        print("\n" + shlex.join(plan["command"]), flush=True)
        if args.dry_run:
            return 0
        # Launches use committed code so collaborator runs identify one reproducible revision.
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT):
            raise ValueError("Commit the working tree before launching a scaling run")
        revision = source_revision(ROOT)
        output = Path(plan["output_dir"])
        if not plan["resume"] and any(output.glob(f"{plan['run_name']}_*")):
            raise ValueError("Output already contains checkpoints; use --resume or a new --output-root")
        if plan["resume"] and not (Path(plan["resume"]) / "scaling_state.json").is_file():
            raise ValueError("The full WSD resume checkpoint is not present on this node")
        output.mkdir(parents=True, exist_ok=True)
        with (output / f".active_node{args.node_rank}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("This run already has an active launcher for this node rank") from None
            corpus = verify_data(args.data, Path(args.output_root).expanduser().resolve()
                                 / f".data_verified_node{args.node_rank}.json")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            prefix = output / f"launch_node{args.node_rank}_{stamp}"
            manifest = dict(plan, git_commit=revision, dataset=corpus, launcher_args=vars(args))
            prefix.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
            env = {k: v for k, v in os.environ.items() if not k.startswith(("FS_", "FSB_"))}
            env.update(plan["environment"])
            env.setdefault("NCCL_DEBUG", "WARN")
            env.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            with prefix.with_suffix(".log").open("x") as log, subprocess.Popen(
                    plan["command"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
    except (ValueError, OSError) as error:
        p.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
