#!/usr/bin/env python3
"""Launch a matched 341M rebuttal arm and retain its exact command/inputs."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.rebuttal.verify_training_data import EXPECTED_BYTES, EXPECTED_SHA256
from experiments.scaling.provenance import source_revision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["swan", "nope", "fs", "no_erf", "dense_lr"])
    p.add_argument("--data", required=True)
    p.add_argument("--data-receipt", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=28000)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--port", type=int, default=29561)
    p.add_argument("--per-gpu", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    world = a.nodes * a.gpus_per_node
    if min(world, a.per_gpu, a.steps) < 1 or 128 % (world * a.per_gpu):
        p.error("Topology and microbatch must divide the fixed 128-sequence global batch")
    if not 0 <= a.rank < a.nodes or (a.nodes > 1 and a.master_addr == "127.0.0.1"):
        p.error("Multi-node training requires a reachable master and valid node rank")
    data = Path(a.data).resolve()
    receipt = json.loads(Path(a.data_receipt).read_text())
    st = data.stat()
    identity = dict(path=str(data), bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)
    if any(receipt.get(k) != v for k, v in identity.items()) or (
            receipt.get("sha256") != EXPECTED_SHA256 or st.st_size != EXPECTED_BYTES):
        p.error("Data identity changed or the full checksum receipt is missing/incorrect")
    accum = 128 // (world * a.per_gpu)
    run = Path(a.runs).resolve() / a.name
    arch = ["--attn_type", "swan", "--swan_scale_base", "8192",
            "--swan_rope_theta", "1000000"] if a.arm == "swan" else (
        ["--attn_type", "standard", "--no_rope"] if a.arm == "nope" else
        ["--attn_type", "fsattn", "--long_range_variant", "cell_topk",
         "--cell_select", "score", "--cell_topk", "4", "--cell_stride", "32",
         "--sel_strict_causal", "1", "--fs_nope_longrange", "--fs_uniform_sr",
         "--sigma_init", "0.01" if a.arm == "no_erf" else "4", "--require_freesolo"])
    if a.arm == "dense_lr":
        arch.append("--lr_dense_control")
    command = [
        sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu",
        "--num_machines", str(a.nodes), "--num_processes", str(world),
        "--machine_rank", str(a.rank), "--main_process_ip", a.master_addr,
        "--main_process_port", str(a.port), "--same_network", "--mixed_precision", "bf16",
        "--dynamo_backend", "no", "--num_cpu_threads_per_process", "2",
        str(ROOT / "experiments/scaling/train_dist.py"), "--config", "340m", *arch,
        "--window_size", "512", "--max_seq_len", "4096", "--vocab", "32000",
        "--per_gpu_batch_size", str(a.per_gpu), "--grad_accum_steps", str(accum),
        "--max_steps", str(a.steps), "--lr", "3e-4", "--min_lr", "3e-5",
        "--warmup_steps", "1000", "--weight_decay", "0.1", "--grad_clip", "1",
        "--seed", "42", "--num_workers", "2", "--save_every", str(a.save_every),
        "--eval_every", str(a.eval_every), "--log_every", str(a.log_every),
        "--pretokenized", str(data), "--run_name", a.name, "--output_dir", str(run),
        "--data_sha256", EXPECTED_SHA256,
        *(["--resume", a.resume] if a.resume else []),
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FS_", "FSB_", "FSL_"))}
    fixed_env = dict(FS_FUSED_ATTN="1", FS_STRICT_CAUSAL="1", FS_PRUNE_KERNEL="0",
                     FSL_LINEAR_CACHE="0", WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false",
                     OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1",
                     PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                     NCCL_DEBUG="WARN", TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="3600")
    env.update(fixed_env)
    manifest = dict(schema="freesolo_rebuttal_launch/1", arm=a.arm, name=a.name,
                    source_revision=source_revision(), root=str(ROOT), command=command,
                    env=fixed_env, data=receipt, world_size=world, per_gpu=a.per_gpu,
                    accumulation=accum, tokens_per_update=524288, steps=a.steps,
                    training_tokens=a.steps * 524288, host=os.uname().nodename,
                    python=sys.version, packages={name: importlib.metadata.version(name)
                    for name in ("torch", "numpy", "accelerate", "triton", "transformers")},
                    created_at=datetime.now(timezone.utc).isoformat())
    if a.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    manifests = Path(a.runs).resolve() / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    record = manifests / f"{a.name}.rank{a.rank}.{time.time_ns()}.json"
    record.write_text(json.dumps(manifest, indent=2) + "\n")
    print("LAUNCH_MANIFEST " + str(record), flush=True)
    start = time.time()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(datetime.now(timezone.utc).isoformat(timespec="seconds"), line.rstrip(), flush=True)
        code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    manifest.update(returncode=code, seconds=time.time() - start,
                    finished_at=datetime.now(timezone.utc).isoformat())
    record.write_text(json.dumps(manifest, indent=2) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
