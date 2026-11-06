#!/usr/bin/env python3
"""Launch a matched rebuttal arm and retain its exact command and inputs."""

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
from experiments.rebuttal.verify_training_data import CORPORA
from experiments.scaling.provenance import source_revision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["swan", "nope", "fs", "no_erf", "dense_lr"])
    p.add_argument("--data", required=True)
    p.add_argument("--data-receipt", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--scale", choices=["341m", "743m"], default="341m")
    p.add_argument("--pilot-packet", action="store_true",
                   help="Discarded implementation pilot on the verified GB200 packet, at most 50 steps")
    p.add_argument("--data-plan", help="Verified global-batch plan for changed GPU topology")
    p.add_argument("--batch-packets", help="Verified ordered packet directory for progressive corpus transfer")
    p.add_argument("--stable-collectives", action="store_true",
                   help="Use Ring/no-NVLS collectives and non-expandable CUDA allocations")
    p.add_argument("--disable-cudnn-sdpa", action="store_true",
                   help="Use the other PyTorch SDPA backends for the GB200 large-batch runtime")
    p.add_argument("--steps", type=int)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--port", type=int, default=29561)
    p.add_argument("--per-gpu", type=int, default=2)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--save-every", type=int)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    large = a.scale == "743m"
    a.steps = a.steps if a.steps is not None else (61035 if large else 28000)
    a.eval_every = a.eval_every if a.eval_every is not None else (2000 if large else 1000)
    a.save_every = a.save_every if a.save_every is not None else (5000 if large else 2000)
    if a.pilot_packet and (a.steps > 50 or a.resume):
        p.error("Packet pilots must start fresh and stop within 50 updates")
    world = a.nodes * a.gpus_per_node
    if min(world, a.per_gpu, a.steps) < 1 or 128 % (world * a.per_gpu):
        p.error("Topology and microbatch must divide the fixed 128-sequence global batch")
    if not 0 <= a.rank < a.nodes or (a.nodes > 1 and a.master_addr == "127.0.0.1"):
        p.error("Multi-node training requires a reachable master and valid node rank")
    data = Path(a.data).resolve()
    expected_bytes, expected_sha256 = CORPORA["gb200-pilot" if a.pilot_packet else a.scale]
    receipt = json.loads(Path(a.data_receipt).read_text())
    packet_manifest = None
    if a.batch_packets:
        if a.pilot_packet or not a.data_plan:
            p.error("Ordered full-corpus packets require a batch plan and cannot use the pilot packet")
        packet_manifest = json.loads((Path(a.batch_packets) / "manifest.json").read_text())
        if packet_manifest["source"] != receipt or receipt["path"] != str(data):
            p.error("Packets are not bound to the complete source-corpus checksum receipt")
    else:
        st = data.stat()
        identity = dict(path=str(data), bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)
        if any(receipt.get(k) != v for k, v in identity.items()):
            p.error("Data file identity changed after its complete hash pass")
    if receipt.get("sha256") != expected_sha256 or receipt.get("bytes") != expected_bytes:
        p.error("Data identity changed or the full checksum receipt is missing/incorrect")
    plan_manifest = None
    if a.data_plan:
        from experiments.rebuttal.matched_data import MatchedBatchDataset
        if a.batch_packets:
            from experiments.rebuttal.packet_data import PacketBatchDataset
            ds = PacketBatchDataset(a.batch_packets, a.data_plan, 4096, 0, world,
                                     corpus_sha256=expected_sha256, wait_seconds=0)
            ds[0]
            validation = PacketBatchDataset(a.batch_packets, a.data_plan, 4096, 0, world,
                                             validation=True, corpus_sha256=expected_sha256, wait_seconds=0)
            validation[0]
        else:
            ds = MatchedBatchDataset(data, a.data_plan, 4096, 0, world,
                                     corpus_sha256=expected_sha256)
        plan_manifest = json.loads(Path(a.data_plan).read_text())
        if (len(ds.plan) < a.steps or ds.per_rank % a.per_gpu
                or plan_manifest["reference_world"] != (16 if large else 8)
                or plan_manifest["reference_batch"] != (1 if large else 2)):
            p.error("Data plan does not match the historical recipe or launch topology")
    elif not a.pilot_packet and (large or world != 8 or a.per_gpu != 2):
        p.error("This full run requires an explicit matched global-batch data plan")
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
        sys.executable, "-m", "accelerate.commands.launch", *(["--multi_gpu"] if world > 1 else []),
        "--num_machines", str(a.nodes), "--num_processes", str(world),
        "--machine_rank", str(a.rank), "--main_process_ip", a.master_addr,
        "--main_process_port", str(a.port), "--same_network", "--mixed_precision", "bf16",
        "--dynamo_backend", "no", "--num_cpu_threads_per_process", "2",
        str(ROOT / "experiments/scaling/train_dist.py"), "--config", "800m" if large else "340m", *arch,
        "--window_size", "512", "--max_seq_len", "4096", "--vocab", "32000",
        "--per_gpu_batch_size", str(a.per_gpu), "--grad_accum_steps", str(accum),
        "--max_steps", str(a.steps), "--lr", "4e-4" if large else "3e-4",
        "--min_lr", "4e-5" if large else "3e-5",
        "--warmup_steps", "2000" if large else "1000", "--weight_decay", "0.1", "--grad_clip", "1",
        "--seed", "42", "--num_workers", "2", "--save_every", str(a.save_every),
        "--checkpoint_steps", str(a.steps),
        "--eval_every", str(a.eval_every), "--log_every", str(a.log_every),
        "--pretokenized", str(data), "--run_name", a.name, "--output_dir", str(run),
        "--data_sha256", expected_sha256,
        "--fail_on_nonfinite",
        *(["--disable_cudnn_sdpa"] if a.disable_cudnn_sdpa else []),
        *(["--rebuttal_data_plan", str(Path(a.data_plan).resolve())] if a.data_plan else []),
        *(["--rebuttal_batch_packets", str(Path(a.batch_packets).resolve())] if a.batch_packets else []),
        *(["--resume", a.resume] if a.resume else []),
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FS_", "FSB_", "FSL_"))}
    fixed_env = dict(FS_FUSED_ATTN="1", FS_STRICT_CAUSAL="1", FS_PRUNE_KERNEL="0",
                     FSL_LINEAR_CACHE="0", WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false",
                     OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1",
                     PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                     NCCL_DEBUG="WARN", TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="3600")
    if a.stable_collectives:
        fixed_env.update(NCCL_ALGO="Ring", NCCL_NVLS_ENABLE="0",
                         PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False")
    env.update(fixed_env)
    import triton
    manifest = dict(schema="freesolo_rebuttal_launch/1", arm=a.arm, name=a.name,
                    scale=a.scale, pilot_packet=a.pilot_packet, data_plan=plan_manifest,
                    ordered_packets=packet_manifest,
                    source_revision=source_revision(), root=str(ROOT), command=command,
                    env=fixed_env, data=receipt, world_size=world, per_gpu=a.per_gpu,
                    accumulation=accum, tokens_per_update=524288, steps=a.steps,
                    training_tokens=a.steps * 524288, host=os.uname().nodename,
                    python=sys.version, packages={name: importlib.metadata.version(name)
                    for name in ("torch", "numpy", "accelerate", "transformers")},
                    created_at=datetime.now(timezone.utc).isoformat())
    manifest["packages"]["triton"] = triton.__version__
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
