"""Exercise the actual trainer on a tiny CPU model, including 2-to-8-rank resume.

This checks schedule, optimizer restoration and fixed-batch cursors. It does not
replace the full-size GPU acceptance gate before a hardware migration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def worker():
    from experiments.scaling import train_dist
    train_dist.MODEL_CONFIGS["resume_tiny"] = dict(d_model=32, n_layers=2, n_heads=4,
                                                 d_ff=64, max_seq_len=16, vocab_size=101)
    train_dist.main()


def run(output):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from experiments.rebuttal.matched_data import write_plan
    output.mkdir(parents=True, exist_ok=False)
    corpus = output / "tokens.bin"
    np.random.default_rng(123).integers(0, 101, 2000 * 17, dtype=np.uint16).tofile(corpus)
    sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    write_plan(output / "plan", corpus_bytes=corpus.stat().st_size, corpus_sha256=sha,
               seq_len=16, virtual_world=8, reference_batch=2, steps=5)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ACCELERATE_USE_CPU="true",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", WANDB_MODE="disabled",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    commands = []
    for name, world, stop, resume in [("continuous", 2, None, None),
            ("first", 2, 3, None), ("same", 2, None, "first/first_step3"),
            ("expanded", 8, None, "first/first_step3")]:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", str(world), __file__, "--worker", "--config", "resume_tiny",
            "--attn_type", "standard", "--no_rope", "--max_steps", "5", "--warmup_steps", "2",
            "--per_gpu_batch_size", "2", "--grad_accum_steps", str(64 // world),
            "--lr", ".0003", "--min_lr", ".00003", "--num_workers", "0",
            "--pretokenized", str(corpus), "--data_sha256", sha,
            "--rebuttal_data_plan", str(output / "plan/manifest.json"),
            "--eval_every", "100", "--eval_final", "--save_every", "100",
            "--checkpoint_steps", "5", "--log_every", "1", "--fail_on_nonfinite",
            "--output_dir", str(output / name), "--run_name", name]
        if stop is not None:
            command += ["--stop_after_step", str(stop)]
        if resume:
            command += ["--resume", str(output / resume)]
        with (output / (name + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=900)
        commands.append(dict(name=name, world=world, command=command, returncode=result.returncode))
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result.check_returncode()
    reference = load_file(str(output / "continuous/continuous_step5/model.safetensors"))
    comparisons = {}
    for name in ("same", "expanded"):
        weights = load_file(str(output / name / (name + "_step5/model.safetensors")))
        diffs = [(weights[key].float() - val.float()).abs() for key, val in reference.items()]
        maximum = max(float(x.max()) for x in diffs)
        # FP32 accumulation order differs across ranks; BF16 model arithmetic is unchanged.
        assert maximum < (1e-7 if name == "same" else 5e-5), (name, maximum)
        state = torch.load(output / name / (name + "_step5/optimizer.bin"),
                           map_location="cpu", weights_only=True)
        assert all(float(v["step"]) == 5 for v in state["state"].values())
        done = json.loads((output / name / "completion.json").read_text())
        assert done["executed_updates"] == 2 and done["schedule_steps"] == 5
        assert done["total_tokens"] == 5 * 128 * 16
        comparisons[name] = dict(maximum_weight_difference=maximum, completion=done)
    (output / "accepted.json").write_text(json.dumps(dict(status="pass", comparisons=comparisons,
        scope="tiny CPU trainer replay; full-size GPU acceptance remains required"), indent=2) + "\n")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        worker()
    else:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--output", type=Path, required=True)
        run(p.parse_args().output.resolve())
