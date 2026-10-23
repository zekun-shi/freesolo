#!/usr/bin/env python3
"""Turn a training checkpoint into an evaluation checkpoint directory.

Training runs write `model.safetensors` (fp32, unprefixed keys) next to
`optimizer.bin` and the RNG state. The evaluation drivers and the Hub release
want a flat directory holding just the weights plus a `config.json` that records
what produced them. This script does that copy and writes the config, taking the
architecture and tokenizer fields from an existing evaluation checkpoint and the
shape fields from `MODEL_CONFIGS` so the two cannot drift apart.

The parameter count and the weight hash are measured from the file itself rather
than copied, so the config can be checked against the weights.
"""

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.scaling.model import MODEL_CONFIGS  # noqa: E402

WEIGHTS = "model.safetensors"
SHAPE_KEYS = ("d_model", "n_layers", "n_heads", "head_dim", "d_ff")
PREFIXES = ("_orig_mod.", "module.")


def read_header(path):
    with open(path, "rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    return header


def summarize(header):
    tensors = [name for name in header if name != "__metadata__"]
    params = sum(math.prod(header[name]["shape"]) for name in tensors)
    dtypes = sorted({header[name]["dtype"] for name in tensors})
    prefixes = sorted({p for p in PREFIXES for n in tensors if n.startswith(p)})
    return {"tensors": len(tensors), "params": params, "dtypes": dtypes, "prefixes": prefixes}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(32 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_prefixes(source, destination, discard):
    from safetensors.torch import load_file, save_file

    state = load_file(source)
    for prefix in discard:
        state = {name.removeprefix(prefix): value for name, value in state.items()}
    save_file(state, destination)


def git_revision(repo):
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit("no git checkout here; pass --code-revision")
    return result.stdout.strip()


def parse_pairs(values):
    pairs = {}
    for value in values:
        key, _, item = value.partition("=")
        pairs[key] = item
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="training run directory (or its model.safetensors) to package")
    ap.add_argument("--out", required=True, help="evaluation checkpoint directory to write")
    ap.add_argument("--template", required=True,
                    help="config.json of an existing evaluation checkpoint of the same variant")
    ap.add_argument("--variant", required=True, choices=["fs", "fs-l", "fs-b"])
    ap.add_argument("--model-config", required=True,
                    help="MODEL_CONFIGS key giving the shape fields (e.g. 1.3b_fs)")
    ap.add_argument("--eval-flags", default=None,
                    help="flags the evaluation drivers take, space separated (default: template's)")
    ap.add_argument("--env", action="append", default=None, metavar="KEY=VALUE",
                    help="full evaluation environment, repeatable (default: template's)")
    ap.add_argument("--base-model", default=None,
                    help="Hub id of the weights this run started from, or 'none'")
    ap.add_argument("--parameters", type=int, required=True,
                    help="parameter count the training run reported")
    ap.add_argument("--pretrain-steps", type=int, required=True)
    ap.add_argument("--cpt-steps", type=int, default=0)
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to record (default: step.txt in the run directory)")
    ap.add_argument("--code-revision", default=None,
                    help="freesolo commit the checkpoint was trained on (default: current HEAD)")
    ap.add_argument("--extra", action="append", default=[], metavar="FILE",
                    help="further artifact to copy alongside the weights (e.g. union_calibration.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure the checkpoint and print the config without writing")
    args = ap.parse_args()

    run = Path(args.checkpoint)
    source = run / WEIGHTS if run.is_dir() else run
    if not source.is_file():
        raise SystemExit(f"no {WEIGHTS} under {args.checkpoint}")

    header = read_header(source)
    facts = summarize(header)
    if facts["prefixes"]:
        print(f"[export] stripping prefixes: {facts['prefixes']}", file=sys.stderr)
    if facts["dtypes"] != ["F32"]:
        print(f"[export] weights are {facts['dtypes']}, not F32", file=sys.stderr)

    template = json.loads(Path(args.template).read_text())
    config = dict(template)
    shape = MODEL_CONFIGS[args.model_config]
    architecture = dict(template.get("architecture", {}))
    if shape["d_model"] % shape["n_heads"]:
        raise SystemExit(f"d_model {shape['d_model']} is not a multiple of n_heads {shape['n_heads']}")
    architecture.update({
        "n_layers": shape["n_layers"],
        "d_model": shape["d_model"],
        "n_heads": shape["n_heads"],
        "head_dim": shape["d_model"] // shape["n_heads"],
        "d_ff": shape["d_ff"],
    })

    # The file holds buffers too (log_sigma, long_range_scale), so its element count
    # runs slightly ahead of the parameter count the run reported; accept that gap only.
    excess = facts["params"] - args.parameters
    if not 0 <= excess <= args.parameters * 1e-4:
        raise SystemExit(f"weights hold {facts['params']:,} elements, the run reported "
                         f"{args.parameters:,} parameters; wrong checkpoint?")

    step = args.step
    if step is None and (run / "step.txt").is_file():
        step = int((run / "step.txt").read_text().strip())

    tokens_per_step = template["tokens_per_step"]
    steps = {"pretraining": args.pretrain_steps}
    tokens = {"pretraining": args.pretrain_steps * tokens_per_step}
    if args.cpt_steps:
        steps["continued_pretraining"] = args.cpt_steps
        tokens["continued_pretraining"] = args.cpt_steps * tokens_per_step
    tokens["total"] = sum(tokens.values())

    config.update({
        "freesolo_config": args.model_config,
        "variant": args.variant,
        "parameters": args.parameters,
        "vocab_size": shape["vocab_size"],
        "architecture": architecture,
        "eval_flags": args.eval_flags.split() if args.eval_flags else template["eval_flags"],
        "env": parse_pairs(args.env) if args.env else template["env"],
        "base_model": None if args.base_model in (None, "none") else args.base_model,
        "training_steps": steps,
        "training_tokens": tokens,
        "training_step": step,
        "code_revision": args.code_revision or git_revision(Path(__file__).resolve().parents[1]),
        "weights_dtype": "float32",
        "weights_sha256": None,
    })
    config.pop("union_calibration", None)

    print(f"[export] {facts['tensors']} tensors, {facts['params']:,} elements, "
          f"{args.parameters:,} parameters, step {step}, {tokens['total']:,} tokens")
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / WEIGHTS
    if facts["prefixes"]:
        strip_prefixes(source, target, facts["prefixes"])
    else:
        shutil.copy2(source, target)
    for artifact in args.extra:
        shutil.copy2(artifact, out / Path(artifact).name)
    config["weights_sha256"] = file_sha256(target)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"[export] wrote {target} ({target.stat().st_size:,} bytes, "
          f"sha256 {config['weights_sha256']})")


if __name__ == "__main__":
    main()
