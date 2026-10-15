"""Download a FreeSolo Hub checkpoint, generate text, or reproduce its speed."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

_SCALING = Path(__file__).resolve().parent
for _directory in (_SCALING, _SCALING.parents[1]):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(repo_id, revision="main", cache_dir=None):
    """Resolve a revision once and verify the downloaded weights against Hub metadata."""
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    weights = next(f for f in info.siblings if f.rfilename == "model.safetensors")
    expected = weights.lfs.sha256 if weights.lfs else None
    folder = Path(snapshot_download(
        repo_id, revision=info.sha, cache_dir=cache_dir,
        allow_patterns=["config.json", "model.safetensors", "union_calibration.json"],
    ))
    config = json.loads((folder / "config.json").read_text())
    if config.get("model_type") != "freesolo" or config.get("variant") not in ("fs", "fs-l", "fs-b"):
        raise ValueError("Expected a native FreeSolo config.json")
    digest = file_sha256(folder / "model.safetensors")
    if not expected or digest != expected:
        raise ValueError("Downloaded weights do not match the Hub SHA256")
    if config.get("weights_sha256", digest) != digest:
        raise ValueError("Weights do not match config.json")
    if config["variant"] == "fs-b" and not (folder / "union_calibration.json").is_file():
        raise ValueError("FS-B requires union_calibration.json alongside its weights")
    return folder, config, {"model": repo_id, "revision": info.sha, "weights_sha256": digest}


def configure(variant, bounded_cache=False, device="cuda"):
    if bounded_cache and variant != "fs-l":
        raise ValueError("The bounded cache is available only for FS-L")
    os.environ["FS_FUSED_ATTN"] = "1" if device == "cuda" else "0"
    os.environ["FS_STRICT_CAUSAL"] = "1"
    os.environ["FS_PRUNE_KERNEL"] = "1" if variant == "fs-l" else "0"
    os.environ["FSL_LINEAR_CACHE"] = "1" if bounded_cache else "0"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    if variant == "fs-l":
        os.environ.update(FS_PRUNE_POOL="64", FS_PRUNE_WINDOW="8", FS_PRUNE_BLKCHUNK="16")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_native(folder, config, *, device, max_seq_len, weights_sha256):
    import torch
    from safetensors import safe_open
    from experiments.scaling import longctx_infer
    from experiments.scaling.bench_efficiency import serve_kwargs

    variant = config["variant"]
    serve = {"fs": "dense", "fs-l": "sparse", "fs-b": "fsb"}[variant]
    size = config["freesolo_config"]
    longctx_infer.MODEL_CONFIGS[size]["vocab_size"] = config["vocab_size"]
    model = longctx_infer.load_model(
        str(folder), config=size, device=device, max_seq_len=max_seq_len,
        **serve_kwargs(serve, 0.5 if variant == "fs-b" else 0.0),
    )
    with safe_open(folder / "model.safetensors", framework="pt") as weights:
        stored = set(weights.keys())
    missing = set(model.state_dict()) - stored
    allowed = {"lm_head.weight"} | {k for k in model.state_dict() if k.endswith(".fsb_tau")}
    if missing - allowed or stored - set(model.state_dict()):
        raise ValueError(f"Checkpoint keys do not match the model: missing={sorted(missing - allowed)}")
    if model.lm_head.weight.data_ptr() != model.tok_emb.weight.data_ptr():
        raise ValueError("Expected tied input/output embeddings")
    if sum(p.numel() for p in model.parameters()) != config["parameters"]:
        raise ValueError("Parameter count does not match config.json")
    calibration = None
    if variant == "fs-b":
        from experiments.scaling.fsb_release import load_union_calibration
        for module in model.modules():
            if getattr(module, "fsb_enabled", False):
                module.fsb_tau.zero_()
        calibration = load_union_calibration(
            model, folder / "union_calibration.json", checkpoint_sha256=weights_sha256, quantile=0.5,
        )
    # Keep fp32 threshold buffers; the serving stack uses bf16 autocast for CUDA.
    return model.eval(), calibration


def retrieval_prompt():
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"
    return (
        "A special magic number is hidden within the following text. "
        "Make sure to memorize it. I will quiz you about the number afterwards.\n"
        + filler * 8 + "The special magic number for sparkling-otter is: 5437923.\n" + filler * 32
        + "What is the special magic number for sparkling-otter mentioned in the provided text?"
        + " The special magic number for sparkling-otter mentioned in the provided text is"
    )


def environment():
    import torch
    from experiments.scaling.bench_efficiency import attn_flags, fsb_flags
    from experiments.scaling.provenance import source_revision
    packages = {}
    for name in ("freesolo", "torch", "triton", "transformers", "huggingface_hub", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    defaults = {**attn_flags(), **fsb_flags(), "FSB_STAGE_PIPE": "1", "FSL_SR_DECODE_FUSED": "1",
                "FS_PREFILL_GRAPH": "1", "KV_PREALLOC": "1"}
    switches = {k: os.environ.get(k, v) for k, v in defaults.items()}
    switches.update({k: v for k, v in os.environ.items() if k.startswith(("FS_", "FSB_", "FSL_", "KV_"))})
    return {"packages": packages, "source_revision": source_revision(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "execution_env": switches, "tf32_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="zekun-shi/FreeSolo-B-341M")
    parser.add_argument("--revision", default="main", help="Hub model revision; recorded as an immutable commit")
    parser.add_argument("--cache-dir", help="use a new directory for a fresh-download validation")
    return parser


def generate_main():
    parser = common_parser("Load FreeSolo directly from Hugging Face and generate text.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prompt", help="default: a short retrieval example with expected answer 5437923")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--bounded-cache", action="store_true", help="FS-L only: use its bounded streaming cache")
    parser.add_argument("--output", default="generation.json")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    folder, config, provenance = download_model(args.model, args.revision, args.cache_dir)
    if args.bounded_cache and args.device != "cuda":
        parser.error("--bounded-cache requires CUDA")
    if config["variant"] == "fs-b" and args.device != "cuda":
        parser.error("FS-B inference requires CUDA; CPU generation is supported for FS and FS-L")
    configure(config["variant"], args.bounded_cache, args.device)
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], revision=config["tokenizer_revision"])
    prompt = args.prompt if args.prompt is not None else retrieval_prompt()
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, calibration = load_native(folder, config, device=args.device,
                                    max_seq_len=ids.numel() + args.max_new_tokens + 1,
                                    weights_sha256=provenance["weights_sha256"])
    if args.device == "cuda":
        from experiments.scaling.kv_cache import cached_generate as generate
    else:
        from experiments.scaling.longctx_infer import greedy_generate as generate
    result = generate(model, ids, args.max_new_tokens, device=args.device,
                      dtype="bfloat16" if args.device == "cuda" else "float32",
                      eos_token_id=tokenizer.eos_token_id)
    tokens = result[0, ids.numel():].tolist()
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    passed = "5437923" in answer if args.prompt is None else None
    report = dict(provenance, variant=config["variant"], device=args.device, environment=environment(),
                  tokenizer={"repo": config["tokenizer"], "revision": config["tokenizer_revision"]},
                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), prompt_tokens=ids.numel(),
                  generated_ids=tokens, generated_text=answer, retrieval_passed=passed, calibration=calibration)
    write_json(args.output, report)
    print(answer)
    print(f"Model revision: {provenance['revision']}; result: {args.output}")
    if passed is False:
        raise SystemExit("Retrieval example did not recover 5437923; see the recorded generation.")


def benchmark_main():
    parser = common_parser("Reproduce FreeSolo CUDA-graph speed from Hub weights and portable calibration.")
    parser.add_argument("--lengths", default="65536,131072")
    parser.add_argument("--output", default="hf-benchmark.json")
    args = parser.parse_args()
    lengths = [int(n) for n in args.lengths.split(",")]
    if not lengths or min(lengths) < 1:
        parser.error("--lengths must contain positive token counts")
    folder, config, provenance = download_model(args.model, args.revision, args.cache_dir)
    configure(config["variant"])
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serve = {"fs": "dense", "fs-l": "sparse", "fs-b": "fsb"}[config["variant"]]
    command = [sys.executable, "-m", "experiments.scaling.bench_efficiency", "--checkpoint", str(folder),
               "--config", config["freesolo_config"], "--serve", serve, "--lengths", args.lengths,
               "--cap40", "0", "--warm", "1", "--nmed", "3", "--steps", "256", "--out", str(output)]
    if serve == "fsb":
        command += ["--fsb_union_tau_quantile", "0.5", "--fsb_union_tau_file",
                    str(folder / "union_calibration.json")]
    write_json(output.with_suffix(".provenance.json"), dict(provenance, command=command, environment=environment()))
    subprocess.run(command, check=True)
    rows = json.loads(output.read_text())
    for row in rows:
        print(f"{row['L']:>7} tokens: prefill[graph] {row.get('prefill_graph_ktoks')} Ktok/s; "
              f"decode[graph] {row.get('decode_graph_tps')} tok/s")
    if serve == "fsb" and any(not isinstance(row.get(key), (float, int))
                              for row in rows for key in ("prefill_graph_ktoks", "decode_graph_tps")):
        raise SystemExit("A graph benchmark failed; inspect the output before comparing speeds.")


if __name__ == "__main__":
    generate_main()
