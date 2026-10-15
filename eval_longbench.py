import argparse, ast, hashlib, importlib.metadata, json, os, re, string, sys, time
os.environ.pop("FS_FUSED_ATTN", None)
import torch
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "experiments", "scaling"))
sys.path.insert(0, _HERE)
import longctx_infer as _lci
from kv_cache import cached_generate
from decode_graph import graphed_generate

TOK_NAME = "fla-hub/transformer-1.3B-100B"
ROOT = Path(_HERE)
_ENV_PREFIXES = ("FS_", "FSB_", "FSL_", "KV_", "PYTORCH_CUDA", "CUDA_VISIBLE_DEVICES")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += sorted((ROOT / "experiments/scaling").glob("*.py"))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def official_metrics(path):
    # exec only the four required scorer definitions of the pinned metrics.py, skipping the
    # unused Chinese/ROUGE imports; scoring itself is not rewritten
    from fuzzywuzzy import fuzz
    tree = ast.parse(Path(path).read_text())
    names = {"normalize_answer", "f1_score", "qa_f1_score", "code_sim_score"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = dict(re=re, string=string, Counter=Counter, fuzz=fuzz)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_fs(args):
    _lci.MODEL_CONFIGS[args.config]["vocab_size"] = 32000
    prune = int(args.architecture == "FS-L")
    os.environ["FS_PRUNE_KERNEL"] = str(prune)
    os.environ["FS_PRUNE_POOL"] = str(args.pool if prune else 0)
    os.environ["FS_PRUNE_WINDOW"] = "8"
    os.environ["FS_PRUNE_BLKCHUNK"] = "16"
    os.environ["FS_STRICT_CAUSAL"] = "1"
    os.environ["FS_FUSED_ATTN"] = "0"
    os.environ["FS_FUSED_SELECT"] = "0"
    os.environ["FSL_LINEAR_CACHE"] = str(prune)
    kw = dict(window_size=512, long_range_variant="cell_topk", cell_stride=32, cell_topk=4,
              cell_select="score", sel_strict_causal=1, use_rope=True, fs_uniform_sr=True,
              rope_long_range=False, cell_prune_kernel=prune, cell_prune_pool=args.pool if prune else 0,
              cell_prune_window=8, cell_prune_blkchunk=16, fs_fused_attn=0)
    is_fsb = args.architecture in ("FS-B", "FS-B-single-forest")
    if is_fsb:
        kw.update(fsb_enabled=True, fsb_policy="fixed", fsb_width=4, fsb_delta=0,
                  fsb_tree="forest", fsb_node_score="bound", fsb_bound_radius_coef=8.0,
                  fsb_leaf_radius=0, fsb_prefill_blkchunk=32)
    if args.architecture == "FS-B":
        kw.update(fsb_width=16, fsb_read_top=4, fsb_rerank_score="lse_pos",
                  fsb_rerank_query="raw", fsb_union_width=16, fsb_union_tau_quantile=0.5)
    model = _lci.load_model(None, config=args.config, attn_type="fsattn", device="cuda",
                            max_seq_len=4096, **kw).eval()
    state = _lci._clean_keys(_lci._unwrap_state_dict(_lci._load_raw_state_dict(args.checkpoint, "cpu")))
    if "lm_head.weight" not in state and model.lm_head.weight is model.tok_emb.weight:
        state["lm_head.weight"] = state["tok_emb.weight"]
    added = []
    if is_fsb:
        # the only new persistent state of the zero-converted FS checkpoint is a zero
        # admission threshold; no learned router
        for name, value in model.state_dict().items():
            if name.endswith(".fsb_tau") and name not in state:
                state[name] = torch.zeros_like(value, device="cpu")
                added.append(name)
    model.load_state_dict(state, strict=True)
    del state
    if is_fsb:
        for module in model.modules():
            if getattr(module, "fsb_enabled", False):
                module.fsb_tau.zero_()
    if args.architecture == "FS-B":
        from experiments.scaling.fsb_release import load_union_calibration
        checkpoint = Path(args.checkpoint)
        if checkpoint.is_dir():
            checkpoint /= "model.safetensors"
        load_union_calibration(model, args.fsb_union_tau_file,
                               checkpoint_sha256=digest(checkpoint), quantile=0.5)
    if args.weight_dtype != "float32":
        for parameter in model.parameters():
            parameter.data = parameter.data.to(getattr(torch, args.weight_dtype))
    return model, {"attention": kw, "added_zero_tau_buffers": added,
                   "parameters": sum(p.numel() for p in model.parameters())}


def fs_decode_mode(args, prompt_length):
    if args.architecture == "FS-L":
        return "bounded_graph" if args.graph and prompt_length >= 512 else "bounded_eager"
    return "cached"


def generate_fs(model, ids, max_new, args, eos):
    if fs_decode_mode(args, ids.numel()) == "bounded_graph":
        return graphed_generate(model, ids, max_new, dtype="bfloat16",
                                eos_token_id=eos, linear=True)
    return cached_generate(model, ids, max_new, dtype="bfloat16", eos_token_id=eos,
                           linear=args.architecture == "FS-L",
                           lean_prefill=args.architecture != "FS-L" and ids.numel() >= 16384,
                           prefill_chunk_size=2048)


def longbench_records(args, tok):
    directory = Path(args.data)
    metadata = json.loads((directory / "manifest.json").read_text())
    data_path = directory / (args.task + "_e.jsonl")
    assert digest(data_path) == metadata["files"][data_path.name]
    for name, expected in metadata["files"].items():
        if name.startswith("protocol/"):
            assert digest(directory / name) == expected
    records = [json.loads(line) for line in data_path.read_text().splitlines()]
    templates = json.loads((directory / "protocol/dataset2prompt.json").read_text())
    caps = json.loads((directory / "protocol/dataset2maxlen.json").read_text())
    indices = metadata["tasks"][args.task]["pilot_indices"] if args.pilot else list(range(len(records)))
    if args.limit:
        indices = indices[:args.limit]
    scorers = official_metrics(directory / "protocol/metrics.py")
    metric = scorers["code_sim_score" if args.task == "repobench-p" else "qa_f1_score"]
    for index in indices[args.shard::args.shards]:
        record = records[index]
        prompt_record = dict(record)
        if args.without_context:
            prompt_record["context"] = ""
        prompt = templates[args.task].format(**prompt_record)
        ids = tok.encode(prompt)
        full_length = len(ids)
        truncated = full_length > args.input_limit
        if truncated:
            half = args.input_limit // 2
            prompt = (tok.decode(ids[:half], skip_special_tokens=True)
                      + tok.decode(ids[-half:], skip_special_tokens=True))
            ids = tok.encode(prompt)
        yield dict(index=index, record_id=record.get("_id"), tokens=ids, prompt=prompt,
                   raw_tokens=full_length, truncated=truncated, benchmark_length=record["length"],
                   document_sha256=hashlib.sha256(record["context"].encode()).hexdigest(),
                   answers=record["answers"], max_new=caps[args.task]), metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", choices=["FS", "FS-L", "FS-B", "FS-B-single-forest"], required=True)
    p.add_argument("--fsb-union-tau-file", help="portable union calibration JSON, required for two-forest FS-B")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="800m")
    p.add_argument("--pool", type=int, default=64)
    p.add_argument("--weight-dtype", default="float32")
    p.add_argument("--graph", action="store_true", help="bounded FS-L decode graph (FS-L only)")
    p.add_argument("--data", required=True, help="directory prepared by experiments/scaling/prepare_longbench.py")
    p.add_argument("--task", required=True)
    p.add_argument("--pilot", action="store_true", help="the manifest's length-stratified 50-example pilot subset")
    p.add_argument("--without-context", action="store_true", help="question-only qa or current-file-only code control")
    p.add_argument("--scorer-wheel", help="pinned, isolated fuzzywuzzy wheel; does not modify the shared environment")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--input-limit", type=int, default=32768)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.architecture == "FS-B" and not args.fsb_union_tau_file:
        p.error("FS-B requires --fsb-union-tau-file; the historical ablation is FS-B-single-forest")
    assert 0 <= args.shard < args.shards
    if args.graph and args.architecture != "FS-L":
        p.error("--graph here denotes the bounded FS-L graph, not the FS-B internal graph")
    if args.scorer_wheel:
        sys.path.insert(0, args.scorer_wheel)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(TOK_NAME, trust_remote_code=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        def emit(record):
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        hashes = source_hashes()
        emit(dict(kind="manifest", args=vars(args), source_hashes=hashes,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  hardware=torch.cuda.get_device_name(), eos=tok.eos_token_id,
                  tokenizer=TOK_NAME,
                  scorer_wheel_sha256=digest(args.scorer_wheel) if args.scorer_wheel else None,
                  source_revision=os.environ.get("FS_SOURCE_GIT"),
                  data_sha256=digest(Path(args.data) / "manifest.json")))
        model, load_info = load_fs(args)
        emit(dict(kind="load", **load_info, env={k: v for k, v in os.environ.items()
                  if k.startswith(_ENV_PREFIXES)},
                  fuzzywuzzy=importlib.metadata.version("fuzzywuzzy")))
        scores, prefix_scores = [], {}
        for item, metric in longbench_records(args, tok):
            ids = torch.tensor(item.pop("tokens"), dtype=torch.long)
            prompt = item.pop("prompt")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            generated = generate_fs(model, ids, item["max_new"], args, tok.eos_token_id)
            generated_ids = generated[0, ids.numel():].tolist()
            del generated
            torch.cuda.synchronize()
            prediction = tok.decode(generated_ids, skip_special_tokens=True)
            score = max((metric(prediction, answer) for answer in item["answers"]), default=0.0)
            scores.append(score)
            prefix_scores.setdefault(item["index"], []).append(score)
            emit(dict(kind="example", **item, prompt_tokens=ids.numel(), generated_ids=generated_ids,
                      decode_mode=fs_decode_mode(args, ids.numel()),
                      prediction=prediction, score=score, seconds=time.perf_counter()-t0,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      copies_other_value=any(v.lower() in prediction.lower()
                                             for v in item.get("other_values", [])),
                      prompt_text_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                      prompt_tokens_sha256=hashlib.sha256(
                          json.dumps(ids.tolist(), separators=(",", ":")).encode()).hexdigest()))
            print("PROGRESS", args.architecture, args.task, len(scores), score, flush=True)
            del ids
            torch.cuda.empty_cache()
        assert hashes == source_hashes(), "Source changed during evaluation"
        if not scores:
            raise RuntimeError("No evaluation examples")
        emit(dict(kind="summary", status="ok", n=len(scores), score=100*sum(scores)/len(scores),
                  n_prefixes=len(prefix_scores),
                  prefix_mean_score=100 * sum(sum(v) / len(v)
                                              for v in prefix_scores.values()) / len(prefix_scores)))
        print("COMPLETE", args.architecture, args.task, len(scores), flush=True)


if __name__ == "__main__":
    main()
