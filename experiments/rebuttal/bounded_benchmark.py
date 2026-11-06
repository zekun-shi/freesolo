from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/scaling"))

import torch
from kv_cache import _trunk_cached, cached_generate, linear_prefill, make_caches
from longctx_infer import _clean_keys, _lm_head_fp32, _load_raw_state_dict, load_model

from experiments.scaling.model import MODEL_CONFIGS


def state_bytes(caches, extra=None):
    seen = set()
    total = 0

    def visit(value):
        nonlocal total
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            key = (value.device, storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, (tuple, list)):
            for v in value:
                visit(v)

    for cache in caches:
        for key, value in cache.__dict__.items():
            if key != "attn":
                visit(value)
    visit(extra)
    return total


def autocast(dtype):
    return (torch.autocast("cuda", dtype=dtype) if dtype != torch.float32
            else contextlib.nullcontext())


def manifest(args):
    sources = ["src/fsattn/attention.py", "src/fsattn/linear_cache.py", "src/fsattn/prefix_topp.py",
               "src/utils/local_attention.py", "experiments/scaling/model.py", "experiments/scaling/longctx_infer.py",
               "experiments/scaling/kv_cache.py", "experiments/scaling/decode_graph.py",
               "experiments/rebuttal/bounded_benchmark.py", "experiments/scaling/linear_decode_graph.py"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=False)
    weight = Path(args.checkpoint) / "model.safetensors"
    h = hashlib.sha256()
    with weight.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return {"args": vars(args), "torch": torch.__version__, "cuda": torch.version.cuda,
                "checkpoint_sha256": h.hexdigest(),
                "hardware": torch.cuda.get_device_name(0), "weight_dtype": args.weight_dtype,
                "rotary_dtype": "float32",
                "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                "git": revision.stdout.strip() if revision.returncode == 0 else (ROOT / "SOURCE_COMMIT").read_text().strip(),
                "env": {k: v for k, v in os.environ.items() if k.startswith(("FS_", "FSL_"))}}


def build(args):
    MODEL_CONFIGS[args.config]["vocab_size"] = 32000
    model = load_model(
        None, config=args.config, attn_type="fsattn", device="cuda", max_seq_len=4096,
        strict=True, window_size=512, long_range_variant="cell_topk", cell_stride=32, cell_topk=4,
        cell_select="score", sel_strict_causal=1, use_rope=True, fs_uniform_sr=True,
        rope_long_range=False, cell_prune_kernel=1, cell_prune_pool=args.pool,
        cell_prune_window=8, cell_prune_blkchunk=16, fs_fused_attn=args.fused,
    ).eval()
    state = _clean_keys(_load_raw_state_dict(args.checkpoint, "cpu"))
    if "lm_head.weight" not in state and model.lm_head.weight is model.tok_emb.weight:
        state["lm_head.weight"] = state["tok_emb.weight"]
    model.load_state_dict(state, strict=True)
    if args.weight_dtype != "float32":
        for parameter in model.parameters():
            parameter.data = parameter.data.to(getattr(torch, args.weight_dtype))
    return model


@torch.no_grad()
def prefill(model, ids, linear, chunk_size, reserve=0):
    cache = make_caches(model, linear=linear, reserve=reserve)
    if linear:
        hidden = linear_prefill(model, ids, cache, chunk_size=chunk_size)
    else:
        hidden = _trunk_cached(model, ids.to("cuda"), cache)[:, -1:].clone()
    return hidden, cache


def prompts(args, length):
    if args.data_root:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B", local_files_only=True)
        for task in args.tasks.split(","):
            file = Path(args.data_root) / f"L{length}" / task / "validation.jsonl"
            records = [json.loads(line) for line in file.open()][:args.examples]
            for i, record in enumerate(records):
                text = record["input"] + record.get("answer_prefix", "")
                token_ids = tok.encode(text)
                prompt_hash = hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()
                yield torch.tensor(token_ids)[None], {
                    "task": task, "example": i, "expected": record["outputs"], "prompt_sha256": prompt_hash}, tok
    else:
        for i in range(args.examples):
            generator = torch.Generator().manual_seed(971 + length + i)
            yield torch.randint(0, 32000, (1, length), generator=generator), {"example": i}, None


@torch.no_grad()
def parity(args, model, emit):
    for length in args.lengths:
        for ids, record, tok in prompts(args, length):
            outputs = {}
            for linear in (False, True):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with autocast(args.dtype):
                    hidden, cache = prefill(model, ids, linear, args.chunk_size, args.steps + 1)
                    logits = _lm_head_fp32(model, hidden, 4096)[:, -1]
                    outputs[linear] = {"logits": logits.cpu(), "bytes": state_bytes(cache)}
                    teacher_logits = []
                    teacher = torch.arange(args.steps, device="cuda")[None] + 123
                    for step in range(args.steps):
                        hidden = _trunk_cached(model, teacher[:, step:step + 1], cache)
                        teacher_logits.append(_lm_head_fp32(model, hidden, 4096)[:, -1].cpu())
                    outputs[linear]["teacher"] = torch.stack(teacher_logits)
                torch.cuda.synchronize()
                outputs[linear]["seconds"] = time.perf_counter() - start
                outputs[linear]["peak_bytes"] = torch.cuda.max_memory_allocated()
                outputs[linear]["state_after_decode"] = state_bytes(cache)
                del hidden, logits, cache
                generated = cached_generate(model, ids, args.new_tokens, dtype=args.dtype, linear=linear,
                                            prefill_chunk_size=args.chunk_size)
                outputs[linear]["tokens"] = generated[:, ids.shape[1]:].cpu()
                del generated
            ref, got = outputs[False], outputs[True]
            delta = got["teacher"].float() - ref["teacher"].float()
            item = dict(kind="parity", nominal_length=length, tokens=ids.shape[1], **record,
                        prefill_max_abs=(got["logits"] - ref["logits"]).abs().max().item(),
                        decode_max_abs=delta.abs().max().item(), decode_rms=delta.square().mean().sqrt().item(),
                        teacher_argmax_agreement=(got["teacher"].argmax(-1) == ref["teacher"].argmax(-1)).float().mean().item(),
                        greedy_equal=torch.equal(got["tokens"], ref["tokens"]),
                        legacy_state_bytes=ref["bytes"], linear_state_bytes=got["bytes"],
                        legacy_peak_bytes=ref["peak_bytes"], linear_peak_bytes=got["peak_bytes"],
                        linear_state_after_decode=got["state_after_decode"],
                        legacy_seconds=ref["seconds"], linear_seconds=got["seconds"])
            if tok:
                a, b = tok.decode(ref["tokens"][0]), tok.decode(got["tokens"][0])
                item.update(legacy_answer=a, linear_answer=b,
                            legacy_correct=all(x.lower() in a.lower() for x in record["expected"]),
                            linear_correct=all(x.lower() in b.lower() for x in record["expected"]))
            emit(item)


@torch.no_grad()
def accuracy(args, model, emit):
    if not args.data_root:
        raise ValueError("Accuracy mode requires RULER prompts via --data-root")
    for length in args.lengths:
        for ids, record, tok in prompts(args, length):
            for linear in (False, True) if args.legacy else (True,):
                start = time.perf_counter()
                if linear and args.graph:
                    from decode_graph import graphed_generate
                    generated = graphed_generate(model, ids, args.new_tokens, dtype=args.dtype,
                                                 linear=True, eos_token_id=tok.eos_token_id)
                else:
                    generated = cached_generate(
                        model, ids, args.new_tokens, dtype=args.dtype, linear=linear,
                        eos_token_id=tok.eos_token_id, lean_prefill=(not linear and length >= 16384),
                        prefill_chunk_size=args.chunk_size)
                answer = tok.decode(generated[0, ids.shape[1]:])
                refs = record["expected"]
                score = sum(str(ref).lower() in answer.lower() for ref in refs) / max(1, len(refs))
                emit(dict(kind="accuracy", mode="linear" if linear else "legacy",
                          graph=bool(linear and args.graph),
                          nominal_length=length, tokens=ids.shape[1], **record,
                          answer=answer, score=score, seconds=time.perf_counter() - start))
                del generated


@torch.no_grad()
def graph_parity(args, model, emit):
    from linear_decode_graph import LinearGraphedDecoder
    for length in args.lengths:
        for ids, record, tok in prompts(args, length):
            graph = LinearGraphedDecoder(model, ids, Lmax=length + args.steps + 1, dtype=args.dtype)
            graph.prefill()
            with autocast(args.dtype):
                hidden, cache = prefill(model, ids, True, args.chunk_size)
                first = _lm_head_fp32(model, hidden, 4096)[:, -1].argmax(-1)
            first_equal = torch.equal(first, graph.next_tok)
            max_abs, squared, count, agreements = 0., 0., 0, 0
            for step in range(args.steps):
                token = torch.full((1, 1), 123 + step, device="cuda", dtype=torch.long)
                with autocast(args.dtype):
                    hidden = _trunk_cached(model, token, cache)
                    ref = _lm_head_fp32(model, hidden, 4096)
                graph.next_tok = token[:, 0]
                graph.step()
                delta = graph.static_logits.float() - ref.float()
                max_abs = max(max_abs, delta.abs().max().item())
                squared += delta.square().sum().item()
                count += delta.numel()
                agreements += int(torch.equal(graph.static_logits.argmax(-1), ref.argmax(-1)))
            emit(dict(kind="graph_parity", nominal_length=length, tokens=ids.shape[1], **record,
                      first_argmax_equal=first_equal, decode_max_abs=max_abs,
                      decode_rms=(squared / count) ** .5,
                      teacher_argmax_agreement=agreements / args.steps,
                      linear_state_bytes=state_bytes(cache),
                      graph_state_bytes=state_bytes(graph.cache, graph.layer_bufs)))
            del graph, cache, hidden, ref, delta
            gc.collect()
            torch.cuda.empty_cache()


@torch.no_grad()
def benchmark(args, model, emit):
    for length in args.lengths:
        ids = torch.randint(0, 32000, (1, length), generator=torch.Generator().manual_seed(length))
        for linear in (True, False) if args.legacy else (True,):
            mode = "linear" if linear else "legacy"
            samples = []
            for trial in range(args.warmup + args.repeats):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with autocast(args.dtype):
                    hidden, cache = prefill(model, ids, linear, args.chunk_size, args.steps + 16)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated()
                allocated = state_bytes(cache)
                next_id = torch.full((1, 1), 123, device="cuda", dtype=torch.long)
                with autocast(args.dtype):
                    for _ in range(8):
                        hidden = _trunk_cached(model, next_id, cache)
                    torch.cuda.synchronize()
                    begin_decode = time.perf_counter()
                    for _ in range(args.steps):
                        hidden = _trunk_cached(model, next_id, cache)
                        logits = _lm_head_fp32(model, hidden, 4096)[:, -1]
                        next_id = logits.argmax(-1, keepdim=True)
                    torch.cuda.synchronize()
                decode_seconds = time.perf_counter() - begin_decode
                if trial >= args.warmup:
                    samples.append({"prefill_seconds": elapsed, "decode_seconds": decode_seconds,
                                        "peak_bytes": peak, "state_bytes": allocated,
                                        "decode_peak_bytes": torch.cuda.max_memory_allocated(),
                                        "state_after_decode": state_bytes(cache)})
                del cache, hidden
            emit({"kind": "benchmark", "mode": mode, "length": length, "samples": samples,
                      "prefill_tok_s": length / statistics.median(x["prefill_seconds"] for x in samples),
                      "decode_tok_s": args.steps / statistics.median(x["decode_seconds"] for x in samples)})


@torch.no_grad()
def graph_benchmark(args, model, emit):
    from linear_decode_graph import LinearGraphedDecoder
    for length in args.lengths:
        ids = torch.randint(0, 32000, (1, length), generator=torch.Generator().manual_seed(length))
        samples = []
        for trial in range(args.warmup + args.repeats):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            graph = LinearGraphedDecoder(model, ids, Lmax=length + args.steps + 16, dtype=args.dtype)
            graph.prefill()
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - begin
            prefill_peak = torch.cuda.max_memory_allocated()
            for _ in range(8):
                graph.step()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for _ in range(args.steps):
                graph.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            if trial >= args.warmup:
                samples.append({"prefill_seconds": prefill_seconds, "decode_seconds": elapsed,
                                    "peak_bytes": prefill_peak,
                                    "decode_peak_bytes": torch.cuda.max_memory_allocated(),
                                    "state_bytes": state_bytes(graph.cache, graph.layer_bufs),
                                    "allocated_after_decode": torch.cuda.memory_allocated()})
            del graph
        emit({"kind": "benchmark", "mode": "linear_graph", "length": length, "samples": samples,
                  "prefill_tok_s": length / statistics.median(x["prefill_seconds"] for x in samples),
                  "decode_tok_s": args.steps / statistics.median(x["decode_seconds"] for x in samples)})


@torch.no_grad()
def profile_prefill(args, model, emit):
    """Trace the existing path separately from uninstrumented wall timings.

    The automatic profiler's FLOPs omit custom kernels and are explicitly a
    partial counter. Raw counts below expose work inside the bounded cache;
    complete model accounting is a separate experiment.
    """
    from collections import Counter
    from torch.profiler import profile, ProfilerActivity, record_function
    from src.fsattn.linear_cache import LinearFSCache

    counts = Counter()
    originals = {}

    def wrap(name, original):
        def call(cache, *a, **kw):
            counts[name + "_calls"] += 1
            if name == "select" and cache.pool is not None:
                q = a[0]
                b, h, n, d = q.shape
                c = cache.pool["ids"].shape[-1]
                counts["selector_cell_pairs"] += b * h * n * c
                counts["selector_dot_flops_fma2"] += 2 * b * h * n * c * d
                counts["selector_sort_items_two_sorts"] += 2 * b * h * n * c
                counts["selector_erf_evaluations"] += b * h * n * c
            elif name == "_cells":
                b, h, n, d = a[0].shape
                counts["summary_key_elements"] += b * h * n * d
                counts["summary_norm_sqrt_evaluations"] += b * h * n
            elif name == "_short_range":
                b, h, n, d = a[0].shape
                old = 0 if cache._sr_k is None else cache._sr_k.shape[2]
                counts["sr_rectangular_pairs"] += b * h * n * (old + n)
                counts["sr_causal_window_pairs"] += b * h * sum(min(cache.W, cache.pos + i) + 1 for i in range(n))
            elif name == "_long_range" and cache.pool is not None:
                b, h, n, d = a[0].shape
                k = min(cache.attn.cell_topk, cache.eligible)
                counts["lr_selected_token_slots"] += b * h * n * k * cache.s
                counts["lr_qk_av_flops_fma2"] += 4 * b * h * n * k * cache.s * d
            elif name == "prepare":
                target = cache.pos // cache.Bq * cache.Bq // cache.s
                if target > cache.eligible:
                    old = 0 if cache._top is None else cache._top["ids"].shape[-1]
                    b, h = cache._pending_k.shape[:2]
                    counts["prune_sort_items_two_sorts"] += 2 * b * h * (old + target - cache.eligible)
            with record_function("bounded/" + name):
                return original(cache, *a, **kw)
        return call

    for length in args.lengths:
        ids = torch.randint(0, 32000, (1, length), generator=torch.Generator().manual_seed(length))
        with autocast(args.dtype):
            hidden, cache = prefill(model, ids, True, args.chunk_size)
        del hidden, cache
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        counts.clear()
        for name in ("prepare", "_cells", "select", "_short_range", "_long_range", "ingest"):
            originals[name] = getattr(LinearFSCache, name)
            setattr(LinearFSCache, name, wrap(name, originals[name]))
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, with_flops=True, profile_memory=True) as prof:
                with record_function("bounded/prefill"), autocast(args.dtype):
                    hidden, cache = prefill(model, ids, True, args.chunk_size)
                    logits = _lm_head_fp32(model, hidden, 4096)
                torch.cuda.synchronize()
        finally:
            for name, original in originals.items():
                setattr(LinearFSCache, name, original)
        trace = str(Path(args.output).with_suffix("")) + f".L{length}.trace.json"
        prof.export_chrome_trace(trace)
        events = []
        for ev in prof.key_averages():
            events.append(dict(name=ev.key, count=ev.count, cpu_us=ev.cpu_time_total,
                               self_cpu_us=ev.self_cpu_time_total,
                               device_us=ev.device_time_total, self_device_us=ev.self_device_time_total,
                               profiler_flops=ev.flops))
        events.sort(key=lambda x: x["self_device_us"], reverse=True)
        emit(dict(kind="prefill_profile", length=length, trace=trace, events=events,
                  counts=dict(counts), linear_state_bytes=state_bytes(cache),
                  automatic_flops_scope="partial; excludes custom Triton/CUDA work, not a total-model FLOP result",
                  integer_scope="sort element visits, not the sorting implementation's comparison count"))
        del prof, hidden, cache, logits
        gc.collect()
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="340m")
    p.add_argument("--mode", choices=["parity", "accuracy", "benchmark", "graph-parity", "graph-benchmark", "profile"], default="parity")
    p.add_argument("--output", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--weight-dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--pool", type=int, default=64)
    p.add_argument("--fused", type=int, default=1)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--new-tokens", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--examples", type=int, default=1)
    p.add_argument("--data-root")
    p.add_argument("--tasks", default="niah_single_1,niah_single_2,niah_single_3")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--graph", action="store_true", help="Use bounded graph decode in accuracy mode")
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        def emit(item):
            line = json.dumps(item, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)
        emit(dict(kind="manifest", **manifest(args)))
        args.dtype = getattr(torch, args.dtype)
        model = build(args)
        {"parity": parity, "accuracy": accuracy, "benchmark": benchmark,
         "graph-parity": graph_parity, "graph-benchmark": graph_benchmark,
         "profile": profile_prefill}[args.mode](args, model, emit)


if __name__ == "__main__":
    main()
