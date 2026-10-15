import argparse, hashlib, json, os, random, sys
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.scaling.model import FreeSoloLM, MODEL_CONFIGS, check_params_loaded


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="accelerate ckpt DIR (model.safetensors) or .pt file")
    p.add_argument("--config", default="340m")
    p.add_argument("--attn_type", default="fsattn", choices=["fsattn"])
    p.add_argument("--long_range_variant", default="cell_topk", choices=["cell_topk"])
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--cell_stride", type=int, default=64, help="token-cell size of the long-range branch")
    p.add_argument("--cell_topk", type=int, default=1, help="# cells kept for exact token readout")
    p.add_argument("--no_rope", action="store_true", help="disable RoPE in the short-range branch")
    p.add_argument("--fs_nope_longrange", action="store_true", help="NoPE long-range branch (rope_long_range=False)")
    p.add_argument("--fs_uniform_sr", action="store_true",
                   help="uniform-window short-range branch (no distance bias)")
    p.add_argument("--cell_select", type=str, default="score", choices=["score"])
    p.add_argument("--cell_prune_kernel", type=int, default=None)
    p.add_argument("--cell_prune_pool", type=int, default=None)
    p.add_argument("--cell_prune_window", type=int, default=None)
    p.add_argument("--cell_prune_blkchunk", type=int, default=None)
    p.add_argument("--sel_strict_causal", type=int, default=None)
    p.add_argument("--use_yarn", action="store_true", help="YaRN rope extension")
    p.add_argument("--yarn_scale", type=float, default=1.0)
    p.add_argument("--yarn_orig_max_seq_len", type=int, default=2048)
    p.add_argument("--max_seq_len", type=int, default=4096, help="eval context cap (rope cache + truncation)")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge")
    p.add_argument("--limit", type=int, default=0, help="0=all eval examples; else first N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    # FS-B (tree-search long-range selection) on an fsattn cell_topk checkpoint, mirroring eval_ruler.py's
    # fsb_kw / tau handling; without --fsb nothing below is used and the harness is unchanged
    p.add_argument("--fsb", action="store_true", help="FS-B long-range branch (src/fsattn/fsb_tree.py)")
    p.add_argument("--fsb_policy", default="fixed", choices=["fixed", "learned"])
    p.add_argument("--fsb_width", type=int, default=4)
    p.add_argument("--fsb_delta", type=int, default=0)
    p.add_argument("--fsb_widths", default="1,2,4")
    p.add_argument("--fsb_deltas", default="0,1,2")
    p.add_argument("--fsb_node_score", default="bound", choices=["bound", "centroid"])
    p.add_argument("--fsb_bound_radius_coef", type=float, default=1.0)
    p.add_argument("--fsb_leaf_radius", type=int, default=0)
    p.add_argument("--fsb_tree", default="forest", choices=["rebuild", "forest"])
    p.add_argument("--fsb_tau_override", type=float, default=None, help="fill the main forest's fsb_tau (e.g. 0)")
    p.add_argument("--fsb_read_top", type=int, default=0)
    p.add_argument("--fsb_rerank_score", default="leaf", choices=["leaf", "lse", "lse_pos", "max"])
    p.add_argument("--fsb_rerank_query", default="route", choices=["route", "raw"])
    p.add_argument("--fsb_union_width", type=int, default=0)
    p.add_argument("--fsb_union_tau_quantile", type=float, default=0.0)
    p.add_argument("--fsb_union_tau_file", help="portable union calibration JSON for this checkpoint")
    p.add_argument("--fsb_tau_cache_dir", default=None, help="tau cache dir (else $FSB_TAU_CACHE_DIR)")
    p.add_argument("--fsb_tau_calib_batches", type=int, default=8)
    p.add_argument("--fsb_tau_calib_len", type=int, default=4096)
    p.add_argument("--fsb_tau_calib_seed", type=int, default=1234)
    p.add_argument("--fsb_tau_calib_data", default="./data/fineweb_edu_llama32k.bin")
    args = p.parse_args()
    if args.fsb_union_tau_file and not (args.fsb and args.fsb_union_tau_quantile > 0
                                        and args.fsb_union_width > 0 and args.fsb_read_top > 0):
        p.error("--fsb_union_tau_file requires FS-B with a filtered second forest and reranking")
    return args


_SHA_CACHE = {}


def _file_sha256_once(path):
    # sha256 of a file, once per process per (real path, size, mtime); None if the file does not exist
    if not path or not os.path.isfile(path):
        return None
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def fsb_kwargs(args):
    # the FS-B attention kwargs, built exactly like eval_ruler.py's fsb_kw (same keys, same omissions)
    return dict(fsb_enabled=True, fsb_policy=args.fsb_policy, fsb_width=args.fsb_width, fsb_delta=args.fsb_delta,
                fsb_widths=args.fsb_widths, fsb_deltas=args.fsb_deltas, fsb_node_score=args.fsb_node_score,
                fsb_tree=args.fsb_tree, fsb_bound_radius_coef=args.fsb_bound_radius_coef,
                fsb_leaf_radius=args.fsb_leaf_radius,
                **({"fsb_read_top": args.fsb_read_top} if args.fsb_read_top > 0 else {}),
                **({"fsb_rerank_score": args.fsb_rerank_score} if args.fsb_rerank_score != "leaf" else {}),
                **({"fsb_rerank_query": args.fsb_rerank_query} if args.fsb_rerank_query != "route" else {}),
                **({"fsb_union_width": args.fsb_union_width} if args.fsb_union_width > 0 else {}),
                **({"fsb_union_tau_quantile": args.fsb_union_tau_quantile} if args.fsb_union_tau_quantile > 0 else {}))


def fsb_setup(model, args):
    # the main forest's tau override and the second forest's per-head tau from the (canonical) cache, as
    # eval_ruler.py does; refuses to recalibrate silently when no cache dir is given
    if args.fsb_tau_override is not None:
        n = 0
        for m in model.modules():
            if getattr(m, "fsb_enabled", False):
                m.fsb_tau.fill_(float(args.fsb_tau_override)); n += 1
        print(f"[fsb] fsb_tau overridden to {args.fsb_tau_override} in {n} layers", flush=True)
    if args.fsb_union_tau_quantile > 0:
        from experiments.scaling import fsb_tau_calib
        cdir = args.fsb_tau_cache_dir or os.environ.get("FSB_TAU_CACHE_DIR")
        if not cdir and not args.fsb_union_tau_file:
            raise SystemExit("--fsb_union_tau_quantile needs --fsb_tau_cache_dir or FSB_TAU_CACHE_DIR (the canonical tau file)")
        csha = _file_sha256_once(os.path.join(args.checkpoint, "model.safetensors"))
        if args.fsb_union_tau_file:
            from experiments.scaling.fsb_release import load_union_calibration
            info = load_union_calibration(model, args.fsb_union_tau_file, checkpoint_sha256=csha,
                                          quantile=args.fsb_union_tau_quantile)
            info["loaded_from_cache"] = True
            cdir = args.fsb_union_tau_file
        else:
            info = fsb_tau_calib.calibrate_or_load(
                model, ckpt_sha=csha, quantile=args.fsb_union_tau_quantile, n_batches=args.fsb_tau_calib_batches,
                length=args.fsb_tau_calib_len, seed=args.fsb_tau_calib_seed, data_path=args.fsb_tau_calib_data, cache_dir=cdir,
                device="cuda", autocast_dtype=torch.bfloat16, target="fsb_tau_union")
        _, allv = fsb_tau_calib.tau_summary(model, target="fsb_tau_union")
        print(f"[fsb] union tau q={args.fsb_union_tau_quantile:g} {'cached' if info['loaded_from_cache'] else 'COMPUTED'} "
              f"from {cdir}: min {allv[0]:.4f} mean {allv[1]:.4f} max {allv[2]:.4f}", flush=True)


def build_model(args, device):
    cfg = MODEL_CONFIGS[args.config].copy(); cfg["vocab_size"] = 32000
    cfg["max_seq_len"] = args.max_seq_len
    kw = {
        "window_size": args.window_size,
        "long_range_variant": args.long_range_variant,
        "cell_stride": args.cell_stride,
        "cell_topk": args.cell_topk,
        "cell_select": args.cell_select,
    }
    for _pk in ("cell_prune_kernel", "cell_prune_pool", "cell_prune_window",
                "cell_prune_blkchunk", "sel_strict_causal"):
        _pv = getattr(args, _pk, None)
        if _pv is not None:
            kw[_pk] = _pv
    if args.no_rope:
        kw["use_rope"] = False
    if args.fs_nope_longrange:
        kw["rope_long_range"] = False
    if args.fs_uniform_sr:
        kw["fs_uniform_sr"] = True
    if args.fsb:
        if args.attn_type != "fsattn" or args.long_range_variant != "cell_topk":
            raise SystemExit("--fsb needs --attn_type fsattn --long_range_variant cell_topk")
        # FS-B eval env as in eval_ruler.py: no training-only CUDA-graph forest stage on these forwards
        os.environ.setdefault("FSB_TRAIN_GRAPH", "0")
        os.environ.setdefault("FSB_DECODE_GRAPH", "1")
        kw.update(fsb_kwargs(args))
    if args.use_yarn:
        kw["use_yarn"] = True
        kw["yarn_scale"] = args.yarn_scale
        kw["yarn_orig_max_seq_len"] = args.yarn_orig_max_seq_len
    model = FreeSoloLM(attn_type=args.attn_type, **kw, **cfg).to(device)
    ck = args.checkpoint
    if os.path.isdir(ck):
        from safetensors.torch import load_file
        sd = load_file(os.path.join(ck, "model.safetensors"))
    else:
        obj = torch.load(ck, map_location=device, weights_only=False)
        sd = obj.get("model_state_dict", obj)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    check_params_loaded(model, missing, where=f"eval_downstream({ck})")
    model.eval()
    print(f"loaded {ck} (missing={len(missing)}, unexpected={len(unexpected)}, "
          f"attn={args.attn_type}, max_seq_len={args.max_seq_len}, yarn={args.use_yarn})")
    if args.fsb:
        fsb_setup(model, args)
    return model


def _hs_preprocess(text):
    import re as _re
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = _re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def hs_fields(ex):
    ctx = _hs_preprocess(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"])
    cands = [" " + _hs_preprocess(e) for e in ex["endings"]]
    return ctx, cands, int(ex["label"])


def arc_fields(ex):
    ctx = "Question: " + ex["question"] + "\nAnswer:"
    labels = ex["choices"]["label"]
    cands = [" " + c for c in ex["choices"]["text"]]
    gold = labels.index(ex["answerKey"]) if ex["answerKey"] in labels else 0
    return ctx, cands, gold


def piqa_fields(ex):
    ctx = "Question: " + ex["goal"] + "\nAnswer:"
    cands = [" " + ex["sol1"], " " + ex["sol2"]]
    return ctx, cands, int(ex["label"])


def wino_fields(ex):
    s = ex["sentence"]
    p = s.index("_")
    pre, post = s[:p], s[p + 1:]
    ctxs = [pre + ex["option1"], pre + ex["option2"]]
    cands = [post, post]
    gold = int(ex["answer"]) - 1
    return ctxs, cands, gold


def obqa_fields(ex):
    ctx = ex["question_stem"]
    labels = ex["choices"]["label"]
    cands = [" " + c for c in ex["choices"]["text"]]
    gold = labels.index(ex["answerKey"]) if ex["answerKey"] in labels else 0
    return ctx, cands, gold


def sciq_fields(ex):
    sup = (ex.get("support") or "").strip()
    ctx = (sup + "\n" if sup else "") + "Question: " + ex["question"] + "\nAnswer:"
    cands = [" " + ex["distractor1"], " " + ex["distractor2"],
             " " + ex["distractor3"], " " + ex["correct_answer"]]
    return ctx, cands, 3


def boolq_fields(ex):
    ctx = ex["passage"] + "\nQuestion: " + ex["question"] + "?\nAnswer:"
    cands = [" no", " yes"]
    return ctx, cands, int(ex["label"])


TASKS = {
    "hellaswag":     ("Rowan/hellaswag",   None,            "validation", "train", hs_fields,   "acc_norm"),
    "piqa":          ("baber/piqa",         None,            "validation", "train", piqa_fields, "acc"),
    "arc_easy":      ("allenai/ai2_arc",   "ARC-Easy",      "test",       "train", arc_fields,  "acc"),
    "arc_challenge": ("allenai/ai2_arc",   "ARC-Challenge", "test",       "train", arc_fields,  "acc_norm"),
    "winogrande":    ("allenai/winogrande","winogrande_xl", "validation", "train", wino_fields, "acc"),
    "openbookqa":    ("allenai/openbookqa","main",          "test",       "train", obqa_fields, "acc_norm"),
    "sciq":          ("allenai/sciq",       None,            "test",       "train", sciq_fields, "acc"),
    "boolq":         ("aps/super_glue",    "boolq",         "validation", "train", boolq_fields,"acc"),
}


def main():
    args = parse_args()
    device = "cuda"
    dtype = torch.bfloat16
    random.seed(args.seed)
    model = build_model(args, device)
    tok = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B", trust_remote_code=True)
    MAXLEN = args.max_seq_len

    @torch.no_grad()
    def score(context, completion):
        ctx_ids = tok.encode(context)
        full_ids = tok.encode(context + completion)
        comp_ids = full_ids[len(ctx_ids):]
        if not comp_ids:
            return (-1e30, -1e30)
        if len(full_ids) > MAXLEN:
            keep = MAXLEN - len(comp_ids)
            full_ids = (ctx_ids[-keep:] if keep > 0 else []) + comp_ids
        x = torch.tensor(full_ids[:-1], device=device).unsqueeze(0)
        y = torch.tensor(full_ids[1:], device=device).unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=dtype):
            logits = model(x)
        logp = F.log_softmax(logits.float(), dim=-1)
        n = len(comp_ids)
        ll = logp[0, -n:].gather(-1, y[0, -n:].unsqueeze(-1)).squeeze(-1).sum().item()
        char_len = max(len(completion), 1)
        return (ll, ll / char_len)

    def fewshot_prefix(train_ds, fields, k):
        if k == 0:
            return ""
        parts = []
        for j in random.sample(range(len(train_ds)), k):
            ctx, cands, gold = fields(train_ds[j])
            c = ctx[gold] if isinstance(ctx, (list, tuple)) else ctx
            parts.append(c + cands[gold])
        return "\n\n".join(parts) + "\n\n"

    results = {"config": vars(args), "tasks": {}}
    for task in [t for t in args.tasks.split(",") if t]:
        name, subset, split, tsplit, fields, metric = TASKS[task]
        load = (lambda s: load_dataset(name, subset, split=s)) if subset else (lambda s: load_dataset(name, split=s))
        ds, train_ds = load(split), load(tsplit)
        prefix = fewshot_prefix(train_ds, fields, args.num_fewshot)
        n_eval = len(ds) if args.limit == 0 else min(args.limit, len(ds))
        correct = correct_norm = tot = 0
        ctx_lens = []
        for i in range(n_eval):
            ctx, cands, gold = fields(ds[i])
            if isinstance(ctx, (list, tuple)):
                full_ctxs = [prefix + c for c in ctx]
            else:
                full_ctxs = [prefix + ctx] * len(cands)
            ctx_lens.append(len(tok.encode(full_ctxs[0])))
            scored = [score(full_ctxs[j], cands[j]) for j in range(len(cands))]
            raw = [s[0] for s in scored]
            nrm = [s[1] for s in scored]
            if max(range(len(cands)), key=lambda j: raw[j]) == gold:
                correct += 1
            if max(range(len(cands)), key=lambda j: nrm[j]) == gold:
                correct_norm += 1
            tot += 1
        acc = correct / tot
        acc_norm = correct_norm / tot
        primary = acc_norm if metric == "acc_norm" else acc
        mean_ctx = sum(ctx_lens) / len(ctx_lens)
        results["tasks"][task] = {"metric": metric, "score": primary,
                                  "acc": acc, "acc_norm": acc_norm, "n": tot,
                                  "num_fewshot": args.num_fewshot, "mean_ctx_tokens": mean_ctx}
        print(f"{task}: {metric}={primary:.4f} (acc={acc:.4f} acc_norm={acc_norm:.4f}, "
              f"n={tot}, {args.num_fewshot}-shot, mean_ctx={mean_ctx:.0f} tok)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump(results, open(args.output, "w"), indent=2)
        print("wrote", args.output)


if __name__ == "__main__":
    main()
