import os, sys, time, json, argparse, statistics, gc, hashlib

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# deployed serve config (mirrors the eval driver)
os.environ.setdefault("FS_STRICT_CAUSAL", "1")
os.environ.setdefault("FS_PRUNE_POOL", "64")
os.environ.setdefault("FS_PRUNE_WINDOW", "8")
os.environ.setdefault("FS_PRUNE_BLKCHUNK", "16")

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

DT = torch.bfloat16

# inference execution switches shared by every serve config (window, trunk glue, readout, FS-L selection, cache cell
# statistics; the FS / FS-L stream forks: prefill window beside the long-range branch, graphed decode read beside the
# window); "0" restores the eager path. FSL_KV_BOUNDED=1: the FS-L graph decode keeps only the K / V cells it can still
# read (src/fsattn/kv_bounded.py)
ATTN_FLAGS = {"FSL_WINDOW_BATCHED": "1", "FSL_TRUNK_FUSED": "1", "FSL_NORM_ONEPASS": "1", "FSL_ROPE_FUSED": "1",
              "FSL_PREFILL_CELL_REUSE": "1", "FSL_READOUT_FUSED": "1", "FSL_WINDOW_STREAM": "0",
              "FSL_FAST_SELECT": "1", "FSL_FUSED_CONTENT": "1", "FSL_FAST_CACHE_STATS": "1",
              "FS_WINDOW_OVERLAP": "1", "FS_DECODE_FORK": "1", "FSL_KV_BOUNDED": "1"}


def attn_flags():
    return {k: os.environ.get(k, v) for k, v in ATTN_FLAGS.items()}


# FS-B execution switches and their defaults: the table eval_ruler.py records in meta.fsb_execution_env
# (tests/test_ruler_execution_metadata.py checks that the two agree)
FSB_FLAGS = {"FSB_FAST_BEAM": "1", "FSB_BEAM_CUDA": "1", "FSB_BEAM_ROWS": "1", "FSB_TRAVERSE_TRITON": "1",
             "FSB_FAST_RERANK": "1", "FSB_FAST_CACHE": "1", "FSB_FAST_BUILD": "1", "FSB_BUILD_CUDA": "1", "FSB_FAST_ROWS": "1",
             "FSB_DECODE_WHOLE": "1", "FSB_DECODE_GRAPH": "1", "FSB_DECODE_GRAPH_CHECK": "0", "FSB_DECODE_V2": "1",
             "FSB_DECODE_FORK": "1", "FSB_DECODE_JOIN": "1", "FSB_DECODE_ASYNC": "1", "FSB_DECODE_V3": "1",
             "FSB_TRAIN_GRAPH": "0",
             "FSB_ROWS_IMPL": "cuda", "FSB_ROWS_DEFER": "1", "FSB_ROWS_LOSKIP": "1", "FSB_ROWS_LSE_SPLIT": "4",
             "FSB_BEAM_COMPACT": "1", "FSB_CHECK_COMPACT": "0", "FSB_FAST_PREFILL": "1",
             "FSB_FAST_PREFILL_BLKCHUNK": "128", "FSB_FAST_PREFILL_SCRATCH_MB": "2048", "FSB_PREFILL_STATS": "0",
             "FSB_WINDOW_OVERLAP": "1", "FSB_BUILD_PREFILL_TRIM": "1", "FSB_PREFILL_NOCOPY": "1", "FSB_ROWS_V3": "1",
             "FSB_BEAM_V2": "1", "FSB_BEAM_V2_EXACT": "1", "FSB_BUILD_SMALL": "1", "FSB_BUILD_UNIFORM": "1",
             "FSB_BEAM_LONG": "1"}


def fsb_flags():
    return {k: os.environ.get(k, v) for k, v in FSB_FLAGS.items()}


def fsl_fast_status():
    # whether the FS-L CUDA paths ran or fell back (an NVRTC failure only warns), after a measurement
    try:
        from src.fsattn.fsl_select_cuda import status
        return status()
    except Exception as e:
        return {"error": repr(e)}


def fsb_fast_status():
    # the FS-B fallback latches (a CUDA kernel that failed to build only warns) and the fast prefill's fallback
    try:
        from src.fsattn.fsb_prefill import fast_status
        return fast_status()
    except Exception as e:
        return {"error": repr(e)}


def serve_kwargs(serve, fsb_union_tau_quantile=0.0):
    # SR-512 = RoPE + 512 window, NoPE long-range score top-4 stride-32; "fsb" adds the paper FS-B serve config
    kw = dict(window_size=512, long_range_variant="cell_topk", cell_stride=32, cell_topk=4,
              cell_select="score", sel_strict_causal=1, use_rope=True,
              fs_uniform_sr=True, rope_long_range=False)
    if serve == "sparse":
        kw.update(cell_prune_kernel=1, cell_prune_pool=64, cell_prune_window=8, cell_prune_blkchunk=16)
    elif serve == "fsb":
        kw.update(cell_prune_kernel=0, fsb_enabled=True, fsb_policy="fixed", fsb_width=16, fsb_delta=0,
                  fsb_node_score="bound", fsb_bound_radius_coef=8.0, fsb_tree="forest",
                  fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_union_width=16)
        if fsb_union_tau_quantile > 0:
            kw["fsb_union_tau_quantile"] = fsb_union_tau_quantile
    else:  # dense: prune OFF -> exact dense top-k over ALL completed cells
        kw.update(cell_prune_kernel=0)
    return kw


def load(max_len, serve, ckpt, fsb_union_tau_quantile=0.0, fused=True, config="340m"):
    os.environ["FS_FUSED_ATTN"] = "1" if fused else "0"
    os.environ["FS_PRUNE_KERNEL"] = "1" if serve == "sparse" else "0"
    import experiments.scaling.model as _esm
    _esm.MODEL_CONFIGS[config]["vocab_size"] = 32000
    from longctx_infer import load_model, extend_rope_cache
    m = load_model(ckpt, config=config, attn_type="fsattn", device="cuda",
                   max_seq_len=max_len + 256, **serve_kwargs(serve, fsb_union_tau_quantile))
    extend_rope_cache(m, max_len + 256)
    return m.to(DT).eval()


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def apply_fsb_tau(m, args, out):
    # --fsb_tau_override 0 of the eval driver: admit every eligible cell
    n = 0
    for mod in m.modules():
        if getattr(mod, "fsb_enabled", False):
            mod.fsb_tau.fill_(0.0); n += 1
    print(f"[FS-B] fsb_tau overridden to 0 in {n} layers", flush=True)
    if args.fsb_union_tau_quantile <= 0:
        return None
    sf = os.path.join(args.checkpoint, "model.safetensors")
    if not os.path.isfile(sf):
        raise SystemExit(f"{sf} not found (the union-tau cache is keyed by its sha256)")
    if args.fsb_union_tau_file:
        from experiments.scaling.fsb_release import load_union_calibration
        info = load_union_calibration(m, args.fsb_union_tau_file,
                                      checkpoint_sha256=_file_sha256(sf),
                                      quantile=args.fsb_union_tau_quantile)
        return {"loaded_from_cache": True, "cache_file": args.fsb_union_tau_file, **info}
    if not args.fsb_tau_calib_data:
        raise SystemExit("--fsb_union_tau_quantile needs --fsb_tau_calib_data (pretokenized uint16 corpus .bin)"
                         " or --fsb_union_tau_file (the checkpoint's union_calibration.json)")
    from experiments.scaling.fsb_tau_calib import calibrate_or_load
    cdir = (args.fsb_tau_cache_dir or os.environ.get("FSB_TAU_CACHE_DIR")
            or os.path.dirname(os.path.abspath(out)))
    return calibrate_or_load(m, ckpt_sha=_file_sha256(sf), quantile=args.fsb_union_tau_quantile,
                             n_batches=args.fsb_tau_calib_batches, length=args.fsb_tau_calib_len,
                             seed=args.fsb_tau_calib_seed, data_path=args.fsb_tau_calib_data,
                             cache_dir=cdir, device="cuda", target="fsb_tau_union")


def make_trunk(compiled):
    # fresh (compiled) trunk per model/len so dynamo guards do not leak across reloads;
    # suppress_errors => the custom kernels graph-break instead of crashing
    from kv_cache import _trunk_cached
    if not compiled:
        return _trunk_cached
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
    dynamo.config.cache_size_limit = 64
    return torch.compile(_trunk_cached, dynamic=False, fullgraph=False)


def _gb_count():
    try:
        from torch._dynamo.utils import counters
        return sum(counters.get("graph_break", {}).values())
    except Exception:
        return -1


def _reset_mem():
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def bench_prefill(m, L, trunk, n_med=3, warm=8):
    from kv_cache import make_caches
    from longctx_infer import _lm_head_fp32
    torch.manual_seed(1234 + L)
    x = torch.randint(0, m.vocab_size, (1, L), device="cuda")
    ntail = min(64, L)
    tail = [None]

    def one(save_tail=False):
        cache = make_caches(m, retain_kv=False)
        with torch.amp.autocast("cuda", dtype=DT):
            h = trunk(m, x, cache)
            if save_tail:
                tail[0] = h[:, -ntail:, :].detach().clone()
            _ = _lm_head_fp32(m, h[:, -1:, :], 4096)[:, 0, :]
        return cache

    _reset_mem()
    for _ in range(warm):
        one(); torch.cuda.synchronize()
    ts = []
    for i in range(n_med):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        one(save_tail=(i == n_med - 1)); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    dt = statistics.median(ts)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    ref_logits = None
    if tail[0] is not None:
        with torch.amp.autocast("cuda", dtype=DT):
            ref_logits = _lm_head_fp32(m, tail[0], 4096)
        tail[0] = None
    return dict(ktoks=L / dt / 1e3, ms=dt * 1e3, peak_gib=peak, reps=ts, ref_logits=ref_logits)


@torch.no_grad()
def bench_prefill_graph(m, L, n_med=3, warm=8, ref_logits=None):
    from decode_graph import GraphedPrefiller
    torch.manual_seed(1234 + L)
    x = torch.randint(0, m.vocab_size, (1, L), device="cuda")
    _reset_mem()
    gp = GraphedPrefiller(m, L, B=1, device="cuda", dtype=DT, tail_tokens=min(64, L), retain_kv=False)
    try:
        gp.capture(x)
        for _ in range(warm):
            gp.run(x); torch.cuda.synchronize()
        ts = []
        for _ in range(n_med):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            gp.run(x); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        dt = statistics.median(ts)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        res = dict(ktoks=L / dt / 1e3, ms=dt * 1e3, peak_gib=peak, reps=ts)
        if ref_logits is not None:
            got = gp.tail_logits()
            res["max_dlogit"] = float((got - ref_logits).abs().max().item())
            res["top1_match"] = f"{int((got.argmax(-1) == ref_logits.argmax(-1)).sum().item())}/{got.shape[-2]}"
        return res
    finally:
        del gp
        for b in m.blocks:
            b.attn.fsattn._fsb_stats = None
        gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def bench_decode(m, ctx, serve, trunk, decode_mode="eager", steps=96, n_med=3, warm=8):
    # eager: fused single-token KV-cache decode (the cached_generate loop, prefill untimed).
    # graph: one captured CUDA graph per decode step (sparse -> GraphedDecoder, fsb -> FSBGraphedDecoder, re-captured
    # at each query block after the eager forest update; both included in the timing).
    from kv_cache import make_caches
    from longctx_infer import _lm_head_fp32
    torch.manual_seed(4321 + ctx)
    prompt = torch.randint(0, m.vocab_size, (1, ctx), device="cuda")
    _reset_mem()

    total_steps = warm + n_med * steps          # every gd.step()/eager step advances one position
    rates = None
    kv = {}
    if serve in ("dense", "sparse", "fsb") and decode_mode == "graph":
        from decode_graph import FSBGraphedDecoder, GraphedDecoder
        if serve == "fsb":
            gd = FSBGraphedDecoder(m, prompt, total_steps + 16, device="cuda", dtype=DT)
        else:
            gd = GraphedDecoder(m, prompt, Lmax=ctx + total_steps + 16, device="cuda", dtype=DT)
        gd.prefill()
        for _ in range(warm):
            gd.step()
        torch.cuda.synchronize()
        rates = []
        for _ in range(n_med):
            t0 = time.perf_counter()
            for _ in range(steps):
                gd.step()
            torch.cuda.synchronize()
            rates.append(steps / (time.perf_counter() - t0))
        if serve != "fsb":
            kv = {"kv_bounded": gd.bounded_kv, "kv_faults": list(gd.kv_faults())}
    else:
        os.environ["FS_FUSED_ATTN"] = "1"
        if serve == "fsb":
            os.environ["FSB_DECODE_GRAPH"] = "1" if decode_mode == "graph" else "0"
        try:
            # KV reserved for every decoded token and pre-allocated, as cached_generate does (otherwise the first decode
            # token doubles every layer's KV buffer and the peak counts that copy)
            cache = make_caches(m, reserve=total_steps + 16)
            for c in cache:
                if hasattr(c, "preallocate"):
                    c.preallocate(1, ctx, DT, "cuda")
            with torch.amp.autocast("cuda", dtype=DT):
                h = trunk(m, prompt, cache)
                nxt = _lm_head_fp32(m, h[:, -1:, :], 4096)[:, 0, :].argmax(-1).unsqueeze(1)
                for _ in range(warm):
                    h = trunk(m, nxt, cache)
                    nxt = _lm_head_fp32(m, h[:, -1:, :], 4096)[:, 0, :].argmax(-1).unsqueeze(1)
                torch.cuda.synchronize()
                rates = []
                for _ in range(n_med):
                    t0 = time.perf_counter()
                    for _ in range(steps):
                        h = trunk(m, nxt, cache)
                        nxt = _lm_head_fp32(m, h[:, -1:, :], 4096)[:, 0, :].argmax(-1).unsqueeze(1)
                    torch.cuda.synchronize()
                    rates.append(steps / (time.perf_counter() - t0))
        finally:
            if serve == "fsb":
                os.environ.pop("FSB_DECODE_GRAPH", None)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return dict(tps=statistics.median(rates), peak_gib=peak, reps=rates, **kv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="accelerate safetensors dir (vocab 32000)")
    ap.add_argument("--config", default="340m", help="model size in experiments/scaling/model.py MODEL_CONFIGS")
    ap.add_argument("--serve", choices=["dense", "sparse", "fsb"], required=True)
    ap.add_argument("--lengths", default="4096,16384,65536,131072")
    ap.add_argument("--cap40", type=int, default=1, help="1=cap allocatable to 40GB (0.5 of 80GB)")
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--nmed", type=int, default=3)
    ap.add_argument("--warm", type=int, default=8, help="untimed prefill passes / decode steps before timing")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip_decode", type=int, default=0)
    ap.add_argument("--skip_prefill", type=int, default=0)
    ap.add_argument("--compile", type=int, default=0, help="1=torch.compile the trunk (prefill+eager decode)")
    ap.add_argument("--fsb_union_tau_quantile", type=float, default=0.0,
                    help="fsb serve only: q > 0 calibrates fsb_tau_union on training chunks before benching")
    ap.add_argument("--fsb_tau_calib_data", default=None, help="pretokenized uint16 corpus .bin for the tau calibration")
    ap.add_argument("--fsb_tau_calib_batches", type=int, default=8)
    ap.add_argument("--fsb_tau_calib_len", type=int, default=4096)
    ap.add_argument("--fsb_tau_calib_seed", type=int, default=1234)
    ap.add_argument("--fsb_tau_cache_dir", default=None)
    ap.add_argument("--fsb_union_tau_file", default="",
                    help="portable union calibration JSON for this checkpoint; skips corpus calibration")
    args = ap.parse_args()
    if args.fsb_union_tau_file and (args.serve != "fsb" or args.fsb_union_tau_quantile <= 0):
        ap.error("--fsb_union_tau_file requires --serve fsb and a positive --fsb_union_tau_quantile")

    if args.cap40 and torch.cuda.get_device_properties(0).total_memory > (50 << 30):
        torch.cuda.set_per_process_memory_fraction(0.5, 0)  # 40 of 80 GB
    if args.serve in ("dense", "sparse"):
        os.environ["FS_FUSED_SELECT"] = "1"
    if args.serve == "fsb":
        os.environ.setdefault("FSB_TRAIN_GRAPH", "0")     # eval-mode forwards must not grow CUDA graphs
    os.environ.setdefault("FSL_SLIDING_MODE", "chunked")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} dev={torch.cuda.get_device_name(0)} "
          f"serve={args.serve} cap40={args.cap40} compile={args.compile} "
          f"FUSED_SELECT={os.environ.get('FS_FUSED_SELECT','0')} "
          f"SLIDING_MODE={os.environ.get('FSL_SLIDING_MODE')} "
          + " ".join(f"{k}={v}" for k, v in attn_flags().items())
          + ("".join(f" {k}={v}" for k, v in fsb_flags().items()) if args.serve == "fsb" else ""), flush=True)

    lengths = [int(x) for x in args.lengths.split(",")]
    label = {"sparse": "FS-L", "fsb": "FS-B"}.get(args.serve, "FS")
    out = args.out or f"/tmp/efficiency_{args.serve}_cap{args.cap40}_compile{args.compile}.json"
    decode_impl = {"dense": {"eager": "eager trunk+lm_head", "graph": "GraphedDecoder (graphed single-token decode)"},
                   "sparse": {"eager": "eager trunk+lm_head", "graph": "GraphedDecoder (graphed single-token decode)"},
                   "fsb": {"eager": "eager trunk+lm_head (FSB_DECODE_GRAPH=0)",
                           "graph": "FSBGraphedDecoder (one graph per decode step, re-captured per query block)"}}
    rows = []
    for L in lengths:
        rec = {"serve": args.serve, "config": args.config, "label": label, "L": L, "cap40": args.cap40, "compile": args.compile,
               "checkpoint": args.checkpoint, "sliding_mode": os.environ.get("FSL_SLIDING_MODE"),
               "attn_flags": attn_flags(), **({"fsb_flags": fsb_flags()} if args.serve == "fsb" else {})}
        m = None
        try:
            m = load(L, args.serve, args.checkpoint, args.fsb_union_tau_quantile, fused=True, config=args.config)
            if args.serve == "fsb":
                info = apply_fsb_tau(m, args, out)
                if info is not None:
                    rec["fsb_tau_union"] = {"quantile": args.fsb_union_tau_quantile,
                                            "loaded_from_cache": info["loaded_from_cache"],
                                            "cache_file": info["cache_file"]}
        except torch.cuda.OutOfMemoryError:
            rec["load"] = "OOM"; rows.append(rec); print(f"[{label} L={L}] LOAD OOM", flush=True)
            torch.cuda.empty_cache(); continue

        trunk = make_trunk(args.compile)

        if not args.skip_prefill:
            ref_logits = None
            try:
                pr = bench_prefill(m, L, trunk, n_med=args.nmed, warm=args.warm)
                ref_logits = pr.get("ref_logits")
                rec["prefill_ktoks"] = round(pr["ktoks"], 2)
                rec["prefill_peak_gib"] = round(pr["peak_gib"], 2)
                rec["prefill_ms"] = round(pr["ms"], 1)
                rec["prefill_graph_breaks"] = _gb_count()
                print(f"[{label} L={L}] prefill {pr['ktoks']:6.2f} Ktok/s  "
                      f"{pr['ms']:8.1f} ms  peak {pr['peak_gib']:.2f} GiB  gb={rec['prefill_graph_breaks']}", flush=True)
            except torch.cuda.OutOfMemoryError:
                rec["prefill_ktoks"] = "OOM"; rec["prefill_peak_gib"] = "OOM"
                print(f"[{label} L={L}] prefill OOM", flush=True)
                torch.cuda.empty_cache()
            if not args.compile and os.environ.get("FS_PREFILL_GRAPH", "1") != "0":
                try:
                    pg = bench_prefill_graph(m, L, n_med=args.nmed, warm=args.warm, ref_logits=ref_logits)
                    rec["prefill_graph_ktoks"] = round(pg["ktoks"], 2)
                    rec["prefill_graph_peak_gib"] = round(pg["peak_gib"], 2)
                    rec["prefill_graph_ms"] = round(pg["ms"], 1)
                    if "max_dlogit" in pg:
                        rec["prefill_graph_max_dlogit"] = round(pg["max_dlogit"], 6)
                        rec["prefill_graph_top1_match"] = pg["top1_match"]
                    p_str = (f"  max|dlogit|={pg['max_dlogit']:.4g} top1={pg['top1_match']}"
                             if "max_dlogit" in pg else "")
                    print(f"[{label} L={L}] prefill[graph] {pg['ktoks']:6.2f} Ktok/s  "
                          f"{pg['ms']:8.1f} ms  peak {pg['peak_gib']:.2f} GiB{p_str}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    rec["prefill_graph_ktoks"] = "OOM"; rec["prefill_graph_peak_gib"] = "OOM"
                    print(f"[{label} L={L}] prefill[graph] OOM", flush=True)
                    torch.cuda.empty_cache()
            del ref_logits

        if not args.skip_decode:
            for dm in (["eager", "graph"] if args.serve in ("dense", "sparse", "fsb") else ["eager"]):
                try:
                    de = bench_decode(m, L, args.serve, trunk, decode_mode=dm, steps=args.steps, n_med=args.nmed,
                                      warm=args.warm)
                    rec[f"decode_{dm}_tps"] = round(de["tps"], 2)
                    rec[f"decode_{dm}_peak_gib"] = round(de["peak_gib"], 2)
                    rec[f"decode_{dm}_impl"] = decode_impl[args.serve][dm]
                    kv_str = ""
                    if "kv_bounded" in de:
                        rec[f"decode_{dm}_kv_bounded"] = de["kv_bounded"]
                        rec[f"decode_{dm}_kv_faults"] = de["kv_faults"]
                        kv_str = f"  kv_bounded={int(de['kv_bounded'])} faults={de['kv_faults']}"
                    print(f"[{label} L={L}] decode[{dm}]  {de['tps']:6.2f} tok/s  "
                          f"peak {de['peak_gib']:.2f} GiB  reps={[round(r,1) for r in de['reps']]}{kv_str}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    rec[f"decode_{dm}_tps"] = "OOM"; rec[f"decode_{dm}_peak_gib"] = "OOM"
                    print(f"[{label} L={L}] decode[{dm}] OOM", flush=True)
                    torch.cuda.empty_cache()

        rec["fsl_fast_status"] = fsl_fast_status()
        if args.serve == "fsb":
            rec["fsb_fast_status"] = fsb_fast_status()
        rows.append(rec)
        del m, trunk; gc.collect(); torch.cuda.empty_cache()
        try:
            import torch._dynamo as dynamo; dynamo.reset()
        except Exception:
            pass

    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("WROTE", out, flush=True)
    print("SUMMARY", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
