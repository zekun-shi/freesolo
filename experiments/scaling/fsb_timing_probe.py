import argparse
import collections
import functools
import importlib
import json
import os
import re
import runpy
import statistics
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
os.environ["FSB_DECODE_WHOLE"] = "0"     # the stage timers wrap the per-layer decode functions, not one step graph

# (stage, module under the attention package, qualname); absent functions are skipped and reported
STAGES = (
    ("forest_build", "fsb_tree", "build_forest"),
    ("forest_build", "fsb_tree", "prefill_forests"),
    ("forest_build", "fsb_tree", "_forest_build_stage"),
    ("forest_ingest", "fsb_tree", "ForestCache.ingest"),
    ("admission_append", "fsb_tree", "ForestCache._append_admitted"),
    ("forest_pack", "fsb_tree", "_flatten_forest"),
    ("forest_pack", "fsb_tree", "ForestCache._pack_lean"),
    ("spine_build", "fsb_tree", "build_spines"),
    ("traversal", "fsb_tree", "traverse_forest"),
    ("traversal", "fsb_tree", "_forest_chunk_stage"),
    ("traversal", "fsb_tree", "traverse"),
    ("chunk_trees", "fsb_tree", "build_trees"),
    ("chunk_trees", "fsb_build_cuda", "forests_cuda"),
    ("rerank", "fsb_tree", "rerank_top"),
    ("token_scores", "fsb_rerank", "cell_token_scores"),
    ("readout", "fsb_tree", "masked_readout"),
    ("readout", "fsb_tree", "_fsb_decode_readout"),
    ("readout", "fsb_fast_step", "decode_readout_fast"),
    ("lr_prefill", "fsb_tree", "fsb_long_range"),
    ("lr_decode", "fsb_tree", "fsb_decode_step"),
    ("decode_capture", "fsb_tree", "_capture_decode_graph"),
)
DISCOVER_MODULES = ("fsb_tree", "fsb_hybrid", "fsb_pool")
DISCOVER = (("hybrid_merge", re.compile(r"quota|merge")), ("pool_build", re.compile(r"pool")))
DISCOVER_EXCLUDE = re.compile(r"^_fsb_graph_pool$|_kw$|^union_candidates$")
ENV_PREFIXES = ("FS_", "FSB_", "FSL_", "KV_", "PYTORCH_CUDA", "CUDA_VISIBLE_DEVICES")


def _scalar(v):
    if isinstance(v, (list, tuple)):
        return all(isinstance(x, (bool, int, float, str)) for x in v)
    return isinstance(v, (bool, int, float, str)) or v is None


def fsb_resolved(model):
    # resolved fsb_* attributes of the first FS-B layer, plus min/mean/max of fsb_tau over all FS-B
    # layers (after any tau override); None for a non-FS-B model
    layers = [m for m in model.modules() if getattr(m, "fsb_enabled", False)]
    if not layers:
        return None
    a0 = layers[0]
    attrs = {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(vars(a0).items())
             if k.startswith("fsb_") and _scalar(v)}
    for k in ("cell_stride", "cell_topk", "_cell_block", "window_size", "recompute_long_range"):
        if _scalar(getattr(a0, k, None)):
            attrs[k] = getattr(a0, k)
    out = {"n_layers": len(layers), "attrs": attrs}
    try:
        import torch
        tau = torch.stack([m.fsb_tau.detach().float().flatten() for m in layers])
        out["tau"] = {"min": float(tau.min()), "mean": float(tau.mean()), "max": float(tau.max())}
        if float(getattr(a0, "fsb_union_tau_quantile", 0.0) or 0.0) > 0:
            tu = torch.stack([m.fsb_tau_union.detach().float().flatten() for m in layers])
            out["tau_union"] = {"min": float(tu.min()), "mean": float(tu.mean()), "max": float(tu.max())}
    except Exception:
        pass
    return out


class StageTimer:
    def __init__(self, torch):
        self.torch = torch
        self.active = False
        self.phase = "prefill"
        self.recs = []                                   # (phase, stage, ev0, ev1)
        self.captured = collections.Counter()
        self.trunk_decode = []                           # per decode token: (ev0, ev1) of the trunk call
        self.wrapped = collections.defaultdict(list)
        self.depth = collections.Counter()              # open timed calls per stage: only the outermost is timed

    def wrap(self, stage, fn, label):
        torch = self.torch
        self.wrapped[stage].append(label)

        @functools.wraps(fn)
        def w(*a, **k):
            if not self.active:
                return fn(*a, **k)
            if torch.cuda.is_current_stream_capturing():
                self.captured[stage] += 1
                return fn(*a, **k)
            if self.depth[stage]:
                return fn(*a, **k)
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            self.depth[stage] += 1
            try:
                out = fn(*a, **k)
            finally:
                self.depth[stage] -= 1
            e1.record()
            self.recs.append((self.phase, stage, e0, e1))
            return out
        w._fsb_timing_orig = fn
        return w

    def begin(self):
        self.recs, self.trunk_decode = [], []
        self.captured.clear()
        self.phase = "prefill"
        self.active = True

    def end(self):
        self.torch.cuda.synchronize()
        self.active = False
        agg = {"prefill": collections.defaultdict(lambda: [0.0, 0]), "decode": collections.defaultdict(lambda: [0.0, 0])}
        for ph, st, e0, e1 in self.recs:
            a = agg[ph][st]
            a[0] += e0.elapsed_time(e1); a[1] += 1
        trunk_dec = [e0.elapsed_time(e1) for e0, e1 in self.trunk_decode]
        return ({ph: {st: {"ms": round(v[0], 4), "calls": v[1]} for st, v in sorted(d.items())} for ph, d in agg.items()},
                trunk_dec, dict(self.captured))


def _resolve(mod, qualname):
    obj = mod
    parts = qualname.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def install(timer, pkg, extra):
    # patch the FS-B stage functions of package ``pkg`` (the model's, e.g. src.fsattn); returns the absent names
    absent = []
    mods = {}
    for name in set(m for _, m, _ in STAGES) | set(DISCOVER_MODULES):
        try:
            mods[name] = importlib.import_module(f"{pkg}.{name}")
        except ImportError:
            mods[name] = None
    done = set()
    for stage, mname, qual in STAGES:
        mod = mods.get(mname)
        try:
            owner, attr = _resolve(mod, qual)
            fn = getattr(owner, attr)
        except AttributeError:
            absent.append(f"{mname}.{qual}"); continue
        if mod is None:
            absent.append(f"{mname}.{qual}"); continue
        setattr(owner, attr, timer.wrap(stage, fn, f"{mname}.{qual}"))
        done.add((mname, qual))
    import inspect
    for mname in DISCOVER_MODULES:
        mod = mods.get(mname)
        if mod is None:
            continue
        for fname, fn in sorted(vars(mod).items()):
            if (not inspect.isfunction(fn) or fn.__module__ != mod.__name__ or (mname, fname) in done
                    or DISCOVER_EXCLUDE.search(fname)):
                continue
            for stage, rx in DISCOVER:
                if rx.search(fname):
                    setattr(mod, fname, timer.wrap(stage, fn, f"{mname}.{fname}"))
                    done.add((mname, fname))
                    break
    for spec in extra:
        stage, target = spec.split("=", 1)
        mname, qual = target.split(":", 1)
        mod = importlib.import_module(mname if "." in mname else f"{pkg}.{mname}")
        owner, attr = _resolve(mod, qual)
        setattr(owner, attr, timer.wrap(stage, getattr(owner, attr), f"{mname}.{qual}"))
    return absent


def _med(xs):
    return round(statistics.median(xs), 4) if xs else None


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--"); pargs, dargs = argv[:i], argv[i + 1:]
    else:
        pargs, dargs = argv, []
    ap = argparse.ArgumentParser(description="FS-B per-stage timing probe (eval-only, opt-in; no library change)")
    ap.add_argument("--decode", type=int, default=32, help="decode tokens per timed generation (after the prefill)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--synthetic_len", type=int, default=0, help="L > 0: L random prompt ids instead of the data prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--driver", default=os.path.join(_ROOT, "eval_ruler.py"))
    ap.add_argument("--wrap", action="append", default=[], help="STAGE=module:qualname (module relative to the model "
                                                                 "package unless dotted)")
    a = ap.parse_args(pargs)
    assert a.decode >= 0 and a.reps >= 1 and a.warmup >= 0

    tmp = tempfile.mkdtemp(prefix="fsb_timing_", dir="/tmp")
    has = lambda f: f in dargs
    if a.synthetic_len > 0:
        if not has("--length"):
            dargs += ["--length", str(a.synthetic_len)]
        if not has("--data_dir"):
            os.makedirs(os.path.join(tmp, "data", "synthetic"))
            with open(os.path.join(tmp, "data", "synthetic", "validation.jsonl"), "w") as f:
                f.write(json.dumps({"input": "synthetic", "outputs": ["x"], "answer_prefix": ""}) + "\n")
            dargs += ["--data_dir", os.path.join(tmp, "data"), "--tasks", "synthetic"]
        L = int(dargs[dargs.index("--length") + 1])
        if a.synthetic_len + a.decode + 1 > L + 256:
            ap.error(f"--synthetic_len {a.synthetic_len} + --decode {a.decode} + 1 exceeds the model's max_seq_len "
                     f"(--length {L} + 256)")
    if not has("--output"):
        dargs += ["--output", os.path.join(tmp, "driver.json")]
    if not has("--limit"):
        dargs += ["--limit", "1"]

    for p in (_ROOT, _HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import kv_cache

    timer = StageTimer(torch)
    state = {"installed": False, "absent": [], "examples": [], "fsb": None, "Bq": None}
    orig_gen = kv_cache.cached_generate

    trunk_w = timer.wrap("trunk", kv_cache._trunk_cached, "kv_cache._trunk_cached")
    trunk_orig = kv_cache._trunk_cached

    def trunk(model, idx, cache):
        if not timer.active:
            return trunk_orig(model, idx, cache)
        timer.phase = "prefill" if idx.shape[1] > 1 else "decode"
        n0 = len(timer.recs)
        out = trunk_w(model, idx, cache)
        if timer.phase == "decode" and len(timer.recs) > n0 and timer.recs[-1][1] == "trunk":
            timer.trunk_decode.append(timer.recs[-1][2:])
        return out

    kv_cache._trunk_cached = trunk
    kv_cache._lm_head_fp32 = timer.wrap("lm_head", kv_cache._lm_head_fp32, "kv_cache._lm_head_fp32")
    gen = torch.Generator(device="cpu").manual_seed(a.seed)

    def timed_generate(model, prompt_ids, max_new_tokens, *args, **kw):
        if not state["installed"]:
            layers = [m for m in model.modules() if getattr(m, "fsb_enabled", False)]
            if layers:
                pkg = type(layers[0]).__module__.rsplit(".", 1)[0]
                state["absent"] = install(timer, pkg, a.wrap)
                state["Bq"] = int(layers[0]._cell_block)
            state["fsb"] = fsb_resolved(model)
            state["installed"] = True
        ids = prompt_ids
        if a.synthetic_len > 0:
            ids = torch.randint(3, 32000, (a.synthetic_len,), generator=gen, dtype=torch.long)
        P = int(ids.numel())
        kw = dict(kw); kw["eos_token_id"] = None
        reps = []
        out = None
        for r in range(a.warmup + a.reps):
            timed = r >= a.warmup
            del out
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            if timed:
                timer.begin()
            w0 = time.perf_counter()
            out = orig_gen(model, ids, a.decode + 1, *args, **kw)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - w0) * 1e3
            if not timed:
                continue
            stages, trunk_dec, captured = timer.end()
            n_dec = len(trunk_dec)
            pre, dec = stages["prefill"], stages["decode"]
            ms = lambda d, k: d.get(k, {}).get("ms", 0.0)
            Bq = state["Bq"] or 128
            bnd = [k for k in range(n_dec) if k == 0 or (P + k) % Bq == 0]    # decode steps at a query-block change
            steady = [t for k, t in enumerate(trunk_dec) if k not in bnd]
            reps.append(dict(
                wall_ms=round(wall, 3), peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                prefill_total_ms=round(ms(pre, "trunk") + ms(pre, "lm_head"), 4),
                decode_tokens=n_dec,
                decode_total_ms=round(ms(dec, "trunk") + ms(dec, "lm_head"), 4),
                decode_per_token_ms=round((ms(dec, "trunk") + ms(dec, "lm_head")) / max(1, n_dec), 4),
                decode_trunk_steady_median_ms=_med(steady),
                decode_trunk_block_change_ms=[round(trunk_dec[k], 4) for k in bnd],
                prefill=pre, decode=dec,
                decode_per_token={k_: round(v_["ms"] / max(1, n_dec), 4) for k_, v_ in dec.items()},
                captured_calls=captured))
        ex = dict(index=len(state["examples"]), prompt_tokens=P, synthetic=a.synthetic_len > 0,
                  lean_prefill=bool(kw.get("lean_prefill", False)), reps=reps)
        ex["median"] = _summ(reps)
        state["examples"].append(ex)
        print(f"[fsb_timing] ex {ex['index']} P={P} prefill {ex['median']['prefill_total_ms']} ms, decode "
              f"{ex['median']['decode_per_token_ms']} ms/token ({a.decode} tokens, {a.reps} reps)", flush=True)
        return out

    kv_cache.cached_generate = timed_generate
    sys.argv = [a.driver] + dargs
    print(f"[fsb_timing] driver {a.driver} {' '.join(dargs)}", flush=True)
    runpy.run_path(a.driver, run_name="__main__")

    allreps = [r for ex in state["examples"] for r in ex["reps"]]
    res = dict(probe="fsb_timing_probe", argv=list(sys.orig_argv) if hasattr(sys, "orig_argv") else None,
               driver=a.driver, driver_args=dargs, decode_tokens=a.decode, reps=a.reps, warmup=a.warmup,
               synthetic_len=a.synthetic_len, torch=torch.__version__,
               gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               code_rev=os.environ.get("CODE_REV") or None,
               env={k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)},
               fsb=state["fsb"], stages_wrapped={k: v for k, v in sorted(timer.wrapped.items())},
               stages_absent=state["absent"],
               notes=("inclusive CUDA-event ms summed over layers; forest_build / forest_ingest include chunk_trees; "
                      "forest_ingest also includes admission_append and forest_pack (allocation and packing); "
                      "traversal includes rerank + token_scores; lr_prefill includes forest_build / spine_build / "
                      "traversal / readout; lr_decode includes forest_ingest / spine_build / decode_capture; trunk "
                      "includes lr_*; decode_per_token_ms averages over the N tokens including the block-change steps "
                      "(decode_trunk_block_change_ms; steady state: decode_trunk_steady_median_ms); with "
                      "FSB_DECODE_GRAPH=1 per-token decode stages replay inside CUDA graphs, invisible to the stage "
                      "timers except the eager warm-up at each block change"),
               summary=_summ(allreps), examples=state["examples"])
    line = json.dumps(res)
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        with open(a.json_out, "w") as f:
            f.write(line + "\n")
    print("FSB_TIMING_JSON " + line, flush=True)


def _summ(reps):
    if not reps:
        return {}
    keys = ("wall_ms", "prefill_total_ms", "decode_total_ms", "decode_per_token_ms", "decode_trunk_steady_median_ms",
            "peak_alloc_gib")
    out = {k: _med([r[k] for r in reps if r.get(k) is not None]) for k in keys}
    st_p = sorted({s for r in reps for s in r["prefill"]})
    st_d = sorted({s for r in reps for s in r["decode_per_token"]})
    out["prefill_stage_ms"] = {s: _med([r["prefill"].get(s, {}).get("ms", 0.0) for r in reps]) for s in st_p}
    out["decode_stage_ms_per_token"] = {s: _med([r["decode_per_token"].get(s, 0.0) for r in reps]) for s in st_d}
    return out


if __name__ == "__main__":
    main()
