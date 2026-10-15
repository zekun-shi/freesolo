import os, sys, json, re, argparse, hashlib, time, traceback, tempfile
os.environ.setdefault("FS_FUSED_ATTN", "1")
import torch
from transformers import AutoTokenizer
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "experiments", "scaling"))
sys.path.insert(0, _HERE)
from longctx_infer import load_model
from kv_cache import cached_generate
import longctx_infer as _lci

TOK_NAME = "fla-hub/transformer-1.3B-100B"

_SHA_CACHE = {}

def _fsb_execution_env():
    # Resolved configuration, not a claim that every kernel's eligibility checks passed.
    defaults = {"FSB_FAST_BEAM": "1", "FSB_BEAM_CUDA": "1", "FSB_BEAM_ROWS": "1", "FSB_TRAVERSE_TRITON": "1",
                "FSB_FAST_RERANK": "1", "FSB_FAST_CACHE": "1",
                "FSB_FAST_BUILD": "1", "FSB_BUILD_CUDA": "1", "FSB_FAST_ROWS": "1", "FSB_DECODE_WHOLE": "1",
                "FSB_DECODE_GRAPH": "1", "FSB_DECODE_GRAPH_CHECK": "0", "FSB_TRAIN_GRAPH": "0",
                "FSB_ROWS_IMPL": "cuda", "FSB_ROWS_DEFER": "1", "FSB_ROWS_LOSKIP": "1", "FSB_ROWS_LSE_SPLIT": "4",
                "FSB_BEAM_COMPACT": "1", "FSB_CHECK_COMPACT": "0", "FSB_FAST_PREFILL": "1",
                "FSB_FAST_PREFILL_BLKCHUNK": "128", "FSB_FAST_PREFILL_SCRATCH_MB": "2048", "FSB_PREFILL_STATS": "0"}
    return {name: os.environ.get(name, default) for name, default in defaults.items()}

def _attn_execution_env():
    # inference execution switches of every variant (window, trunk glue, prefill readout and its launch shape, FS-L
    # selection, KV-cache cell statistics); "0" restores the eager path
    defaults = {"FSL_SLIDING_MODE": "chunked", "FSL_WINDOW_BATCHED": "1", "FSL_TRUNK_FUSED": "1",
                "FSL_NORM_ONEPASS": "1", "FSL_ROPE_FUSED": "1", "FSL_PREFILL_CELL_REUSE": "1",
                "FSL_READOUT_FUSED": "1", "FSL_WINDOW_STREAM": "0", "FSL_READOUT_G": "64", "FSL_READOUT_WARPS": "4",
                "FSL_READOUT_STAGES": "2", "FSL_FAST_SELECT": "1", "FSL_FUSED_CONTENT": "1",
                "FSL_FAST_CACHE_STATS": "1"}
    return {name: os.environ.get(name, default) for name, default in defaults.items()}

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

def max_new_for(task):
    if task.startswith("niah"): return 128
    if task == "vt": return 30
    if task == "cwe": return 120
    if task == "fwe": return 50
    if task in ("qa", "qa_1", "qa_2"): return 32
    return 64

def scorer_for(task):
    if task in ("qa", "qa_1", "qa_2"):
        return "string_match_part"
    return "string_match_all"

def postprocess_pred(s):
    s = s.strip()
    return re.sub(r'[\x00-\x1f]', '\n', s).strip()

def score_one(pred, outputs, metric):
    low = postprocess_pred(pred).lower()
    refs = [str(r).lower() for r in outputs]
    if metric == "string_match_part":
        return max([1.0 if r in low else 0.0 for r in refs]) if refs else 0.0
    return sum(1.0 if r in low else 0.0 for r in refs) / max(1, len(refs))

def build_attn_kwargs(cell_topk=4, window_size=128, use_rope=False, rope_long_range=None,
                      fs_uniform_sr=False, sel_content_only=False, fsb=None):
    kw = dict(window_size=window_size, long_range_variant="cell_topk", cell_stride=32, cell_topk=cell_topk,
              cell_select="score", sel_strict_causal=1, use_rope=use_rope)
    if fs_uniform_sr:
        kw["fs_uniform_sr"] = True
    if rope_long_range is not None:
        kw["rope_long_range"] = rope_long_range
    if sel_content_only:
        kw["sel_content_only"] = True
    if fsb:
        kw.update(fsb)   # FS-B ckpts: fsb_enabled + policy/menu (tau + controller weights from the ckpt)
    return kw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="340m", help="MODEL_CONFIGS key (340m, 800m, ...)")
    ap.add_argument("--window_size", type=int, default=128, help="short-range window size")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--cell_topk", type=int, default=4,
                    help="# long-range cells kept for exact token readout (default 4 = training value).")
    ap.add_argument("--use_rope", action="store_true",
                    help="RoPE in the short-range window (use_rope=True).")
    ap.add_argument("--fs_uniform_sr", action="store_true",
                    help="Uniform-window short-range branch (no distance bias).")
    ap.add_argument("--fs_nope_longrange", action="store_true",
                    help="NoPE on the long-range branch (rope_long_range=False).")
    ap.add_argument("--sel_content_only", action="store_true",
                    help="Select long-range cells by content (q.k_bar) only, dropping the "
                         "distance-weighted selection term. Must match a checkpoint trained "
                         "with --sel_content_only.")
    # FS-B long-range branch (threshold admission + content tree + beam traversal); the menu/policy
    # must match training, tau + controller weights load from the checkpoint
    ap.add_argument("--fsb", action="store_true", help="FS-B long-range branch (src/fsattn/fsb_tree.py)")
    ap.add_argument("--fsb_policy", default="learned", choices=["fixed", "learned"])
    ap.add_argument("--fsb_width", type=int, default=4)
    ap.add_argument("--fsb_delta", type=int, default=0)
    ap.add_argument("--fsb_widths", default="1,2,4")
    ap.add_argument("--fsb_deltas", default="0,1,2")
    ap.add_argument("--fsb_node_score", default="bound", choices=["bound", "centroid"])
    ap.add_argument("--fsb_bound_radius_coef", type=float, default=1.0,
                    help="weight of the ||q|| r_v term in the optimistic node bound (1.0 = the bound as trained)")
    ap.add_argument("--fsb_leaf_radius", type=int, default=0,
                    help="1: cells carry the radius of their own token keys (must match training)")
    ap.add_argument("--fsb_tree", default="forest", choices=["rebuild", "forest"],
                    help="FS-B tree organisation (must match training)")
    ap.add_argument("--fsb_prefill_blkchunk", type=int, default=None,
                    help="query blocks per prefill chunk (memory bound)")
    ap.add_argument("--fsb_read_top", type=int, default=0,
                    help="overselect: the beam runs with --fsb_width, then only the K best of its cells by the routing "
                         "leaf score are read (e.g. --fsb_width 8 --fsb_read_top 4); 0 = off")
    ap.add_argument("--fsb_rerank_score", default="leaf", choices=["leaf", "lse", "lse_pos", "max"],
                    help="overselect re-rank score (needs --fsb_read_top): leaf = the mean-key leaf score; lse / max = "
                         "logsumexp / max over the cell's raw token scores; lse_pos = lse + the position term")
    ap.add_argument("--fsb_rerank_query", default="route", choices=["route", "raw"],
                    help="overselect re-rank query (needs --fsb_read_top): route = the routing query the beam traverses "
                         "with; raw = the raw q scores the re-rank (identical without a routing adapter)")
    ap.add_argument("--fsb_union_width", type=int, default=0,
                    help="overselect union candidate pool (needs --fsb_read_top): the re-rank's candidates are the main "
                         "beam's cells union the cells of a second beam of width U run with the same routing query "
                         "(duplicates dropped); 0 = off")
    ap.add_argument("--fsb_hybrid", default="off", choices=["off", "pool", "union", "quota"],
                    help="hybrid FS-L pool + FS-B tree selector (needs --fsb_read_top, forest, fixed policy; raw "
                         "re-rank query with an adapter): pool = the pool top-K only; union = pool shortlist union "
                         "the tree candidates; quota = reserve --fsb_pool_quota pool slots, the tree fills the rest")
    ap.add_argument("--fsb_pool_size", type=int, default=None,
                    help="hybrid pool: top-P cells by max key norm (library default 64; 0 = every eligible cell)")
    ap.add_argument("--fsb_pool_recent", type=int, default=None, help="hybrid pool: recent-cell window (library default 8)")
    ap.add_argument("--fsb_pool_short", type=int, default=None,
                    help="hybrid union: pool shortlist size unioned with the tree candidates (library default 4)")
    ap.add_argument("--fsb_pool_quota", type=int, default=None,
                    help="hybrid quota: reserved pool slots (library default 1; 1..--fsb_read_top)")
    ap.add_argument("--fsb_route_adapter", default="none", choices=["none", "lowrank", "full"],
                    help="FS-B routing adapter of a router-distilled checkpoint (must match training); "
                         "none = raw-q routing")
    ap.add_argument("--fsb_route_rank", type=int, default=32, help="lowrank adapter rank (must match training)")
    ap.add_argument("--fsb_route_alpha", type=float, default=1.0,
                    help="adapter scale at eval (0 = raw-q routing with the adapter weights loaded; with "
                         "--fsb_route_learn_c 1, alpha 0 still routes with the learned c_h)")
    ap.add_argument("--fsb_route_learn_c", type=int, default=0,
                    help="1: the checkpoint carries a learned per-head radius coefficient (fsb_route_ctheta) that "
                         "replaces --fsb_bound_radius_coef in every traversal")
    ap.add_argument("--fsb_route_fixed_c", type=int, default=0,
                    help="1 (needs --fsb_route_learn_c 1): load the learned c_h but route with the fixed "
                         "--fsb_bound_radius_coef in every traversal")
    ap.add_argument("--fsb_rerank_rr", default="none", choices=["none", "linear"],
                    help="residual re-ranker on the overselect's lse_pos score (needs --fsb_read_top, "
                         "--fsb_rerank_score lse_pos and, with an adapter, --fsb_rerank_query raw); weights load "
                         "from --ckpt or --fsb_router_file")
    ap.add_argument("--fsb_router_file", default=None,
                    help="router-only checkpoint (fsb_route.safetensors, or a dir holding it) overlaid onto --ckpt "
                         "after loading, with a strict key check")
    ap.add_argument("--fsb_router_allow_mismatch", type=int, default=0,
                    help="1: overlay a --fsb_router_file whose recorded re-ranker identity differs from this eval's "
                         "--ckpt and flags (reported instead of refused)")
    ap.add_argument("--fsb_tau_override", type=float, default=None,
                    help="set every layer/head admission threshold to this value after loading (0 = admit every "
                         "eligible cell), overriding the checkpoint's fsb_tau")
    ap.add_argument("--fsb_tau_quantile", type=float, default=0.0,
                    help="> 0: set every layer/head tau to this quantile of its max-key-norm distribution over "
                         "--fsb_tau_calib_batches training chunks of --fsb_tau_calib_len tokens before the eval; "
                         "cells below tau never enter the tree (the hybrid pool ignores tau). The tensor is cached "
                         "(--fsb_tau_cache_dir) keyed by ckpt sha / q / N / len / seed. 0 = off; mutually exclusive "
                         "with --fsb_tau_override")
    ap.add_argument("--fsb_tau_calib_batches", type=int, default=8, help="calibration chunks N")
    ap.add_argument("--fsb_tau_calib_len", type=int, default=4096, help="tokens per calibration chunk")
    ap.add_argument("--fsb_tau_calib_seed", type=int, default=1234, help="seed of the chunk draw")
    ap.add_argument("--fsb_tau_calib_data", default="./data/fineweb_edu_llama32k.bin",
                    help="pretokenized uint16 corpus; chunks come from its first 98%% (inside the trainer's train split)")
    ap.add_argument("--fsb_tau_cache_dir", default=None,
                    help="directory of the cached tau tensor (default: $FSB_TAU_CACHE_DIR, else the --output directory)")
    ap.add_argument("--fsb_read_margin", type=float, default=0.0,
                    help="nats: after the exact re-rank (needs --fsb_read_top) read every candidate whose score is "
                         "within this margin of the best candidate's, at least --fsb_read_min and at most "
                         "--fsb_read_top cells; 0 = off; refused with --fsb_hybrid pool / quota")
    ap.add_argument("--fsb_read_min", type=int, default=1,
                    help="minimum cells read under --fsb_read_margin (1..--fsb_read_top)")
    ap.add_argument("--fsb_beam_quota", type=int, default=0,
                    help="k > 0 (needs --fsb_read_top, forest, fixed policy; k <= --fsb_read_top and <= --fsb_width): "
                         "the first k valid cells of the main beam are always read, the exact re-rank order fills the "
                         "remaining slots; refused with --fsb_hybrid pool and --fsb_read_margin; 0 = off")
    ap.add_argument("--fsb_union_tau_quantile", type=float, default=0.0,
                    help="q > 0 (needs --fsb_union_width > 0, --fsb_read_top > 0, forest, fixed policy): the second "
                         "beam traverses a second forest over the cells admitted by a per-head threshold calibrated "
                         "at this quantile on training text (the same calibration, cache and --fsb_tau_calib_* "
                         "settings as --fsb_tau_quantile; stored in fsb_tau_union, so the main forest keeps its own "
                         "tau); 0 = off")
    ap.add_argument("--fsb_union_tau_file", help="portable union calibration JSON for this checkpoint; skips corpus calibration")
    ap.add_argument("--fsb_union_quota", type=int, default=0,
                    help="k > 0 (needs --fsb_union_width > 0 and --fsb_read_top > 0, forest, fixed policy; "
                         "k <= --fsb_read_top): k read slots reserved for the best second-beam-only cells, re-ranked "
                         "among themselves, the main beam's re-rank order fills the rest; refused with --fsb_hybrid "
                         "pool / union, --fsb_read_margin and --fsb_rerank_rr; 0 = off")
    ap.add_argument("--fsb_union_prior_scale", type=float, default=1.0,
                    help="s in [0, 1] (needs --fsb_union_width > 0 and --fsb_read_top > 0, forest, fixed policy): "
                         "the exact re-rank's position / key-norm prior term of the second-beam-only candidates is "
                         "multiplied by s (0 = they compete on content alone) while the main beam's keep the full "
                         "lse_pos; 1 = off")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    if a.fsb_union_tau_file and not (a.fsb and a.fsb_union_tau_quantile > 0):
        ap.error("--fsb_union_tau_file requires --fsb and a positive --fsb_union_tau_quantile")
    if not a.fsb:
        # FS-B-only flags are ignored without --fsb: refuse instead of recording dead flags
        _stray = [f_ for f_, on_ in (("--fsb_hybrid", a.fsb_hybrid != "off"), ("--fsb_pool_size", a.fsb_pool_size is not None),
                                     ("--fsb_pool_recent", a.fsb_pool_recent is not None),
                                     ("--fsb_pool_short", a.fsb_pool_short is not None),
                                     ("--fsb_pool_quota", a.fsb_pool_quota is not None),
                                     ("--fsb_rerank_rr", a.fsb_rerank_rr != "none"),
                                     ("--fsb_router_file", a.fsb_router_file is not None),
                                     ("--fsb_router_allow_mismatch", bool(a.fsb_router_allow_mismatch)),
                                     ("--fsb_tau_quantile", a.fsb_tau_quantile > 0),
                                     ("--fsb_read_margin", a.fsb_read_margin > 0),
                                     ("--fsb_read_min", a.fsb_read_min != 1),
                                     ("--fsb_beam_quota", a.fsb_beam_quota > 0),
                                     ("--fsb_union_tau_quantile", a.fsb_union_tau_quantile > 0),
                                     ("--fsb_union_quota", a.fsb_union_quota > 0),
                                     ("--fsb_union_prior_scale", a.fsb_union_prior_scale != 1.0)) if on_]
        if _stray:
            ap.error(f"{' '.join(_stray)} need --fsb (they would be silently ignored)")
    if a.fsb_union_tau_quantile < 0 or a.fsb_union_tau_quantile >= 1:
        ap.error(f"--fsb_union_tau_quantile {a.fsb_union_tau_quantile}: 0 (off) or a quantile in (0, 1)")
    if a.fsb_union_tau_quantile > 0 and (a.fsb_union_width <= 0 or a.fsb_read_top <= 0):
        ap.error("--fsb_union_tau_quantile filters the SECOND beam's forest: it needs --fsb_union_width > 0 and --fsb_read_top > 0")
    if a.fsb_union_quota < 0:
        ap.error(f"--fsb_union_quota {a.fsb_union_quota}: 0 (off) or the number of reserved union slots (1..--fsb_read_top)")
    if a.fsb_union_quota > 0 and (a.fsb_union_width <= 0 or a.fsb_read_top <= 0):
        ap.error("--fsb_union_quota reserves read slots for the SECOND beam's cells: it needs --fsb_union_width > 0 and --fsb_read_top > 0")
    if a.fsb_union_quota > a.fsb_read_top:
        ap.error(f"--fsb_union_quota {a.fsb_union_quota} > --fsb_read_top {a.fsb_read_top} (the reserved union slots must fit the read)")
    if not (0.0 <= a.fsb_union_prior_scale <= 1.0):
        ap.error(f"--fsb_union_prior_scale {a.fsb_union_prior_scale}: 1 (off) or a scale in [0, 1]")
    if a.fsb_union_prior_scale != 1.0 and (a.fsb_union_width <= 0 or a.fsb_read_top <= 0):
        ap.error("--fsb_union_prior_scale scales the SECOND beam's candidates' prior term: it needs --fsb_union_width > 0 and "
                 "--fsb_read_top > 0")
    if a.fsb_route_fixed_c and not a.fsb_route_learn_c:
        ap.error("--fsb_route_fixed_c 1 needs --fsb_route_learn_c 1 (without a learned c_h the coefficient is already fixed)")
    if a.fsb_tau_quantile > 0 and a.fsb_tau_override is not None:
        ap.error("--fsb_tau_quantile and --fsb_tau_override are mutually exclusive (one calibrates tau, the other sets it)")
    if a.fsb_tau_quantile < 0 or a.fsb_tau_quantile >= 1:
        ap.error(f"--fsb_tau_quantile {a.fsb_tau_quantile}: 0 (off) or a quantile in (0, 1)")
    if a.fsb:
        # the training-only CUDA-graph forest stage would blow memory on these no-cache forwards; decode graph
        # replay is bit-identical to eager decode. A value the caller exported wins.
        os.environ.setdefault("FSB_TRAIN_GRAPH", "0")
        os.environ.setdefault("FSB_DECODE_GRAPH", "1")
        fsb_env = _fsb_execution_env()
        print("[freesolo] FS-B eval env: " + " ".join(f"{k}={v}" for k, v in fsb_env.items())
              + " " + " ".join(f"{k}={os.environ.get(k, '<unset>')}" for k in (
                  "FS_STRICT_CAUSAL", "FS_PRUNE_KERNEL", "FS_PRUNE_POOL")), flush=True)
    attn_env = _attn_execution_env()
    print("[freesolo] attention eval env: " + " ".join(f"{k}={v}" for k, v in attn_env.items()), flush=True)
    _lci.MODEL_CONFIGS[a.config]["vocab_size"] = 32000
    fsb_kw = (dict(fsb_enabled=True, fsb_policy=a.fsb_policy, fsb_width=a.fsb_width, fsb_delta=a.fsb_delta,
                   fsb_widths=a.fsb_widths, fsb_deltas=a.fsb_deltas, fsb_node_score=a.fsb_node_score,
                   fsb_tree=a.fsb_tree, fsb_bound_radius_coef=a.fsb_bound_radius_coef, fsb_leaf_radius=a.fsb_leaf_radius,
                   **({"fsb_prefill_blkchunk": a.fsb_prefill_blkchunk} if a.fsb_prefill_blkchunk is not None else {}),
                   **({"fsb_read_top": a.fsb_read_top} if a.fsb_read_top > 0 else {}),
                   **({"fsb_rerank_score": a.fsb_rerank_score} if a.fsb_rerank_score != "leaf" else {}),
                   **({"fsb_rerank_query": a.fsb_rerank_query} if a.fsb_rerank_query != "route" else {}),
                   **({"fsb_union_width": a.fsb_union_width} if a.fsb_union_width > 0 else {}),
                   **({"fsb_read_margin": a.fsb_read_margin} if a.fsb_read_margin > 0 else {}),
                   **({"fsb_read_min": a.fsb_read_min} if a.fsb_read_min != 1 else {}),
                   **({"fsb_beam_quota": a.fsb_beam_quota} if a.fsb_beam_quota > 0 else {}),
                   **({"fsb_union_tau_quantile": a.fsb_union_tau_quantile} if a.fsb_union_tau_quantile > 0 else {}),
                   **({"fsb_union_quota": a.fsb_union_quota} if a.fsb_union_quota > 0 else {}),
                   **({"fsb_union_prior_scale": a.fsb_union_prior_scale} if a.fsb_union_prior_scale != 1.0 else {}),
                   **({"fsb_hybrid": a.fsb_hybrid} if a.fsb_hybrid != "off" else {}),
                   **{k_: getattr(a, k_) for k_ in ("fsb_pool_size", "fsb_pool_recent", "fsb_pool_short", "fsb_pool_quota")
                      if getattr(a, k_) is not None},
                   **({"fsb_route_adapter": a.fsb_route_adapter, "fsb_route_rank": a.fsb_route_rank,
                       "fsb_route_alpha": a.fsb_route_alpha} if a.fsb_route_adapter != "none" else {}),
                   **({"fsb_route_learn_c": 1} if a.fsb_route_learn_c else {}),
                   **({"fsb_rerank_rr": a.fsb_rerank_rr} if a.fsb_rerank_rr != "none" else {}))
              if a.fsb else None)
    attn_kw = build_attn_kwargs(a.cell_topk, window_size=a.window_size, use_rope=a.use_rope,
                                rope_long_range=(False if a.fs_nope_longrange else None),
                                fs_uniform_sr=a.fs_uniform_sr,
                                sel_content_only=a.sel_content_only, fsb=fsb_kw)

    tok = AutoTokenizer.from_pretrained(TOK_NAME, trust_remote_code=True)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 2
    lean = a.length >= 16384
    model = load_model(a.ckpt, config=a.config, attn_type="fsattn", device="cuda",
                       max_seq_len=a.length + 256, **attn_kw)
    if a.fsb_router_file and not a.fsb:
        ap.error("--fsb_router_file needs --fsb")
    _rkeys = None
    if a.fsb and (a.fsb_router_file or a.fsb_rerank_rr != "none"):
        # router tensors from --fsb_router_file (strict overlay) or --ckpt: every fsb_route_* tensor must be
        # loaded (a missing key keeps its zero init = the identity)
        from src.fsattn.fsb_route import router_key_check
        _rrep, _rkeys = router_key_check(model, a.ckpt, a.fsb_router_file, allow_mismatch=bool(a.fsb_router_allow_mismatch))
        print(f"[freesolo] {_rrep}", flush=True)
        if a.fsb_rerank_rr != "none":
            _rw = torch.stack([_m.fsb_route_rr_w.detach().float() for _m in model.modules() if getattr(_m, "fsb_enabled", False)])
            print(f"[freesolo] residual re-ranker {a.fsb_rerank_rr}: |w| mean {float(_rw.abs().mean()):.4e} max "
                  f"{float(_rw.abs().max()):.4e} over {_rw.shape[0]} layers", flush=True)
    elif a.fsb:
        # no re-ranker in the model: a checkpoint carrying re-ranker tensors would silently rank with the identity
        from src.fsattn.fsb_route import router_key_check
        if os.path.isfile(os.path.join(a.ckpt, "model.safetensors")):
            _rkeys = router_key_check(model, a.ckpt, None)[1]
    if a.fsb and a.fsb_route_adapter != "none":
        # the adapter weights MUST come from the checkpoint (a missing key would silently leave B / W at zero = no-op)
        from safetensors import safe_open
        _sf = a.fsb_router_file or os.path.join(a.ckpt, "model.safetensors")
        if _rkeys is not None:
            _keys = _rkeys                                               # the source the router tensors came from
        else:
            with safe_open(_sf, framework="pt") as _f:
                _keys = {k.replace("_orig_mod.", "").replace("module.", "") for k in _f.keys()}
        _need = [n for n, _ in model.named_parameters() if n.rsplit(".", 1)[-1].startswith("fsb_route_")]
        _miss = [n for n in _need if n not in _keys]
        assert _need and not _miss, f"--fsb_route_adapter {a.fsb_route_adapter}: {len(_miss)}/{len(_need)} router params missing from {_sf}: {_miss[:4]}"
        _has_c = any(k.endswith("fsb_route_ctheta") for k in _keys)
        assert _has_c == bool(a.fsb_route_learn_c), (f"{_sf} {'carries' if _has_c else 'has no'} a learned radius coefficient "
                                                     f"(fsb_route_ctheta) but --fsb_route_learn_c {a.fsb_route_learn_c}")
        if a.fsb_route_learn_c:
            from src.fsattn.fsb_route import radius_summary
            _cs = radius_summary(model)
            print(f"[freesolo] learned radius coefficient c_h loaded: mean {_cs[0]:.4f} [{_cs[1]:.4f}, {_cs[2]:.4f}] "
                  f"(--fsb_bound_radius_coef {a.fsb_bound_radius_coef} is only its init)", flush=True)
            if a.fsb_route_fixed_c:
                # set before the first forward, so no graph ever holds a c buffer
                for _m in model.modules():
                    if getattr(_m, "fsb_enabled", False):
                        _m.fsb_route_learn_c = False
                print(f"[freesolo] --fsb_route_fixed_c 1: the learned c_h is IGNORED; every traversal routes with the fixed "
                      f"--fsb_bound_radius_coef {a.fsb_bound_radius_coef}", flush=True)
            elif a.fsb_route_alpha == 0.0:
                print("[freesolo] NOTE: --fsb_route_alpha 0 with the learned c_h: raw-q routing but NOT the backbone's own "
                      "routing (add --fsb_route_fixed_c 1 for that)", flush=True)
        from src.fsattn.fsb_route import adapter_head_norms
        _hn = torch.stack([adapter_head_norms(_m) for _m in model.modules() if getattr(_m, "fsb_enabled", False)])
        print(f"[freesolo] routing adapter {a.fsb_route_adapter} r={a.fsb_route_rank} alpha={a.fsb_route_alpha}: "
              f"{len(_need)} tensors loaded, ||adapter||_F mean {_hn.mean():.4f} max {_hn.max():.4f}", flush=True)
    if a.fsb and a.fsb_tau_override is not None:
        _nt = 0
        for _m in model.modules():
            if getattr(_m, "fsb_enabled", False):
                _m.fsb_tau.fill_(float(a.fsb_tau_override)); _nt += 1
        print(f"[freesolo] fsb_tau overridden to {a.fsb_tau_override} in {_nt} layers", flush=True)
    if a.fsb and a.fsb_tau_quantile > 0:
        # per-head tau of the main forest from training chunks (cached; keyed by the checkpoint sha)
        import fsb_tau_calib
        _cdir = a.fsb_tau_cache_dir or os.environ.get("FSB_TAU_CACHE_DIR") or os.path.dirname(os.path.abspath(a.output))
        _csha = _file_sha256_once(os.path.join(a.ckpt, "model.safetensors"))
        if _csha is None:
            ap.error(f"--fsb_tau_quantile: {a.ckpt}/model.safetensors not found (the tau cache is keyed by its sha256)")
        _tau_info = fsb_tau_calib.calibrate_or_load(
            model, ckpt_sha=_csha, quantile=a.fsb_tau_quantile, n_batches=a.fsb_tau_calib_batches,
            length=a.fsb_tau_calib_len, seed=a.fsb_tau_calib_seed, data_path=a.fsb_tau_calib_data, cache_dir=_cdir,
            device="cuda", autocast_dtype=torch.bfloat16)
        _per, _all = fsb_tau_calib.tau_summary(model)
        _afl = _tau_info.get("admit_frac_calib_layer") or [float("nan")] * len(_per)
        print(f"[fsb-tau] q={a.fsb_tau_quantile:g} N={a.fsb_tau_calib_batches} len={a.fsb_tau_calib_len} "
              f"seed={a.fsb_tau_calib_seed} {'cached' if _tau_info['loaded_from_cache'] else 'computed'} "
              f"tau over layers x heads: min {_all[0]:.4f} mean {_all[1]:.4f} max {_all[2]:.4f}; "
              f"admitted fraction on the calibration cells {_tau_info.get('admit_frac_calib_mean', float('nan')):.3f}", flush=True)
        for _li, (_mn, _me, _mx) in enumerate(_per):
            print(f"[fsb-tau] layer {_li:2d}: tau min {_mn:.4f} mean {_me:.4f} max {_mx:.4f}  admit_calib {_afl[_li]:.3f}", flush=True)
    if a.fsb and a.fsb_union_tau_quantile > 0:
        # the second beam's per-head threshold fsb_tau_union from the same training chunks / cache (the main
        # forest's fsb_tau is untouched by this call)
        import fsb_tau_calib
        _cdir = a.fsb_tau_cache_dir or os.environ.get("FSB_TAU_CACHE_DIR") or os.path.dirname(os.path.abspath(a.output))
        _csha = _file_sha256_once(os.path.join(a.ckpt, "model.safetensors"))
        if _csha is None:
            ap.error(f"--fsb_union_tau_quantile: {a.ckpt}/model.safetensors not found (the tau cache is keyed by its sha256)")
        if a.fsb_union_tau_file:
            from experiments.scaling.fsb_release import load_union_calibration
            _utau_info = load_union_calibration(model, a.fsb_union_tau_file, checkpoint_sha256=_csha,
                                                quantile=a.fsb_union_tau_quantile)
            _utau_info["loaded_from_cache"] = True
        else:
            _utau_info = fsb_tau_calib.calibrate_or_load(
                model, ckpt_sha=_csha, quantile=a.fsb_union_tau_quantile, n_batches=a.fsb_tau_calib_batches,
                length=a.fsb_tau_calib_len, seed=a.fsb_tau_calib_seed, data_path=a.fsb_tau_calib_data, cache_dir=_cdir,
                device="cuda", autocast_dtype=torch.bfloat16, target="fsb_tau_union")
        _uper, _uall = fsb_tau_calib.tau_summary(model, target="fsb_tau_union")
        _mper, _mall = fsb_tau_calib.tau_summary(model)
        _uafl = _utau_info.get("admit_frac_calib_layer") or [float("nan")] * len(_uper)
        print(f"[fsb-utau] union q={a.fsb_union_tau_quantile:g} N={a.fsb_tau_calib_batches} len={a.fsb_tau_calib_len} "
              f"seed={a.fsb_tau_calib_seed} {'cached' if _utau_info['loaded_from_cache'] else 'computed'} "
              f"tau_union over layers x heads: min {_uall[0]:.4f} mean {_uall[1]:.4f} max {_uall[2]:.4f}; admitted fraction "
              f"on the calibration cells {_utau_info.get('admit_frac_calib_mean', float('nan')):.3f}; the MAIN forest's tau "
              f"min {_mall[0]:.4f} mean {_mall[1]:.4f} max {_mall[2]:.4f}", flush=True)
        for _li, (_mn, _me, _mx) in enumerate(_uper):
            print(f"[fsb-utau] layer {_li:2d}: tau_union min {_mn:.4f} mean {_me:.4f} max {_mx:.4f}  admit_calib {_uafl[_li]:.3f}", flush=True)
    if a.fsb:
        from src.fsattn.fsb_tree import fsb_read_stats_reset, fsb_read_stats
    print(f"[freesolo] {a.ckpt} tok={TOK_NAME} vocab={tok.vocab_size} eos={eos} lean={lean} "
          f"attn={attn_kw}", flush=True)

    def _fsb_admit_frac(key="admit_frac"):
        # the last prefill's admitted-cell fraction (admit_frac: the main forest; admit_frac_union: the
        # filtered second forest of --fsb_union_tau_quantile): per FS-B layer (mean over heads), mean over layers
        _st = [_m._fsb_stats[key] for _m in model.modules()
               if getattr(_m, "fsb_enabled", False) and _m._fsb_stats and key in _m._fsb_stats]
        if not _st:
            return None, None
        _lay = torch.stack(_st).float().mean(1).tolist()
        return _lay, sum(_lay) / len(_lay)

    results = {}
    _first_prompt = True
    for task in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        jf = os.path.join(a.data_dir, task, "validation.jsonl")
        if not os.path.exists(jf):
            results[task] = {"status": "missing_jsonl"}; print(f"  {task}: MISSING", flush=True); continue
        recs = [json.loads(x) for x in open(jf)]
        if a.limit: recs = recs[:a.limit]
        mx = min(max_new_for(task), a.max_new); metric = scorer_for(task)
        scores = []; t0 = time.time(); status = "ok"
        # FS-B per-example read count (cells read per (layer, head, query) row over prefill + decode) and the
        # prefill's admitted-cell fraction (the noise filter's footprint)
        _rd = {"sum": 0.0, "rows": 0.0, "max": 0.0, "af": [], "afu": [], "ut": 0.0, "utf": [], "uq": 0.0, "uqf": []}
        try:
            for ri, r in enumerate(recs):
                prompt = r["input"] + r.get("answer_prefix", "")
                ids = torch.tensor(tok.encode(prompt), dtype=torch.long)
                P = ids.numel()
                if a.fsb:
                    fsb_read_stats_reset(model)
                out = cached_generate(model, ids, mx, device="cuda", dtype="bfloat16",
                                      eos_token_id=eos, lean_prefill=lean)
                pred = tok.decode(out[0, P:])
                scores.append(score_one(pred, r["outputs"], metric))
                if a.fsb:
                    _rs = fsb_read_stats(model)
                    _afl, _af = _fsb_admit_frac()
                    _rd["sum"] += _rs["mean"] * _rs["rows"]; _rd["rows"] += _rs["rows"]; _rd["max"] = max(_rd["max"], _rs["max"])
                    if _af is not None:
                        _rd["af"].append(_af)
                    _aful = _afu = None
                    if a.fsb_union_tau_quantile > 0 or a.fsb_union_quota > 0:
                        # the read cells that came from the second beam only (a fraction of every cell read)
                        _rd["ut"] += _rs["utau_cells"]; _rd["utf"].append(_rs["utau_frac"])
                    if a.fsb_union_tau_quantile > 0:
                        # the filtered second forest's footprint: its admitted fraction
                        _aful, _afu = _fsb_admit_frac("admit_frac_union")
                        if _afu is not None:
                            _rd["afu"].append(_afu)
                    if a.fsb_union_quota > 0:
                        # the union quota's footprint: the read cells that came through the reserved union slots
                        _rd["uq"] += _rs["uq_cells"]; _rd["uqf"].append(_rs["uq_frac"])
                    if _first_prompt:
                        _first_prompt = False
                        if _afl is not None:
                            print(f"[fsb-admit] first prompt ({task} L{a.length} #{ri}, P={P}): admitted-cell fraction per layer "
                                  + " ".join(f"{v:.3f}" for v in _afl) + f" | mean {_af:.4f}", flush=True)
                        print(f"[fsb-read] first prompt: cells read per row mean {_rs['mean']:.3f} max {_rs['max']:.0f} over "
                              f"{_rs['rows']:.0f} rows; per layer " + " ".join(f"{v:.2f}" for v in _rs["layer_mean"]), flush=True)
                        if _aful is not None:
                            print(f"[fsb-utau] first prompt: second forest admitted-cell fraction per layer "
                                  + " ".join(f"{v:.3f}" for v in _aful) + f" | mean {_afu:.4f}; reads from the second beam only "
                                  f"{_rs['utau_cells']:.0f} of {_rs['mean'] * _rs['rows']:.0f} cells read ({_rs['utau_frac']:.4f})", flush=True)
                        if a.fsb_union_quota > 0:
                            print(f"[fsb-uq] first prompt: reads through the {a.fsb_union_quota} reserved union slot(s) "
                                  f"{_rs['uq_cells']:.0f} of {_rs['mean'] * _rs['rows']:.0f} cells read ({_rs['uq_frac']:.4f}); "
                                  f"from the second beam only {_rs['utau_cells']:.0f} ({_rs['utau_frac']:.4f})", flush=True)
                del out, ids
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            status = "OOM"; torch.cuda.empty_cache()
        except Exception as e:
            status = "ERR:" + str(e)[:120]; traceback.print_exc()
        sc = round(sum(scores) / len(scores) * 100, 2) if scores else None
        results[task] = {"score": sc, "n": len(scores), "seconds": round(time.time() - t0),
                         "status": status, "metric": metric, "max_new": mx}
        _pk = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        results[task]["peak_alloc_gib"] = round(_pk, 2)
        _rdmsg = ""
        if a.fsb and _rd["rows"] > 0:
            results[task]["read_cells_mean"] = round(_rd["sum"] / _rd["rows"], 4)
            results[task]["read_cells_max"] = _rd["max"]
            if _rd["af"]:
                results[task]["admit_frac_mean"] = round(sum(_rd["af"]) / len(_rd["af"]), 4)
            _rdmsg = (f" read_cells mean={results[task]['read_cells_mean']:.3f} max={_rd['max']:.0f}"
                      + (f" admit_frac={results[task]['admit_frac_mean']:.4f}" if _rd["af"] else ""))
            if (a.fsb_union_tau_quantile > 0 or a.fsb_union_quota > 0) and _rd["utf"]:
                results[task]["utau_read_frac_mean"] = round(sum(_rd["utf"]) / len(_rd["utf"]), 4)
                results[task]["utau_read_cells_mean"] = round(_rd["ut"] / len(_rd["utf"]), 2)
                if _rd["afu"]:
                    results[task]["admit_frac_union_mean"] = round(sum(_rd["afu"]) / len(_rd["afu"]), 4)
                _rdmsg += (f" utau_read_frac={results[task]['utau_read_frac_mean']:.4f}"
                           + (f" admit_frac_union={results[task]['admit_frac_union_mean']:.4f}" if _rd["afu"] else ""))
            if a.fsb_union_quota > 0 and _rd["uqf"]:
                results[task]["uq_read_frac_mean"] = round(sum(_rd["uqf"]) / len(_rd["uqf"]), 4)
                results[task]["uq_read_cells_mean"] = round(_rd["uq"] / len(_rd["uqf"]), 2)
                _rdmsg += f" uq_read_frac={results[task]['uq_read_frac_mean']:.4f}"
        print(f"[freesolo L{a.length}] {task}: score={sc} n={len(scores)} metric={metric} "
              f"mx={mx} {results[task]['seconds']}s {status} peak_alloc={_pk:.1f}GiB{_rdmsg}", flush=True)

    variant = "FS-B" if a.fsb else ("FS-L" if int(os.environ.get("FS_PRUNE_KERNEL", "0")) else "FS")
    report = {"meta": {"ckpt": os.path.abspath(a.ckpt), "length": a.length, "tokenizer": TOK_NAME, "eos": eos,
                        "variant": variant, "config": a.config, "limit": a.limit,
                        "attn_execution_env": attn_env,
                        **({"fsb_execution_env": fsb_env} if a.fsb else {}),
                        **({"union_tau_sha256": _utau_info["tau_sha256"]}
                           if a.fsb and a.fsb_union_tau_quantile > 0 else {}),
                        "attn_kwargs": attn_kw, "cell_topk": a.cell_topk,
                        "lean_prefill": lean, "fsb_route_fixed_c": int(a.fsb_route_fixed_c),
                        **({"fsb_router_file": a.fsb_router_file} if a.fsb_router_file else {}),
                        **({"fsb_union_quota": a.fsb_union_quota} if a.fsb_union_quota > 0 else {}),
                        **({"fsb_union_prior_scale": a.fsb_union_prior_scale} if a.fsb_union_prior_scale != 1.0 else {})},
               "results": results}
    with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(os.path.abspath(a.output)),
                                     prefix=".ruler-", suffix=".json", delete=False) as stream:
        temporary = stream.name
        json.dump(report, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, a.output)
    failed = [task for task, result in results.items()
              if result.get("status") != "ok" or result.get("n", 0) == 0]
    if failed or not results:
        raise SystemExit(f"RULER evaluation incomplete: {failed or 'no tasks'}; report: {a.output}")
    print("FREESOLO_DONE", a.output, flush=True)

if __name__ == "__main__":
    main()
