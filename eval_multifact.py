import argparse, hashlib, json, os, sys, time
os.environ.pop("FS_FUSED_ATTN", None)   # masked-pool fsattn path
import torch
from transformers import AutoTokenizer
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "experiments", "scaling"))
sys.path.insert(0, _HERE)
import longctx_infer as _lci
from longctx_infer import load_model
from kv_cache import FSKVCache, cached_generate, make_caches, _trunk_cached

TOK_NAME = "fla-hub/transformer-1.3B-100B"
DECODE_TOKENS = 16
LM_CHUNK = 4096
_ENV_PREFIXES = ("FS_", "FSB_", "FSL_", "KV_", "PYTORCH_CUDA", "CUDA_VISIBLE_DEVICES")

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


# FSKVCache state audit (experiments/scaling/kv_cache.py); clone_cache enforces coverage.
_CLONE_TENSORS = ("_k_buf", "_v_buf", "_kbar", "_ceil", "_cell_k", "_spread")
_CLONE_SCALARS = ("s", "W", "pos", "lean_prefill", "reserve", "_len", "fsb_leaf_radius")
_DERIVED_VIEWS = ("_k_hist", "_v_hist")
_SHARED = ("attn",)
_SINGLE_USE = ("_prealloc",)
_DROPPED = ("_lr_decode_pool", "_fsb_tree", "_fsb_forest", "_fsb_train_graphs")


def clone_cache(cache):
    # Fork one layer's FSKVCache: clone every tensor, re-derive views, share the module.
    # * _k_buf/_v_buf: raw K/V history (B,H,cap,D); decode appends IN PLACE (_grow writes
    #   _k_buf[:, :, _len]), so they are cloned, never shared.
    # * _k_hist/_v_hist: views into _k_buf/_v_buf[:, :, :_len]; re-derived on the clone's own
    #   buffers (cloning the view would orphan it from future appends).
    # * _kbar/_ceil/_cell_k/_spread: cell summaries (_spread exists only with FS-B leaf
    #   radius); reassigned via torch.cat, but cloned anyway so the fork owns its storage.
    # * _prealloc: single-use prefill preallocation; always consumed (None) at fork time.
    # * _lr_decode_pool: decode candidate-pool stash keyed by block start; each fork rebuilds
    #   its own, mirroring the fresh-cache-per-generation contract.
    # * _fsb_tree/_fsb_forest/_fsb_train_graphs: FS-B tree/forest/decode-graph state, lazily
    #   attached and bound to the original buffers; None or absent on the FS/FS-L fork path
    #   and never cloned.
    # * attn: module reference, shared read-only.
    assert type(cache) is FSKVCache, f"clone_cache audits FSKVCache only, got {type(cache)}"
    known = (set(_CLONE_TENSORS) | set(_CLONE_SCALARS) | set(_DERIVED_VIEWS)
             | set(_SHARED) | set(_SINGLE_USE) | set(_DROPPED))
    extra = sorted(set(vars(cache)) - known)
    assert not extra, f"FSKVCache grew unaudited state {extra}; update clone_cache"
    fork = object.__new__(FSKVCache)
    fork.attn = cache.attn
    for name in _CLONE_SCALARS:
        setattr(fork, name, getattr(cache, name))
    for name in _CLONE_TENSORS:
        t = getattr(cache, name)
        setattr(fork, name, None if t is None else t.clone())
    fork._k_hist = None if cache._k_hist is None else fork._k_buf[:, :, :fork._len]
    fork._v_hist = None if cache._v_hist is None else fork._v_buf[:, :, :fork._len]
    for name in _SINGLE_USE:
        assert getattr(cache, name) is None, f"clone_cache with a pending {name}"
        setattr(fork, name, None)
    if fork._k_buf is not None:
        assert fork._k_buf.data_ptr() != cache._k_buf.data_ptr()
    return fork


def caches_equal(x, y):
    # bit-equality of two layer caches' prefix state (positions + every stored tensor)
    if (x.pos, x._len) != (y.pos, y._len):
        return False
    for name in _CLONE_TENSORS:
        tx, ty = getattr(x, name), getattr(y, name)
        if (tx is None) != (ty is None):
            return False
        if tx is not None and not torch.equal(tx, ty):
            return False
    return True


def prefill(model, caches, prefix_ids):
    # one cached prefill; only the populated cache state is kept
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = _trunk_cached(model, prefix_ids, caches)
    assert torch.isfinite(hidden[:, -1]).all()
    del hidden


def answer(model, fork, q_ids):
    # question tokens into the forked cache (per-token decode append), then DECODE_TOKENS
    # greedy tokens through the fp32 lm-head; no eos stop; returns the generated ids
    step = q_ids
    generated = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(DECODE_TOKENS):
            hidden = _trunk_cached(model, step, fork)
            logits = _lci._lm_head_fp32(model, hidden[:, -1:, :], LM_CHUNK)[:, 0, :]   # (1,V) fp32
            nxt = int(logits.argmax())
            generated.append(nxt)
            step = torch.tensor([[nxt]], device=q_ids.device)
    return generated


def selfcheck(model, caches, qids):
    # fork isolation: answer question 0 on fork 1, re-fork from the original, answer question 1
    # on fork 2; the original prefix cache must stay bit-identical (layer 0)
    assert len(qids) >= 2, "selfcheck needs a prefix with >=2 questions"
    ref = clone_cache(caches[0])
    fork1 = [clone_cache(c) for c in caches]
    answer(model, fork1, qids[0])
    assert caches_equal(caches[0], ref), "fork 1 decode MUTATED the original prefix cache"
    del fork1
    fork2 = [clone_cache(c) for c in caches]
    assert caches_equal(fork2[0], ref), "re-fork after question 1 lost prefill-state equality"
    answer(model, fork2, qids[1])
    assert caches_equal(caches[0], ref), "fork 2 decode MUTATED the original prefix cache"
    del fork2, ref
    print("SELFCHECK_OK fork isolation verified (layer 0, 2 questions)", flush=True)


def build_attn_kwargs(cell_topk=4, window_size=128, use_rope=False, rope_long_range=None,
                      fs_uniform_sr=False, fsb=None):
    kw = dict(window_size=window_size, long_range_variant="cell_topk", cell_stride=32, cell_topk=cell_topk,
              cell_select="score", sel_strict_causal=1, use_rope=use_rope)
    if fs_uniform_sr:
        kw["fs_uniform_sr"] = True
    if rope_long_range is not None:
        kw["rope_long_range"] = rope_long_range
    if fsb:
        kw.update(fsb)
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["FS-B", "FS-L", "FS"])
    ap.add_argument("--protocol", required=True, choices=["capeval340", "rebuttal800"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, choices=["340m", "800m"])
    ap.add_argument("--data", required=True, help="gen_multifact_data .jsonl (prefix, facts, questions)")
    ap.add_argument("--output", required=True, help=".jsonl (manifest, one row per question, summary)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--selfcheck", action="store_true",
                    help="fork-isolation self-test on the first prefix (capeval340, FS/FS-L)")
    # FS-B selector flags; defaults are the library defaults, the harness passes its values
    ap.add_argument("--window_size", type=int, default=512)
    ap.add_argument("--cell_topk", type=int, default=4)
    ap.add_argument("--use_rope", action="store_true")
    ap.add_argument("--fs_uniform_sr", action="store_true")
    ap.add_argument("--fs_nope_longrange", action="store_true")
    ap.add_argument("--fsb", action="store_true")
    ap.add_argument("--fsb_policy", default="learned", choices=["fixed", "learned"])
    ap.add_argument("--fsb_width", type=int, default=4)
    ap.add_argument("--fsb_delta", type=int, default=0)
    ap.add_argument("--fsb_widths", default="1,2,4")
    ap.add_argument("--fsb_deltas", default="0,1,2")
    ap.add_argument("--fsb_node_score", default="bound", choices=["bound", "centroid"])
    ap.add_argument("--fsb_bound_radius_coef", type=float, default=1.0)
    ap.add_argument("--fsb_leaf_radius", type=int, default=0)
    ap.add_argument("--fsb_tree", default="forest", choices=["rebuild", "forest"])
    ap.add_argument("--fsb_read_top", type=int, default=0)
    ap.add_argument("--fsb_rerank_score", default="leaf", choices=["leaf", "lse", "lse_pos", "max"])
    ap.add_argument("--fsb_rerank_query", default="route", choices=["route", "raw"])
    ap.add_argument("--fsb_union_width", type=int, default=0)
    ap.add_argument("--fsb_union_tau_quantile", type=float, default=0.0)
    ap.add_argument("--fsb_route_adapter", default="none", choices=["none"])
    ap.add_argument("--fsb_route_alpha", type=float, default=1.0)
    ap.add_argument("--fsb_tau_override", type=float, default=None)
    ap.add_argument("--fsb_tau_calib_batches", type=int, default=8)
    ap.add_argument("--fsb_tau_calib_len", type=int, default=4096)
    ap.add_argument("--fsb_tau_calib_seed", type=int, default=1234)
    ap.add_argument("--fsb_tau_calib_data", default=None,
                    help="pretokenized uint16 corpus for --fsb_union_tau_quantile calibration")
    ap.add_argument("--fsb_tau_cache_dir", default=None)
    a = ap.parse_args()

    if (a.arch == "FS-B") != a.fsb:
        ap.error("--arch FS-B needs --fsb, and --fsb needs --arch FS-B")
    fork_mode = a.protocol == "capeval340" and a.arch != "FS-B"
    if a.selfcheck and not fork_mode:
        ap.error("--selfcheck exercises the prefix-cache fork (capeval340, FS/FS-L only)")
    want_prune = a.arch == "FS-L"
    if int(os.environ.get("FS_PRUNE_KERNEL", "0")) != int(want_prune):
        ap.error(f"--arch {a.arch} needs FS_PRUNE_KERNEL={int(want_prune)}")
    if want_prune and (os.environ.get("FS_PRUNE_POOL") != "64" or os.environ.get("FS_PRUNE_WINDOW") != "8"):
        ap.error("--arch FS-L needs FS_PRUNE_POOL=64 FS_PRUNE_WINDOW=8")
    if not want_prune and os.environ.get("FS_PRUNE_POOL", "0") not in ("", "0"):
        ap.error(f"--arch {a.arch} must not set FS_PRUNE_POOL")
    if os.environ.get("FS_STRICT_CAUSAL") != "1":
        ap.error("FS_STRICT_CAUSAL=1 is required (as in every paper run)")
    if os.environ.get("FSL_LINEAR_CACHE", "0") != "0":
        ap.error("FSL_LINEAR_CACHE must be unset/0 (masked implementation)")
    if a.fsb_union_tau_quantile > 0 and (a.fsb_union_width <= 0 or a.fsb_read_top <= 0):
        ap.error("--fsb_union_tau_quantile needs --fsb_union_width > 0 and --fsb_read_top > 0")
    if a.fsb_union_tau_quantile > 0 and not a.fsb_tau_calib_data:
        ap.error("--fsb_union_tau_quantile needs --fsb_tau_calib_data")

    if a.fsb:
        os.environ.setdefault("FSB_TRAIN_GRAPH", "0")
        os.environ.setdefault("FSB_DECODE_GRAPH", "1")
    torch.set_grad_enabled(False)
    _lci.MODEL_CONFIGS[a.config]["vocab_size"] = 32000
    fsb_kw = (dict(fsb_enabled=True, fsb_policy=a.fsb_policy, fsb_width=a.fsb_width, fsb_delta=a.fsb_delta,
                   fsb_widths=a.fsb_widths, fsb_deltas=a.fsb_deltas, fsb_node_score=a.fsb_node_score,
                   fsb_tree=a.fsb_tree, fsb_bound_radius_coef=a.fsb_bound_radius_coef, fsb_leaf_radius=a.fsb_leaf_radius,
                   **({"fsb_read_top": a.fsb_read_top} if a.fsb_read_top > 0 else {}),
                   **({"fsb_rerank_score": a.fsb_rerank_score} if a.fsb_rerank_score != "leaf" else {}),
                   **({"fsb_rerank_query": a.fsb_rerank_query} if a.fsb_rerank_query != "route" else {}),
                   **({"fsb_union_width": a.fsb_union_width} if a.fsb_union_width > 0 else {}),
                   **({"fsb_union_tau_quantile": a.fsb_union_tau_quantile} if a.fsb_union_tau_quantile > 0 else {}))
              if a.fsb else None)
    attn_kw = build_attn_kwargs(a.cell_topk, window_size=a.window_size, use_rope=a.use_rope,
                                rope_long_range=(False if a.fs_nope_longrange else None),
                                fs_uniform_sr=a.fs_uniform_sr, fsb=fsb_kw)
    tok = AutoTokenizer.from_pretrained(TOK_NAME, trust_remote_code=True)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 2
    model = load_model(a.ckpt, config=a.config, attn_type="fsattn", device="cuda",
                       max_seq_len=a.max_seq_len, **attn_kw)

    # load_model is non-strict: prove that only FS-B's threshold buffers were missing and nothing was dropped
    sd = _lci._clean_keys(_lci._unwrap_state_dict(_lci._load_raw_state_dict(a.ckpt, "cpu")))
    model_keys = set(model.state_dict().keys())
    missing = sorted(model_keys - set(sd))
    unexpected = sorted(set(sd) - model_keys)
    del sd
    bad_missing = [k for k in missing if not k.endswith((".fsb_tau", ".fsb_tau_union"))]
    tied = model.lm_head.weight is model.tok_emb.weight
    if tied:
        bad_missing = [k for k in bad_missing if k != "lm_head.weight"]
    assert not bad_missing and not unexpected, (bad_missing[:8], unexpected[:8])

    ckpt_file = os.path.join(a.ckpt, "model.safetensors")
    ckpt_sha = _file_sha256_once(ckpt_file)
    utau_info = None
    if a.fsb:
        from src.fsattn.fsb_route import router_key_check
        router_key_check(model, a.ckpt, None)
        if a.fsb_tau_override is not None:
            n_t = 0
            for m in model.modules():
                if getattr(m, "fsb_enabled", False):
                    m.fsb_tau.fill_(float(a.fsb_tau_override)); n_t += 1
            print(f"[multifact] fsb_tau overridden to {a.fsb_tau_override} in {n_t} layers", flush=True)
        if a.fsb_union_tau_quantile > 0:
            import fsb_tau_calib
            cdir = a.fsb_tau_cache_dir or os.environ.get("FSB_TAU_CACHE_DIR") or os.path.dirname(os.path.abspath(a.output))
            if ckpt_sha is None:
                ap.error(f"--fsb_union_tau_quantile: {ckpt_file} not found (the tau cache is keyed by its sha256)")
            utau_info = fsb_tau_calib.calibrate_or_load(
                model, ckpt_sha=ckpt_sha, quantile=a.fsb_union_tau_quantile, n_batches=a.fsb_tau_calib_batches,
                length=a.fsb_tau_calib_len, seed=a.fsb_tau_calib_seed, data_path=a.fsb_tau_calib_data, cache_dir=cdir,
                device="cuda", autocast_dtype=torch.bfloat16, target="fsb_tau_union")
            _, uall = fsb_tau_calib.tau_summary(model, target="fsb_tau_union")
            _, mall = fsb_tau_calib.tau_summary(model)
            print(f"[multifact] tau_union {'cached' if utau_info['loaded_from_cache'] else 'computed'}: min {uall[0]:.4f} "
                  f"mean {uall[1]:.4f} max {uall[2]:.4f}; main tau min {mall[0]:.4f} max {mall[2]:.4f}", flush=True)
        from src.fsattn.fsb_tree import fsb_read_stats_reset, fsb_read_stats

    records = [json.loads(x) for x in open(a.data)]
    if a.limit:
        records = records[:a.limit]
    eos_stop = a.protocol == "rebuttal800"
    src_files = [os.path.abspath(__file__)] + [os.path.join(_HERE, p) for p in (
        "experiments/scaling/kv_cache.py", "experiments/scaling/longctx_infer.py",
        "experiments/scaling/model.py", "experiments/scaling/fsb_tau_calib.py",
        "src/fsattn/attention.py", "src/fsattn/fsb_tree.py", "src/fsattn/fsb_route.py")]
    out = open(a.output, "x")

    def emit(row):
        out.write(json.dumps(row) + "\n"); out.flush()

    emit(dict(kind="manifest", args=vars(a), argv=sys.argv, attn_kwargs=attn_kw,
              env={k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIXES)},
              torch=torch.__version__, cuda=torch.version.cuda, hardware=torch.cuda.get_device_name(),
              tokenizer=TOK_NAME, eos=eos, eos_stop=eos_stop, fork=fork_mode,
              ckpt_sha256=ckpt_sha, load_missing=missing, lm_head_tied=tied,
              data_sha256=_file_sha256_once(a.data),
              source_sha256={os.path.relpath(p, _HERE): _file_sha256_once(p) for p in src_files},
              union_tau_calib=utau_info,
              params=sum(p.numel() for p in model.parameters())))

    n_q = n_c = n_wrong_other = 0
    prefix_acc = []
    t0 = time.time()
    checked = False
    for pi, rec in enumerate(records):
        qs = rec["questions"]
        assert len({q["fact_idx"] for q in qs}) == len(qs)
        values = [f["value"] for f in rec["facts"]]
        hits = []
        if fork_mode:
            # one prefill per prefix; each question is answered from a deep-copy fork of the
            # end-of-prefix cache (16 greedy tokens, no eos stop)
            prefix_list = tok(rec["prefix"])["input_ids"]
            q_lists = [tok("\n\n" + q["q"], add_special_tokens=False)["input_ids"] for q in qs]
            prefix_ids = torch.tensor(prefix_list, dtype=torch.long, device="cuda")[None]
            qids = [torch.tensor(ql, dtype=torch.long, device="cuda")[None] for ql in q_lists]
            n = prefix_ids.shape[1]
            # pre-reserve decode slack so a fork's first append never doubles the K/V buffers
            reserve = max(t.shape[1] for t in qids) + DECODE_TOKENS + 8
            assert n + reserve <= a.max_seq_len, (n, reserve, a.max_seq_len)
            caches = make_caches(model, reserve=reserve)
            prefill(model, caches, prefix_ids)
            if a.selfcheck and not checked:
                selfcheck(model, caches, qids)
                emit(dict(kind="selfcheck", prefix_index=pi, layers=len(caches), passed=True))
                checked = True
            for qi, q in enumerate(qs):
                ids_list = prefix_list + q_lists[qi]
                te = time.perf_counter()
                fork = [clone_cache(c) for c in caches]
                gen_ids = answer(model, fork, qids[qi])
                del fork
                pred = tok.decode(gen_ids)
                correct = q["answer"].lower() in pred.lower()
                others = [v for v in values if v != q["answer"]]
                copies_other = any(v.lower() in pred.lower() for v in others)
                emit(dict(kind="example", prefix_index=pi, question_index=qi, fact_idx=q["fact_idx"],
                          gold=q["answer"], pred=pred, generated_ids=gen_ids, correct=correct,
                          copies_other_value=copies_other, prompt_tokens=len(ids_list),
                          prompt_tokens_sha256=hashlib.sha256(
                              json.dumps(ids_list, separators=(",", ":")).encode()).hexdigest(),
                          seconds=round(time.perf_counter() - te, 3)))
                hits.append(int(correct))
                n_q += 1; n_c += int(correct); n_wrong_other += int((not correct) and copies_other)
            prefix_acc.append(sum(hits) / len(hits))
            print(f"PROGRESS {a.arch} {pi} {sum(hits)}/{len(hits)} cum {n_c}/{n_q} {time.time() - t0:.0f}s", flush=True)
            del caches, prefix_ids, qids
            torch.cuda.empty_cache()
            continue
        if a.protocol == "capeval340":
            prefix_ids = tok(rec["prefix"])["input_ids"]
        else:
            prefix_ids = tok.encode(rec["prefix"])
        for qi, q in enumerate(qs):
            if a.protocol == "capeval340":
                q_ids = tok("\n\n" + q["q"], add_special_tokens=False)["input_ids"]
            else:
                q_ids = tok.encode(q["q"], add_special_tokens=False)
                assert len(prefix_ids) + len(q_ids) + 16 <= 4096, f"strict length gate failed at prefix {pi}"
            ids_list = prefix_ids + q_ids
            assert len(ids_list) + a.max_new <= a.max_seq_len
            ids = torch.tensor(ids_list, dtype=torch.long)
            P = ids.numel()
            if a.fsb:
                fsb_read_stats_reset(model)
            te = time.perf_counter()
            gen = cached_generate(model, ids, a.max_new, device="cuda", dtype="bfloat16",
                                  eos_token_id=(eos if eos_stop else None), lean_prefill=False)
            gen_ids = gen[0, P:].tolist()
            del gen
            pred = tok.decode(gen_ids, skip_special_tokens=True) if eos_stop else tok.decode(gen_ids)
            correct = q["answer"].lower() in pred.lower()
            others = [v for v in values if v != q["answer"]]
            copies_other = any(v.lower() in pred.lower() for v in others)
            row = dict(kind="example", prefix_index=pi, question_index=qi, fact_idx=q["fact_idx"],
                       gold=q["answer"], pred=pred, generated_ids=gen_ids, correct=correct,
                       copies_other_value=copies_other, prompt_tokens=P,
                       prompt_tokens_sha256=hashlib.sha256(
                           json.dumps(ids_list, separators=(",", ":")).encode()).hexdigest(),
                       seconds=round(time.perf_counter() - te, 3))
            if a.fsb:
                rs = fsb_read_stats(model)
                if rs is not None:
                    row.update(read_cells_mean=round(rs["mean"], 4), read_cells_max=rs["max"],
                               utau_read_frac=round(rs["utau_frac"], 4))
            emit(row)
            hits.append(int(correct))
            n_q += 1; n_c += int(correct); n_wrong_other += int((not correct) and copies_other)
            del ids
            torch.cuda.empty_cache()
        prefix_acc.append(sum(hits) / len(hits))
        print(f"PROGRESS {a.arch} {pi} {sum(hits)}/{len(hits)} cum {n_c}/{n_q} {time.time() - t0:.0f}s", flush=True)
    summary = dict(kind="summary", status="ok", n_prefixes=len(records), n_questions=n_q, n_correct=n_c,
                   accuracy=100.0 * n_c / n_q, prefix_mean_accuracy=100.0 * sum(prefix_acc) / len(prefix_acc),
                   n_wrong=n_q - n_c, n_wrong_with_other_value=n_wrong_other, seconds=round(time.time() - t0, 1),
                   peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    emit(summary)
    out.close()
    print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
