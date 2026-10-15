import os


def build_attn_kwargs(a):
    attn_type = a.attn_type
    attn_kwargs = {}
    if attn_type == "fsattn":
        attn_kwargs["window_size"] = a.window_size
        attn_kwargs["long_range_variant"] = a.long_range_variant
        attn_kwargs["sigma_init"] = a.sigma_init
        attn_kwargs["cell_stride"] = getattr(a, "cell_stride", 64)
        attn_kwargs["cell_topk"] = getattr(a, "cell_topk", 1)
        attn_kwargs["cell_select"] = getattr(a, "cell_select", "score")
        for _pk in ("cell_prune_kernel", "cell_prune_pool", "cell_prune_window", "cell_prune_blkchunk",
                    "sel_strict_causal"):
            _pv = getattr(a, _pk, None)
            if _pv is not None:
                attn_kwargs[_pk] = _pv
        if getattr(a, "disable_short_range", False):
            attn_kwargs["disable_short_range"] = True
        if getattr(a, "disable_long_range", False):
            attn_kwargs["disable_long_range"] = True
        attn_kwargs["rope_long_range"] = not getattr(a, "fs_nope_longrange", False)
        attn_kwargs["fs_uniform_sr"] = bool(getattr(a, "fs_uniform_sr", False))
        attn_kwargs["sel_content_only"] = bool(getattr(a, "sel_content_only", False))
        # FS-B: threaded only under --fsb; every knob left at None is not threaded, so the
        # library defaults hold and every existing run resolves the same attn_kwargs.
        if getattr(a, "fsb", False):
            attn_kwargs["fsb_enabled"] = True
            for _fk in ("fsb_widths", "fsb_deltas", "fsb_policy", "fsb_width", "fsb_delta",
                        "fsb_node_score", "fsb_route_cost", "fsb_policy_temp", "fsb_readout_blkchunk",
                        "fsb_tree", "fsb_prefill_blkchunk", "fsb_rl", "fsb_ppo_store_dtype",
                        "fsb_bound_radius_coef", "fsb_leaf_radius",
                        "fsb_route_adapter", "fsb_route_rank", "fsb_route_alpha", "fsb_rd_weight",
                        "fsb_rd_rows", "fsb_rd_loss", "fsb_rd_far", "fsb_rd_pos", "fsb_rd_temp",
                        "fsb_rd_topn", "fsb_rd_node", "fsb_rd_node_weight", "fsb_route_learn_c",
                        "fsb_read_top", "fsb_rerank_score", "fsb_rerank_query", "fsb_union_width",
                        "fsb_union_tau_quantile"):
                _fv = getattr(a, _fk, None)
                if _fv is not None:
                    attn_kwargs[_fk] = _fv
            for _fk in ("fsb_hybrid", "fsb_pool_size", "fsb_pool_recent", "fsb_pool_short",
                        "fsb_pool_quota"):
                _fv = getattr(a, _fk, None)
                if _fv is not None:
                    attn_kwargs[_fk] = _fv
            if getattr(a, "fsb_rerank_rr", None) is not None:
                attn_kwargs["fsb_rerank_rr"] = a.fsb_rerank_rr
        if getattr(a, "fs_nope", False):
            attn_kwargs["use_rope"] = False
    if getattr(a, "no_rope", False):
        attn_kwargs["use_rope"] = False
    return attn_type, attn_kwargs


_CONFIG_TO_ARG = [
    ("model.config", "config"),
    ("attn.attn_type", "attn_type"),
    ("attn.long_range_variant", "long_range_variant"),
    ("attn.window_size", "window_size"),
    ("attn.sigma_init", "sigma_init"),
    ("attn.cell_stride", "cell_stride"),
    ("attn.cell_topk", "cell_topk"),
    ("attn.cell_select", "cell_select"),
    ("attn.sel_strict_causal", "sel_strict_causal"),
    ("attn.fs_uniform_sr", "fs_uniform_sr"),
    ("attn.fs_nope_longrange", "fs_nope_longrange"),
    ("training.per_gpu_batch_size", "per_gpu_batch_size"),
    ("training.grad_accum_steps", "grad_accum_steps"),
    ("training.lr", "lr"),
    ("training.min_lr", "min_lr"),
    ("training.weight_decay", "weight_decay"),
    ("training.max_steps", "max_steps"),
    ("training.warmup_steps", "warmup_steps"),
    ("training.grad_clip", "grad_clip"),
    ("training.eval_every", "eval_every"),
    ("training.log_every", "log_every"),
    ("training.save_every", "save_every"),
    ("training.seed", "seed"),
    ("data.pretokenized", "pretokenized"),
    ("data.data_subset", "data_subset"),
    ("data.num_workers", "num_workers"),
    ("runtime.compile", "compile"),
    ("runtime.no_long_range_recompute", "no_long_range_recompute"),
    ("runtime.gradient_checkpointing", "gradient_checkpointing"),
    ("output.output_dir", "output_dir"),
    ("output.run_name", "run_name"),
    ("output.wandb_project", "wandb_project"),
    ("output.mlflow_uri", "mlflow_uri"),
    ("output.resume", "resume"),
]


def _scaling_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _resolve(path):
    candidates = [path,
                  os.path.join(_scaling_dir(), path),
                  os.path.join(os.getcwd(), path)]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return path


def is_config_mode(config_value):
    return isinstance(config_value, str) and config_value.endswith((".yaml", ".yml"))


def load_merged_config(variant_path, overrides=None):
    from omegaconf import OmegaConf
    base = OmegaConf.load(os.path.join(_scaling_dir(), "configs", "base.yaml"))
    variant = OmegaConf.load(_resolve(variant_path))
    conf = OmegaConf.merge(base, variant)
    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(list(overrides)))
    return conf


def apply_config(args, extras=None):
    if not is_config_mode(getattr(args, "config", None)):
        return None
    from omegaconf import OmegaConf
    conf = load_merged_config(args.config, extras)
    missing = object()
    for dotpath, attr in _CONFIG_TO_ARG:
        val = OmegaConf.select(conf, dotpath, default=missing)
        if val is missing:
            continue
        setattr(args, attr, val)
    use_rope = OmegaConf.select(conf, "attn.use_rope", default=None)
    if use_rope is not None:
        args.no_rope = not bool(use_rope)
        args.fs_nope = False
    return conf
