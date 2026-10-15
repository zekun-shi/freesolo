from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.utils.checkpoint

NEG = -1e9

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False


if _HAVE_TRITON:
    @triton.jit
    def _head_rowmat_kernel(X, Wt, OUT, n_per_head, H, D, E, DP: tl.constexpr, EP: tl.constexpr):
        r = tl.program_id(0)
        h = (r // n_per_head) % H
        d = tl.arange(0, DP)
        e = tl.arange(0, EP)
        x = tl.load(X + r * D + d, mask=d < D, other=0.0)
        w = tl.load(Wt + h * D * E + d[:, None] * E + e[None, :], mask=(d[:, None] < D) & (e[None, :] < E), other=0.0)
        p = x[:, None] * w                                               # (DP, EP)
        # pairwise halving over the input dim: p[:h] + p[h:] (a size-2 reduction = one IEEE add per element)
        if DP >= 256:
            p = tl.sum(tl.reshape(p, (2, 128, EP)), axis=0)
        if DP >= 128:
            p = tl.sum(tl.reshape(p, (2, 64, EP)), axis=0)
        if DP >= 64:
            p = tl.sum(tl.reshape(p, (2, 32, EP)), axis=0)
        if DP >= 32:
            p = tl.sum(tl.reshape(p, (2, 16, EP)), axis=0)
        if DP >= 16:
            p = tl.sum(tl.reshape(p, (2, 8, EP)), axis=0)
        if DP >= 8:
            p = tl.sum(tl.reshape(p, (2, 4, EP)), axis=0)
        if DP >= 4:
            p = tl.sum(tl.reshape(p, (2, 2, EP)), axis=0)
        if DP >= 2:
            p = tl.sum(tl.reshape(p, (2, 1, EP)), axis=0)
        y = tl.sum(p, axis=0)                                            # one row left: identity
        tl.store(OUT + r * E + e, y, mask=e < E)


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def det_head_matmul(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # out[b,h,i,e] = sum_d x[b,h,i,d] W[h,d,e]: fixed zero-padded pairwise-halving sum over d, bit-identical
    # for a row computed in a batch or alone (Triton on CUDA for D, E <= 256, Python halving otherwise)
    B, H, n, D = x.shape
    E = W.shape[-1]
    x = x.float().contiguous()
    W = W.float().contiguous()
    if x.numel() == 0:
        return x.new_zeros(B, H, n, E)
    if _HAVE_TRITON and x.is_cuda and D <= 256 and E <= 256:
        out = torch.empty(B, H, n, E, dtype=torch.float32, device=x.device)
        _head_rowmat_kernel[(B * H * n,)](x, W, out, n, H, D, E, DP=_pow2(D), EP=_pow2(E), num_warps=4)
        return out
    from .det_reduce import _py_halving
    prod = x[..., :, None] * W[None, :, None, :, :]                      # (B,H,n,D,E)
    return _py_halving(prod.movedim(-2, -1))                             # halving over D


def route_init(attn, n_heads: int, head_dim: int, kwargs: dict):
    # attach the routing adapter / distillation config (called from fsb_init); adapter none creates no parameter
    mode = str(kwargs.get("fsb_route_adapter", "none") or "none")
    assert mode in ("none", "lowrank", "full"), mode
    attn.fsb_route_adapter = mode
    attn.fsb_route_rank = int(kwargs.get("fsb_route_rank", 32))
    attn.fsb_route_alpha = float(kwargs.get("fsb_route_alpha", 1.0))
    attn.fsb_rd_weight = float(kwargs.get("fsb_rd_weight", 0.0))
    attn.fsb_rd_rows = int(kwargs.get("fsb_rd_rows", 16))
    attn.fsb_rd_loss = str(kwargs.get("fsb_rd_loss", "kl"))
    assert attn.fsb_rd_loss in ("kl", "topk"), attn.fsb_rd_loss
    attn.fsb_rd_far = bool(int(kwargs.get("fsb_rd_far", 0)))
    attn.fsb_rd_pos = bool(int(kwargs.get("fsb_rd_pos", 0)))
    attn.fsb_rd_temp = float(kwargs.get("fsb_rd_temp", 1.0))
    attn.fsb_rd_topn = int(kwargs.get("fsb_rd_topn", 16))
    assert attn.fsb_rd_topn >= 4, "fsb_rd_topn must be >= 4 (the recall metric and the topk loss use the teacher top-4)"
    attn.fsb_rd_chunk = int(kwargs.get("fsb_rd_chunk", os.environ.get("FSB_RD_CHUNK", "256")))
    attn.fsb_rd_node = int(kwargs.get("fsb_rd_node", 0) or 0)
    _nw = kwargs.get("fsb_rd_node_weight", None)
    attn.fsb_rd_node_weight = 1.0 if _nw is None else float(_nw)
    attn.fsb_rd_node_blkchunk = int(kwargs.get("fsb_rd_node_blkchunk", os.environ.get("FSB_RD_NODE_BLKCHUNK", "16")))
    attn.fsb_route_learn_c = bool(int(kwargs.get("fsb_route_learn_c", 0) or 0))
    attn._fsb_rd = None
    attn._fsb_rd_gen = None                      # optional torch.Generator for the row sampling
    if attn.fsb_rd_weight > 0 and mode == "none":
        raise ValueError("fsb_rd_weight > 0 needs a routing adapter (fsb_route_adapter lowrank|full)")
    if attn.fsb_rd_node not in (0, 1):
        raise ValueError(f"fsb_rd_node must be 0 or 1, got {attn.fsb_rd_node}")
    if attn.fsb_rd_node:
        if not attn.fsb_rd_weight > 0:
            raise ValueError("fsb_rd_node 1 needs fsb_rd_weight > 0 (the node term is part of the distillation loss)")
        if getattr(attn, "fsb_tree", "rebuild") != "forest":
            raise ValueError("fsb_rd_node 1 needs fsb_tree forest (the node loss walks the streaming forest)")
        if getattr(attn, "fsb_policy_mode", "fixed") != "fixed" or getattr(attn, "fsb_delta", 0) != 0:
            raise ValueError("fsb_rd_node 1 needs the fixed policy with fsb_delta 0 (every internal node is expanded; the "
                             "deployment W/D0 beam)")
    if attn.fsb_route_learn_c:
        if mode == "none":
            raise ValueError("fsb_route_learn_c 1 needs a routing adapter (the coefficient is a router parameter)")
        if getattr(attn, "fsb_node_score", "bound") != "bound" or not getattr(attn, "fsb_bound_radius_coef", 1.0) > 0:
            raise ValueError("fsb_route_learn_c 1 needs fsb_node_score bound and fsb_bound_radius_coef > 0")
    if attn.fsb_rd_weight > 0 and getattr(attn, "fsb_leaf_radius", False):
        raise ValueError("fsb_rd_weight > 0 is not supported with fsb_leaf_radius 1 (the student omits the leaf "
                         "radius term of the beam's leaf score)")
    rr_init(attn, n_heads, kwargs)                                      # residual re-ranker: independent of the adapter
    if mode == "none":
        return
    H, Dh = n_heads, head_dim
    # own generator: the adapter init does not consume the global RNG
    g = torch.Generator().manual_seed(int(kwargs.get("fsb_route_seed", 20260928)))
    if mode == "lowrank":
        r = attn.fsb_route_rank
        attn.fsb_route_A = nn.Parameter(torch.randn(H, Dh, r, generator=g) / math.sqrt(Dh))
        attn.fsb_route_B = nn.Parameter(torch.zeros(H, r, Dh))
    else:
        attn.fsb_route_W = nn.Parameter(torch.zeros(H, Dh, Dh))
    attn.fsb_route_logtemp = nn.Parameter(torch.zeros(H))            # loss-only student temperature (log)
    if attn.fsb_route_learn_c:
        c0 = float(attn.fsb_bound_radius_coef)
        th0 = float(torch.tensor(math.log(math.expm1(c0)), dtype=torch.float32))   # softplus^-1(c0), fp32
        attn._fsb_ctheta0 = th0
        attn._fsb_c_s0 = -math.expm1(-c0)                               # sigmoid(softplus^-1(c0)) = 1 - exp(-c0)
        attn.fsb_route_ctheta = nn.Parameter(torch.full((H,), th0))     # c_h = softplus(theta_h), == c0 at init
    if attn.fsb_rd_node:
        attn.fsb_route_nlogtemp = nn.Parameter(torch.zeros(H))       # loss-only node-level student temperature (log)


def rr_init(attn, n_heads: int, kwargs: dict):
    # residual re-ranker of the overselect's final read (fsb_rerank_rr): zero-init weights meant for the router
    # machinery (named fsb_route_*), created without a routing adapter
    from .fsb_rerank import RR_MODES, RR_NF
    mode = str(kwargs.get("fsb_rerank_rr", "none") or "none")
    if mode not in RR_MODES:
        raise ValueError(f"FS-B: fsb_rerank_rr = {mode!r} ({'|'.join(RR_MODES)})")
    attn.fsb_rerank_rr = mode
    if mode == "none":
        return
    if getattr(attn, "fsb_read_top", 0) <= 0:
        raise ValueError("FS-B: fsb_rerank_rr re-ranks the overselect's candidates; it needs fsb_read_top > 0")
    if getattr(attn, "fsb_rerank_score", "leaf") != "lse_pos":
        raise ValueError(f"FS-B: fsb_rerank_rr is a residual on the lse_pos score; got fsb_rerank_score "
                         f"{getattr(attn, 'fsb_rerank_score', 'leaf')!r}")
    if getattr(attn, "fsb_route_adapter", "none") != "none" and getattr(attn, "fsb_rerank_query", "route") != "raw":
        raise ValueError("FS-B: fsb_rerank_rr with a routing adapter needs fsb_rerank_query raw (the re-ranker's "
                         "score and features use the raw query)")
    attn.fsb_route_rr_w = nn.Parameter(torch.zeros(n_heads, RR_NF))
    attn.fsb_route_rr_logtemp = nn.Parameter(torch.zeros(n_heads))


def rr_all_zero(model) -> bool:
    # True when the model has no reranker weights or every fsb_route_rr_w is exactly zero (the identity)
    ws = [p for n, p in model.named_parameters() if n.endswith("fsb_route_rr_w")]
    return (not ws) or all(float(p.detach().abs().max()) == 0.0 for p in ws)


def route_radius(attn):
    # learned radius coefficient c_h = softplus(theta_h), exactly fsb_bound_radius_coef at init; None when fixed
    if not getattr(attn, "fsb_route_learn_c", False):
        return None
    th = attn.fsb_route_ctheta
    with torch.autocast(device_type=th.device.type, enabled=False):
        u = th.float() - attn._fsb_ctheta0                                # exactly 0 at init
        return (attn.fsb_bound_radius_coef + torch.log1p(attn._fsb_c_s0 * torch.expm1(u))).clamp(min=0.0)


def radius_rows(attn, head_id, c_h=None):
    # radius coefficient for traversal rows with heads head_id: the fixed scalar, or learned c_h[head_id] detached
    if not getattr(attn, "fsb_route_learn_c", False):
        return attn.fsb_bound_radius_coef
    if c_h is None:
        with torch.no_grad():
            c_h = route_radius(attn)
    return c_h.detach()[head_id]


def rd_generator(model, device, seed, rank):
    # same rank-distinct generator (seed * 1000 + rank) for every distilling FS-B layer; None when none distills
    layers = [m for m in model.modules()
              if getattr(m, "fsb_enabled", False) and getattr(m, "fsb_rd_weight", 0.0) > 0]
    if not layers:
        return None
    gen = torch.Generator(device=device).manual_seed(int(seed) * 1000 + int(rank))
    for m in layers:
        m._fsb_rd_gen = gen
    return gen


def route_on(attn) -> bool:
    return getattr(attn, "fsb_route_adapter", "none") != "none" and getattr(attn, "fsb_route_alpha", 0.0) != 0.0


@torch.no_grad()
def fsb_route_q(attn, Qf: torch.Tensor) -> torch.Tensor:
    # routing query: Qf itself (same object) when the adapter is off, else q + alpha * adapter_h(q), fixed order
    if not route_on(attn):
        return Qf
    with torch.autocast(device_type=Qf.device.type, enabled=False):
        if attn.fsb_route_adapter == "full":
            d = det_head_matmul(Qf, attn.fsb_route_W.detach())
        else:
            d = det_head_matmul(det_head_matmul(Qf, attn.fsb_route_A.detach()), attn.fsb_route_B.detach())
        return Qf.float() + attn.fsb_route_alpha * d


def _route_q_grad(attn, q: torch.Tensor, alpha: float) -> torch.Tensor:
    # differentiable q_r (plain einsum) for the distillation student
    if attn.fsb_route_adapter == "full":
        d = torch.einsum("bhnd,hde->bhne", q, attn.fsb_route_W.float())
    else:
        d = torch.einsum("bhnr,hre->bhne", torch.einsum("bhnd,hdr->bhnr", q, attn.fsb_route_A.float()),
                         attn.fsb_route_B.float())
    return q + alpha * d


def adapter_head_norms(attn) -> torch.Tensor:
    # (H,) Frobenius norm of the effective per-head adapter matrix (lowrank: A_h B_h)
    with torch.no_grad():
        if attn.fsb_route_adapter == "full":
            M_ = attn.fsb_route_W.float()
        else:
            M_ = torch.bmm(attn.fsb_route_A.float(), attn.fsb_route_B.float())
        return M_.flatten(1).norm(dim=-1)


def route_param_names(model):
    # names of the routing-only parameters (last name component starts with "fsb_route_")
    return [n for n, _ in model.named_parameters() if n.rsplit(".", 1)[-1].startswith("fsb_route_")]


def load_router_params(model, path):
    # copy the fsb_route_* tensors of a router checkpoint dir into the model in place (strict shape check)
    from safetensors.torch import load_file
    f = os.path.join(path, "fsb_route.safetensors")
    if not os.path.exists(f):
        f = os.path.join(path, "model.safetensors")
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in load_file(f).items()}
    sd = {k: v for k, v in sd.items() if k.rsplit(".", 1)[-1].startswith("fsb_route_")}
    params = dict(model.named_parameters())
    names = route_param_names(model)
    bad = [k for k in sd if k not in params or tuple(params[k].shape) != tuple(sd[k].shape)]
    if not sd or bad:
        raise ValueError(f"init_router_from {f}: {len(sd)} router tensors, {len(bad)} absent from the model or of another "
                         f"shape: {bad[:4]}")
    with torch.no_grad():
        for k, v in sd.items():
            params[k].copy_(v.to(params[k].dtype))
    kept = sorted({n.rsplit(".", 1)[-1] for n in names if n not in sd})
    return (f"init_router_from {f}: loaded {len(sd)} router tensors ({sorted({k.rsplit('.', 1)[-1] for k in sd})}); "
            f"kept at init: {kept if kept else 'none'}")


def router_file_path(path):
    # router checkpoint path: a file, or a directory holding fsb_route.safetensors
    return os.path.join(path, "fsb_route.safetensors") if os.path.isdir(path) else path


def _router_keys(f):
    from safetensors import safe_open
    with safe_open(f, framework="pt") as f_:
        return {k.replace("_orig_mod.", "").replace("module.", "") for k in f_.keys()}


# identity of a trained re-ranker: which backbone and which selection system its features / labels come from
RR_IDENTITY_KEY = "fsb_rr_identity"                                     # safetensors metadata key of a trained re-ranker
_SHA_CACHE = {}


def ckpt_weights_sha256(ckpt):
    # sha256 of the checkpoint's weight file, cached per process by (real path, size, mtime); None when absent
    import hashlib
    if not ckpt:
        return None
    f = os.path.join(ckpt, "model.safetensors") if os.path.isdir(ckpt) else ckpt
    if not os.path.isfile(f):
        return None
    st = os.stat(f)
    key = (os.path.realpath(f), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def adapter_sha256(model):
    # sha256 of the non-reranker fsb_route_* tensors over (name, shape, fp32 bytes) in name order; "none" when none
    import hashlib
    ps = sorted((n, p) for n, p in model.named_parameters()
                if n.rsplit(".", 1)[-1].startswith("fsb_route_") and not n.rsplit(".", 1)[-1].startswith("fsb_route_rr_"))
    if not ps:
        return "none"
    h = hashlib.sha256()
    for n, p in ps:
        t = p.detach().float().contiguous().cpu()
        h.update(n.encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def rr_select_config(model) -> dict:
    # the selection config a reranker's features and labels are defined for, from the first FS-B layer
    from .fsb_rerank import RR_FEATURES
    a = next((m for m in model.modules() if getattr(m, "fsb_enabled", False)), None)
    if a is None:
        return {}
    adapter = getattr(a, "fsb_route_adapter", "none")
    rq = getattr(a, "fsb_rerank_query", "route")
    cfg = {"fsb_width": int(a.fsb_width), "fsb_union_width": int(getattr(a, "fsb_union_width", 0) or 0),
           "fsb_read_top": int(getattr(a, "fsb_read_top", 0) or 0), "fsb_rerank_score": getattr(a, "fsb_rerank_score", "leaf"),
           # without an adapter the route query IS the raw query (identical bits)
           "fsb_rerank_query": "raw" if adapter == "none" else rq, "fsb_route_adapter": adapter,
           "fsb_hybrid": getattr(a, "fsb_hybrid", "off"), "features": list(RR_FEATURES)}
    if cfg["fsb_hybrid"] != "off":
        for k in ("fsb_pool_size", "fsb_pool_recent", "fsb_pool_short", "fsb_pool_quota"):
            cfg[k] = int(getattr(a, k))
    if int(getattr(a, "fsb_beam_quota", 0) or 0) > 0:                    # only when on
        cfg["fsb_beam_quota"] = int(a.fsb_beam_quota)
    if float(getattr(a, "fsb_union_tau_quantile", 0.0) or 0.0) > 0:      # only when on
        cfg["fsb_union_tau_quantile"] = float(a.fsb_union_tau_quantile)
    if int(getattr(a, "fsb_union_quota", 0) or 0) > 0:                   # only when on
        cfg["fsb_union_quota"] = int(a.fsb_union_quota)
    if float(getattr(a, "fsb_union_prior_scale", 1.0)) != 1.0:           # only when on
        cfg["fsb_union_prior_scale"] = float(a.fsb_union_prior_scale)
    return cfg


def rr_identity(model, ckpt) -> dict:
    # {base_sha256 (weight file of the base checkpoint), base (path, informational), adapter_sha256, config}
    return {"base_sha256": ckpt_weights_sha256(ckpt), "base": (os.path.realpath(ckpt) if ckpt else None),
            "adapter_sha256": adapter_sha256(model), "config": rr_select_config(model)}


def identity_mismatch(ident: dict, model=None, ckpt=None, check_adapter=False) -> list:
    # ways a recorded reranker identity differs from the model and the base checkpoint; [] = consistent
    bad = []
    if model is not None and ident.get("config") is not None:
        cur = rr_select_config(model)
        diff = {k: (ident["config"].get(k), cur.get(k)) for k in sorted(set(ident["config"]) | set(cur))
                if ident["config"].get(k) != cur.get(k)}
        if diff:
            bad.append(f"selection config differs (recorded, this model): {diff}")
    if check_adapter and model is not None and ident.get("adapter_sha256") is not None:
        cur = adapter_sha256(model)
        if cur != ident["adapter_sha256"]:
            bad.append(f"routing adapter tensors differ (sha256 {str(ident['adapter_sha256'])[:12]} recorded, "
                       f"{cur[:12]} here)")
    if ckpt is not None and ident.get("base_sha256"):
        cur = ckpt_weights_sha256(ckpt)
        if cur != ident["base_sha256"]:
            bad.append(f"base checkpoint differs: recorded {ident.get('base')} (sha256 {ident['base_sha256'][:12]}), "
                       f"got {ckpt} (sha256 {str(cur)[:12]})")
    return bad


def router_file_identity(path):
    # reranker identity recorded in a router file's safetensors metadata (RR_IDENTITY_KEY), or None
    import json
    from safetensors import safe_open
    f = router_file_path(path)
    with safe_open(f, framework="pt") as f_:
        md = f_.metadata() or {}
    return json.loads(md[RR_IDENTITY_KEY]) if RR_IDENTITY_KEY in md else None


def overlay_router_file(model, path, ckpt=None, allow_mismatch=False):
    # strict eval-time router overlay: every model fsb_route_* param in the file, same shapes, no extra tensor
    from safetensors.torch import load_file
    f = router_file_path(path)
    if not os.path.isfile(f):
        raise ValueError(f"--fsb_router_file {path}: no router file ({f})")
    ident = router_file_identity(f)
    idrep = "no re-ranker identity in the file (base / config unchecked)"
    if ident is not None:
        bad = identity_mismatch(ident, model=model, ckpt=ckpt)
        if bad and not allow_mismatch:
            raise ValueError(f"--fsb_router_file {f}: trained for another system: " + "; ".join(bad)
                             + " (pass --fsb_router_allow_mismatch 1 to overlay it anyway)")
        idrep = (("IDENTITY MISMATCH ALLOWED: " + "; ".join(bad)) if bad else
                 f"identity ok (base sha256 {str(ident.get('base_sha256'))[:12]}{'' if ckpt else ' unchecked: no ckpt'}, "
                 f"selection config equal)")
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in load_file(f).items()}
    sd = {k: v for k, v in sd.items() if k.rsplit(".", 1)[-1].startswith("fsb_route_")}
    params = dict(model.named_parameters())
    need = route_param_names(model)
    missing = [n for n in need if n not in sd]
    extra = [k for k in sd if k not in params]
    shape = [k for k in sd if k in params and tuple(params[k].shape) != tuple(sd[k].shape)]
    if not need or missing or extra or shape:
        raise ValueError(f"--fsb_router_file {f}: the model has {len(need)} router tensors; missing from the file "
                         f"{len(missing)} {missing[:4]}, in the file but not in the model {len(extra)} {extra[:4]}, shape "
                         f"mismatch {len(shape)} {shape[:4]}")
    with torch.no_grad():
        for k in need:
            params[k].copy_(sd[k].to(params[k].dtype))
    kinds = sorted({k.rsplit(".", 1)[-1] for k in need})
    return (f"router file {f}: loaded {len(need)} router tensors {kinds} (strict: all present, shapes match, no extra "
            f"router tensor); {idrep}", set(sd))


def router_key_check(model, ckpt, router_file=None, allow_mismatch=False):
    # every fsb_route_* parameter must load from router_file (strict overlay) or ckpt's model.safetensors
    need = route_param_names(model)
    has_rr = any(n.rsplit(".", 1)[-1].startswith("fsb_route_rr_") for n in need)
    ck = os.path.join(ckpt, "model.safetensors") if (ckpt and os.path.isdir(ckpt)) else None
    ck_keys = _router_keys(ck) if (ck and os.path.isfile(ck)) else None
    ck_route = {k for k in ck_keys if k.rsplit(".", 1)[-1].startswith("fsb_route_")} if ck_keys is not None else set()
    if not has_rr and any(k.rsplit(".", 1)[-1].startswith("fsb_route_rr_") for k in ck_route):
        raise ValueError(f"{ck} carries residual re-ranker tensors (fsb_route_rr_*) but the model has no re-ranker: pass "
                         "--fsb_rerank_rr linear (else it would rank with the identity)")
    if router_file:
        rep, keys = overlay_router_file(model, router_file, ckpt=ckpt, allow_mismatch=allow_mismatch)
        return rep, keys
    if not need:
        return "no router tensors in the model", ck_route if ck_keys is not None else None
    if ck_keys is None:
        raise ValueError(f"the model has {len(need)} router tensors but {ckpt} has no model.safetensors to load them "
                         "from (pass --fsb_router_file)")
    missing = [n for n in need if n not in ck_keys]
    if missing:
        raise ValueError(f"{len(missing)}/{len(need)} router tensors missing from {ck}: {missing[:4]} (a missing tensor "
                         "keeps its init: zero adapter / zero re-ranker = the identity; pass --fsb_router_file)")
    kinds = sorted({n.rsplit(".", 1)[-1] for n in need})
    return f"router tensors from {ck}: {len(need)} {kinds}", ck_route


def radius_summary(model):
    # (mean, min, max) of the learned radius coefficients over all FS-B layers/heads, or None
    cs = [route_radius(m) for m in model.modules() if getattr(m, "fsb_enabled", False) and getattr(m, "fsb_route_learn_c", False)]
    if not cs:
        return None
    with torch.no_grad():
        c = torch.stack([c_.detach() for c_ in cs])
    return float(c.mean()), float(c.min()), float(c.max())


def _pos_term(ceil_c, lam, sigv, qpos, s):
    # lambda_h a_c erf(|i - (c+0.5) s| / (2 sigma_h)) for rows at positions qpos (nr,) -> (B,H,nr,M)
    M = ceil_c.shape[-1]
    cen = (torch.arange(M, device=ceil_c.device, dtype=torch.float32) + 0.5) * s
    erf = torch.erf((qpos.view(1, 1, -1, 1) - cen.view(1, 1, 1, M)).abs() / (2.0 * sigv.view(1, -1, 1, 1)))
    return lam.view(1, -1, 1, 1) * ceil_c[:, :, None, :] * erf


def _elig_mask(E_row, admit_full):
    # cells c < E_row (prefix per row) that are admitted -> (B,H,nr,M) bool
    M = admit_full.shape[-1]
    c = torch.arange(M, device=E_row.device)
    return (c[None, :] < E_row[:, None])[None, None] & admit_full[:, :, None, :]


def _student_loss(qrows, Kbar, ceil_c, lam, sigv, qpos, E_row, admit_full, top_idx, top_valid, logp_top,
                  logp_rest, has_rest, row_ok, logtemp, P1, P2, *, attn, alpha, s, mode):
    # student loss over the sampled rows: qrows (B,H,nr,Dh) detached, P1/P2 the adapter params; returns the sum
    with torch.autocast(device_type=qrows.device.type, enabled=False):
        qr = _route_q_grad(attn, qrows, alpha)
        r = torch.einsum("bhnd,bhmd->bhnm", qr, Kbar) * attn.scale + _pos_term(ceil_c, lam, sigv, qpos, s)
        r = r / logtemp.float().exp().view(1, -1, 1, 1)
        elig = _elig_mask(E_row, admit_full)
        r = r.masked_fill(~elig, NEG)
        lse = torch.logsumexp(r, dim=-1)                                  # (B,H,nr)
        if mode == "kl":
            ls_top = torch.gather(r, -1, top_idx) - lse[..., None]
            p_top = logp_top.exp()
            kl = torch.where(top_valid, p_top * (logp_top - ls_top), torch.zeros_like(ls_top)).sum(-1)
            r_rest = r.scatter(-1, top_idx, NEG)
            lq_rest = torch.logsumexp(r_rest, dim=-1) - lse
            kl = kl + torch.where(has_rest, logp_rest.exp() * (logp_rest - lq_rest), torch.zeros_like(kl))
            per_row = kl
        else:                                                            # Plackett-Luce over the teacher top-4
            rm = r
            per_row = torch.zeros_like(lse); nk = torch.zeros_like(lse)
            for k in range(4):
                ck = top_idx[..., k:k + 1]; vk = top_valid[..., k]
                lk = torch.logsumexp(rm, dim=-1) - torch.gather(r, -1, ck).squeeze(-1)
                per_row = per_row + torch.where(vk, lk, torch.zeros_like(lk)); nk = nk + vk.float()
                rm = rm.scatter(-1, ck, NEG)
            per_row = per_row / nk.clamp(min=1.0)
        per_row = torch.where(row_ok, per_row, torch.zeros_like(per_row))
        return per_row.sum()


# node-level distillation of the beam's decisions (fsb_rd_node 1)
NODE_LEVELS = 16                               # node recall is logged for traversal iterations 1..16
RD_STATS_BASE = 11
# stats layout: base | node KL sum, recall den, decision levels, trace mismatches, traced rows | node recall L1..16
RD_STATS_NODE = RD_STATS_BASE + 5 + NODE_LEVELS
_LAYOUT_CACHE = {}


def _flat_layout(F, device):
    # static flat node order of a block tree (per forest layout); cached per (ks, bases, leafbases, Ntot, Ltot, device)
    key = (tuple(F["ks"]), tuple(sorted(F["bases"].items())), tuple(sorted(F["leafbases"].items())), F["Ntot"],
           F["Ltot"], str(device))
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    kf, nf, lf, bf, pf = [], [], [], [], []
    for k in F["ks"]:
        nodes = (2 << k) - 1
        n = torch.arange(nodes)
        kf.append(torch.full((nodes,), k)); nf.append(n)
        lf.append(torch.log2((n + 1).float()).floor().long())
        bf.append(torch.full((nodes,), F["bases"][k])); pf.append(torch.full((nodes,), F["nodes_per"][k]))
    cat = lambda xs: torch.cat(xs).to(device) if xs else torch.zeros(0, dtype=torch.long, device=device)
    lay = {"kf": cat(kf), "nf": cat(nf), "lf": cat(lf), "bf": cat(bf), "pf": cat(pf), "K": F["Kmax"],
           "S": F["Kmax"] + 1, "ks": list(F["ks"]),
           "leafbase_t": torch.tensor([F["leafbases"].get(k, 0) for k in range(F["Kmax"] + 1)], device=device)}
    lay["Nc"] = int(lay["kf"].numel())
    # teacher pyramid (see _node_teacher): the leaves of chunk k padded to P = 2^K slots, reduced pairwise K times;
    # node (k, level l, index i) is entry i of row k at pyramid level jj = k - l
    K = F["Kmax"]; P = 1 << K
    assert list(F["ks"]) == list(range(K + 1)), F["ks"]
    off = [0]
    for jj in range(K):
        off.append(off[-1] + (K + 1) * (P >> jj))
    off_t = torch.tensor(off, device=device)
    jj = lay["kf"] - lay["lf"]
    ii = lay["nf"] - (torch.bitwise_left_shift(torch.ones_like(lay["lf"]), lay["lf"]) - 1)
    lay["tidx"] = off_t[jj] + lay["kf"] * torch.bitwise_right_shift(torch.full_like(jj, P), jj) + ii
    kk_ = torch.arange(K + 1, device=device)
    lay["ridx"] = off_t[kk_] + kk_ * torch.bitwise_right_shift(torch.full_like(kk_, P), kk_)
    lay["arP"] = torch.arange(P, device=device)
    lay["size_t"] = torch.bitwise_left_shift(torch.ones_like(kk_), kk_)
    if len(_LAYOUT_CACHE) > 64:
        _LAYOUT_CACHE.clear()
    _LAYOUT_CACHE[key] = lay
    return lay


def _tree_layout(mb, K):
    # binary-counter tree structure of admitted sizes mb -> (present, j, croot_depth, sp_exists, sp_depth), (..., K+1)
    ks = torch.arange(K + 1, device=mb.device)
    one = torch.ones_like(ks)
    bits = (mb[..., None] >> ks) & 1
    above = bits.flip(-1).cumsum(-1).flip(-1) - bits                    # set bits strictly above k
    lower = (mb[..., None] & (torch.bitwise_left_shift(one, ks) - 1)) > 0
    present = bits.bool()
    j = (mb[..., None] & ~(torch.bitwise_left_shift(one, ks + 1) - 1)) >> ks
    return present, j, above + lower.long(), present & lower, above


def _node_structure(lay, mb, W_row):
    # per-(t, block) flat-node structure; decision level: its nodes plus the leaves above it outnumber W_row
    K = lay["K"]
    present, j, crd, spx, spd = _tree_layout(mb, K)
    kf, lf = lay["kf"], lay["lf"]
    ex = torch.cat([present[..., kf], spx], -1)
    dep = torch.cat([crd[..., kf] + lf, spd], -1)
    leaf = torch.cat([(lf == kf).expand_as(present[..., kf]), torch.zeros_like(spx)], -1) & ex
    D1 = K + 2
    cnt_d = torch.zeros(*dep.shape[:-1], D1, dtype=torch.long, device=dep.device).scatter_add_(-1, dep, ex.long())
    lv_d = torch.zeros_like(cnt_d).scatter_add_(-1, dep, leaf.long())
    dec = ((cnt_d + lv_d.cumsum(-1) - lv_d) > W_row[..., None]) & (cnt_d > 0)   # (..., D1)
    decv = dec.gather(-1, dep) & ex
    return {"ex": ex, "dep": dep, "decv": decv, "j": j, "present": present, "spx": spx, "n_dec": dec.sum(-1)}


def _node_teacher(tcell, Fleaf, lay, j, present, spx):
    # teacher subtree masses T_v = logsumexp over the eligible cells under v -> (T,G,R,Nf); NEG (up to +K log 2)
    # for nodes without an eligible cell (absent nodes are masked by the caller)
    T, G, R, M = tcell.shape
    K = lay["K"]; P = 1 << K
    Ltot = Fleaf.shape[1]
    ar, size = lay["arP"], lay["size_t"]
    lidx = ((lay["leafbase_t"] + j * size)[..., None] + ar).clamp(max=Ltot - 1)          # (T,G,K+1,P)
    cells = Fleaf.gather(1, lidx.reshape(T, -1)).view(T, G, K + 1, P)
    ok = (ar < size[:, None]) & present[..., None] & (cells >= 0)
    cells = torch.where(ok, cells, torch.full_like(cells, M))
    tc = torch.cat([tcell, tcell.new_full((T, G, R, 1), NEG)], -1)                       # column M = sentinel
    lv = tc.gather(-1, cells.view(T, G, 1, -1).expand(T, G, R, -1)).view(T, G, R, K + 1, P)
    pyr = [lv.reshape(T, G, R, -1)]
    for _ in range(K):
        lv = torch.logaddexp(lv[..., 0::2], lv[..., 1::2])
        pyr.append(lv.reshape(T, G, R, -1))
    flat = torch.cat(pyr, -1)
    Tc = flat[..., lay["tidx"]]                                                            # (T,G,R,Nc)
    roots = flat[..., lay["ridx"]].masked_fill(~present[:, :, None, :], NEG)              # (T,G,R,K+1)
    sp = torch.logcumsumexp(roots, dim=-1).masked_fill(~spx[:, :, None, :], NEG)
    return torch.cat([Tc, sp], -1)


def _seg_lse(x, oh, dep):
    # per-level log-sum-exp: every level shifted by its own detached max (order-independent scatter amax seeded
    # with NEG), then float64 one-hot matmuls (no atomics); each level's sum >= 1, so no level underflows
    T_, G_, R_, Nf = x.shape
    idx = dep[:, :, None, :].expand(T_, G_, R_, Nf)
    with torch.no_grad():
        m = x.new_full((T_, G_, R_, oh.shape[-1]), NEG).scatter_reduce(-1, idx, x, "amax", include_self=True).double()
        mn = m.gather(-1, idx)                                            # (T,G,R,Nf) each node's level max
    e = torch.exp(x.double() - mn)
    lse = torch.log(torch.matmul(e, oh).clamp(min=1e-300)) + m                                # (T,G,R,D1)
    return torch.matmul(lse, oh.transpose(-1, -2)).float()


def _node_chunk_loss(q, tcell, Fnode, Fleaf, SPn, mb, W_row, qpos, lam_t, sig_t, *, attn, alpha, s, lay):
    # node-level KL summed over the rows of one chunk of query blocks; grad reaches the adapter, the learned
    # radius coefficient and the node temperature (all read from attn)
    B, H, _, Dh = q.shape
    T = B * H
    G, R = qpos.shape
    NC = Fnode.shape[-1]
    Ntot = Fnode.shape[1]
    with torch.autocast(device_type=q.device.type, enabled=False):
        with torch.no_grad():
            st = _node_structure(lay, mb, W_row)
            refc = (lay["bf"] + st["j"][..., lay["kf"]] * lay["pf"] + lay["nf"]).clamp(min=0, max=Ntot - 1)   # (T,G,Nc)
            rid = (torch.arange(T, device=q.device).view(T, 1, 1) * Ntot + refc).reshape(-1)
            A = Fnode.reshape(-1, NC).index_select(0, rid).view(T, G, lay["Nc"], NC)      # row gather
            A = torch.cat([A, SPn], 2)                                    # (T,G,Nf,NC)
            mu = A[..., :Dh]; rad = A[..., Dh]
            pmin_pos = (A[..., Dh + 2] + 0.5) * s
            posv = lam_t.view(T, 1, 1, 1) * A[..., Dh + 1][:, :, None, :] * torch.erf(
                (qpos[None, :, :, None] - pmin_pos[:, :, None, :]).abs() / (2.0 * sig_t.view(T, 1, 1, 1)))
            del A
            Tn = _node_teacher(tcell, Fleaf, lay, st["j"], st["present"], st["spx"]) / attn.fsb_rd_temp
            tval = st["ex"][:, :, None, :] & (Tn > NEG / 2)
            keep = tval & st["decv"][:, :, None, :]
            oh = torch.nn.functional.one_hot(st["dep"], lay["K"] + 2).double()            # (T,G,Nf,D1) level one-hot
            Tt = Tn.masked_fill(~tval, NEG)
            lpT = Tt - _seg_lse(Tt, oh, st["dep"])
            pT = lpT.exp()
        qr = _route_q_grad(attn, q, alpha).reshape(T, G, R, Dh)
        U = torch.einsum("tgrd,tgnd->tgrn", qr, mu) * attn.scale
        if attn.fsb_node_score == "bound":
            c = route_radius(attn)
            qn = (qr * qr).sum(-1).clamp(min=1e-30).sqrt()               # (T,G,R)
            c_t = (c.view(1, H).expand(B, H).reshape(T, 1, 1) if c is not None else attn.fsb_bound_radius_coef)
            U = U + (c_t * qn)[..., None] * rad[:, :, None, :] * attn.scale
        U = (U + posv) / attn.fsb_route_nlogtemp.float().exp().view(1, H).expand(B, H).reshape(T, 1, 1, 1)
        U = torch.where(tval, U, torch.full_like(U, NEG))
        lpU = U - _seg_lse(U, oh, st["dep"])
        kl = torch.where(keep, pT * (lpT - lpU), torch.zeros_like(U))
        return kl.sum()


def _node_trace_stats(attn, F, SP, lay, Qr_rows, rows, R, nbm, sel_rows, top1, ok1, lam, sigv, s, st_all, row_ok,
                      Qraw_rows=None, Kp=None, Bq=None):
    # no-grad node-recall stats of the forward's beam on the sampled rows: eager re-traversal with a per-iteration
    # trace (checked against the forward's selection); returns the (5 + NODE_LEVELS,) tail of the stats vector
    from .fsb_tree import traverse_forest, _forest_gather
    T = top1.shape[0] * top1.shape[1]
    H = top1.shape[1]
    Dh = Qr_rows.shape[-1]
    dev = Qr_rows.device
    Tq = T * nbm
    q_tr = Qr_rows.reshape(T, nbm, R, Dh).reshape(Tq, R, Dh).float()
    qpos = rows.float().view(1, nbm, R).expand(T, -1, -1).reshape(Tq, R).contiguous()
    head = (torch.arange(T, device=dev) % H).repeat_interleave(nbm)
    height = SP["height"]
    D = (height - attn.fsb_delta).clamp(min=0)
    W = torch.minimum(torch.full_like(D, attn.fsb_width), torch.bitwise_left_shift(torch.ones_like(D), D))
    W = W[:, None].expand(Tq, R).contiguous()
    delta = (height - D)[:, None].expand(Tq, R).contiguous()
    trace = []
    rkw = {}
    rt = getattr(attn, "fsb_read_top", 0)
    if rt > 0:
        # overselect: the forward read the re-ranked top of the beam's candidates; re-rank the re-run's candidates the same way
        rr = getattr(attn, "fsb_rerank_score", "leaf")
        rq = None
        if getattr(attn, "fsb_rerank_query", "route") == "raw" and Qraw_rows is not None and Qraw_rows is not Qr_rows:
            rq = Qraw_rows.reshape(T, nbm, R, Dh).reshape(Tq, R, Dh).float()
        rkw = dict(read_top=rt, rerank_score=rr, tokK=(Kp.detach() if rr != "leaf" else None), rq=rq)
        if getattr(attn, "fsb_union_width", 0) > 0:                        # fsb_union_width: the same union candidate pool
            from .fsb_tree import union_kw
            rkw.update(union_kw(attn))
        from .fsb_tree import hybrid_on, hybrid_kw, hybrid_pool, pool_order
        if hybrid_on(attn):                                                # fsb_hybrid: the same pool rows and merge
            assert Bq is not None, "the hybrid node trace needs the query block Bq"
            order = F["pool_order"] if "pool_order" in F else pool_order(F["cellA"])
            E_q = ((torch.arange(1, nbm + 1, device=dev) * Bq) // s)[None, :].expand(T, nbm).reshape(Tq)
            pool_q = hybrid_pool(order, torch.arange(T, device=dev).repeat_interleave(nbm), E_q, attn.fsb_pool_size,
                                 attn.fsb_pool_recent)
            rkw.update(hybrid_kw(attn, pool_q))
        from .fsb_tree import rr_kw, margin_kw, beam_kw                    # fsb_rerank_rr: the same residual re-ranker
        rkw.update(rr_kw(attn))
        rkw.update(margin_kw(attn))                                        # fsb_read_margin: the same adaptive read count
        rkw.update(beam_kw(attn))                                          # fsb_beam_quota: the same reserved beam picks
        from .fsb_tree import union_tau_on, union_tau_spines_kw, union_quota_kw, union_prior_kw
        if union_tau_on(attn) and F.get("utau_F") is not None:             # fsb_union_tau_quantile: the same second forest
            rkw.update(union_tau_spines_kw(attn, F, F["utau_mba"][:, 1:nbm + 1].reshape(Tq),
                                           torch.arange(T, device=dev).repeat_interleave(nbm), R))
        rkw.update(union_quota_kw(attn))                                   # fsb_union_quota: the same split re-rank
        rkw.update(union_prior_kw(attn))                                   # fsb_union_prior_scale: the same prior scale
    sel_tr, _, _ = traverse_forest(F, SP, q_tr, qpos, lam.float()[head].view(Tq, 1, 1), sigv.float()[head].view(Tq, 1, 1),
                                   attn.scale, s, W, delta, attn.fsb_W_max, attn.fsb_delta_max, attn.fsb_node_score,
                                   radius_coef=radius_rows(attn, head), trace=trace, **rkw)
    mism = (sel_tr != sel_rows.reshape(Tq, R, -1)).any(-1).sum().float()
    # position of the teacher's top cell in the admitted sequence and in every chunk size's leaf table
    M = F["M"]; K = lay["K"]
    c1 = top1.reshape(T, nbm * R).clamp(min=0)
    adm_sorted = F["adm"].masked_fill(F["adm"] < 0, M).contiguous()
    a_star = torch.searchsorted(adm_sorted, c1.contiguous()).view(Tq, R)
    inv = torch.full((T, K + 1, M + 1), -1, dtype=torch.long, device=dev)
    for k in lay["ks"]:
        L0 = F["leafbases"][k]; n = F["Cmaxs"][k] << k
        cells = F["leaf"][:, L0:L0 + n]
        inv[:, k].scatter_(1, cells.masked_fill(cells < 0, M), torch.arange(L0, L0 + n, device=dev).expand(T, -1).contiguous())
    t_of = torch.arange(T, device=dev).repeat_interleave(nbm)[:, None, None]   # (Tq,1,1)
    c1q = c1.view(Tq, R)[:, :, None]
    ok = ok1.reshape(Tq, R)
    hits = torch.zeros(NODE_LEVELS, device=dev)
    for it, ref in enumerate(trace[:NODE_LEVELS]):
        Wm = ref.shape[-1]
        at = _forest_gather(F, SP, ref.reshape(Tq, R * Wm), cols=slice(Dh + 6, Dh + 9)).view(Tq, R, Wm, 3)
        src, off, ln = at[..., 0].long(), at[..., 1].long(), at[..., 2].long()
        kn = (torch.searchsorted(lay["leafbase_t"], off.reshape(-1), right=True) - 1).clamp(min=0, max=K).view_as(off)
        pstar = inv.view(-1)[((t_of * (K + 1) + kn) * (M + 1) + c1q).reshape(-1)].view_as(off)
        a_ = a_star[:, :, None]
        inside = torch.where(src == 1, (off <= pstar) & (pstar < off + ln), (off <= a_) & (a_ < off + ln))
        hit = ((ref != -1) & (src > 0) & inside).any(-1) & ok
        hits[it] = hit.sum().float()
    if len(trace) < NODE_LEVELS and trace:                                 # frontier is final after the last iteration
        hits[len(trace):] = hits[len(trace) - 1]
    n_dec = (st_all["n_dec"].reshape(T, nbm, 1).expand(T, nbm, R).reshape(top1.shape) * row_ok).sum().float()
    head5 = torch.stack([ok.sum().float(), n_dec, mism, torch.tensor(float(Tq * R), device=dev)])
    return torch.cat([head5, hits])


def fsb_rd_loss(attn, Qf, Kp, Kbar, ceil_c, admit_full, admit_blk, sigma, sel_full, N, Bq, nb, s, Qr=None, forest=None):
    # router-distillation loss of one FS-B layer on the current forward; returns {loss (grad only to fsb_route_*),
    # stats (11,) or, with fsb_rd_node, (32,), rows}; None when nb < 2
    B, H, _, Dh = Qf.shape
    dev = Qf.device
    M = Kbar.shape[2]
    R = max(1, min(attn.fsb_rd_rows, Bq))
    if nb < 2:
        return None
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        # rows: R distinct offsets per query block b >= 1 (shared by batch and heads), block order
        u = torch.rand(nb - 1, Bq, device=dev, generator=attn._fsb_rd_gen)
        off = u.argsort(dim=-1)[:, :R].sort(dim=-1).values
        rows = (torch.arange(1, nb, device=dev)[:, None] * Bq + off).reshape(-1)   # (nr,)
        nr = rows.numel()
        blk = rows // Bq
        qpos = rows.float()
        E_row = (blk * Bq) // s                                           # cells finished before the block
        if attn.fsb_rd_far:
            E_row = torch.minimum(E_row, torch.div(rows - attn.window_size, s, rounding_mode="floor").clamp(min=0))
        E_row = E_row.clamp(max=M)
        lam = attn.sel_scale.detach().float()
        sigv = sigma.detach().float()
        # teacher: exact per-cell log-sum-exp of the raw token scores, chunked over rows
        teach = torch.full((B, H, nr, M), NEG, device=dev)
        rc = max(R, (max(attn.fsb_rd_chunk, R) // R) * R)
        Kf = Kp.float()                                                   # one fp32 copy of the keys per layer
        for j0 in range(0, nr, rc):
            j1 = min(j0 + rc, nr)
            b_hi = 1 + (j1 - 1) // R                                      # block of the chunk's last row (host)
            Emax = min(M, (b_hi * Bq) // s)
            if Emax <= 0:
                continue
            qc = Qf[:, :, rows[j0:j1]]                                    # (B,H,rc,Dh)
            sc = torch.matmul(qc, Kf[:, :, :Emax * s].transpose(-1, -2)) * attn.scale
            teach[:, :, j0:j1, :Emax] = sc.view(B, H, j1 - j0, Emax, s).logsumexp(-1)
            del sc
        del Kf
        elig = _elig_mask(E_row, admit_full)                              # (B,H,nr,M)
        if attn.fsb_rd_pos:
            teach = teach + _pos_term(ceil_c, lam, sigv, qpos, s)
        node = bool(getattr(attn, "fsb_rd_node", 0)) and forest is not None
        tcell = teach.masked_fill(~elig, NEG) if node else None          # node teacher: the same per-cell score
        tt = (teach / attn.fsb_rd_temp).masked_fill(~elig, NEG)
        n_elig = elig.sum(-1)                                             # (B,H,nr)
        row_ok = n_elig > 0
        topn = min(attn.fsb_rd_topn, M)
        vals, top_idx = tt.topk(topn, dim=-1)                             # sorted descending
        top_valid = vals > NEG / 2
        lse_t = torch.logsumexp(tt, dim=-1)
        logp_top = torch.where(top_valid, vals - lse_t[..., None], torch.full_like(vals, NEG))
        rest = tt.scatter(-1, top_idx, NEG)
        logp_rest = torch.logsumexp(rest, dim=-1) - lse_t
        has_rest = n_elig > top_valid.sum(-1)
        qrows = Qf[:, :, rows].contiguous()
        # recall of the forward's beam on the teacher top-4, per tree-size stratum
        sel = sel_full[:, :, rows].long()                                 # (B,H,nr,kk)
        t4, v4 = top_idx[..., :4], top_valid[..., :4]
        hit = ((t4[..., :, None] == sel[..., None, :]) & (sel[..., None, :] >= 0)).any(-1) & v4
        hits = hit.sum(-1).float(); den = v4.sum(-1).float()
        m_row = torch.gather(admit_blk.sum(-1), 2, blk.view(1, 1, -1).expand(B, H, -1)).float()   # tree size
        strata = (m_row < 64, (m_row >= 64) & (m_row <= 256), m_row > 256)
        # exhaustive leaf-score top-4: the routing score over every eligible cell (what the loss trains directly)
        qr_rows = (Qf if Qr is None else Qr)[:, :, rows]
        leaf = (torch.einsum("bhnd,bhmd->bhnm", qr_rows, Kbar.float()) * attn.scale
                + _pos_term(ceil_c.float(), lam, sigv, qpos, s)).masked_fill(~elig, NEG)
        l4 = leaf.topk(min(4, M), dim=-1).indices
        l4 = l4.masked_fill(torch.gather(leaf, -1, l4) <= NEG / 2, -1)
        lhit = ((t4[..., :, None] == l4[..., None, :]) & (l4[..., None, :] >= 0)).any(-1) & v4
        lhits = lhit.sum(-1).float()
        del leaf
        st, lst = [], []
        for sm in strata:
            st += [(hits * sm).sum(), (den * sm).sum()]
            lst.append((lhits * sm).sum())
    P1 = attn.fsb_route_W if attn.fsb_route_adapter == "full" else attn.fsb_route_A
    P2 = None if attn.fsb_route_adapter == "full" else attn.fsb_route_B
    with torch.enable_grad():
        fn = lambda *a_: _student_loss(*a_, attn=attn, alpha=attn.fsb_route_alpha, s=s, mode=attn.fsb_rd_loss)
        args = (qrows, Kbar.detach().float(), ceil_c.detach().float(), lam, sigv, qpos, E_row, admit_full,
                top_idx, top_valid, logp_top, logp_rest, has_rest, row_ok, attn.fsb_route_logtemp, P1, P2)
        if os.environ.get("FSB_RD_RECOMPUTE", "1") == "1":
            tot = torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        else:
            tot = fn(*args)
        nrow = row_ok.sum().float()
        loss = tot / nrow.clamp(min=1.0)
    stats = torch.stack(st + [tot.detach(), nrow] + lst)
    if node:
        # node-level KL of the beam's decisions (per chunk of query blocks, recomputed in backward)
        F_, m_b_all = forest
        from .fsb_tree import build_spines
        nbm = nb - 1
        T = B * H
        with torch.no_grad():
            lay = _flat_layout(F_, dev)
            mb = m_b_all[:, 1:nb].contiguous()                             # (T, nbm) the beam's tree sizes
            SP = build_spines(F_, mb.reshape(-1), torch.arange(T, device=dev).repeat_interleave(nbm))
            Dd = (SP["height"] - attn.fsb_delta).clamp(min=0)
            W_row = torch.minimum(torch.full_like(Dd, attn.fsb_width), torch.bitwise_left_shift(torch.ones_like(Dd), Dd)).view(T, nbm)
            st_all = _node_structure(lay, mb, W_row)
            tail = _node_trace_stats(attn, F_, SP, lay, (Qf if Qr is None else Qr)[:, :, rows], rows, R, nbm,
                                     sel_full[:, :, rows], top_idx[..., 0], top_valid[..., 0], lam, sigv, s, st_all, row_ok,
                                     Qraw_rows=(None if Qr is None or Qr is Qf else qrows), Kp=Kp, Bq=Bq)
            del st_all
            tc4 = tcell.view(T, nbm, R, M)
            SPn = SP["node"].view(T, nbm, SP["node"].shape[1], -1)
            qpos2 = qpos.view(nbm, R)
            lam_t = lam.view(1, H).expand(B, H).reshape(T); sig_t = sigv.view(1, H).expand(B, H).reshape(T)
        Gc = max(1, int(attn.fsb_rd_node_blkchunk))
        rec = os.environ.get("FSB_RD_RECOMPUTE", "1") == "1"
        ntot = None
        with torch.enable_grad():
            fn_n = lambda *a_: _node_chunk_loss(*a_, attn=attn, alpha=attn.fsb_route_alpha, s=s, lay=lay)
            for g0 in range(0, nbm, Gc):
                g1 = min(nbm, g0 + Gc)
                args_n = (qrows[:, :, g0 * R:g1 * R], tc4[:, g0:g1], F_["node"], F_["leaf"], SPn[:, g0:g1], mb[:, g0:g1],
                          W_row[:, g0:g1], qpos2[g0:g1], lam_t, sig_t)
                part = torch.utils.checkpoint.checkpoint(fn_n, *args_n, use_reentrant=False) if rec else fn_n(*args_n)
                ntot = part if ntot is None else ntot + part
            loss = loss + attn.fsb_rd_node_weight * ntot / nrow.clamp(min=1.0)
        stats = torch.cat([stats, ntot.detach().view(1), tail])
    return {"loss": loss, "stats": stats, "rows": rows}


def fsb_rd_collect(model, clear=True):
    # mean over the FS-B layers of the last forward's distillation losses and the summed stats; (None, None) if none
    losses, stats = [], None
    for m in model.modules():
        rd = getattr(m, "_fsb_rd", None) if getattr(m, "fsb_enabled", False) else None
        if rd is None:
            continue
        losses.append(rd["loss"])
        stats = rd["stats"] if stats is None else stats + rd["stats"]
        if clear:
            m._fsb_rd = None
    if not losses:
        return None, None
    return torch.stack(losses).mean(), stats


def rd_stats_str(st) -> str:
    v = [float(x) for x in st]
    rec = lambda h, d: (h / d if d > 0 else float("nan"))
    return (f"rd {v[6] / max(v[7], 1.0):.4f} | beam top4 recall <64 {rec(v[0], v[1]):.3f} 64-256 {rec(v[2], v[3]):.3f} "
            f">256 {rec(v[4], v[5]):.3f} all {rec(v[0] + v[2] + v[4], v[1] + v[3] + v[5]):.3f} | leaf top4 recall "
            f"{rec(v[8], v[1]):.3f} / {rec(v[9], v[3]):.3f} / {rec(v[10], v[5]):.3f} all "
            f"{rec(v[8] + v[9] + v[10], v[1] + v[3] + v[5]):.3f} | n {v[1]:.0f} / {v[3]:.0f} / {v[5]:.0f}"
            + (node_stats_str(v) if len(v) >= RD_STATS_NODE else ""))


def node_stats_str(v) -> str:
    rec = lambda h, d: (h / d if d > 0 else float("nan"))
    hits = v[RD_STATS_BASE + 5:RD_STATS_BASE + 5 + NODE_LEVELS]
    last = len(hits)
    while last > 1 and hits[last - 1] == hits[last - 2]:
        last -= 1                                                         # constant tail = frontier already final
    lv = " ".join(f"{rec(h, v[12]):.3f}" for h in hits[:last])
    return (f" || node KL {v[11] / max(v[7], 1.0):.4f} (decision levels/row {v[13] / max(v[7], 1.0):.2f}) | node recall "
            f"L1..L{last} {lv} | trace mismatch {v[14]:.0f}/{v[15]:.0f}")
