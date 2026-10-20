from __future__ import annotations

import math
import os
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.compile_guard import compiling

NEG = -1e9


class FSBPolicy(nn.Module):
    def __init__(self, head_dim: int, n_heads: int, n_actions: int, hidden: int = 64, emb: int = 16):
        super().__init__()
        self.head_emb = nn.Embedding(n_heads, emb)
        self.inp = nn.Linear(head_dim + emb + 2, hidden)
        self.out = nn.Linear(hidden, n_actions)
        self.val = nn.Linear(hidden, 1)

    def forward(self, qn, head_id, extra):
        Bq = qn.shape[1]
        e = self.head_emb(head_id)[:, None, :].expand(-1, Bq, -1)
        x = torch.cat([qn, e, extra[:, None, :].expand(-1, Bq, -1)], dim=-1)
        h = F.gelu(self.inp(x))
        return self.out(h), self.val(h).squeeze(-1)


def _parse_ints(v, dflt):
    if v is None:
        return list(dflt)
    if isinstance(v, str):
        return [int(t) for t in v.split(",") if t.strip()]
    return [int(t) for t in v]


def fsb_init(attn, n_heads: int, head_dim: int, kwargs: dict):
    attn.fsb_widths = _parse_ints(kwargs.get("fsb_widths"), (1, 2, 4))
    attn.fsb_deltas = _parse_ints(kwargs.get("fsb_deltas"), (0, 1, 2))
    attn.fsb_W_max = max(attn.fsb_widths)
    attn.fsb_delta_max = max(attn.fsb_deltas)
    attn.fsb_kk = attn.fsb_W_max * (1 << attn.fsb_delta_max)
    attn.fsb_read_top = int(kwargs.get("fsb_read_top", 0) or 0)
    if attn.fsb_kk > 16 and attn.fsb_read_top <= 0:
        raise ValueError(f"FS-B: W_max*2^Delta_max = {attn.fsb_kk} > 16 (fused kernel slot limit)")
    attn.fsb_policy_mode = str(kwargs.get("fsb_policy", "fixed"))
    assert attn.fsb_policy_mode in ("fixed", "learned"), attn.fsb_policy_mode
    attn.fsb_width = int(kwargs.get("fsb_width", attn.fsb_W_max))
    attn.fsb_delta = int(kwargs.get("fsb_delta", 0))
    if attn.fsb_read_top > 0:
        wm = max(attn.fsb_W_max, attn.fsb_width if attn.fsb_policy_mode == "fixed" else 1)
        attn.fsb_W_max = 1 << (wm - 1).bit_length()
        cand = attn.fsb_W_max * (1 << attn.fsb_delta_max)
        if attn.fsb_read_top > 16:
            raise ValueError(f"FS-B: fsb_read_top = {attn.fsb_read_top} > 16 (fused kernel slot limit)")
        if cand > 64:
            raise ValueError(f"FS-B overselect: beam candidates W_max*2^Delta_max = {cand} > 64")
        if attn.fsb_read_top > cand:
            raise ValueError(f"FS-B: fsb_read_top = {attn.fsb_read_top} > the beam's {cand} candidate cells")
        attn.fsb_kk = attn.fsb_read_top
    attn.fsb_rerank_score = str(kwargs.get("fsb_rerank_score", "leaf") or "leaf")
    if attn.fsb_rerank_score not in ("leaf", "lse", "lse_pos", "max"):
        raise ValueError(f"FS-B: fsb_rerank_score = {attn.fsb_rerank_score!r} (leaf|lse|lse_pos|max)")
    if attn.fsb_rerank_score != "leaf" and attn.fsb_read_top <= 0:
        raise ValueError(f"FS-B: fsb_rerank_score {attn.fsb_rerank_score} re-ranks the overselect candidates; "
                         "it needs fsb_read_top > 0")
    attn.fsb_rerank_query = str(kwargs.get("fsb_rerank_query", "route") or "route")
    if attn.fsb_rerank_query not in ("route", "raw"):
        raise ValueError(f"FS-B: fsb_rerank_query = {attn.fsb_rerank_query!r} (route|raw)")
    if attn.fsb_rerank_query != "route" and attn.fsb_read_top <= 0:
        raise ValueError(f"FS-B: fsb_rerank_query {attn.fsb_rerank_query} is the overselect re-rank's query; "
                         "it needs fsb_read_top > 0")
    attn.fsb_union_width = int(kwargs.get("fsb_union_width", 0) or 0)
    attn.fsb_union_W_max = 0
    if attn.fsb_union_width < 0:
        raise ValueError(f"FS-B: fsb_union_width = {attn.fsb_union_width} < 0")
    if attn.fsb_union_width > 0:
        if attn.fsb_read_top <= 0:
            raise ValueError("FS-B: fsb_union_width widens the overselect re-rank's candidate pool; it needs fsb_read_top > 0")
        attn.fsb_union_W_max = 1 << (attn.fsb_union_width - 1).bit_length()
        if attn.fsb_union_W_max * (1 << attn.fsb_delta_max) > 64:
            raise ValueError(f"FS-B: union beam candidates {attn.fsb_union_W_max}*2^{attn.fsb_delta_max} > 64")
    attn.fsb_read_margin = float(kwargs.get("fsb_read_margin", 0.0) or 0.0)
    _rm = kwargs.get("fsb_read_min", None)
    attn.fsb_read_min = 1 if _rm is None else int(_rm)
    if attn.fsb_read_margin < 0:
        raise ValueError(f"FS-B: fsb_read_margin = {attn.fsb_read_margin} < 0")
    if attn.fsb_read_margin > 0:
        if attn.fsb_read_top <= 0:
            raise ValueError("FS-B: fsb_read_margin trims the overselect re-rank's read set; it needs fsb_read_top > 0")
        if not 1 <= attn.fsb_read_min <= attn.fsb_read_top:
            raise ValueError(f"FS-B: fsb_read_min = {attn.fsb_read_min} (fsb_read_margin: 1..fsb_read_top = {attn.fsb_read_top})")
    elif attn.fsb_read_min != 1:
        raise ValueError(f"FS-B: fsb_read_min = {attn.fsb_read_min} is the floor of the adaptive read count; it needs "
                         "fsb_read_margin > 0")
    attn.fsb_node_score = str(kwargs.get("fsb_node_score", "bound"))
    assert attn.fsb_node_score in ("bound", "centroid"), attn.fsb_node_score
    attn.fsb_bound_radius_coef = float(kwargs.get("fsb_bound_radius_coef", 1.0))
    attn.fsb_leaf_radius = bool(int(kwargs.get("fsb_leaf_radius", 0)))
    attn.fsb_route_cost = float(kwargs.get("fsb_route_cost", 0.5))
    attn.fsb_readout_blkchunk = int(kwargs.get("fsb_readout_blkchunk", 4))
    attn.fsb_policy_temp = float(kwargs.get("fsb_policy_temp", 1.0))
    attn.fsb_tree = str(kwargs.get("fsb_tree", "rebuild"))
    assert attn.fsb_tree in ("rebuild", "forest"), attn.fsb_tree
    _pb = kwargs.get("fsb_prefill_blkchunk", None)
    attn.fsb_prefill_blkchunk = int(_pb) if _pb is not None else int(os.environ.get("FSB_PREFILL_BLKCHUNK", "32"))
    # given (kwarg or FSB_PREFILL_BLKCHUNK), the stage loop's chunk size is also the memory bound of the fast inference
    # prefill's stages (fsb_prefill._stage_blocks); left unset, that path keeps its own FSB_FAST_PREFILL_BLKCHUNK
    attn.fsb_prefill_blkchunk_set = _pb is not None or "FSB_PREFILL_BLKCHUNK" in os.environ
    menu = [(w, d) for d in attn.fsb_deltas for w in attn.fsb_widths]
    attn.register_buffer("fsb_actions", torch.tensor(menu, dtype=torch.long), persistent=False)
    attn.register_buffer("fsb_tau", torch.zeros(n_heads), persistent=True)
    attn.fsb_policy = FSBPolicy(head_dim, n_heads, len(menu)) if attn.fsb_policy_mode == "learned" else None
    attn.fsb_rl = str(kwargs.get("fsb_rl", "reinforce"))
    assert attn.fsb_rl in ("reinforce", "ppo"), attn.fsb_rl
    attn.fsb_ppo_store_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[str(kwargs.get("fsb_ppo_store_dtype", "bf16"))]
    attn._fsb_ppo_rec = []
    attn._fsb_ppo_store = []
    attn._fsb_ppo_nmb = 0
    attn.recompute_long_range = False
    attn._fsb_stats = None
    attn._fsb_calib = None
    from .fsb_route import route_init
    route_init(attn, n_heads, head_dim, kwargs)
    beam_quota_init(attn, kwargs)
    hybrid_init(attn, kwargs)
    union_tau_init(attn, n_heads, kwargs)
    union_quota_init(attn, kwargs)
    union_prior_init(attn, kwargs)


def union_tau_init(attn, n_heads: int, kwargs: dict):
    attn.fsb_union_tau_quantile = float(kwargs.get("fsb_union_tau_quantile", 0.0) or 0.0)
    attn.register_buffer("fsb_tau_union", torch.zeros(n_heads), persistent=False)
    q = attn.fsb_union_tau_quantile
    if q < 0 or q >= 1:
        raise ValueError(f"FS-B: fsb_union_tau_quantile = {q} (0 = off, else a quantile in (0, 1))")
    if q == 0:
        return
    if int(getattr(attn, "fsb_union_width", 0) or 0) <= 0:
        raise ValueError("FS-B: fsb_union_tau_quantile filters the SECOND beam's forest; it needs fsb_union_width > 0")
    if int(getattr(attn, "fsb_read_top", 0) or 0) <= 0:
        raise ValueError("FS-B: fsb_union_tau_quantile needs fsb_read_top > 0 (the union candidates are re-ranked)")
    if getattr(attn, "fsb_tree", "rebuild") != "forest":
        raise ValueError("FS-B: fsb_union_tau_quantile needs fsb_tree forest (the rebuild tree paths are not wired)")
    if getattr(attn, "fsb_policy_mode", "fixed") != "fixed":
        raise ValueError("FS-B: fsb_union_tau_quantile needs the fixed policy (the second forest's stop depth is fsb_delta)")
    if getattr(attn, "fsb_hybrid", "off") == "pool":
        raise ValueError("FS-B: fsb_hybrid pool runs no beam; fsb_union_tau_quantile filters the second beam's forest")


def union_tau_on(attn) -> bool:
    return float(getattr(attn, "fsb_union_tau_quantile", 0.0) or 0.0) > 0


def union_tau_kw(attn, F2, SP2, Bq: int) -> dict:
    if F2 is None or SP2 is None or not union_tau_on(attn):
        return {}
    h2 = SP2["height"]
    D2 = (h2 - int(attn.fsb_delta)).clamp(min=0)
    return {"union_F": F2, "union_SP": SP2, "union_delta": (h2 - D2)[:, None].expand(h2.shape[0], Bq)}


def union_tau_spines_kw(attn, F: dict, m_b2, t_idx, Bq: int) -> dict:
    F2 = F.get("utau_F") if F is not None else None
    if F2 is None or m_b2 is None or not union_tau_on(attn):
        return {}
    return union_tau_kw(attn, F2, build_spines(F2, m_b2, t_idx), Bq)


@torch.no_grad()
def utau_build(attn, Kbar_s, ceil_s, E, spread_s=None):
    T, M, Dh = Kbar_s.shape
    H = attn.fsb_tau_union.shape[0]
    admit_u = ceil_s >= attn.fsb_tau_union.view(1, H, 1).expand(T // H, H, 1).reshape(T, 1)
    from .fsb_build_cuda import forests_cuda
    got = forests_cuda(Kbar_s, ceil_s, spread_s, [attn.fsb_tau_union], E)
    if got is not None:
        return got[0][0], got[0][1], admit_u
    F2 = build_forest(Kbar_s, ceil_s, admit_u, spread=spread_s)
    F2["chunks"] = {}
    adm2 = F2["adm"].masked_fill(F2["adm"] < 0, M)
    m_b2_all = torch.searchsorted(adm2, E[None, :].expand(T, -1).contiguous())
    return F2, m_b2_all, admit_u


def utau_prov_add(prov_out: list, sel: torch.Tensor, sel_unew: torch.Tensor):
    from .fsb_rerank import PROV_UTAU
    bit = _member(sel, sel_unew).to(torch.int8) * PROV_UTAU
    if prov_out:
        prov_out[-1] = prov_out[-1] + bit
    else:
        prov_out.append(bit.masked_fill(sel < 0, -1))


def fsb_utau_acc(attn, prov):
    acc = getattr(attn, "_fsb_read_acc", None)
    if acc is None or prov is None:
        return
    from .fsb_rerank import PROV_UTAU, PROV_UNIONQ
    p = prov.detach().reshape(-1).to(torch.int16)
    ok = p >= 0
    acc[3].add_((ok & ((p & PROV_UTAU) != 0)).sum().float())
    acc[4].add_((ok & ((p & PROV_UNIONQ) != 0)).sum().float())


def union_quota_init(attn, kwargs: dict):
    attn.fsb_union_quota = int(kwargs.get("fsb_union_quota", 0) or 0)
    k = attn.fsb_union_quota
    if k < 0:
        raise ValueError(f"FS-B: fsb_union_quota = {k} < 0")
    if k == 0:
        return
    rt = int(getattr(attn, "fsb_read_top", 0) or 0)
    if int(getattr(attn, "fsb_union_width", 0) or 0) <= 0:
        raise ValueError("FS-B: fsb_union_quota reserves read slots for the SECOND beam's cells; it needs fsb_union_width > 0")
    if rt <= 0:
        raise ValueError("FS-B: fsb_union_quota splits the overselect re-rank's read; it needs fsb_read_top > 0")
    if getattr(attn, "fsb_tree", "rebuild") != "forest":
        raise ValueError("FS-B: fsb_union_quota needs fsb_tree forest (the rebuild tree paths are not wired)")
    if getattr(attn, "fsb_policy_mode", "fixed") != "fixed":
        raise ValueError("FS-B: fsb_union_quota needs the fixed policy")
    if k > rt:
        raise ValueError(f"FS-B: fsb_union_quota = {k} > fsb_read_top = {rt} (the reserved union slots must fit the read)")
    hy = getattr(attn, "fsb_hybrid", "off")
    if hy in ("pool", "union"):
        raise ValueError(f"FS-B: fsb_hybrid {hy} {'runs no beam' if hy == 'pool' else 're-ranks one common shortlist'}; "
                         "fsb_union_quota needs the main / second beam split (use off or quota)")
    if float(getattr(attn, "fsb_read_margin", 0.0) or 0.0) > 0:
        raise ValueError("FS-B: fsb_read_margin trims the re-rank's read set, but fsb_union_quota fixes the read count by its "
                         "merge (reserved union slots + the main order): use one of the two")
    if getattr(attn, "fsb_rerank_rr", "none") != "none":
        raise ValueError("FS-B: fsb_rerank_rr forms its features over one shortlist; fsb_union_quota re-ranks the main and "
                         "the second-beam-only lists apart (not wired)")
    bq = int(getattr(attn, "fsb_beam_quota", 0) or 0)
    kl = int(getattr(attn, "fsb_pool_quota", 0) or 0) if hy == "quota" else 0
    if kl + bq + k > rt:
        raise ValueError(f"FS-B: fsb_pool_quota + fsb_beam_quota + fsb_union_quota = {kl} + {bq} + {k} > fsb_read_top = {rt} "
                         "(the reserved pool, beam and union slots must fit the read)")


def union_quota_on(attn) -> bool:
    return int(getattr(attn, "fsb_union_quota", 0) or 0) > 0


def union_quota_kw(attn) -> dict:
    k = int(getattr(attn, "fsb_union_quota", 0) or 0)
    return {"union_quota": k} if k > 0 else {}


def _priority_merge(parts, K: int) -> torch.Tensor:
    m = torch.cat(parts, dim=-1)
    C = m.shape[-1]
    tri = torch.tril(torch.ones(C, C, dtype=torch.bool, device=m.device), diagonal=-1)
    dup = ((m[..., :, None] == m[..., None, :]) & tri).any(-1)
    m = m.masked_fill(dup, -1)
    keep = (m < 0).to(torch.int32).argsort(dim=-1, stable=True)
    return m.gather(-1, keep)[..., :K]


def union_quota_list(ures: torch.Tensor, Tm: torch.Tensor, urem: torch.Tensor) -> torch.Tensor:
    return torch.cat([ures, Tm, urem], dim=-1)


def union_quota_merge(ures: torch.Tensor, Tm: torch.Tensor, urem: torch.Tensor, K: int, bres=None, L=None,
                      K_L: int = 0) -> torch.Tensor:
    parts = []
    if L is not None:
        parts.append(L[..., :K_L])
    if bres is not None:
        parts.append(bres)
    parts += [ures, Tm, urem]
    if L is not None:
        parts.append(L[..., K_L:])
    return _priority_merge(parts, K)


def union_quota_prov_add(prov_out: list, sel: torch.Tensor, sel_unew: torch.Tensor, ures: torch.Tensor):
    from .fsb_rerank import PROV_UNIONQ
    utau_prov_add(prov_out, sel, sel_unew)
    prov_out[-1] = prov_out[-1] + _member(sel, ures).to(torch.int8) * PROV_UNIONQ


def union_prior_init(attn, kwargs: dict):
    v = kwargs.get("fsb_union_prior_scale", None)
    attn.fsb_union_prior_scale = 1.0 if v is None else float(v)
    s = attn.fsb_union_prior_scale
    if not (0.0 <= s <= 1.0):
        raise ValueError(f"FS-B: fsb_union_prior_scale = {s} (1 = off, else a scale in [0, 1])")
    if s == 1.0:
        return
    if int(getattr(attn, "fsb_union_width", 0) or 0) <= 0:
        raise ValueError("FS-B: fsb_union_prior_scale scales the SECOND beam's candidates' prior term; it needs fsb_union_width > 0")
    if int(getattr(attn, "fsb_read_top", 0) or 0) <= 0:
        raise ValueError("FS-B: fsb_union_prior_scale needs fsb_read_top > 0 (the union candidates are re-ranked)")
    if getattr(attn, "fsb_tree", "rebuild") != "forest":
        raise ValueError("FS-B: fsb_union_prior_scale needs fsb_tree forest (the rebuild tree paths are not wired)")
    if getattr(attn, "fsb_policy_mode", "fixed") != "fixed":
        raise ValueError("FS-B: fsb_union_prior_scale needs the fixed policy")
    if getattr(attn, "fsb_hybrid", "off") == "pool":
        raise ValueError("FS-B: fsb_hybrid pool runs no beam; fsb_union_prior_scale scales the second beam's candidates")


def union_prior_on(attn) -> bool:
    return float(getattr(attn, "fsb_union_prior_scale", 1.0)) != 1.0


def union_prior_kw(attn) -> dict:
    return {"union_prior": float(attn.fsb_union_prior_scale)} if union_prior_on(attn) else {}


def union_prior_vec(Cm: int, Cu: int, s: float, device) -> torch.Tensor:
    return torch.cat([torch.ones(1, 1, Cm, device=device), torch.full((1, 1, Cu), float(s), device=device)], dim=-1)


def _prior_extend(ps: torch.Tensor, Cx: int) -> torch.Tensor:
    return torch.cat([ps, torch.ones(1, 1, Cx, device=ps.device)], dim=-1)


def beam_quota_init(attn, kwargs: dict):
    attn.fsb_beam_quota = int(kwargs.get("fsb_beam_quota", 0) or 0)
    k = attn.fsb_beam_quota
    if k < 0:
        raise ValueError(f"FS-B: fsb_beam_quota = {k} < 0")
    if k == 0:
        return
    rt = int(getattr(attn, "fsb_read_top", 0) or 0)
    if rt <= 0:
        raise ValueError("FS-B: fsb_beam_quota reserves the main beam's picks inside the overselect re-rank's read; it needs "
                         "fsb_read_top > 0")
    if getattr(attn, "fsb_tree", "rebuild") != "forest":
        raise ValueError("FS-B: fsb_beam_quota needs fsb_tree forest (the rebuild tree paths are not wired)")
    if getattr(attn, "fsb_policy_mode", "fixed") != "fixed":
        raise ValueError("FS-B: fsb_beam_quota needs the fixed policy (the main beam's width is fsb_width)")
    if k > rt:
        raise ValueError(f"FS-B: fsb_beam_quota = {k} > fsb_read_top = {rt} (the reserved picks must fit the read)")
    if k > int(attn.fsb_width):
        raise ValueError(f"FS-B: fsb_beam_quota = {k} > fsb_width = {attn.fsb_width} (the main beam retains at most fsb_width "
                         "cells at Delta 0)")
    if float(getattr(attn, "fsb_read_margin", 0.0) or 0.0) > 0:
        raise ValueError("FS-B: fsb_read_margin trims the re-rank's read set, but fsb_beam_quota fixes the read count by its "
                         "merge (reserved picks + the re-rank order): use one of the two")


def beam_kw(attn) -> dict:
    k = int(getattr(attn, "fsb_beam_quota", 0) or 0)
    return {"beam_quota": k} if k > 0 else {}


def beam_reserved(sel_main: torch.Tensor, k: int) -> torch.Tensor:
    keep = (sel_main < 0).to(torch.int32).argsort(dim=-1, stable=True)
    return sel_main.gather(-1, keep)[..., :k]


def beam_quota_merge(bres: torch.Tensor, Tr: torch.Tensor, K: int, L=None, K_L: int = 0) -> torch.Tensor:
    parts = [bres, Tr] if L is None else [L[..., :K_L], bres, Tr, L[..., K_L:]]
    m = torch.cat(parts, dim=-1)
    C = m.shape[-1]
    tri = torch.tril(torch.ones(C, C, dtype=torch.bool, device=m.device), diagonal=-1)
    dup = ((m[..., :, None] == m[..., None, :]) & tri).any(-1)
    m = m.masked_fill(dup, -1)
    keep = (m < 0).to(torch.int32).argsort(dim=-1, stable=True)
    return m.gather(-1, keep)[..., :K]


def beam_prov(sel: torch.Tensor, bres: torch.Tensor) -> torch.Tensor:
    from .fsb_rerank import PROV_BEAMQ
    p = _member(sel, bres).to(torch.int8) * PROV_BEAMQ
    return p.masked_fill(sel < 0, -1)


HYBRID_MODES = ("off", "pool", "union", "quota")


def hybrid_init(attn, kwargs: dict):
    def _int(name, dflt):
        v = kwargs.get(name, None)
        return dflt if v is None else int(v)
    mode = str(kwargs.get("fsb_hybrid", "off") or "off")
    if mode not in HYBRID_MODES:
        raise ValueError(f"FS-B: fsb_hybrid = {mode!r} ({'|'.join(HYBRID_MODES)})")
    attn.fsb_hybrid = mode
    attn.fsb_pool_size = _int("fsb_pool_size", 64)
    attn.fsb_pool_recent = _int("fsb_pool_recent", 8)
    attn.fsb_pool_short = _int("fsb_pool_short", 4)
    attn.fsb_pool_quota = _int("fsb_pool_quota", 1)
    if mode == "off":
        return
    rt = int(getattr(attn, "fsb_read_top", 0) or 0)
    if rt <= 0:
        raise ValueError(f"FS-B: fsb_hybrid {mode} selects a fixed number of cells; it needs fsb_read_top > 0")
    if getattr(attn, "fsb_tree", "rebuild") != "forest":
        raise ValueError(f"FS-B: fsb_hybrid {mode} needs fsb_tree forest (the rebuild tree paths are not wired)")
    if getattr(attn, "fsb_policy_mode", "fixed") != "fixed":
        raise ValueError(f"FS-B: fsb_hybrid {mode} needs the fixed policy")
    if getattr(attn, "fsb_route_adapter", "none") != "none" and getattr(attn, "fsb_rerank_query", "route") != "raw":
        raise ValueError(f"FS-B: fsb_hybrid {mode} with a routing adapter needs fsb_rerank_query raw (the pool and the "
                         "final re-rank must score with the raw query, never the routing query)")
    if attn.fsb_pool_size < 0 or attn.fsb_pool_recent < 0:
        raise ValueError(f"FS-B: fsb_pool_size ({attn.fsb_pool_size}) and fsb_pool_recent ({attn.fsb_pool_recent}) "
                         "must be >= 0")
    if attn.fsb_pool_size == 0 and attn.fsb_pool_recent > 0:
        attn.fsb_pool_recent = 0
    if mode == "union" or (mode == "quota" and getattr(attn, "fsb_rerank_rr", "none") != "none"):
        cp = attn.fsb_pool_size + attn.fsb_pool_recent
        if attn.fsb_pool_short < 1 or (attn.fsb_pool_size > 0 and attn.fsb_pool_short > cp):
            raise ValueError(f"FS-B: fsb_pool_short = {attn.fsb_pool_short} ({mode}"
                             f"{' + fsb_rerank_rr' if mode == 'quota' else ''}: 1..fsb_pool_size + fsb_pool_recent = {cp})")
    if mode == "quota" and not 1 <= attn.fsb_pool_quota <= rt:
        raise ValueError(f"FS-B: fsb_pool_quota = {attn.fsb_pool_quota} (quota: 1..fsb_read_top = {rt})")
    if mode == "pool" and getattr(attn, "fsb_rd_weight", 0.0) > 0:
        raise ValueError("FS-B: fsb_hybrid pool reads no tree; router distillation (fsb_rd_weight > 0) has nothing to train")
    if mode == "pool" and getattr(attn, "fsb_rerank_rr", "none") != "none":
        raise ValueError("FS-B: fsb_hybrid pool orders its read by FS-L's score g only; the residual re-ranker "
                         "(fsb_rerank_rr) has nothing to re-rank")
    if mode in ("pool", "quota") and float(getattr(attn, "fsb_read_margin", 0.0) or 0.0) > 0:
        raise ValueError(f"FS-B: fsb_read_margin trims the re-rank's read set, but fsb_hybrid {mode} fixes the read count "
                         "by its merge (pool: L[:K]; quota: the reserved pool slots + the tree list): use off or union")
    bq = int(getattr(attn, "fsb_beam_quota", 0) or 0)
    if mode == "pool" and bq > 0:
        raise ValueError("FS-B: fsb_hybrid pool runs no beam; fsb_beam_quota reserves the main beam's picks (use union or quota)")
    if mode == "quota" and bq > 0 and attn.fsb_pool_quota + bq > rt:
        raise ValueError(f"FS-B: fsb_pool_quota + fsb_beam_quota = {attn.fsb_pool_quota} + {bq} > fsb_read_top = {rt} "
                         "(the reserved pool and beam slots must fit the read)")


def hybrid_on(attn) -> bool:
    return getattr(attn, "fsb_hybrid", "off") != "off"


def hybrid_kw(attn, pool) -> dict:
    if not hybrid_on(attn):
        return {}
    return {"hybrid": attn.fsb_hybrid, "pool": pool, "pool_short": attn.fsb_pool_short, "pool_quota": attn.fsb_pool_quota}


@torch.no_grad()
def pool_order(cellA: torch.Tensor) -> torch.Tensor:
    from .prefix_topp import stable_topk_slots
    M = cellA.shape[1]
    return stable_topk_slots(cellA, torch.arange(M, device=cellA.device).view(1, M), M)


@torch.no_grad()
def hybrid_pool(order: torch.Tensor, t_idx: torch.Tensor, E: torch.Tensor, P: int, Wc: int) -> torch.Tensor:
    Tp = t_idx.shape[0]
    M = order.shape[1]
    dev = order.device
    E = E.view(Tp, 1).long()
    ids = torch.arange(M, device=dev).view(1, M)
    if P <= 0:
        return torch.where(ids < E, ids, torch.full_like(ids, -1)).expand(Tp, M).contiguous()
    o = order[t_idx]
    csum = (o < E).long().cumsum(-1).contiguous()
    tgt = torch.arange(1, P + 1, device=dev).view(1, P).expand(Tp, P).contiguous()
    pos = torch.searchsorted(csum, tgt)
    top = o.gather(1, pos.clamp(max=M - 1)).masked_fill(pos >= M, -1)
    if Wc > 0:
        near = E - 1 - torch.arange(Wc, device=dev).view(1, Wc)
        near = near.masked_fill(near < 0, -1)
        dup = ((near[:, :, None] == top[:, None, :]) & (near[:, :, None] >= 0)).any(-1)
        cand = torch.cat([top, near.masked_fill(dup, -1)], dim=-1)
    else:
        cand = top
    keep = (cand < 0).to(torch.int32).argsort(dim=-1, stable=True)
    return cand.gather(-1, keep)


@torch.no_grad()
def pool_scores(pool: torch.Tensor, pq: torch.Tensor, qpos, lam, sigv, cellK, cellA, t_idx, scale: float, s: int):
    from .fsb_rerank import cell_token_scores
    Tp, Bq, _ = pq.shape
    Cp = pool.shape[-1]
    sel = pool[:, None, :].expand(Tp, Bq, Cp)
    Mc = cellA.shape[1]
    content = cell_token_scores(pq, sel, cellK, t_idx, scale, 1, "max")
    gid = t_idx.view(Tp, 1, 1) * Mc + sel.clamp(min=0, max=Mc - 1)
    a = cellA.reshape(-1)[gid.reshape(-1)].view(Tp, Bq, Cp)
    pos_c = (sel.float() + 0.5) * s
    erf = torch.erf((qpos[..., None] - pos_c).abs() / (2.0 * sigv))
    return (content + lam * a * erf).masked_fill(sel < 0, float("-inf"))


@torch.no_grad()
def pool_rank(pool: torch.Tensor, pq: torch.Tensor, qpos, lam, sigv, cellK, cellA, t_idx, scale: float, s: int, K: int):
    from .prefix_topp import stable_topk_slots
    Tp, Bq, _ = pq.shape
    Cp = pool.shape[-1]
    step = max(1, (1 << 24) // max(1, Bq * Cp))
    outs = []
    for r0 in range(0, Tp, step):
        r1 = min(Tp, r0 + step)
        sl = slice(r0, r1)
        g = pool_scores(pool[sl], pq[sl], qpos[sl], lam[sl], sigv[sl], cellK, cellA, t_idx[sl], scale, s)
        ids = pool[sl][:, None, :].expand_as(g)
        k = min(K, Cp)
        slot = stable_topk_slots(g, ids, k)
        L = ids.gather(-1, slot).masked_fill(g.gather(-1, slot) == float("-inf"), -1)
        if k < K:
            L = torch.cat([L, L.new_full((r1 - r0, Bq, K - k), -1)], dim=-1)
        outs.append(L)
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def _member(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a[..., :, None] == b[..., None, :]) & (b[..., None, :] >= 0)).any(-1) & (a >= 0)


def hybrid_prov(sel: torch.Tensor, tree, plist, bres=None) -> torch.Tensor:
    in_p = _member(sel, plist)
    in_t = _member(sel, tree) if tree is not None else torch.zeros_like(in_p)
    prov = torch.where(in_t & in_p, 2, torch.where(in_p, 1, 0))
    if bres is not None:
        from .fsb_rerank import PROV_BEAMQ
        prov = prov + _member(sel, bres).long() * PROV_BEAMQ
    return prov.masked_fill(sel < 0, -1).to(torch.int8)


def quota_merge(L: torch.Tensor, Tr: torch.Tensor, K_L: int, K: int) -> torch.Tensor:
    m = torch.cat([L[..., :K_L], Tr, L[..., K_L:]], dim=-1)
    C = m.shape[-1]
    tri = torch.tril(torch.ones(C, C, dtype=torch.bool, device=m.device), diagonal=-1)
    dup = ((m[..., :, None] == m[..., None, :]) & tri).any(-1)
    m = m.masked_fill(dup, -1)
    keep = (m < 0).to(torch.int32).argsort(dim=-1, stable=True)
    return m.gather(-1, keep)[..., :K]


def hybrid_codes(cand: torch.Tensor, tree, plist, bres=None) -> torch.Tensor:
    from .fsb_rerank import PROV_TREE, PROV_POOL, PROV_BEAMQ
    in_p = _member(cand, plist)
    in_t = _member(cand, tree) if tree is not None else torch.zeros_like(in_p)
    codes = in_t.long() * PROV_TREE + in_p.long() * PROV_POOL
    if bres is not None:
        codes = codes + _member(cand, bres).long() * PROV_BEAMQ
    return codes


def hybrid_shortlist(sel_tree, plist, bres=None):
    C = plist if sel_tree is None else union_candidates(sel_tree, plist)
    return C, hybrid_codes(C, sel_tree, plist, bres)


@torch.no_grad()
def hybrid_select(sel_tree, pool, pq, qpos, lam, sigv, F: dict, SP: dict, scale: float, s: int, read_top: int,
                  rerank_score: str, tokK, mode: str, pool_short: int, pool_quota: int, prov_out=None, rr_w=None,
                  read_margin: float = 0.0, read_min: int = 1, bres=None, tree_rank=None, prior_scale=None):
    tabs = (F["cellK"], F["cellA"], SP["t_idx"])
    mkw = {"read_margin": read_margin, "read_min": read_min} if read_margin > 0 else {}
    if mode == "pool":
        assert rr_w is None, "fsb_hybrid pool has no re-rank: fsb_rerank_rr is refused at construction"
        assert not mkw, "fsb_hybrid pool reads a fixed count: fsb_read_margin is refused at construction"
        assert bres is None, "fsb_hybrid pool runs no beam: fsb_beam_quota is refused at construction"
        assert tree_rank is None, "fsb_hybrid pool runs no beam: fsb_union_quota is refused at construction"
        assert prior_scale is None, "fsb_hybrid pool runs no beam: fsb_union_prior_scale is refused at construction"
        L = pool_rank(pool, pq, qpos, lam, sigv, *tabs, scale, s, read_top)
        out, plist = L, L
    elif mode == "union":
        assert bres is None or not mkw, "fsb_beam_quota with fsb_read_margin is refused at construction"
        assert tree_rank is None, "fsb_hybrid union re-ranks one common shortlist: fsb_union_quota is refused at construction"
        plist = pool_rank(pool, pq, qpos, lam, sigv, *tabs, scale, s, pool_short)
        cand = union_candidates(sel_tree, plist)
        pkw = {"prior_scale": _prior_extend(prior_scale, plist.shape[-1])} if prior_scale is not None else {}
        if rr_w is None:
            out = rerank_top(cand, pq, qpos, lam, sigv, *tabs, scale, s, read_top, score=rerank_score, tokK=tokK, **mkw, **pkw)
        else:
            out = rerank_top(cand, pq, qpos, lam, sigv, *tabs, scale, s, read_top, score=rerank_score, tokK=tokK,
                             rr_w=rr_w, prov=hybrid_codes(cand, sel_tree, plist, bres), **mkw, **pkw)[0]
        if bres is not None:
            out = beam_quota_merge(bres, out, read_top)
    elif mode == "quota":
        assert not mkw, "fsb_hybrid quota reads a fixed count: fsb_read_margin is refused at construction"
        plist = pool_rank(pool, pq, qpos, lam, sigv, *tabs, scale, s, read_top)
        if tree_rank is not None:
            assert rr_w is None, "fsb_rerank_rr with fsb_union_quota is refused at construction"
            Tr = tree_rank
        elif rr_w is None:
            Tr = rerank_top(sel_tree, pq, qpos, lam, sigv, *tabs, scale, s, read_top, score=rerank_score, tokK=tokK,
                            **({"prior_scale": prior_scale} if prior_scale is not None else {}))
        else:
            from .fsb_rerank import PROV_TREE
            pshort = plist if pool_short == read_top else pool_rank(pool, pq, qpos, lam, sigv, *tabs, scale, s, pool_short)
            cand, codes = hybrid_shortlist(sel_tree, pshort, bres)
            pkw = {"prior_scale": _prior_extend(prior_scale, pshort.shape[-1])} if prior_scale is not None else {}
            Tr = rerank_top(cand, pq, qpos, lam, sigv, *tabs, scale, s, read_top, score=rerank_score, tokK=tokK,
                            rr_w=rr_w, prov=codes, scope=(codes & PROV_TREE) != 0, **pkw)[0]
        if bres is None:
            out = quota_merge(plist, Tr, pool_quota, read_top)
        else:
            out = beam_quota_merge(bres, Tr, read_top, plist, pool_quota)
    else:
        raise ValueError(f"fsb_hybrid {mode!r}")
    if prov_out is not None:
        prov_out.append(hybrid_prov(out, sel_tree, plist, bres))
    return out


def _det_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    from .det_reduce import det_sum
    return det_sum(x, dim)


def _det_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _det_sum(a * b, -1)


def _det_norm(a: torch.Tensor) -> torch.Tensor:
    return _det_sum(a * a, -1).sqrt()


@torch.no_grad()
def build_trees(Kbar: torch.Tensor, ceil_c: torch.Tensor, admit: torch.Tensor, L: int, spread=None) -> dict:
    if Kbar.is_cuda:
        from .fsb_build_fast import build_trees_fast, fast_build_ok
        if fast_build_ok(Kbar, L):
            return build_trees_fast(Kbar, ceil_c, admit, L, spread=spread)
    T, M, Dh = Kbar.shape
    dev = Kbar.device
    cap0 = 1 << L
    assert M <= cap0, (M, L)
    ids = torch.arange(M, device=dev)
    m0 = admit.sum(-1)
    key = torch.where(admit, ids[None, :], ids[None, :] + M)
    order = key.argsort(dim=-1)
    packed0 = torch.where(ids[None, :] < m0[:, None], order, torch.full_like(order, -1))
    slots = torch.full((T, cap0), -1, dtype=torch.long, device=dev)
    slots[:, :M] = packed0
    mu_l, rad_l, am_l, pm_l, cnt_l, packed_l = [], [], [], [], [], []
    for l in range(L + 1):
        nodes = 1 << l
        cap = cap0 >> l
        S = slots.view(T, nodes, cap)
        valid = S >= 0
        m = valid.sum(-1)
        Sc = S.clamp(min=0)
        X = torch.gather(Kbar, 1, Sc.view(T, -1)[..., None].expand(-1, -1, Dh)).view(T, nodes, cap, Dh)
        X = X * valid[..., None]
        mu = _det_sum(X, 2) / m.clamp(min=1)[..., None].float()
        Xc = (X - mu[:, :, None, :]) * valid[..., None]
        d2 = _det_sum(Xc * Xc, -1)
        if spread is None:
            rad = d2.amax(-1).sqrt()
        else:
            sp = torch.gather(spread, 1, Sc.view(T, -1)).view(T, nodes, cap)
            rad = (d2.sqrt() + sp).masked_fill(~valid, 0.0).amax(-1)
        ce = torch.gather(ceil_c, 1, Sc.view(T, -1)).view(T, nodes, cap)
        amax = ce.masked_fill(~valid, NEG).amax(-1).clamp(min=0.0)
        pmin = S.masked_fill(~valid, M).amin(-1)
        mu_l.append(mu); rad_l.append(rad); am_l.append(amax); pm_l.append(pmin); cnt_l.append(m)
        packed_l.append(slots.clone())
        if l == L:
            break
        far = d2.masked_fill(~valid, -1.0).argmax(-1)
        v = torch.gather(Xc, 2, far[..., None, None].expand(-1, -1, 1, Dh)).squeeze(2)
        v = v / _det_norm(v)[..., None].clamp_min(1e-6)
        for _ in range(2):
            pv = _det_sum(Xc * v[:, :, None, :], -1)
            v = _det_sum(Xc * pv[..., None], 2)
            v = v / _det_norm(v)[..., None].clamp_min(1e-6)
        proj = _det_sum(Xc * v[:, :, None, :], -1).masked_fill(~valid, float("inf"))
        order = proj.argsort(dim=-1, stable=True)
        Ss = torch.gather(S, 2, order)
        vs = Ss >= 0
        i = torch.arange(cap, device=dev).view(1, 1, cap)
        nleft = ((m + 1) // 2)[..., None]
        half = cap // 2
        dest = torch.where(i < nleft, i.expand(T, nodes, cap), half + i - nleft)
        dest = torch.where(vs, dest, torch.full_like(dest, cap))
        new = torch.full((T, nodes, cap + 1), -1, dtype=torch.long, device=dev)
        new.scatter_(2, dest, Ss)
        slots = new[:, :, :cap].reshape(T, cap0)
    height = torch.log2(m0.clamp(min=1).float()).ceil().long()
    return {"L": L, "M": M, "mu": mu_l, "rad": rad_l, "amax": am_l, "pmin": pm_l, "cnt": cnt_l,
            "packed": torch.cat(packed_l, dim=1), "m0": m0, "height": height}


@torch.no_grad()
def traverse(tr: dict, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor, sigv: torch.Tensor,
             scale: float, s: int, W: torch.Tensor, D: torch.Tensor, W_max: int, delta_max: int,
             node_score: str = "bound", radius_coef: float = 1.0, read_top: int = 0, cell_tabs=None,
             rerank_score: str = "leaf", tokK=None, rq=None, union_width: int = 0, union_W_max: int = 0, rr_w=None,
             read_margin: float = 0.0, read_min: int = 1):
    sel, n_visit = _rebuild_beam(tr, q, qpos, lam, sigv, scale, s, W, D, W_max, delta_max, node_score, radius_coef)
    if union_width > 0:
        Wu = torch.minimum(torch.full_like(D, union_width), torch.bitwise_left_shift(torch.ones_like(D), D))
        sel_u, nv_u = _rebuild_beam(tr, q, qpos, lam, sigv, scale, s, Wu, D, union_W_max, delta_max, node_score,
                                    radius_coef)
        sel = union_candidates(sel, sel_u)
        n_visit = n_visit + nv_u
    n_read = (sel >= 0).sum(-1)
    if read_top > 0:
        _rq = q if rq is None else rq
        if rr_w is None and read_margin <= 0:
            from .fsb_fast_step import fast_rerank_ok, rerank_top_fast
            if fast_rerank_ok(sel, _rq, tokK, s, read_top, rerank_score):
                sel = rerank_top_fast(sel, _rq, qpos, lam, sigv, cell_tabs[1], cell_tabs[2], scale, s, read_top,
                                      rerank_score, tokK)
                n_read = (sel >= 0).sum(-1)
                return sel, n_visit, n_read
        sel = rerank_top(sel, _rq, qpos, lam, sigv, cell_tabs[0], cell_tabs[1], cell_tabs[2], scale, s, read_top,
                         score=rerank_score, tokK=tokK, **({"rr_w": rr_w} if rr_w is not None else {}),
                         **({"read_margin": read_margin, "read_min": read_min} if read_margin > 0 else {}))
        n_read = (sel >= 0).sum(-1)
    return sel, n_visit, n_read


def union_candidates(sel_a: torch.Tensor, sel_b: torch.Tensor) -> torch.Tensor:
    dup = (sel_b[..., :, None] == sel_a[..., None, :]).any(-1)
    return torch.cat([sel_a, sel_b.masked_fill(dup, -1)], dim=-1)


def union_kw(attn) -> dict:
    return {"union_width": int(getattr(attn, "fsb_union_width", 0) or 0),
            "union_W_max": int(getattr(attn, "fsb_union_W_max", 0) or 0)}


def compact_slots_ok(W_max: int, union_W_max: int, delta_eff: int, read_top: int) -> bool:
    # the compact layout's candidate slots, (W_max + union_W_max) * 2^delta_eff (union_W_max = 0 without a union
    # beam), hold the whole read: rerank_top, its read margin and union_candidates' re-rank need read_top of them
    # (fsb_init only guarantees read_top <= W_max * 2^delta_max)
    return (int(W_max) + int(union_W_max)) << int(delta_eff) >= int(read_top)


def beam_delta_eff(attn):
    # traverse_forest's delta_eff for the eager forest prefill of a fixed-policy module (delta = min(height,
    # fsb_delta) for both beams) whose candidates go to a re-rank that reads only their slot order: the beams then
    # emit 2^fsb_delta cell slots per frontier slot instead of 2^fsb_delta_max. None = the full layout (any other
    # configuration, fewer compact slots than fsb_read_top, FSB_BEAM_COMPACT=0, or under torch.compile)
    if (compiling() or getattr(attn, "fsb_policy_mode", "fixed") != "fixed"
            or getattr(attn, "fsb_tree", "rebuild") != "forest"
            or int(getattr(attn, "fsb_read_top", 0) or 0) <= 0 or hybrid_on(attn) or beam_kw(attn)
            or union_quota_on(attn) or rr_kw(attn) or union_prior_on(attn)
            or os.environ.get("FSB_BEAM_COMPACT", "1") != "1"):
        return None
    d = max(0, min(int(attn.fsb_delta), int(attn.fsb_delta_max)))
    uw = int(getattr(attn, "fsb_union_W_max", 0) or 0) if int(getattr(attn, "fsb_union_width", 0) or 0) > 0 else 0
    if not compact_slots_ok(attn.fsb_W_max, uw, d, attn.fsb_read_top):
        return None
    return d


def rr_kw(attn) -> dict:
    if getattr(attn, "fsb_rerank_rr", "none") == "none":
        return {}
    return {"rr_w": attn.fsb_route_rr_w}


def margin_kw(attn) -> dict:
    if float(getattr(attn, "fsb_read_margin", 0.0) or 0.0) <= 0:
        return {}
    return {"read_margin": float(attn.fsb_read_margin), "read_min": int(getattr(attn, "fsb_read_min", 1))}


def _margin_trim(out: torch.Tensor, top_v: torch.Tensor, K: int, read_margin: float, read_min: int) -> torch.Tensor:
    if read_margin <= 0:
        return out
    tv = top_v[..., :K]
    keep = (tv >= tv[..., :1] - read_margin) | (torch.arange(K, device=out.device) < read_min)
    return out.masked_fill(~keep, -1)


@torch.no_grad()
def _rebuild_beam(tr: dict, q, qpos, lam, sigv, scale: float, s: int, W, D, W_max: int, delta_max: int,
                  node_score: str, radius_coef):
    T, Bq, Dh = q.shape
    L = tr["L"]
    dev = q.device
    root_ok = tr["cnt"][0][:, 0] > 0
    beam = torch.full((T, Bq, W_max), -1, dtype=torch.long, device=dev)
    beam[:, :, 0] = torch.where(root_ok, torch.zeros_like(root_ok, dtype=torch.long),
                                torch.full_like(root_ok, -1, dtype=torch.long))[:, None]
    n_visit = torch.zeros(T, Bq, device=dev)
    qn = _det_norm(q)[..., None]
    if torch.is_tensor(radius_coef):
        qn = qn * radius_coef.view(-1, 1, 1)
    elif radius_coef != 1.0:
        qn = qn * radius_coef
    two = torch.tensor([0, 1], device=dev)
    slot = torch.arange(W_max, device=dev).view(1, 1, W_max)
    for l in range(L):
        active = D > l
        ch = (beam[..., None] * 2 + two).view(T, Bq, 2 * W_max)
        ch_ok = beam.repeat_interleave(2, dim=-1) >= 0
        chc = ch.clamp(min=0)
        idx = chc.view(T, Bq * 2 * W_max)
        c_cnt = torch.gather(tr["cnt"][l + 1], 1, idx).view(T, Bq, 2 * W_max)
        ok = ch_ok & (c_cnt > 0)
        mu_c = torch.gather(tr["mu"][l + 1], 1, idx[..., None].expand(-1, -1, Dh)).view(T, Bq, 2 * W_max, Dh)
        content = _det_dot(q[:, :, None, :], mu_c) * scale
        if node_score == "bound":
            rad_c = torch.gather(tr["rad"][l + 1], 1, idx).view(T, Bq, 2 * W_max)
            content = content + qn * rad_c * scale
        pmin_pos = (torch.gather(tr["pmin"][l + 1], 1, idx).view(T, Bq, 2 * W_max).float() + 0.5) * s
        erf = torch.erf((qpos[..., None] - pmin_pos).abs() / (2.0 * sigv))
        am_c = torch.gather(tr["amax"][l + 1], 1, idx).view(T, Bq, 2 * W_max)
        score = (content + lam * am_c * erf).masked_fill(~ok, NEG)
        top = score.topk(W_max, dim=-1)
        nb_ = torch.gather(ch, -1, top.indices)
        slot_ok = (top.values > NEG / 2) & (slot < W[..., None])
        nb_ = nb_.masked_fill(~slot_ok, -1)
        n_visit = n_visit + active.float() * ok.sum(-1).float()
        beam = torch.where(active[..., None], nb_, beam)
    R = 1 << delta_max
    cap0 = 1 << L
    capD = torch.bitwise_left_shift(torch.ones_like(D), (L - D).clamp(min=0))
    start = (D * cap0)[..., None] + beam.clamp(min=0) * capD[..., None]
    t = torch.arange(R, device=dev)
    pos = start[..., None] + t
    valid = (beam[..., None] >= 0) & (t < capD[..., None, None])
    packed = tr["packed"]
    cells = torch.gather(packed, 1, pos.view(T, -1).clamp(max=packed.shape[1] - 1)).view(T, Bq, W_max, R)
    cells = cells.masked_fill(~valid, -1)
    return cells.view(T, Bq, W_max * R), n_visit


@torch.no_grad()
def rerank_top(sel: torch.Tensor, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor, sigv: torch.Tensor,
               cellK: torch.Tensor, cellA: torch.Tensor, t_idx: torch.Tensor, scale: float, s: int, K: int,
               score: str = "leaf", tokK=None, rr_w=None, prov=None, scope=None, read_margin: float = 0.0,
               read_min: int = 1, prior_scale=None):
    Tp, Bq, C = sel.shape
    Mc, Dh = cellK.shape[1], cellK.shape[2]
    ok = sel >= 0
    gid = t_idx.view(Tp, 1, 1) * Mc + sel.clamp(min=0, max=Mc - 1)
    if rr_w is not None:
        from .fsb_rerank import rr_inputs, rr_features, rr_residual
        assert score == "lse_pos" and tokK is not None, "the residual re-ranker needs the lse_pos token re-rank"
        inp = rr_inputs(sel, q, qpos, lam, sigv, cellK, cellA, t_idx, scale, s, tokK)
        pos = inp["pos"] if prior_scale is None else inp["pos"] * prior_scale
        u = (inp["lse"] + pos).masked_fill(~ok, float("-inf"))
        r = rr_residual(rr_features(inp, sel, qpos, s, prov), rr_w, t_idx)
        u = u + r.masked_fill(~ok, 0.0)
        if scope is not None:
            u = u.masked_fill(~scope, float("-inf"))
        top_v, top_i = torch.sort(u, dim=-1, descending=True, stable=True)
        out = torch.gather(sel, -1, top_i[..., :K])
        out = _margin_trim(out.masked_fill(top_v[..., :K] == float("-inf"), -1), top_v, K, read_margin, read_min)
        if prov is None:
            return out
        return out, torch.gather(prov, -1, top_i[..., :K]).masked_fill(out < 0, -1)
    assert scope is None, "rerank_top: scope is a residual re-ranker option (rr_w)"
    if score == "leaf":
        kb = cellK.reshape(-1, Dh)[gid.reshape(-1)].view(Tp, Bq, C, Dh)
        a = cellA.reshape(-1)[gid.reshape(-1)].view(Tp, Bq, C)
        content = _det_dot(q[:, :, None, :], kb) * scale
    else:
        from .fsb_rerank import cell_token_scores
        assert tokK is not None, "token-level re-rank needs the raw token keys (tokK)"
        content = cell_token_scores(q, sel, tokK, t_idx, scale, s, score)
        a = cellA.reshape(-1)[gid.reshape(-1)].view(Tp, Bq, C) if score == "lse_pos" else None
    if score in ("lse", "max"):
        u = content.masked_fill(~ok, float("-inf"))
    else:
        pos_c = (sel.float() + 0.5) * s
        erf = torch.erf((qpos[..., None] - pos_c).abs() / (2.0 * sigv))
        pt = lam * a * erf
        if prior_scale is not None:
            pt = pt * prior_scale
        u = (content + pt).masked_fill(~ok, float("-inf"))
    top_v, top_i = torch.sort(u, dim=-1, descending=True, stable=True)
    out = torch.gather(sel, -1, top_i[..., :K])
    out = _margin_trim(out.masked_fill(top_v[..., :K] == float("-inf"), -1), top_v, K, read_margin, read_min)
    if prov is None:
        return out
    return out, torch.gather(prov, -1, top_i[..., :K]).masked_fill(out < 0, -1)


# packed node attribute layout (fp32): [mu (Dh) | rad | amax | pmin | cnt | child_l | child_r | src | off | len]
_NA = 9


def _pack_nodes(mu, rad, amax, pmin, cnt, chl, chr_, src, off, ln):
    return torch.cat([mu, torch.stack([rad, amax, pmin.float(), cnt.float(), chl.float(), chr_.float(),
                                       src.float(), off.float(), ln.float()], dim=-1)], dim=-1)


def _grouped_chunk_build(Kbar, ceil_c, adm, m_tot, ks, spread=None):
    out = {}
    for grp in _grouped_chunk_build_iter(Kbar, ceil_c, adm, m_tot, ks, spread=spread):
        out.update(grp)
    return out


def _grouped_chunk_build_iter(Kbar, ceil_c, adm, m_tot, ks, spread=None):
    T, M, Dh = Kbar.shape
    dev = Kbar.device
    g = int(os.environ.get("FSB_BUILD_GROUP", "4"))
    if g <= 0:
        groups = [(ks, max(ks))]
    else:
        groups = {}
        for k in ks:
            groups.setdefault(k // g, []).append(k)
        groups = [(v, max(v)) for _, v in sorted(groups.items())]
    for group, Kpad in groups:
        cap = 1 << Kpad
        Kb_l, ce_l, ok_l, sp_l, meta = [], [], [], [], []
        r0 = 0
        for k in group:
            size = 1 << k
            Cmax = M >> k
            pos = torch.arange(Cmax * size, device=dev)
            cid = torch.gather(adm, 1, pos[None, :].expand(T, -1))
            ok = (cid >= 0) & (pos[None, :] < ((m_tot >> k) << k)[:, None])
            cid = cid.masked_fill(~ok, -1).view(T, Cmax, size)
            rows = T * Cmax
            Kb = torch.gather(Kbar, 1, cid.clamp(min=0).view(T, -1)[..., None].expand(-1, -1, Dh)).reshape(rows, size, Dh)
            ce = torch.gather(ceil_c, 1, cid.clamp(min=0).view(T, -1)).reshape(rows, size)
            sp = torch.gather(spread, 1, cid.clamp(min=0).view(T, -1)).reshape(rows, size) if spread is not None else None
            okr = (cid >= 0).reshape(rows, size)
            if size < cap:
                Kb = torch.cat([Kb, Kb.new_zeros(rows, cap - size, Dh)], 1)
                ce = torch.cat([ce, ce.new_zeros(rows, cap - size)], 1)
                okr = torch.cat([okr, torch.zeros(rows, cap - size, dtype=torch.bool, device=dev)], 1)
                if sp is not None:
                    sp = torch.cat([sp, sp.new_zeros(rows, cap - size)], 1)
            Kb_l.append(Kb); ce_l.append(ce); ok_l.append(okr); sp_l.append(sp); meta.append((k, slice(r0, r0 + rows), Cmax, cid)); r0 += rows
        Kb_c, ce_c, ok_c = torch.cat(Kb_l, 0), torch.cat(ce_l, 0), torch.cat(ok_l, 0)
        sp_c = torch.cat(sp_l, 0) if spread is not None else None
        del Kb_l, ce_l, ok_l, sp_l
        tr = build_trees(Kb_c, ce_c, ok_c, Kpad, spread=sp_c)
        del Kb_c, ce_c, ok_c, sp_c
        yield {k: (tr, rs, Cmax, cid, Kpad) for k, rs, Cmax, cid in meta}
        del tr


@torch.no_grad()
def build_forest(Kbar: torch.Tensor, ceil_c: torch.Tensor, admit: torch.Tensor, spread=None) -> dict:
    T, M, Dh = Kbar.shape
    dev = Kbar.device
    ids = torch.arange(M, device=dev)
    m_tot = admit.sum(-1)
    key = torch.where(admit, ids[None, :], ids[None, :] + M)
    order = key.argsort(dim=-1)
    adm = torch.where(ids[None, :] < m_tot[:, None], order, torch.full_like(order, -1))
    F = {"adm": adm, "m_tot": m_tot, "M": M, "T": T, "Dh": Dh, "ks": [], "chunks": {}}
    Kmax = int(math.floor(math.log2(max(M, 1))))
    F["Kmax"] = Kmax
    ks = [k for k in range(Kmax + 1) if (M >> k) > 0]
    if ks:
        for grp in _grouped_chunk_build_iter(Kbar, ceil_c, adm, m_tot, ks, spread=spread):
            for k, (tr, rs, Cmax, cid, Kpad) in grp.items():
                F["chunks"][k] = _chunk_tables(tr, k, T, Cmax, cid, rs, Kpad)
            del grp, tr
        F["ks"] = list(ks)
    _flatten_forest(F)
    return F


_STATIC_CACHE = {}


def _static_chunk(k: int, Cmax: int, T: int, nbase: int, lbase: int, dev):
    key = ("chunk", k, Cmax, T, nbase, lbase, str(dev))
    hit = _STATIC_CACHE.get(key)
    if hit is not None:
        return hit
    size = 1 << k; nodes = 2 * size - 1
    n = torch.arange(nodes, device=dev)
    lvl = torch.log2((n + 1).float()).floor().long()
    jj = n + 1 - (1 << lvl)
    nlen = torch.bitwise_left_shift(torch.ones_like(lvl), k - lvl)
    noff = jj * nlen
    internal = n < size - 1
    j = torch.arange(Cmax, device=dev)
    g = nbase + j[:, None] * nodes
    chl = torch.where(internal[None], g + 2 * n[None] + 1, torch.full((Cmax, nodes), -1, device=dev))
    chr_ = torch.where(internal[None], g + 2 * n[None] + 2, torch.full((Cmax, nodes), -1, device=dev))
    off = lbase + j[:, None] * size + noff[None]
    blk = torch.stack([chl.float(), chr_.float(), torch.ones(Cmax, nodes, device=dev), off.float(),
                       nlen[None].expand(Cmax, -1).float()], dim=-1)
    hit = {"static": blk, "cnt": nlen.float()}
    if len(_STATIC_CACHE) > 512:
        _STATIC_CACHE.clear()
    _STATIC_CACHE[key] = hit
    return hit


def _chunk_tables(tr: dict, k: int, T: int, Cmax: int, cid, rows=slice(None), Kpad=None) -> dict:
    size = 1 << k
    Kpad = k if Kpad is None else Kpad
    cap0 = 1 << Kpad
    Dh = tr["mu"][0].shape[-1]
    mu = torch.cat([tr["mu"][l][rows] for l in range(k + 1)], dim=1).view(T, Cmax, -1, Dh)
    rad = torch.cat([tr["rad"][l][rows] for l in range(k + 1)], dim=1).view(T, Cmax, -1)
    amax = torch.cat([tr["amax"][l][rows] for l in range(k + 1)], dim=1).view(T, Cmax, -1)
    cidf = cid.reshape(T * Cmax, size)
    segL = tr["packed"][rows][:, k * cap0:(k + 1) * cap0].view(-1, size, cap0 >> k)[:, :, 0]
    leafg = torch.gather(cidf, 1, segL.clamp(min=0)).masked_fill(segL < 0, 1 << 30)
    pmin = torch.cat([leafg.view(-1, 1 << l, size >> l).amin(-1) for l in range(k + 1)], dim=1).view(T, Cmax, -1)
    leaf = leafg.masked_fill(leafg >= (1 << 30), -1).view(T, Cmax, size)
    return {"mu": mu, "rad": rad, "amax": amax, "pmin": pmin, "leaf": leaf, "Cmax": Cmax, "k": k}


def _flatten_forest(F: dict):
    T, Dh = F["T"], F["Dh"]
    dev = F["adm"].device
    packs, leafs = [], []
    bases, nodes_per, leafbases, Cmaxs = {}, {}, {}, {}
    nbase = 0; lbase = 0
    for k in F["ks"]:
        C = F["chunks"][k]; Cmax = C["Cmax"]; size = 1 << k; nodes = 2 * size - 1
        bases[k] = nbase; nodes_per[k] = nodes; leafbases[k] = lbase; Cmaxs[k] = Cmax
        st = _static_chunk(k, Cmax, T, nbase, lbase, dev)
        dyn = torch.stack([C["rad"], C["amax"], C["pmin"].float(), st["cnt"].view(1, 1, nodes).expand(T, Cmax, nodes)], dim=-1)
        pk = torch.cat([C["mu"], dyn, st["static"][None].expand(T, -1, -1, -1)], dim=-1)
        packs.append(pk.reshape(T, Cmax * nodes, Dh + _NA)); leafs.append(C["leaf"].reshape(T, -1))
        nbase += Cmax * nodes; lbase += Cmax * size
    if F["ks"]:
        F["node"] = torch.cat(packs, 1) if len(packs) > 1 else packs[0]
        F["leaf"] = torch.cat(leafs, 1) if len(leafs) > 1 else leafs[0]
    else:
        F["node"] = torch.zeros(T, 1, Dh + _NA, device=dev); F["node"][:, :, Dh + 4:Dh + 6] = -1
        F["leaf"] = torch.full((T, 1), -1, dtype=torch.long, device=dev)
    F.pop("node_mu", None); F.pop("node_aux", None)          # the packed rebuild supersedes cached beam views
    F["Ntot"] = F["node"].shape[1]; F["Ltot"] = F["leaf"].shape[1]
    F["bases"] = bases; F["nodes_per"] = nodes_per; F["leafbases"] = leafbases; F["Cmaxs"] = Cmaxs


def _beam_views(F: dict):
    # the beams' node tables: mu in bf16 (their single source of truth for the means) and the 9 packed aux fields
    # in fp32. A packed fp32 F (build_forest's) converts once and caches the pair on itself, with each radius
    # inflated by the bf16 rounding error of its mean so q.mu + rad * |q| stays a valid bound
    mu = F.get("node_mu")
    if mu is None:
        node = F["node"]
        Dh = F["Dh"]
        src = node[..., :Dh]
        mu = torch.empty(src.shape, dtype=torch.bfloat16, device=node.device)
        mu.copy_(src)
        aux = node[..., Dh:].contiguous()                 # a slice copy: already detached from node
        eps = torch.empty(node.shape[:-1], dtype=torch.float32, device=node.device)
        for t in range(src.shape[0]):                     # row chunks keep the fp32 temporaries small
            eps[t] = (src[t] - mu[t].float()).norm(dim=-1)
        aux[..., 0] += eps
        F["node_mu"], F["node_aux"] = mu, aux
    return F["node_mu"], F["node_aux"]


@torch.no_grad()
def build_spines(F: dict, m_b: torch.Tensor, t_idx: torch.Tensor, out: dict | None = None) -> dict:
    dev = m_b.device
    Tp = m_b.shape[0]
    K = F["Kmax"]; Dh = F["Dh"]; Nt = F["Ntot"]
    S = K + 1
    ckey = ("spine", S, tuple(sorted(F["bases"].items())), tuple(sorted(F["nodes_per"].items())), str(dev))
    consts = _STATIC_CACHE.get(ckey)
    if consts is None:
        hk = torch.tensor([k in F["bases"] for k in range(S)], device=dev)
        consts = (torch.arange(S, device=dev),
                  torch.tensor([F["bases"].get(k, 0) for k in range(S)], device=dev),
                  torch.tensor([F["nodes_per"].get(k, 1) for k in range(S)], device=dev),
                  hk,
                  hk.to(torch.int8))
        _STATIC_CACHE[ckey] = consts
    if m_b.is_cuda:
        from .fsb_build_fast import build_spines_fast, fast_spines_ok
        if fast_spines_ok(F, m_b):
            return build_spines_fast(F, m_b, t_idx, consts, out=out)
    # a packed forest's fp32 rows stay exact here; only the split lean arena upcasts its bf16 means
    if F.get("node") is not None:
        node_flat = F["node"].reshape(-1, Dh + _NA)
    else:
        mu32, aux32 = _beam_views(F)
        node_flat = torch.cat([mu32.reshape(-1, Dh).float(), aux32.reshape(-1, _NA)], dim=-1)
    ks, base_t, npr_t, has_k = consts[:4]
    off_all = m_b[:, None] & ~((1 << (ks[None] + 1)) - 1)
    j_all = off_all >> ks[None]
    jb = F.get("jbase")
    if jb is not None:
        j_all = j_all - jb[t_idx]
    croot_all = base_t[None] + j_all * npr_t[None]
    present_all = ((m_b[:, None] >> ks[None]) & 1).bool() & has_k[None]
    gid = t_idx[:, None] * Nt + croot_all.clamp(min=0, max=Nt - 1)
    C = node_flat[gid.reshape(-1)].view(Tp, S, Dh + _NA)
    sp = torch.zeros(Tp, S, Dh + _NA, device=dev)
    sp[:, :, Dh + 4:Dh + 6] = -1
    r = torch.zeros(Tp, Dh + 4, device=dev)
    r[:, Dh + 2] = float(1 << 30)
    r_ref = torch.full((Tp,), -1, dtype=torch.long, device=dev)
    for k in range(S):
        present = present_all[:, k]
        c = C[:, k, :Dh + 4]
        c_cnt = C[:, k, Dh + 3]; r_cnt = r[:, Dh + 3]
        mk = present & (r_cnt > 0)
        tot = (c_cnt + r_cnt).clamp(min=1)
        n_mu = (c_cnt[:, None] * c[:, :Dh] + r_cnt[:, None] * r[:, :Dh]) / tot[:, None]
        nrm = _det_norm(torch.stack([c[:, :Dh] - n_mu, r[:, :Dh] - n_mu], 0))
        n_rad = torch.maximum(c[:, Dh] + nrm[0], r[:, Dh] + nrm[1])
        n_am = torch.maximum(c[:, Dh + 1], r[:, Dh + 1]); n_pm = torch.minimum(c[:, Dh + 2], r[:, Dh + 2])
        n_cnt = c_cnt + r_cnt
        nvec = torch.cat([n_mu, torch.stack([n_rad, n_am, n_pm, n_cnt], -1)], -1)
        spk = torch.cat([nvec, torch.stack([croot_all[:, k].float(), r_ref.float(), torch.full_like(n_rad, 2.0),
                                            off_all[:, k].float(), n_cnt], -1)], -1)
        sp[:, k] = torch.where(mk[:, None], spk, sp[:, k])
        cvec = c
        r = torch.where(mk[:, None], nvec, torch.where(present[:, None], cvec, r))
        r_ref = torch.where(mk, torch.full_like(r_ref, -(k + 2)), torch.where(present, croot_all[:, k], r_ref))
    root = torch.where(m_b > 0, r_ref, torch.full_like(r_ref, -1))
    k1 = torch.log2(m_b.clamp(min=1).float()).floor().long()
    pop = present_all.sum(-1)
    height = torch.where(m_b > 1, k1 + (pop >= 2).long(), torch.zeros_like(k1))
    return {"node": sp, "root": root, "height": height, "m_b": m_b, "t_idx": t_idx}


def _forest_gather(F: dict, SP: dict, ref: torch.Tensor, cols=None, tabs=None):
    Tp, N = ref.shape
    Dh = F["Dh"]; Nt = F["Ntot"]; Sp = SP["node"].shape[1]
    cols = slice(0, Dh + _NA) if cols is None else cols
    lo, hi = cols.start or 0, cols.stop
    # tabs: the caller's beam tables (bf16 means + inflated radii). Given, every beam backend scores the same
    # bits; absent, a packed forest's exact fp32 rows (references/tests) or the lean arena's upcast
    gid_c = SP["t_idx"][:, None] * Nt + ref.clamp(min=0, max=Nt - 1)
    gid = gid_c.reshape(-1)
    if tabs is None and F.get("node") is not None:
        Ac = F["node"].reshape(-1, Dh + _NA)[gid][:, lo:hi]
    else:
        mu32, aux32 = tabs if tabs is not None else _beam_views(F)
        Ac = torch.cat([mu32.reshape(-1, Dh)[gid].float(), aux32.reshape(-1, _NA)[gid]], dim=-1)[:, lo:hi]
    gid_s = torch.arange(Tp, device=ref.device)[:, None] * Sp + (-ref - 2).clamp(min=0, max=Sp - 1)
    As = SP["node"].reshape(-1, Dh + _NA)[gid_s.reshape(-1)][:, lo:hi]
    A = torch.where((ref >= 0).reshape(-1, 1), Ac, As).view(Tp, N, hi - lo)
    none = ref == -1
    A = A.masked_fill(none[..., None], 0.0)
    if lo <= Dh + 4 and hi >= Dh + 6:
        A[..., Dh + 4 - lo:Dh + 6 - lo] = A[..., Dh + 4 - lo:Dh + 6 - lo].masked_fill(none[..., None], -1.0)
    return A


def _forest_attrs(F: dict, SP: dict, ref: torch.Tensor):
    A = _forest_gather(F, SP, ref); Dh = F["Dh"]
    return {"mu": A[..., :Dh], "rad": A[..., Dh], "amax": A[..., Dh + 1], "pmin": A[..., Dh + 2].long(),
            "cnt": A[..., Dh + 3].long(), "child": A[..., Dh + 4:Dh + 6].long(), "src": A[..., Dh + 6].long(),
            "off": A[..., Dh + 7].long(), "len": A[..., Dh + 8].long()}


def _check_compact(cells, nvis, full, W_max: int, delta_max: int, delta_eff: int):
    # FSB_CHECK_COMPACT=1: the compact beam output (2^delta_eff slots per frontier slot) is the full layout's first
    # 2^delta_eff slots of every frontier slot, the rest of which are empty, with the same visit counts
    R, Re = 1 << delta_max, 1 << delta_eff
    f = full[0].view(*full[0].shape[:-1], W_max, R)
    assert torch.equal(f[..., :Re].reshape(cells.shape), cells), "FSB_CHECK_COMPACT: compact beam cells differ"
    assert bool((f[..., Re:] == -1).all()), "FSB_CHECK_COMPACT: a frontier node holds more than 2^delta_eff cells"
    assert torch.equal(full[1], nvis), "FSB_CHECK_COMPACT: visit counts differ"


def _union_width(SPu: dict, du: torch.Tensor, union_width: int) -> torch.Tensor:
    # the union beam's per-query width: union_width, capped by the subtree size at the read depth
    Du = (SPu["height"][:, None] - du).clamp(min=0)
    return torch.minimum(torch.full_like(Du, union_width), torch.bitwise_left_shift(torch.ones_like(Du), Du))


@torch.no_grad()
def traverse_forest(F: dict, SP: dict, q: torch.Tensor, qpos: torch.Tensor, lam: torch.Tensor,
                    sigv: torch.Tensor, scale: float, s: int, W: torch.Tensor, delta: torch.Tensor,
                    W_max: int, delta_max: int, node_score: str = "bound", radius_coef=1.0, trace=None,
                    read_top: int = 0, rerank_score: str = "leaf", tokK=None, rq=None, union_width: int = 0,
                    union_W_max: int = 0, rr_w=None, hybrid: str = "off", pool=None, pool_short: int = 4,
                    pool_quota: int = 1, prov_out=None, read_margin: float = 0.0, read_min: int = 1,
                    beam_quota: int = 0, union_F=None, union_SP=None, union_delta=None, union_quota: int = 0,
                    union_prior: float = 1.0, union_W=None, rows: bool = False, rows_ovf=None, delta_eff=None,
                    rows_q_bf16: bool = False):
    # rows: the caller allows the block-shared prefill re-rank (fsb_rows; numerically equivalent, not bit-identical):
    # eager inference prefill only, never inside a CUDA-graph stage, training or a traced re-traversal. rows_ovf: a
    # list that receives the rows re-rank's deferred capacity flag (fsb_rows.union_rerank_rows); when it reads True
    # the result is invalid and the caller must recompute it without the rows path (fsb_long_range recomputes the
    # layer with the per-stage check). rows_q_bf16: the caller guarantees that the re-rank query (rq, else q) holds
    # bf16 values, which the rows re-rank's LSE kernel then multiplies without a low part.
    # delta_eff: the caller guarantees delta <= delta_eff for both beams (the fixed policy: delta = min(height,
    # fsb_delta)); the beams then emit 2^delta_eff cell slots per frontier slot instead of 2^delta_max. Exact: every
    # final frontier node holds at most 2^delta cells (a node expanded at iteration i entered at iteration i - 1, so
    # its depth is i - 1; the descent runs Kmax + 1 iterations and nodes at depth Kmax + 1 are leaves), so the dropped
    # slots are always -1 and the valid cells keep their relative slot order, which is all the re-rank reads (ties go
    # to the first slot; provenance is slot >= Ca). Applied only where the candidates are re-ranked with no
    # slot-layout-dependent option and the compact slots hold read_top cells (every re-rank path returns read_top
    # columns only from at least as many slots); FSB_CHECK_COMPACT=1 also runs the full layout and asserts the equality
    mkw = {"read_margin": read_margin, "read_min": read_min} if read_margin > 0 else {}
    if (delta_eff is not None and 0 <= delta_eff < delta_max and read_top > 0 and hybrid == "off" and beam_quota == 0
            and union_quota == 0 and rr_w is None and union_prior == 1.0 and trace is None
            and compact_slots_ok(W_max, union_W_max if union_width > 0 else 0, delta_eff, read_top)):
        dm_beam = delta_eff
    else:
        dm_beam = delta_max
    if hybrid == "pool":
        assert beam_quota == 0, "fsb_hybrid pool runs no beam: fsb_beam_quota is refused at construction"
        assert union_F is None, "fsb_hybrid pool runs no beam: fsb_union_tau_quantile is refused at construction"
        assert union_quota == 0, "fsb_hybrid pool runs no beam: fsb_union_quota is refused at construction"
        Tp, Bq = q.shape[0], q.shape[1]
        sel = hybrid_select(None, pool, q if rq is None else rq, qpos, lam, sigv, F, SP, scale, s, read_top, rerank_score,
                            tokK, hybrid, pool_short, pool_quota, prov_out, **({"rr_w": rr_w} if rr_w is not None else {}),
                            **mkw)
        return sel, torch.zeros(Tp, Bq, device=q.device), (sel >= 0).sum(-1)
    chk = dm_beam != delta_max and os.environ.get("FSB_CHECK_COMPACT", "0") == "1"
    sel_unew = sel_u = pair = Wu = None
    if union_width > 0:
        if union_F is not None:
            Fu, SPu, du = union_F, union_SP, union_delta
        else:
            Fu, SPu, du = F, SP, delta
        from .fsb_fast_step import forest_beam_pair, forest_beam_pair_ok
        if trace is None and forest_beam_pair_ok(F, SP, Fu, SPu, q, W_max, union_W_max, dm_beam):
            Wu = _union_width(SPu, du, union_width) if union_W is None else union_W
            # both beams in one launch, in the layout the beams below would emit (2^dm_beam slots per frontier slot)
            pair = forest_beam_pair(F, SP, W, delta, W_max, Fu, SPu, Wu, du, union_W_max, q, qpos, lam, sigv, scale,
                                    s, dm_beam, node_score, radius_coef)
    if pair is not None:
        (sel, n_visit), (sel_u, nv_u) = pair
    else:
        sel, n_visit = _forest_beam(F, SP, q, qpos, lam, sigv, scale, s, W, delta, W_max, dm_beam, node_score,
                                    radius_coef, trace)
    if chk:
        _check_compact(sel, n_visit, _forest_beam(F, SP, q, qpos, lam, sigv, scale, s, W, delta, W_max, delta_max,
                                                  node_score, radius_coef, None), W_max, delta_max, dm_beam)
    bres = beam_reserved(sel, beam_quota) if beam_quota > 0 else None
    bkw = {"bres": bres} if bres is not None else {}
    if union_width > 0:
        if pair is None:
            if Wu is None:
                Wu = _union_width(SPu, du, union_width) if union_W is None else union_W
            sel_u, nv_u = _forest_beam(Fu, SPu, q, qpos, lam, sigv, scale, s, Wu, du, union_W_max, dm_beam,
                                       node_score, radius_coef, None, torch_loop=trace is not None)
        if chk:
            _check_compact(sel_u, nv_u, _forest_beam(Fu, SPu, q, qpos, lam, sigv, scale, s, Wu, du, union_W_max,
                                                     delta_max, node_score, radius_coef, None),
                           union_W_max, delta_max, dm_beam)
        n_visit = n_visit + nv_u
    if (rows and read_top > 0 and hybrid == "off" and bres is None and union_quota == 0 and rr_w is None and not mkw
            and union_prior == 1.0 and q.shape[1] > 1 and rerank_score != "leaf" and trace is None):
        from .fsb_rows import rows_ok, union_rerank_rows
        _rq = q if rq is None else rq
        if rows_ok(_rq, tokK, s, read_top, rerank_score):
            # batched prefill: block-shared union + re-rank (numerically equivalent, not bit-identical)
            res = union_rerank_rows(sel, sel_u, _rq, qpos, lam, sigv, F["cellA"], SP["t_idx"], scale, s, read_top,
                                    rerank_score, tokK, prov=(union_F is not None and prov_out is not None),
                                    ovf_out=rows_ovf, q_bf16=rows_q_bf16)
            if res is not None:
                if res[1] is not None:
                    prov_out.append(res[1])
                return res[0], n_visit, (res[0] >= 0).sum(-1)
    if sel_u is not None:
        Cm = sel.shape[-1]
        sel = union_candidates(sel, sel_u)
        sel_unew = sel[..., Cm:]
    pskw = {}
    if union_prior != 1.0:
        assert union_width > 0 and read_top > 0, "fsb_union_prior_scale needs a union beam and read_top (refused at construction)"
        pskw = {"prior_scale": union_prior_vec(Cm, sel.shape[-1] - Cm, union_prior, q.device)}
    ures = uqkw = None
    if union_quota > 0:
        assert union_width > 0 and read_top > 0 and sel_unew is not None and rr_w is None and not mkw, \
            "fsb_union_quota needs a union beam and read_top, no re-ranker and no read margin (refused at construction)"
        rqq = (q if rq is None else rq).float()
        tabs = (F["cellK"], F["cellA"], SP["t_idx"])
        Tu = rerank_top(sel_unew, rqq, qpos, lam, sigv, *tabs, scale, s, read_top, score=rerank_score, tokK=tokK,
                        **({"prior_scale": pskw["prior_scale"][..., Cm:]} if pskw else {}))
        Tm = rerank_top(sel[..., :Cm], rqq, qpos, lam, sigv, *tabs, scale, s, read_top, score=rerank_score, tokK=tokK)
        ures, urem = Tu[..., :union_quota], Tu[..., union_quota:]
        uqkw = {"tree_rank": union_quota_list(ures, Tm, urem)}
    if hybrid != "off":
        sel = hybrid_select(sel, pool, (q if rq is None else rq).float(), qpos, lam, sigv, F, SP, scale, s, read_top, rerank_score,
                            tokK, hybrid, pool_short, pool_quota, prov_out, **({"rr_w": rr_w} if rr_w is not None else {}),
                            **mkw, **bkw, **(uqkw or {}), **pskw)
    elif read_top > 0:
        if ures is not None:
            sel = union_quota_merge(ures, Tm, urem, read_top, bres=bres)
        else:
            _rq = q if rq is None else rq
            _fast = rr_w is None and not mkw and not pskw
            if _fast:
                from .fsb_fast_step import fast_rerank_ok, rerank_top_fast
                _fast = fast_rerank_ok(sel, _rq, tokK, s, read_top, rerank_score)
            if _fast:
                sel = rerank_top_fast(sel, _rq, qpos, lam, sigv, F["cellA"], SP["t_idx"], scale, s, read_top,
                                      rerank_score, tokK)
            else:
                sel = rerank_top(sel, _rq.float(), qpos, lam, sigv, F["cellK"], F["cellA"], SP["t_idx"], scale, s, read_top,
                                 score=rerank_score, tokK=tokK, **({"rr_w": rr_w} if rr_w is not None else {}),
                                 **mkw, **pskw)
            if bres is not None:
                sel = beam_quota_merge(bres, sel, read_top)
        if bres is not None and prov_out is not None:
            prov_out.append(beam_prov(sel, bres))
    if ures is not None and prov_out is not None:
        union_quota_prov_add(prov_out, sel, sel_unew, ures)
    elif union_F is not None and prov_out is not None:
        utau_prov_add(prov_out, sel, sel_unew)
    n_read = (sel >= 0).sum(-1)
    return sel, n_visit, n_read


@torch.no_grad()
def _forest_beam(F: dict, SP: dict, q, qpos, lam, sigv, scale: float, s: int, W, delta, W_max: int, delta_max: int,
                 node_score: str, radius_coef, trace, torch_loop: bool = False):
    Tp, Bq, Dh = q.shape
    dev = q.device
    use_kernel = q.is_cuda and os.environ.get("FSB_TRAVERSE_TRITON", "1") == "1" and trace is None and not torch_loop
    if use_kernel:
        from .fsb_traverse_triton import level_step, _HAVE_TRITON
        use_kernel = _HAVE_TRITON
    if use_kernel:
        from .fsb_fast_step import fast_beam_ok, forest_beam_fused
        if fast_beam_ok(F, SP, q, W_max, delta_max):
            return forest_beam_fused(F, SP, q, qpos, lam, sigv, scale, s, W, delta, W_max, delta_max,
                                     node_score, radius_coef)
    R = 1 << delta_max
    budget = torch.bitwise_left_shift(torch.ones_like(delta), delta)
    ref = torch.full((Tp, Bq, W_max), -1, dtype=torch.long, device=dev)
    ref[:, :, 0] = SP["root"][:, None]
    NC = Dh + _NA
    n_visit = torch.zeros(Tp, Bq, device=dev)
    bt = _beam_views(F)                                   # every beam backend scores the same table bits
    gath = lambda refs, cols=None: _forest_gather(F, SP, refs, cols, tabs=bt)
    if use_kernel:
        qn2 = _det_norm(q)
        if torch.is_tensor(radius_coef):
            qn2 = qn2 * radius_coef.view(-1, 1)
        elif radius_coef != 1.0:
            qn2 = qn2 * radius_coef
        Wl = W.contiguous(); bud = budget.contiguous()
        chunk_mu, chunk_ax = bt; spine = SP["node"]
        for _ in range(F["Kmax"] + 1):
            ref, nv = level_step(ref, q, qn2, qpos.contiguous(), lam, sigv, Wl, bud, SP["t_idx"], chunk_mu,
                                 chunk_ax, spine, scale, s, W_max, node_score)
            n_visit = n_visit + nv
        A = gath(ref.view(Tp, Bq * W_max)).view(Tp, Bq, W_max, NC)
    else:
        A = gath(ref.view(Tp, Bq * W_max)).view(Tp, Bq, W_max, NC)
    qn = _det_norm(q)[..., None]
    if torch.is_tensor(radius_coef):
        qn = qn * radius_coef.view(-1, 1, 1)
    elif radius_coef != 1.0:
        qn = qn * radius_coef
    slot = torch.arange(W_max, device=dev).view(1, 1, W_max)
    BIG = 1e30
    for _ in range(0 if use_kernel else F["Kmax"] + 1):
        active = ref != -1
        expand = active & (A[..., Dh + 3] > budget[..., None])
        carried = active & ~expand
        ch = A[..., Dh + 4:Dh + 6].long()
        chv = torch.where(expand[..., None], ch, torch.full_like(ch, -1)).view(Tp, Bq, 2 * W_max)
        CA = gath(chv.view(Tp, Bq * 2 * W_max)).view(Tp, Bq, 2 * W_max, NC)
        content = _det_dot(q[:, :, None, :], CA[..., :Dh]) * scale
        if node_score == "bound":
            content = content + qn * CA[..., Dh] * scale
        pmin_pos = (CA[..., Dh + 2] + 0.5) * s
        erf = torch.erf((qpos[..., None] - pmin_pos).abs() / (2.0 * sigv))
        score = (content + lam * CA[..., Dh + 1] * erf).masked_fill(chv == -1, NEG)
        cand_ref = torch.cat([ref.masked_fill(~carried, -1), chv], dim=-1)
        cand_sc = torch.cat([torch.where(carried, torch.full_like(score[..., :W_max], BIG), torch.full_like(score[..., :W_max], NEG)), score], dim=-1)
        top_v, top_i = torch.sort(cand_sc, dim=-1, descending=True, stable=True)
        top_v, top_i = top_v[..., :W_max], top_i[..., :W_max]
        slot_ok = (top_v > NEG / 2) & (slot < W[..., None])
        ref = torch.gather(cand_ref, -1, top_i).masked_fill(~slot_ok, -1)
        A = torch.gather(torch.cat([A, CA], dim=2), 2, top_i[..., None].expand(-1, -1, -1, NC))
        A = A.masked_fill(~slot_ok[..., None], 0.0)
        n_visit = n_visit + (chv != -1).sum(-1).float()
        if trace is not None:
            trace.append(ref.clone())
    src = A[..., Dh + 6].long(); off = A[..., Dh + 7].long(); ln = A[..., Dh + 8].long()
    t = torch.arange(R, device=dev)
    pos = off[..., None] + t
    valid = (src[..., None] > 0) & (t < ln[..., None]) & (ref[..., None] != -1)
    t_idx = SP["t_idx"]
    Lt = F["Ltot"]; Mm = F["M"]
    leaf_flat = F["leaf"].reshape(-1); adm_flat = F["adm"].reshape(-1)
    gl = (t_idx[:, None, None, None] * Lt + pos.clamp(min=0, max=Lt - 1)).view(Tp, -1)
    ga = (t_idx[:, None, None, None] * Mm + pos.clamp(min=0, max=Mm - 1)).view(Tp, -1)
    from_leaf = (src[..., None].expand(-1, -1, -1, R) == 1).reshape(Tp, -1)
    cells = torch.where(from_leaf, leaf_flat[gl], adm_flat[ga]).view(Tp, Bq, W_max, R)
    cells = cells.masked_fill(~valid, -1)
    return cells.view(Tp, Bq, W_max * R), n_visit


class ForestCache:
    """Streaming forest state with reusable buffers enabled by FSB_FAST_CACHE."""

    def __init__(self, T, Dh, device, lean=False):
        self.T, self.Dh, self.dev = T, Dh, device
        self.adm = torch.full((T, 0), -1, dtype=torch.long, device=device)
        self.m_tot = torch.zeros(T, dtype=torch.long, device=device)
        self.n_seen = 0
        self.chunks = {}
        self.F = None; self.blk_start = -1; self.SP = None
        self.pool = None
        self.utau = None
        self.live = False
        self.lean = bool(lean)
        self.fast = os.environ.get("FSB_FAST_CACHE", "1") == "1"
        self.m_cap = 0
        self.k_cap = 0
        self._lean_meta = {}
        self._arena_mu = None
        self._arena_aux = None
        self._leaf_arena = None
        self._arena_k = -1
        self._t_idx = None
        self._sp_stable = None
        self._cellA_g = None
        self._cells_seen = 0
        self._m_max_h = None
        self._height_h = None
        self._pad_buf = None
        self._pad_src = None
        self._zero_e = None
        self._emp_buf = None
        self._emp_k0_ptr = None
        self._mc_buf = None
        self._mc_nn = None
        self._pend_mc = None
        self._jbase_buf = None
        self._task_buf = None
        self._tau_src = None
        self._tau32 = None

    def _get_tau32(self, tau):
        key = (tau.data_ptr(), int(tau._version), tuple(tau.shape), tau.dtype)
        if self._tau_src != key or self._tau32 is None:
            from .fsb_build_cuda import _tau32
            self._tau32 = _tau32(tau).reshape(self.T).contiguous()
            self._tau_src = key
        return self._tau32

    @torch.no_grad()
    def _append_admitted(self, ceil_all, M_vis, tau):
        if self.fast and ceil_all.is_cuda and not torch.cuda.is_current_stream_capturing():
            from .fsb_build_fast import append_admitted_fast
            ret = append_admitted_fast(self, ceil_all, M_vis, self._get_tau32(tau))
            if ret is not None:
                return ret
        T = self.T
        nn = M_vis - self.n_seen
        rkey = ("arange_1d", nn, str(self.dev))
        r = _STATIC_CACHE.get(rkey)
        if r is None:
            r = torch.arange(nn, device=self.dev)[None]
            _STATIC_CACHE[rkey] = r
        ce = ceil_all[:, :, self.n_seen:M_vis].reshape(T, -1).float()
        ids = self.n_seen + r
        adm_new = ce >= tau.view(T, 1)
        n_new = adm_new.sum(-1)
        key = torch.where(adm_new, ids, ids + M_vis)
        order = key.argsort(-1)
        newseq = torch.where(r < n_new[:, None], order + self.n_seen, -1)
        cap = self.adm.shape[1]
        if self.fast and M_vis <= cap:
            grown = self.adm
        else:
            capacity = max(M_vis, self.m_cap, 2 * cap, 1) if self.fast else cap + nn
            grown = torch.full((T, capacity), -1, dtype=torch.long, device=self.dev)
            grown[:, :cap] = self.adm
        grown.scatter_(1, self.m_tot[:, None] + r, newseq)
        self.adm = grown; self.m_tot.add_(n_new); self.n_seen = M_vis
        return n_new

    def _ensure_lean_arena(self, Kmax):
        T, Dh, dev = self.T, self.Dh, self.dev
        cap_tasks = max(256, 16 * T) + 4
        if self._task_buf is None or self._task_buf.shape[0] < cap_tasks:
            self._task_buf = torch.empty((cap_tasks, 10), dtype=torch.int32, device=dev)
        if Kmax > self._arena_k:
            old_k = self._arena_k
            ncap = (1 << (Kmax + 2)) - Kmax - 3
            lcap = (1 << (Kmax + 1)) - 1
            node_mu = torch.zeros(T, ncap, Dh, dtype=torch.bfloat16, device=dev)
            node_aux = torch.zeros(T, ncap, _NA, device=dev)
            node_aux[..., 4:6] = -1
            leaf = torch.full((T, lcap), -1, dtype=torch.long, device=dev)
            if self._arena_mu is not None:
                node_mu[:, :self._arena_mu.shape[1]] = self._arena_mu
                node_aux[:, :self._arena_aux.shape[1]] = self._arena_aux
                leaf[:, :self._leaf_arena.shape[1]] = self._leaf_arena
            for k in range(old_k + 1, Kmax + 1):
                size = 1 << k
                nodes = 2 * size - 1
                nbase = 2 * size - k - 2
                lbase = size - 1
                st = _static_chunk(k, 1, T, nbase, lbase, dev)
                dst = node_aux[:, nbase:nbase + nodes]
                dst[..., 3] = st["cnt"]
                dst[..., 4:] = st["static"]
            self._arena_mu, self._arena_aux, self._leaf_arena, self._arena_k = node_mu, node_aux, leaf, Kmax

    def _get_emp_buf(self, Kbar_all):
        T, Dh, dev = self.T, self.Dh, self.dev
        emp = self._emp_buf
        k0_ptr = Kbar_all.data_ptr()
        if emp is None or self._emp_k0_ptr != k0_ptr or emp.shape != (T, 1, Dh + 3):
            emp = torch.zeros(T, 1, Dh + 3, device=dev)
            emp[..., :Dh] = torch.copysign(emp[..., :Dh], Kbar_all[:, :, 0].reshape(T, 1, Dh).float())
            emp[..., Dh + 2] = float(1 << 30)
            self._emp_buf = emp
            self._emp_k0_ptr = k0_ptr
        return emp

    def _pack_lean_cuda(self, F_cuda, Kbar_all, Kmax, ks):
        self._ensure_lean_arena(Kmax)
        emp = self._get_emp_buf(Kbar_all)
        ks_t = tuple(ks)
        ckey = ("pack_lean_fast", F_cuda["Ntot"], F_cuda["Ltot"], ks_t, str(self.dev))
        idx_c = _STATIC_CACHE.get(ckey)
        if idx_c is None:
            dev = self.dev
            k_n = torch.cat([torch.full((2 * (1 << k) - 1,), k, dtype=torch.int32, device=dev) for k in ks_t])
            base_n = torch.cat([torch.full((2 * (1 << k) - 1,), F_cuda["bases"][k], dtype=torch.int32, device=dev) for k in ks_t])
            str_n = torch.cat([torch.full((2 * (1 << k) - 1,), 2 * (1 << k) - 1, dtype=torch.int32, device=dev) for k in ks_t])
            u_n = torch.cat([torch.arange(2 * (1 << k) - 1, dtype=torch.int32, device=dev) for k in ks_t])
            dst_n = torch.cat([torch.arange(2 * (1 << k) - k - 2, 2 * (1 << k) - k - 2 + 2 * (1 << k) - 1, dtype=torch.int32, device=dev) for k in ks_t])
            k_l = torch.cat([torch.full((1 << k,), k, dtype=torch.int32, device=dev) for k in ks_t])
            base_l = torch.cat([torch.full((1 << k,), F_cuda["leafbases"][k], dtype=torch.int32, device=dev) for k in ks_t])
            str_l = torch.cat([torch.full((1 << k,), 1 << k, dtype=torch.int32, device=dev) for k in ks_t])
            u_l = torch.cat([torch.arange(1 << k, dtype=torch.int32, device=dev) for k in ks_t])
            dst_l = torch.cat([torch.arange((1 << k) - 1, (1 << k) - 1 + (1 << k), dtype=torch.int32, device=dev) for k in ks_t])
            idx_c = (k_n, base_n, str_n, u_n, dst_n, k_l, base_l, str_l, u_l, dst_l)
            _STATIC_CACHE[ckey] = idx_c
        from .fsb_build_fast import pack_lean_chunks_fast
        return pack_lean_chunks_fast(self, F_cuda, emp, Kmax, idx_c)

    def _store_arena_rows(self, sl, mu32, rad, amax, pmin):
        # packed fp32 rows -> the split arena: mu rounded to bf16 (the beams' single source of truth) and the
        # radius inflated by the rounding error, so q.mu + rad * |q| stays a valid bound of every member's score
        mu_w = mu32.float().to(torch.bfloat16)
        eps = (mu32.float() - mu_w.float()).norm(dim=-1)
        self._arena_mu[:, sl] = mu_w
        ax = self._arena_aux[:, sl]
        ax[..., 0] = rad + eps
        ax[..., 1] = amax
        ax[..., 2] = pmin

    def _pack_lean(self, F, dirty):
        Dh = self.Dh
        self._ensure_lean_arena(F["Kmax"])
        ks_t = tuple(F["ks"])
        lkey = ("lean_lay", ks_t)
        lay = _STATIC_CACHE.get(lkey)
        if lay is None:
            bases, leafbases, nodes_per, Cmaxs = {}, {}, {}, {}
            for k in ks_t:
                size = 1 << k
                nodes = 2 * size - 1
                bases[k] = 2 * size - k - 2
                leafbases[k] = size - 1
                nodes_per[k] = nodes
                Cmaxs[k] = 1
            lay = (bases, leafbases, nodes_per, Cmaxs)
            _STATIC_CACHE[lkey] = lay
        bases, leafbases, nodes_per, Cmaxs = lay
        contig = F["chunks"].get("_contig")
        if contig is not None:
            k0, row, lf = contig
            n0 = 2 * (1 << k0) - k0 - 2
            l0 = (1 << k0) - 1
            self._store_arena_rows(slice(n0, n0 + row.shape[1]), row[..., :Dh], row[..., Dh], row[..., Dh + 1],
                                   row[..., Dh + 2])
            self._leaf_arena[:, l0:l0 + lf.shape[1]] = lf
        else:
            for k in ks_t:
                if k not in dirty:
                    continue
                size = 1 << k
                nodes = 2 * size - 1
                nbase = 2 * size - k - 2
                lbase = size - 1
                C = F["chunks"][k]
                sl = slice(nbase, nbase + nodes)
                if "row" in C:
                    r = C["row"][:, 0]
                    self._store_arena_rows(sl, r[..., :Dh], r[..., Dh], r[..., Dh + 1], r[..., Dh + 2])
                else:
                    self._store_arena_rows(sl, C["mu"][:, 0], C["rad"][:, 0], C["amax"][:, 0], C["pmin"][:, 0])
                self._leaf_arena[:, lbase:lbase + size] = C["leaf"][:, 0]
        F.update(node_mu=self._arena_mu, node_aux=self._arena_aux, leaf=self._leaf_arena,
                 Ntot=self._arena_mu.shape[1], Ltot=self._leaf_arena.shape[1],
                 bases=bases, leafbases=leafbases, nodes_per=nodes_per,
                 Cmaxs=Cmaxs)

    def _t_idx_rows(self):
        # the spine builder's row sequence on one persistent tensor (a decode graph reads its pointer)
        if self._t_idx is None or self._t_idx.shape[0] != self.T:
            self._t_idx = torch.arange(self.T, device=self.dev)
        return self._t_idx

    def _stable_spines(self):
        # build_spines with the outputs written into the previous build's tensors when their shapes match: a
        # whole-step decode graph captured over the previous block's spines reads the same memory at the new values
        old = self._sp_stable
        if self.lean and (not getattr(self, "_grew", True) or getattr(self, "_sp_built", False)) and old is not None and self.SP is old:
            return self.SP
        try:
            sp = build_spines(self.F, self.m_tot, self._t_idx_rows(), out=old)
        except TypeError:
            sp = build_spines(self.F, self.m_tot, self._t_idx_rows())
        if sp is old:
            self.SP = old
        elif old is not None and old["node"].shape == sp["node"].shape:
            old["node"].copy_(sp["node"]); old["root"].copy_(sp["root"]); old["height"].copy_(sp["height"])
            old["m_b"] = self.m_tot
            self.SP = old
        else:
            self.SP = sp
            self._sp_stable = sp
        return self.SP

    def _cellA_stable(self, ceil_all, M_vis):
        # the rerank kernels' ceiling table on a capacity-strided buffer: they bake the tensor's pointer, strides
        # and leading shape into a decode graph's launches, so the graph survives a block change while only the
        # values grow (one completed cell at a time; cells < M_vis are the only ones ever selected). F["cellA"]
        # keeps its exact M_vis-sized contract; this buffer never replaces it outside the fused decode calls.
        T, dev = self.T, self.dev
        slot = self._cellA_g
        c_ptr = ceil_all.data_ptr()
        if slot is None or slot.shape[1] < M_vis:
            need = max(M_vis, self.m_cap)
            cap = max(64, 1 << (need - 1).bit_length()) if need > 0 else 64
            slot = torch.zeros(T, cap, dtype=torch.float32, device=dev)
            self._cellA_g = slot
            self._cells_seen = 0
        elif getattr(self, "_cellA_src_ptr", None) != c_ptr or M_vis < self._cells_seen:
            self._cells_seen = 0
        self._cellA_src_ptr = c_ptr
        if M_vis > self._cells_seen:
            n = M_vis - self._cells_seen
            if ceil_all.stride(0) == ceil_all.shape[1] * ceil_all.stride(1):
                c_src = ceil_all.reshape(T, -1)[:, self._cells_seen:M_vis]
            else:
                c_src = ceil_all[:, :, self._cells_seen:M_vis].reshape(T, n)
            slot[:, self._cells_seen:M_vis].copy_(c_src)
            self._cells_seen = M_vis
        return slot

    def _pad_cells(self, Kbar_all, ceil_all, M_vis, spread_all=None):
        owner = (self._pad_src() if callable(self._pad_src) else self._pad_src) if self._pad_src is not None else self
        if owner is None:
            owner = self
        T, Dh = self.T, self.Dh
        M_pad = max(512, 1 << (M_vis - 1).bit_length())
        buf = owner._pad_buf
        has_sp = spread_all is not None
        if buf is None or buf[0].shape[1] != M_pad or (has_sp and buf[2] is None) or buf[3] > M_vis:
            Kp = Kbar_all.new_empty((T, M_pad, Dh), dtype=torch.float32)
            cp = ceil_all.new_empty((T, M_pad), dtype=torch.float32)
            if M_pad > M_vis:
                cp[:, M_vis:].fill_(float("-inf"))
            sp = spread_all.new_empty((T, M_pad), dtype=torch.float32) if has_sp else None
            seen = 0
        else:
            Kp, cp, sp, seen = buf
        if M_vis > seen:
            n = M_vis - seen
            if Kbar_all.stride(0) == Kbar_all.shape[1] * Kbar_all.stride(1):
                k_src = Kbar_all.reshape(T, -1, Dh)[:, seen:M_vis]
            else:
                k_src = Kbar_all[:, :, seen:M_vis].reshape(T, n, Dh)
            Kp[:, seen:M_vis].copy_(k_src)
            if ceil_all.stride(0) == ceil_all.shape[1] * ceil_all.stride(1):
                c_src = ceil_all.reshape(T, -1)[:, seen:M_vis]
            else:
                c_src = ceil_all[:, :, seen:M_vis].reshape(T, n)
            cp[:, seen:M_vis].copy_(c_src)
            if has_sp:
                if spread_all.stride(0) == spread_all.shape[1] * spread_all.stride(1):
                    s_src = spread_all.reshape(T, -1)[:, seen:M_vis]
                else:
                    s_src = spread_all[:, :, seen:M_vis].reshape(T, n)
                sp[:, seen:M_vis].copy_(s_src)
            seen = M_vis
        owner._pad_buf = (Kp, cp, sp, seen)
        return Kp, cp, (sp if has_sp else None)

    def _cuda_forest(self, Kbar_all, ceil_all, M_vis, spread_all, taus, sel=None, lean=None):
        # the CUDA build of one ingest: the visible cells padded with never-admitted cells (ceilings -inf) to a
        # power-of-two M. sel = [(k, [j per head or None])] restricts the build to those chunks (a decode ingest's
        # dirty aligned chunks: the only rows _cuda_chunk_all reads); without it, or when the chunk build is not
        # covered, the whole forest is built. None -> the per-level build runs; never in capture.
        T, Dh, dev = self.T, self.Dh, self.dev
        if not Kbar_all.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        if (lean is not None and Kbar_all.dtype == torch.float32 and ceil_all.dtype == torch.float32
                and Kbar_all.stride(0) == Kbar_all.shape[1] * Kbar_all.stride(1)
                and Kbar_all.stride(2) == Dh and Kbar_all.stride(3) == 1 and Kbar_all.stride(1) % Dh == 0
                and ceil_all.stride(0) == ceil_all.shape[1] * ceil_all.stride(1) and ceil_all.stride(2) == 1
                and (spread_all is None or (spread_all.dtype == torch.float32
                     and spread_all.stride(0) == spread_all.shape[1] * spread_all.stride(1)
                     and spread_all.stride(2) == 1))):
            Kbar = Kbar_all.reshape(T, -1, Dh)[:, :M_vis]
            ceil_c = ceil_all.reshape(T, -1)[:, :M_vis]
            spread_c = spread_all.reshape(T, -1)[:, :M_vis] if spread_all is not None else None
        elif self.fast and self.lean:
            Kbar, ceil_c, spread_c = self._pad_cells(Kbar_all, ceil_all, M_vis, spread_all)
        else:
            M_pad = max(512, 1 << (M_vis - 1).bit_length())
            Kbar = Kbar_all[:, :, :M_vis].reshape(T, M_vis, Dh).float()
            ceil_c = ceil_all[:, :, :M_vis].reshape(T, M_vis).float()
            spread_c = spread_all[:, :, :M_vis].reshape(T, M_vis).float() if spread_all is not None else None
            if M_pad > M_vis:
                Kp, cp = Kbar.new_zeros(T, M_pad, Dh), ceil_c.new_full((T, M_pad), float("-inf"))
                Kp[:, :M_vis] = Kbar; cp[:, :M_vis] = ceil_c
                Kbar, ceil_c = Kp, cp
                if spread_c is not None:
                    sp = spread_c.new_zeros(T, M_pad)
                    sp[:, :M_vis] = spread_c
                    spread_c = sp
        if self._zero_e is None:
            self._zero_e = torch.zeros(1, dtype=torch.long, device=dev)
        from .fsb_build_cuda import forests_cuda
        kw = {"sel": sel, "lean": lean} if lean is not None else {"sel": sel}
        got = forests_cuda(Kbar, ceil_c, spread_c, taus, self._zero_e, **kw)
        return None if got is None else got[0][0]

    def _cuda_chunk_all(self, F_cuda, Kbar_all, jbase, bits, ks):
        # _cuda_chunk for every dirty level in one pair of gathers over the whole-forest build: the per-level rows
        # (aligned-chunk node rows at (k, j)) are concatenated across levels, gathered and masked once, then sliced
        # back per level — identical values to the per-level gathers, a fraction of the launches. Rows without the
        # chunk hold the reference build's all-empty chunk (signed zeros of cell 0's rows, pmin 2^30, leaves -1);
        # _pack_lean adds the static columns at the lean bases
        T, Dh, dev = self.T, self.Dh, self.dev
        ks_t = tuple(ks)
        ckey = ("chunk_all", F_cuda["Ntot"], F_cuda["Ltot"], ks_t, str(dev))
        idx_c = _STATIC_CACHE.get(ckey)
        if idx_c is None:
            k_n = torch.cat([torch.full((2 * (1 << k) - 1,), k, dtype=torch.long, device=dev) for k in ks_t])
            base_n = torch.cat([torch.full((2 * (1 << k) - 1,), F_cuda["bases"][k], dtype=torch.long, device=dev) for k in ks_t])[None]
            str_n = torch.cat([torch.full((2 * (1 << k) - 1,), 2 * (1 << k) - 1, dtype=torch.long, device=dev) for k in ks_t])[None]
            u_n = torch.cat([torch.arange(2 * (1 << k) - 1, device=dev) for k in ks_t])[None]
            k_l = torch.cat([torch.full((1 << k,), k, dtype=torch.long, device=dev) for k in ks_t])
            base_l = torch.cat([torch.full((1 << k,), F_cuda["leafbases"][k], dtype=torch.long, device=dev) for k in ks_t])[None]
            str_l = torch.cat([torch.full((1 << k,), 1 << k, dtype=torch.long, device=dev) for k in ks_t])[None]
            u_l = torch.cat([torch.arange(1 << k, device=dev) for k in ks_t])[None]
            idx_c = (k_n, base_n, str_n, u_n, k_l, base_l, str_l, u_l)
            _STATIC_CACHE[ckey] = idx_c
        k_n, base_n, str_n, u_n, k_l, base_l, str_l, u_l = idx_c
        nrs = base_n + jbase[:, k_n] * str_n + u_n
        lrs = base_l + jbase[:, k_l] * str_l + u_l
        hms = bits[:, k_n]
        hml = bits[:, k_l]
        node = F_cuda.get("node")
        if node is not None:
            row = node.gather(1, nrs[..., None].expand(-1, -1, Dh + 3))
        else:
            # the split whole-forest build: bf16 means upcast (the arena re-rounds to the same bits, its radius
            # slack passing through node_aux[0] untouched) alongside the fp32 aggregate columns
            row = torch.cat([F_cuda["node_mu"].gather(1, nrs[..., None].expand(-1, -1, Dh)).float(),
                             F_cuda["node_aux"].gather(1, nrs[..., None].expand(-1, -1, 3))], dim=-1)
        lf = F_cuda["leaf"].gather(1, lrs)
        emp = self._get_emp_buf(Kbar_all)
        row = torch.where(hms[..., None], row, emp)
        lf = torch.where(hml, lf, -1)
        if ks[-1] - ks[0] + 1 == len(ks):
            return {"_contig": (ks[0], row, lf)}
        chunks, no, lo = {}, 0, 0
        for k in ks:
            size = 1 << k; nodes = 2 * size - 1
            chunks[k] = {"row": row[:, no:no + nodes].unsqueeze(1),
                         "leaf": lf[:, lo:lo + size].unsqueeze(1), "Cmax": 1, "k": k}
            no += nodes; lo += size
        return chunks

    @torch.no_grad()
    def _ingest_lean(self, Kbar_all, ceil_all, M_vis, tau, spread_all=None):
        T, Dh, dev = self.T, self.Dh, self.dev
        pend_mc = self._pend_mc
        self._pend_mc = None
        if pend_mc is not None:
            nn_h, m_h = pend_mc
            nn = True
        else:
            nn = self._append_admitted(ceil_all, M_vis, tau) if M_vis > self.n_seen else None
            if self.F is not None and nn is None:
                self._grew = False
                return
            # one host read per ingest: the admitted counts and m_tot together give grew, Kmax, the level bits, the
            # aligned chunk indices and the reuse comparison, so the levels below never sync again
            if nn is not None and self._mc_buf is not None and nn is self._mc_nn:
                mc_l = self._mc_buf.cpu().tolist()
                nn_h, m_h = mc_l[0], mc_l[1]
            else:
                mc = (torch.stack([nn, self.m_tot]) if nn is not None else self.m_tot[None]).cpu()
                m_h = mc[-1].tolist()
                nn_h = mc[0].tolist() if nn is not None else None
        m = self.m_tot
        m_max = max(m_h)
        self._m_max_h = m_max
        self._height_h = tuple((mi - 1).bit_length() if mi > 1 else 0 for mi in m_h)
        grew = nn is not None and any(nn_h)
        if self.F is not None and not grew:
            self._grew = False
            return
        self._grew = True
        self._sp_built = False
        m_upper = m_max + max(0, self.m_cap - M_vis) if self.k_cap > 0 else m_max
        prev_k = self.F["Kmax"] if self.F is not None else 0
        Kmax = max(int(math.floor(math.log2(m_upper))), prev_k) if m_upper > 0 else prev_k
        bits_h = [[(mi >> k) & 1 for k in range(Kmax + 1)] for mi in m_h]
        jb_h = [[((mi >> (k + 1)) << 1) if b else 0 for k, b in enumerate(row)] for mi, row in zip(m_h, bits_h)]
        need = [any(row[k] for row in bits_h) for k in range(Kmax + 1)]
        old = self.F
        chunks, meta, dirty = {}, {}, set()
        Kbar = ceil_c = spread_c = None
        reuse, F_cuda = {}, None
        packed_cuda = False
        if self.fast:
            ks_n = [k for k in range(Kmax + 1) if need[k]]
            for k in ks_n:
                meta[k] = (tuple(row[k] for row in jb_h), tuple(bool(row[k]) for row in bits_h))
            if old is not None:
                reuse = {k: self._lean_meta.get(k) == meta[k] for k in ks_n if k in self._lean_meta}
            dset = [k for k in ks_n if not reuse.get(k, False)]
            if dset:
                # one partial build of this forest's own dirty chunks (the pair latch joined both taus' whole-forest
                # builds; the chunk build is cheap enough to run per forest, and each forest's aligned chunks differ
                # with its admission, so a shared selection would need both forests' post-ingest state up front)
                self._ensure_lean_arena(Kmax)
                if self._jbase_buf is None or self._jbase_buf.shape != (T, Kmax + 1):
                    self._jbase_buf = torch.empty((T, Kmax + 1), dtype=torch.long, device=dev)
                S_sp = Kmax + 1
                NC_sp = Dh + _NA
                sp_out = self._sp_stable
                if sp_out is None or sp_out["node"].shape != (T, S_sp, NC_sp):
                    sp_out = {
                        "node": torch.empty((T, S_sp, NC_sp), dtype=torch.float32, device=dev),
                        "root": torch.empty((T,), dtype=torch.long, device=dev),
                        "height": torch.empty((T,), dtype=torch.long, device=dev),
                        "m_b": self.m_tot,
                        "t_idx": self._t_idx_rows(),
                    }
                    self._sp_stable = sp_out
                else:
                    sp_out["m_b"] = self.m_tot
                    sp_out["t_idx"] = self._t_idx_rows()
                M_pad = max(512, 1 << (M_vis - 1).bit_length())
                lean_arg = (self._arena_mu, self._arena_aux, self._leaf_arena, self.adm, self.m_tot,
                            self._jbase_buf, Kmax, M_pad,
                            sp_out["node"], sp_out["root"], sp_out["height"], self._task_buf)
                sel = [(k, [row[k] if bts[k] else None for row, bts in zip(jb_h, bits_h)]) for k in dset]
                F_cuda = self._cuda_forest(Kbar_all, ceil_all, M_vis, spread_all, [self._get_tau32(tau)], sel=sel,
                                           lean=lean_arg)
            if dset and F_cuda is not None:
                if F_cuda.get("_lean_direct"):
                    jbase = F_cuda["jbase"]
                    if F_cuda.get("_sp_built"):
                        self.SP = self._sp_stable
                        self._sp_built = True
                else:
                    jbase = self._pack_lean_cuda(F_cuda, Kbar_all, Kmax, dset)
                if jbase is not None:
                    F_cuda = None
                    dset = []
                    packed_cuda = True
            if not packed_cuda:
                jb_bits = torch.tensor([jb_h, bits_h], dtype=torch.long, device=dev)
                jbase, bits = jb_bits[0], jb_bits[1].bool()
                if dset and F_cuda is not None:
                    chunks.update(self._cuda_chunk_all(F_cuda, Kbar_all, jbase, bits, dset))
                    F_cuda = None
                    dirty.update(dset)
                    dset = []
            ks_loop = dset
        else:
            jbase = torch.tensor(jb_h, dtype=torch.long, device=dev)
            bits = torch.tensor(bits_h, dtype=torch.bool, device=dev).t()
            ks_loop = range(Kmax + 1)
        for k in ks_loop:
            if not need[k]:
                continue
            size = 1 << k; nodes = 2 * size - 1
            has = bits[:, k] if self.fast else bits[k]
            if self.fast:
                j = jbase[:, k]
            else:
                j = torch.where(has, (m >> (k + 1)) << 1, torch.zeros_like(m))
                jbase[:, k] = j
                prev = self._lean_meta.get(k)
                if old is not None and prev is not None and torch.equal(prev[0], j) and torch.equal(prev[1], has):
                    b = old["bases"][k]; lb = old["leafbases"][k]
                    if old.get("node") is None:
                        # the fast stream's previous stage is the split arena: bf16 means, fp32 aux
                        ndm = old["node_mu"][:, b:b + nodes]; nda = old["node_aux"][:, b:b + nodes]
                        chunks[k] = {"mu": ndm.reshape(T, 1, nodes, Dh), "rad": nda[..., 0].reshape(T, 1, nodes),
                                     "amax": nda[..., 1].reshape(T, 1, nodes), "pmin": nda[..., 2].reshape(T, 1, nodes),
                                     "leaf": old["leaf"][:, lb:lb + size].reshape(T, 1, size), "Cmax": 1, "k": k}
                    else:
                        # the flattened (non-fast) stream keeps the previous stage as a packed fp32 table
                        nd = old["node"][:, b:b + nodes]
                        chunks[k] = {"mu": nd[..., :Dh].reshape(T, 1, nodes, Dh),
                                     "rad": nd[..., Dh].reshape(T, 1, nodes),
                                     "amax": nd[..., Dh + 1].reshape(T, 1, nodes),
                                     "pmin": nd[..., Dh + 2].reshape(T, 1, nodes),
                                     "leaf": old["leaf"][:, lb:lb + size].reshape(T, 1, size), "Cmax": 1, "k": k}
                    meta[k] = (j, has)
                    continue
                meta[k] = (j, has)
            if Kbar is None:
                Kbar = Kbar_all[:, :, :M_vis].reshape(T, M_vis, Dh).float()
                ceil_c = ceil_all[:, :, :M_vis].reshape(T, M_vis).float()
                spread_c = spread_all[:, :, :M_vis].reshape(T, M_vis).float() if spread_all is not None else None
            pos = j[:, None] * size + torch.arange(size, device=dev)[None]
            cid = torch.gather(self.adm, 1, pos.clamp(max=self.adm.shape[1] - 1))
            cid = torch.where(has[:, None], cid, torch.full_like(cid, -1))
            ok = cid >= 0
            Kb = torch.gather(Kbar, 1, cid.clamp(min=0)[..., None].expand(-1, -1, Dh))
            ce_ = torch.gather(ceil_c, 1, cid.clamp(min=0))
            sp_ = torch.gather(spread_c, 1, cid.clamp(min=0)) if spread_c is not None else None
            tr = build_trees(Kb, ce_, ok, k, spread=sp_)
            chunks[k] = _chunk_tables(tr, k, T, 1, cid.view(T, 1, size), slice(None), k)
            dirty.add(k)
            del tr
        F = {"adm": self.adm if self.adm.shape[1] > 0 else torch.full((T, 1), -1, dtype=torch.long, device=dev),
             "m_tot": m, "M": max(self.adm.shape[1], 1), "T": T, "Dh": Dh, "Kmax": Kmax,
             "ks": sorted(meta), "chunks": chunks}
        if self.fast:
            self._pack_lean(F, dirty)
        else:
            _flatten_forest(F)
        F["chunks"] = {}
        F["jbase"] = jbase
        self.F = F; self._lean_meta = meta

    @torch.no_grad()
    def ingest(self, Kbar_all, ceil_all, M_vis, tau, spread_all=None, keep_cells=False):
        if self.lean:
            self._ingest_lean(Kbar_all, ceil_all, M_vis, tau, spread_all)
            if keep_cells:
                T, Dh = self.T, self.Dh
                if self.fast and Kbar_all.stride(0) == Kbar_all.shape[1] * Kbar_all.stride(1) and Kbar_all.dtype == torch.float32:
                    self.F["cellK"] = Kbar_all.reshape(T, -1, Dh)[:, :M_vis]
                else:
                    self.F["cellK"] = Kbar_all[:, :, :M_vis].reshape(T, M_vis, Dh).float()
                if self.fast:
                    slot = self._cellA_stable(ceil_all, M_vis)
                    self.F["cellA"] = slot[:, :M_vis]
                else:
                    self.F["cellA"] = ceil_all[:, :, :M_vis].reshape(T, M_vis).float()
            return

        T, Dh = self.T, self.Dh
        if M_vis > self.n_seen:
            self._append_admitted(ceil_all, M_vis, tau)
        Kbar = Kbar_all[:, :, :M_vis].reshape(T, M_vis, Dh).float()
        ceil_c = ceil_all[:, :, :M_vis].reshape(T, M_vis).float()
        spread_c = spread_all[:, :, :M_vis].reshape(T, M_vis).float() if spread_all is not None else None
        if self.F is None:
            admit = torch.zeros(T, M_vis + 1, dtype=torch.bool, device=self.dev)
            admit.scatter_(1, torch.where(self.adm >= 0, self.adm, torch.full_like(self.adm, M_vis)), True)
            admit = admit[:, :M_vis].contiguous()
            F = build_forest(Kbar, ceil_c, admit, spread=spread_c)
            self.adm = F["adm"]; self.m_tot = F["m_tot"]
            for k in F["ks"]:
                C = F["chunks"][k]
                self.chunks[k] = {"mu": C["mu"].clone(), "rad": C["rad"].clone(), "amax": C["amax"].clone(),
                                  "pmin": C["pmin"].clone(), "leaf": C["leaf"].clone(),
                                  "cap": C["Cmax"], "built": (self.m_tot >> k).clone()}
            self.F = F
            if keep_cells:
                F["cellK"] = Kbar; F["cellA"] = ceil_c
            return
        changed = False
        Kmax = int(math.floor(math.log2(max(int(self.m_tot.max().item()), 1))))
        for k in range(Kmax + 1):
            size = 1 << k
            Ck = self.m_tot >> k
            if k not in self.chunks:
                self.chunks[k] = {"built": torch.zeros(T, dtype=torch.long, device=self.dev), "cap": 0}
            C = self.chunks[k]
            need = int(Ck.max().item())
            if need > C["cap"]:
                cap = max(need, 2 * C["cap"], 1)
                for name, shape in (("mu", (T, cap, 2 * size - 1, Dh)), ("rad", (T, cap, 2 * size - 1)), ("amax", (T, cap, 2 * size - 1)),
                                    ("pmin", (T, cap, 2 * size - 1)), ("leaf", (T, cap, size))):
                    dt = torch.long if name in ("pmin", "leaf") else torch.float32
                    new = torch.zeros(shape, dtype=dt, device=self.dev) if name != "leaf" else torch.full(shape, -1, dtype=dt, device=self.dev)
                    if C["cap"] > 0:
                        new[:, :C["cap"]] = C[name]
                    C[name] = new
                C["cap"] = cap
            while bool((C["built"] < Ck).any()):
                pend = C["built"] < Ck
                j = C["built"]
                pos = j[:, None] * size + torch.arange(size, device=self.dev)[None]
                cid = torch.gather(self.adm, 1, pos.clamp(max=self.adm.shape[1] - 1))
                cid = torch.where(pend[:, None], cid, torch.full_like(cid, -1))
                ok = cid >= 0
                Kb = torch.gather(Kbar, 1, cid.clamp(min=0)[..., None].expand(-1, -1, Dh))
                ce_ = torch.gather(ceil_c, 1, cid.clamp(min=0))
                sp_ = torch.gather(spread_c, 1, cid.clamp(min=0)) if spread_c is not None else None
                tr = build_trees(Kb, ce_, ok, k, spread=sp_)
                tab = _chunk_tables(tr, k, T, 1, cid.view(T, 1, size), slice(None), k)
                jj = j.clamp(max=C["cap"] - 1)
                rows = torch.arange(T, device=self.dev)
                for name in ("mu", "rad", "amax", "pmin", "leaf"):
                    cur = C[name][rows, jj]
                    C[name][rows, jj] = torch.where(pend.view(T, *([1] * (cur.dim() - 1))), tab[name][:, 0], cur)
                C["built"] = C["built"] + pend.long()
                changed = True
        if changed or self.F is None:
            F = {"adm": self.adm if self.adm.shape[1] > 0 else torch.full((T, 1), -1, dtype=torch.long, device=self.dev),
                 "m_tot": self.m_tot, "M": max(self.adm.shape[1], 1), "T": T, "Dh": Dh, "Kmax": Kmax, "ks": [], "chunks": {}}
            for k in range(Kmax + 1):
                C = self.chunks.get(k)
                if C is None or C["cap"] == 0:
                    continue
                F["ks"].append(k)
                F["chunks"][k] = {"mu": C["mu"], "rad": C["rad"], "amax": C["amax"], "pmin": C["pmin"], "leaf": C["leaf"], "Cmax": C["cap"], "k": k}
            _flatten_forest(F)
            self.F = F
        if keep_cells:
            self.F["cellK"] = Kbar; self.F["cellA"] = ceil_c


def choose_actions(attn, q: torch.Tensor, head_id: torch.Tensor, m0: torch.Tensor, height: torch.Tensor,
                   L: int, training: bool):
    T, Bq, Dh = q.shape
    dev = q.device
    if attn.fsb_policy_mode == "fixed":
        D = (height - attn.fsb_delta).clamp(min=0)
        Wt = torch.minimum(torch.full_like(D, attn.fsb_width), torch.bitwise_left_shift(torch.ones_like(D), D))
        return Wt[:, None].expand(T, Bq), D[:, None].expand(T, Bq), {}
    A = attn.fsb_actions
    Wa, Da = A[:, 0], A[:, 1]
    Dnode = (height[:, None] - Da[None, :]).clamp(min=0)
    valid = (Da[None, :] <= height[:, None]) & (Wa[None, :] <= torch.bitwise_left_shift(torch.ones_like(Dnode), Dnode))
    valid = valid & (m0[:, None] > 0)
    valid[:, 0] |= (m0 == 0)
    with torch.autocast(device_type=q.device.type, enabled=False):
        qn = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
        extra = torch.stack([torch.log1p(m0.float()) / math.log(257.0),
                             height.float() / 8.0], dim=-1)
        logits, value = attn.fsb_policy(qn, head_id, extra)
        logits = (logits / attn.fsb_policy_temp).masked_fill(~valid[:, None, :], NEG)
        logp_all = torch.log_softmax(logits, dim=-1)
        if training:
            a = torch.multinomial(logp_all.exp().view(-1, logp_all.shape[-1]), 1).view(T, Bq)
        else:
            a = logits.argmax(dim=-1)
        logp = torch.gather(logp_all, -1, a[..., None]).squeeze(-1)
        ent = -(logp_all.exp() * logp_all.masked_fill(~valid[:, None, :], 0.0)).sum(-1)
        if training and attn.fsb_rl == "ppo":
            attn._fsb_ppo_rec.append({"qn": qn.detach().to(attn.fsb_ppo_store_dtype), "head_id": head_id,
                                      "extra": extra.detach(), "valid": valid, "action": a.to(torch.uint8),
                                      "logp_old": logp.detach()})
    Wq = Wa[a]
    Dq = (height[:, None] - Da[a]).clamp(min=0)
    return Wq, Dq, {"logp": logp, "entropy": ent, "value": value, "action": a}


def _admit_frac(n_adm, eligible):
    ne = eligible.float().sum(-1)
    frac = n_adm.float() / ne.clamp(min=1)[None, None]
    w = (ne > 0).float()[None, None].expand_as(frac)
    return (frac * w).sum(dim=(0, 2)) / w.sum(dim=(0, 2)).clamp(min=1)


def _qf_rows(Q, q0, q1, N):
    qs = Q[:, :, q0:min(q1, N)].float()
    if q1 > N:
        qs = torch.cat([qs, qs.new_zeros(qs.shape[0], qs.shape[1], q1 - max(q0, N), qs.shape[3])], dim=2)
    return qs


def masked_readout(attn, Q, Kp, Vp, sel_full, s, Bq, N):
    B, H, Npad, kk = sel_full.shape
    Dh = Q.shape[-1]
    dev = Q.device
    nb = Npad // Bq
    NBCH = max(1, int(getattr(attn, "fsb_readout_blkchunk", 4)))
    Qp = Q if Npad == N else torch.cat([Q, Q.new_zeros(B, H, Npad - N, Dh)], dim=2)
    out = Q.new_zeros(B, H, Npad, Dh)
    off = torch.arange(s, device=dev)
    for b0 in range(0, nb, NBCH):
        b1 = min(b0 + NBCH, nb)
        q0, q1 = b0 * Bq, b1 * Bq
        nq = q1 - q0
        sel = sel_full[:, :, q0:q1].long()
        tok = (sel.clamp(min=0)[..., None] * s + off).view(B, H, nq, kk * s)
        gidx = tok.reshape(B, H, nq * kk * s, 1).expand(-1, -1, -1, Dh)
        Kc = torch.gather(Kp, 2, gidx).view(B, H, nq, kk * s, Dh)
        Vc = torch.gather(Vp, 2, gidx).view(B, H, nq, kk * s, Dh)
        sc = torch.einsum("bhqd,bhqkd->bhqk", Qp[:, :, q0:q1], Kc).float() * attn.scale
        qabs = torch.arange(q0, q1, device=dev).view(1, 1, nq, 1)
        allow = (sel >= 0).repeat_interleave(s, dim=-1) & (tok < qabs)
        sc = sc.masked_fill(~allow, NEG)
        w = torch.softmax(sc, dim=-1).to(Vc.dtype)
        ob = torch.einsum("bhqk,bhqkd->bhqd", w, Vc)
        ob = ob * allow.any(dim=-1, keepdim=True).to(ob.dtype)
        out[:, :, q0:q1] = ob.to(out.dtype)
    return attn.lr_norm(out[:, :, :N].to(Q.dtype))


def readout_handoff(attn, state=None):
    # The prefill readout's hand-off (FSAttn._begin_lr): the window output that the fused readout consumes when it
    # returns W_o's input (_lr_combine_sr, _lr_combined) and the cell statistics handed to the KV cache
    # (_cell_stats_out). Without state: a snapshot. With a snapshot: restore it, for a caller that discards a readout
    # it already queued and redoes the layer (the deferred rows capacity check), so the redo combines again and the
    # cache keeps one copy of the statistics. An in-place hand-off (_lr_inplace_sr: the readout writes W_o's input
    # over the window output) is snapshotted with a copy of the window output, since the first readout overwrites it.
    if state is None:
        cs = getattr(attn, "_cell_stats_out", None)
        sr = getattr(attn, "_lr_combine_sr", None)
        inplace = bool(getattr(attn, "_lr_inplace_sr", False))
        return (sr.clone() if (inplace and sr is not None) else sr, getattr(attn, "_lr_combined", False),
                None if cs is None else len(cs), inplace)
    attn._lr_combine_sr, attn._lr_combined = state[0], state[1]
    if len(state) > 3:
        attn._lr_inplace_sr = state[3]
    cs = getattr(attn, "_cell_stats_out", None)
    if cs is not None and state[2] is not None:
        del cs[state[2]:]
    return state


def fsb_long_range(attn, Q, K, V, sigma, N, kv_padded=None, _stage_check=False):
    B, H, _, Dh = Q.shape
    s = attn.cell_stride
    dev = Q.device
    M = (N + s - 1) // s
    pad = M * s - N
    if pad and kv_padded is not None and kv_padded[0].shape[2] >= M * s and kv_padded[1].shape[2] >= M * s:
        Kp, Vp = kv_padded[0][:, :, :M * s], kv_padded[1][:, :, :M * s]
    elif pad:
        Kp = torch.cat([K, K.new_zeros(B, H, pad, Dh)], dim=2)
        Vp = torch.cat([V, V.new_zeros(B, H, pad, Dh)], dim=2)
    else:
        Kp, Vp = K, V
    Bq = min(attn._cell_block, N)
    nb = (N + Bq - 1) // Bq
    Npad = nb * Bq
    T = B * H
    forest = attn.fsb_tree == "forest"
    read_top = getattr(attn, "fsb_read_top", 0)
    rr_score = getattr(attn, "fsb_rerank_score", "leaf") if read_top > 0 else "leaf"
    tokK = Kp.detach() if rr_score != "leaf" else None
    from .fsb_prefill import fsb_prefill_fast, prefill_fast_ok
    if prefill_fast_ok(attn, Q, tokK, Bq):                     # eager inference prefill (FSB_FAST_PREFILL=0: this loop)
        return fsb_prefill_fast(attn, Q, K, V, Kp, Vp, sigma, N, tokK)
    lr0 = readout_handoff(attn)                                 # restored if the deferred capacity check redoes the layer
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        from .fsl_select_cuda import cache_cell_stats
        st_c = cache_cell_stats(Kp, M, s) if (not attn.fsb_leaf_radius and not torch.is_grad_enabled()) else None
        if st_c is not None:
            Kbar, ceil_c = st_c
        else:
            Kpv = Kp.view(B, H, M, s, Dh)
            Kbar = Kpv.mean(dim=3).float()
            ceil_c = Kpv.float().norm(dim=-1).amax(dim=-1)
        if getattr(attn, "_cell_stats_out", None) is not None and not pad:
            attn._cell_stats_out.append((K, Kbar, ceil_c))      # FSKVCache.prefill: the same ops on the same view
        spread = None
        if attn.fsb_leaf_radius:
            dK = Kpv.float() - Kbar[..., None, :]
            spread = _det_sum(dK * dK, -1).amax(dim=-1).sqrt()
        if getattr(attn, "_fsb_calib", None) is not None:
            attn._fsb_calib.append(ceil_c.detach().transpose(0, 1).reshape(H, -1).cpu())
        blk_start = torch.arange(nb, device=dev, dtype=torch.float32) * Bq
        cell_end = (torch.arange(M, device=dev, dtype=torch.float32) + 1.0) * s
        eligible = cell_end[None, :] <= blk_start[:, None]
        admit_full = ceil_c >= attn.fsb_tau.view(1, H, 1)
        E_blk = (blk_start / s).floor().long()
        csum = admit_full.long().cumsum(-1)
        n_adm = torch.where(E_blk[None, None] > 0, csum.gather(-1, (E_blk.clamp(min=1, max=M) - 1)[None, None].expand(B, H, nb)),
                            torch.zeros((), dtype=torch.long, device=dev))
        del csum
        L = max(1, int(math.ceil(math.log2(M))))
        from .fsb_route import fsb_route_q, route_radius, radius_rows, route_on
        route = route_on(attn)
        c_h = route_radius(attn)
        rr_raw = read_top > 0 and getattr(attn, "fsb_rerank_query", "route") == "raw" and route
        _graph = (forest and dev.type == "cuda" and os.environ.get("FSB_TRAIN_GRAPH", "1") == "1"
                  and (attn.training or os.environ.get("FSB_EVAL_GRAPH", "0") == "1"))
        Qf = Qr = None
        if _graph:
            Qf = _qf_rows(Q, 0, Npad, N)
            Qr = fsb_route_q(attn, Qf)
        if forest and not _graph:
            F, m_b_all = prefill_forests(attn, Kbar.reshape(T, M, Dh), ceil_c.reshape(T, M), (blk_start / s).floor().long(),
                                         spread.reshape(T, M) if spread is not None else None)
            if read_top > 0:
                F["cellK"] = Kbar.reshape(T, M, Dh); F["cellA"] = ceil_c.reshape(T, M)
            if hybrid_on(attn):
                F["pool_order"] = pool_order(F["cellA"])
        n_adm_u = None
        if union_tau_on(attn):
            csum_u = (ceil_c >= attn.fsb_tau_union.view(1, H, 1)).long().cumsum(-1)
            n_adm_u = torch.where(E_blk[None, None] > 0, csum_u.gather(-1, (E_blk.clamp(min=1, max=M) - 1)[None, None].expand(B, H, nb)),
                                  torch.zeros((), dtype=torch.long, device=dev))
            del csum_u
    NBCH = attn.fsb_prefill_blkchunk if attn.fsb_prefill_blkchunk > 0 else nb
    NBCH = min(NBCH, nb)
    sel_full = torch.full((B, H, Npad, attn.fsb_kk), -1, dtype=torch.long, device=dev)
    nv_l, nr_l, has_l, pol_l, ut_l = [], [], [], [], []
    attn._fsb_ppo_rec = []
    results = []
    if _graph:
        results = _forest_chunks_graphed(attn, Qr, Kbar, ceil_c, sigma, N, nb, Bq, NBCH, blk_start, spread,
                                         Qpol=(None if Qr is Qf else Qf), c_h=c_h, Kp=tokK, rr_raw=rr_raw)
    else:
        # the rows re-rank's capacity flags (one per stage, only when rows can overflow) are read at the end, after
        # the readout is queued; probes and training keep the per-stage host check
        ovf = None if (_stage_check or attn.training or getattr(attn, "_fsb_calib", None) is not None
                       or getattr(attn, "_fsb_sel_override", None) is not None) else []
        d_eff = beam_delta_eff(attn)
        for b0 in range(0, nb, NBCH):
            b1 = min(b0 + NBCH, nb); nbc = b1 - b0; Tc = T * nbc
            with torch.no_grad():
                if Q.dtype == torch.bfloat16 and forest and not route and attn.fsb_policy_mode == "fixed" and not attn.training:
                    q0, q1 = b0 * Bq, b1 * Bq
                    nq = min(q1, N) - q0
                    if nq == nbc * Bq and Q[:, :, q0:q1].is_contiguous():
                        q = Q[:, :, q0:q1].view(Tc, Bq, Dh)
                    else:
                        q = torch.empty((Tc, Bq, Dh), dtype=torch.bfloat16, device=dev)
                        qv = q.view(B, H, nbc * Bq, Dh)
                        qv[:, :, :nq].copy_(Q[:, :, q0:q0 + nq])
                        if nq < nbc * Bq:
                            qv[:, :, nq:].zero_()
                    qp = q
                    rq_bf16 = True
                else:
                    qf_c = _qf_rows(Q, b0 * Bq, b1 * Bq, N)
                    qr_c = fsb_route_q(attn, qf_c)
                    q = qr_c.reshape(B, H, nbc, Bq, Dh).reshape(Tc, Bq, Dh)
                    qp = q if qr_c is qf_c else qf_c.reshape(B, H, nbc, Bq, Dh).reshape(Tc, Bq, Dh)
                    rq_bf16 = Q.dtype == torch.bfloat16 and (rr_raw or qr_c is qf_c)     # re-rank query = Q.float()
                    del qf_c, qr_c
                qpos = (blk_start[b0:b1, None] + torch.arange(Bq, device=dev, dtype=torch.float32)[None, :])
                qpos = qpos[None, None].expand(B, H, nbc, Bq).reshape(Tc, Bq)
                head_id = torch.arange(H, device=dev)[None, :, None].expand(B, H, nbc).reshape(Tc)
                lam = attn.sel_scale.float()[head_id].view(Tc, 1, 1)
                sigv = sigma.float()[head_id].view(Tc, 1, 1)
                if forest:
                    t_idx = torch.arange(T, device=dev).repeat_interleave(nbc)
                    m_b = m_b_all[:, b0:b1].reshape(Tc)
                    SP = build_spines(F, m_b, t_idx)
                    m0, height = m_b, SP["height"]
                    pool_c = None
                    if hybrid_on(attn):
                        E_rows = E_blk[b0:b1][None, :].expand(T, nbc).reshape(Tc)
                        pool_c = hybrid_pool(F["pool_order"], t_idx, E_rows, attn.fsb_pool_size, attn.fsb_pool_recent)
                else:
                    tr = build_trees(Kbar[:, :, None].expand(B, H, nbc, M, Dh).reshape(Tc, M, Dh),
                                     ceil_c[:, :, None].expand(B, H, nbc, M).reshape(Tc, M),
                                     (eligible[b0:b1][None, None] & admit_full[:, :, None, :]).reshape(Tc, M), L,
                                     spread=spread[:, :, None].expand(B, H, nbc, M).reshape(Tc, M) if spread is not None else None)
                    m0, height = tr["m0"], tr["height"]
            W, D, pol = choose_actions(attn, qp, head_id, m0, height, L, training=attn.training)
            with torch.no_grad():
                rc = radius_rows(attn, head_id, c_h)
                if forest:
                    delta = (height[:, None] - D).clamp(min=0)
                    ukw = union_tau_spines_kw(attn, F, F["utau_mba"][:, b0:b1].reshape(Tc) if "utau_mba" in F else None, t_idx, Bq)
                    uqkw = union_quota_kw(attn)
                    pl = [] if (ukw or uqkw) else None
                    sel, n_visit, n_read = traverse_forest(F, SP, q, qpos, lam, sigv, attn.scale, s, W, delta,
                                                           attn.fsb_W_max, attn.fsb_delta_max, attn.fsb_node_score,
                                                           radius_coef=rc, read_top=read_top,
                                                           rerank_score=rr_score, tokK=tokK, rq=(qp if rr_raw else None),
                                                           **union_kw(attn), **hybrid_kw(attn, pool_c), **rr_kw(attn),
                                                           **margin_kw(attn), **beam_kw(attn), **ukw, **uqkw,
                                                           **union_prior_kw(attn), rows=not attn.training,
                                                           rows_ovf=ovf, delta_eff=d_eff, rows_q_bf16=rq_bf16,
                                                           **({"prov_out": pl} if pl is not None else {}))
                    if pl:
                        ut_l.append(pl[0].view(B, H, nbc * Bq, -1))
                else:
                    tabs = None
                    if read_top > 0:
                        tabs = (Kbar.reshape(T, M, Dh), ceil_c.reshape(T, M),
                                torch.arange(T, device=dev).repeat_interleave(nbc))
                    sel, n_visit, n_read = traverse(tr, q, qpos, lam, sigv, attn.scale, s, W, D,
                                                    attn.fsb_W_max, attn.fsb_delta_max, attn.fsb_node_score,
                                                    radius_coef=rc, read_top=read_top, cell_tabs=tabs,
                                                    rerank_score=rr_score, tokK=tokK, rq=(qp if rr_raw else None),
                                                    **union_kw(attn), **rr_kw(attn), **margin_kw(attn))
            results.append((b0, b1, nbc, m0, sel, n_visit, n_read, pol))
    pend = None
    if not _graph and ovf:
        # device -> pinned host copy of the flags behind the stage loop; the host waits for it only after this
        # layer's statistics and readout are queued, so the GPU is not idle while it waits. The copy runs on the
        # current stream of the flags' device (Q's, not necessarily the current device), so the event is recorded there
        fl = torch.stack(ovf)
        pend = (torch.empty(fl.shape, dtype=torch.bool, pin_memory=True), torch.cuda.Event(),
                getattr(attn, "_fsb_read_acc", None))
        pend[0].copy_(fl, non_blocking=True)
        pend[1].record(torch.cuda.current_stream(fl.device))
        if pend[2] is not None:
            pend = pend + (pend[2].clone(),)
    _rd = getattr(attn, "fsb_rd_weight", 0.0) > 0 and attn.training
    if not _rd:
        F = SP = tr = m_b_all = adm_sorted = None
    for b0, b1, nbc, m0, sel, n_visit, n_read, pol in results:
        Tc = T * nbc
        sel_full[:, :, b0 * Bq:b1 * Bq] = sel.view(B, H, nbc * Bq, attn.fsb_kk)
        nv_l.append(n_visit.view(B, H, nbc * Bq)); nr_l.append(n_read.view(B, H, nbc * Bq))
        has_l.append((m0 > 0)[:, None].expand(Tc, Bq).reshape(B, H, nbc * Bq))
        pol_l.append({k_: v_.view(B, H, nbc * Bq) for k_, v_ in pol.items()})
    if attn._fsb_ppo_rec:
        assert len(attn._fsb_ppo_rec) == len(results), (len(attn._fsb_ppo_rec), len(results))
        for rec, (b0, b1, *_r) in zip(attn._fsb_ppo_rec, results):
            rec["cols"] = (b0 * Bq, b1 * Bq)
    n_visit = torch.cat(nv_l, dim=2); n_read = torch.cat(nr_l, dim=2)
    fsb_read_acc(attn, n_read[:, :, :N])
    if ut_l:
        fsb_utau_acc(attn, torch.cat(ut_l, dim=2)[:, :, :N])
    real = (torch.arange(Npad, device=dev) < N).view(1, 1, Npad)
    has = torch.cat(has_l, dim=2) & real
    cost = (n_read.float() * s + attn.fsb_route_cost * n_visit) / float(attn.cell_topk * s)
    st = {"has": has[:, :, :N], "cost": cost[:, :, :N], "n_read": n_read[:, :, :N].float(),
          "n_visit": n_visit[:, :, :N],
          "admit_frac": _admit_frac(n_adm, eligible),
          "m0_mean": n_adm.float().mean()}
    if n_adm_u is not None:
        st["admit_frac_union"] = _admit_frac(n_adm_u, eligible)
    if pol_l and pol_l[0]:
        for k_ in pol_l[0]:
            st[k_] = torch.cat([p_[k_] for p_ in pol_l], dim=2)[:, :, :N]
    attn._fsb_stats = st
    if _rd:
        from .fsb_route import fsb_rd_loss
        fr = None
        if getattr(attn, "fsb_rd_node", 0) and forest:
            fr = (attn._fsb_train_graphs["build"]["F"], attn._fsb_train_graphs["build"]["m_b_all"]) if _graph else (F, m_b_all)
        if Qf is None:
            Qf = _qf_rows(Q, 0, Npad, N)
            Qr = fsb_route_q(attn, Qf)
        admit = eligible[None, None] & admit_full[:, :, None, :]
        attn._fsb_rd = fsb_rd_loss(attn, Qf, Kp, Kbar, ceil_c, admit_full, admit, sigma, sel_full, N, Bq, nb, s, Qr=Qr,
                                   forest=fr)
    elif getattr(attn, "_fsb_rd", None) is not None:
        attn._fsb_rd = None
    ov = getattr(attn, "_fsb_sel_override", None)
    if ov is not None:
        attn._fsb_sel_override = None
        if ov.get("rows") is not None:
            ov["row_sel"] = sel_full[:, :, ov["rows"]].clone()
        ov["seen"] = sel_full[ov["v"], ov["h"], ov["i"]].clone()
        sel_full[ov["v"], ov["h"], ov["i"]] = ov["sel"].to(sel_full.dtype)
    _fused = bool(attn._prune_cfg("fs_fused_attn", "FS_FUSED_ATTN", 0))
    if _fused:
        out = attn._cell_fused_lr(Q, K, V, sel_full.to(torch.int32).contiguous(), s, Bq)
    else:
        out = masked_readout(attn, Q, Kp, Vp, sel_full, s, Bq, N)
    if pend is not None:
        pend[1].synchronize()
        if bool(pend[0].any()):
            # a row with more than fsb_rows.UCAP distinct candidates (never seen in practice): recompute the layer
            # with the per-stage check, which redoes exactly the overflowing stages without the rows path, after
            # undoing this pass's read-statistics updates and its readout's hand-off
            if pend[2] is not None:
                pend[2].copy_(pend[3])
            readout_handoff(attn, lr0)
            return fsb_long_range(attn, Q, K, V, sigma, N, kv_padded, _stage_check=True)
    return out


_FSB_POOL = None


def _fsb_graph_pool():
    global _FSB_POOL
    if _FSB_POOL is None:
        _FSB_POOL = torch.cuda.graph_pool_handle()
    return _FSB_POOL


@torch.no_grad()
def prefill_forests(attn, Kbar_s, ceil_s, E, spread_s=None):
    # the prefill forest (admission fsb_tau) and, with union tau on, the second beam's forest, with the per-block
    # prefix counts m_b = searchsorted(adm, E); one CUDA build of both when FSB_BUILD_CUDA covers the inputs
    T, M, Dh = Kbar_s.shape
    from .fsb_build_cuda import forests_cuda
    got = forests_cuda(Kbar_s, ceil_s, spread_s, [attn.fsb_tau] + ([attn.fsb_tau_union] if union_tau_on(attn) else []), E)
    if got is not None:
        F, m_b_all = got[0]
        if len(got) > 1:
            F["utau_F"], F["utau_mba"] = got[1]
        return F, m_b_all
    H = attn.fsb_tau.shape[0]
    admit_full = ceil_s >= attn.fsb_tau.view(1, H, 1).expand(T // H, H, 1).reshape(T, 1)
    F = build_forest(Kbar_s, ceil_s, admit_full, spread=spread_s)
    F["chunks"] = {}
    adm_sorted = F["adm"].masked_fill(F["adm"] < 0, M)
    m_b_all = torch.searchsorted(adm_sorted, E[None, :].expand(T, -1).contiguous())
    if union_tau_on(attn):
        F["utau_F"], F["utau_mba"], _ = utau_build(attn, Kbar_s, ceil_s, E, spread_s)
    return F, m_b_all


@torch.no_grad()
def _forest_build_stage(attn, Kbar_s, ceil_s, blk_start, s, spread_s=None):
    F, m_b_all = prefill_forests(attn, Kbar_s, ceil_s, (blk_start / s).floor().long(), spread_s)
    if getattr(attn, "fsb_read_top", 0) > 0:
        F["cellK"] = Kbar_s; F["cellA"] = ceil_s
    if hybrid_on(attn):
        F["pool_order"] = pool_order(ceil_s)
    return F, m_b_all


@torch.no_grad()
def _forest_chunk_stage(attn, F, m_b, t_idx, q, qpos, lam, sigv, W, D, s, fixed, radius_coef=None, tokK=None, rq=None,
                        pool=None, m_b2=None):
    SP = build_spines(F, m_b, t_idx)
    Tc, Bq = q.shape[0], q.shape[1]
    if fixed:
        D = (SP["height"] - attn.fsb_delta).clamp(min=0)[:, None].expand(Tc, Bq)
        W = torch.minimum(torch.full_like(D, attn.fsb_width), torch.bitwise_left_shift(torch.ones_like(D), D))
    delta = (SP["height"][:, None] - D).clamp(min=0)
    sel, n_visit, n_read = traverse_forest(F, SP, q, qpos, lam, sigv, attn.scale, s, W, delta,
                                           attn.fsb_W_max, attn.fsb_delta_max, attn.fsb_node_score,
                                           radius_coef=attn.fsb_bound_radius_coef if radius_coef is None else radius_coef,
                                           read_top=getattr(attn, "fsb_read_top", 0),
                                           rerank_score=(getattr(attn, "fsb_rerank_score", "leaf") if tokK is not None else "leaf"),
                                           tokK=tokK, rq=rq, **union_kw(attn), **hybrid_kw(attn, pool), **rr_kw(attn),
                                           **margin_kw(attn), **beam_kw(attn), **union_tau_spines_kw(attn, F, m_b2, t_idx, Bq),
                                           **union_quota_kw(attn), **union_prior_kw(attn))
    return SP["height"], sel, n_visit, n_read


def _forest_chunks_graphed(attn, Qf, Kbar, ceil_c, sigma, N, nb, Bq, NBCH, blk_start, spread=None, Qpol=None, c_h=None,
                           Kp=None, rr_raw=False):
    B, H, _, Dh = Qf.shape
    s = attn.cell_stride
    dev = Qf.device
    T = B * H
    M = Kbar.shape[2]
    fixed = attn.fsb_policy_mode == "fixed"
    key = (T, M, N, nb, Bq, NBCH, attn.fsb_policy_mode, attn.fsb_width, attn.fsb_delta, attn.fsb_node_score, attn.fsb_bound_radius_coef, attn.fsb_leaf_radius,
           attn.fsb_W_max, attn.fsb_delta_max, str(dev), c_h is not None, getattr(attn, "fsb_read_top", 0))
    if Kp is not None:
        key = key + (getattr(attn, "fsb_rerank_score", "leaf"), tuple(Kp.shape), Kp.dtype)
    if rr_raw:
        assert Qpol is not None, "rr_raw needs the raw query (Qpol)"
        key = key + ("rerank_query_raw",)
    if getattr(attn, "fsb_union_width", 0) > 0:
        key = key + ("union", attn.fsb_union_width, attn.fsb_union_W_max)
    if float(getattr(attn, "fsb_read_margin", 0.0) or 0.0) > 0:
        key = key + ("read_margin", float(attn.fsb_read_margin), int(attn.fsb_read_min))
    if int(getattr(attn, "fsb_beam_quota", 0) or 0) > 0:
        key = key + ("beam_quota", int(attn.fsb_beam_quota))
    if union_tau_on(attn):
        key = key + ("union_tau", float(attn.fsb_union_tau_quantile), attn.fsb_tau_union.data_ptr())
    if union_quota_on(attn):
        key = key + ("union_quota", int(attn.fsb_union_quota))
    if union_prior_on(attn):
        key = key + ("union_prior", float(attn.fsb_union_prior_scale))
    hyb = hybrid_on(attn)
    if hyb:
        key = key + ("hybrid", attn.fsb_hybrid, attn.fsb_pool_size, attn.fsb_pool_recent, attn.fsb_pool_short,
                     attn.fsb_pool_quota, getattr(attn, "fsb_rerank_score", "leaf"))
    if getattr(attn, "fsb_rerank_rr", "none") != "none":
        key = key + ("rerank_rr", attn.fsb_rerank_rr, attn.fsb_route_rr_w.data_ptr())
    cache = getattr(attn, "_fsb_train_graphs", None)
    if cache is None or cache.get("key") != key:
        cache = {"key": key, "chunks": [], "c": (c_h.detach().float().clone() if c_h is not None else None)}
        attn._fsb_train_graphs = cache
        pool = _fsb_graph_pool()
        Kbar_s = Kbar.reshape(T, M, Dh).clone(); ceil_s = ceil_c.reshape(T, M).clone(); blk_s = blk_start.clone()
        spread_s = spread.reshape(T, M).clone() if spread is not None else None
        Kp_s = Kp.contiguous().clone() if Kp is not None else None
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _forest_build_stage(attn, Kbar_s, ceil_s, blk_s, s, spread_s)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            F, m_b_all = _forest_build_stage(attn, Kbar_s, ceil_s, blk_s, s, spread_s)
        g.replay()
        torch.cuda.synchronize()
        cache["build"] = {"g": g, "Kbar": Kbar_s, "ceil": ceil_s, "spread": spread_s, "F": F, "m_b_all": m_b_all, "blk_start": blk_s,
                          "Kp": Kp_s}
        for b0 in range(0, nb, NBCH):
            b1 = min(b0 + NBCH, nb); nbc = b1 - b0; Tc = T * nbc
            q_s = torch.zeros(Tc, Bq, Dh, device=dev)
            rq_s = torch.zeros(Tc, Bq, Dh, device=dev) if rr_raw else None
            qpos = (blk_start[b0:b1, None] + torch.arange(Bq, device=dev, dtype=torch.float32)[None, :])
            qpos = qpos[None, None].expand(B, H, nbc, Bq).reshape(Tc, Bq).contiguous()
            head_id = torch.arange(H, device=dev)[None, :, None].expand(B, H, nbc).reshape(Tc)
            t_idx = torch.arange(T, device=dev).repeat_interleave(nbc)
            W_s = torch.full((Tc, Bq), attn.fsb_width, dtype=torch.long, device=dev)
            D_s = torch.zeros(Tc, Bq, dtype=torch.long, device=dev)
            sig_s = sigma.detach().float().clone()
            E_s = (blk_start[b0:b1] / s).floor().long()[None, :].expand(T, nbc).reshape(Tc).contiguous() if hyb else None
            def stage(_F=F, _mba=m_b_all, _b0=b0, _b1=b1, _t_idx=t_idx, _q=q_s, _qpos=qpos, _head=head_id, _sig=sig_s, _W=W_s, _D=D_s,
                      _c=cache["c"], _tok=Kp_s, _rq=rq_s, _E=E_s):
                for _ff in (_F, _F.get("utau_F")):
                    if _ff is not None and _ff.get("node") is not None:
                        # the build graph rewrites the packed tables on every replay (incl. in-place tau changes),
                        # so the beams' bf16 views are re-derived inside this capture rather than baked in from warmup
                        _ff.pop("node_mu", None); _ff.pop("node_aux", None)
                m_b_ = _mba[:, _b0:_b1].reshape(Tc)
                lam_ = attn.sel_scale.detach().float()[_head].view(Tc, 1, 1)
                sigv = _sig[_head].view(Tc, 1, 1)
                rc_ = _c[_head] if _c is not None else None
                pool_ = (hybrid_pool(_F["pool_order"], _t_idx, _E, attn.fsb_pool_size, attn.fsb_pool_recent)
                         if _E is not None else None)
                m_b2_ = _F["utau_mba"][:, _b0:_b1].reshape(Tc) if _F.get("utau_mba") is not None else None
                h_, sel_, nv_, nr_ = _forest_chunk_stage(attn, _F, m_b_, _t_idx, _q, _qpos, lam_, sigv, _W, _D, s, fixed, rc_,
                                                         tokK=_tok, rq=_rq, pool=pool_, m_b2=m_b2_)
                return h_, sel_, nv_, nr_, m_b_
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                stage()
            torch.cuda.current_stream().wait_stream(side)
            gc_ = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gc_, pool=pool):
                height, sel, n_visit, n_read, m_b = stage()
            gc_.replay(); torch.cuda.synchronize()
            cache["chunks"].append({"g": gc_, "q": q_s, "rq": rq_s, "W": W_s, "D": D_s, "sigma": sig_s, "b0": b0, "b1": b1, "nbc": nbc,
                                    "height": height, "sel": sel, "n_visit": n_visit, "n_read": n_read,
                                    "head_id": head_id, "t_idx": t_idx, "m_b": m_b, "qpos": qpos, "E": E_s})
    bd = cache["build"]
    if cache["c"] is not None:
        cache["c"].copy_(c_h.detach().float())
    bd["Kbar"].copy_(Kbar.reshape(T, M, Dh)); bd["ceil"].copy_(ceil_c.reshape(T, M))
    if bd["spread"] is not None:
        bd["spread"].copy_(spread.reshape(T, M))
    if bd.get("Kp") is not None:
        bd["Kp"].copy_(Kp)
    bd["g"].replay()
    F = bd["F"]
    out = []
    for ch in cache["chunks"]:
        b0, b1, nbc = ch["b0"], ch["b1"], ch["nbc"]; Tc = T * nbc
        ch["q"].copy_(Qf[:, :, b0 * Bq:b1 * Bq].reshape(B, H, nbc, Bq, Dh).reshape(Tc, Bq, Dh))
        if ch["rq"] is not None:
            ch["rq"].copy_(Qpol[:, :, b0 * Bq:b1 * Bq].reshape(B, H, nbc, Bq, Dh).reshape(Tc, Bq, Dh))
        ch["sigma"].copy_(sigma.detach().float())
        pol = {}
        if not fixed:
            with torch.no_grad():
                m_b_now = bd["m_b_all"][:, b0:b1].reshape(Tc)
                height = build_spines(F, m_b_now, ch["t_idx"])["height"]
            qpol = ch["q"] if Qpol is None else Qpol[:, :, b0 * Bq:b1 * Bq].reshape(B, H, nbc, Bq, Dh).reshape(Tc, Bq, Dh)
            W, D, pol = choose_actions(attn, qpol, ch["head_id"], m_b_now, height, F["Kmax"] + 1, training=attn.training)
            ch["W"].copy_(W); ch["D"].copy_(D)
        ch["g"].replay()
        out.append((b0, b1, nbc, ch["m_b"].clone(), ch["sel"].clone(), ch["n_visit"].clone(), ch["n_read"].clone(), pol))
    return out


@torch.no_grad()
def fsb_decode_step(attn, q_t, cache, sigma, pos):
    B, H, _, Dh = q_t.shape
    s = attn.cell_stride
    dev = q_t.device
    Bq = attn._cell_block
    blk_start = (pos // Bq) * Bq
    if attn.fsb_tree == "forest":
        return _fsb_decode_step_forest(attn, q_t, cache, sigma, pos)
    tr = getattr(cache, "_fsb_tree", None)
    if tr is None or tr["blk_start"] != blk_start:
        Kbar_all, ceil_all, ncell = cache.completed_cells()
        M_vis = min(blk_start // s, ncell) if ncell > 0 else 0
        if M_vis <= 0:
            tr = {"blk_start": blk_start, "empty": True}
        else:
            Kbar = Kbar_all[:, :, :M_vis].float().reshape(B * H, M_vis, Dh)
            ceil_c = ceil_all[:, :, :M_vis].float().reshape(B * H, M_vis)
            admit = ceil_c >= attn.fsb_tau.view(1, H, 1).expand(B, H, 1).reshape(B * H, 1)
            L = max(1, int(math.ceil(math.log2(M_vis))))
            sp_all = cache.completed_spread() if attn.fsb_leaf_radius else None
            tr = {"blk_start": blk_start, "empty": False, "L": L,
                  "trees": build_trees(Kbar, ceil_c, admit, L,
                                       spread=sp_all[:, :, :M_vis].float().reshape(B * H, M_vis) if sp_all is not None else None)}
            if getattr(attn, "fsb_read_top", 0) > 0:
                tr["cell_tabs"] = (Kbar, ceil_c, torch.arange(B * H, device=dev))
        cache._fsb_tree = tr
    if tr["empty"]:
        fsb_read_acc(attn, torch.zeros(B * H, device=dev))
        return q_t.new_zeros(B, H, 1, Dh)
    trees, L = tr["trees"], tr["L"]
    T = B * H
    q = q_t[:, :, 0, :].float().reshape(T, 1, Dh)
    from .fsb_route import fsb_route_q, route_on
    qr = fsb_route_q(attn, q_t.float()).reshape(T, 1, Dh) if route_on(attn) else q
    qpos = torch.full((T, 1), float(pos), device=dev)
    head_id = torch.arange(H, device=dev)[None, :].expand(B, H).reshape(T)
    lam = attn.sel_scale.float()[head_id].view(T, 1, 1)
    sigv = sigma.float()[head_id].view(T, 1, 1)
    W, D, _ = choose_actions(attn, q, head_id, trees["m0"], trees["height"], L, training=False)
    from .fsb_route import radius_rows
    rt = getattr(attn, "fsb_read_top", 0)
    rr = getattr(attn, "fsb_rerank_score", "leaf") if rt > 0 else "leaf"
    rq = q if (rt > 0 and getattr(attn, "fsb_rerank_query", "route") == "raw" and qr is not q) else None
    sel, _, _ = traverse(trees, qr, qpos, lam, sigv, attn.scale, s, W, D,
                         attn.fsb_W_max, attn.fsb_delta_max, attn.fsb_node_score,
                         radius_coef=radius_rows(attn, head_id), read_top=rt,
                         cell_tabs=tr.get("cell_tabs"), rerank_score=rr,
                         tokK=(cache.lr_tokens()[0] if rr != "leaf" else None), rq=rq,
                         **union_kw(attn), **rr_kw(attn), **margin_kw(attn))
    fsb_read_acc(attn, (sel >= 0).sum(-1))
    return _fsb_decode_readout(attn, q_t, cache, sel, pos)


def _fsb_decode_readout(attn, q_t, cache, sel, pos, full_buffer=False):
    if (getattr(cache, "use_int8", False) and q_t.is_cuda and torch.cuda.is_current_stream_capturing()):
        raise RuntimeError("the int8 FS-B reference readout must not be captured: it gathers from a "
                           "length-sliced materialization whose baked shape goes stale at replay; "
                           "decode_readout_fast must cover this configuration instead")
    B, H, _, Dh = q_t.shape
    s = attn.cell_stride
    dev = q_t.device
    kk = sel.shape[-1]
    sel = sel.view(B, H, kk)
    if full_buffer and getattr(cache, "read_full_kv", None) is not None:
        Kp, Vp = cache.read_full_kv()
    elif full_buffer and getattr(cache, "_k_buf", None) is not None:
        Kp, Vp = cache._k_buf, cache._v_buf
    else:
        Kp, Vp = cache.lr_tokens()
    off = torch.arange(s, device=dev)
    tok = (sel.clamp(min=0)[..., None] * s + off).view(B, H, kk * s)
    gidx = tok[..., None].expand(-1, -1, -1, Dh)
    Kc = torch.gather(Kp, 2, gidx)
    Vc = torch.gather(Vp, 2, gidx)
    sc = torch.einsum("bhd,bhkd->bhk", q_t[:, :, 0, :], Kc).float() * attn.scale
    allow = (sel >= 0).repeat_interleave(s, dim=-1) & (tok < pos)
    sc = sc.masked_fill(~allow, NEG)
    w = torch.softmax(sc, dim=-1).to(Vc.dtype)
    ob = torch.einsum("bhk,bhkd->bhd", w, Vc).unsqueeze(2)
    ob = ob * allow.any(dim=-1).view(B, H, 1, 1).to(ob.dtype)
    return attn.lr_norm(ob.to(q_t.dtype))


def _capture_decode_graph(g, fn):
    if os.environ.get("FSB_DECODE_CAPTURE_GC", "0") == "1":
        with torch.cuda.graph(g):
            return fn()
    import gc
    if torch.cuda.graph.default_capture_stream is None:
        torch.cuda.graph.default_capture_stream = torch.cuda.Stream()
    # a collected CUDA-object destructor (an old graph, stream or event) mid-capture invalidates it;
    # flush the collector up front and hold it off across the capture
    if os.environ.get("FSB_DECODE_CAPTURE_GC") != "0":
        gc.collect()
        torch.cuda.synchronize()
    gc.disable()
    try:
        with torch.cuda.stream(torch.cuda.graph.default_capture_stream):
            g.capture_begin()
            try:
                out = fn()
            finally:
                g.capture_end()
    finally:
        gc.enable()
    return out


@torch.no_grad()
def _forest_token(attn, q_t, cache, sigma, pos, fc, c_h=None, pos_f=None, sr=None):
    B, H, _, Dh = q_t.shape
    s = attn.cell_stride
    dev = q_t.device
    T = B * H
    from .fsb_route import fsb_route_q, route_on, radius_rows
    ro = route_on(attn)
    pos_t = pos if torch.is_tensor(pos) else torch.full((), float(pos), device=dev)
    qpos = (pos_f if pos_f is not None else pos_t.float()).view(1, 1).expand(T, 1)
    utau = getattr(fc, "utau", None)
    # query-independent per-block values (fixed policy): recomputed whenever the forests, spines or the
    # head parameters change, so every token sees exactly the values it would compute itself
    key = (getattr(fc, "serial", 0), B, H, attn.sel_scale._version, attn.sel_scale.data_ptr(),
           attn.log_sigma._version, attn.log_sigma.data_ptr(), attn.fsb_width,
           attn.fsb_delta, int(getattr(attn, "fsb_union_width", 0) or 0))
    tk = getattr(fc, "_tok", None)
    q = qr = None
    if attn.fsb_policy_mode == "fixed" and tk is not None and tk["key"] == key:
        head_id, lam, W, delta, ukw = tk["head_id"], tk["lam"], tk["W"], tk["delta"], tk["ukw"]
    else:
        q = q_t[:, :, 0, :].float().reshape(T, 1, Dh)
        qr = fsb_route_q(attn, q_t.float()).reshape(T, 1, Dh) if ro else q
        head_id = torch.arange(H, device=dev)[None, :].expand(B, H).reshape(T)
        lam = attn.sel_scale.float()[head_id].view(T, 1, 1)
        W, D, _ = choose_actions(attn, q, head_id, fc.m_tot, fc.SP["height"], fc.F["Kmax"] + 1, training=False)
        delta = (fc.SP["height"][:, None] - D).clamp(min=0)
        ukw = union_tau_kw(attn, utau.F, utau.SP, 1) if utau is not None else {}
        if ukw and int(getattr(attn, "fsb_union_width", 0) or 0) > 0:
            Du = (ukw["union_SP"]["height"][:, None] - ukw["union_delta"]).clamp(min=0)
            ukw["union_W"] = torch.minimum(torch.full_like(Du, int(attn.fsb_union_width)),
                                           torch.bitwise_left_shift(torch.ones_like(Du), Du))
        if attn.fsb_policy_mode == "fixed":
            fc._tok = tk = {"key": key, "head_id": head_id, "lam": lam, "W": W, "delta": delta, "ukw": ukw}
    sigv = tk["sigv"] if (tk is not None and sigma is tk.get("sigma") and "sigv" in tk) else sigma.float()[head_id].view(T, 1, 1)
    rr = getattr(attn, "fsb_rerank_score", "leaf") if getattr(attn, "fsb_read_top", 0) > 0 else "leaf"
    tokK = None
    if rr != "leaf":
        if getattr(cache, "use_int8", False):
            tokK = cache.rerank_keys()          # the int8 read state for the fused kernels
        else:
            tokK = cache._k_buf if getattr(cache, "_k_buf", None) is not None else cache.lr_tokens()[0]
    hkw = hybrid_kw(attn, fc.pool)
    bkw = beam_kw(attn)
    uqkw = union_quota_kw(attn)
    upkw = union_prior_kw(attn)
    rc = radius_rows(attn, head_id, c_h)
    prov = [] if (hkw or bkw or ukw or uqkw) else None
    d_eff = beam_delta_eff(attn)
    sel = None
    if (attn.fsb_policy_mode == "fixed" and tk is not None and tk["key"] == key and rr != "leaf" and q_t.is_cuda
            and not (hkw or bkw or uqkw or upkw or rr_kw(attn) or margin_kw(attn))):
        if qr is None:
            if ro:
                q = q_t[:, :, 0, :].float().reshape(T, 1, Dh)
                qr = fsb_route_q(attn, q_t.float()).reshape(T, 1, Dh)
            elif q_t.dtype in (torch.float32, torch.bfloat16) and q_t.is_contiguous():
                q = qr = q_t.view(T, 1, Dh)
            else:
                q = qr = q_t[:, :, 0, :].float().reshape(T, 1, Dh)
        rq = q if (getattr(attn, "fsb_read_top", 0) > 0 and getattr(attn, "fsb_rerank_query", "route") == "raw" and qr is not q) else None
        sel = _forest_token_plain(attn, fc, tk, qr, qr if rq is None else rq, qpos, lam, sigv, rc, rr, tokK, ukw, prov,
                                  d_eff=d_eff)
    if sel is None:
        if qr is None or q.dtype != torch.float32:
            q = q_t[:, :, 0, :].float().reshape(T, 1, Dh)
            qr = fsb_route_q(attn, q_t.float()).reshape(T, 1, Dh) if ro else q
        rq = q if (getattr(attn, "fsb_read_top", 0) > 0 and getattr(attn, "fsb_rerank_query", "route") == "raw" and qr is not q) else None
        sel, _, _ = traverse_forest(fc.F, fc.SP, qr, qpos, lam, sigv, attn.scale, s, W, delta,
                                    attn.fsb_W_max, attn.fsb_delta_max, attn.fsb_node_score,
                                    radius_coef=rc, read_top=getattr(attn, "fsb_read_top", 0),
                                    rerank_score=rr, tokK=(cache.lr_tokens()[0] if getattr(cache, "use_int8", False)
                                                           else tokK),
                                    rq=rq, delta_eff=d_eff, **union_kw(attn), **hkw,
                                    **rr_kw(attn), **margin_kw(attn), **bkw, **ukw, **uqkw, **upkw,
                                    **({"prov_out": prov} if prov is not None else {}))
    fc._last_sel = sel
    if prov is not None:
        fc._last_prov = prov[0]
    from .fsb_fast_step import decode_readout_fast
    rout = decode_readout_fast(attn, cache, q_t, sel, pos_t, sr=sr)
    return rout if rout is not None else _fsb_decode_readout(attn, q_t, cache, sel, pos_t, full_buffer=True)


def _forest_token_plain(attn, fc, tk, q, rq, qpos, lam, sigv, rc, rr, tokK, ukw, prov, d_eff=None):
    # traverse_forest for one decode query per row in the plain configuration (fixed policy, overselect re-rank,
    # optional second beam; no hybrid pool, quotas, margins, prior scale or residual re-ranker), on the fused
    # kernels: the same operations in the same order, minus the discarded visit and read counts, with the
    # per-block budgets reused. None when a fused kernel does not apply.
    from .fsb_fast_step import (fast_beam_ok, fast_rerank_ok, fast_union_rerank_ok, forest_beam_fused,
                                forest_beam_pair, forest_beam_pair_ok, rerank_top_fast, union_rerank_fast)
    from .fsb_traverse_triton import _HAVE_TRITON
    if not _HAVE_TRITON or os.environ.get("FSB_TRAVERSE_TRITON", "1") != "1":
        return None
    F, SP = fc.F, fc.SP
    W_max, ns, s = attn.fsb_W_max, attn.fsb_node_score, attn.cell_stride
    d_max = attn.fsb_delta_max if d_eff is None else d_eff
    uw = int(getattr(attn, "fsb_union_width", 0) or 0)
    Fu, SPu = (ukw["union_F"], ukw["union_SP"]) if ukw else (F, SP)
    if not (fast_beam_ok(F, SP, q, W_max, d_max) and (uw == 0 or fast_beam_ok(Fu, SPu, q, attn.fsb_union_W_max, d_max))):
        return None
    if "bud" not in tk:
        tk["bud"] = torch.bitwise_left_shift(torch.ones_like(tk["delta"]), tk["delta"]).contiguous()
        if uw > 0:
            du = ukw["union_delta"] if ukw else tk["delta"]
            if not ukw:
                Du = (SP["height"][:, None] - du).clamp(min=0)
                tk["union_W"] = torch.minimum(torch.full_like(Du, uw), torch.bitwise_left_shift(torch.ones_like(Du), Du))
            tk["bud_u"] = torch.bitwise_left_shift(torch.ones_like(du), du).contiguous()
    sel_u = sel_unew = pair = None
    if uw > 0:
        du = ukw["union_delta"] if ukw else tk["delta"]
        Wu = ukw["union_W"] if ukw else tk["union_W"]
        if forest_beam_pair_ok(F, SP, Fu, SPu, q, W_max, attn.fsb_union_W_max, d_max):
            pair = forest_beam_pair(F, SP, tk["W"], tk["delta"], W_max, Fu, SPu, Wu, du, attn.fsb_union_W_max,
                                    q, qpos, lam, sigv, attn.scale, s, d_max, ns, rc,
                                    bud=tk["bud"], bud_u=tk["bud_u"])
    if pair is not None:
        (sel, _), (sel_u, _) = pair
    else:
        sel, _ = forest_beam_fused(F, SP, q, qpos, lam, sigv, attn.scale, s, tk["W"], tk["delta"], W_max, d_max, ns,
                                   rc, bud=tk["bud"])
        if uw > 0:
            sel_u, _ = forest_beam_fused(Fu, SPu, q, qpos, lam, sigv, attn.scale, s, Wu, du, attn.fsb_union_W_max,
                                         d_max, ns, rc, bud=tk["bud_u"])
    K = int(attn.fsb_read_top)
    cellA = fc._cellA_g if getattr(fc, "_cellA_g", None) is not None else F["cellA"]
    if fast_union_rerank_ok(sel.shape[-1], 0 if sel_u is None else sel_u.shape[-1], rq, tokK, s, K, rr):
        # union, re-rank and the second-beam provenance in two kernels around torch.erf
        want = bool(ukw) and prov is not None
        sel, pv = union_rerank_fast(sel, sel_u, rq, qpos, lam, sigv, cellA, SP["t_idx"], attn.scale, s, K, rr,
                                    tokK, prov=want)
        if want:
            prov.append(pv)
        return sel
    if sel_u is not None:
        Cm = sel.shape[-1]
        sel = union_candidates(sel, sel_u)
        sel_unew = sel[..., Cm:]
    if fast_rerank_ok(sel, rq, tokK, s, K, rr):
        sel = rerank_top_fast(sel, rq, qpos, lam, sigv, F["cellA"], SP["t_idx"], attn.scale, s, K, rr, tokK)
    else:
        sel = rerank_top(sel, rq.float(), qpos, lam, sigv, F["cellK"], F["cellA"], SP["t_idx"], attn.scale, s, K,
                         score=rr, tokK=tokK)
    if ukw and prov is not None:
        utau_prov_add(prov, sel, sel_unew)
    return sel


def _decode_sel_cfg(attn) -> tuple:
    rr = getattr(attn, "fsb_route_rr_w", None)
    ut = getattr(attn, "fsb_tau_union", None) if union_tau_on(attn) else None
    return (getattr(attn, "fsb_hybrid", "off"), getattr(attn, "fsb_pool_size", None), getattr(attn, "fsb_pool_recent", None),
            getattr(attn, "fsb_pool_short", None), getattr(attn, "fsb_pool_quota", None), getattr(attn, "fsb_read_top", 0),
            getattr(attn, "fsb_rerank_score", "leaf"), getattr(attn, "fsb_rerank_query", "route"),
            getattr(attn, "fsb_union_width", 0), getattr(attn, "fsb_rerank_rr", "none"),
            rr.data_ptr() if rr is not None else None,
            float(getattr(attn, "fsb_read_margin", 0.0) or 0.0), int(getattr(attn, "fsb_read_min", 1) or 1),
            int(getattr(attn, "fsb_beam_quota", 0) or 0),
            float(getattr(attn, "fsb_union_tau_quantile", 0.0) or 0.0),
            (ut.data_ptr(), int(ut._version)) if ut is not None else None,
            int(getattr(attn, "fsb_union_quota", 0) or 0),
            float(getattr(attn, "fsb_union_prior_scale", 1.0)))


@torch.no_grad()
def _forest_budgets(attn, fc, B, H, dev):
    # _forest_token's per-block traversal state of the fixed policy (query-independent: the block's spines, the
    # cell totals and the head scales), refreshed in place at every fsb_decode_prepare so a whole-step decode
    # graph that captured this layer keeps reading the same tensors at the new block's values. Values are exactly
    # those _forest_token (and _forest_token_plain's lazy "bud" branch) computes at token time. Learned policies
    # keep their per-query, per-token budgets: this refreshes nothing and any graph re-captures on their pointers.
    T = B * H
    key = (getattr(fc, "serial", 0), B, H, attn.sel_scale._version, attn.sel_scale.data_ptr(),
           attn.log_sigma._version, attn.log_sigma.data_ptr(), attn.fsb_width,
           attn.fsb_delta, int(getattr(attn, "fsb_union_width", 0) or 0))
    tk = getattr(fc, "_tok", None)
    if attn.fsb_policy_mode != "fixed" or fc.SP is None or fc.F is None:
        return tk
    if tk is not None and tk["key"] == key:
        return tk
    utau = getattr(fc, "utau", None)
    h_main = getattr(fc, "_height_h", None) if getattr(fc, "lean", False) else None
    h_utau = (getattr(utau, "_height_h", None) if getattr(utau, "lean", False) else None) if utau is not None else ()
    cfg_key = key[1:] + (union_tau_on(attn),)
    if h_main is not None and h_utau is not None:
        d_cfg = int(attn.fsb_delta)
        w_cfg = int(attn.fsb_width)
        uw_cfg = int(getattr(attn, "fsb_union_width", 0) or 0)
        eff_main = tuple((min(w_cfg, 1 << max(0, h - d_cfg)), min(h, d_cfg), min(uw_cfg, 1 << max(0, h - d_cfg))) for h in h_main)
        eff_utau = tuple((min(uw_cfg, 1 << max(0, h - d_cfg)), min(h, d_cfg)) for h in h_utau)
        h_key = (eff_main, eff_utau)
    else:
        h_key = None
    if tk is not None and h_key is not None and tk.get("_cfg_key") == cfg_key:
        ukw = tk.get("ukw")
        if ukw and utau is not None:
            ukw["union_F"] = utau.F
            ukw["union_SP"] = utau.SP
        if tk.get("_h_key") == h_key:
            tk["key"] = key
            return tk
        if tk.get("_eff_main") != eff_main:
            tk["W"].copy_(torch.tensor([[w_t] for w_t, _, _ in eff_main], dtype=torch.long))
            tk["delta"].copy_(torch.tensor([[d_t] for _, d_t, _ in eff_main], dtype=torch.long))
            tk["bud"].copy_(torch.tensor([[1 << d_t] for _, d_t, _ in eff_main], dtype=torch.long))
        if uw_cfg > 0:
            if utau is not None and union_tau_on(attn):
                if ukw is None:
                    ukw = {"union_F": utau.F, "union_SP": utau.SP}
                if tk.get("_eff_utau") != eff_utau or "union_delta" not in ukw or "union_W" not in ukw:
                    du_c = torch.tensor([[du_t] for _, du_t in eff_utau], dtype=torch.long)
                    uw_c = torch.tensor([[uw_t] for uw_t, _ in eff_utau], dtype=torch.long)
                    bu_c = torch.tensor([[1 << du_t] for _, du_t in eff_utau], dtype=torch.long)
                    if "union_delta" in ukw:
                        ukw["union_delta"].copy_(du_c)
                    else:
                        ukw["union_delta"] = du_c.to(dev)
                    if "union_W" in ukw:
                        ukw["union_W"].copy_(uw_c)
                    else:
                        ukw["union_W"] = uw_c.to(dev)
                    tk["bud_u"].copy_(bu_c)
                tk["ukw"] = ukw
            elif tk.get("_eff_main") != eff_main and tk.get("union_W") is not None:
                tk["union_W"].copy_(torch.tensor([[uw_t] for _, _, uw_t in eff_main], dtype=torch.long))
                tk["bud_u"].copy_(torch.tensor([[1 << d_t] for _, d_t, _ in eff_main], dtype=torch.long))
        tk["key"] = key
        tk["_h_key"] = h_key
        tk["_eff_main"] = eff_main
        tk["_eff_utau"] = eff_utau
        return tk
    head_id = torch.arange(H, device=dev)[None, :].expand(B, H).reshape(T)
    lam = attn.sel_scale.float()[head_id].view(T, 1, 1)
    sig = attn.sigma
    sigv = sig.float()[head_id].view(T, 1, 1)
    W, D, _ = choose_actions(attn, lam, head_id, fc.m_tot, fc.SP["height"], fc.F["Kmax"] + 1, training=False)
    delta = (fc.SP["height"][:, None] - D).clamp(min=0)
    bud = torch.bitwise_left_shift(torch.ones_like(delta), delta)
    ukw = union_tau_kw(attn, utau.F, utau.SP, 1) if utau is not None else {}
    uw = int(getattr(attn, "fsb_union_width", 0) or 0)
    union_W = bud_u = None
    if uw > 0:
        du = ukw["union_delta"] if ukw else delta
        if not ukw:
            Du = (fc.SP["height"][:, None] - du).clamp(min=0)
            union_W = torch.minimum(torch.full_like(Du, uw), torch.bitwise_left_shift(torch.ones_like(Du), Du))
        else:
            Du = (ukw["union_SP"]["height"][:, None] - du).clamp(min=0)
            union_W = torch.minimum(torch.full_like(Du, uw), torch.bitwise_left_shift(torch.ones_like(Du), Du))
        bud_u = torch.bitwise_left_shift(torch.ones_like(du), du)

    def _put(old_t, new_t):
        if old_t is not None and old_t.shape == new_t.shape and old_t.dtype == new_t.dtype:
            old_t.copy_(new_t)
            return old_t
        return new_t

    if tk is None:
        tk = {}
    ukw_old = tk.get("ukw") or {}
    tk["head_id"] = _put(tk.get("head_id"), head_id)
    tk["lam"] = _put(tk.get("lam"), lam)
    tk["sigma"] = _put(tk.get("sigma"), sig)
    tk["sigv"] = _put(tk.get("sigv"), sigv)
    tk["W"] = _put(tk.get("W"), W)
    tk["delta"] = _put(tk.get("delta"), delta)
    tk["bud"] = _put(tk.get("bud"), bud)
    if ukw:
        ukw["union_delta"] = _put(ukw_old.get("union_delta"), ukw["union_delta"])
        if union_W is not None:
            ukw["union_W"] = _put(ukw_old.get("union_W"), union_W)
        tk.pop("union_W", None)
    else:
        tk["union_W"] = _put(tk.get("union_W"), union_W) if union_W is not None else None
    if bud_u is not None:
        tk["bud_u"] = _put(tk.get("bud_u"), bud_u)
    tk["ukw"] = ukw
    tk["key"] = key
    tk["_cfg_key"] = cfg_key
    tk["_h_key"] = h_key
    tk["_eff_main"] = h_key[0] if h_key is not None else None
    tk["_eff_utau"] = h_key[1] if h_key is not None else None
    fc._tok = tk
    return tk



def _plain_consumption(attn, fc) -> bool:
    # _forest_token's fused plain-path decision recomputed on the host (which tensors a cached whole-step graph
    # reads follows from it: the plain path reads the stable cellA slot, the reference paths the fresh F entries)
    from .fsb_fast_step import fast_beam_ok, fast_union_rerank_ok
    from .fsb_traverse_triton import _HAVE_TRITON
    tk = getattr(fc, "_tok", None)
    if not (_HAVE_TRITON and os.environ.get("FSB_TRAVERSE_TRITON", "1") == "1"
            and attn.fsb_policy_mode == "fixed" and tk is not None
            and getattr(attn, "fsb_read_top", 0) > 0
            and getattr(attn, "fsb_rerank_score", "leaf") != "leaf"):
        return False
    if (hybrid_on(attn) or int(getattr(attn, "fsb_beam_quota", 0) or 0) > 0
            or getattr(fc, "pool", None) is not None or union_quota_on(attn) or union_prior_on(attn)
            or rr_kw(attn) or margin_kw(attn)):
        return False
    F, SP = fc.F, fc.SP
    if F is None or SP is None:
        return False
    uw = int(getattr(attn, "fsb_union_width", 0) or 0)
    ukw = tk["ukw"]
    Fu, SPu = (ukw["union_F"], ukw["union_SP"]) if ukw else (F, SP)
    q0 = tk["lam"]
    d_eff = beam_delta_eff(attn)
    d_max = attn.fsb_delta_max if d_eff is None else d_eff
    if not (fast_beam_ok(F, SP, q0, attn.fsb_W_max, d_max)
            and (uw == 0 or fast_beam_ok(Fu, SPu, q0, attn.fsb_union_W_max, d_max))):
        return False
    R = 1 << d_max
    ca = attn.fsb_W_max * R
    cb = (attn.fsb_union_W_max * R) if uw > 0 else 0
    rr = getattr(attn, "fsb_rerank_score", "leaf")
    return fast_union_rerank_ok(ca, cb, q0, True, attn.cell_stride, int(attn.fsb_read_top), rr)


def fsb_graph_registry(attn, cache):
    # the identity of every tensor a captured whole-step decode program reads through this layer's forest cache,
    # plus the host-side branch bits that selected them. FSBGraphedDecoder replays the previous block's graph only
    # while the registry is unchanged: fsb_decode_prepare refreshes the slot tensors in place (same pointers, new
    # values); any structural change (arena growth, path switches, live flips) forces a re-capture.
    def sig(t):
        return None if not torch.is_tensor(t) else (t.data_ptr(), tuple(t.shape), t.stride())
    fc = getattr(cache, "_fsb_forest", None)
    if fc is None:
        return None
    F, SP, tk = fc.F, fc.SP, getattr(fc, "_tok", None)
    plain = _plain_consumption(attn, fc)
    reg = [bool(getattr(fc, "_live", False)), F is None, SP is None, getattr(fc, "pool", None) is None, plain,
           sig(getattr(cache, "_k_buf", None)), sig(getattr(cache, "_v_buf", None)),
           sig(getattr(cache, "_qk_buf", None)), sig(getattr(cache, "_qv_buf", None)),
           sig(getattr(cache, "_k_hot", None)), sig(getattr(cache, "_v_hot", None)),
           sig(getattr(cache, "_nq_t", None))]
    if F is not None:
        reg += [sig(F.get("node_mu")), sig(F.get("node_aux")), sig(F.get("leaf")), sig(F.get("adm")),
                F.get("Kmax"), F.get("M"), F.get("Ltot")]
        if plain and getattr(fc, "_cellA_g", None) is not None:
            reg += [sig(fc._cellA_g)]
        else:
            reg += [sig(F.get("cellA")), sig(F.get("cellK"))]
    if SP is not None:
        reg += [sig(SP["node"]), sig(SP["root"]), sig(SP["t_idx"]), sig(SP["height"]), sig(fc.m_tot)]
    if tk is not None:
        reg += [sig(tk.get("W")), sig(tk.get("bud")), sig(tk.get("bud_u")), sig(tk.get("union_W")),
                sig(tk.get("lam")), sig(tk.get("sigma")), sig(tk.get("sigv")), sig(tk.get("head_id"))]
        ukw = tk.get("ukw") or {}
        reg += [sig(ukw.get("union_delta")), sig(ukw.get("union_W"))]
    u = getattr(fc, "utau", None)
    reg.append(u is None)
    if u is not None:
        uF, uSP = u.F, u.SP
        reg += [bool(getattr(u, "_live", False)), uF is None, uSP is None]
        if uF is not None:
            reg += [sig(uF.get("node_mu")), sig(uF.get("node_aux")), sig(uF.get("leaf")), sig(uF.get("adm")),
                    sig(u.m_tot), uF.get("Kmax"), uF.get("M"), uF.get("Ltot")]
        if uSP is not None:
            reg += [sig(uSP["node"]), sig(uSP["root"]), sig(uSP["t_idx"]), sig(uSP["height"])]
    return tuple(reg)


@torch.no_grad()
def fsb_decode_prepare_all(attns, caches, B, H, Dh, dev, pos):
    T = B * H
    staged = []
    for attn, cache in zip(attns, caches):
        s = attn.cell_stride
        Bq = attn._cell_block
        blk_start = (pos // Bq) * Bq
        fc = getattr(cache, "_fsb_forest", None)
        if fc is None:
            fc = ForestCache(T, Dh, dev, lean=os.environ.get("FSB_DECODE_LEAN", "1") == "1")
            cache._fsb_forest = fc
        if fc.fast and fc.lean and getattr(cache, "_k_buf", None) is not None:
            fc.m_cap = max(fc.m_cap, cache._k_buf.shape[2] // s)
        if fc.fast and fc.lean and getattr(cache, "_qk_buf", None) is not None:
            fc.m_cap = max(fc.m_cap, cache._qk_buf.shape[2] // s)
        g_cap = getattr(cache, "_fsb_graph_m_cap", 0)
        if fc.fast and fc.lean and g_cap > 0:
            fc.m_cap = max(fc.m_cap, g_cap)
            fc.k_cap = max(fc.k_cap, int(math.floor(math.log2(max(g_cap, 1)))))
        if fc.fast and fc.lean and fc.blk_start != blk_start:
            _, ceil_all, ncell = cache.completed_cells()
            M_vis = min(blk_start // s, ncell) if ncell > 0 else 0
            if M_vis > 0 and ceil_all.is_cuda and not torch.cuda.is_current_stream_capturing():
                if M_vis > fc.n_seen:
                    tau_m = attn.fsb_tau.view(1, H).expand(B, H).reshape(T)
                    nn = fc._append_admitted(ceil_all, M_vis, tau_m)
                    if fc._mc_buf is not None and nn is fc._mc_nn:
                        staged.append(fc)
                    elif nn is not None:
                        fc._pend_mc = (nn.cpu().tolist(), fc.m_tot.cpu().tolist())
                if union_tau_on(attn):
                    if fc.utau is None:
                        fc.utau = ForestCache(T, Dh, dev, lean=fc.lean)
                    fc.utau._pad_src = weakref.ref(fc)
                    fc.utau.m_cap = fc.m_cap
                    fc.utau.k_cap = fc.k_cap
                    if M_vis > fc.utau.n_seen:
                        tau_u = attn.fsb_tau_union.view(1, H).expand(B, H).reshape(T)
                        nnu = fc.utau._append_admitted(ceil_all, M_vis, tau_u)
                        if fc.utau._mc_buf is not None and nnu is fc.utau._mc_nn:
                            staged.append(fc.utau)
                        elif nnu is not None:
                            fc.utau._pend_mc = (nnu.cpu().tolist(), fc.utau.m_tot.cpu().tolist())
    if staged:
        all_mc = torch.stack([f._mc_buf for f in staged]).cpu().tolist()
        for f, mc_val in zip(staged, all_mc):
            f._pend_mc = mc_val
    return [fsb_decode_prepare(a, c, B, H, Dh, dev, pos) for a, c in zip(attns, caches)]


@torch.no_grad()
def fsb_decode_prepare(attn, cache, B, H, Dh, dev, pos):
    # bring the decode forests of this layer to the query block of position pos (ingest the newly completed cells,
    # spines, second forest, pool) -> (forest cache, whether the block has anything to read)
    s = attn.cell_stride
    Bq = attn._cell_block
    blk_start = (pos // Bq) * Bq
    T = B * H
    fc = getattr(cache, "_fsb_forest", None)
    if fc is None:
        fc = ForestCache(T, Dh, dev, lean=os.environ.get("FSB_DECODE_LEAN", "1") == "1"); cache._fsb_forest = fc
    if fc.fast and fc.lean and getattr(cache, "_k_buf", None) is not None:
        fc.m_cap = max(fc.m_cap, cache._k_buf.shape[2] // s)
    if fc.fast and fc.lean and getattr(cache, "_qk_buf", None) is not None:
        fc.m_cap = max(fc.m_cap, cache._qk_buf.shape[2] // s)
    g_cap = getattr(cache, "_fsb_graph_m_cap", 0)
    if fc.fast and fc.lean and g_cap > 0:
        fc.m_cap = max(fc.m_cap, g_cap)
        fc.k_cap = max(fc.k_cap, int(math.floor(math.log2(max(g_cap, 1)))))
    cfg = _decode_sel_cfg(attn)

    def _utau_block(Kbar_all, ceil_all, M_vis, fresh=False):
        if not union_tau_on(attn) or M_vis <= 0:
            fc.utau = None
            return
        if fresh or fc.utau is None:
            fc.utau = ForestCache(T, Dh, dev, lean=fc.lean)
        fc.utau._pad_src = weakref.ref(fc)
        fc.utau.m_cap = fc.m_cap
        fc.utau.k_cap = fc.k_cap
        fc.utau.ingest(Kbar_all, ceil_all, M_vis, attn.fsb_tau_union.view(1, H).expand(B, H).reshape(T),
                       spread_all=cache.completed_spread() if attn.fsb_leaf_radius else None)
        fc.utau._stable_spines()
        fc.utau.live = (fc.utau._m_max_h if fc.utau.lean and fc.utau._m_max_h is not None
                        else int(fc.utau.m_tot.max().item())) > 0

    if fc.blk_start != blk_start:
        Kbar_all, ceil_all, ncell = cache.completed_cells()
        M_vis = min(blk_start // s, ncell) if ncell > 0 else 0
        if M_vis > 0:
            tau_m = attn.fsb_tau.view(1, H).expand(B, H).reshape(T)
            fc.ingest(Kbar_all, ceil_all, M_vis, tau_m,
                      spread_all=cache.completed_spread() if attn.fsb_leaf_radius else None,
                      keep_cells=getattr(attn, "fsb_read_top", 0) > 0)
            fc._stable_spines()
            fc.live = (fc._m_max_h if fc.lean and fc._m_max_h is not None else int(fc.m_tot.max().item())) > 0
            if hybrid_on(attn):
                fc.pool = hybrid_pool(pool_order(fc.F["cellA"]), torch.arange(T, device=dev),
                                      torch.full((T,), M_vis, dtype=torch.long, device=dev), attn.fsb_pool_size,
                                      attn.fsb_pool_recent)
            _utau_block(Kbar_all, ceil_all, M_vis)
        else:
            fc.SP = None
            fc.pool = None
            fc.utau = None
        fc.blk_start = blk_start
        fc.graph = None
        fc.serial = getattr(fc, "serial", 0) + 1
    elif getattr(fc, "sel_cfg", cfg) != cfg:
        Kbar_all, ceil_all, ncell = cache.completed_cells()
        M_vis = min(blk_start // s, ncell) if ncell > 0 else 0
        if fc.SP is not None and hybrid_on(attn):
            fc.pool = hybrid_pool(pool_order(fc.F["cellA"]), torch.arange(T, device=dev),
                                  torch.full((T,), M_vis, dtype=torch.long, device=dev), attn.fsb_pool_size,
                                  attn.fsb_pool_recent)
        else:
            fc.pool = None
        _utau_block(Kbar_all, ceil_all, M_vis if fc.SP is not None else 0, fresh=True)
        fc.graph = None
        fc.serial = getattr(fc, "serial", 0) + 1
    fc._pad_buf = None
    fc.sel_cfg = cfg
    live = not (fc.SP is None or (not fc.live and not hybrid_on(attn)
                                  and not (getattr(fc, "utau", None) is not None and fc.utau.live)))
    fc._live = live
    _forest_budgets(attn, fc, B, H, dev)
    return fc, live


@torch.no_grad()
def fsb_decode_token(attn, q_t, cache, sigma, pos_t, pos_f=None, sr=None):
    # the per-token FS-B read of a whole-step decode graph (the block's forests prepared by fsb_decode_prepare)
    fc = cache._fsb_forest
    B, H, _, Dh = q_t.shape
    if not getattr(fc, "_live", False):
        fsb_read_acc(attn, torch.zeros(B * H, device=q_t.device))
        return q_t.new_zeros(B, H, 1, Dh)
    from .fsb_route import route_radius
    out = _forest_token(attn, q_t, cache, sigma, pos_t, fc, c_h=route_radius(attn), pos_f=pos_f, sr=sr)
    if getattr(attn, "_fsb_read_acc", None) is not None:          # the read statistics of the eager path, in-graph
        fsb_read_acc(attn, (fc._last_sel >= 0).sum(-1))
        if getattr(fc, "utau", None) is not None or union_quota_on(attn):
            fsb_utau_acc(attn, getattr(fc, "_last_prov", None))
    return out


@torch.no_grad()
def _fsb_decode_step_forest(attn, q_t, cache, sigma, pos):
    B, H, _, Dh = q_t.shape
    T = B * H
    dev = q_t.device
    fc, live = fsb_decode_prepare(attn, cache, B, H, Dh, dev, pos)
    stats = getattr(attn, "_fsb_read_acc", None) is not None
    if not live:
        fsb_read_acc(attn, torch.zeros(T, device=dev))
        return q_t.new_zeros(B, H, 1, Dh)
    use_graph = (os.environ.get("FSB_DECODE_GRAPH", "1") == "1" and q_t.is_cuda
                 and cache._k_buf is not None and not getattr(attn, "_fsb_graph_broken", False))
    from .fsb_route import route_radius
    c_now = route_radius(attn)
    if not use_graph:
        out = _forest_token(attn, q_t, cache, sigma, pos, fc, c_h=c_now)
        if stats:
            fsb_read_acc(attn, (fc._last_sel >= 0).sum(-1))
            if getattr(fc, "utau", None) is not None or union_quota_on(attn):
                fsb_utau_acc(attn, getattr(fc, "_last_prov", None))
        return out
    G = getattr(fc, "graph", None)
    if G is not None and (G["kptr"] != cache._k_buf.data_ptr() or G["vptr"] != cache._v_buf.data_ptr()
                          or G["shape"] != tuple(q_t.shape) or G["dtype"] != q_t.dtype):
        G = None
    if G is None:
        q_s = q_t.clone(); pos_s = torch.tensor(float(pos), device=dev); sig_s = sigma.detach().clone()
        c_s = c_now.clone() if c_now is not None else None
        try:
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                _forest_token(attn, q_s, cache, sig_s, pos_s, fc, c_h=c_s)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            out_s = _capture_decode_graph(g, lambda: _forest_token(attn, q_s, cache, sig_s, pos_s, fc, c_h=c_s))
        except RuntimeError as exc:
            torch.cuda.synchronize()
            fc.graph = None
            attn._fsb_graph_broken = True
            print(f"[fsb-graph] capture failed at pos {pos} ({type(exc).__name__}: {exc}); "
                  "falling back to eager decode for this module", flush=True)
            out = _forest_token(attn, q_t, cache, sigma, pos, fc, c_h=c_now)
            if stats:
                fsb_read_acc(attn, (fc._last_sel >= 0).sum(-1))
                if getattr(fc, "utau", None) is not None or union_quota_on(attn):
                    fsb_utau_acc(attn, getattr(fc, "_last_prov", None))
            return out
        G = {"g": g, "q": q_s, "pos": pos_s, "sigma": sig_s, "c": c_s, "out": out_s, "kptr": cache._k_buf.data_ptr(),
             "vptr": cache._v_buf.data_ptr(), "shape": tuple(q_t.shape), "dtype": q_t.dtype}
        fc.graph = G
    G["q"].copy_(q_t); G["pos"].fill_(float(pos)); G["sigma"].copy_(sigma)
    if G.get("c") is not None:
        G["c"].copy_(c_now)
    G["g"].replay()
    out = G["out"].clone()
    if stats:
        fsb_read_acc(attn, (fc._last_sel >= 0).sum(-1))
        if getattr(fc, "utau", None) is not None or union_quota_on(attn):
            fsb_utau_acc(attn, getattr(fc, "_last_prov", None))
    if os.environ.get("FSB_DECODE_GRAPH_CHECK", "0") == "1":
        sel_g = fc._last_sel.clone()
        ref = _forest_token(attn, q_t, cache, sigma, pos, fc, c_h=c_now)
        d = (out - ref).abs().max().item(); ds = int((sel_g != fc._last_sel).sum().item())
        if d > 0 or ds > 0:
            print(f"[fsb-graph-check] pos {pos}: replay != eager, out max diff {d:.3e}, sel slots differing {ds}", flush=True)
    return out


def _fsb_layers(model):
    for m in model.modules():
        if getattr(m, "fsb_enabled", False):
            yield m


def fsb_reward_tokens(loss_tok, l0_tok, mode="loss", clip=5.0):
    if mode == "loss":
        return loss_tok
    assert mode == "gain" and l0_tok is not None, mode
    return -(l0_tok.detach().float() - loss_tok.detach().float()).clamp(0.0, float(clip))


def fsb_policy_loss(model, loss_tok, eta=0.1, entropy_beta=0.01, value_coef=0.5, adv_norm=False, stats=None):
    total = loss_tok.new_zeros(())
    lt = loss_tok.detach().float()[:, None, :]
    acc = {"Z": 0.0, "value": 0.0, "adv_raw_std": 0.0, "adv_abs": 0.0, "reward": 0.0, "n_layers": 0} if stats is not None else None
    for m in _fsb_layers(model):
        st = m._fsb_stats
        if not st or "logp" not in st:
            continue
        mask = st["has"].float()
        n = mask.sum().clamp(min=1.0)
        Z = lt + eta * st["cost"]
        adv = (Z - st["value"]).detach()
        if acc is not None:
            with torch.no_grad():
                mu_r = (adv * mask).sum() / n
                acc["Z"] += float((Z.detach() * mask).sum() / n); acc["value"] += float((st["value"].detach() * mask).sum() / n)
                acc["adv_raw_std"] += float((((adv - mu_r) ** 2) * mask).sum().div(n).sqrt())
                acc["reward"] += float((-lt.expand_as(mask) * mask).sum() / n); acc["n_layers"] += 1
        if adv_norm:
            mu = (adv * mask).sum() / n
            sd = (((adv - mu) ** 2) * mask).sum().div(n).sqrt()
            adv = (adv - mu) / (sd + 1e-6)
        if acc is not None:
            acc["adv_abs"] += float((adv.abs() * mask).sum() / n)
        pl = (adv * st["logp"] * mask).sum() / n
        vl = ((st["value"] - Z.detach()) ** 2 * mask).sum() / n
        el = (st["entropy"] * mask).sum() / n
        total = total + pl + value_coef * vl - entropy_beta * el
        if getattr(m, "fsb_rl", "reinforce") == "ppo" and m._fsb_ppo_rec:
            _fsb_ppo_finalize(m, adv, Z.detach(), st["has"], n)
    if acc is not None and acc["n_layers"]:
        nl = acc.pop("n_layers")
        stats.update({k_: v_ / nl for k_, v_ in acc.items()})
    return total


def _fsb_ppo_finalize(m, adv, Z, has, n):
    B, H, N = adv.shape
    Npad = max(rec["cols"][1] for rec in m._fsb_ppo_rec)
    if Npad > N:
        adv, Z, has = (F.pad(t_, (0, Npad - N)) for t_ in (adv, Z, has))
    for rec in m._fsb_ppo_rec:
        c0, c1 = rec.pop("cols")
        Tc, Bq = rec["logp_old"].shape
        rec["adv"] = adv[:, :, c0:c1].reshape(Tc, Bq).contiguous()
        rec["Z"] = Z[:, :, c0:c1].reshape(Tc, Bq).contiguous()
        rec["mask"] = has[:, :, c0:c1].reshape(Tc, Bq).contiguous()
        rec["n"] = n.detach()
        m._fsb_ppo_store.append(rec)
    m._fsb_ppo_rec = []
    m._fsb_ppo_nmb += 1


def fsb_ppo_surrogate(model, clip=0.2, entropy_beta=0.01, value_coef=0.5, group_queries=1 << 20):
    total = None
    dev = None
    acc = None
    for m in _fsb_layers(model):
        R = m._fsb_ppo_store
        if not R:
            continue
        nmb = max(m._fsb_ppo_nmb, 1)
        dev = R[0]["logp_old"].device
        if acc is None:
            acc = torch.zeros(4, device=dev); mdev = torch.zeros((), device=dev)
        groups, cur, rows = [], [], 0
        for r_ in R:
            T_, Bq_ = r_["logp_old"].shape
            if cur and rows + T_ * Bq_ > group_queries:
                groups.append(cur); cur, rows = [], 0
            cur.append(r_); rows += T_ * Bq_
        groups.append(cur)
        for G in groups:
            cat = (lambda k_: torch.cat([r_[k_] for r_ in G], dim=0)) if len(G) > 1 else (lambda k_: G[0][k_])
            with torch.autocast(device_type=dev.type, enabled=False):
                valid = cat("valid")
                logits, value = m.fsb_policy(cat("qn").float(), cat("head_id"), cat("extra"))
                logits = (logits / m.fsb_policy_temp).masked_fill(~valid[:, None, :], NEG)
                logp_all = torch.log_softmax(logits, dim=-1)
                logp = torch.gather(logp_all, -1, cat("action").long()[..., None]).squeeze(-1)
                ent = -(logp_all.exp() * logp_all.masked_fill(~valid[:, None, :], 0.0)).sum(-1)
                mask = cat("mask").float(); adv = cat("adv"); Z = cat("Z")
                w = torch.cat([(1.0 / r_["n"]).expand(r_["logp_old"].shape[0]) for r_ in G])[:, None] / nmb
                wm = w * mask
                ratio = torch.exp(logp - cat("logp_old"))
                surr = torch.maximum(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv)
                term = (surr * wm).sum() + value_coef * ((value - Z) ** 2 * wm).sum() - entropy_beta * (ent * wm).sum()
            total = term if total is None else total + term
            with torch.no_grad():
                d_ = (ratio - 1.0).abs() * mask
                acc += torch.stack([mask.sum(), (ratio * mask).sum(), ((d_ > clip).float() * mask).sum(),
                                    (((ratio - 1.0) - torch.log(ratio)) * mask).sum()])
                mdev = torch.maximum(mdev, d_.max())
    if total is None:
        return torch.zeros((), device=dev), {"n": 0}
    n_all, r_sum, clipped, kl = acc.tolist(); n_all = max(n_all, 1.0)
    return total, {"n": n_all, "ratio_mean": r_sum / n_all, "ratio_maxdev": float(mdev), "clipfrac": clipped / n_all,
                   "approx_kl": kl / n_all}


def fsb_ppo_clear(model):
    for m in _fsb_layers(model):
        m._fsb_ppo_rec = []; m._fsb_ppo_store = []; m._fsb_ppo_nmb = 0


def fsb_ppo_store_bytes(model):
    tot = 0
    for m in _fsb_layers(model):
        for rec in m._fsb_ppo_store + m._fsb_ppo_rec:
            tot += sum(v_.numel() * v_.element_size() for v_ in rec.values() if torch.is_tensor(v_))
    return tot


def fsb_ctrl_params(model):
    return [p_ for n_, p_ in model.named_parameters() if "fsb_policy" in n_]


def fsb_read_acc(attn, n_read):
    acc = getattr(attn, "_fsb_read_acc", None)
    if acc is None:
        return
    n = n_read.detach().reshape(-1).float()
    acc[0].add_(n.sum()); acc[1].add_(float(n.numel())); acc[2].copy_(torch.maximum(acc[2], n.max()))


def fsb_read_stats_reset(model, on=True):
    for m in _fsb_layers(model):
        m._fsb_read_acc = torch.zeros(5, device=m.fsb_tau.device) if on else None


def fsb_read_stats(model):
    accs = [m._fsb_read_acc for m in _fsb_layers(model) if getattr(m, "_fsb_read_acc", None) is not None]
    if not accs:
        return None
    a = torch.stack(accs).cpu()
    rows = float(a[:, 1].sum())
    tot = float(a[:, 0].sum()); ut = float(a[:, 3].sum()); uq = float(a[:, 4].sum())
    return {"mean": tot / max(rows, 1.0), "max": float(a[:, 2].max()), "rows": rows,
            "layer_mean": (a[:, 0] / a[:, 1].clamp(min=1.0)).tolist(),
            "utau_cells": ut, "utau_frac": ut / max(tot, 1.0), "uq_cells": uq, "uq_frac": uq / max(tot, 1.0)}


@torch.no_grad()
def fsb_log_stats(model):
    agg = {"n_read": [], "n_visit": [], "cost": [], "admit_frac": [], "m0": [], "ent": [], "act": []}
    for m in _fsb_layers(model):
        st = m._fsb_stats
        if not st:
            continue
        agg["admit_frac"].append(st["admit_frac"].mean().item())
        agg["m0"].append(st["m0_mean"].item())
        if "has" not in st:
            continue              # fsb_prefill's eval prefill keeps only these two without FSB_PREFILL_STATS=1
        mask = st["has"].float(); n = mask.sum().clamp(min=1.0)
        agg["n_read"].append(((st["n_read"] * mask).sum() / n).item())
        agg["n_visit"].append(((st["n_visit"] * mask).sum() / n).item())
        agg["cost"].append(((st["cost"] * mask).sum() / n).item())
        if "entropy" in st:
            agg["ent"].append(((st["entropy"] * mask).sum() / n).item())
            nA = int(m.fsb_actions.shape[0])
            hist = torch.bincount(st["action"][st["has"]].flatten(), minlength=nA).float()
            agg["act"].append(hist / hist.sum().clamp(min=1.0))
    out = {k: (sum(v) / len(v) if v else float("nan")) for k, v in agg.items() if k != "act"}
    if agg["act"]:
        out["act_hist"] = torch.stack(agg["act"]).mean(0).tolist()
    return out


@torch.no_grad()
def fsb_calibrate_tau(model, batches, quantile, reduce_fn=None, target="fsb_tau"):
    layers = list(_fsb_layers(model))
    if not layers or quantile <= 0:
        return None
    if target not in ("fsb_tau", "fsb_tau_union"):
        raise ValueError(f"fsb_calibrate_tau: target {target!r} (fsb_tau | fsb_tau_union)")
    for m in layers:
        m._fsb_calib = []
    was_training = model.training
    model.eval()
    for x in batches:
        model(x)
    model.train(was_training)
    taus, fracs, n_cells = [], [], 0
    for m in layers:
        buf = getattr(m, target)
        a = torch.cat(m._fsb_calib, dim=1)
        tau = torch.quantile(a.float(), float(quantile), dim=1)
        if reduce_fn is not None:
            tau = reduce_fn(tau.to(buf.device))
        buf.copy_(tau.to(buf.device, buf.dtype))
        t_ = buf.detach().float().cpu().clone()
        taus.append(t_); fracs.append((a.float() >= t_[:, None]).float().mean(1)); n_cells = int(a.shape[1])
        m._fsb_calib = None
        m._fsb_stats = None
    return {"tau": torch.stack(taus), "admit_frac": torch.stack(fracs), "n_cells": n_cells}
