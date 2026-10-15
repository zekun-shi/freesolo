from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice as _ld              # triton >= 3.1
    except Exception:
        from triton.language.extra.cuda import libdevice as _ld        # triton 3.0
    from .fsb_traverse_triton import _halve_rows
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False

MODES = ("leaf", "lse", "lse_pos", "max")


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


if _HAVE_TRITON:
    @triton.jit
    def _cell_tok_kernel(Q, SEL, TIDX, KEY, OUT, C, Bq, Hh, sb, sh, sl, scale, S,
                         OUTM, OUTK, CELLK, sct, scm,
                         DH: tl.constexpr, DHP: tl.constexpr, SP: tl.constexpr, LSE: tl.constexpr,
                         EXTRA: tl.constexpr):
        # EXTRA (residual re-ranker features): also store the max token score into OUTM and the mean-key score
        # (q . kbar_c) * scale (CELLK[seq, cell] fp32 per-cell means, strides sct / scm) into OUTK; -inf where cell < 0
        pid = tl.program_id(0)                                           # = (t' * Bq + b) * C + candidate
        r = pid // C
        cell = tl.load(SEL + pid)
        if cell >= 0:
            seq = tl.load(TIDX + r // Bq)                                # sequence b*H + h of the tree row
            bi = seq // Hh
            hi = seq - bi * Hh
            d = tl.arange(0, DHP)
            j = tl.arange(0, SP)
            q = tl.load(Q + r * DH + d, mask=d < DH, other=0.0)          # (DHP,) fp32, zero-padded
            tok = cell * S + j
            k = tl.load(KEY + bi * sb + hi * sh + tok[:, None] * sl + d[None, :],
                        mask=(j[:, None] < S) & (d[None, :] < DH), other=0.0).to(tl.float32)
            sc = _halve_rows(k * q[None, :], DHP) * scale                # (SP,) fp32 product, halving dot, * scale
            sc = tl.where(j < S, sc, float("-inf"))
            m = tl.max(sc, axis=0)
            if LSE:
                e = tl.where(j < S, _ld.exp(sc - m), 0.0)
                tot = tl.sum(_halve_rows(tl.reshape(e, (1, SP)), SP), axis=0)
                res = m + _ld.log(tot)
            else:
                res = m
            tl.store(OUT + pid, res)
            if EXTRA:
                kb = tl.load(CELLK + seq * sct + cell * scm + d, mask=d < DH, other=0.0)
                mk = tl.sum(_halve_rows(tl.reshape(kb * q, (1, DHP)), DHP), axis=0) * scale
                tl.store(OUTM + pid, m)
                tl.store(OUTK + pid, mk)
        else:
            tl.store(OUT + pid, float("-inf"))
            if EXTRA:
                tl.store(OUTM + pid, float("-inf"))
                tl.store(OUTK + pid, float("-inf"))


def _key4(tokK: torch.Tensor) -> torch.Tensor:
    # (B,H,L,Dh) or (T,L,Dh) -> (B,H,L,Dh) view with unit last-dim stride (copies only if it is not)
    K4 = tokK if tokK.dim() == 4 else tokK.unsqueeze(0)
    return K4 if K4.stride(-1) == 1 else K4.contiguous()


def use_triton(q: torch.Tensor, s: int) -> bool:
    return (_HAVE_TRITON and q.is_cuda and q.shape[-1] <= 256 and s <= 256
            and os.environ.get("FSB_RERANK_TRITON", "1") == "1")


@torch.no_grad()
def cell_token_scores_torch(q, sel, tokK, t_idx, scale: float, s: int, mode: str, cellK=None) -> torch.Tensor:
    # reference / fallback of cell_token_scores (the same arithmetic), chunked over (row, candidate) entries
    from .fsb_tree import _det_sum
    Tp, Bq, C = sel.shape
    Dh = q.shape[-1]
    K4 = _key4(tokK)
    Hh = K4.shape[1]
    dev = q.device
    n = Tp * Bq * C
    out = torch.full((n,), float("-inf"), dtype=torch.float32, device=dev)
    extra = cellK is not None
    if extra:
        outm = torch.full((n,), float("-inf"), dtype=torch.float32, device=dev)
        outk = torch.full((n,), float("-inf"), dtype=torch.float32, device=dev)
        Mc = cellK.shape[1]
    budget = int(os.environ.get("FSB_RERANK_CHUNK", str(1 << 24)))
    step = max(1, budget // (s * Dh))
    qf = q.float().reshape(Tp * Bq, Dh)
    selr = sel.reshape(n)
    seq_row = t_idx.view(Tp, 1).expand(Tp, Bq).reshape(-1)             # (rows,)
    j = torch.arange(s, device=dev)
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        c = selr[i0:i1]
        ok = c >= 0
        row = torch.arange(i0, i1, device=dev) // C
        sq = seq_row[row]
        bi = torch.div(sq, Hh, rounding_mode="floor"); hi = sq - bi * Hh
        tok = c.clamp(min=0)[:, None] * s + j                          # (n', s)
        k = K4[bi[:, None], hi[:, None], tok]                           # (n', s, Dh) gathered in the key dtype
        sc = _det_sum(k.float() * qf[row][:, None, :], -1) * scale      # (n', s)
        m = sc.amax(-1)
        res = m + torch.log(_det_sum(torch.exp(sc - m[:, None]), -1)) if mode in ("lse", "lse_pos") else m
        out[i0:i1] = torch.where(ok, res, torch.full_like(res, float("-inf")))
        if extra:
            kb = cellK[sq, c.clamp(min=0, max=Mc - 1)].float()              # (n', Dh) the cell's fp32 mean key
            mk = _det_sum(kb * qf[row], -1) * scale
            outm[i0:i1] = torch.where(ok, m, torch.full_like(m, float("-inf")))
            outk[i0:i1] = torch.where(ok, mk, torch.full_like(mk, float("-inf")))
    if extra:
        return out.view(Tp, Bq, C), outm.view(Tp, Bq, C), outk.view(Tp, Bq, C)
    return out.view(Tp, Bq, C)


@torch.no_grad()
def cell_token_scores(q: torch.Tensor, sel: torch.Tensor, tokK: torch.Tensor, t_idx: torch.Tensor,
                      scale: float, s: int, mode: str, cellK=None):
    # token-level score of every candidate -> (T',Bq,C) fp32, -inf where sel < 0; with cellK also (max, mean-key)
    assert mode in ("lse", "lse_pos", "max"), mode
    Tp, Bq, C = sel.shape
    Dh = q.shape[-1]
    extra = cellK is not None
    if Tp * Bq * C == 0:
        e_ = torch.full((Tp, Bq, C), float("-inf"), dtype=torch.float32, device=q.device)
        return (e_, e_.clone(), e_.clone()) if extra else e_
    if not use_triton(q, s):
        return cell_token_scores_torch(q, sel, tokK, t_idx, scale, s, mode, cellK=cellK)
    K4 = _key4(tokK)
    out = torch.empty(Tp, Bq, C, dtype=torch.float32, device=q.device)
    if extra:
        cK = cellK if (cellK.dtype == torch.float32 and cellK.stride(-1) == 1) else cellK.float().contiguous()
        outm = torch.empty_like(out); outk = torch.empty_like(out)
        xargs = (outm, outk, cK, cK.stride(0), cK.stride(1))
    else:
        xargs = (out, out, out, 0, 0)                                    # unused (EXTRA=False specialization)
    _cell_tok_kernel[(Tp * Bq * C,)](q.float().contiguous(), sel.contiguous(), t_idx.contiguous(), K4, out,
                                     C, Bq, K4.shape[1], K4.stride(0), K4.stride(1), K4.stride(2), float(scale), s,
                                     *xargs,
                                     DH=Dh, DHP=_pow2(Dh), SP=_pow2(s), LSE=(mode != "max"), EXTRA=extra,
                                     num_warps=4, enable_fp_fusion=False)
    return (out, outm, outk) if extra else out


# residual re-ranker (fsb_rerank_rr): lse_pos score of a candidate plus a per-(layer, head) residual r(phi)
# columns of fsb_route_rr_w (H, RR_NF); the first four are centred by the row max over the row's valid candidates
RR_FEATURES = ("lse", "max", "meankey", "pos", "a_c", "logdist", "lse_minus_max", "slot", "prov_pool", "prov_both")
RR_NF = len(RR_FEATURES)
RR_MODES = ("none", "linear")
# prov codes of a candidate (prov tensor, same shape as sel): bit 0 = found by the tree, bit 1 = by the pool (None:
# every candidate tree-only); higher bits are informational: the features read only bits 0-1
PROV_TREE, PROV_POOL, PROV_BEAMQ, PROV_UTAU, PROV_UNIONQ = 1, 2, 4, 8, 16


@torch.no_grad()
def rr_inputs(sel, q, qpos, lam, sigv, cellK, cellA, t_idx, scale: float, s: int, tokK):
    # lse_pos inputs of the reranker for candidates sel (T',Bq,C): lse, mx, mk (-inf where sel < 0), a, pos, ok
    Tp, Bq, C = sel.shape
    Mc = cellK.shape[1]
    ok = sel >= 0
    gid = t_idx.view(Tp, 1, 1) * Mc + sel.clamp(min=0, max=Mc - 1)
    lse, mx, mk = cell_token_scores(q, sel, tokK, t_idx, scale, s, "lse_pos", cellK=cellK)
    a = cellA.reshape(-1)[gid.reshape(-1)].view(Tp, Bq, C)
    pos_c = (sel.float() + 0.5) * s
    erf = torch.erf((qpos[..., None] - pos_c).abs() / (2.0 * sigv))
    pos = lam * a * erf
    return {"lse": lse, "mx": mx, "mk": mk, "a": a, "pos": pos, "ok": ok}


def _rowmax_centre(x, ok):
    m = x.masked_fill(~ok, float("-inf")).amax(-1, keepdim=True)         # exact (order-free); -inf for an empty row
    return torch.where(ok, x - m, torch.zeros_like(x))


@torch.no_grad()
def rr_features(inp: dict, sel, qpos, s: int, prov=None):
    # (T',Bq,C,RR_NF) fp32 features in the RR_FEATURES order, 0 on invalid slots (never NaN / inf)
    ok = inp["ok"]
    Tp, Bq, C = sel.shape
    dist = (qpos[..., None] - (sel.float() + 0.5) * s).abs() / float(s)
    slot = (torch.arange(C, device=sel.device, dtype=torch.float32) / float(C)).view(1, 1, C).expand(Tp, Bq, C)
    if prov is None:
        pp = pb = torch.zeros(Tp, Bq, C, dtype=torch.float32, device=sel.device)
    else:
        pv = prov.long()
        pp = ((pv & PROV_POOL) != 0).float()
        pb = (((pv & PROV_POOL) != 0) & ((pv & PROV_TREE) != 0)).float()
    cols = [_rowmax_centre(inp["lse"], ok), _rowmax_centre(inp["mx"], ok), _rowmax_centre(inp["mk"], ok),
            _rowmax_centre(inp["pos"], ok), inp["a"], torch.log1p(dist),
            torch.where(ok, inp["lse"] - inp["mx"], torch.zeros_like(inp["lse"])), slot, pp, pb]
    phi = torch.stack(cols, dim=-1)
    return torch.where(ok[..., None], phi, torch.zeros_like(phi))


@torch.no_grad()
def rr_residual(phi, rr_w, t_idx):
    # r = det-sum over features of phi * w[h]; exactly 0 (+-0.0) where w == 0 (a zero reranker keeps lse_pos order)
    from .fsb_tree import _det_sum
    w = rr_w.detach().float()
    H = w.shape[0]
    wr = w[t_idx % H]                                                    # (T', F)
    return _det_sum(phi * wr[:, None, None, :], -1)
