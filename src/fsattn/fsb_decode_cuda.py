"""The FS-B decode read of one layer as one CUDA kernel (FSB_DECODE_V2=1, the default; 0 keeps the kernel chain).

decode_read_fused runs, for every (batch, head) row of a decode token, both beam descents, the union re-rank, the
top-K selection and the readout with lr_norm and the window combine in one launch (fsb_decode.cu), in place of the
chain of _forest_token's plain path (forest_beam_pair, union_rerank_fast, decode_readout_fast): the same read cells and
provenance, bitwise, and the readout's F8-class numerics. A decode step's beam descents are latency-bound (a dozen
dependent levels of loads); the kernel scores each child whole in one lane, so a level is one round of loads, and
prefetches the next level's candidates into L2 while it scores, and the chain's three dependent launches per layer
become one. It covers the int8 long-range KV store at head dim 64, cells of 32 tokens, beams of at most 16 slots
(2^delta cells each, 32 per beam at most) and 1, 2, 4 or 8 read cells; anything else returns None and the chain runs.

FSB_DECODE_V3=1 (the default; 0 builds the second version) builds the kernel's third version (DEC_V3 in
fsb_decode.cu): the same read cells and outputs bit for bit with a shorter chain per descent. While a level keeps every
child it visits, its nodes follow from the tree's shape alone; those top levels are batched (one round of loads, one
pass of scores, the frontier order replayed), the shape walk is cached per row in the forest's plan buffer
(ForestCache._plan) and reused while the row's shape holds, and the preamble issues the query's load first.

FSB_DECODE_SHARE=1 (the default) builds the kernel with one copy of the descent code for both descents and one copy
of the re-rank code for both re-rank passes (0 keeps a copy per pass), and has the re-rank load each cell's prior
ceiling with its key rows (DEC_SHARE in
fsb_decode.cu): the same read cells and outputs bit for bit, and a launch runs from 80 KB of code instead of 148 KB.
Inside a decode step the other layers' weights evict the kernel's code from L2 between its launches, so every launch
fetches the instructions it runs from DRAM on its critical path.
"""
from __future__ import annotations

import ctypes
import os
import warnings

import torch

from ..utils.compile_guard import compiling
from . import fsb_beam_cuda as bc

_SRC = bc._load_cu_sections("fsb_decode.cu")["_DECODE_KERNEL"]
_NWARP = 16                                              # warps per (batch, head) row
_fns: dict = {}
_failed: list = []                                       # the kernel failed to build: the chain runs from then on
_TIMES = None        # diagnostic hook: an int64 CUDA tensor (rows, 256) -> the DEC_TIMING build records phase clocks
_CELLS = None        # test hook: an int64 CUDA tensor (rows, 2 W_max 2^delta) -> both beams' cells (main, then union)
_LEV0 = None         # test hook: an int32 CUDA tensor (rows, 2) -> the levels each descent's prefix covered (0: none)


def v2_on() -> bool:
    return os.environ.get("FSB_DECODE_V2", "1") == "1" and not _failed and not compiling()


def v3_on() -> bool:
    # FSB_DECODE_V3=1 (the default): the kernel's third version (fsb_decode.cu, DEC_V3), the same read cells and
    # outputs bit for bit; 0 builds the second version
    return os.environ.get("FSB_DECODE_V3", "1") == "1"


def share_on() -> bool:
    # FSB_DECODE_SHARE=1 (the default; 0 builds a separate copy per pass): the build whose descents and re-rank
    # passes share their code and whose re-rank loads the prior ceilings with the key rows (fsb_decode.cu,
    # DEC_SHARE), the same read cells and outputs bit for bit
    return os.environ.get("FSB_DECODE_SHARE", "1") == "1"


class _DecBeam(ctypes.Structure):
    # struct DecBeam of fsb_decode.cu (one beam's tables), passed by value
    _fields_ = ([(k, ctypes.c_void_p) for k in ("ROOT", "WBEAM", "BUDGET", "TIDX", "CHUNK", "CHUNK_AX", "SPINE",
                                                "LEAF", "ADM", "HGT", "PLAN")]
                + [(k, ctypes.c_longlong) for k in ("Ntot", "Sp", "Ltot", "Mm")]
                + [("KMAXP1", ctypes.c_int), ("pad_", ctypes.c_int)])


def decode_source(DH: int, WM: int, R: int, KK: int, bound: bool, rc_t: bool, lse: bool, pos: bool, has_sr: bool,
                  w_bf16: bool, lrs_bf16: bool, timing: bool, prov_bit: int, v3: bool = False,
                  share: bool = False) -> str:
    defs = dict(DH=DH, DHP=DH, NA=9, NC=DH + 9, WM=WM, R=R, KK=KK, SCELL=32, BOUND=int(bound), RC_T=int(rc_t),
                LSE_ON=int(lse), POS_ON=int(pos), HAS_SR=int(has_sr), W_BF16=int(w_bf16), LRS_BF16=int(lrs_bf16),
                DEC_TIMING=int(timing), PROV_BIT=int(prov_bit), NWARP=_NWARP, DEC_V3=int(v3), DEC_SHARE=int(share))
    return "".join(f"#define {k} {v}\n" for k, v in defs.items()) + bc._HELPERS + _SRC


_PLAN_INTS = 256                                         # fsb_decode.cu's PLAN_STRIDE in 32-bit words


def _plan_of(fc, T: int, dev):
    # the forest cache's (rows, _PLAN_INTS) plan buffer of the V3 kernel (zeros: no row's walk cached), made at first
    # use; inside a graph capture a missing one stays missing (the kernel then walks at every launch)
    plan = getattr(fc, "_plan", None)
    if plan is None and not torch.cuda.is_current_stream_capturing():
        plan = fc._plan = torch.zeros((T, _PLAN_INTS), dtype=torch.int32, device=dev)
    return plan


def _beam_struct(F: dict, SP: dict, W: torch.Tensor, bud: torch.Tensor, keep: list, plan=None) -> _DecBeam:
    # plan: the forest's (rows, _PLAN_INTS) int32 cache of the V3 kernel's prefix walk, or None
    root, wb, bd, ti = SP["root"].contiguous(), W.contiguous(), bud.contiguous(), SP["t_idx"].contiguous()
    hgt = SP["height"].contiguous()
    keep += [root, wb, bd, ti, hgt]
    pl = 0
    if (plan is not None and plan.dtype == torch.int32 and plan.is_contiguous() and plan.dim() == 2
            and plan.shape[1] == _PLAN_INTS and plan.shape[0] >= root.shape[0] and plan.device == root.device):
        pl = plan.data_ptr()
    return _DecBeam(root.data_ptr(), wb.data_ptr(), bd.data_ptr(), ti.data_ptr(), F["node_mu"].data_ptr(),
                    F["node_aux"].data_ptr(), SP["node"].data_ptr(), F["leaf"].data_ptr(), F["adm"].data_ptr(),
                    hgt.data_ptr(), pl, F["node_mu"].shape[1], SP["node"].shape[1], F["Ltot"], F["M"],
                    F["Kmax"] + 1, 0)


def _beam_ok(F: dict, SP: dict, W: torch.Tensor, bud: torch.Tensor, T: int, dev) -> bool:
    mu, aux, node = F["node_mu"], F["node_aux"], SP["node"]
    return (mu.dtype == torch.bfloat16 and aux.dtype == node.dtype == torch.float32 and mu.is_contiguous()
            and aux.is_contiguous() and node.is_contiguous() and mu.data_ptr() % 16 == 0
            and mu.shape[-1] == 64 and aux.shape[-1] == 9 and node.shape[-1] == 73
            and mu.shape[0] >= T and node.shape[0] >= T and SP["root"].shape[0] >= T and SP["t_idx"].shape[0] >= T
            and W.dtype == bud.dtype == torch.int64 and tuple(W.shape) == tuple(bud.shape) == (T, 1)
            and F["leaf"].dtype == F["adm"].dtype == torch.int64
            and SP["height"].dtype == torch.int64 and SP["height"].shape[0] >= T
            and all(t.device == dev for t in (mu, aux, node, W, bud, F["leaf"], F["adm"], SP["root"], SP["t_idx"],
                                               SP["height"])))


@torch.no_grad()
def decode_read_fused(attn, fc, tk: dict, cache, q_t: torch.Tensor, q: torch.Tensor, qpos: torch.Tensor,
                      lam: torch.Tensor, sigv: torch.Tensor, rc, rr: str, tokK, ukw: dict, sr, pos_t: torch.Tensor,
                      d_eff):
    # -> (out, read cells (T,1,K) int64, provenance (T,1,K) int8) for one decode query per (batch, head) row, or
    # None when the fused read does not apply (host checks only before the launch; nothing compiles in a capture).
    # q: the beams' and the re-rank's shared query (q_t viewed (T,1,Dh)); out has decode_readout_fast's layout
    from .fsb_fast_step import fast_union_rerank_ok, forest_beam_pair_ok
    from .fsb_rerank import PROV_UTAU
    from .fsb_tree import _beam_views
    from .kv_int8 import Int8Keys
    if not (v2_on() and q_t.is_cuda and q_t.dtype == torch.bfloat16 and q_t.dim() == 4 and q_t.shape[2] == 1
            and q_t.is_contiguous() and getattr(cache, "use_int8", False) and isinstance(tokK, Int8Keys)
            and torch.is_tensor(pos_t) and pos_t.dim() == 0):
        return None
    B, H, _, Dh = q_t.shape
    T, s, K = B * H, int(attn.cell_stride), int(attn.fsb_read_top)
    uw = int(getattr(attn, "fsb_union_width", 0) or 0)
    if not (Dh == 64 and s == 32 and int(cache.s) == 32 and 1 <= K <= 8 and (K & (K - 1)) == 0 and uw > 0
            and rr in ("lse", "lse_pos", "max") and q.shape == (T, 1, Dh) and q.dtype == torch.bfloat16
            and q.is_contiguous()):
        return None
    F, SP = fc.F, fc.SP
    Fu, SPu = (ukw["union_F"], ukw["union_SP"]) if ukw else (F, SP)
    WM, d_max = int(attn.fsb_W_max), int(attn.fsb_delta_max if d_eff is None else d_eff)
    R = 1 << d_max
    if not (forest_beam_pair_ok(F, SP, Fu, SPu, q, WM, attn.fsb_union_W_max, d_max) and WM * R <= 32
            and K <= 2 * WM * R and fast_union_rerank_ok(WM * R, WM * R, q, tokK, s, K, rr)):
        return None
    if "bud" not in tk or "bud_u" not in tk:
        return None
    W, bud, bud_u = tk["W"], tk["bud"], tk["bud_u"]
    Wu = ukw["union_W"] if ukw else tk["union_W"]
    _beam_views(F)
    _beam_views(Fu)
    dev = q_t.device
    if not (_beam_ok(F, SP, W, bud, T, dev) and _beam_ok(Fu, SPu, Wu, bud_u, T, dev)):
        return None
    qk, qv, sk, sv = cache._qk_buf, cache._qv_buf, cache._ks_buf, cache._vs_buf
    khot, vhot, nq_t = cache._k_hot, cache._v_hot, cache._nq_t
    cellA = fc._cellA_g if getattr(fc, "_cellA_g", None) is not None else F["cellA"]
    lnw, lrs = attn.lr_norm.scale, attn.long_range_scale.view(-1)
    if not (qk.dtype == qv.dtype == torch.int8 and sk.dtype == sv.dtype == khot.dtype == vhot.dtype == torch.bfloat16
            and qk.dim() == qv.dim() == 4 and qk.stride(3) == qv.stride(3) == 1 and qk.stride(2) == qv.stride(2) == Dh
            and qk.data_ptr() % 16 == 0 and qv.data_ptr() % 16 == 0 and sk.stride(-1) == sv.stride(-1) == 1
            and khot.stride(-1) == vhot.stride(-1) == 1 and nq_t.dtype == torch.int64 and nq_t.dim() == 0
            and cellA.dtype == torch.float32 and cellA.dim() == 2 and cellA.shape[0] >= T
            and lnw.dtype in (torch.bfloat16, torch.float32) and lnw.is_contiguous() and lnw.numel() == Dh
            and lrs.dtype in (torch.bfloat16, torch.float32) and lrs.is_contiguous()
            and qpos.dtype == lam.dtype == sigv.dtype == torch.float32 and lam.numel() == sigv.numel() == T
            and qpos.dim() == 2 and qpos.shape == (T, 1) and all(t.device == dev for t in (
                qk, qv, sk, sv, khot, vhot, nq_t, cellA, lnw, lrs, qpos, lam, sigv, pos_t))):
        return None
    has_sr = (sr is not None and sr.shape == q_t.shape and sr.dtype == q_t.dtype and sr.stride(3) == 1)
    rc_t = torch.is_tensor(rc)
    w_bf16, lrs_bf16 = lnw.dtype == torch.bfloat16, lrs.dtype == torch.bfloat16
    times = _TIMES
    cfg = (dev.index, Dh, WM, R, K, attn.fsb_node_score == "bound", rc_t, rr != "max", rr == "lse_pos", has_sr,
           w_bf16, lrs_bf16, times is not None, v3_on(), share_on())
    fn = _fns.get(cfg)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            return None                                                      # never compile inside a graph capture
        try:
            fn = _fns[cfg] = bc.load_function(dev, decode_source(*cfg[1:-2], PROV_UTAU, v3=cfg[-2], share=cfg[-1]),
                                              "fsb_decode_fused")
        except Exception as e:                                               # no NVRTC / driver: the chain runs
            if not _failed:
                _failed.append(repr(e))
                warnings.warn(f"FS-B fused decode read unavailable, using the kernel chain: {e!r}")
            return None
    if qpos.stride(0) == 0:
        qp, qp_s = qpos, 0
    else:
        qp, qp_s = qpos.contiguous(), 1
    pos_i = pos_t if pos_t.dtype == torch.int64 else pos_t.to(torch.int64)
    lm, sg = lam.reshape(-1).contiguous(), sigv.reshape(-1).contiguous()
    rct = rc.reshape(-1).float().contiguous() if rc_t else q
    keep = [qp, pos_i, lm, sg, rct]
    # V3: the two descents cache their prefix walks in their own forests' plans (one forest for both: the second walks)
    ut = getattr(fc, "utau", None)
    pa = _plan_of(fc, T, dev) if cfg[-1] else None
    pb = _plan_of(ut, T, dev) if cfg[-1] and ut is not None and Fu is ut.F else None
    BA = _beam_struct(F, SP, W, bud, keep, pa)
    BB = _beam_struct(Fu, SPu, Wu, bud_u, keep, pb)
    q3 = q_t[:, :, 0, :]
    if has_sr:
        out = torch.empty(B, 1, H * Dh, device=dev, dtype=q_t.dtype)
        sob, soh = H * Dh, Dh
        sr3 = sr[:, :, 0, :]
        srp, srb, srh = sr3.data_ptr(), sr3.stride(0), sr3.stride(1)
    else:
        out = torch.empty(B, H, 1, Dh, device=dev, dtype=q_t.dtype)
        sob, soh = out.stride(0), out.stride(1)
        srp, srb, srh = q3.data_ptr(), 0, 0
    sel = torch.empty(T, 1, K, dtype=torch.int64, device=dev)
    pv = torch.empty(T, 1, K, dtype=torch.int8, device=dev)
    P, I64, F32, I32 = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_float, ctypes.c_int
    args = [BA, BB, P(q.data_ptr()), I32(1), P(rct.data_ptr()), F32(1.0 if rc_t else float(rc)), P(qp.data_ptr()),
            I64(qp_s), P(lm.data_ptr()), P(sg.data_ptr()), F32(float(attn.scale)), F32(float(s)),
            P(qk.data_ptr()), P(sk.data_ptr()), P(khot.data_ptr()), P(nq_t.data_ptr()),
            P(cellA.data_ptr()), I64(cellA.stride(0)), I64(cellA.stride(1)), I32(sk.shape[2]),
            I64(qk.stride(0)), I64(qk.stride(1)), I64(qk.stride(2)), I64(sk.stride(0)), I64(sk.stride(1)),
            I64(khot.stride(0)), I64(khot.stride(1)), I64(khot.stride(2)),
            P(qv.data_ptr()), P(sv.data_ptr()), P(vhot.data_ptr()),
            I64(qv.stride(0)), I64(qv.stride(1)), I64(qv.stride(2)), I64(sv.stride(0)), I64(sv.stride(1)),
            I64(vhot.stride(0)), I64(vhot.stride(1)), I64(vhot.stride(2)),
            P(pos_i.data_ptr()), P(lnw.data_ptr()), P(srp), I64(srb), I64(srh), P(lrs.data_ptr()),
            P(out.data_ptr()), I64(sob), I64(soh), F32(float(attn.lr_norm.eps)), I32(H),
            P(sel.data_ptr()), P(pv.data_ptr()), P(_CELLS.data_ptr() if _CELLS is not None else 0),
            P(_LEV0.data_ptr() if _LEV0 is not None else 0), P(times.data_ptr() if times is not None else 0)]
    bc.launch(fn, T, 32 * _NWARP, dev, args)
    return out, sel, pv
