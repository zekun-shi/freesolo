"""Bounded long-range KV storage for the masked FS-L graph decode (FSL_KV_BOUNDED=1; GraphedDecoder only).

Masked FS-L (cell_prune_kernel=1) reads, per head, the cells of its block pool and its local window. The pool of the
query block that starts at token b holds topk(ceil[:M], P) (M = b // s cells completed before the block; a cell's
ceiling is its largest key norm, fixed when the cell completes) and the Wc newest of those M cells; every token of
the block reads the W tokens before it and the open cell. torch.topk takes the cells above its k-th value, then the
cells at it in index order: the P largest under the total order (ceiling descending, index ascending). Later blocks
only add cells to the ranking, so a completed cell outside a block's top P is outside every later top P, and once
it is also older than the cells the block's window and newest Wc reach, no later step reads it: its K / V can go.

The store keeps exactly the cells a later step can still read. Per layer: S slots of s rows (int8 K and V with one
bf16 scale per slot and tensor, kv_int8's per-cell quantization, so every value read is the one the full int8 store
returns; the cell's fp32 mean key for the pool), a cell -> slot and a slot -> cell table (int32, -1 for none), every
completed cell's ceiling (fp32: the top-P input) and the open cell's bf16 rows (the int8 store's hot buffer). Reads
go through the cell -> slot table: token t of completed cell c is row slot[c] * s + t % s. S = P + max(ceil(W / s),
Wc) + Bq / s + 1 slots hold every cell kept at once: after a block's refresh at most its top P and the max(ceil(W /
s), Wc) cells before the block that its window and newest Wc reach, and Bq / s cells complete before the next
refresh.

The bookkeeping runs eagerly between graph replays, on the device and without host syncs: the prefill stores the
cells its first decode block can read (SlotStore.prefill), a completed cell takes the lowest free slot
(store_cell), and each block refresh assembles the pool of trunk_triton.fsl_pool_refresh value for value (the mean
keys gathered through the table) and frees the slots of the cells that left the readable set (pool). An overflow
(no free slot) and a readable pool cell without a slot cannot occur under the bound above; both are counted in a
per-layer device counter (SlotStore.flags) instead of reading out of bounds, and checked by faults().

The decode reads the same values of the same cells as on the full int8 store, so its logits are identical with the
switch on and off (tests/test_kv_bounded.py). A layer of the released models keeps about 6 MB plus the cell-indexed
tables (8 bytes per cell and head), against about 290 MB at 131k tokens with the full int8 store. (The bounded
streaming cache of linear_cache.py, FSL_LINEAR_CACHE=1, is a different configuration: it also bounds the prefill and
selects in its own tie order.)"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as _libdevice
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False


def bounded_on() -> bool:
    return os.environ.get("FSL_KV_BOUNDED", "0") == "1"


def _cfg(attn):
    # (P, Wc, W, Bq, s): top-P width, newest cells, window, query block, cell size of an FS-L layer
    return (attn._prune_cfg("cell_prune_pool", "FS_PRUNE_POOL", 64),
            attn._prune_cfg("cell_prune_window", "FS_PRUNE_WINDOW", 8),
            int(attn.window_size), int(attn._cell_block), int(attn.cell_stride))


def slot_count(attn) -> int:
    P, Wc, W, Bq, s = _cfg(attn)
    return P + max(-(-W // s), Wc) + -(-Bq // s) + 1


def supported(attns, dtype, device) -> bool:
    """Whether every layer's graph decode can run on the bounded store: masked FS-L on CUDA with the int8 store and
    the fused window, pool and long-range decode kernels (their reference fallbacks read the full history)."""
    from ..utils import trunk_triton as _tt
    from .kv_int8 import int8_supported
    if not (_HAVE_TRITON and _tt.available() and torch.device(device).type == "cuda"
            and int8_supported(dtype, device)):
        return False
    if any(os.environ.get(k, "1") == "0" for k in ("FSL_TRUNK_FUSED", "FSL_SR_DECODE_FUSED",
                                                   "FSL_LR_DECODE_FUSED", "FSL_POOL_REFRESH_FUSED")):
        return False
    for a in attns:
        P, Wc, W, Bq, s = _cfg(a)
        half = a.head_dim // 2
        if (getattr(a, "fsb_enabled", False) or a._is_dense_lr() or getattr(a, "sel_content_only", False)
                or not 1 <= a.cell_topk <= 16 or a.head_dim != 64 or s not in (16, 32, 64) or Bq % s
                or P < 1 or Wc < 0 or P + Wc > 128 or not 1 <= W + 1 <= 1024 or half & (half - 1)):
            return False
    return True


class SlotStore:
    """One layer's bounded K / V (see the module docstring). n_cells is the capacity of the cell-indexed tables
    (cells of the whole decode, prompt and generated tokens); n_slots = slot_count(attn) the K / V capacity."""

    def __init__(self, attn, B: int, n_cells: int, device):
        self.P, self.Wc, self.W, self.Bq, self.s = _cfg(attn)
        H, D = attn.n_heads, attn.head_dim
        self.B, self.H, self.D = B, H, D
        self.n_slots = slot_count(attn)
        self.n_cells = max(1, int(n_cells))
        dev = torch.device(device)
        self.qk = torch.zeros(B, H, self.n_slots * self.s, D, dtype=torch.int8, device=dev)
        self.qv = torch.zeros_like(self.qk)
        self.sk = torch.ones(B, H, self.n_slots, dtype=torch.bfloat16, device=dev)
        self.sv = torch.ones_like(self.sk)
        self.kbar = torch.zeros(B, H, self.n_slots, D, dtype=torch.float32, device=dev)
        self.cell_in = torch.full((B, H, self.n_slots), -1, dtype=torch.int32, device=dev)
        self.slot_of = torch.full((B, H, self.n_cells), -1, dtype=torch.int32, device=dev)
        self.ceil = torch.zeros(B, H, self.n_cells, dtype=torch.float32, device=dev)
        self.flags = torch.zeros(2, dtype=torch.int32, device=dev)     # [slot overflows, pool cells without a slot]

    def fits(self, B: int, H: int, D: int, n_cells: int, device) -> bool:
        return ((self.B, self.H, self.D) == (B, H, D) and self.n_cells >= n_cells
                and self.qk.device == torch.device(device))

    def _readable_from(self, blk_start: int, m_vis: int) -> int:
        # the oldest cell the block at blk_start reads outside its top P: its first token's window start, or its
        # oldest of the Wc newest cells
        return max(0, min((blk_start - self.W) // self.s, m_vis - self.Wc))

    @torch.no_grad()
    def prefill(self, K, V, kbar, ceil, N: int):
        """Store the prompt's cells that its first decode block (the block of token N) or a later step can read:
        the top P of the cells completed before that block and every cell from _readable_from on."""
        s = self.s
        ncell = N // s
        self.slot_of.fill_(-1)
        self.cell_in.fill_(-1)
        self.flags.zero_()
        if ncell <= 0:
            return
        if ncell > self.n_cells:
            raise ValueError(f"bounded KV store sized for {self.n_cells} cells, prompt has {ncell}")
        B, H, _, D = K.shape
        self.ceil[:, :, :ncell].copy_(ceil)
        blk = (N // self.Bq) * self.Bq
        m_vis = min(blk // s, ncell)
        keep = torch.zeros((B, H, ncell), dtype=torch.bool, device=K.device)
        keep[:, :, self._readable_from(blk, m_vis) if m_vis > 0 else 0:] = True
        if m_vis > 0:
            # the call the first decode refresh (pool) makes, on the same values: the same cells
            keep.scatter_(2, self.ceil[:, :, :m_vis].topk(min(self.P, m_vis), dim=-1).indices, True)
        slot = torch.cumsum(keep, dim=-1, dtype=torch.int32) - 1
        over = keep & (slot >= self.n_slots)
        self.flags[:1] += over.sum(dtype=torch.int32).view(1)
        self.slot_of[:, :, :ncell].copy_(torch.where(keep & ~over, slot, torch.full_like(slot, -1)))
        del keep, slot, over
        kb = kbar.float()
        _fill_slots_kernel[(ncell, B * H)](
            K, V, kb, self.slot_of, self.qk, self.qv, self.sk, self.sv, self.kbar, self.cell_in, H,
            K.stride(0), K.stride(1), K.stride(2), V.stride(0), V.stride(1), V.stride(2),
            kb.stride(0), kb.stride(1), kb.stride(2),
            self.slot_of.stride(0), self.slot_of.stride(1),
            self.qk.stride(0), self.qk.stride(1), self.qk.stride(2), self.sk.stride(0), self.sk.stride(1),
            self.kbar.stride(0), self.kbar.stride(1), self.kbar.stride(2),
            self.cell_in.stride(0), self.cell_in.stride(1),
            S=s, D=D, DP=triton.next_power_of_2(D), num_warps=4,
        )

    @torch.no_grad()
    def store_cell(self, k_hot, v_hot, kbar, ceil, cell: int):
        """A decode cell completed (the open cell's rows, its mean key (B, H, 1, D) and ceiling (B, H, 1)): record
        the ceiling and quantize the rows into each head's lowest free slot."""
        D = self.D
        if kbar.stride(-1) != 1:
            kbar = kbar.contiguous()
        _store_cell_kernel[(self.B * self.H,)](
            k_hot, v_hot, kbar, ceil, self.qk, self.qv, self.sk, self.sv, self.kbar, self.slot_of, self.cell_in,
            self.ceil, self.flags,
            int(cell), self.H, self.n_slots, self.n_cells,
            k_hot.stride(0), k_hot.stride(1), k_hot.stride(2), v_hot.stride(0), v_hot.stride(1), v_hot.stride(2),
            kbar.stride(0), kbar.stride(1), ceil.stride(0), ceil.stride(1),
            self.qk.stride(0), self.qk.stride(1), self.qk.stride(2), self.sk.stride(0), self.sk.stride(1),
            self.kbar.stride(0), self.kbar.stride(1), self.kbar.stride(2),
            self.slot_of.stride(0), self.slot_of.stride(1), self.cell_in.stride(0), self.cell_in.stride(1),
            self.ceil.stride(0), self.ceil.stride(1),
            S=self.s, D=D, DP=triton.next_power_of_2(D), BLOCK_SL=triton.next_power_of_2(self.n_slots),
            num_warps=4,
        )

    @torch.no_grad()
    def pool(self, blk_start: int, ncell: int, out: dict | None = None) -> dict:
        """FSAttn._build_decode_pool's candidate pool of the block at blk_start (the tensors and values of
        trunk_triton.fsl_pool_refresh, in place in out when it fits), after which the slots of the cells no later
        step reads are freed."""
        s = self.s
        m_vis = min(blk_start // s, ncell, self.n_cells)
        if m_vis <= 0:
            return {"empty": True}
        pk = min(self.P, m_vis)
        top = self.ceil[:, :, :m_vis].topk(pk, dim=-1).indices.contiguous()
        B, H, D = self.B, self.H, self.D
        C = pk + self.Wc
        dev = self.ceil.device
        if (isinstance(out, dict) and not out.get("empty", True) and out.get("C") == C
                and tuple(out["Kbar_cand"].shape) == (B, H, C, D) and out["Kbar_cand"].is_contiguous()
                and out["ceil_cand"].is_contiguous() and out["cen_cand"].is_contiguous()
                and out["ok_cell"].is_contiguous() and out["cand_c"].is_contiguous()):
            res = out
            kt = out.get("Kbar_cand_t")
            if kt is None or tuple(kt.shape) != (B, H, D, C) or not kt.is_contiguous():
                out["Kbar_cand_t"] = torch.empty((B, H, D, C), device=dev, dtype=torch.float32)
            out["M_vis"] = m_vis
        else:
            res = {
                "empty": False, "M_vis": m_vis, "C": C,
                "Kbar_cand": torch.empty((B, H, C, D), device=dev, dtype=torch.float32),
                "Kbar_cand_t": torch.empty((B, H, D, C), device=dev, dtype=torch.float32),
                "ceil_cand": torch.empty((B, H, C), device=dev, dtype=torch.float32),
                "cen_cand": torch.empty((B, H, C), device=dev, dtype=torch.float32),
                "ok_cell": torch.empty((B, H, C), device=dev, dtype=torch.bool),
                "cand_c": torch.empty((B, H, C), device=dev, dtype=torch.int64),
            }
        kc, kt, cl = res["Kbar_cand"], res["Kbar_cand_t"], res["ceil_cand"]
        _pool_slots_kernel[(B * H,)](
            top, self.kbar, self.slot_of, self.cell_in, self.ceil, self.flags,
            kc, kt, cl, res["cen_cand"], res["ok_cell"], res["cand_c"],
            self.kbar.stride(0), self.kbar.stride(1), self.kbar.stride(2),
            self.slot_of.stride(0), self.slot_of.stride(1), self.cell_in.stride(0), self.cell_in.stride(1),
            self.ceil.stride(0), self.ceil.stride(1),
            kc.stride(0), kc.stride(1), kc.stride(2), cl.stride(0), cl.stride(1),
            blk_start // s - 1, float(blk_start), float(s), pk, C, self._readable_from(blk_start, m_vis),
            self.n_slots,
            H=H, D=D, BLOCK_C=max(16, triton.next_power_of_2(C)), BLOCK_D=max(16, triton.next_power_of_2(D)),
            BLOCK_SL=triton.next_power_of_2(self.n_slots), num_warps=4,
        )
        return res

    def held(self) -> torch.Tensor:
        # (B, H) number of cells held per head (a host sync where read)
        return (self.cell_in >= 0).sum(-1)

    def faults(self) -> tuple:
        # (slot overflows, readable pool cells without a slot) so far; a host sync
        f = self.flags.tolist()
        return int(f[0]), int(f[1])

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.qk, self.qv, self.sk, self.sv, self.kbar,
                                                           self.cell_in, self.slot_of, self.ceil, self.flags))


if _HAVE_TRITON:
    from .kv_int8 import _div_rn_f32

    @triton.jit
    def _quant_tile(x):
        # kv_int8._quant_cells_kernel's quantization of one cell, (s, D) fp32 values: scale = bf16(max|x| / 127)
        # (1.0 for an all-zero cell), q = rint(x / scale) as int8, both divisions IEEE
        amax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
        sc = tl.where(amax > 0.0, _div_rn_f32(amax, tl.full((), 127.0, tl.float32)), 1.0)
        sc = sc.to(tl.bfloat16).to(tl.float32)
        q = _libdevice.rint(_div_rn_f32(x, sc)).to(tl.int8)
        return q, sc

    @triton.jit
    def _fill_slots_kernel(K, V, KBAR, SLOT_OF, QK, QV, SK, SV, KB, CELL_IN, H,
                           skb, skh, skn, svb, svh, svn, sbb, sbh, sbc, stb, sth,
                           sqb, sqh, sqn, ssb, ssh, sxb, sxh, sxs, scb, sch,
                           S: tl.constexpr, D: tl.constexpr, DP: tl.constexpr):
        # one prompt cell (program 0) of one head (program 1): quantized into the slot the prefill retention gave it
        # (-1: not kept; the retention already dropped and counted any cell past the last slot), with its mean key
        # and its slot -> cell entry
        c = tl.program_id(0).to(tl.int64)
        bh = tl.program_id(1)
        b = bh // H
        h = bh - b * H
        slot = tl.load(SLOT_OF + b * stb + h * sth + c)
        ok = slot >= 0
        j = tl.arange(0, S)
        d = tl.arange(0, DP)
        m = ok & (j[:, None] >= 0) & (d[None, :] < D)
        rows = c * S + j
        prow = slot.to(tl.int64) * S + j
        k = tl.load(K + b * skb + h * skh + rows[:, None] * skn + d[None, :], mask=m, other=0.0).to(tl.float32)
        qk, sck = _quant_tile(k)
        tl.store(QK + b * sqb + h * sqh + prow[:, None] * sqn + d[None, :], qk, mask=m)
        tl.store(SK + b * ssb + h * ssh + slot, sck.to(tl.bfloat16), mask=ok)
        v = tl.load(V + b * svb + h * svh + rows[:, None] * svn + d[None, :], mask=m, other=0.0).to(tl.float32)
        qv, scv = _quant_tile(v)
        tl.store(QV + b * sqb + h * sqh + prow[:, None] * sqn + d[None, :], qv, mask=m)
        tl.store(SV + b * ssb + h * ssh + slot, scv.to(tl.bfloat16), mask=ok)
        dm = ok & (d < D)
        kb = tl.load(KBAR + b * sbb + h * sbh + c * sbc + d, mask=dm, other=0.0)
        tl.store(KB + b * sxb + h * sxh + slot * sxs + d, kb, mask=dm)
        tl.store(CELL_IN + b * scb + h * sch + slot, c.to(tl.int32), mask=ok)

    @triton.jit(do_not_specialize=["cell"])
    def _store_cell_kernel(HK, HV, KBAR1, CEIL1, QK, QV, SK, SV, KB, SLOT_OF, CELL_IN, CEIL, FLAGS,
                           cell, H, n_slots, n_cells,
                           shb, shh, shn, svb, svh, svn, s1b, s1h, c1b, c1h,
                           sqb, sqh, sqn, ssb, ssh, sxb, sxh, sxs, stb, sth, scb, sch, seb, seh,
                           S: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, BLOCK_SL: tl.constexpr):
        # the completed decode cell of one head: its ceiling into the cell table, its rows quantized into the lowest
        # free slot, and both table entries
        bh = tl.program_id(0)
        b = bh // H
        h = bh - b * H
        si = tl.arange(0, BLOCK_SL)
        occ = tl.load(CELL_IN + b * scb + h * sch + si, mask=si < n_slots, other=0)
        slot = tl.min(tl.where((occ < 0) & (si < n_slots), si, BLOCK_SL), axis=0)
        in_table = cell < n_cells
        ok = (slot < n_slots) & in_table
        tl.atomic_add(FLAGS, 1, mask=~ok)
        tl.store(CEIL + b * seb + h * seh + cell, tl.load(CEIL1 + b * c1b + h * c1h), mask=in_table)
        j = tl.arange(0, S)
        d = tl.arange(0, DP)
        m = ok & (j[:, None] >= 0) & (d[None, :] < D)
        prow = slot * S + j
        k = tl.load(HK + b * shb + h * shh + j[:, None] * shn + d[None, :], mask=m, other=0.0).to(tl.float32)
        qk, sck = _quant_tile(k)
        tl.store(QK + b * sqb + h * sqh + prow[:, None] * sqn + d[None, :], qk, mask=m)
        tl.store(SK + b * ssb + h * ssh + slot, sck.to(tl.bfloat16), mask=ok)
        v = tl.load(HV + b * svb + h * svh + j[:, None] * svn + d[None, :], mask=m, other=0.0).to(tl.float32)
        qv, scv = _quant_tile(v)
        tl.store(QV + b * sqb + h * sqh + prow[:, None] * sqn + d[None, :], qv, mask=m)
        tl.store(SV + b * ssb + h * ssh + slot, scv.to(tl.bfloat16), mask=ok)
        dm = ok & (d < D)
        tl.store(KB + b * sxb + h * sxh + slot * sxs + d, tl.load(KBAR1 + b * s1b + h * s1h + d, mask=dm, other=0.0),
                 mask=dm)
        tl.store(SLOT_OF + b * stb + h * sth + cell, slot, mask=ok)
        tl.store(CELL_IN + b * scb + h * sch + slot, cell, mask=ok)

    @triton.jit
    def _pool_slots_kernel(TOPP, KB, SLOT_OF, CELL_IN, CEIL, FLAGS,
                           KBAR_OUT, KBAR_OUT_T, CEIL_OUT, CEN_OUT, OK_OUT, CAND_OUT,
                           sxb, sxh, sxs, stb, sth, scb, sch, seb, seh, skob, skoh, skoc, scob, scoh,
                           hi, blk_start_f, s_f, Pk, C, c_lo, n_slots,
                           H: tl.constexpr, D: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,
                           BLOCK_SL: tl.constexpr):
        # trunk_triton._fsl_pool_refresh_kernel op for op (candidates: the top P, then the Wc newest cells; visible
        # and first-occurrence flags; ceilings and centres), with the mean keys gathered through the cell -> slot
        # table; then the head's slots of cells below c_lo outside the top P are freed
        bh = tl.program_id(0)
        b = bh // H
        h = bh - b * H
        c = tl.arange(0, BLOCK_C)
        cmask = c < C

        topp_val = tl.load(TOPP + bh * Pk + c, mask=c < Pk, other=0).to(tl.int64)
        near_val = (hi - (c - Pk)).to(tl.int64)
        cand = tl.where(c < Pk, topp_val, near_val)
        cand_valid = (cand >= 0) & cmask
        cc = tl.maximum(cand, 0)

        cen = (cc.to(tl.float32) + 0.5) * s_f
        cend = (cc.to(tl.float32) + 1.0) * s_f
        vis = (cend <= blk_start_f) & cand_valid

        eq = (cc[:, None] == cc[None, :]) & (c[None, :] < c[:, None]) & cmask[:, None] & cmask[None, :]
        dup = tl.sum(eq.to(tl.int32), axis=1) > 0
        ok = vis & (~dup)

        out_off = b * scob + h * scoh + c
        tl.store(CEIL_OUT + out_off, tl.load(CEIL + b * seb + h * seh + cc, mask=cmask, other=0.0), mask=cmask)
        tl.store(CEN_OUT + out_off, cen, mask=cmask)
        tl.store(OK_OUT + out_off, ok, mask=cmask)
        tl.store(CAND_OUT + out_off, cc, mask=cmask)

        slot = tl.load(SLOT_OF + b * stb + h * sth + cc, mask=cmask, other=-1)
        has = cmask & (slot >= 0)
        tl.atomic_add(FLAGS + 1, tl.sum((ok & ~has).to(tl.int32), axis=0))
        d = tl.arange(0, BLOCK_D)
        dmask = d < D
        kbar_vals = tl.load(KB + b * sxb + h * sxh + slot[:, None] * sxs + d[None, :],
                            mask=has[:, None] & dmask[None, :], other=0.0)
        tl.store(KBAR_OUT + b * skob + h * skoh + c[:, None] * skoc + d[None, :], kbar_vals,
                 mask=cmask[:, None] & dmask[None, :])
        tl.store(KBAR_OUT_T + b * skob + h * skoh + d[:, None] * C + c[None, :], tl.trans(kbar_vals),
                 mask=dmask[:, None] & cmask[None, :])

        si = tl.arange(0, BLOCK_SL)
        smask = si < n_slots
        occ = tl.load(CELL_IN + b * scb + h * sch + si, mask=smask, other=-1)
        top = (c < Pk) & cmask
        in_top = tl.sum(((occ[:, None] == cand[None, :]) & top[None, :]).to(tl.int32), axis=1) > 0
        drop = smask & (occ >= 0) & (occ < c_lo) & (~in_top)
        tl.store(CELL_IN + b * scb + h * sch + si, tl.full((BLOCK_SL,), -1, tl.int32), mask=drop)
        tl.store(SLOT_OF + b * stb + h * sth + occ, tl.full((BLOCK_SL,), -1, tl.int32), mask=drop)
