from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_FWD_NUM_WARPS = 1
_FWD_NUM_STAGES = 1


if triton is not None:

    @triton.jit
    def _cell_fused_select_kernel(
        CONTENT,
        CEILC,
        CENC,
        OKCELL,
        CANDC,
        BLKSTART,
        LAM,
        SIGV,
        SEL,
        stride_cb, stride_ch, stride_cn, stride_cq, stride_cc,
        stride_eb, stride_eh, stride_en, stride_ec,
        stride_sb, stride_sh, stride_sr, stride_sk,
        H: tl.constexpr, NBC: tl.constexpr, BQ: tl.constexpr, C: tl.constexpr,
        BLOCK_C: tl.constexpr, KK: tl.constexpr,
        CONTENT_ONLY: tl.constexpr, NEG: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q = pid % BQ
        tmp = pid // BQ
        n = tmp % NBC
        bh = tmp // NBC
        b = bh // H
        h = bh - b * H

        offs_c = tl.arange(0, BLOCK_C)
        c_mask = offs_c < C
        content = tl.load(
            CONTENT + b * stride_cb + h * stride_ch + n * stride_cn + q * stride_cq + offs_c * stride_cc,
            mask=c_mask, other=0.0,
        )
        if CONTENT_ONLY:
            score = content
        else:
            ceil_c = tl.load(
                CEILC + b * stride_eb + h * stride_eh + n * stride_en + offs_c * stride_ec,
                mask=c_mask, other=0.0,
            )
            cen_c = tl.load(
                CENC + b * stride_eb + h * stride_eh + n * stride_en + offs_c * stride_ec,
                mask=c_mask, other=0.0,
            )
            blk_start = tl.load(BLKSTART + n)
            qpos = blk_start + q
            lam = tl.load(LAM + h)
            sigv = tl.load(SIGV + h)
            dist = tl.abs(qpos - cen_c)
            erf = tl.erf(dist / (2.0 * sigv))
            score = content + lam * ceil_c * erf

        okcell = tl.load(
            OKCELL + b * stride_eb + h * stride_eh + n * stride_en + offs_c * stride_ec,
            mask=c_mask, other=0,
        ) != 0
        score = tl.where(okcell & c_mask, score, NEG)

        candc = tl.load(
            CANDC + b * stride_eb + h * stride_eh + n * stride_en + offs_c * stride_ec,
            mask=c_mask, other=0,
        )

        RETIRE = float("-inf")
        sel_base = SEL + b * stride_sb + h * stride_sh + (n * BQ + q) * stride_sr
        for it in tl.static_range(0, KK):
            m = tl.max(score, axis=0)
            is_max = score == m
            best = tl.min(tl.where(is_max, offs_c, C), axis=0)
            best_ok = tl.sum(tl.where(offs_c == best, okcell.to(tl.int32), 0), axis=0) > 0
            best_cell = tl.sum(tl.where(offs_c == best, candc, 0), axis=0)
            out_cell = tl.where(best_ok, best_cell, -1)
            tl.store(sel_base + it * stride_sk, out_cell)
            score = tl.where(offs_c == best, RETIRE, score)


    @triton.jit(do_not_specialize=["Q_POS_BASE", "N"])
    def _cell_indexed_attn_fwd_kernel(
        Q,
        K,
        V,
        SEL,
        BIAS,
        O,
        LSE,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_sb,
        stride_sh,
        stride_sn,
        stride_sk,
        stride_bb,
        stride_bh,
        stride_bn,
        stride_bk,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        stride_lb,
        stride_lh,
        stride_ln,
        Q_POS_BASE,
        N,
        H: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        CELL_SIZE: tl.constexpr,
        QUERY_BLOCK: tl.constexpr,
        K_SELECT: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        q_idx = tl.program_id(0)
        bh_idx = tl.program_id(1)
        b_idx = bh_idx // H
        h_idx = bh_idx - b_idx * H

        q_pos = Q_POS_BASE + q_idx

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        q_ptrs = (
            Q
            + b_idx * stride_qb
            + h_idx * stride_qh
            + q_idx * stride_qn
            + offs_d * stride_qd
        )
        q = tl.load(q_ptrs, mask=d_mask, other=0.0)

        block_start = (q_pos // QUERY_BLOCK) * QUERY_BLOCK
        n_cells = (N + CELL_SIZE - 1) // CELL_SIZE
        offs_s = tl.arange(0, BLOCK_S)

        m_i = tl.full((1,), -float("inf"), tl.float32)
        l_i = tl.zeros((1,), tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)

        for slot in tl.static_range(0, K_SELECT):
            sel_ptr = (
                SEL
                + b_idx * stride_sb
                + h_idx * stride_sh
                + q_idx * stride_sn
                + slot * stride_sk
            )
            cell = tl.load(sel_ptr)
            cell_ok = (
                (cell >= 0)
                & (cell < n_cells)
                & ((cell + 1) * CELL_SIZE <= block_start)
            )

            for prev_slot in tl.static_range(0, K_SELECT):
                prev_ptr = (
                    SEL
                    + b_idx * stride_sb
                    + h_idx * stride_sh
                    + q_idx * stride_sn
                    + prev_slot * stride_sk
                )
                prev = tl.load(prev_ptr)
                prev_ok = (
                    (prev >= 0)
                    & (prev < n_cells)
                    & ((prev + 1) * CELL_SIZE <= block_start)
                )
                if prev_slot < slot:
                    cell_ok = cell_ok & ~(prev_ok & (prev == cell))

            safe_cell = tl.maximum(cell, 0)
            tok = safe_cell * CELL_SIZE + offs_s
            tok_ok = (
                cell_ok
                & (offs_s < CELL_SIZE)
                & (tok < N)
                & (tok < q_pos)
            )

            k_ptrs = (
                K
                + b_idx * stride_kb
                + h_idx * stride_kh
                + tok[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            kval = tl.load(
                k_ptrs,
                mask=tok_ok[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores = tl.sum(kval.to(tl.float32) * q[None, :], axis=1) * SCALE
            if HAS_BIAS:
                bias_val = tl.load(
                    BIAS
                    + b_idx * stride_bb
                    + h_idx * stride_bh
                    + q_idx * stride_bn
                    + slot * stride_bk
                )
                scores = tl.where(tok_ok, scores + bias_val, -float("inf"))
            else:
                scores = tl.where(tok_ok, scores, -float("inf"))

            has_token = tl.sum(tok_ok.to(tl.int32), axis=0) > 0
            tile_m = tl.max(scores, axis=0)
            m_new = tl.where(has_token, tl.maximum(m_i, tile_m), m_i)
            alpha = tl.where(has_token, tl.exp(m_i - m_new), 1.0)
            p = tl.where(tok_ok, tl.exp(scores - m_new), 0.0)

            v_ptrs = (
                V
                + b_idx * stride_vb
                + h_idx * stride_vh
                + tok[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            vval = tl.load(
                v_ptrs,
                mask=tok_ok[:, None] & d_mask[None, :],
                other=0.0,
            )
            acc_new = acc * alpha + tl.sum(
                p[:, None].to(vval.dtype) * vval, axis=0
            ).to(tl.float32)
            l_new = l_i * alpha + tl.sum(p, axis=0)

            acc = tl.where(has_token, acc_new, acc)
            l_i = tl.where(has_token, l_new, l_i)
            m_i = m_new

        row_ok = l_i > 0.0
        out = tl.where(row_ok, acc / l_i, 0.0)
        lse = tl.where(row_ok, m_i + tl.log(l_i), -float("inf"))

        o_ptrs = (
            O
            + b_idx * stride_ob
            + h_idx * stride_oh
            + q_idx * stride_on
            + offs_d * stride_od
        )
        tl.store(o_ptrs, out, mask=d_mask)
        lse_ptr = (
            LSE
            + b_idx * stride_lb
            + h_idx * stride_lh
            + q_idx * stride_ln
            + tl.arange(0, 1)
        )
        tl.store(lse_ptr, lse)

    @triton.jit
    def _cell_indexed_attn_fwd_posp_kernel(
        Q,
        K,
        V,
        SEL,
        BIAS,
        O,
        LSE,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_sb,
        stride_sh,
        stride_sn,
        stride_sk,
        stride_bb,
        stride_bh,
        stride_bn,
        stride_bk,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        stride_lb,
        stride_lh,
        stride_ln,
        POSP,
        N,
        H: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        CELL_SIZE: tl.constexpr,
        QUERY_BLOCK: tl.constexpr,
        K_SELECT: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        # _cell_indexed_attn_fwd_kernel with the query position base read from a device scalar (a 0-d int64
        # tensor) instead of a launch argument, so a captured CUDA graph can track the decode position
        q_idx = tl.program_id(0)
        bh_idx = tl.program_id(1)
        b_idx = bh_idx // H
        h_idx = bh_idx - b_idx * H

        q_pos = tl.load(POSP) + q_idx

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        q_ptrs = (
            Q
            + b_idx * stride_qb
            + h_idx * stride_qh
            + q_idx * stride_qn
            + offs_d * stride_qd
        )
        q = tl.load(q_ptrs, mask=d_mask, other=0.0)

        block_start = (q_pos // QUERY_BLOCK) * QUERY_BLOCK
        n_cells = (N + CELL_SIZE - 1) // CELL_SIZE
        offs_s = tl.arange(0, BLOCK_S)

        m_i = tl.full((1,), -float("inf"), tl.float32)
        l_i = tl.zeros((1,), tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)

        for slot in tl.static_range(0, K_SELECT):
            sel_ptr = (
                SEL
                + b_idx * stride_sb
                + h_idx * stride_sh
                + q_idx * stride_sn
                + slot * stride_sk
            )
            cell = tl.load(sel_ptr)
            cell_ok = (
                (cell >= 0)
                & (cell < n_cells)
                & ((cell + 1) * CELL_SIZE <= block_start)
            )

            for prev_slot in tl.static_range(0, K_SELECT):
                prev_ptr = (
                    SEL
                    + b_idx * stride_sb
                    + h_idx * stride_sh
                    + q_idx * stride_sn
                    + prev_slot * stride_sk
                )
                prev = tl.load(prev_ptr)
                prev_ok = (
                    (prev >= 0)
                    & (prev < n_cells)
                    & ((prev + 1) * CELL_SIZE <= block_start)
                )
                if prev_slot < slot:
                    cell_ok = cell_ok & ~(prev_ok & (prev == cell))

            safe_cell = tl.maximum(cell, 0)
            tok = safe_cell * CELL_SIZE + offs_s
            tok_ok = (
                cell_ok
                & (offs_s < CELL_SIZE)
                & (tok < N)
                & (tok < q_pos)
            )

            k_ptrs = (
                K
                + b_idx * stride_kb
                + h_idx * stride_kh
                + tok[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            kval = tl.load(
                k_ptrs,
                mask=tok_ok[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores = tl.sum(kval.to(tl.float32) * q[None, :], axis=1) * SCALE
            if HAS_BIAS:
                bias_val = tl.load(
                    BIAS
                    + b_idx * stride_bb
                    + h_idx * stride_bh
                    + q_idx * stride_bn
                    + slot * stride_bk
                )
                scores = tl.where(tok_ok, scores + bias_val, -float("inf"))
            else:
                scores = tl.where(tok_ok, scores, -float("inf"))

            has_token = tl.sum(tok_ok.to(tl.int32), axis=0) > 0
            tile_m = tl.max(scores, axis=0)
            m_new = tl.where(has_token, tl.maximum(m_i, tile_m), m_i)
            alpha = tl.where(has_token, tl.exp(m_i - m_new), 1.0)
            p = tl.where(tok_ok, tl.exp(scores - m_new), 0.0)

            v_ptrs = (
                V
                + b_idx * stride_vb
                + h_idx * stride_vh
                + tok[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            vval = tl.load(
                v_ptrs,
                mask=tok_ok[:, None] & d_mask[None, :],
                other=0.0,
            )
            acc_new = acc * alpha + tl.sum(
                p[:, None].to(vval.dtype) * vval, axis=0
            ).to(tl.float32)
            l_new = l_i * alpha + tl.sum(p, axis=0)

            acc = tl.where(has_token, acc_new, acc)
            l_i = tl.where(has_token, l_new, l_i)
            m_i = m_new

        row_ok = l_i > 0.0
        out = tl.where(row_ok, acc / l_i, 0.0)
        lse = tl.where(row_ok, m_i + tl.log(l_i), -float("inf"))

        o_ptrs = (
            O
            + b_idx * stride_ob
            + h_idx * stride_oh
            + q_idx * stride_on
            + offs_d * stride_od
        )
        tl.store(o_ptrs, out, mask=d_mask)
        lse_ptrs = (
            LSE
            + b_idx * stride_lb
            + h_idx * stride_lh
            + q_idx * stride_ln
            + tl.arange(0, 1)
        )
        tl.store(lse_ptrs, lse)

    @triton.jit
    def _cell_indexed_attn_bwd_dq_kernel(
        Q,
        K,
        V,
        SEL,
        BIAS,
        O,
        LSE,
        DO,
        DQ,
        DBIAS,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_sb,
        stride_sh,
        stride_sn,
        stride_sk,
        stride_bb,
        stride_bh,
        stride_bn,
        stride_bk,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        stride_lb,
        stride_lh,
        stride_ln,
        stride_dob,
        stride_doh,
        stride_don,
        stride_dod,
        stride_dqb,
        stride_dqh,
        stride_dqn,
        stride_dqd,
        stride_dbb,
        stride_dbh,
        stride_dbn,
        stride_dbk,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        CELL_SIZE: tl.constexpr,
        QUERY_BLOCK: tl.constexpr,
        K_SELECT: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        STORE_DBIAS: tl.constexpr,
    ):
        q_idx = tl.program_id(0)
        bh_idx = tl.program_id(1)
        b_idx = bh_idx // H
        h_idx = bh_idx - b_idx * H
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D

        q_ptrs = (
            Q
            + b_idx * stride_qb
            + h_idx * stride_qh
            + q_idx * stride_qn
            + offs_d * stride_qd
        )
        o_ptrs = (
            O
            + b_idx * stride_ob
            + h_idx * stride_oh
            + q_idx * stride_on
            + offs_d * stride_od
        )
        do_ptrs = (
            DO
            + b_idx * stride_dob
            + h_idx * stride_doh
            + q_idx * stride_don
            + offs_d * stride_dod
        )
        q = tl.load(q_ptrs, mask=d_mask, other=0.0)
        out = tl.load(o_ptrs, mask=d_mask, other=0.0)
        dout = tl.load(do_ptrs, mask=d_mask, other=0.0)

        lse_ptr = (
            LSE
            + b_idx * stride_lb
            + h_idx * stride_lh
            + q_idx * stride_ln
            + tl.arange(0, 1)
        )
        lse = tl.load(lse_ptr)
        row_ok = lse > -float("inf")
        delta = tl.sum(
            dout.to(tl.float32) * out.to(tl.float32), axis=0
        )

        block_start = (q_idx // QUERY_BLOCK) * QUERY_BLOCK
        n_cells = (N + CELL_SIZE - 1) // CELL_SIZE
        offs_s = tl.arange(0, BLOCK_S)
        dq = tl.zeros((BLOCK_D,), tl.float32)

        for slot in tl.static_range(0, K_SELECT):
            sel_ptr = (
                SEL
                + b_idx * stride_sb
                + h_idx * stride_sh
                + q_idx * stride_sn
                + slot * stride_sk
            )
            cell = tl.load(sel_ptr)
            cell_ok = (
                (cell >= 0)
                & (cell < n_cells)
                & ((cell + 1) * CELL_SIZE <= block_start)
            )
            for prev_slot in tl.static_range(0, K_SELECT):
                prev_ptr = (
                    SEL
                    + b_idx * stride_sb
                    + h_idx * stride_sh
                    + q_idx * stride_sn
                    + prev_slot * stride_sk
                )
                prev = tl.load(prev_ptr)
                prev_ok = (
                    (prev >= 0)
                    & (prev < n_cells)
                    & ((prev + 1) * CELL_SIZE <= block_start)
                )
                if prev_slot < slot:
                    cell_ok = cell_ok & ~(prev_ok & (prev == cell))

            safe_cell = tl.maximum(cell, 0)
            tok = safe_cell * CELL_SIZE + offs_s
            tok_ok = (
                row_ok
                & cell_ok
                & (offs_s < CELL_SIZE)
                & (tok < N)
                & (tok < q_idx)
            )
            k_ptrs = (
                K
                + b_idx * stride_kb
                + h_idx * stride_kh
                + tok[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V
                + b_idx * stride_vb
                + h_idx * stride_vh
                + tok[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            kval = tl.load(
                k_ptrs,
                mask=tok_ok[:, None] & d_mask[None, :],
                other=0.0,
            )
            vval = tl.load(
                v_ptrs,
                mask=tok_ok[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores = tl.sum(
                kval.to(tl.float32) * q[None, :], axis=1
            ) * SCALE
            if HAS_BIAS:
                bias_val = tl.load(
                    BIAS
                    + b_idx * stride_bb
                    + h_idx * stride_bh
                    + q_idx * stride_bn
                    + slot * stride_bk
                )
                scores = scores + bias_val
            p = tl.where(tok_ok, tl.exp(scores - lse), 0.0)
            dp = tl.sum(
                vval.to(tl.float32) * dout[None, :].to(tl.float32),
                axis=1,
            )
            ds = p * (dp - delta)
            dq += (
                tl.sum(ds[:, None] * kval.to(tl.float32), axis=0)
                * SCALE
            )
            if STORE_DBIAS:
                dbias_slot = tl.sum(ds, axis=0)
                dbias_ptr = (
                    DBIAS
                    + b_idx * stride_dbb
                    + h_idx * stride_dbh
                    + q_idx * stride_dbn
                    + slot * stride_dbk
                )
                tl.store(dbias_ptr, dbias_slot)

        dq_ptrs = (
            DQ
            + b_idx * stride_dqb
            + h_idx * stride_dqh
            + q_idx * stride_dqn
            + offs_d * stride_dqd
        )
        tl.store(dq_ptrs, dq, mask=d_mask)

    @triton.jit
    def _cell_indexed_attn_bwd_dkdv_segment_kernel(
        Q,
        K,
        V,
        O,
        LSE,
        DO,
        SORTED_Q,
        SORTED_BIAS,
        UNIQUE_KEYS,
        OFFSETS,
        DK,
        DV,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        stride_lb,
        stride_lh,
        stride_ln,
        stride_dob,
        stride_doh,
        stride_don,
        stride_dod,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dkd,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        stride_dvd,
        N: tl.constexpr,
        H: tl.constexpr,
        M: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        CELL_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_M: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        seg_idx = tl.program_id(0)
        key = tl.load(UNIQUE_KEYS + seg_idx)
        bh_idx = key // M
        cell = key - bh_idx * M
        b_idx = bh_idx // H
        h_idx = bh_idx - b_idx * H
        edge_begin = tl.load(OFFSETS + seg_idx)
        edge_end = tl.load(OFFSETS + seg_idx + 1)

        offs_s = tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, BLOCK_D)
        s_mask = offs_s < CELL_SIZE
        d_mask = offs_d < D
        tok = cell * CELL_SIZE + offs_s
        tok_ok = s_mask & (tok < N)

        k_ptrs = (
            K
            + b_idx * stride_kb
            + h_idx * stride_kh
            + tok[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V
            + b_idx * stride_vb
            + h_idx * stride_vh
            + tok[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        kval = tl.load(
            k_ptrs,
            mask=tok_ok[:, None] & d_mask[None, :],
            other=0.0,
        )
        vval = tl.load(
            v_ptrs,
            mask=tok_ok[:, None] & d_mask[None, :],
            other=0.0,
        )
        kval_t = tl.trans(kval)
        vval_t = tl.trans(vval)

        dk = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
        dv = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
        offs_m = tl.arange(0, BLOCK_M)

        edge_at = edge_begin
        while edge_at < edge_end:
            edge = edge_at + offs_m
            edge_mask = edge < edge_end
            q_idx = tl.load(SORTED_Q + edge, mask=edge_mask, other=0)
            if HAS_BIAS:
                edge_bias = tl.load(
                    SORTED_BIAS + edge, mask=edge_mask, other=0.0
                )

            q_ptrs = (
                Q
                + b_idx * stride_qb
                + h_idx * stride_qh
                + q_idx[:, None] * stride_qn
                + offs_d[None, :] * stride_qd
            )
            o_ptrs = (
                O
                + b_idx * stride_ob
                + h_idx * stride_oh
                + q_idx[:, None] * stride_on
                + offs_d[None, :] * stride_od
            )
            do_ptrs = (
                DO
                + b_idx * stride_dob
                + h_idx * stride_doh
                + q_idx[:, None] * stride_don
                + offs_d[None, :] * stride_dod
            )
            qval = tl.load(
                q_ptrs,
                mask=edge_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            oval = tl.load(
                o_ptrs,
                mask=edge_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            doval = tl.load(
                do_ptrs,
                mask=edge_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            lse_ptrs = (
                LSE
                + b_idx * stride_lb
                + h_idx * stride_lh
                + q_idx * stride_ln
            )
            lse = tl.load(lse_ptrs, mask=edge_mask, other=-float("inf"))

            scores = tl.dot(
                qval, kval_t, input_precision="ieee"
            ) * SCALE
            if HAS_BIAS:
                scores = scores + edge_bias[:, None]
            p = tl.where(
                edge_mask[:, None] & tok_ok[None, :],
                tl.exp(scores - lse[:, None]),
                0.0,
            )
            dp = tl.dot(doval, vval_t, input_precision="ieee")
            delta = tl.sum(
                doval.to(tl.float32) * oval.to(tl.float32), axis=1
            )
            ds = p * (dp - delta[:, None])

            dk += (
                tl.dot(
                    tl.trans(ds.to(qval.dtype)),
                    qval,
                    input_precision="ieee",
                )
                * SCALE
            )
            dv += tl.dot(
                tl.trans(p.to(doval.dtype)),
                doval,
                input_precision="ieee",
            )
            edge_at += BLOCK_M

        dk_ptrs = (
            DK
            + b_idx * stride_dkb
            + h_idx * stride_dkh
            + tok[:, None] * stride_dkn
            + offs_d[None, :] * stride_dkd
        )
        dv_ptrs = (
            DV
            + b_idx * stride_dvb
            + h_idx * stride_dvh
            + tok[:, None] * stride_dvn
            + offs_d[None, :] * stride_dvd
        )
        store_mask = tok_ok[:, None] & d_mask[None, :]
        tl.store(dk_ptrs, dk, mask=store_mask)
        tl.store(dv_ptrs, dv, mask=store_mask)


def cell_fused_select(
    qb: torch.Tensor,
    kbar_cand: torch.Tensor,
    ceil_cand: torch.Tensor,
    cen_cand: torch.Tensor,
    ok_cell: torch.Tensor,
    cand_c: torch.Tensor,
    blk_start: torch.Tensor,
    lam: torch.Tensor,
    sigv: torch.Tensor,
    *,
    kk: int,
    scale: float,
    content_only: bool,
    neg: float = -1e9,
) -> torch.Tensor:
    _require_triton_cuda()
    B, H, nbc, Bq, D = qb.shape
    C = kbar_cand.shape[3]
    KK = min(int(kk), C)
    if not qb.is_cuda:
        raise ValueError("cell_fused_select is a CUDA-only fused kernel")
    if qb.dtype != torch.float32 or kbar_cand.dtype != torch.float32:
        raise TypeError("qb and kbar_cand must be float32 (match the eager score dtype)")
    if C > 512:
        raise ValueError(f"candidate count must be <=512, got {C}")
    content = torch.matmul(qb, kbar_cand.transpose(-1, -2)) * scale
    content = content.contiguous()
    ok_i8 = ok_cell.to(torch.int8).contiguous()
    cand_i32 = cand_c.to(torch.int32).contiguous()
    ceil_f = ceil_cand.contiguous()
    cen_f = cen_cand.contiguous()
    blk_f = blk_start.to(torch.float32).contiguous()
    lam_f = lam.reshape(-1).to(torch.float32).contiguous()
    sigv_f = sigv.reshape(-1).to(torch.float32).contiguous()
    sel = torch.empty((B, H, nbc * Bq, KK), device=qb.device, dtype=torch.int32)

    block_c = max(16, triton.next_power_of_2(C))
    grid = (B * H * nbc * Bq,)
    _cell_fused_select_kernel[grid](
        content, ceil_f, cen_f, ok_i8, cand_i32, blk_f, lam_f, sigv_f, sel,
        *content.stride(),
        *ceil_f.stride(),
        *sel.stride(),
        H=H, NBC=nbc, BQ=Bq, C=C,
        BLOCK_C=block_c, KK=KK,
        CONTENT_ONLY=bool(content_only), NEG=float(neg),
        num_warps=1, num_stages=1,
    )
    return sel


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError(
            "Fused cell attention requires Triton. Use the masked-pool "
            "reference on CPU or install a CUDA PyTorch build with Triton."
        )


def _validate_forward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_cells: torch.Tensor,
    cell_size: int,
    query_block: int,
) -> Tuple[int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B,H,N,D]")
    if k.shape != v.shape:
        raise ValueError(
            f"k and v shapes must match, got {k.shape}, {v.shape}"
        )
    if (q.shape[0], q.shape[1], q.shape[3]) != (k.shape[0], k.shape[1], k.shape[3]):
        raise ValueError(
            f"q, k, v must share batch/head/head_dim, got {q.shape}, {k.shape}, {v.shape}"
        )
    if q.shape[2] > k.shape[2]:
        raise ValueError(
            f"query count {q.shape[2]} must not exceed K/V length {k.shape[2]}"
        )
    if selected_cells.ndim != 4:
        raise ValueError("selected_cells must have shape [B,H,Npad,K]")

    bsz, n_heads, n_ctx, head_dim = q.shape
    if selected_cells.shape[:2] != (bsz, n_heads):
        raise ValueError(
            "selected_cells batch/head dimensions must match q: "
            f"{selected_cells.shape[:2]} vs {(bsz, n_heads)}"
        )
    if selected_cells.shape[2] < n_ctx:
        raise ValueError(
            "selected_cells must cover every real query row: "
            f"{selected_cells.shape[2]} < {n_ctx}"
        )
    if selected_cells.shape[3] <= 0:
        raise ValueError("selected_cells must contain at least one selection slot")
    if selected_cells.dtype not in (torch.int32, torch.int64):
        raise TypeError("selected_cells must use int32 or int64 cell IDs")
    if q.device != k.device or q.device != v.device or q.device != selected_cells.device:
        raise ValueError("q, k, v, and selected_cells must be on the same device")
    if not q.is_cuda:
        raise ValueError("cell_indexed_attn_fwd is a CUDA-only fused kernel")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported q/k/v dtype: {q.dtype}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must use the same dtype")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, and v must be contiguous in their last dimension")
    if cell_size <= 0 or query_block <= 0:
        raise ValueError("cell_size and query_block must be positive")
    if head_dim <= 0 or head_dim > 256:
        raise ValueError(f"head dimension must be in [1,256], got {head_dim}")
    return bsz, n_heads, n_ctx, head_dim


def _validate_bias(
    bias: torch.Tensor | None,
    selected_cells: torch.Tensor,
    q: torch.Tensor,
) -> None:
    if bias is None:
        return
    if bias.shape != selected_cells.shape:
        raise ValueError(
            "bias must have the same [B,H,Npad,K] shape as selected_cells: "
            f"{tuple(bias.shape)} vs {tuple(selected_cells.shape)}"
        )
    if bias.dtype != torch.float32:
        raise TypeError(f"bias must be float32, got {bias.dtype}")
    if bias.device != q.device:
        raise ValueError("bias must be on the Q device")


def cell_indexed_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_cells: torch.Tensor,
    *,
    cell_size: int,
    query_block: int,
    scale: float,
    bias: torch.Tensor | None = None,
    q_pos: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _require_triton_cuda()
    bsz, n_heads, n_ctx, head_dim = _validate_forward_inputs(
        q, k, v, selected_cells, cell_size, query_block
    )
    n_kv = k.shape[2]
    k_select = selected_cells.shape[3]
    if k_select > 16:
        raise ValueError(f"K selection slots must be <=16, got {k_select}")
    _validate_bias(bias, selected_cells, q)

    out = torch.empty(
        (bsz, n_heads, n_ctx, head_dim), device=q.device, dtype=q.dtype
    )
    lse = torch.empty((bsz, n_heads, n_ctx), device=q.device, dtype=torch.float32)
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_s = max(16, triton.next_power_of_2(cell_size))
    if block_s > 128:
        raise ValueError(f"cell_size must be <=128, got {cell_size}")

    has_bias = bias is not None
    bias_arg = bias if has_bias else selected_cells
    bias_strides = (
        bias.stride() if has_bias else selected_cells.stride()
    )

    grid = (n_ctx, bsz * n_heads)
    _cell_indexed_attn_fwd_kernel[grid](
        q,
        k,
        v,
        selected_cells,
        bias_arg,
        out,
        lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *selected_cells.stride(),
        *bias_strides,
        *out.stride(),
        *lse.stride(),
        int(q_pos),
        N=n_kv,
        H=n_heads,
        D=head_dim,
        SCALE=float(scale),
        CELL_SIZE=int(cell_size),
        QUERY_BLOCK=int(query_block),
        K_SELECT=int(k_select),
        BLOCK_D=block_d,
        BLOCK_S=block_s,
        HAS_BIAS=has_bias,
        num_warps=_FWD_NUM_WARPS,
        num_stages=_FWD_NUM_STAGES,
    )
    return out, lse


def cell_indexed_attn_fwd_posp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_cells: torch.Tensor,
    *,
    cell_size: int,
    query_block: int,
    scale: float,
    bias: torch.Tensor | None = None,
    q_pos_ptr: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # cell_indexed_attn_fwd with the position base read from a 0-d int64 device tensor, so a captured CUDA
    # graph replays with the position the decoder's pos_t holds (a python int would be baked at capture)
    _require_triton_cuda()
    if q_pos_ptr.dim() != 0 or q_pos_ptr.device != q.device:
        raise ValueError("q_pos_ptr must be a 0-d tensor on the Q device")
    bsz, n_heads, n_ctx, head_dim = _validate_forward_inputs(
        q, k, v, selected_cells, cell_size, query_block
    )
    n_kv = k.shape[2]
    k_select = selected_cells.shape[3]
    if k_select > 16:
        raise ValueError(f"K selection slots must be <=16, got {k_select}")
    _validate_bias(bias, selected_cells, q)

    out = torch.empty(
        (bsz, n_heads, n_ctx, head_dim), device=q.device, dtype=q.dtype
    )
    lse = torch.empty((bsz, n_heads, n_ctx), device=q.device, dtype=torch.float32)
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_s = max(16, triton.next_power_of_2(cell_size))
    if block_s > 128:
        raise ValueError(f"cell_size must be <=128, got {cell_size}")

    has_bias = bias is not None
    bias_arg = bias if has_bias else selected_cells
    bias_strides = (
        bias.stride() if has_bias else selected_cells.stride()
    )

    grid = (n_ctx, bsz * n_heads)
    _cell_indexed_attn_fwd_posp_kernel[grid](
        q,
        k,
        v,
        selected_cells,
        bias_arg,
        out,
        lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *selected_cells.stride(),
        *bias_strides,
        *out.stride(),
        *lse.stride(),
        q_pos_ptr,
        N=n_kv,
        H=n_heads,
        D=head_dim,
        SCALE=float(scale),
        CELL_SIZE=int(cell_size),
        QUERY_BLOCK=int(query_block),
        K_SELECT=int(k_select),
        BLOCK_D=block_d,
        BLOCK_S=block_s,
        HAS_BIAS=has_bias,
        num_warps=_FWD_NUM_WARPS,
        num_stages=_FWD_NUM_STAGES,
    )
    return out, lse


def _build_cell_routes(
    selected_cells: torch.Tensor,
    *,
    n_ctx: int,
    cell_size: int,
    query_block: int,
    n_heads: int,
    bias: torch.Tensor | None = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    bsz = selected_cells.shape[0]
    ids = selected_cells[:, :, :n_ctx]
    k_select = ids.shape[-1]
    n_cells = (n_ctx + cell_size - 1) // cell_size

    q_idx = torch.arange(
        n_ctx, device=ids.device, dtype=torch.int32
    ).view(1, 1, n_ctx, 1)
    block_start = (q_idx // query_block) * query_block
    valid = (
        (ids >= 0)
        & (ids < n_cells)
        & ((ids + 1) * cell_size <= block_start)
    )
    for slot in range(k_select):
        for prev in range(slot):
            valid[..., slot] &= ~(
                valid[..., prev] & (ids[..., slot] == ids[..., prev])
            )

    bh = torch.arange(
        bsz * n_heads, device=ids.device, dtype=ids.dtype
    ).view(bsz, n_heads, 1, 1)
    keys = bh * n_cells + ids.clamp_min(0)
    queries = q_idx.expand(bsz, n_heads, n_ctx, k_select)
    edge_keys = keys[valid]
    edge_queries = queries[valid]
    edge_bias = bias[:, :, :n_ctx][valid] if bias is not None else None

    sorted_keys, order = torch.sort(edge_keys, stable=True)
    sorted_queries = edge_queries[order].contiguous()
    sorted_bias = (
        edge_bias[order].contiguous() if edge_bias is not None else None
    )
    unique_keys, counts = torch.unique_consecutive(
        sorted_keys, return_counts=True
    )
    offsets = torch.empty(
        unique_keys.numel() + 1,
        device=ids.device,
        dtype=torch.int64,
    )
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return sorted_queries, unique_keys, offsets, sorted_bias


def cell_indexed_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_cells: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    cell_size: int,
    query_block: int,
    scale: float,
    bias: torch.Tensor | None = None,
    needs_dq: bool = True,
    needs_dk: bool = True,
    needs_dv: bool = True,
    needs_dbias: bool = True,
) -> Tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    _require_triton_cuda()
    bsz, n_heads, n_ctx, head_dim = _validate_forward_inputs(
        q, k, v, selected_cells, cell_size, query_block
    )
    if out.shape != q.shape or dout.shape != q.shape:
        raise ValueError("out and dout must have the same logical shape as q")
    if lse.shape != (bsz, n_heads, n_ctx) or lse.dtype != torch.float32:
        raise ValueError("lse must have shape [B,H,N] and dtype float32")
    if out.device != q.device or dout.device != q.device or lse.device != q.device:
        raise ValueError("out, lse, and dout must be on the Q device")
    if out.stride(-1) != 1:
        raise ValueError("out must be contiguous in its last dimension")
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    _validate_bias(bias, selected_cells, q)

    block_d = max(16, triton.next_power_of_2(head_dim))
    block_s = max(16, triton.next_power_of_2(cell_size))
    k_select = selected_cells.shape[-1]
    if k_select > 16:
        raise ValueError(f"K selection slots must be <=16, got {k_select}")
    if block_s > 128:
        raise ValueError(f"cell_size must be <=128, got {cell_size}")

    has_bias = bias is not None
    bias_arg = bias if has_bias else selected_cells
    bias_strides = bias.stride() if has_bias else selected_cells.stride()

    want_dbias = has_bias and needs_dbias
    run_dq_kernel = needs_dq or want_dbias
    dq_fp32 = (
        torch.empty(
            (bsz, n_heads, n_ctx, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        if run_dq_kernel
        else None
    )
    dbias_fp32 = (
        torch.zeros_like(bias) if want_dbias else None
    )
    dbias_arg = dbias_fp32 if want_dbias else selected_cells
    dbias_strides = (
        dbias_fp32.stride() if want_dbias else selected_cells.stride()
    )
    if run_dq_kernel:
        grid_dq = (n_ctx, bsz * n_heads)
        _cell_indexed_attn_bwd_dq_kernel[grid_dq](
            q,
            k,
            v,
            selected_cells,
            bias_arg,
            out,
            lse,
            dout,
            dq_fp32,
            dbias_arg,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *selected_cells.stride(),
            *bias_strides,
            *out.stride(),
            *lse.stride(),
            *dout.stride(),
            *dq_fp32.stride(),
            *dbias_strides,
            N=n_ctx,
            H=n_heads,
            D=head_dim,
            SCALE=float(scale),
            CELL_SIZE=int(cell_size),
            QUERY_BLOCK=int(query_block),
            K_SELECT=int(k_select),
            BLOCK_D=block_d,
            BLOCK_S=block_s,
            HAS_BIAS=has_bias,
            STORE_DBIAS=want_dbias,
            num_warps=1,
            num_stages=1,
        )

    dk_fp32 = None
    dv_fp32 = None
    if needs_dk or needs_dv:
        dk_fp32 = torch.zeros(
            (bsz, n_heads, n_ctx, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        dv_fp32 = torch.zeros_like(dk_fp32)
        sorted_q, unique_keys, offsets, sorted_bias = _build_cell_routes(
            selected_cells,
            n_ctx=n_ctx,
            cell_size=cell_size,
            query_block=query_block,
            n_heads=n_heads,
            bias=bias,
        )
        sorted_bias_arg = sorted_bias if has_bias else sorted_q
        if unique_keys.numel() > 0:
            n_cells = (n_ctx + cell_size - 1) // cell_size
            grid_dkdv = (unique_keys.numel(),)
            _cell_indexed_attn_bwd_dkdv_segment_kernel[grid_dkdv](
                q,
                k,
                v,
                out,
                lse,
                dout,
                sorted_q,
                sorted_bias_arg,
                unique_keys,
                offsets,
                dk_fp32,
                dv_fp32,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *out.stride(),
                *lse.stride(),
                *dout.stride(),
                *dk_fp32.stride(),
                *dv_fp32.stride(),
                N=n_ctx,
                H=n_heads,
                M=n_cells,
                D=head_dim,
                SCALE=float(scale),
                CELL_SIZE=int(cell_size),
                BLOCK_D=block_d,
                BLOCK_S=block_s,
                BLOCK_M=16,
                HAS_BIAS=has_bias,
                num_warps=4,
                num_stages=2,
            )

    dq = dq_fp32.to(q.dtype) if (needs_dq and dq_fp32 is not None) else None
    dk = dk_fp32.to(k.dtype) if needs_dk and dk_fp32 is not None else None
    dv = dv_fp32.to(v.dtype) if needs_dv and dv_fp32 is not None else None
    dbias = dbias_fp32 if want_dbias else None
    return dq, dk, dv, dbias


class _CellIndexedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor | None,
        selected_cells: torch.Tensor,
        cell_size: int,
        query_block: int,
        scale: float,
        q_pos: int = 0,
    ) -> torch.Tensor:
        out, lse = cell_indexed_attn_fwd(
            q,
            k,
            v,
            selected_cells,
            cell_size=cell_size,
            query_block=query_block,
            scale=scale,
            bias=bias,
            q_pos=q_pos,
        )
        ctx.save_for_backward(q, k, v, selected_cells, out, lse, bias)
        ctx.cell_size = int(cell_size)
        ctx.query_block = int(query_block)
        ctx.scale = float(scale)
        ctx.has_bias = bias is not None
        ctx.q_pos = int(q_pos)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout: torch.Tensor):
        if getattr(ctx, "q_pos", 0) != 0:
            raise RuntimeError(
                "indexed_cell_attention backward does not support q_pos!=0 "
                "(shifted-query decode is inference-only)."
            )
        q, k, v, selected_cells, out, lse, bias = ctx.saved_tensors
        need_dbias = ctx.has_bias and ctx.needs_input_grad[3]
        dq, dk, dv, dbias = cell_indexed_attn_bwd(
            q,
            k,
            v,
            selected_cells,
            out,
            lse,
            dout,
            cell_size=ctx.cell_size,
            query_block=ctx.query_block,
            scale=ctx.scale,
            bias=bias,
            needs_dq=ctx.needs_input_grad[0],
            needs_dk=ctx.needs_input_grad[1],
            needs_dv=ctx.needs_input_grad[2],
            needs_dbias=need_dbias,
        )
        return dq, dk, dv, dbias, None, None, None, None, None


@torch._dynamo.disable
def indexed_cell_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_cells: torch.Tensor,
    *,
    cell_size: int,
    query_block: int,
    scale: float,
    bias: torch.Tensor | None = None,
    q_pos: int = 0,
) -> torch.Tensor:
    return _CellIndexedAttention.apply(
        q,
        k,
        v,
        bias,
        selected_cells,
        int(cell_size),
        int(query_block),
        float(scale),
        int(q_pos),
    )


__all__ = [
    "indexed_cell_attention",
    "cell_fused_select",
    "cell_indexed_attn_bwd",
    "cell_indexed_attn_fwd",
    "cell_indexed_attn_fwd_posp",
]
