import os

import torch
import torch.nn.functional as F

from .compile_guard import compiling


_QCHUNK = 256


_FLEX_DISABLED = os.environ.get("FSL_DISABLE_FLEX_SLIDING", "0") == "1"

_flex_attention = None
_create_block_mask = None
_flex_import_tried = False
_flex_compiled = None

_block_mask_cache: dict = {}


def _try_import_flex():
    global _flex_attention, _create_block_mask, _flex_import_tried, _flex_compiled
    if _flex_import_tried:
        return _flex_attention is not None
    _flex_import_tried = True
    if _FLEX_DISABLED:
        return False
    try:
        from torch.nn.attention.flex_attention import (
            create_block_mask as _cbm,
            flex_attention as _fa,
        )
        _flex_attention = _fa
        _create_block_mask = _cbm
        _flex_compiled = torch.compile(_fa, dynamic=False)
    except Exception:
        _flex_attention = None
        _create_block_mask = None
        _flex_compiled = None
    return _flex_attention is not None


def _sliding_window_block_mask(N, W, device):
    key = (int(N), int(W), str(device))
    bm = _block_mask_cache.get(key)
    if bm is not None:
        return bm

    def _mask_mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        in_window = (q_idx - kv_idx) <= W
        return causal & in_window

    bm = _create_block_mask(_mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=device, _compile=True)
    _block_mask_cache[key] = bm
    return bm


def _flex_sliding_window(Q, K, V, W, scale):
    N = Q.shape[2]
    block_mask = _sliding_window_block_mask(N, W, Q.device)
    return _flex_compiled(Q, K, V, block_mask=block_mask, scale=float(scale))


_erfc_score_mod_cache: dict = {}


def _erfc_sliding_score_mod(sigma):
    key = (id(sigma), str(sigma.device))
    fn = _erfc_score_mod_cache.get(key)
    if fn is not None:
        return fn
    sig = sigma.reshape(-1).to(torch.float32)

    def _mod(score, b, h, q_idx, kv_idx):
        d = (q_idx - kv_idx).to(torch.float32)
        arg = d / (2.0 * sig[h])
        bias = torch.log(torch.clamp(torch.erfc(arg), min=1e-10))
        return score + bias.to(score.dtype)

    _erfc_score_mod_cache[key] = _mod
    return _mod


def _flex_sliding_window_erfc(Q, K, V, W, scale, sigma):
    N = Q.shape[2]
    block_mask = _sliding_window_block_mask(N, W, Q.device)
    score_mod = _erfc_sliding_score_mod(sigma)
    return _flex_compiled(Q, K, V, score_mod=score_mod, block_mask=block_mask, scale=float(scale))


_SLIDE_QCHUNK = int(os.environ.get("FSL_SLIDING_QCHUNK", "1024"))

# The memory-efficient SDPA kernel walks 64-query tiles against key tiles of 64 keys (head_dim <= 64, the 64x64 kernel)
# or 128 keys (64 < head_dim <= 128, the 64x128 kernel). A key tile that is fully masked for a query row is an exact
# no-op for that row (test_chunked_window_matches_generic_loop), so every row's output depends only on the tiles that
# hold its own window and on where the tile grid starts. With the window and the chunk length multiples of the key
# tile T, every chunk's keys start on the absolute T-aligned grid, and so do the keys of a T-query problem that starts
# at a multiple of T with exactly its window: same tiles, same values. With 128-key tiles a 64-query problem would
# start its keys mid-tile for every other block (6-13% of outputs differ for head_dim 96 / 128), hence T-query blocks.
_band_cache: dict = {}


def _band(rows, keys, k0_minus_q0, W, dtype, device):
    # (rows, keys) additive band mask, 0 where 0 <= (q0 + r) - (k0 + c) <= W, else -inf (as the chunked path builds it)
    key = (rows, keys, k0_minus_q0, W, dtype, str(device))
    m = _band_cache.get(key)
    if m is None:
        dist = torch.arange(rows, device=device)[:, None] - (torch.arange(keys, device=device)[None, :] + k0_minus_q0)
        band = (dist >= 0) & (dist <= W)
        m = torch.where(band, torch.zeros((), device=device, dtype=dtype),
                        torch.full((), float("-inf"), device=device, dtype=dtype))
        _band_cache[key] = m
    return m


def _window_rows(Q, K, V, q0, q1, W, scale):
    # rows [q0, q1) against keys [max(0, q0 - W), q1), one SDPA call; returns (B, H, q1 - q0, D)
    k0 = max(0, q0 - W)
    m = _band(q1 - q0, q1 - k0, k0 - q0, W, Q.dtype, Q.device)
    return F.scaled_dot_product_attention(Q[:, :, q0:q1], K[:, :, k0:q1], V[:, :, k0:q1], attn_mask=m[None, None],
                                          scale=scale)


def _autocast_inputs(Q, K, V):
    # the casts SDPA's autocast would apply to its inputs (the window views must be taken on the cast tensors)
    if not (Q.is_cuda and torch.is_autocast_enabled()):
        return Q, K, V
    dt = torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else torch.get_autocast_gpu_dtype()
    return tuple(t.to(dt) if (t.is_floating_point() and t.dtype not in (dt, torch.float64)) else t for t in (Q, K, V))


def _batched_block(Q, K, V, W):
    # Query block T of the batched window for these (autocast-cast) inputs, or 0 when it is not taken: T is the key
    # tile of the SDPA kernel for the head dim (64 up to 64, 128 up to 128; larger head dims are not covered), and the
    # window and the chunked path's chunk must be multiples of T (see the note above _band_cache).
    D = Q.shape[-1]
    T = 64 if D <= 64 else 128
    ok = (D <= 128 and D % 8 == 0 and W % T == 0 and _SLIDE_QCHUNK % T == 0
          and Q.dtype in (torch.bfloat16, torch.float16) and K.dtype == Q.dtype and V.dtype == Q.dtype
          and Q.stride(-1) == 1 and K.stride(-1) == 1 and V.stride(-1) == 1)
    return T if ok else 0


def _bh_mergeable(x):
    # (B, H, N, D) whose batch and head index merge into one (B * H) index with a single stride
    return x.shape[0] == 1 or x.stride(0) == x.shape[1] * x.stride(1)


def _batched_sliding_window(Q, K, V, W, scale, T=64):
    # Same values as _chunked_sliding_window / the generic loop, bit for bit (tests/test_window_paths.py), from one
    # SDPA call over all T-query blocks q0 = T j >= W of every batch row (T from _batched_block): each block is its
    # own problem with its window's keys [q0 - W, q0 + T) as an overlapping strided view of K / V and one shared
    # (T, W + T) band, the batch rows folded into the head index. Rows below W and a partial last block take one
    # reference-style call each. Every (row, head, query) is computed independently of the others in the call, so the
    # folding changes no value. Returns a (B, H, N, D) view of a (B, N, H, D) buffer, the layout W_o reads. Inference
    # only: the backward of the overlapping views accumulates with atomics (local_attention takes it only without
    # autograd).
    B, H, N, D = Q.shape
    j0, nfull = W // T, N // T
    nblk = nfull - j0
    out = torch.empty(B, N, H, D, device=Q.device, dtype=Q.dtype)
    out[:, :W].copy_(_window_rows(Q, K, V, 0, W, W, scale).transpose(1, 2))
    if nblk > 0:
        # batch rows fold into the head index only with a single (B * H) stride: the rotated Q / K are contiguous, V
        # (the transposed W_v output) is copied once when B > 1. The copy costs up to ~9% of the window (B=2,
        # N=16384); a loop over the rows instead cost three host-bound calls per row, up to 2.8x slower than the
        # chunked path on short prompts.
        Qm, Km, Vm = (t if _bh_mergeable(t) else t.contiguous() for t in (Q, K, V))
        sq, sk, sv = Qm.stride(), Km.stride(), Vm.stride()
        qv = Qm.as_strided((nblk, B * H, T, D), (T * sq[2], sq[1], sq[2], sq[3]), Qm.storage_offset() + W * sq[2])
        kv = Km.as_strided((nblk, B * H, T + W, D), (T * sk[2], sk[1], sk[2], sk[3]), Km.storage_offset())
        vv = Vm.as_strided((nblk, B * H, T + W, D), (T * sv[2], sv[1], sv[2], sv[3]), Vm.storage_offset())
        m = _band(T, T + W, -W, W, Q.dtype, Q.device)
        o = F.scaled_dot_product_attention(qv, kv, vv, attn_mask=m[None, None], scale=scale)
        out[:, W:nfull * T].view(B, nblk, T, H, D).copy_(o.view(nblk, B, H, T, D).permute(1, 0, 3, 2, 4))
    if nfull * T < N:
        out[:, nfull * T:].copy_(_window_rows(Q, K, V, nfull * T, N, W, scale).transpose(1, 2))
    return out.transpose(1, 2)


def _chunked_sliding_window(Q, K, V, W, scale):
    B, H, N, D = Q.shape
    C = _SLIDE_QCHUNK
    neg = float("-inf")
    outs = []
    for c0 in range(0, N, C):
        c1 = min(c0 + C, N)
        k0 = max(0, c0 - W)
        q = Q[:, :, c0:c1]
        k = K[:, :, k0:c1]
        v = V[:, :, k0:c1]
        qi = torch.arange(c0, c1, device=Q.device)
        kj = torch.arange(k0, c1, device=Q.device)
        dist = qi.unsqueeze(1) - kj.unsqueeze(0)
        band = (dist >= 0) & (dist <= W)
        m = torch.where(band, torch.zeros((), device=Q.device, dtype=Q.dtype),
                        torch.full((), neg, device=Q.device, dtype=Q.dtype))
        m = m.unsqueeze(0).unsqueeze(0)
        outs.append(F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=scale))
    return torch.cat(outs, dim=2)


def local_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int,
    score_bias: torch.Tensor | None = None,
    scale: float = 1.0,
    causal: bool = False,
    sigma: torch.Tensor | None = None,
    fast_sliding: bool = True,
) -> torch.Tensor:
    B, H, N, D = Q.shape
    W = window_size

    if W >= N and score_bias is None:
        return F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=scale)

    _mode = os.environ.get("FSL_SLIDING_MODE", "chunked").lower()
    if (
        fast_sliding
        and _mode == "flex"
        and causal
        and sigma is not None
        and score_bias is None
        and 0 < W < N
        and Q.is_cuda
    ):
        try:
            if _try_import_flex():
                return _flex_sliding_window_erfc(Q, K, V, W, scale, sigma)
        except Exception:
            pass
    if (
        fast_sliding
        and causal
        and sigma is None
        and score_bias is None
        and 0 < W < N
        and Q.is_cuda
        and _mode != "off"
    ):
        try:
            if _mode == "flex":
                if _try_import_flex():
                    return _flex_sliding_window(Q, K, V, W, scale)
            else:
                # (inference only; under torch.compile the chunked path, which inductor compiles as before)
                if (os.environ.get("FSL_WINDOW_BATCHED", "1") != "0" and not torch.is_grad_enabled()
                        and not compiling()):
                    q, k, v = _autocast_inputs(Q, K, V)
                    T = _batched_block(q, k, v, W)
                    if T:
                        return _batched_sliding_window(q, k, v, W, scale, T)
                return _chunked_sliding_window(Q, K, V, W, scale)
        except Exception:
            pass

    neg = float("-inf")
    outs = []
    for c0 in range(0, N, _QCHUNK):
        c1 = min(c0 + _QCHUNK, N)
        k0 = max(0, c0 - W)
        k1 = c1 if causal else min(N, c1 + W)
        q = Q[:, :, c0:c1]
        k = K[:, :, k0:k1]
        v = V[:, :, k0:k1]

        qi = torch.arange(c0, c1, device=Q.device)
        kj = torch.arange(k0, k1, device=Q.device)
        dist = qi.unsqueeze(1) - kj.unsqueeze(0)
        band = dist.abs() <= W
        if causal:
            band = band & (dist >= 0)

        if sigma is not None:
            arg = dist.abs().to(torch.float32).unsqueeze(0) / (2.0 * sigma.view(H, 1, 1))
            bias = torch.log(torch.erfc(arg).clamp(min=1e-10))
            m = torch.where(band.unsqueeze(0), bias,
                            torch.full((), neg, device=Q.device, dtype=bias.dtype))
            m = m.unsqueeze(0).to(Q.dtype)
        elif score_bias is not None:
            idx = (dist + W).clamp(0, 2 * W).long()
            bias = score_bias[:, idx]
            m = torch.where(band.unsqueeze(0), bias,
                            torch.full((), neg, device=Q.device, dtype=bias.dtype))
            m = m.unsqueeze(0).to(Q.dtype)
        else:
            m = torch.where(band, torch.zeros((), device=Q.device, dtype=Q.dtype),
                            torch.full((), neg, device=Q.device, dtype=Q.dtype))
            m = m.unsqueeze(0).unsqueeze(0)

        outs.append(F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=scale))

    return torch.cat(outs, dim=2)
