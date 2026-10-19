"""int8 long-range KV storage for the inference caches (FS_KV_INT8=1 on the graph decode path).

Layout: completed cells of s = cell_stride tokens are quantized per cell with symmetric
round-to-nearest int8 and a bf16 scale per (batch, head, cell, K-or-V):
    q = round(x / scale), scale = max(|x|) / 127 (bf16; empty cell -> 1.0 to keep x == 0)
The open (incomplete) cell and the new decode rows stay bf16 in a small hot buffer, so the
int8 region always spans whole cells. Cell statistics (Kbar / ceilings) are computed by the
cache from the pre-quantization bf16 values, identically to the bf16 path.

Reads gather by token index: idx < n_q reads int8 * scale[idx // s], idx >= n_q reads the
hot buffer at idx - n_q.

torch reference ops are exact (float32 math on bf16 inputs); the Triton kernel reproduces
them (the quantize pass of the prefill store is the hot path).
"""
from __future__ import annotations

import os

import torch

from src.utils.compile_guard import compiling

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as _libdevice
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False


def int8_on() -> bool:
    return os.environ.get("FS_KV_INT8", "1") == "1" and not compiling()


def int8_supported(dtype, device) -> bool:
    return (int8_on() and _HAVE_TRITON and torch.device(device).type == "cuda" and dtype == torch.bfloat16)


class Int8Keys:
    """The rerank tokK of an int8-mode cache: int8 cell rows + bf16 cell scales, the open cell's bf16 hot
    rows, and the device scalar n_q (rows below n_q live in the int8 region: value = q * sc[row // s])."""
    __slots__ = ("q", "sc", "hot", "nq_t", "s")

    def __init__(self, q: torch.Tensor, sc: torch.Tensor, hot: torch.Tensor, nq_t: torch.Tensor, s: int):
        self.q, self.sc, self.hot, self.nq_t, self.s = q, sc, hot, nq_t, s


@torch.no_grad()
def quantize_cells(x: torch.Tensor, s: int):
    # x (B,H,N,D) bf16 -> (q int8 (B,H,nc*s,D), scale bf16 (B,H,nc)), the torch reference
    B, H, N, D = x.shape
    nc = N // s
    xc = x[:, :, :nc * s].reshape(B, H, nc, s, D).float()
    sc = xc.abs().amax(dim=(3, 4)) / 127.0
    sc = torch.where(sc > 0, sc, torch.ones_like(sc)).to(torch.bfloat16)
    q = torch.round(xc / sc[:, :, :, None, None].float()).to(torch.int8)
    return q.reshape(B, H, nc * s, D).contiguous(), sc.contiguous()


@torch.no_grad()
def dequant_cells(q: torch.Tensor, sc: torch.Tensor, s: int, dtype=torch.bfloat16, out: torch.Tensor = None):
    # inverse of quantize_cells (covers the quantized region only); sc may hold extra trailing cells
    B, H, N, D = q.shape
    nc = N // s
    x = q.view(B, H, nc, s, D).float() * sc[:, :, :nc, None, None].float()
    if out is None:
        return x.to(dtype).reshape(B, H, nc * s, D)
    out[:, :, :nc * s].copy_(x.reshape(B, H, nc * s, D))
    return out


if _HAVE_TRITON:
    @triton.jit
    def _div_rn_f32(a, b):
        # IEEE division matching torch's fp32 semantics (plain f32 division in Triton is not correctly rounded)
        return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True,
                                         pack=1)

    @triton.jit
    def _quant_cells_kernel(X, Q, SC, nc, H, sxb, sxh, sxn, sqb, sqh, sqn, scb, sch, scc,
                            S: tl.constexpr, D: tl.constexpr, DP: tl.constexpr):
        b = tl.program_id(0)
        bb = b // H
        bh = b % H
        c = tl.program_id(1)
        j = tl.arange(0, S)
        d = tl.arange(0, DP)
        src = X + bb * sxb + bh * sxh + (c * S + j)[:, None] * sxn + d[None, :]
        x = tl.load(src, mask=(j[:, None] >= 0) & (d[None, :] < D) & ((c * S + j)[:, None] < nc * S), other=0.0)
        xf = x.to(tl.float32)
        amax = tl.max(tl.max(tl.abs(xf), axis=1), axis=0)
        sc = tl.where(amax > 0.0, _div_rn_f32(amax, tl.full((), 127.0, tl.float32)), 1.0)
        sc = sc.to(tl.bfloat16).to(tl.float32)
        q = _libdevice.rint(_div_rn_f32(xf, sc)).to(tl.int8)
        dst = Q + bb * sqb + bh * sqh + (c * S + j)[:, None] * sqn + d[None, :]
        tl.store(dst, q, mask=(d[None, :] < D) & ((c * S + j)[:, None] < nc * S))
        tl.store(SC + bb * scb + bh * sch + c * scc, sc.to(tl.bfloat16))


@torch.no_grad()
def quantize_cells_fast(x: torch.Tensor, s: int, q: torch.Tensor = None, sc: torch.Tensor = None):
    # GPU path of quantize_cells; q/sc may be preallocated slices of the cache buffers
    B, H, N, D = x.shape
    nc = N // s
    if not (_HAVE_TRITON and x.is_cuda and x.dtype == torch.bfloat16 and x.stride(-1) == 1
            and 1 <= nc and (nc * s) * s < (1 << 31)):
        return None
    if q is None:
        q = torch.empty((B, H, nc * s, D), dtype=torch.int8, device=x.device)
    if sc is None:
        sc = torch.empty((B, H, nc), dtype=torch.bfloat16, device=x.device)
    _quant_cells_kernel[(B * H, nc)](x, q, sc, nc, H, x.stride(0), x.stride(1), x.stride(2),
                                     q.stride(0), q.stride(1), q.stride(2),
                                     sc.stride(0), sc.stride(1), sc.stride(2),
                                     S=s, D=D, DP=1 << (D - 1).bit_length(), num_warps=4)
    return q, sc
