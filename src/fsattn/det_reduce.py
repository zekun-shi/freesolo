import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False

_MAX_P = 16384
_TILE = 8192


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _py_halving(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    p = _pow2(n)
    if p != n:
        x = torch.cat([x, x.new_zeros(*x.shape[:-1], p - n)], dim=-1)
    while x.shape[-1] > 1:
        h = x.shape[-1] // 2
        x = x[..., :h] + x[..., h:]
    return x.squeeze(-1)


if _HAVE_TRITON:
    @triton.jit
    def _halving_rows_kernel(X, OUT, R, N, P: tl.constexpr, BR: tl.constexpr):
        # BR contiguous rows of length N per program; each row is zero-padded to P and summed by pairwise
        # halving, x[:h] + x[h:], one IEEE add per element and level
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        offs = tl.arange(0, P)
        x = tl.load(X + r.to(tl.int64)[:, None] * N + offs[None, :],
                    mask=(r[:, None] < R) & (offs[None, :] < N), other=0.0)
        if P >= 16384:
            x = tl.sum(tl.reshape(x, (BR, 2, 8192)), axis=1)
        if P >= 8192:
            x = tl.sum(tl.reshape(x, (BR, 2, 4096)), axis=1)
        if P >= 4096:
            x = tl.sum(tl.reshape(x, (BR, 2, 2048)), axis=1)
        if P >= 2048:
            x = tl.sum(tl.reshape(x, (BR, 2, 1024)), axis=1)
        if P >= 1024:
            x = tl.sum(tl.reshape(x, (BR, 2, 512)), axis=1)
        if P >= 512:
            x = tl.sum(tl.reshape(x, (BR, 2, 256)), axis=1)
        if P >= 256:
            x = tl.sum(tl.reshape(x, (BR, 2, 128)), axis=1)
        if P >= 128:
            x = tl.sum(tl.reshape(x, (BR, 2, 64)), axis=1)
        if P >= 64:
            x = tl.sum(tl.reshape(x, (BR, 2, 32)), axis=1)
        if P >= 32:
            x = tl.sum(tl.reshape(x, (BR, 2, 16)), axis=1)
        if P >= 16:
            x = tl.sum(tl.reshape(x, (BR, 2, 8)), axis=1)
        if P >= 8:
            x = tl.sum(tl.reshape(x, (BR, 2, 4)), axis=1)
        if P >= 4:
            x = tl.sum(tl.reshape(x, (BR, 2, 2)), axis=1)
        if P >= 2:
            x = tl.sum(tl.reshape(x, (BR, 2, 1)), axis=1)
        tl.store(OUT + r, tl.sum(x, axis=1), mask=r < R)

    @triton.jit
    def _halving_strided_kernel(X, OUT, O, N, B, P: tl.constexpr, CO: tl.constexpr):
        # x viewed as (A, N, B): output o = a * B + b sums x[a, :, b] by the same pairwise halving; CO outputs
        # per program, so consecutive b stay contiguous in every loaded row
        o = tl.program_id(0) * CO + tl.arange(0, CO)
        a = o // B
        base = a.to(tl.int64) * N * B + (o - a * B)
        j = tl.arange(0, P)
        x = tl.load(X + base[None, :] + j.to(tl.int64)[:, None] * B,
                    mask=(j[:, None] < N) & (o[None, :] < O), other=0.0)
        if P >= 16384:
            x = tl.sum(tl.reshape(x, (2, 8192, CO)), axis=0)
        if P >= 8192:
            x = tl.sum(tl.reshape(x, (2, 4096, CO)), axis=0)
        if P >= 4096:
            x = tl.sum(tl.reshape(x, (2, 2048, CO)), axis=0)
        if P >= 2048:
            x = tl.sum(tl.reshape(x, (2, 1024, CO)), axis=0)
        if P >= 1024:
            x = tl.sum(tl.reshape(x, (2, 512, CO)), axis=0)
        if P >= 512:
            x = tl.sum(tl.reshape(x, (2, 256, CO)), axis=0)
        if P >= 256:
            x = tl.sum(tl.reshape(x, (2, 128, CO)), axis=0)
        if P >= 128:
            x = tl.sum(tl.reshape(x, (2, 64, CO)), axis=0)
        if P >= 64:
            x = tl.sum(tl.reshape(x, (2, 32, CO)), axis=0)
        if P >= 32:
            x = tl.sum(tl.reshape(x, (2, 16, CO)), axis=0)
        if P >= 16:
            x = tl.sum(tl.reshape(x, (2, 8, CO)), axis=0)
        if P >= 8:
            x = tl.sum(tl.reshape(x, (2, 4, CO)), axis=0)
        if P >= 4:
            x = tl.sum(tl.reshape(x, (2, 2, CO)), axis=0)
        if P >= 2:
            x = tl.sum(tl.reshape(x, (2, 1, CO)), axis=0)
        tl.store(OUT + o, tl.sum(x, axis=0), mask=o < O)


def det_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    # fixed-order pairwise-halving sum, fp32; batched and per-item calls stay bit-identical
    dim = dim % x.dim()
    n = x.shape[dim]
    p = _pow2(n)
    if (not _HAVE_TRITON) or (not x.is_cuda) or p > _MAX_P or x.dtype != torch.float32:
        return _py_halving(x.movedim(dim, -1))
    out = torch.empty(x.shape[:dim] + x.shape[dim + 1:], dtype=torch.float32, device=x.device)
    O = out.numel()
    if O == 0:
        return out
    if n == 0:
        return out.zero_()
    xc = x.contiguous()
    B = 1
    for d in x.shape[dim + 1:]:
        B *= d
    if B == 1:
        br = min(max(1, _TILE // p), _pow2(O))
        _halving_rows_kernel[(triton.cdiv(O, br),)](xc, out, O, n, P=p, BR=br, num_warps=4 if br * p <= 4096 else 8)
    else:
        co = min(max(1, _TILE // p), _pow2(O))
        _halving_strided_kernel[(triton.cdiv(O, co),)](xc, out, O, n, B, P=p, CO=co,
                                                        num_warps=4 if co * p <= 4096 else 8)
    return out
