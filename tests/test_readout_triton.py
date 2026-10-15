"""Fused prefill readout + window combine (src/fsattn/readout_triton.py) against an fp32 reference, compared with the
error of the path it replaces (indexed_cell_attention, then lr_norm, long_range_scale and + sr under bf16 autocast,
cast to bf16 at W_o)."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU (Triton kernel test)", allow_module_level=True)

from src.fsattn.attention import RMSNorm
from src.fsattn.cell_select_triton import indexed_cell_attention
from src.fsattn.readout_triton import readout_combine, readout_ok

S, BQ, KK, D = 32, 128, 4, 64


def _make_sel(B, H, N, g, mode="random"):
    # selections as the selectors write them, plus the cases the kernel must skip: -1, repeats within a query, cells
    # not complete before the query's block, cells past the context, all-invalid rows
    Npad = (N + BQ - 1) // BQ * BQ
    q = torch.arange(Npad, device="cuda")
    hi = (q // BQ) * BQ // S                         # cells [0, hi) are valid for the query's block
    if mode == "shared":                             # every query of a block reads the same 4 cells
        base = torch.randint(0, 1 << 20, (B, H, Npad // BQ, 1, KK), device="cuda", generator=g)
        r = base.expand(B, H, Npad // BQ, BQ, KK).reshape(B, H, Npad, KK)
    elif mode == "spread":                           # every query reads 4 different cells
        r = torch.randint(0, 1 << 20, (B, H, Npad, KK), device="cuda", generator=g)
    else:                                            # a few dozen distinct cells per block, as on real prompts
        pool = torch.randint(0, 1 << 20, (B, H, Npad // BQ, 1, 40), device="cuda", generator=g)
        pick = torch.randint(0, 40, (B, H, Npad // BQ, BQ, KK), device="cuda", generator=g)
        r = torch.gather(pool.expand(-1, -1, -1, BQ, -1), -1, pick).reshape(B, H, Npad, KK)
    sel = torch.where(hi.view(1, 1, Npad, 1) > 0, r % hi.clamp(min=1).view(1, 1, Npad, 1), torch.full_like(r, -1))
    u = torch.rand(B, H, Npad, KK, device="cuda", generator=g)
    sel = torch.where(u < 0.05, torch.full_like(sel, -1), sel)                         # empty slots
    sel = torch.where((u >= 0.05) & (u < 0.08), hi.view(1, 1, Npad, 1) + 3, sel)       # not complete yet
    sel = torch.where((u >= 0.08) & (u < 0.10), torch.full_like(sel, 1 << 22), sel)    # past the context
    sel[..., 1] = torch.where(u[..., 1] < 0.2, sel[..., 0], sel[..., 1])               # repeated cell
    sel[:, :, 5::97] = -1                                                              # all-invalid rows
    return sel


def _ref_fp32(Q, K, V, sel, sr, w, g, eps, scale):
    # exact attention over each query's distinct valid cells (fp32), fp32 RMSNorm * w * g + sr
    B, H, N, Dh = Q.shape
    n_cells = (K.shape[2] + S - 1) // S
    out = torch.empty(B, N, H, Dh, device="cuda", dtype=torch.float32)
    off = torch.arange(S, device="cuda")
    Kf, Vf = K.float(), V.float()
    for q0 in range(0, N, 1024):
        q1 = min(N, q0 + 1024)
        qp = torch.arange(q0, q1, device="cuda")
        bs = (qp // BQ) * BQ
        c = sel[:, :, q0:q1].long()
        ok = (c >= 0) & (c < n_cells) & ((c + 1) * S <= bs.view(1, 1, -1, 1))
        dup = torch.zeros_like(ok)
        for j in range(1, KK):
            for i in range(j):
                dup[..., j] |= ok[..., i] & (c[..., i] == c[..., j])
        ok &= ~dup
        tok = (c.clamp(min=0, max=n_cells - 1)[..., None] * S + off).reshape(B, H, q1 - q0, KK * S)
        tok = tok.clamp(max=K.shape[2] - 1)
        tok_ok = ok[..., None].expand(B, H, q1 - q0, KK, S).reshape(B, H, q1 - q0, KK * S)
        idx = tok[..., None].expand(-1, -1, -1, -1, Dh)
        Kg = torch.gather(Kf[:, :, None].expand(B, H, q1 - q0, K.shape[2], Dh), 3, idx)
        Vg = torch.gather(Vf[:, :, None].expand(B, H, q1 - q0, K.shape[2], Dh), 3, idx)
        lg = (Q[:, :, q0:q1, None, :].float() * Kg).sum(-1) * scale
        lg = lg.masked_fill(~tok_ok, float("-inf"))
        p = torch.softmax(lg, dim=-1).nan_to_num(0.0)
        o = (p[..., None] * Vg).sum(3)
        o = o * tok_ok.any(-1, keepdim=True)
        lr = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * w.float()
        out[:, q0:q1] = (sr[:, :, q0:q1].float() + g.float().view(1, H, 1, 1) * lr).transpose(1, 2)
    return out.reshape(B, N, H * Dh)


def _old_path(Q, K, V, sel, sr, norm, g, scale, query_block=BQ):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = indexed_cell_attention(Q, K, V, sel, cell_size=S, query_block=query_block, scale=scale)
        lr = norm(o.to(Q.dtype))
        y = sr + g.unsqueeze(0) * lr
        B, H, N, Dh = Q.shape
        return y.transpose(1, 2).contiguous().view(B, N, H * Dh).to(torch.bfloat16)


def _inputs(B, H, N, seed, mode="random", sr_layout="window", sel_dtype=torch.int64):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # model layouts: Q / K / V are transposed (B, N, H, D) linear outputs
    Q, K, V = ((torch.randn(B, N, H, D, device="cuda", generator=g) * sc).to(torch.bfloat16).transpose(1, 2)
               for sc in (1.0, 1.0, 1.0))
    sel = _make_sel(B, H, N, g, mode).to(sel_dtype)
    if sr_layout == "window":   # batched window output: (B, H, N, D) view of a (B, N, H, D) buffer
        sr = (torch.randn(B, N, H, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16).transpose(1, 2)
    else:
        sr = (torch.randn(B, H, N, D, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    norm = RMSNorm(D).cuda()
    with torch.no_grad():
        norm.scale.copy_(1 + 0.2 * torch.randn(D, device="cuda", generator=g))
    norm = norm.to(torch.bfloat16)
    lrs = (0.1 + 0.05 * torch.randn(H, 1, 1, device="cuda", generator=g)).to(torch.bfloat16)
    return Q, K, V, sel, sr, norm, lrs


def _ulp_bf16(x):
    return torch.exp2(torch.floor(torch.log2(x.abs().clamp(min=1e-30))) - 7)


@pytest.mark.parametrize("B,H,N,mode,sr_layout,sel_dtype", [
    (1, 4, 300, "random", "window", torch.int64),
    (2, 4, 1000, "random", "contig", torch.int32),
    (1, 16, 4096, "random", "window", torch.int64),
    (1, 4, 4096, "spread", "window", torch.int32),
    (1, 4, 4096, "shared", "contig", torch.int64),
    (1, 2, 16384 + 77, "random", "window", torch.int64),
])
@torch.no_grad()
def test_readout_combine_error_vs_fp32(B, H, N, mode, sr_layout, sel_dtype):
    Q, K, V, sel, sr, norm, lrs = _inputs(B, H, N, N + B * 7, mode, sr_layout, sel_dtype)
    scale = D ** -0.5
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert readout_ok(Q, K, V, sel, sr, cell_size=S, query_block=BQ)
        new = readout_combine(Q, K, V, sel, sr, norm.scale, lrs, norm.eps, scale=scale, cell_size=S, query_block=BQ)
        again = [readout_combine(Q, K, V, sel, sr, norm.scale, lrs, norm.eps, scale=scale, cell_size=S,
                                 query_block=BQ) for _ in range(2)]
    old = _old_path(Q, K, V, sel, sr, norm, lrs, scale)
    ref = _ref_fp32(Q, K, V, sel, sr, norm.scale, lrs, norm.eps, scale)
    assert new.shape == (B, N, H * D) and new.dtype == torch.bfloat16 and new.is_contiguous()
    assert all(torch.equal(new, a) for a in again)                      # deterministic
    e_new = (new.float() - ref).abs()
    e_old = (old.float() - ref).abs()
    # no worse than the replaced path: max error within one bf16 ulp of max |ref|, mean error not larger
    assert e_new.max() <= e_old.max() + _ulp_bf16(ref.abs().max()), (e_new.max().item(), e_old.max().item())
    assert e_new.mean() <= e_old.mean(), (e_new.mean().item(), e_old.mean().item())
    # rows without any valid cell are exactly sr
    rows0 = torch.arange(min(N, BQ), device="cuda")
    assert torch.equal(new[:, rows0], sr[:, :, rows0].transpose(1, 2).reshape(B, -1, H * D).to(torch.bfloat16))


@torch.no_grad()
def test_readout_combine_fp32_parameters():
    # RULER regime: fp32 lr_norm weight and long_range_scale under bf16 autocast (bf16 activations)
    B, H, N = 1, 4, 4096
    Q, K, V, sel, sr, norm, lrs = _inputs(B, H, N, 21)
    norm, lrs = norm.float(), lrs.float()
    scale = D ** -0.5
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert readout_ok(Q, K, V, sel, sr, cell_size=S, query_block=BQ)
        new = readout_combine(Q, K, V, sel, sr, norm.scale, lrs, norm.eps, scale=scale, cell_size=S, query_block=BQ)
    old = _old_path(Q, K, V, sel, sr, norm, lrs, scale)
    ref = _ref_fp32(Q, K, V, sel, sr, norm.scale, lrs, norm.eps, scale)
    e_new, e_old = (new.float() - ref).abs(), (old.float() - ref).abs()
    assert e_new.max() <= e_old.max() + _ulp_bf16(ref.abs().max())
    assert e_new.mean() <= e_old.mean()


@torch.no_grad()
def test_readout_combine_causal():
    # perturbing token p changes no query before p
    B, H, N, p = 1, 4, 2048, 1300
    Q, K, V, sel, sr, norm, lrs = _inputs(B, H, N, 11)
    scale = D ** -0.5
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y1 = readout_combine(Q, K, V, sel, sr, norm.scale, lrs, norm.eps, scale=scale, cell_size=S, query_block=BQ)
        K2, V2 = K.clone(), V.clone()
        K2[:, :, p] += 3.0
        V2[:, :, p] += 3.0
        y2 = readout_combine(Q, K2, V2, sel, sr, norm.scale, lrs, norm.eps, scale=scale, cell_size=S, query_block=BQ)
    assert torch.equal(y1[:, :p], y2[:, :p])
    assert not torch.equal(y1[:, p:], y2[:, p:])


@torch.no_grad()
def test_readout_ok_gates():
    Q, K, V, sel, sr, norm, lrs = _inputs(1, 2, 512, 3)
    assert not readout_ok(Q, K, V, sel, sr, cell_size=S, query_block=BQ)          # no autocast
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert readout_ok(Q, K, V, sel, sr, cell_size=S, query_block=BQ)
        assert not readout_ok(Q, K, V, sel, sr, cell_size=S, query_block=BQ, q_pos=5)
        assert not readout_ok(Q, K, V, sel, sr.float(), cell_size=S, query_block=BQ)
        with torch.enable_grad():
            assert not readout_ok(Q, K, V, sel, sr, cell_size=S, query_block=BQ)


@torch.no_grad()
def test_readout_short_prompts():
    # Prompts of 2-129 tokens: the callers pass query_block = min(128, N), and the kernel needs a power-of-two group
    # of at least 16 queries (2-63 tokens other than 16 and 32 failed to compile); the other lengths take the eager
    # readout. Up to 128 tokens every query sits in block 0, which has no complete cell, so the fused output is sr
    # and equals the eager path's bit for bit; at 129 the last query reads cells 0-3.
    import src.fsattn.readout_triton as rt
    scale = D ** -0.5
    taken = []
    for N in range(2, 130):
        bq = min(BQ, N)
        Q, K, V, sel, sr, norm, lrs = _inputs(1, 2, N, 1000 + N)
        G = min(rt._G, bq)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ok = readout_ok(Q, K, V, sel, sr, cell_size=S, query_block=bq)
            assert ok == (G >= 16 and (G & (G - 1)) == 0 and bq % G == 0), N
            if not ok:          # the eager readout, as on main (tests/test_short_prompts.py runs it at every length)
                continue
            new = readout_combine(Q, K, V, sel, sr, norm.scale, lrs, norm.eps, scale=scale, cell_size=S,
                                  query_block=bq)
        old = _old_path(Q, K, V, sel, sr, norm, lrs, scale, query_block=bq)
        taken.append(N)
        if N <= BQ:
            assert torch.equal(new, old), N
            assert torch.equal(new, sr.transpose(1, 2).reshape(1, N, -1)), N
        else:
            ref = _ref_fp32(Q, K, V, sel, sr, norm.scale, lrs, norm.eps, scale)
            e_new, e_old = (new.float() - ref).abs(), (old.float() - ref).abs()
            assert e_new.max() <= e_old.max() + _ulp_bf16(ref.abs().max()), N
            assert torch.equal(new[:, :BQ], old[:, :BQ]), N
    assert taken == [16, 32, 64, 128, 129]


@torch.no_grad()
def test_readout_ok_group_size(monkeypatch):
    # FSL_READOUT_G must give a power-of-two group of at least 16 queries dividing the query block; a valid one changes
    # which queries share a program, not any query's result beyond the kernel's tolerance
    import src.fsattn.readout_triton as rt
    B, H, N = 1, 2, 1000
    Q, K, V, sel, sr, norm, lrs = _inputs(B, H, N, 4)
    scale = D ** -0.5
    ref = _ref_fp32(Q, K, V, sel, sr, norm.scale, lrs, norm.eps, scale)
    e_old = (_old_path(Q, K, V, sel, sr, norm, lrs, scale).float() - ref).abs()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for g, ok in ((8, False), (48, False), (100, False), (16, True), (32, True), (64, True), (128, True)):
            monkeypatch.setattr(rt, "_G", g)
            assert readout_ok(Q, K, V, sel, sr, cell_size=S, query_block=BQ) == ok, g
            if ok:
                new = readout_combine(Q, K, V, sel, sr, norm.scale, lrs, norm.eps, scale=scale, cell_size=S,
                                      query_block=BQ)
                e_new = (new.float() - ref).abs()
                assert e_new.max() <= e_old.max() + _ulp_bf16(ref.abs().max()), g
                assert e_new.mean() <= e_old.mean(), g
