import torch

if not torch.cuda.is_available():
    import pytest
    pytest.skip("needs a CUDA GPU (Triton kernel test)", allow_module_level=True)

from src.fsattn.attention import FSAttn


# All checks of this file run in one test, in their original order, so every seeded
# random draw (and therefore every checked value) is the same as when the file is run
# as a script.
def test_fused_integration():
    dev = "cuda"
    torch.manual_seed(0)
    d_model, H, N = 64, 4, 256
    Dh = d_model // H
    BLK = 64


    def mk(**kw):
        m = FSAttn(d_model=d_model, n_heads=H, window_size=16, causal=True,
                           long_range_variant="cell_topk", use_rope=False, lr_scale_mode="one",
                           cell_select="score", cell_stride=16, cell_topk=4, sel_strict_causal=1, **kw).to(dev).eval()
        m._cell_block = BLK
        return m


    m = mk()
    Q = torch.randn(1, H, N, Dh, device=dev)
    K = torch.randn(1, H, N, Dh, device=dev)
    V = torch.randn(1, H, N, Dh, device=dev)
    lr_ref = m._lr_cell_topk(Q, K, V, m.sigma, N)
    m.fs_fused_attn = 1
    lr_fus = m._lr_cell_topk(Q, K, V, m.sigma, N)
    d = (lr_ref - lr_fus).abs().max().item()
    print(f"F1 dense strict: |fused - masked-full| = {d:.2e}")
    assert d < 2e-3, f"fused != masked-full: {d}"
    print("F1 OK  (fused indexed-cell == masked-full over the same selected tokens)")

    mf = mk(fs_fused_attn=1)
    x = torch.randn(1, N, d_model, device=dev)
    y1 = mf(x)
    x2 = x.clone(); x2[:, 200] += 5.0
    leak = (y1[:, :200] - mf(x2)[:, :200]).abs().max().item()
    print(f"F2 fused full-module prefix leak @<200 = {leak:.2e}")
    assert leak < 1e-5, f"fused module leaks future: {leak}"
    print("F2 OK  (fused module strictly causal end-to-end)")

    xg = torch.randn(1, N, d_model, device=dev, requires_grad=True)
    mg = mk(fs_fused_attn=1); mg.train()
    mg(xg).square().mean().backward()
    g = xg.grad.norm().item()
    print(f"F3 fused grad norm = {g:.3f}")
    import math
    assert math.isfinite(g) and g > 0
    print("F3 OK  (trainable through the fused kernel)")

    def mk_on(**kw):
        return mk(cell_prune_kernel=1, cell_prune_pool=8, cell_prune_window=2, cell_prune_blkchunk=256, **kw)

    ms = mk_on()
    Qs = torch.randn(1, H, N, Dh, device=dev)
    Ks = torch.randn(1, H, N, Dh, device=dev)
    Vs = torch.randn(1, H, N, Dh, device=dev)
    lr_mp = ms._lr_cell_topk(Qs, Ks, Vs, ms.sigma, N)
    ms.fs_fused_attn = 1
    lr_fs = ms._lr_cell_topk(Qs, Ks, Vs, ms.sigma, N)
    ds = (lr_mp - lr_fs).abs().max().item()
    print(f"S1 sparse strict: |fused - masked-pool| = {ds:.2e}")
    assert ds < 2e-3, f"sparse fused != masked-pool: {ds}"
    print("S1 OK  (fused indexed-cell == masked-pool over the same selected cells)")

    msf = mk_on(fs_fused_attn=1)
    xs = torch.randn(1, N, d_model, device=dev)
    ys1 = msf(xs)
    xs2 = xs.clone(); xs2[:, 200] += 5.0
    leak_s = (ys1[:, :200] - msf(xs2)[:, :200]).abs().max().item()
    print(f"S2 sparse fused full-module prefix leak @<200 = {leak_s:.2e}")
    assert leak_s < 1e-5, f"sparse fused module leaks future: {leak_s}"
    print("S2 OK  (sparse fused module strictly causal end-to-end)")

    print("FUSED INTEGRATION (dense + sparse strict) TESTS PASSED")


if __name__ == "__main__":
    test_fused_integration()
