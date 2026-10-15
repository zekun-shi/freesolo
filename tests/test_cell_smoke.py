import math
import torch
import torch.nn.functional as F
from src.fsattn.attention import FSAttn


# All checks of this file run in one test, in their original order, so every seeded
# random draw (and therefore every checked value) is the same as when the file is run
# as a script.
def test_cell_smoke():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d_model, H, N = 64, 4, 256
    Dh = d_model // H


    def mk(**kw):
        m = FSAttn(d_model=d_model, n_heads=H, window_size=16, causal=True,
                   long_range_variant="cell_topk", use_rope=False,
                   lr_scale_mode="one", **kw).to(dev).eval()
        return m


    m1 = mk(cell_stride=16, cell_topk=4)
    x = torch.randn(2, N, d_model, device=dev)
    y = m1(x)
    assert y.shape == (2, N, d_model) and torch.isfinite(y).all()
    print("T1 forward shape+finite OK", tuple(y.shape))

    torch.manual_seed(1)
    Q = torch.zeros(1, H, N, Dh, device=dev)
    K = torch.randn(1, H, N, Dh, device=dev) * 0.1
    V = torch.zeros(1, H, N, Dh, device=dev)
    p = 100
    nk = torch.randn(H, Dh, device=dev)
    nv = torch.randn(H, Dh, device=dev)
    K[0, :, p] = nk
    V[0, :, p] = nv
    Q[0, :, -1] = nk

    for kk in (1, 4):
        m = mk(cell_stride=16, cell_topk=kk)
        m._cell_block = 64
        o = m._lr_cell_topk(Q, K, V, m.sigma, N)
        cos = F.cosine_similarity(o[0, :, -1], nv, dim=-1).mean().item()
        print(f"T2 cell_topk(k={kk}) needle cos = {cos:+.3f}")
        assert cos > 0.9, f"cell_topk(k={kk}) not exact-recovering the needle: {cos}"
        xt = torch.randn(2, N, d_model, device=dev)
        yt = m(xt); xt2 = xt.clone(); xt2[:, 200:] += 5.0
        dd = (yt[:, :150] - m(xt2)[:, :150]).abs().max().item()
        assert dd < 1e-4, f"cell_topk(k={kk}) causality leak {dd}"
        xg2 = torch.randn(1, N, d_model, device=dev, requires_grad=True); m.train()
        m(xg2).square().mean().backward()
        g = xg2.grad.norm().item()
        assert math.isfinite(g) and g > 0
        m.eval()
    print("T2 cell_topk exact recovery + causal + grads OK")

    print("ALL CELL SMOKE TESTS PASSED")


if __name__ == "__main__":
    test_cell_smoke()
