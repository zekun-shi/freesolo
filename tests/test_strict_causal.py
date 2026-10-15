import math
import torch
import torch.nn.functional as F
from src.fsattn.attention import FSAttn


# All checks of this file run in one test, in their original order, so every seeded
# random draw (and therefore every checked value) is the same as when the file is run
# as a script.
def test_strict_causal():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d_model, H, N = 64, 4, 256
    Dh = d_model // H
    BLK = 64


    def mk(strict, **kw):
        m = FSAttn(d_model=d_model, n_heads=H, window_size=16, causal=True,
                           long_range_variant="cell_topk", use_rope=False, lr_scale_mode="one",
                           cell_select="score", cell_stride=16, cell_topk=4,
                           sel_strict_causal=bool(strict), **kw).to(dev).eval()
        m._cell_block = BLK
        return m


    def lr(m, Q, K, V):
        return m._lr_cell_topk(Q, K, V, m.sigma, N)


    def intra_block_leak(strict):
        m = mk(strict)
        torch.manual_seed(2)
        Q = torch.randn(1, H, N, Dh, device=dev)
        K = torch.randn(1, H, N, Dh, device=dev)
        V = torch.randn(1, H, N, Dh, device=dev)
        o1 = lr(m, Q, K, V)
        Qp, Kp, Vp = Q.clone(), K.clone(), V.clone()
        Qp[0, :, 190] += 5.0; Kp[0, :, 190] += 5.0; Vp[0, :, 190] += 5.0
        o2 = lr(m, Qp, Kp, Vp)
        return (o1[0, :, 130:161] - o2[0, :, 130:161]).abs().max().item()


    leak_off = intra_block_leak(False)
    leak_on = intra_block_leak(True)
    print(f"T7 intra-block leak:  block-causal={leak_off:.2e}   strict-causal={leak_on:.2e}")
    assert leak_on < 1e-5, f"strict-causal STILL leaks intra-block: {leak_on}"
    assert leak_off > 1e-3, f"golden-gate FAIL: block-causal did not leak ({leak_off}) — flag not changing behavior?"
    print("T7 OK  (strict-causal removes the intra-block leak; flag is live)")

    torch.manual_seed(1)
    Q = torch.zeros(1, H, N, Dh, device=dev)
    K = torch.randn(1, H, N, Dh, device=dev) * 0.1
    V = torch.zeros(1, H, N, Dh, device=dev)
    p = 100
    nk = torch.randn(H, Dh, device=dev)
    nv = torch.randn(H, Dh, device=dev)
    K[0, :, p] = nk; V[0, :, p] = nv; Q[0, :, -1] = nk
    m = mk(True)
    cos = F.cosine_similarity(lr(m, Q, K, V)[0, :, -1], nv, dim=-1).mean().item()
    print(f"T8 strict-causal needle cos = {cos:+.3f}")
    assert cos > 0.9, f"strict-causal broke needle recovery: {cos}"
    print("T8 OK")

    xg = torch.randn(1, N, d_model, device=dev, requires_grad=True)
    mg = mk(True); mg.train()
    mg(xg).square().mean().backward()
    g = xg.grad.norm().item()
    print(f"T9 strict-causal grad norm = {g:.3f}")
    assert math.isfinite(g) and g > 0
    print("T9 OK")

    mc = mk(True)
    xt = torch.randn(2, N, d_model, device=dev)
    yt = mc(xt)
    xt2 = xt.clone(); xt2[:, 200:] += 5.0
    dd = (yt[:, :150] - mc(xt2)[:, :150]).abs().max().item()
    print(f"T10 strict-causal cross-block leak@<150 = {dd:.2e}")
    assert dd < 1e-4, f"strict-causal cross-block leak {dd}"
    print("T10 OK")

    mb = mk(True)
    torch.manual_seed(3)
    Qr = torch.randn(1, H, N, Dh, device=dev)
    Kr = torch.randn(1, H, N, Dh, device=dev)
    Vr = torch.randn(1, H, N, Dh, device=dev)
    o0 = lr(mb, Qr, Kr, Vr)
    b0 = o0[0, :, :BLK].abs().max().item()
    print(f"T11 block-0 LR |max| = {b0:.2e}")
    assert b0 < 1e-5, f"block-0 LR output nonzero (invalid-filler leak not fixed): {b0}"
    print("T11 OK  (block 0 has zero LR; in-block history deferred to SR)")

    def full_prefix_leak(strict, p=200):
        m = mk(strict)
        torch.manual_seed(4)
        x = torch.randn(1, N, d_model, device=dev)
        y1 = m(x)
        x2 = x.clone(); x2[:, p] += 5.0
        return (y1[:, :p] - m(x2)[:, :p]).abs().max().item()
    fl_off = full_prefix_leak(False)
    fl_on = full_prefix_leak(True)
    print(f"T12 full-module prefix leak @<200: block-causal={fl_off:.2e}  strict={fl_on:.2e}")
    assert fl_on < 1e-5, f"strict full-module prefix leak {fl_on}"
    assert fl_off > 1e-3, f"golden-gate: expected block-causal to leak intra-block, got {fl_off}"
    print("T12 OK  (strict module prefix-invariant incl. intra-block; block-causal leaks)")

    from experiments.scaling.model import FreeSoloLM
    def lm_prefix_leak(strict, p=200, Ln=256):
        torch.manual_seed(5)
        lm = FreeSoloLM.from_config("debug", attn_type="fsattn", long_range_variant="cell_topk",
                                 use_rope=False, lr_scale_mode="one", cell_select="score",
                                 cell_stride=16, cell_topk=4, window_size=64,
                                 sel_strict_causal=int(strict)).to(dev).eval()
        ids = torch.randint(0, 50257, (1, Ln), device=dev)
        with torch.no_grad():
            lg1 = lm(ids)
            ids2 = ids.clone(); ids2[0, p] = (int(ids2[0, p]) + 123) % 50257
            lg2 = lm(ids2)
        logit_leak = (lg1[0, :p] - lg2[0, :p]).abs().max().item()
        tgt = ids[0, 1:p]
        ce1 = F.cross_entropy(lg1[0, :p - 1], tgt).item()
        ce2 = F.cross_entropy(lg2[0, :p - 1], tgt).item()
        return logit_leak, abs(ce1 - ce2)
    ll_on, ce_on = lm_prefix_leak(True)
    print(f"T13 full-LM strict: prefix-logit leak={ll_on:.2e}  shifted-CE Δ={ce_on:.2e}")
    assert ll_on < 1e-4, f"strict full-LM prefix-logit leak {ll_on}"
    assert ce_on < 1e-5, f"strict full-LM shifted-CE not invariant {ce_on}"
    print("T13 OK  (full-LM prefix logits + shifted-CE invariant to suffix under strict-causal)")

    def mk_on(strict, pool=8, win=2, blkchunk=256, **kw):
        m = mk(strict, cell_prune_kernel=1, cell_prune_pool=pool, cell_prune_window=win,
               cell_prune_blkchunk=blkchunk, **kw)
        return m

    def intra_block_leak_on(strict):
        m = mk_on(strict)
        torch.manual_seed(2)
        Q = torch.randn(1, H, N, Dh, device=dev)
        K = torch.randn(1, H, N, Dh, device=dev)
        V = torch.randn(1, H, N, Dh, device=dev)
        o1 = lr(m, Q, K, V)
        Qp, Kp2, Vp = Q.clone(), K.clone(), V.clone()
        Qp[0, :, 190] += 5.0; Kp2[0, :, 190] += 5.0; Vp[0, :, 190] += 5.0
        o2 = lr(m, Qp, Kp2, Vp)
        return (o1[0, :, 130:161] - o2[0, :, 130:161]).abs().max().item()
    on_on = intra_block_leak_on(True)
    mg_bc, mg_st = mk_on(False), mk_on(True)
    torch.manual_seed(2)
    Qg = torch.randn(1, H, N, Dh, device=dev); Kg = torch.randn(1, H, N, Dh, device=dev); Vg = torch.randn(1, H, N, Dh, device=dev)
    flagdiff = (lr(mg_bc, Qg, Kg, Vg) - lr(mg_st, Qg, Kg, Vg)).abs().max().item()
    print(f"T14 O(N) strict intra-block leak={on_on:.2e}   block-causal-vs-strict output diff={flagdiff:.2e}")
    assert on_on < 1e-5, f"strict O(N) STILL leaks intra-block: {on_on}"
    assert flagdiff > 1e-3, f"golden-gate FAIL: block-causal O(N) == strict O(N), flag not live ({flagdiff})"
    print("T14 OK  (strict O(N) intra-block leak ~0; strict flag changes the O(N) selection)")

    mb2 = mk_on(True)
    torch.manual_seed(3)
    Qr = torch.randn(1, H, N, Dh, device=dev)
    Kr = torch.randn(1, H, N, Dh, device=dev)
    Vr = torch.randn(1, H, N, Dh, device=dev)
    b0on = lr(mb2, Qr, Kr, Vr)[0, :, :BLK].abs().max().item()
    print(f"T15 O(N) block-0 LR |max| = {b0on:.2e}")
    assert b0on < 1e-5, f"strict O(N) block-0 LR nonzero: {b0on}"
    print("T15 OK  (strict O(N) block 0 has zero LR)")

    def full_prefix_leak_on(strict, p=200):
        m = mk_on(strict)
        torch.manual_seed(4)
        x = torch.randn(1, N, d_model, device=dev)
        y1 = m(x)
        x2 = x.clone(); x2[:, p] += 5.0
        return (y1[:, :p] - m(x2)[:, :p]).abs().max().item()
    flon_on = full_prefix_leak_on(True)
    print(f"T16 O(N) strict full-module prefix leak @<200 = {flon_on:.2e}")
    assert flon_on < 1e-5, f"strict O(N) full-module prefix leak {flon_on}"
    print("T16 OK  (strict O(N) module prefix-invariant incl. intra-block; flag-live proven in T14)")

    M_cells = (N + 16 - 1) // 16
    torch.manual_seed(7)
    Qe = torch.randn(1, H, N, Dh, device=dev)
    Ke = torch.randn(1, H, N, Dh, device=dev)
    Ve = torch.randn(1, H, N, Dh, device=dev)
    m_dense = mk(True)
    m_on_full = mk_on(True, pool=M_cells, win=2)
    o_dense = lr(m_dense, Qe, Ke, Ve)
    o_on = lr(m_on_full, Qe, Ke, Ve)
    eqv = (o_dense - o_on).abs().max().item()
    print(f"T17 |strict-O(N) - dense-strict| (full pool) = {eqv:.2e}")
    assert eqv < 1e-4, f"strict O(N) full-pool != dense strict: {eqv}"
    print("T17 OK  (strict O(N) with full pool reproduces the dense strict path)")

    print("ALL STRICT-CAUSAL TESTS PASSED")


if __name__ == "__main__":
    test_strict_causal()
