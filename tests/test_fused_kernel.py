import torch

if not torch.cuda.is_available():
    import pytest
    pytest.skip("needs a CUDA GPU (Triton kernel test)", allow_module_level=True)

from src.fsattn.cell_select_triton import indexed_cell_attention


# All checks of this file run in one test, in their original order, so every seeded
# random draw (and therefore every checked value) is the same as when the file is run
# as a script.
def test_fused_kernel():
    dev = "cuda"
    torch.manual_seed(0)


    def make_sel(B, H, N, Bq, s, K):
        n_cells = (N + s - 1) // s
        sel = torch.full((B, H, N, K), -1, dtype=torch.int32, device=dev)
        for p in range(N):
            hi = (p // Bq) * Bq // s - 1
            if hi < 0:
                continue
            avail = hi + 1
            for b in range(B):
                for h in range(H):
                    kk = min(K, avail)
                    perm = torch.randperm(avail, device=dev)[:kk]
                    sel[b, h, p, :kk] = perm.to(torch.int32)
        return sel


    def ref_indexed_attn(q, k, v, sel, s, Bq, scale):
        B, H, N, D = q.shape
        K = sel.shape[-1]
        KK = K * s
        NEG = -1e30
        qpos = torch.arange(N, device=dev)
        block_start = (qpos // Bq) * Bq
        n_cells = (N + s - 1) // s
        off = torch.arange(s, device=dev)
        cell = sel.clamp(min=0).long()
        tok = (cell.unsqueeze(-1) * s + off).reshape(B, H, N, KK)
        cell_ok = (sel >= 0) & (sel < n_cells) & (((sel.long() + 1) * s) <= block_start.view(1, 1, N, 1))
        tok_ok = cell_ok.unsqueeze(-1).expand(B, H, N, K, s).reshape(B, H, N, KK)
        tok_ok = tok_ok & (tok < N) & (tok < qpos.view(1, 1, N, 1))
        tokc = tok.clamp(0, N - 1)
        idx = tokc.unsqueeze(-1).expand(B, H, N, KK, D)
        Kg = torch.gather(k.unsqueeze(2).expand(B, H, N, N, D), 3, idx)
        Vg = torch.gather(v.unsqueeze(2).expand(B, H, N, N, D), 3, idx)
        logits = (q.unsqueeze(3) * Kg).sum(-1).float() * scale
        logits = logits.masked_fill(~tok_ok, NEG)
        w = torch.softmax(logits, dim=-1).to(v.dtype)
        out = (w.unsqueeze(-1) * Vg).sum(3)
        out = out * tok_ok.any(-1, keepdim=True).to(out.dtype)
        return out


    def run(dtype, tol):
        B, H, N, D, Bq, s, K = 2, 4, 256, 64, 64, 16, 4
        scale = 1.0 / (D ** 0.5)
        q0 = torch.randn(B, H, N, D, device=dev)
        k0 = torch.randn(B, H, N, D, device=dev)
        v0 = torch.randn(B, H, N, D, device=dev)
        sel = make_sel(B, H, N, Bq, s, K)
        go = torch.randn(B, H, N, D, device=dev)

        def prep(t):
            t = t.to(dtype).clone().requires_grad_(True)
            return t
        qa, ka, va = prep(q0), prep(k0), prep(v0)
        qb, kb, vb = prep(q0), prep(k0), prep(v0)

        o_ref = ref_indexed_attn(qa, ka, va, sel, s, Bq, scale)
        o_ker = indexed_cell_attention(qb, kb, vb, sel, cell_size=s, query_block=Bq, scale=scale)
        df = (o_ref.float() - o_ker.float()).abs().max().item()

        o_ref.backward(go.to(dtype)); o_ker.backward(go.to(dtype))
        dq = (qa.grad.float() - qb.grad.float()).abs().max().item()
        dk = (ka.grad.float() - kb.grad.float()).abs().max().item()
        dv = (va.grad.float() - vb.grad.float()).abs().max().item()
        b0 = o_ker[:, :, :Bq].abs().max().item()
        print(f"{dtype}: |out|={df:.2e}  dQ={dq:.2e} dK={dk:.2e} dV={dv:.2e}  block0={b0:.2e}")
        assert df < tol and dq < tol and dk < tol and dv < tol, f"{dtype} mismatch > {tol}"
        assert b0 < 1e-6, "block-0 not zero"


    def causality(p=130):
        B, H, N, D, Bq, s, K = 1, 4, 256, 64, 64, 16, 4
        scale = 1.0 / (D ** 0.5)
        q = torch.randn(B, H, N, D, device=dev)
        k = torch.randn(B, H, N, D, device=dev)
        v = torch.randn(B, H, N, D, device=dev)
        sel = make_sel(B, H, N, Bq, s, K)
        o1 = indexed_cell_attention(q, k, v, sel, cell_size=s, query_block=Bq, scale=scale)
        qp, kp, vp = q.clone(), k.clone(), v.clone()
        qp[:, :, p] += 7.0; kp[:, :, p] += 7.0; vp[:, :, p] += 7.0
        o2 = indexed_cell_attention(qp, kp, vp, sel, cell_size=s, query_block=Bq, scale=scale)
        before = (o1[:, :, :p] - o2[:, :, :p]).abs().max().item()
        after = (o1[:, :, p:] - o2[:, :, p:]).abs().max().item()
        print(f"causality: |Δ queries<{p}|={before:.2e}   |Δ queries>={p}|={after:.2e}")
        assert before < 1e-6, f"FUTURE-TOKEN LEAK: query<{p} changed by {before}"
        assert after > 1e-3, f"golden-gate: perturbation not live ({after})"
        print(f"causality OK  (kernel token-causal: token {p} cannot affect any earlier query)")


    print("== fused kernel vs independent PyTorch reference ==")
    run(torch.float32, 2e-4)
    run(torch.float16, 3e-2)
    run(torch.bfloat16, 8e-2)
    print("FUSED KERNEL MATCHES INDEPENDENT REFERENCE (fwd + dQ/dK/dV, fp32/fp16/bf16)")
    causality()


if __name__ == "__main__":
    test_fused_kernel()
