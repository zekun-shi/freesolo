import torch
from src.fsattn.prefix_topp import (
    causal_prefix_topp_rescan, causal_prefix_topp_incremental, _completed_hi,
)


# All checks of this file run in one test, in their original order, so every seeded
# random draw (and therefore every checked value) is the same as when the file is run
# as a script.
def test_prefix_topp():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)


    def sets_per_block(idx):
        B, H, nb, P = idx.shape
        return [[frozenset(int(x) for x in idx[b, h, n].tolist() if x >= 0)
                 for n in range(nb)] for b in range(B) for h in range(H)]


    configs = [
        (1, 4, 64, 16, 64, 8),
        (2, 3, 128, 32, 128, 16),
        (1, 2, 512, 32, 128, 64),
        (1, 1, 53, 6, 17, 5),
    ]
    for (B, H, M, s, Bq, Pk) in configs:
        N = M * s
        nb = (N + Bq - 1) // Bq
        ceil_c = torch.rand(B, H, M, device=dev) * 5.0
        cell_end = (torch.arange(M, device=dev, dtype=torch.float32) + 1.0) * s
        blk_start = torch.arange(nb, device=dev, dtype=torch.float32) * Bq
        r = causal_prefix_topp_rescan(ceil_c, cell_end, blk_start, Pk)
        inc = causal_prefix_topp_incremental(ceil_c, s, Bq, nb, Pk)
        sr, si = sets_per_block(r), sets_per_block(inc)
        ok = all(a == b for row_r, row_i in zip(sr, si) for a, b in zip(row_r, row_i))
        print(f"P1 cfg B{B}H{H}M{M}s{s}Bq{Bq}P{Pk} nb{nb}: set-match={ok}")
        assert ok, f"incremental != rescan for cfg {(B,H,M,s,Bq,Pk)}"
    print("P1 OK  (incremental top-P SET == rescan, all configs incl. non-multiple)")

    B, H, M, s, Bq, Pk = 1, 2, 64, 16, 64, 8
    N = M * s; nb = (N + Bq - 1) // Bq
    ceil_c = torch.rand(B, H, M, device=dev)
    inc = causal_prefix_topp_incremental(ceil_c, s, Bq, nb, Pk)
    assert (inc[:, :, 0, :] < 0).all(), "block 0 must have no completed cells (all -1)"
    for b in range(nb):
        hi = _completed_hi(b, Bq, s)
        valid = inc[0, 0, b][inc[0, 0, b] >= 0]
        assert (valid <= hi).all(), f"block {b} selected a cell > hi={hi}"
        assert valid.numel() == min(Pk, hi + 1) if hi >= 0 else valid.numel() == 0
    print("P2 OK  (block-0 empty; every block's cells within its completed prefix, right count)")

    inc2 = causal_prefix_topp_incremental(ceil_c, s, Bq, nb, Pk)
    assert torch.equal(inc, inc2), "incremental not deterministic"
    ceil_tie = torch.ones(1, 1, 32, device=dev)
    nbt = (32 * 16 + 63) // 64
    inct = causal_prefix_topp_incremental(ceil_tie, 16, 64, nbt, 8)
    inct2 = causal_prefix_topp_incremental(ceil_tie, 16, 64, nbt, 8)
    assert torch.equal(inct, inct2), "tie case not deterministic"
    hi_last = _completed_hi(nbt - 1, 64, 16)
    last = inct[0, 0, -1][inct[0, 0, -1] >= 0]
    assert last.numel() == min(8, hi_last + 1) and (last <= hi_last).all() and (last >= 0).all()
    assert last.unique().numel() == last.numel(), "tie selection has duplicate cells"
    print("P3 OK  (deterministic; exact-tie selection valid + duplicate-free)")

    B, H, M, s, Bq, Pk = 1, 1, 4096, 32, 128, 64
    N = M * s; nb = (N + Bq - 1) // Bq
    merges = 0
    prev_hi = -1
    for b in range(nb):
        hi = _completed_hi(b, Bq, s)
        if hi > prev_hi:
            merges += (Pk + (hi - prev_hi))
            prev_hi = hi
    rescan_work = nb * M
    print(f"P4 N={N} nb={nb} M={M}: incremental merged-elems={merges}  rescan-scanned={rescan_work}"
          f"  ratio={rescan_work/merges:.0f}x")
    assert merges < rescan_work / 10, "incremental not sufficiently sub-rescan"
    print("P4 OK  (incremental work << rescan; O(N) vs O(N^2/(Bq*s)))")

    print("ALL PREFIX-TOPP TESTS PASSED")


if __name__ == "__main__":
    test_prefix_topp()
