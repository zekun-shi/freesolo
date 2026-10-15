import torch

_NEG = -1e30


def stable_topk_slots(scores, cell_ids, k):
    ids = cell_ids.expand_as(scores)
    order = ids.masked_fill(ids < 0, torch.iinfo(ids.dtype).max).argsort(
        dim=-1, stable=True)
    values = scores.masked_fill(ids < 0, -torch.inf).gather(-1, order)
    rank = values.argsort(dim=-1, descending=True, stable=True)[..., :k]
    return order.gather(-1, rank)


def _completed_hi(b, Bq, s):
    return (b * Bq) // s - 1


def causal_prefix_topp_rescan(ceil_c, cell_end, blk_start, Pk):
    B, H, M = ceil_c.shape
    P = min(Pk, M)
    visM = cell_end[None, :] <= blk_start[:, None]
    key = ceil_c[:, :, None, :].expand(B, H, blk_start.shape[0], M).clone()
    key = key.masked_fill(~visM[None, None], _NEG)
    ids = torch.arange(M, device=ceil_c.device).view(1, 1, 1, M)
    idx = stable_topk_slots(key, ids, P)
    return idx.masked_fill(key.gather(-1, idx) <= _NEG / 2.0, -1)


def causal_prefix_topp_incremental(ceil_c, s, Bq, nb, Pk):
    B, H, M = ceil_c.shape
    dev = ceil_c.device
    P = min(Pk, M)
    idx_all = torch.arange(M, device=dev)

    run_ceil = torch.full((B, H, P), _NEG, device=dev, dtype=ceil_c.dtype)
    run_idx = torch.full((B, H, P), -1, device=dev, dtype=torch.long)
    out = torch.full((B, H, nb, P), -1, device=dev, dtype=torch.long)
    prev_hi = -1
    for b in range(nb):
        hi = min(M - 1, _completed_hi(b, Bq, s))
        if hi > prev_hi:
            lo = prev_hi + 1
            n_new = hi + 1 - lo
            new_ceil = ceil_c[:, :, lo:hi + 1]
            new_idx = idx_all[lo:hi + 1].view(1, 1, n_new).expand(B, H, n_new)
            cat_ceil = torch.cat([run_ceil, new_ceil], dim=-1)
            cat_idx = torch.cat([run_idx, new_idx], dim=-1)
            top = stable_topk_slots(cat_ceil, cat_idx, P)
            run_ceil = cat_ceil.gather(-1, top)
            run_idx = cat_idx.gather(-1, top)
            prev_hi = hi
        out[:, :, b, :] = run_idx
    return out
