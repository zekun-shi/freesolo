"""Teacher-forced completion likelihood through the existing bounded FS-L cache."""
import torch
import torch.nn.functional as F

from src.fsattn.linear_cache import LinearFSCache


@torch.no_grad()
def completion_logprobs(model, full_ids, completion_tokens, *, chunk_size=2048,
                       dtype=torch.bfloat16):
    if chunk_size < 1 or completion_tokens < 1:
        raise ValueError("Positive chunk size and completion length are required")
    device = model.tok_emb.weight.device
    seq = torch.as_tensor(full_ids, dtype=torch.long, device=device).reshape(1, -1)
    if seq.shape[1] < 2:
        raise ValueError("Likelihood needs a predictor token and a target token")
    caches = []
    for block in model.blocks:
        if block.attn.attn_type != 'fsattn':
            raise ValueError("Bounded likelihood requires FS-L in every layer")
        attn = block.attn.fsattn
        if (getattr(attn, 'fsb_enabled', False) or
                not attn._prune_cfg('cell_prune_kernel', 'FS_PRUNE_KERNEL', 0)):
            raise ValueError("Bounded likelihood requires native FS-L pruning")
        caches.append(LinearFSCache(attn))
    n = min(completion_tokens, seq.shape[1] - 1)
    first = seq.shape[1] - 1 - n
    scores = []
    with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        # Independent caches per candidate; prefix tokens are never scored.
        for start in range(0, first, chunk_size):
            model(seq[:, start:min(start + chunk_size, first)], cache=caches)
        for start in range(first, seq.shape[1] - 1, chunk_size):
            end = min(start + chunk_size, seq.shape[1] - 1)
            logits = model(seq[:, start:end], cache=caches)
            logp = F.log_softmax(logits.float(), dim=-1)
            scores.append(logp.gather(-1, seq[:, start + 1:end + 1, None]).squeeze(-1))
    result = torch.cat(scores, dim=1)[0]
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite bounded completion likelihood")
    return result
