"""KV caches for the matched global NoPE and SWAN baselines."""

import torch


class DenseKVCache:
    def __init__(self, max_tokens, window_size=None):
        if max_tokens < 1:
            raise ValueError("Cache capacity must be positive")
        self.max_tokens = int(max_tokens)
        self.window_size = window_size
        self.seqlen = 0
        self.k = self.v = None

    def update(self, k, v):
        count = k.shape[2]
        if self.seqlen and count != 1:
            raise ValueError("After prefill, the baseline cache accepts one token per update")
        if self.seqlen + count > self.max_tokens:
            raise ValueError("Baseline cache capacity exceeded")
        if self.window_size is not None:
            if not self.seqlen:
                all_k, all_v = k, v
            else:
                all_k = torch.cat((self.k, k), dim=2)
                all_v = torch.cat((self.v, v), dim=2)
                all_k, all_v = all_k[:, :, -self.window_size - 1:], all_v[:, :, -self.window_size - 1:]
            # Return the full prefill to score every query, but retain only the window.
            self.k = all_k[:, :, -self.window_size - 1:].clone()
            self.v = all_v[:, :, -self.window_size - 1:].clone()
        else:
            if self.k is None:
                shape = (*k.shape[:2], self.max_tokens, k.shape[-1])
                self.k, self.v = k.new_empty(shape), v.new_empty(shape)
            self.k[:, :, self.seqlen:self.seqlen + count].copy_(k)
            self.v[:, :, self.seqlen:self.seqlen + count].copy_(v)
            all_k, all_v = self.k[:, :, :self.seqlen + count], self.v[:, :, :self.seqlen + count]
        self.seqlen += count
        return all_k, all_v


def make_baseline_caches(model, max_tokens):
    caches = []
    for block in model.blocks:
        if block.attn.attn_type not in ("standard", "swan"):
            raise ValueError("Baseline caches require standard or SWAN attention throughout")
        caches.append(DenseKVCache(max_tokens, block.attn.standard.window_size))
    return caches


@torch.no_grad()
def baseline_cached_generate(model, prompt_ids, max_new_tokens, device="cuda", *,
                             dtype="bfloat16", eos_token_id=None):
    """Greedy baseline generation with the same head, precision and EOS rules as FS."""
    from .longctx_infer import _as_2d_long, _autocast_ctx, _lm_head_fp32, _resolve_device

    device = _resolve_device(device)
    model.eval()
    seq = _as_2d_long(prompt_ids, device)
    if seq.shape[1] == 0 or max_new_tokens < 0:
        raise ValueError("Generation needs a nonempty prompt and nonnegative token budget")
    if max_new_tokens == 0:
        return seq
    dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    caches = make_baseline_caches(model, seq.shape[1] + max_new_tokens)
    finished = torch.zeros(seq.shape[0], dtype=torch.bool, device=device)
    generated, step_in = [], seq
    with _autocast_ctx(device, dt):
        for _ in range(max_new_tokens):
            hidden = model.drop(model.tok_emb(step_in))
            for i, block in enumerate(model.blocks):
                hidden = block(hidden, cache=caches[i])
            # Avoid a prompt-length by vocabulary allocation at long contexts.
            hidden = model.ln_final(hidden[:, -1:, :])
            logits = _lm_head_fp32(model, hidden, 4096)[:, 0, :]
            nxt = logits.argmax(dim=-1)
            if eos_token_id is not None:
                nxt = torch.where(finished, torch.full_like(nxt, eos_token_id), nxt)
                finished |= nxt == eos_token_id
            generated.append(nxt)
            step_in = nxt.unsqueeze(1)
            if eos_token_id is not None and bool(finished.all()):
                break
    return torch.cat([seq, torch.stack(generated, dim=1)], dim=1)
