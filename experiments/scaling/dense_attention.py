"""Dense attention and the published SWAN local/global attention primitives."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class DenseAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, causal=True, use_rope=True,
                 window_size=None, rope_theta=10000.0, infer_scale_base=0.0):
        super().__init__()
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("Dense RoPE attention needs an even, integral head dimension")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.use_rope = use_rope
        self.window_size = window_size
        self.rope_theta = float(rope_theta)
        self.infer_scale_base = float(infer_scale_base)
        if window_size is not None and (window_size < 1 or not causal):
            raise ValueError("Sliding-window attention requires a positive causal window")
        if self.rope_theta <= 1 or not math.isfinite(self.rope_theta):
            raise ValueError("RoPE base must be finite and greater than one")
        if self.infer_scale_base != 0 and (
                self.infer_scale_base <= 1 or not math.isfinite(self.infer_scale_base)):
            raise ValueError("Inference scaling base must be zero (off) or greater than one")
        if self.infer_scale_base and (use_rope or window_size is not None):
            raise ValueError("SWAN inference scaling applies only to global NoPE layers")
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        batch, length, width = x.shape
        q, k, v = [projection(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
                   for projection in (self.W_q, self.W_k, self.W_v)]
        offset = 0 if cache is None else cache.seqlen
        if self.use_rope:
            # Keep frequencies in fp32 even when weights are stored in bf16.
            with torch.autocast(device_type=x.device.type, enabled=False):
                frequency = 1.0 / (self.rope_theta ** (
                    torch.arange(0, self.head_dim, 2, device=x.device).float() / self.head_dim))
                angles = torch.outer(torch.arange(offset, offset + length, device=x.device).float(), frequency)
                cos, sin = angles.cos()[None, None], angles.sin()[None, None]
            def rotate(t):
                first, second = t.chunk(2, dim=-1)
                return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(t.dtype)
            q, k = rotate(q), rotate(k)
        if not self.training and self.infer_scale_base:
            # SWAN section 2.2: multiply attention scores by log_a(a+n),
            # per absolute query position. Training uses the ordinary scale.
            positions = torch.arange(offset, offset + length, device=x.device, dtype=torch.float32)
            factor = torch.log(positions + self.infer_scale_base) / math.log(self.infer_scale_base)
            q = (q.float() * factor[None, None, :, None]).to(q.dtype)
        if cache is not None:
            if offset and length != 1:
                raise ValueError("Dense cached attention supports single-token decode after prefill")
            k, v = cache.update(k, v)
        if self.window_size is not None and length > 1:
            from src.utils.local_attention import local_attention
            out = local_attention(q, k, v, window_size=self.window_size,
                                  scale=self.head_dim ** -0.5, causal=True)
        else:
            if self.window_size is not None:
                k, v = k[:, :, -self.window_size - 1:], v[:, :, -self.window_size - 1:]
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal and length > 1)
        return self.W_o(out.transpose(1, 2).contiguous().view(batch, length, width))
