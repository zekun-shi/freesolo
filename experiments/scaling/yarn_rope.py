import math

import torch
import torch.nn as nn


def _find_correction_dim(num_rotations: float, dim: int, base: float,
                         original_max_seq_len: int) -> float:
    return (dim * math.log(original_max_seq_len / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _find_correction_range(low_rot: float, high_rot: float, dim: int, base: float,
                          original_max_seq_len: int) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, base, original_max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, original_max_seq_len))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_mask(low: float, high: float, size: int,
                     dtype=torch.float32, device=None) -> torch.Tensor:
    if low == high:
        high += 0.001
    ramp = (torch.arange(size, dtype=dtype, device=device) - low) / (high - low)
    return torch.clamp(ramp, 0.0, 1.0)


def yarn_get_mscale(scale: float) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def compute_yarn_inv_freq(
    dim: int,
    base: float,
    scale: float,
    original_max_seq_len: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    dtype=torch.float32,
    device=None,
) -> torch.Tensor:
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scale * pos_freqs)

    low, high = _find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_seq_len
    )
    extrap_mask = 1.0 - _linear_ramp_mask(low, high, dim // 2, dtype=dtype, device=device)

    inv_freq = (
        inv_freq_interpolation * (1.0 - extrap_mask)
        + inv_freq_extrapolation * extrap_mask
    )
    return inv_freq


class YaRNRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scale: float = 1.0,
        original_max_seq_len: int | None = None,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attn_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale = float(scale)
        self.original_max_seq_len = (
            int(original_max_seq_len) if original_max_seq_len is not None else int(max_seq_len)
        )
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attn_factor = attn_factor

        self.mscale = yarn_get_mscale(self.scale) * attn_factor

        inv_freq = compute_yarn_inv_freq(
            dim=dim,
            base=base,
            scale=self.scale,
            original_max_seq_len=self.original_max_seq_len,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos() * self.mscale, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * self.mscale, persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def build_rope(
    head_dim: int,
    max_seq_len: int,
    use_yarn: bool = False,
    yarn_scale: float = 1.0,
    yarn_orig_max_seq_len: int | None = None,
    yarn_beta_fast: float = 32.0,
    yarn_beta_slow: float = 1.0,
    yarn_attn_factor: float = 1.0,
    rope_base: float = 10000.0,
):
    if not use_yarn:
        yarn_scale = 1.0
    orig = yarn_orig_max_seq_len if yarn_orig_max_seq_len is not None else max_seq_len
    return YaRNRotaryEmbedding(
        dim=head_dim,
        max_seq_len=max_seq_len,
        base=rope_base,
        scale=yarn_scale,
        original_max_seq_len=orig,
        beta_fast=yarn_beta_fast,
        beta_slow=yarn_beta_slow,
        attn_factor=yarn_attn_factor,
    )
