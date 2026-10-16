import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.fsattn.attention import FSAttn
from src.utils import trunk_triton as _tt


def check_params_loaded(model: nn.Module, missing, where: str = "checkpoint") -> None:
    """Raise if a non-strict load left any trainable parameter at its random init.

    Missing buffers are tolerated; a parameter tied to a tensor that was loaded (lm_head.weight
    <-> tok_emb.weight) counts as loaded. Checkpoints must use the FreeSolo parameter names
    (the released safetensors do).
    """
    missing = set(missing)
    if not missing:
        return
    loaded_ptrs = {t.data_ptr() for k, t in model.state_dict(keep_vars=True).items()
                   if k not in missing}
    params = dict(model.named_parameters(remove_duplicate=False))
    bad = sorted(k for k in missing
                 if k in params and params[k].data_ptr() not in loaded_ptrs)
    if bad:
        raise RuntimeError(
            f"{where}: {len(bad)} model parameters were not found in the checkpoint and would "
            f"stay randomly initialised (e.g. {bad[:4]}). The checkpoint's key names do not "
            f"match FreeSoloLM; use the released FreeSolo safetensors.")


MODEL_CONFIGS = {
    "150m": dict(
        d_model=896, n_layers=12, n_heads=14, d_ff=2560,
        max_seq_len=4096, vocab_size=32000,
    ),
    "debug": dict(
        d_model=256, n_layers=4, n_heads=4, d_ff=688,
        max_seq_len=1024, vocab_size=50257,
    ),
    "small": dict(
        d_model=768, n_layers=12, n_heads=12, d_ff=2048,
        max_seq_len=2048, vocab_size=50257,
    ),
    "proxy256": dict(
        d_model=256, n_layers=24, n_heads=16, d_ff=704,
        max_seq_len=2048, vocab_size=50257,
    ),
    "proxy512": dict(
        d_model=512, n_layers=24, n_heads=16, d_ff=1408,
        max_seq_len=2048, vocab_size=50257,
    ),
    "340m": dict(
        d_model=1024, n_layers=24, n_heads=16, d_ff=2816,
        max_seq_len=2048, vocab_size=50257,
    ),
    "800m": dict(
        d_model=1536, n_layers=24, n_heads=16, d_ff=4224,
        max_seq_len=2048, vocab_size=50257,
    ),
    "1.3b": dict(
        d_model=2048, n_layers=24, n_heads=16, d_ff=5504,
        max_seq_len=2048, vocab_size=50257,
    ),
    # the trained FreeSolo 1.3B (fs13b_fineweb_56b_20261007): Llama-2 vocabulary, 4096-token training context
    "1.3b_fs": dict(
        d_model=2048, n_layers=24, n_heads=16, d_ff=5632,
        max_seq_len=4096, vocab_size=32000,
    ),
    "3b": dict(
        d_model=2560, n_layers=36, n_heads=20, d_ff=6912,
        max_seq_len=2048, vocab_size=50257,
    ),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if not torch.is_grad_enabled() and x.is_cuda and x.dim() == 3 and x.shape[1] > 8192:
            N = x.shape[1]
            c0 = self.forward(x[:, :8192])
            out = torch.empty((x.shape[0], N, x.shape[2]), device=x.device, dtype=c0.dtype)
            out[:, :8192].copy_(c0)
            del c0
            for s in range(8192, N, 8192):
                e = min(s + 8192, N)
                out[:, s:e].copy_(self.forward(x[:, s:e]))
            return out
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _ensure_gu_cache(self):
        wg, wu = self.w_gate.weight, self.w_up.weight
        d_ff = wg.shape[0]
        key = (wg.data_ptr(), wg._version, wu.data_ptr(), wu._version, wg.dtype)
        c = getattr(self, "_w_gu_cache", None)
        if c is None or c[0] != key:
            gu_w = torch.cat([wg.data, wu.data], dim=0)
            self.w_gate.weight.data = gu_w[:d_ff]
            self.w_up.weight.data = gu_w[d_ff:]
            wg, wu = self.w_gate.weight, self.w_up.weight
            key = (wg.data_ptr(), wg._version, wu.data_ptr(), wu._version, wg.dtype)
            c = (key, gu_w)
            self._w_gu_cache = c
        return c[1], d_ff

    def forward(self, x):
        if (getattr(self, "_use_gu_cache", False) and x.is_cuda and _tt._rows(x) == 1
                and not torch.is_grad_enabled() and self.w_gate.bias is None and self.w_up.bias is None
                and os.environ.get("FSL_TRUNK_FUSED", "1") != "0"):
            gu_w, d_ff = self._ensure_gu_cache()
            gu = F.linear(x, gu_w)
            g, u = gu[..., :d_ff], gu[..., d_ff:]
            h = _tt.silu_mul(g, u) if _tt.silu_mul_ok(g, u) else F.silu(g) * u
            return self.dropout(self.w_down(h))
        g, u = self.w_gate(x), self.w_up(x)
        if not torch.is_grad_enabled():
            del x
            # no-grad CUDA inference: silu and the product in one pass, bit for bit (trunk_triton.silu_mul)
            h = _tt.silu_mul(g, u, out=g) if _tt.silu_mul_ok(g, u) else F.silu(g) * u
            del g, u
            out = self.w_down(h)
            del h
            return self.dropout(out)
        h = _tt.silu_mul(g, u) if _tt.silu_mul_ok(g, u) else F.silu(g) * u
        return self.dropout(self.w_down(h))


class AttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        attn_type: str = "fsattn",
        causal: bool = True,
        layer_idx: int = 0,
        **attn_kwargs,
    ):
        super().__init__()
        if attn_type not in ("fsattn", "standard"):
            raise ValueError(f"unknown attn_type {attn_type!r}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = attn_kwargs.pop("use_rope", True)
        self.attn_type = attn_type
        if attn_type == "standard":
            from experiments.scaling.dense_attention import DenseAttention
            self.standard = DenseAttention(d_model, n_heads, max_seq_len,
                                           causal=causal, use_rope=self.use_rope)
            return
        self.fsattn = FSAttn(
            d_model, n_heads, causal=causal,
            bias=False,
            use_rope=self.use_rope,
            max_seq_len=max_seq_len,
            **attn_kwargs,
        )

    def forward(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        if self.attn_type == "standard":
            return self.standard(x, cache=cache)
        return self.fsattn(x, cache=cache)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        attn_type: str,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        layer_idx: int = 0,
        **attn_kwargs,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = AttentionWithRoPE(
            d_model, n_heads, max_seq_len,
            attn_type=attn_type, causal=True, layer_idx=layer_idx, **attn_kwargs,
        )
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.gradient_checkpointing = gradient_checkpointing

    def _forward(self, x, cache=None):
        if _tt.rms_norm_ok(x, self.ln1.weight):
            # no-grad CUDA inference under autocast: each norm's output goes only to autocast linears (W_q / W_k /
            # W_v, w_gate / w_up), which would each cast it; write it once in the autocast dtype instead, from one
            # fused pass (bit for bit: trunk_triton.rms_norm_cast)
            dt = _tt.autocast_dtype()
            xn1 = _tt.rms_norm_cast(x, self.ln1.weight, self.ln1.eps, dt)
            xn1._can_release = True
            attn_out = self.attn(xn1, cache=cache)
            del xn1
            N = x.shape[1] if x.dim() == 3 else 0
            ffn_env = int(os.environ.get("FSL_FFN_CHUNK", "0"))
            ffn_chunk = ffn_env if ffn_env > 0 else (max(16384, N // 4) if N > 16384 else 0)
            if N > ffn_chunk > 0:
                x.add_(attn_out)
                del attn_out
                for s in range(0, N, ffn_chunk):
                    xc = x[:, s:min(s + ffn_chunk, N)]
                    xc.add_(self.ffn(_tt.rms_norm_cast(xc, self.ln2.weight, self.ln2.eps, dt)))
                return x
            x = x + attn_out
            del attn_out
            x = x + self.ffn(_tt.rms_norm_cast(x, self.ln2.weight, self.ln2.eps, dt))
            return x
        x = x + self.attn(self.ln1(x), cache=cache)
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x, cache=None):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x, cache=cache)


class FreeSoloLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        d_ff: int = 2816,
        max_seq_len: int = 2048,
        attn_type: str = "fsattn",
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        **attn_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        _attn_pattern = attn_kwargs.pop("attn_pattern", None)
        self.attn_pattern = _attn_pattern
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, max_seq_len,
                (_attn_pattern[i % len(_attn_pattern)] if _attn_pattern else attn_type),
                dropout, gradient_checkpointing,
                layer_idx=i,
                **attn_kwargs,
            )
            for i in range(n_layers)
        ])

        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight

        self._mup_output_mult = 1.0

        self._fsb_policy_layers = [m for m in self.modules() if getattr(m, "fsb_policy", None) is not None]
        self._fsb_route_layers = [m for m in self.modules() if getattr(m, "fsb_route_adapter", "none") != "none"]

        self._init_weights()
        self._dedup_rope_buffers()

    def _dedup_rope_buffers(self):
        shared = {}
        for m in self.modules():
            if hasattr(m, "_build_cache") and hasattr(m, "cos_cached") and m.cos_cached is not None:
                key = (
                    getattr(m, "dim", None),
                    getattr(m, "base", None),
                    getattr(m, "scale", None),
                    getattr(m, "original_max_seq_len", None),
                    getattr(m, "beta_fast", None),
                    getattr(m, "beta_slow", None),
                    getattr(m, "attn_factor", None),
                    m.cos_cached.device,
                    m.cos_cached.dtype,
                    m.cos_cached.shape,
                )
                if key in shared:
                    m.cos_cached, m.sin_cached, m.inv_freq = shared[key]
                else:
                    shared[key] = (m.cos_cached, m.sin_cached, getattr(m, "inv_freq", None))

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse=recurse)
        self._dedup_rope_buffers()
        return out

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "tok_emb" not in name and "long_range_scale" not in name \
                    and "fsb_route_" not in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, cache=None) -> torch.Tensor:
        x = self.drop(self.tok_emb(idx))

        for i, block in enumerate(self.blocks):
            x = block(x, cache=None if cache is None else cache[i])

        x = self.ln_final(x)
        logits = self.lm_head(x) * self._mup_output_mult
        if self.training and self._fsb_policy_layers:
            tie = None
            for m in self._fsb_policy_layers:
                st = m._fsb_stats
                if st and "logp" in st:
                    t = st["logp"].sum() + st["value"].sum()
                    tie = t if tie is None else tie + t
            if tie is not None:
                logits = logits + tie.to(logits.dtype) * 0.0
        if self.training and self._fsb_route_layers and torch.is_grad_enabled():
            tie = None
            for m in self._fsb_route_layers:
                rd = getattr(m, "_fsb_rd", None)
                if rd is not None and rd["loss"].requires_grad:
                    tie = rd["loss"] if tie is None else tie + rd["loss"]
            if tie is not None:
                logits = logits + tie.to(logits.dtype) * 0.0
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config_name: str, attn_type: str = "fsattn", **kwargs):
        cfg = MODEL_CONFIGS[config_name].copy()
        cfg.update(kwargs)
        return cls(attn_type=attn_type, **cfg)
