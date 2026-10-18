from __future__ import annotations

import contextlib
import glob
import json
import os
import sys
import time
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.scaling.model import FreeSoloLM, MODEL_CONFIGS, check_params_loaded

__all__ = [
    "DEFAULT_MAX_SEQ_LEN",
    "load_model",
    "extend_rope_cache",
    "trunk_hidden_states",
    "logits_at_positions",
    "last_token_logits",
    "forward_logits",
    "greedy_generate",
    "score_continuation",
    "run_length_sweep",
    "DEFAULT_SWEEP_LENGTHS",
]

DEFAULT_MAX_SEQ_LEN = 131072
DEFAULT_SWEEP_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]


def _resolve_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _resolve_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return {
        "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "f16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }[str(dtype).lower()]


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _as_2d_long(ids, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(ids):
        ids = torch.as_tensor(ids, dtype=torch.long)
    ids = ids.to(device=device, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    elif ids.dim() != 2:
        raise ValueError(f"input_ids must be 1-D or 2-D, got shape {tuple(ids.shape)}")
    return ids


_NON_MODEL_FILES = (
    "optimizer", "scheduler", "scaler", "random_states", "sampler",
)


def _looks_like_model_file(path: str) -> bool:
    base = os.path.basename(path).lower()
    return not any(tag in base for tag in _NON_MODEL_FILES)


def _load_safetensors_file(path: str, device: torch.device) -> dict:
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read {path}: install `safetensors` to load this checkpoint."
        ) from exc
    return load_file(path, device=str(device))


def _load_sharded(index_path: str, loader: Callable[[str], dict]) -> dict:
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    sd: dict = {}
    base_dir = os.path.dirname(index_path)
    loaded_shards: dict[str, dict] = {}
    for key, shard_name in weight_map.items():
        if shard_name not in loaded_shards:
            loaded_shards[shard_name] = loader(os.path.join(base_dir, shard_name))
        sd[key] = loaded_shards[shard_name][key]
    return sd


def _load_raw_state_dict(path: str, device: torch.device) -> dict:
    if os.path.isdir(path):
        for idx_name, loader in (
            ("model.safetensors.index.json",
             lambda p: _load_safetensors_file(p, device)),
            ("pytorch_model.bin.index.json",
             lambda p: torch.load(p, map_location=device, weights_only=False)),
        ):
            idx = os.path.join(path, idx_name)
            if os.path.exists(idx):
                return _load_sharded(idx, loader)

        for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt"):
            cand = os.path.join(path, name)
            if os.path.exists(cand):
                if cand.endswith(".safetensors"):
                    return _load_safetensors_file(cand, device)
                obj = torch.load(cand, map_location=device, weights_only=False)
                return _unwrap_state_dict(obj)

        for pattern in ("*.safetensors", "*.bin", "*.pt"):
            hits = sorted(
                p for p in glob.glob(os.path.join(path, pattern))
                if _looks_like_model_file(p)
            )
            if not hits:
                continue
            if pattern == "*.safetensors":
                sd: dict = {}
                for p in hits:
                    sd.update(_load_safetensors_file(p, device))
                return sd
            sd = {}
            for p in hits:
                sd.update(_unwrap_state_dict(
                    torch.load(p, map_location=device, weights_only=False)))
            return sd

        raise FileNotFoundError(
            f"No model weight file found in accelerate dir: {path}\n"
            f"  contents: {sorted(os.listdir(path))}"
        )

    if path.endswith(".safetensors"):
        return _load_safetensors_file(path, device)
    obj = torch.load(path, map_location=device, weights_only=False)
    return _unwrap_state_dict(obj)


def _unwrap_state_dict(obj) -> dict:
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _clean_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k.replace("_orig_mod.", "")
        if nk.startswith("module."):
            nk = nk[len("module."):]
        out[nk] = v
    return out


def load_model(
    checkpoint: str | None,
    config: str = "340m",
    attn_type: str = "fsattn",
    device="cuda",
    *,
    max_seq_len: int | None = None,
    eval_mode: bool = True,
    strict: bool = False,
    verbose: bool = True,
    n_layers: int | None = None,
    **attn_kwargs,
) -> FreeSoloLM:
    device = _resolve_device(device)
    if config not in MODEL_CONFIGS:
        raise KeyError(f"Unknown config '{config}'. Choices: {list(MODEL_CONFIGS)}")

    cfg = MODEL_CONFIGS[config].copy()
    if max_seq_len is not None:
        cfg["max_seq_len"] = max_seq_len
    if n_layers is not None:
        cfg["n_layers"] = n_layers

    model = FreeSoloLM(attn_type=attn_type, **cfg, **attn_kwargs).to(device)

    if checkpoint is not None:
        raw = _load_raw_state_dict(checkpoint, device)
        meta = {}
        if not os.path.isdir(checkpoint) and checkpoint.endswith((".pt", ".bin")):
            try:
                obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if isinstance(obj, dict):
                    meta = {k: obj[k] for k in ("step", "val_loss", "epoch")
                            if k in obj}
            except Exception:
                meta = {}

        sd = _clean_keys(_unwrap_state_dict(raw))
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        check_params_loaded(model, missing, where=f"load_model({checkpoint})")
        if verbose:
            n_params = sum(p.numel() for p in model.parameters())
            extra = f", meta={meta}" if meta else ""
            print(f"[load_model] {config}/{attn_type}: {n_params:,} params "
                  f"from {checkpoint}{extra}")
            if missing:
                print(f"[load_model]   missing keys ({len(missing)}): "
                      f"{missing[:6]}{' ...' if len(missing) > 6 else ''}")
            if unexpected:
                print(f"[load_model]   unexpected keys ({len(unexpected)}): "
                      f"{unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")

    if eval_mode:
        model.eval()
    return model


def extend_rope_cache(model: torch.nn.Module, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
    shared = {}
    for m in model.modules():
        if hasattr(m, "_build_cache") and hasattr(m, "cos_cached"):
            if m.cos_cached.shape[0] < max_seq_len:
                m._build_cache(max_seq_len)
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
                m.cos_cached, m.sin_cached = shared[key]
            else:
                shared[key] = (m.cos_cached, m.sin_cached)
    return model


@torch.no_grad()
def trunk_hidden_states(
    model: FreeSoloLM,
    input_ids: torch.Tensor,
    *,
    dtype="bfloat16",
) -> torch.Tensor:
    device = next(model.parameters()).device
    dt = _resolve_dtype(dtype)
    idx = _as_2d_long(input_ids, device)
    with _autocast_ctx(device, dt):
        x = model.drop(model.tok_emb(idx))
        for block in model.blocks:
            x = block(x)
        x = model.ln_final(x)
    return x


def _lm_head_fp32(model: FreeSoloLM, hidden: torch.Tensor, lm_chunk_size: int) -> torch.Tensor:
    w = model.lm_head.weight
    W = w if (torch.is_autocast_enabled() and w.dtype == torch.get_autocast_dtype("cuda")) else w.float()
    P = hidden.shape[1]
    if P <= lm_chunk_size:
        return F.linear(hidden.float(), W)
    outs = []
    for s in range(0, P, lm_chunk_size):
        outs.append(F.linear(hidden[:, s:s + lm_chunk_size, :].float(), W))
    return torch.cat(outs, dim=1)


@torch.no_grad()
def logits_at_positions(
    model: FreeSoloLM,
    input_ids: torch.Tensor,
    positions: Sequence[int] | torch.Tensor,
    *,
    dtype="bfloat16",
    lm_chunk_size: int = 4096,
) -> torch.Tensor:
    device = next(model.parameters()).device
    idx = _as_2d_long(input_ids, device)
    N = idx.shape[1]
    pos = torch.as_tensor(list(positions) if not torch.is_tensor(positions)
                          else positions, dtype=torch.long, device=device)
    pos = torch.where(pos < 0, pos + N, pos)
    if (pos < 0).any() or (pos >= N).any():
        raise IndexError(f"position(s) out of range for seq_len={N}: {pos.tolist()}")

    hidden = trunk_hidden_states(model, idx, dtype=dtype)
    sel = hidden.index_select(1, pos)
    return _lm_head_fp32(model, sel, lm_chunk_size)


@torch.no_grad()
def last_token_logits(
    model: FreeSoloLM,
    input_ids: torch.Tensor,
    *,
    dtype="bfloat16",
) -> torch.Tensor:
    out = logits_at_positions(model, input_ids, [-1], dtype=dtype)
    return out[:, 0, :]


@torch.no_grad()
def forward_logits(
    model: FreeSoloLM,
    input_ids: torch.Tensor,
    positions: Sequence[int] | torch.Tensor | None = None,
    *,
    dtype="bfloat16",
    lm_chunk_size: int = 4096,
) -> torch.Tensor:
    if positions is not None:
        return logits_at_positions(
            model, input_ids, positions, dtype=dtype, lm_chunk_size=lm_chunk_size)
    hidden = trunk_hidden_states(model, input_ids, dtype=dtype)
    return _lm_head_fp32(model, hidden, lm_chunk_size)


@torch.no_grad()
def greedy_generate(
    model: FreeSoloLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    device="cuda",
    *,
    dtype="bfloat16",
    eos_token_id: int | None = None,
) -> torch.Tensor:
    device = _resolve_device(device)
    model.eval()
    seq = _as_2d_long(prompt_ids, device)
    B = seq.shape[0]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits = last_token_logits(model, seq, dtype=dtype)
        next_tok = logits.argmax(dim=-1)
        if eos_token_id is not None:
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_token_id),
                                   next_tok)
            finished = finished | (next_tok == eos_token_id)
        seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break
    return seq


@torch.no_grad()
def score_continuation(
    model: FreeSoloLM,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    *,
    dtype="bfloat16",
    lm_chunk_size: int = 4096,
) -> dict:
    device = next(model.parameters()).device
    prompt = _as_2d_long(prompt_ids, device)
    answer = _as_2d_long(answer_ids, device)
    if prompt.shape[0] != 1 or answer.shape[0] != 1:
        raise ValueError("score_continuation handles a single example "
                         "(prompt/answer must be 1-D or batch size 1).")
    P = prompt.shape[1]
    A = answer.shape[1]
    if A == 0:
        raise ValueError("answer_ids is empty.")

    seq = torch.cat([prompt, answer], dim=1)
    positions = torch.arange(P - 1, P + A - 1, device=device)
    logits = logits_at_positions(
        model, seq, positions, dtype=dtype, lm_chunk_size=lm_chunk_size)
    logits = logits[0]

    log_probs = F.log_softmax(logits, dim=-1)
    tgt = answer[0]
    tok_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    preds = logits.argmax(dim=-1)
    correct = (preds == tgt)

    total_lp = float(tok_lp.sum().item())
    avg_lp = total_lp / A
    return {
        "token_logprobs": tok_lp.tolist(),
        "total_logprob": total_lp,
        "avg_logprob": avg_lp,
        "perplexity": float(torch.exp(torch.tensor(-avg_lp)).item()),
        "predictions": preds.tolist(),
        "correct": correct.tolist(),
        "num_correct": int(correct.sum().item()),
        "num_tokens": int(A),
        "accuracy": float(correct.float().mean().item()),
    }


@torch.no_grad()
def run_length_sweep(
    model: FreeSoloLM,
    build_example_fn: Callable[[int, torch.device], object],
    lengths: Iterable[int] = tuple(DEFAULT_SWEEP_LENGTHS),
    *,
    dtype="bfloat16",
    device=None,
    score_fn: Callable[[FreeSoloLM, object, str], dict] | None = None,
    grow_rope: bool = True,
    verbose: bool = True,
) -> list[dict]:
    if device is None:
        device = next(model.parameters()).device
    device = _resolve_device(device)
    lengths = list(lengths)
    model.eval()

    if grow_rope and lengths:
        extend_rope_cache(model, max(lengths))

    def _default_score(m, example, dt):
        ids = example["input_ids"] if isinstance(example, dict) else example
        logits = last_token_logits(m, ids, dtype=dt)
        ids2d = _as_2d_long(ids, device)
        return {
            "seq_len_actual": int(ids2d.shape[1]),
            "argmax_last": int(logits[0].argmax().item()),
        }

    scorer = score_fn or _default_score
    records: list[dict] = []

    for L in lengths:
        rec: dict = {"length": L, "status": "ok", "seconds": None,
                     "peak_mem_gb": None}
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        try:
            example = build_example_fn(L, device)
            metrics = scorer(model, example, dtype)
            rec.update(metrics or {})
            rec["seconds"] = time.time() - t0
            if device.type == "cuda":
                rec["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            if verbose:
                mem = f"{rec['peak_mem_gb']:.2f} GB" if rec["peak_mem_gb"] else "n/a"
                print(f"[sweep] L={L:>7}: ok  ({rec['seconds']:.1f}s, peak {mem})")
        except BaseException as exc:
            rec["seconds"] = time.time() - t0
            if _is_oom(exc):
                rec["status"] = "oom"
                rec["error"] = "CUDA out of memory"
                if verbose:
                    print(f"[sweep] L={L:>7}: OOM -- skipping")
            else:
                rec["status"] = "error"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                if verbose:
                    print(f"[sweep] L={L:>7}: ERROR -- {rec['error']}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        records.append(rec)

    return records


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={dev}")

    model = load_model(None, config="debug", attn_type="fsattn", device=dev,
                       window_size=32, long_range_variant="cell_topk", cell_stride=8, cell_topk=2)
    extend_rope_cache(model, 4096)

    prompt = torch.randint(0, 50257, (24,))
    answer = torch.randint(0, 50257, (8,))

    sc = score_continuation(model, prompt, answer, dtype="float32")
    print(f"[smoke] score_continuation: avg_logprob={sc['avg_logprob']:.3f} "
          f"acc={sc['accuracy']:.3f} ntok={sc['num_tokens']}")

    gen = greedy_generate(model, prompt, max_new_tokens=5, device=dev,
                          dtype="float32")
    print(f"[smoke] greedy_generate -> shape {tuple(gen.shape)}")

    lt = last_token_logits(model, prompt, dtype="float32")
    print(f"[smoke] last_token_logits -> shape {tuple(lt.shape)}")

    def build(L, device):
        return torch.randint(0, 50257, (1, L), device=device)

    recs = run_length_sweep(model, build, lengths=[64, 128, 256],
                            dtype="float32", device=dev)
    print(f"[smoke] sweep statuses: {[(r['length'], r['status']) for r in recs]}")
    print("[smoke] OK")
