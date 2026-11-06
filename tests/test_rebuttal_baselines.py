import copy
import math

import pytest
import torch

from experiments.scaling.baseline_cache import DenseKVCache, make_baseline_caches, baseline_cached_generate
from experiments.scaling.dense_attention import DenseAttention
from experiments.scaling.model import FreeSoloLM


def reference(attn, x):
    b, n, _ = x.shape
    h, d = attn.n_heads, attn.head_dim
    q, k, v = [layer(x).reshape(b, n, h, d).transpose(1, 2)
               for layer in (attn.W_q, attn.W_k, attn.W_v)]
    if attn.use_rope:
        angle = torch.outer(torch.arange(n, device=x.device).float(),
                            attn.rope_theta ** (-torch.arange(d // 2, device=x.device).float() * 2 / d))
        phase = torch.polar(torch.ones_like(angle), angle)
        def rotate(t):
            z = torch.complex(t[..., :d // 2].float(), t[..., d // 2:].float()) * phase
            return torch.cat((z.real, z.imag), -1).to(t.dtype)
        q, k = rotate(q), rotate(k)
    score = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(d)
    if not attn.training and attn.infer_scale_base:
        factors = [math.log(attn.infer_scale_base + i, attn.infer_scale_base) for i in range(n)]
        score = score * torch.tensor(factors, device=x.device)[None, None, :, None]
    dist = torch.arange(n, device=x.device)[:, None] - torch.arange(n, device=x.device)[None, :]
    allowed = dist >= 0
    if attn.window_size is not None:
        allowed &= dist <= attn.window_size
    out = score.masked_fill(~allowed, -torch.inf).softmax(-1) @ v.float()
    return attn.W_o(out.transpose(1, 2).reshape(b, n, h * d).to(x.dtype))


@pytest.mark.parametrize("kind", ["nope", "local", "scaled_nope"])
@pytest.mark.parametrize("training", [True, False])
def test_attention_and_gradients_against_explicit_mask(kind, training):
    torch.manual_seed(31)
    kwargs = {"use_rope": kind == "local", "window_size": 4 if kind == "local" else None,
              "rope_theta": 1000000.0, "infer_scale_base": 16 if kind == "scaled_nope" else 0}
    actual = DenseAttention(32, 4, 64, **kwargs).train(training)
    expected = copy.deepcopy(actual)
    x = torch.randn(2, 17, 32, requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    out, ref = actual(x), reference(expected, z)
    torch.testing.assert_close(out, ref, atol=3e-7, rtol=2e-5)
    target = torch.randn_like(out)
    (out * target).sum().backward()
    (ref * target).sum().backward()
    torch.testing.assert_close(x.grad, z.grad, atol=8e-7, rtol=2e-5)
    for p, r in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(p.grad, r.grad, atol=3e-6, rtol=3e-5)


def small_model(kind):
    return FreeSoloLM(vocab_size=64, d_model=32, n_layers=8, n_heads=4, d_ff=64,
                      max_seq_len=64, attn_type=kind, window_size=4,
                      use_rope=False, swan_scale_base=16)


def test_swan_uses_published_global_start_pattern_and_theta():
    model = small_model("swan")
    assert [i for i, b in enumerate(model.blocks) if not b.attn.standard.use_rope] == [0, 4]
    for i, b in enumerate(model.blocks):
        a = b.attn.standard
        assert a.window_size == (None if i % 4 == 0 else 4)
        assert a.infer_scale_base == (16 if i % 4 == 0 else 0)
        assert a.rope_theta == 1000000.0


@pytest.mark.parametrize("kind", ["standard", "swan"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_logits_match_full_prefix_and_preserve_causality(kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    model = small_model(kind).eval().to(device)
    tokens = torch.randint(0, 64, (2, 23), device=device)
    caches = make_baseline_caches(model, tokens.shape[1])
    with torch.no_grad():
        whole = model(tokens)
        changed = tokens.clone()
        changed[:, 9:] = (changed[:, 9:] + 1) % 64
        torch.testing.assert_close(whole[:, :9], model(changed)[:, :9], atol=1e-6, rtol=1e-5)
        pieces = [model(tokens[:, :7], cache=caches)]
        for i in range(7, tokens.shape[1]):
            pieces.append(model(tokens[:, i:i + 1], cache=caches))
        cached = torch.cat(pieces, 1)
    torch.testing.assert_close(cached, whole, atol=2e-6, rtol=2e-5)
    for cache in caches:
        assert cache.seqlen == 23
        if cache.window_size is not None:
            assert cache.k.shape[2] <= cache.window_size + 1
    with pytest.raises(ValueError, match="capacity"):
        caches[0].update(torch.zeros(2, 4, 1, 8, device=device),
                         torch.zeros(2, 4, 1, 8, device=device))


def test_inference_scaling_does_not_change_training():
    torch.manual_seed(4)
    a = DenseAttention(32, 4, 64, use_rope=False, infer_scale_base=16)
    b = copy.deepcopy(a)
    b.infer_scale_base = 0
    x = torch.randn(1, 29, 32)
    torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)
    a.eval(); b.eval()
    assert not torch.allclose(a(x)[:, 1:], b(x)[:, 1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ["standard", "swan"])
def test_bf16_cached_decode_matches_full_prefix(kind):
    torch.manual_seed(42)
    model = small_model(kind).eval().cuda()
    tokens = torch.randint(0, 64, (1, 37), device="cuda")
    caches = make_baseline_caches(model, 37)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(tokens)
        outputs = [model(tokens[:, :11], cache=caches)]
        outputs += [model(tokens[:, i:i + 1], cache=caches) for i in range(11, 37)]
    torch.testing.assert_close(torch.cat(outputs, 1).float(), full.float(), atol=0.003, rtol=0.02)


def test_model_reload_preserves_baseline_outputs(tmp_path, monkeypatch):
    from experiments.scaling.longctx_infer import load_model, MODEL_CONFIGS
    model = small_model("swan").eval()
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    cfg = dict(vocab_size=64, d_model=32, n_layers=8, n_heads=4, d_ff=64, max_seq_len=64)
    monkeypatch.setitem(MODEL_CONFIGS, "_rebuttal_test", cfg)
    restored = load_model(str(checkpoint), config="_rebuttal_test", attn_type="swan", device="cpu",
                          window_size=4, swan_scale_base=16)
    x = torch.randint(0, 64, (1, 19))
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["standard", "swan"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_greedy_generation_matches_full_prefix_and_stops_at_eos(kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(51)
    model = small_model(kind).eval().to(device)
    prompt = torch.randint(0, 64, (2, 13), device=device)
    expected = prompt.clone()
    with torch.no_grad():
        for _ in range(7):
            nxt = model(expected)[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat([expected, nxt], dim=1)
    actual = baseline_cached_generate(model, prompt, 7, device=device, dtype="float32")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    eos = int(expected[0, prompt.shape[1]])
    stopped = baseline_cached_generate(model, prompt[0], 7, device=device,
                                       dtype="float32", eos_token_id=eos)
    torch.testing.assert_close(stopped, expected[:1, :prompt.shape[1] + 1], atol=0, rtol=0)
    unchanged = baseline_cached_generate(model, prompt, 0, device=device)
    torch.testing.assert_close(unchanged, prompt, atol=0, rtol=0)
