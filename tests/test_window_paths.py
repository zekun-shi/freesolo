import os

import pytest
import torch

from src.utils import local_attention as la

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@gpu
@pytest.mark.parametrize("N", [700, 3000, 5121])
@pytest.mark.parametrize("chunk", [256, 1024, 1536])
def test_chunked_window_matches_generic_loop(monkeypatch, N, chunk):
    # the chunked band-mask path (FS-B's window branch) gives the generic per-256-query loop's output bit for bit
    g = torch.Generator(device="cuda").manual_seed(N)
    Q, K, V = (torch.randn(1, 4, N, 64, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3))
    ref = la.local_attention(Q, K, V, 512, scale=0.125, causal=True, sigma=None, fast_sliding=False)
    monkeypatch.setattr(la, "_SLIDE_QCHUNK", chunk)
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
    monkeypatch.setenv("FSL_WINDOW_BATCHED", "0")
    got = la.local_attention(Q, K, V, 512, scale=0.125, causal=True, sigma=None, fast_sliding=True)
    assert torch.equal(ref, got)


def _model_qkv(B, H, N, D, dtype, seed):
    # the model's layouts: rotated Q / K are contiguous (B, H, N, D); V is the transposed W_v output, (B, N, H, D) in memory
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", generator=g).to(dtype)
    K = torch.randn(B, H, N, D, device="cuda", generator=g).to(dtype)
    V = torch.randn(B, N, H, D, device="cuda", generator=g).to(torch.bfloat16).transpose(1, 2)
    return Q, K, V


def _spy_batched(monkeypatch):
    # the outputs of the batched calls that returned. local_attention falls back to the generic loop (the tests'
    # reference) on any exception, so a test must check that the value it got is the batched path's own output
    calls = []
    orig = la._batched_sliding_window

    def spy(*a, **k):
        out = orig(*a, **k)
        calls.append(out)
        return out
    monkeypatch.setattr(la, "_batched_sliding_window", spy)
    return calls


def _ran_batched(calls, got):
    return len(calls) == 1 and calls[0] is got


@gpu
@pytest.mark.parametrize("N", [513, 520, 575, 576, 577, 700, 1024, 3000, 5121, 16384, 16461])
@pytest.mark.parametrize("B", [1, 2])
def test_batched_window_matches_loop_and_chunked(monkeypatch, N, B):
    # one SDPA call over all 64-query blocks (overlapping strided K / V windows, one shared band) gives the generic
    # loop's and the chunked path's output bit for bit
    Q, K, V = _model_qkv(B, 16 if N >= 16384 else 4, N, 64, torch.bfloat16, N + B)
    ref = la.local_attention(Q, K, V, 512, scale=0.125, causal=True, sigma=None, fast_sliding=False)
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
    monkeypatch.setenv("FSL_WINDOW_BATCHED", "0")
    chk = la.local_attention(Q, K, V, 512, scale=0.125, causal=True, sigma=None, fast_sliding=True)
    monkeypatch.setenv("FSL_WINDOW_BATCHED", "1")
    calls = _spy_batched(monkeypatch)
    with torch.no_grad():
        got = la.local_attention(Q, K, V, 512, scale=0.125, causal=True, sigma=None, fast_sliding=True)
    assert _ran_batched(calls, got)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.equal(ref, chk)
    assert torch.equal(ref, got)
    # the output is a (B, H, N, D) view of a (B, N, H, D) buffer: the transpose W_o reads needs no copy
    assert got.transpose(1, 2).is_contiguous()


@gpu
@pytest.mark.parametrize("W,N", [(64, 300), (64, 1000), (128, 129), (128, 1000), (256, 3000), (1024, 5000)])
def test_batched_window_other_widths(monkeypatch, W, N):
    Q, K, V = _model_qkv(1, 4, N, 64, torch.bfloat16, W * 7 + N)
    ref = la.local_attention(Q, K, V, W, scale=0.125, causal=True, sigma=None, fast_sliding=False)
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
    monkeypatch.setenv("FSL_WINDOW_BATCHED", "1")
    calls = _spy_batched(monkeypatch)
    with torch.no_grad():
        got = la.local_attention(Q, K, V, W, scale=0.125, causal=True, sigma=None, fast_sliding=True)
    assert _ran_batched(calls, got)
    assert torch.equal(ref, got)


@gpu
@pytest.mark.parametrize("N", [700, 5121, 16384])
def test_batched_window_fp32_inputs_under_autocast(monkeypatch, N):
    # RULER regime: fp32 parameters, so the rotated Q / K are fp32 and SDPA's autocast casts them to bf16
    Q, K, V = _model_qkv(1, 16 if N >= 16384 else 4, N, 64, torch.float32, 3 * N)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = la.local_attention(Q, K, V, 512, scale=0.125, causal=True, sigma=None, fast_sliding=False)
        monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
        monkeypatch.setenv("FSL_WINDOW_BATCHED", "1")
        calls = _spy_batched(monkeypatch)
        got = la.local_attention(Q, K, V, 512, scale=0.125, causal=True, sigma=None, fast_sliding=True)
    assert _ran_batched(calls, got)
    assert ref.dtype == got.dtype == torch.bfloat16
    assert torch.equal(ref, got)


@gpu
def test_batched_window_not_taken_for_unaligned_width(monkeypatch):
    Q, K, V = _model_qkv(1, 4, 1000, 64, torch.bfloat16, 5)
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
    monkeypatch.setenv("FSL_WINDOW_BATCHED", "1")
    calls = _spy_batched(monkeypatch)
    with torch.no_grad():
        got = la.local_attention(Q, K, V, 500, scale=0.125, causal=True, sigma=None, fast_sliding=True)
    assert not calls
    assert torch.equal(la._chunked_sliding_window(Q, K, V, 500, 0.125), got)


def _qkv_layout(B, H, N, D, seed, qk="bhnd", dtype=torch.bfloat16):
    # qk "bhnd": rotated Q / K (contiguous, as the RoPE models hand them over); "bnhd": transposed linear outputs (the
    # no-RoPE models). V is always the transposed W_v output.
    g = torch.Generator(device="cuda").manual_seed(seed)
    if qk == "bhnd":
        Q, K = (torch.randn(B, H, N, D, device="cuda", generator=g).to(dtype) for _ in range(2))
    else:
        Q, K = (torch.randn(B, N, H, D, device="cuda", generator=g).to(dtype).transpose(1, 2) for _ in range(2))
    V = torch.randn(B, N, H, D, device="cuda", generator=g).to(dtype).transpose(1, 2)
    return Q, K, V


def _expected_block(D, W):
    # the SDPA kernel's key tile for the head dim: 64 keys up to 64, 128 keys up to 128; wider heads are not covered
    if D > 128:
        return 0
    T = 64 if D <= 64 else 128
    return T if W % T == 0 else 0


@gpu
@pytest.mark.parametrize("D", [16, 32, 64, 96, 128, 256])
@pytest.mark.parametrize("B,H,W,N,qk", [
    (1, 4, 512, 3000, "bhnd"), (2, 4, 512, 5121, "bhnd"), (3, 4, 512, 1000, "bnhd"), (1, 8, 512, 16384, "bhnd"),
    (2, 4, 128, 1001, "bnhd"), (1, 4, 192, 2000, "bhnd"), (2, 4, 256, 2048, "bhnd"), (1, 4, 1024, 4000, "bnhd"),
    (2, 4, 512, 513, "bhnd"), (3, 4, 512, 640, "bhnd"), (1, 4, 64, 700, "bhnd"),
])
def test_batched_window_head_dims(monkeypatch, D, B, H, W, N, qk):
    # every head dim of the configs (16 proxy256, 32 proxy512, 64 340m, 96 800m, 128 1.3b / 3b): the default is the
    # chunked path's output bit for bit. The batched path is taken where its query blocks match the kernel's key tile
    # (with 64-query blocks, head dims 96 / 128 differ from the chunked path in 6-13% of the outputs)
    Q, K, V = _qkv_layout(B, H, N, D, N * 7 + D + W + B, qk)
    sc = D ** -0.5
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
    with torch.no_grad():
        chk = la._chunked_sliding_window(Q, K, V, W, sc)
        monkeypatch.setenv("FSL_WINDOW_BATCHED", "1")
        calls = _spy_batched(monkeypatch)
        got = la.local_attention(Q, K, V, W, scale=sc, causal=True, sigma=None, fast_sliding=True)
        T = _expected_block(D, W)
        assert la._batched_block(Q, K, V, W) == T
        assert _ran_batched(calls, got) if T else not calls
        assert torch.equal(chk, got)
        if T:
            ref = la.local_attention(Q, K, V, W, scale=sc, causal=True, sigma=None, fast_sliding=False)
            assert torch.equal(ref, got)


@pytest.fixture
def deterministic():
    # The memory-efficient SDPA backward splits the keys over several CTAs when its grid is small and accumulates dQ
    # with atomics, so even the chunked path's dQ differs from run to run (18 of 20 runs at B=2, H=4, N=2048, W=512 on
    # an A100). Deterministic mode keeps one split. (Nothing here calls cuBLAS, whose deterministic-mode check reads
    # CUBLAS_WORKSPACE_CONFIG once per process: the training-step test runs in its own process for that.)
    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


@gpu
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("B,N,W", [(2, 2048, 512), (1, 3000, 128)])
def test_window_backward_matches_chunked(monkeypatch, deterministic, D, B, N, W):
    # training (autograd on): the default switches run the chunked path, so the output and dQ / dK / dV are the
    # FSL_WINDOW_BATCHED=0 values bit for bit (the batched path's backward through overlapping views accumulates with
    # atomics: 55-60% of dK / dV differed, and differed from run to run). Deterministic mode: see the fixture.
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
    g = torch.Generator(device="cuda").manual_seed(D + N)
    H = 4
    Q0 = torch.randn(B, H, N, D, device="cuda", generator=g)                 # fp32 parameters regime, as in training
    K0 = torch.randn(B, H, N, D, device="cuda", generator=g)
    V0 = torch.randn(B, N, H, D, device="cuda", generator=g).to(torch.bfloat16).transpose(1, 2)
    go = torch.randn(B, H, N, D, device="cuda", generator=g).to(torch.bfloat16)
    calls = _spy_batched(monkeypatch)

    def run(flag):
        monkeypatch.setenv("FSL_WINDOW_BATCHED", flag)
        Q, K = Q0.clone().requires_grad_(True), K0.clone().requires_grad_(True)
        V = V0.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = la.local_attention(Q, K, V, W, scale=D ** -0.5, causal=True, sigma=None, fast_sliding=True)
        o.backward(go)
        return o.detach(), Q.grad, K.grad, V.grad

    got, ref = run("1"), run("0")
    assert not calls
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


_TRAIN_STEP = r"""
import os, sys
root = sys.argv[1]
sys.path[:0] = [root, os.path.join(root, "experiments", "scaling")]
import torch
import src.utils.local_attention as la
from experiments.scaling.model import FreeSoloLM

torch.use_deterministic_algorithms(True)
calls = []
orig = la._batched_sliding_window
def spy(*a, **k):
    calls.append(1)
    return orig(*a, **k)
la._batched_sliding_window = spy
SWITCHES = ("FSL_WINDOW_BATCHED", "FSL_TRUNK_FUSED", "FSL_NORM_ONEPASS", "FSL_ROPE_FUSED", "FSL_PREFILL_CELL_REUSE",
            "FSL_READOUT_FUSED")

def step(flag):
    for name in SWITCHES:
        if flag is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = flag
    torch.manual_seed(0)
    m = FreeSoloLM(vocab_size=1000, d_model=256, n_layers=2, n_heads=4, d_ff=688, max_seq_len=2048,
                   attn_type="fsattn", window_size=512, long_range_variant="cell_topk", cell_stride=32,
                   cell_topk=4, cell_select="score", sel_strict_causal=1, use_rope=True, rope_long_range=False,
                   fs_uniform_sr=True).cuda().train()
    x = torch.randint(0, 1000, (2, 2048), device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(x)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, 1000), x[:, 1:].reshape(-1))
    loss.backward()
    return loss.detach(), {n: p.grad for n, p in m.named_parameters() if p.grad is not None}

l_def, g_def = step(None)
l_off, g_off = step("0")
import json
print("RESULT " + json.dumps({"batched_calls": len(calls), "loss_equal": bool(torch.equal(l_def, l_off)),
                              "n_grads": len(g_def), "same_params": sorted(g_def) == sorted(g_off),
                              "differ": [n for n in g_def if n in g_off and not torch.equal(g_def[n], g_off[n])]}))
"""


@gpu
def test_training_step_ignores_inference_switches():
    # One FreeSoloLM training step (RoPE window + NoPE long range, as configs/attn/fsattn_rope.yaml) under bf16
    # autocast: the loss and every parameter gradient are the same with the inference switches at their defaults as
    # with all of them off, and the batched window is never called. Deterministic mode (see the fixture above) in a
    # fresh process, since its cuBLAS check reads CUBLAS_WORKSPACE_CONFIG once per process.
    import json
    import subprocess
    import sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FSL_")}
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", FSL_SLIDING_MODE="chunked", FS_STRICT_CAUSAL="1")
    r = subprocess.run([sys.executable, "-c", _TRAIN_STEP, root], env=env, capture_output=True, text=True,
                       timeout=900)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert r.returncode == 0 and line is not None, (r.returncode, r.stdout[-2000:], r.stderr[-4000:])
    res = json.loads(line[len("RESULT "):])
    assert res["batched_calls"] == 0 and res["loss_equal"] and res["same_params"] and res["n_grads"] > 0, res
    assert res["differ"] == [], res
