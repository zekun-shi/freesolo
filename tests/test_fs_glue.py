"""FS (dense) and FS-L (sparse) prefills that keep no KV cache with K and V taken in the projections' token-major
layout (FS_GLUE_KV, FSAttn._kv_token_major_ok / _project_into) against the same model with the switch off, bit for
bit: hidden states, every layer's selected cells, the KV cache the prefill leaves, eager and CUDA-graph captured
prefills (replayed on new inputs), at the released attention shape (16 heads of 64, cells of 32, K = 4, window 512)
with partial query blocks, padded cells, prompts shorter than the window, two sequences, the graphed prefill's shared
K / V buffers and the selection, readout and window fallbacks; plus the copies the switch removes, where it must not
apply and the graphed decode, whose prefill keeps its cache."""
import os
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_REPO, "experiments", "scaling"), _REPO]

import bench_efficiency as be  # noqa: E402
from experiments.scaling.model import FreeSoloLM  # noqa: E402
from src.fsattn.attention import FSAttn  # noqa: E402


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    v = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}.get(t.dtype)
    return t.view(v) if v is not None else t


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _serve(monkeypatch, serve):
    # the bench's switches for its FS / FS-L serve configs (bench_efficiency.load and main)
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "1" if serve == "sparse" else "0")
    monkeypatch.setenv("FS_FUSED_SELECT", "1")
    monkeypatch.setenv("FSL_SLIDING_MODE", "chunked")
    monkeypatch.setenv("FS_GLUE_KV", "0")


def _model(serve, L, n_layers=2, seed=13):
    # the released attention shape (16 heads of 64) in the bench's FS / FS-L serve config, a small trunk; the norm
    # weights and long-range scales are spread so that the long-range branch is visible in the output
    torch.manual_seed(seed)
    m = FreeSoloLM(vocab_size=512, d_model=1024, n_layers=n_layers, n_heads=16, d_ff=512, max_seq_len=L + 512,
                   attn_type="fsattn", **be.serve_kwargs(serve)).cuda().to(torch.bfloat16).eval()
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    with torch.no_grad():
        for blk in m.blocks:
            a = blk.attn.fsattn
            a.lr_norm.scale.copy_(0.8 + 0.4 * torch.rand(a.head_dim, device="cuda", generator=g))
            a.long_range_scale.copy_(0.5 + torch.rand(a.n_heads, 1, 1, device="cuda", generator=g))
    from longctx_infer import extend_rope_cache
    extend_rope_cache(m, L + 256)
    return m


def _tokens(L, seed, B=1, ties=False):
    g = torch.Generator().manual_seed(seed)
    if ties:                                    # one 64-token pattern repeated: every layer-0 cell has a twin
        return torch.randint(0, 512, (B, 64), generator=g).repeat(1, -(-L // 64))[:, :L].cuda()
    return torch.randint(0, 512, (B, L), generator=g).cuda()


def _caches(m, B, L, retain, prealloc):
    # the caches of an eager prefill: per-layer preallocated K / V buffers when the cache is kept (cached_generate),
    # the graphed prefill's one pair of K / V buffers shared by every layer when it is not (GraphedPrefiller)
    from kv_cache import make_caches
    cs = make_caches(m, retain_kv=retain)
    if prealloc and retain:
        for c in cs:
            c.preallocate(B, L, torch.bfloat16, "cuda", keep=True)
    elif prealloc:
        a0 = m.blocks[0].attn.fsattn
        kb = torch.empty((B, a0.n_heads, L, a0.head_dim), dtype=torch.bfloat16, device="cuda")
        vb = torch.empty_like(kb)
        for c in cs:
            c._prealloc, c._keep_prealloc = (B, L, kb, vb, None, None), True
    return cs


def _prefill(monkeypatch, m, x, retain, prealloc):
    # eager prefill (the bench's trunk): hidden states, every layer's selected cells and the cache it leaves
    from kv_cache import _trunk_cached
    sels = []
    orig = FSAttn._cell_fused_lr

    def hook(self, Q, K, V, sel, s, Bq, *a, **k):
        sels.append(sel.clone())
        return orig(self, Q, K, V, sel, s, Bq, *a, **k)

    monkeypatch.setattr(FSAttn, "_cell_fused_lr", hook)
    cs = _caches(m, x.shape[0], x.shape[1], retain, prealloc)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = _trunk_cached(m, x, cs)
    monkeypatch.setattr(FSAttn, "_cell_fused_lr", orig)
    out = {"h": h.clone(), **{f"sel{i}": s for i, s in enumerate(sels)}}
    if retain:
        for j, c in enumerate(cs):
            for name in ("_kbar", "_ceil", "_k_hist", "_v_hist", "_cell_k"):
                t = getattr(c, name, None)
                if torch.is_tensor(t):
                    out[f"{name}{j}"] = t.clone()
    return out


def _check_switch(monkeypatch, m, x, retain, prealloc, n_sel):
    monkeypatch.setenv("FS_GLUE_KV", "0")
    ref = _prefill(monkeypatch, m, x, retain, prealloc)
    assert len([k for k in ref if k.startswith("sel")]) == n_sel
    assert torch.isfinite(ref["h"].float()).all()
    monkeypatch.setenv("FS_GLUE_KV", "1")
    got = _prefill(monkeypatch, m, x, retain, prealloc)
    assert got.keys() == ref.keys()
    for k in ref:
        assert _same(got[k], ref[k]), k


_CASES = [(8192, 1, False, False), (8000, 1, False, False), (4100, 1, False, False), (8192, 1, False, True),
          (4096, 1, True, False), (4000, 1, True, False), (3000, 1, True, False), (300, 1, True, False),
          (4608, 2, False, False), (3000, 2, True, False)]
_CASE_IDS = ["long", "long_partial_block", "long_padded_cells", "long_tied_cells", "shared_buffers",
             "shared_buffers_partial_block", "shared_buffers_padded_cells", "shorter_than_window", "batch2",
             "batch2_shared_buffers_padded_cells"]


@pytest.mark.parametrize("serve", ["dense", "sparse"])
@pytest.mark.parametrize("L,B,prealloc,ties", _CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("retain", [False, True], ids=["prefill_only", "keep_cache"])
def test_fs_glue_eager_prefill_matches_off(monkeypatch, serve, L, B, prealloc, ties, retain):
    # FS_GLUE_KV=1 against 0: same hidden states, selected cells (in order) and cache, bit for bit. Prompts above 4096
    # tokens without buffers take the projections' own output (the switch drops its .contiguous() copies), the others
    # the graphed prefill's shared K / V buffers when no cache is kept (the switch writes the projections into them
    # token-major instead of copying); a kept cache leaves the switch without effect
    _serve(monkeypatch, serve)
    m = _model(serve, L)
    x = _tokens(L, 21 + L, B, ties)
    _check_switch(monkeypatch, m, x, retain, prealloc, n_sel=len(m.blocks))


_FALLBACKS = {"legacy_select": {"FSL_FAST_SELECT": "0"}, "cublas_content": {"FSL_FUSED_CONTENT": "0"},
              "triton_select": {"FS_FUSED_SELECT": "0"}, "eager_readout": {"FSL_READOUT_FUSED": "0"},
              "serial_window": {"FS_WINDOW_OVERLAP": "0"}, "torch_rope": {"FSL_ROPE_FUSED": "0"},
              "chunked_window": {"FSL_ROPE_FUSED": "0", "FSL_WINDOW_BATCHED": "0"},
              "generic_window": {"FSL_ROPE_FUSED": "0", "FSL_SLIDING_MODE": "off"},
              "unfused": {"FS_FUSED_ATTN": "0"}}


@pytest.mark.parametrize("serve", ["dense", "sparse"])
@pytest.mark.parametrize("fallback", list(_FALLBACKS))
@pytest.mark.parametrize("L,prealloc", [(4608, False), (3000, True)], ids=["long", "shared_buffers_padded_cells"])
def test_fs_glue_fallbacks_match_off(monkeypatch, serve, fallback, L, prealloc):
    # the selection, readout and window fallbacks read the token-major K and V through their strides (torch glue,
    # cuBLAS, SDPA) and give the switch-off values bit for bit
    _serve(monkeypatch, serve)
    for name, value in _FALLBACKS[fallback].items():
        monkeypatch.setenv(name, value)
    m = _model(serve, L)
    x = _tokens(L, 7 + L)
    _check_switch(monkeypatch, m, x, False, prealloc, n_sel=0 if fallback == "unfused" else len(m.blocks))


@pytest.mark.parametrize("serve", ["dense", "sparse"])
@pytest.mark.parametrize("L,B,retain", [(4608, 1, False), (4100, 1, False), (3000, 2, False), (4608, 1, True)],
                         ids=["prefill_only", "prefill_only_padded_cells", "prefill_only_batch2", "keep_cache"])
def test_fs_glue_graphed_prefill_matches_eager_off(monkeypatch, serve, L, B, retain):
    # the switch inside a captured prefill (the forked window; the shared K / V buffers that every layer overwrites when
    # no cache is kept): replays on the captured and on new inputs equal the eager prefill with the switch off bit for
    # bit (above 4096 tokens it copies K and V into the (B, H, N, D) layout)
    from decode_graph import GraphedPrefiller
    from kv_cache import _trunk_cached, make_caches
    from longctx_infer import _lm_head_fp32
    _serve(monkeypatch, serve)
    m = _model(serve, L, n_layers=3)
    x1, x2 = _tokens(L, 31, B), _tokens(L, 32, B)
    monkeypatch.setenv("FS_GLUE_KV", "1")
    gp = GraphedPrefiller(m, L, B=B, device="cuda", dtype=torch.bfloat16, tail_tokens=64, retain_kv=retain)
    gp.capture(x1)
    monkeypatch.setenv("FS_GLUE_KV", "0")
    for x in (x1, x2, x1):
        got_last = gp.run(x).clone()
        got_tail = gp.tail_logits().clone()
        cs = make_caches(m, retain_kv=retain)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = _trunk_cached(m, x, cs)
            ref_last = _lm_head_fp32(m, h[:, -1:, :], 4096)[:, 0, :]
            ref_tail = _lm_head_fp32(m, h[:, -64:, :], 4096)
        assert torch.isfinite(ref_tail).all()
        assert _same(got_last, ref_last) and _same(got_tail, ref_tail)
        if retain:
            for cg, ce in zip(gp.cache, cs):
                for name in ("_kbar", "_ceil", "_k_hist", "_v_hist"):
                    assert _same(getattr(cg, name), getattr(ce, name)), name
    del gp


def _copies(fn) -> int:
    # (B, N, H, D) -> (B, H, N, D) layout copies: elementwise direct-copy kernels of one call
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
               and "direct_copy" in e.name and "elementwise_kernel<128" in e.name)


@pytest.mark.parametrize("serve", ["dense", "sparse"])
@pytest.mark.parametrize("prealloc", [False, True], ids=["projection_output", "shared_buffers"])
def test_fs_glue_drops_the_layout_copies_only_without_cache(monkeypatch, serve, prealloc):
    # FS_GLUE_KV=1 removes the K and V layout copies of every layer of a prefill that keeps no cache (the shared K / V
    # buffers of the graphed prefill or the projections' own output) and changes nothing when the cache keeps K / V
    from kv_cache import _trunk_cached
    _serve(monkeypatch, serve)
    L, n_layers = 8192, 2
    m = _model(serve, L, n_layers=n_layers)
    x = _tokens(L, 5)

    def run(retain):
        cs = _caches(m, 1, L, retain, prealloc)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _trunk_cached(m, x, cs)

    n = {}
    for kv in ("0", "1"):
        monkeypatch.setenv("FS_GLUE_KV", kv)
        for retain in (False, True):
            run(retain)
            n[kv, retain] = _copies(lambda: run(retain))
    assert n["0", False] - n["1", False] == 2 * n_layers, n
    assert n["0", True] == n["1", True], n


def test_glue_kv_switches_are_per_variant(monkeypatch):
    # FS_GLUE_KV steers the FS and FS-L layers only, FSB_GLUE_KV the FS-B layers only, and neither applies when the
    # cache keeps K / V, under autograd or to a CPU input
    class Cache:
        def __init__(self, retain):
            self.retain_kv = retain

    x = torch.zeros(1, 8, 1024, device="cuda", dtype=torch.bfloat16)
    for serve in ("dense", "sparse", "fsb"):
        torch.manual_seed(0)
        m = FreeSoloLM(vocab_size=64, d_model=1024, n_layers=1, n_heads=16, d_ff=64, max_seq_len=128,
                       attn_type="fsattn", **be.serve_kwargs(serve, 0.5 if serve == "fsb" else 0.0))
        a = m.blocks[0].attn.fsattn
        for fs in ("0", "1"):
            for fsb in ("0", "1"):
                monkeypatch.setenv("FS_GLUE_KV", fs)
                monkeypatch.setenv("FSB_GLUE_KV", fsb)
                with torch.no_grad():
                    assert a._kv_token_major_ok(x, Cache(False)) == ((fsb if serve == "fsb" else fs) == "1")
                    assert not a._kv_token_major_ok(x, Cache(True))
                    assert not a._kv_token_major_ok(x.cpu(), Cache(False))
                assert not a._kv_token_major_ok(x, Cache(False))


@pytest.mark.parametrize("serve", ["dense", "sparse"])
@torch.no_grad()
def test_fs_glue_graphed_decode_unchanged(monkeypatch, serve):
    # the graphed decode's prefill keeps its cache, so the switch leaves its teacher-forced steps as they were, bit for
    # bit, across query-block boundaries
    from decode_graph import GraphedDecoder
    _serve(monkeypatch, serve)
    L = 1024
    m = _model(serve, L)
    prompt = _tokens(L, 41)
    forced = torch.randint(0, 512, (200,), generator=torch.Generator().manual_seed(43)).cuda()
    runs = {}
    for kv in ("0", "1"):
        monkeypatch.setenv("FS_GLUE_KV", kv)
        gd = GraphedDecoder(m, prompt, Lmax=L + len(forced) + 16, device="cuda", dtype=torch.bfloat16)
        gd.prefill()
        logits = []
        for t in forced:
            gd.next_tok = t.view(1)
            gd.step()
            logits.append(gd.static_logits.clone())
        runs[kv] = torch.stack(logits)
        del gd
    assert torch.isfinite(runs["0"]).all()
    assert _same(runs["0"], runs["1"])
