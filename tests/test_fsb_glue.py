"""The FS-B prefill's work around its selection kernels: K and V taken in the projections' token-major layout when the
prefill keeps no KV cache (FSB_GLUE_KV, FSAttn._kv_token_major_ok / _project_into) and the layer's cell statistics and
forests built on a side stream forked right after the K projection in a captured prefill (FSB_GLUE_FORK,
fsb_prefill.fork_layer_forests), against the same model with both switches off, bit for bit: hidden states, every
layer's selected cells, the KV cache the prefill leaves, eager and CUDA-graph captured prefills (replayed on new
inputs), at the released attention shape (16 heads of 64, cells of 32, K = 4, beam width 16) with partial query
blocks, padded cells, several stages, two sequences and tied cells; plus the work each switch removes and where it
must not apply."""
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
from src.fsattn import fsb_prefill as fp  # noqa: E402
from src.fsattn.attention import FSAttn  # noqa: E402

pytestmark = pytest.mark.skipif(not fp._HAVE_TRITON, reason="needs Triton")

SWITCHES = ("FSB_GLUE_KV", "FSB_GLUE_FORK")


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    v = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}.get(t.dtype)
    return t.view(v) if v is not None else t


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _set(monkeypatch, on):
    for name in SWITCHES:
        monkeypatch.setenv(name, "1" if name in on else "0")


def _model(L, n_layers=2, seed=13):
    # the released attention shape (16 heads of 64) in the bench's FS-B serve config, a small trunk
    torch.manual_seed(seed)
    m = FreeSoloLM(vocab_size=512, d_model=1024, n_layers=n_layers, n_heads=16, d_ff=512, max_seq_len=L + 512,
                   attn_type="fsattn", **be.serve_kwargs("fsb", 0.5)).cuda().to(torch.bfloat16).eval()
    from longctx_infer import extend_rope_cache
    extend_rope_cache(m, L + 256)
    return m


def _calibrate(m, x):
    # fsb_tau 0 (the bench's override) and the union threshold at each head's median cell ceiling
    from kv_cache import _trunk_cached, make_caches
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        cs = make_caches(m)
        _trunk_cached(m, x, cs)
        for blk, c in zip(m.blocks, cs):
            blk.attn.fsattn.fsb_tau.fill_(0.0)
            blk.attn.fsattn.fsb_tau_union.copy_(c._ceil[0].float().median(dim=-1).values)


def _tokens(L, seed, ties=False, B=1):
    g = torch.Generator().manual_seed(seed)
    if ties:                                    # one 64-token pattern repeated: every layer-0 cell has a twin
        return torch.randint(0, 512, (B, 64), generator=g).repeat(1, -(-L // 64))[:, :L].cuda()
    return torch.randint(0, 512, (B, L), generator=g).cuda()


def _prefill(monkeypatch, m, x, retain):
    # eager prefill (the bench's trunk): hidden states, every layer's selected cells and the cache it leaves
    from kv_cache import _trunk_cached, make_caches
    sels = []
    orig = FSAttn._cell_fused_lr

    def hook(self, Q, K, V, sel, s, Bq, *a, **k):
        sels.append(sel.clone())
        return orig(self, Q, K, V, sel, s, Bq, *a, **k)

    monkeypatch.setattr(FSAttn, "_cell_fused_lr", hook)
    cs = make_caches(m, retain_kv=retain)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = _trunk_cached(m, x, cs)
    monkeypatch.setattr(FSAttn, "_cell_fused_lr", orig)
    out = {"h": h.clone(), **{f"sel{i}": s for i, s in enumerate(sels)}}
    if retain:
        for j, c in enumerate(cs):
            for name in ("_kbar", "_ceil", "_k_hist", "_v_hist", "_qk_buf", "_qv_buf", "_ks_buf", "_vs_buf"):
                t = getattr(c, name, None)
                if torch.is_tensor(t):
                    out[f"{name}{j}"] = t.clone()
    for blk in m.blocks:
        assert getattr(blk.attn.fsattn, "_fsb_fork", None) is None      # every fork was joined
    return out


@pytest.mark.parametrize("L,blk,ties,B", [(4096, "128", False, 1), (8000, "16", False, 1), (4000, "8", False, 1),
                                          (3000, "8", False, 1), (4096, "8", True, 1), (300, "128", False, 1),
                                          (4608, "8", False, 2)],
                         ids=["one_stage", "long_partial_block", "partial_block", "padded_cells", "tied_cells",
                              "short", "batch2"])
@pytest.mark.parametrize("retain", [False, True], ids=["prefill_only", "keep_cache"])
def test_glue_eager_prefill_matches_off(monkeypatch, L, blk, ties, B, retain):
    # every switch combination against both off: same hidden states, selected cells (in order) and cache, bit for bit
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "0")
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", blk)
    m = _model(L)
    x = _tokens(L, 21 + L, ties, B)
    _set(monkeypatch, ())
    _calibrate(m, x)
    ref = _prefill(monkeypatch, m, x, retain)
    assert len([k for k in ref if k.startswith("sel")]) == 2
    for on in (("FSB_GLUE_KV",), ("FSB_GLUE_FORK",), SWITCHES):
        _set(monkeypatch, on)
        got = _prefill(monkeypatch, m, x, retain)
        assert got.keys() == ref.keys(), on
        for k in ref:
            assert _same(got[k], ref[k]), (on, k)


@pytest.mark.parametrize("retain", [False, True], ids=["prefill_only", "keep_cache"])
def test_glue_graphed_prefill_matches_eager_off(monkeypatch, retain):
    # both switches inside a captured prefill (several stages on the stage pipe, the forked window, the bench's shared
    # K / V buffers that every layer overwrites when no cache is kept): replays on the captured and on new inputs equal
    # the eager prefill with the switches off bit for bit
    from decode_graph import GraphedPrefiller
    from kv_cache import _trunk_cached, make_caches
    from longctx_infer import _lm_head_fp32
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "0")
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "8")     # 32 query blocks: four stages per layer
    L = 4096
    m = _model(L, n_layers=3)
    x1, x2 = _tokens(L, 31), _tokens(L, 32)
    _set(monkeypatch, ())
    _calibrate(m, x1)
    _set(monkeypatch, SWITCHES)
    gp = GraphedPrefiller(m, L, B=1, device="cuda", dtype=torch.bfloat16, tail_tokens=64, retain_kv=retain)
    gp.capture(x1)
    _set(monkeypatch, ())
    for x in (x1, x2, x1):
        got_last = gp.run(x).clone()
        got_tail = gp.tail_logits().clone()
        cs = make_caches(m, retain_kv=retain)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = _trunk_cached(m, x, cs)
            ref_last = _lm_head_fp32(m, h[:, -1:, :], 4096)[:, 0, :]
            ref_tail = _lm_head_fp32(m, h[:, -64:, :], 4096)
        assert _same(got_last, ref_last) and _same(got_tail, ref_tail)
        if retain:
            for cg, ce in zip(gp.cache, cs):
                assert _same(cg._kbar, ce._kbar) and _same(cg._ceil, ce._ceil)
    for blk in m.blocks:
        assert getattr(blk.attn.fsattn, "_fsb_fork", None) is None


@pytest.mark.parametrize("N,extra", [(4096, 0), (3000, 0), (3000, 40), (1, 0)])
@pytest.mark.parametrize("autocast", [True, False])
def test_project_into_matches_linear(N, extra, autocast):
    # the projection written into a (B, H, cap, D) buffer's storage is the module's F.linear output bit for bit, as a
    # (B, H, N, D) view of that storage
    torch.manual_seed(N + extra)
    lin = torch.nn.Linear(1024, 1024, bias=False).cuda().to(torch.bfloat16)
    x = torch.randn(1, N, 1024, device="cuda").to(torch.bfloat16)
    buf = torch.full((1, 16, N + extra, 64), float("nan"), device="cuda", dtype=torch.bfloat16)
    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if autocast else torch.no_grad()
    with torch.no_grad(), ctx:
        ref = lin(x).view(1, N, 16, 64).transpose(1, 2)
        got = FSAttn._project_into(lin, x, buf)
    assert got.data_ptr() == buf.data_ptr() and got.shape == ref.shape and got.stride() == ref.stride()
    assert _same(got, ref)


def _copies(fn) -> int:
    # (B, N, H, D) -> (B, H, N, D) layout copies: elementwise direct-copy kernels of one call
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
               and "direct_copy" in e.name and "elementwise_kernel<128" in e.name)


@pytest.mark.parametrize("prealloc", [False, True])
def test_glue_kv_drops_the_layout_copies_only_without_cache(monkeypatch, prealloc):
    # FSB_GLUE_KV=1 removes the K and V layout copies of every layer of a prefill that keeps no cache (the shared K / V
    # buffers of the graphed prefill or the projections' own output) and changes nothing when the cache keeps K / V
    from kv_cache import _trunk_cached, make_caches
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "0")
    L, n_layers = 8192, 2
    m = _model(L, n_layers=n_layers)
    x = _tokens(L, 5)
    _set(monkeypatch, ())
    _calibrate(m, x)

    def run(retain):
        cs = make_caches(m, retain_kv=retain)
        if prealloc and not retain:                          # GraphedPrefiller's buffers for retain_kv=False
            a0 = m.blocks[0].attn.fsattn
            kb = torch.empty((1, a0.n_heads, L, a0.head_dim), dtype=torch.bfloat16, device="cuda")
            vb = torch.empty_like(kb)
            for c in cs:
                c._prealloc, c._keep_prealloc = (1, L, kb, vb, None, None), True
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _trunk_cached(m, x, cs)

    n = {}
    for kv in ("0", "1"):
        monkeypatch.setenv("FSB_GLUE_KV", kv)
        for retain in (False, True):
            run(retain)
            n[kv, retain] = _copies(lambda: run(retain))
    assert n["0", False] - n["1", False] == 2 * n_layers, n
    assert n["0", True] == n["1", True], n


@pytest.mark.parametrize("forced", [False, True], ids=["capture", "eager_forced"])
def test_glue_fork_is_taken_where_the_fast_prefill_runs(monkeypatch, forced):
    # FSB_GLUE_FORK=1 builds each layer's forests once, on the fork stream, inside a captured prefill (and eagerly when
    # the capture-only gate is forced open), and the fast prefill takes them; the eager pass before the capture, padded
    # cells and the switch off keep the build on the long-range branch
    from decode_graph import GraphedPrefiller
    from kv_cache import _trunk_cached, make_caches
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "0")
    if forced:
        monkeypatch.setattr(fp, "_fork_now", lambda: True)
    calls = []
    real = fp._layer_forests

    def spy(*a, **k):
        side = fp._FORK_STREAMS.get(str(torch.device("cuda", torch.cuda.current_device())))
        calls.append((torch.cuda.is_current_stream_capturing(),
                      side is not None and torch.cuda.current_stream().cuda_stream == side.cuda_stream))
        return real(*a, **k)

    monkeypatch.setattr(fp, "_layer_forests", spy)
    for L, fork in ((4096, "1"), (3000, "1"), (4096, "0")):
        m = _model(L)
        x = _tokens(L, 9)
        _set(monkeypatch, ())
        _calibrate(m, x)
        monkeypatch.setenv("FSB_GLUE_FORK", fork)
        calls.clear()
        n = len(m.blocks)
        forked = fork == "1" and L % 32 == 0
        if forced:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _trunk_cached(m, x, make_caches(m, retain_kv=False))
            assert calls == [(False, forked)] * n, (L, fork, calls)
        else:
            gp = GraphedPrefiller(m, L, B=1, device="cuda", dtype=torch.bfloat16, tail_tokens=64, retain_kv=False)
            gp.capture(x)                                    # an eager warm-up pass, then the capture
            assert calls == [(False, False)] * n + [(True, forked)] * n, (L, fork, calls)
            gp.run(x)
            del gp
        for blk in m.blocks:
            assert getattr(blk.attn.fsattn, "_fsb_fork", None) is None


def test_glue_fork_unused_is_joined(monkeypatch):
    # a fork whose forests the long-range branch does not take (here its own gate sends it to the stage loop) is
    # still joined before the layer's output, and the output is the switch-off output (the capture-only gate forced
    # open, so the eager prefill forks)
    from kv_cache import _trunk_cached, make_caches
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "0")
    monkeypatch.setattr(fp, "_fork_now", lambda: True)
    L = 2048
    m = _model(L)
    x = _tokens(L, 3)
    _set(monkeypatch, ())
    _calibrate(m, x)
    joins = []
    real = fp.join_layer_forests

    def spy(attn):
        fk = real(attn)
        joins.append(fk is not None)
        return fk

    monkeypatch.setattr(fp, "join_layer_forests", spy)
    out = {}
    for fork in ("0", "1"):
        monkeypatch.setenv("FSB_GLUE_FORK", fork)
        joins.clear()
        orig_ok = fp.prefill_fast_ok

        def ok(attn, Q, tokK, Bq):                           # the fork's check passes, the branch's own one fails
            return orig_ok(attn, Q, tokK, Bq) if Q is tokK else False

        monkeypatch.setattr(fp, "prefill_fast_ok", ok)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out[fork] = _trunk_cached(m, x, make_caches(m, retain_kv=False)).clone()
        monkeypatch.setattr(fp, "prefill_fast_ok", orig_ok)
        assert (sum(joins) == len(m.blocks)) == (fork == "1"), (fork, joins)
        for blk in m.blocks:
            assert getattr(blk.attn.fsattn, "_fsb_fork", None) is None
    assert _same(out["0"], out["1"])


def test_glue_fork_forced_eager_matches_off(monkeypatch):
    # the forked build with the capture-only gate forced open in an eager prefill of several stages and the cache kept:
    # hidden states, selected cells and cache bit for bit as with the switch off
    monkeypatch.setenv("FS_FUSED_ATTN", "1")
    monkeypatch.setenv("FS_PRUNE_KERNEL", "0")
    monkeypatch.setenv("FSB_FAST_PREFILL_BLKCHUNK", "8")
    L = 4096
    m = _model(L)
    x = _tokens(L, 77)
    _set(monkeypatch, ())
    _calibrate(m, x)
    real_now = fp._fork_now
    for retain in (False, True):
        _set(monkeypatch, ())
        ref = _prefill(monkeypatch, m, x, retain)
        monkeypatch.setattr(fp, "_fork_now", lambda: True)
        _set(monkeypatch, SWITCHES)
        got = _prefill(monkeypatch, m, x, retain)
        monkeypatch.setattr(fp, "_fork_now", real_now)
        assert got.keys() == ref.keys()
        for k in ref:
            assert _same(got[k], ref[k]), (retain, k)
