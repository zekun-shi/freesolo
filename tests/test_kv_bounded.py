"""The bounded FS-L K / V store (src/fsattn/kv_bounded.py, FSL_KV_BOUNDED=1) against the full int8 store it replaces:
the cells it holds after the prefill and after every block refresh (the top P under torch.topk's tie order and the
cells the window and the newest Wc reach), the values held (kv_int8's quantization cell for cell), the pool it
assembles (trunk_triton.fsl_pool_refresh value for value), the window and long-range decode kernels reading through
its table, and GraphedDecoder's teacher-forced logits, which must be identical with the flag on and off. Kernels
are checked at the released models' attention shape (16 heads of 64, 32-token cells, window 512, top 4 of a
64 + 8 pool)."""
import os
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "scaling"))

from src.fsattn import kv_bounded as kvb  # noqa: E402
from src.fsattn import kv_int8  # noqa: E402
from src.utils import trunk_triton as tt  # noqa: E402

if not kvb._HAVE_TRITON:
    pytest.skip("needs Triton", allow_module_level=True)

H, D, S, W, P, WC, BQ = 16, 64, 32, 512, 64, 8, 128


def _attn(seed=3, w_dtype=torch.bfloat16, cfg=(S, W, P, WC)):
    from src.fsattn.attention import FSAttn
    torch.manual_seed(seed)
    s, w, p, wc = cfg
    a = FSAttn(d_model=H * D, n_heads=H, window_size=w, causal=True, use_rope=True, rope_long_range=False,
               fs_uniform_sr=True, max_seq_len=16384, cell_stride=s, cell_topk=4, sel_strict_causal=1,
               cell_prune_kernel=1, cell_prune_pool=p, cell_prune_window=wc).cuda().to(w_dtype).eval()
    with torch.no_grad():
        a.lr_norm.scale.copy_(0.8 + 0.4 * torch.randn(D, device="cuda"))
        a.long_range_scale.copy_(0.2 * torch.randn(H, 1, 1, device="cuda"))
        a.sel_scale.copy_(0.5 + torch.randn(H, device="cuda"))
    return a


def _kv(N, seed, ties=False, B=1):
    # random keys / values; ties: every row one of 6 vectors, so most cells share their largest key norm
    g = torch.Generator(device="cuda").manual_seed(seed)
    if ties:
        tab = torch.randn(6, D, device="cuda", generator=g) * 2
        idx = torch.randint(0, 6, (B, H, N), device="cuda", generator=g)
        K = tab[idx].bfloat16()
    else:
        K = (torch.randn(B, H, N, D, device="cuda", generator=g) * 2).bfloat16()
    V = (torch.randn(B, H, N, D, device="cuda", generator=g) * 2).bfloat16()
    return K, V


def _stats(K, ncell, s=S):
    B = K.shape[0]
    Kc = K[:, :, :ncell * s].reshape(B, H, ncell, s, D)
    return Kc.mean(dim=3).float(), Kc.float().norm(dim=-1).amax(dim=-1)


def _readable(ceil, ncell, blk, cfg=(S, W, P, WC)):
    # the cells the block at blk (and every later step) can read: its top P, and every cell from the first one its
    # window or its newest-cell candidates reach
    s, w, p, wc = cfg
    B = ceil.shape[0]
    m = min(blk // s, ncell)
    want = torch.zeros(B, H, ncell, dtype=torch.bool, device="cuda")
    want[:, :, max(0, min((blk - w) // s, m - wc)) if m > 0 else 0:] = True
    if m > 0:
        want.scatter_(2, ceil[:, :, :m].topk(min(p, m), dim=-1).indices, True)
    return want


def _check_held(st, K, V, kbar, ncell, want):
    # the held set is exactly want, every held cell's slot carries its int8 rows, scales and mean key
    B = K.shape[0]
    S = st.s
    held = st.slot_of[:, :, :ncell] >= 0
    assert torch.equal(held, want)
    assert (st.slot_of[:, :, ncell:] < 0).all()
    assert torch.equal((st.cell_in >= 0).sum(-1), want.sum(-1))
    assert st.faults() == (0, 0)
    qk, sk = kv_int8.quantize_cells(K, S)
    qv, sv = kv_int8.quantize_cells(V, S)
    cid = st.cell_in.long()
    occ = cid >= 0
    c = cid.clamp(min=0)
    n = st.n_slots

    def at_cells(x, width):
        return x.reshape(B, H, -1, width).gather(2, c[..., None].expand(B, H, n, width))

    assert torch.equal(st.slot_of.gather(2, c).long()[occ], torch.arange(n, device="cuda").expand(B, H, n)[occ])
    for got, ref in ((st.qk, qk), (st.qv, qv)):
        assert torch.equal(got.reshape(B, H, n, S * D)[occ], at_cells(ref[:, :, :ncell * S], S * D)[occ])
    for got, ref in ((st.sk, sk), (st.sv, sv)):
        assert torch.equal(got[occ], ref.gather(2, c)[occ])
    assert torch.equal(st.kbar[occ], at_cells(kbar, D)[occ])


@pytest.mark.parametrize("M", [100, 4096, 25000])
def test_topk_takes_tied_ceilings_in_index_order(M):
    # the property the store's retention rests on: torch.topk of a head's ceilings returns the P largest under
    # (ceiling descending, index ascending), also on the multi-block path (16 slices of >= 20000 cells)
    g = torch.Generator(device="cuda").manual_seed(M)
    for levels in (3, 50, 0):
        ceil = (torch.randint(0, levels, (1, H, M), device="cuda", generator=g).float() if levels
                else torch.rand(1, H, M, device="cuda", generator=g))
        k = min(P, M)
        got = ceil.topk(k, dim=-1).indices.sort(-1).values
        want = torch.sort(-ceil, dim=-1, stable=True).indices[..., :k].sort(-1).values
        assert torch.equal(got, want), (M, levels)


@pytest.mark.parametrize("N", [700, 3000, 4096, 4096 + 77, 8192 + 31])
@pytest.mark.parametrize("ties", [False, True])
def test_prefill_holds_the_readable_cells_as_the_int8_store(N, ties):
    a = _attn()
    K, V = _kv(N, seed=N + ties, ties=ties)
    ncell = N // S
    kbar, ceil = _stats(K, ncell)
    st = kvb.SlotStore(a, 1, (N + 600) // S, "cuda")
    assert st.n_slots == P + W // S + BQ // S + 1
    st.prefill(K, V, kbar, ceil, N)
    assert torch.equal(st.ceil[:, :, :ncell], ceil)
    _check_held(st, K, V, kbar, ncell, _readable(ceil, ncell, N // BQ * BQ))


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("cfg", [(S, W, P, WC), (64, 500, 37, 12), (16, 64, 16, 8), (32, 96, 64, 8)])
def test_blocks_refresh_like_the_reference_pool_and_drop_unreadable_cells(ties, cfg):
    # prompt, then 40 decode blocks of completed cells: after every refresh the pool equals fsl_pool_refresh over
    # the full statistics, the store holds exactly the readable cells, and never more than its slots (the released
    # cell size / window / pool, and others: a window that is not a whole number of cells, newest cells reaching
    # further back than the window, cells smaller than the window's tile)
    S, W, P, WC = cfg
    a = _attn(cfg=cfg)
    N0, nblk = 4096 + 40, 40
    Nmax = N0 + nblk * BQ + BQ
    K, V = _kv(Nmax, seed=7 + ties, ties=ties)
    st = kvb.SlotStore(a, 1, Nmax // S + 1, "cuda")
    assert st.n_slots == P + max(-(-W // S), WC) + BQ // S + 1
    ncell = N0 // S
    kbar, ceil = _stats(K, Nmax // S, S)
    st.prefill(K[:, :, :N0], V[:, :, :N0], kbar[:, :, :ncell], ceil[:, :, :ncell], N0)
    pool = None
    peak = 0
    for blk in range(N0 // BQ * BQ, N0 // BQ * BQ + nblk * BQ + 1, BQ):
        while ncell < blk // S:                          # the cells the decode completed before this block
            c0 = ncell * S
            st.store_cell(K[:, :, c0:c0 + S], V[:, :, c0:c0 + S], kbar[:, :, ncell:ncell + 1],
                          ceil[:, :, ncell:ncell + 1], ncell)
            ncell += 1
            peak = max(peak, int(st.held().max()))
        m = min(blk // S, ncell)
        pk = min(P, m)
        top = ceil[:, :, :m].topk(pk, dim=-1).indices
        ref = tt.fsl_pool_refresh(top, kbar[:, :, :ncell], ceil[:, :, :ncell], m, blk, S, pk, WC)
        pool = st.pool(blk, ncell, out=pool)
        for k in ("Kbar_cand", "Kbar_cand_t", "ceil_cand", "cen_cand", "ok_cell", "cand_c"):
            assert torch.equal(pool[k], ref[k]), (blk, k)
        assert pool["M_vis"] == m and pool["C"] == pk + WC
        _check_held(st, K[:, :, :ncell * S], V[:, :, :ncell * S], kbar[:, :, :ncell], ncell,
                    _readable(ceil, ncell, blk, cfg))
    assert peak <= st.n_slots


def test_store_cell_takes_the_lowest_free_slot_and_counts_overflow():
    a = _attn()
    N = 4096
    K, V = _kv(N + 2 * S, seed=11)
    kbar, ceil = _stats(K, N // S + 2)
    st = kvb.SlotStore(a, 1, N // S + 2, "cuda")
    st.prefill(K[:, :, :N], V[:, :, :N], kbar[:, :, :N // S], ceil[:, :, :N // S], N)
    free = (st.cell_in < 0).int().argmax(-1)
    c = N // S
    st.store_cell(K[:, :, N:N + S], V[:, :, N:N + S], kbar[:, :, c:c + 1], ceil[:, :, c:c + 1], c)
    assert torch.equal(st.slot_of[:, :, c].long(), free)
    assert torch.equal(st.ceil[:, :, c], ceil[:, :, c])
    qk, sk = kv_int8.quantize_cells(K[:, :, N:N + S], S)
    sl = int(free[0, 5])
    assert torch.equal(st.qk[0, 5, sl * S:(sl + 1) * S], qk[0, 5])
    assert torch.equal(st.sk[0, 5, sl], sk[0, 5, 0])
    st.cell_in.fill_(0)                                  # no free slot: the cell is counted, nothing is written
    before = st.qk.clone()
    st.store_cell(K[:, :, N + S:N + 2 * S], V[:, :, N + S:N + 2 * S], kbar[:, :, c + 1:c + 2],
                  ceil[:, :, c + 1:c + 2], c + 1)
    assert st.faults()[0] == 1 * H and torch.equal(st.qk, before)
    assert (st.slot_of[:, :, c + 1] < 0).all()


def _window_args(N, seed):
    # a prompt of N tokens in both stores (full int8 rows / bounded slots), the open cell's rows, and a new token
    a = _attn(seed)
    K, V = _kv(N, seed=seed)
    ncell = N // S
    nq = ncell * S
    kbar, ceil = _stats(K, ncell)
    qk = torch.zeros(1, H, N + 256, D, dtype=torch.int8, device="cuda")
    qv = torch.zeros_like(qk)
    sk = torch.ones(1, H, (N + 256) // S, dtype=torch.bfloat16, device="cuda")
    sv = torch.ones_like(sk)
    kv_int8.quantize_cells_fast(K[:, :, :nq], S, qk[:, :, :nq], sk[:, :, :ncell])
    kv_int8.quantize_cells_fast(V[:, :, :nq], S, qv[:, :, :nq], sv[:, :, :ncell])
    st = kvb.SlotStore(a, 1, (N + 256) // S, "cuda")
    st.prefill(K, V, kbar, ceil, N)
    kh = torch.zeros(1, H, S, D, dtype=torch.bfloat16, device="cuda")
    vh = torch.zeros_like(kh)
    kh[:, :, :N - nq] = K[:, :, nq:N]
    vh[:, :, :N - nq] = V[:, :, nq:N]
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    q, k, v = (torch.randn(1, H, 1, D, device="cuda", generator=g).bfloat16() for _ in range(3))
    nq_t = torch.full((), nq, dtype=torch.int64, device="cuda")
    pos_t = torch.full((), N, dtype=torch.int64, device="cuda")
    return a, st, (qk, qv, sk, sv, kh, vh, nq_t), q, k, v, pos_t


def _bf(x):
    return x.bfloat16().float()


def _rope_bf16(x, c, sn):
    # _apply_rotary_emb with every product, difference and sum rounded to bf16 (x fp32 values, c / sn bf16 rows)
    h = D // 2
    x1, x2, c, sn = x[..., :h], x[..., h:], c[:, :h].float(), sn[:, :h].float()
    return torch.cat([_bf(_bf(x1 * c) - _bf(x2 * sn)), _bf(_bf(x2 * c) + _bf(x1 * sn))], dim=-1)


def _window_ref(a, q, k, v, qk, sk, qv, sv, kh, vh, nq, pos):
    # torch reference of the int8 window decode (uniform window, RoPE, bf16 autocast): rows below n_q are int8 *
    # scale in fp32, the open cell's rows and the new token bf16; keys rotated in bf16 steps, values rounded to bf16,
    # then _sr_decode_from_buf's scores, softmax and readout
    idx = torch.arange(pos - W, pos + 1, device="cuda")
    q8 = idx < nq
    i8, cl, ih = idx.clamp(max=nq - 1), (idx // S).clamp(max=nq // S - 1), (idx - nq).clamp(min=0)

    def rows(qx, sx, hot, cur):
        x = torch.where(q8[None, None, :, None], qx[:, :, i8].float() * sx[:, :, cl].float()[..., None],
                        hot[:, :, ih].float())
        return torch.where((idx == pos)[None, None, :, None], cur.float(), x)

    cos, sin = a.rope.cos_cached, a.rope.sin_cached
    kw = _rope_bf16(rows(qk, sk, kh, k), cos[idx], sin[idx]).bfloat16()
    vw = rows(qv, sv, vh, v).bfloat16()
    qr = _rope_bf16(q.float(), cos[pos:pos + 1], sin[pos:pos + 1]).bfloat16()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        scores = torch.einsum("bhd,bhld->bhl", qr[:, :, 0], kw).float() * a.scale
        w = torch.softmax(scores, dim=-1).to(vw.dtype)
        return torch.einsum("bhl,bhld->bhd", w, vw).unsqueeze(2)


@pytest.mark.parametrize("N", [4096, 4096 + 77, 9000 + 31])
@torch.no_grad()
def test_window_decode_through_slots_matches_full_int8_store(N):
    a, st, (qk, qv, sk, sv, kh, vh, nq_t), q, k, v, pos_t = _window_args(N, seed=N % 97)
    win_idx = torch.arange(N - W, N + 1, device="cuda")
    win_pos = win_idx.float()
    cos, sin = a.rope.cos_cached, a.rope.sin_cached
    outs, hots = [], []
    for slots in (None, st.slot_of):
        khot, vhot = kh.clone(), vh.clone()
        args = (qk, sk, khot, qv, sv, vhot) if slots is None else (st.qk, st.sk, khot, st.qv, st.sv, vhot)
        assert tt.sr_decode_int8_ok(q, k, v, *args, nq_t, win_idx, win_pos, cos, slots=slots)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(tt.sr_decode_fused_int8(q, k, v, *args, nq_t, S, pos_t, win_idx, win_pos, a.sigma,
                                                a.scale, W, True, True, cos, sin, slots=slots))
        hots.append((khot, vhot))
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(hots[0][0], hots[1][0]) and torch.equal(hots[0][1], hots[1][1])
    nq = int(nq_t)
    ref = _window_ref(a, q, k, v, qk, sk, qv, sv, kh, vh, nq, N)
    assert torch.equal(outs[1], ref)
    assert torch.equal(hots[1][0][:, :, N - nq], k[:, :, 0]) and torch.equal(hots[1][1][:, :, N - nq], v[:, :, 0])


@pytest.mark.parametrize("N", [4096 + 77, 9000 + 31])
@pytest.mark.parametrize("ln_f32", [False, True])
@torch.no_grad()
def test_long_range_decode_through_slots_matches_full_int8_store(N, ln_f32):
    a, st, (qk, qv, sk, sv, kh, vh, nq_t), q, k, v, pos_t = _window_args(N, seed=N % 89 + 5)
    if ln_f32:
        a.lr_norm.scale.data = a.lr_norm.scale.data.float()
    blk = N // BQ * BQ
    pool = st.pool(blk, N // S)
    sr = torch.randn(1, H, 1, D, device="cuda").bfloat16()
    outs = []
    for slots in (None, st.slot_of):
        state = ((qk, qv, sk, sv, kh, vh, nq_t, S) if slots is None
                 else (st.qk, st.qv, st.sk, st.sv, kh, vh, nq_t, S))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = tt.lr_comb_int8(a, q, sr, pool, pos_t, a.sigma, state, slots=slots)
        assert out is not None
        outs.append(out)
    assert torch.equal(outs[0], outs[1])
    # the bf16 kernel on the dequantized history (int8 * scale rounded to bf16, the values the int8 reads produce)
    Kd = kv_int8.dequant_cells(qk[:, :, :N // S * S], sk, S).contiguous()
    Vd = kv_int8.dequant_cells(qv[:, :, :N // S * S], sv, S).contiguous()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = tt.fsl_lr_decode_combine(a, q, Kd, Vd, sr, pool, pos_t, a.sigma)
    assert torch.equal(outs[1], ref)
    # and the torch path (_lr_decode_from_pool_buf's cell scores, top 4, masked token softmax, lr_norm, then the
    # window combine): the same cells, up to the online softmax's summation order
    B = 1
    C = pool["C"]
    lam = a.sel_scale.float().view(1, H, 1)
    sig = a.sigma.float().view(1, H, 1)
    content = torch.einsum("bhd,bhcd->bhc", q[:, :, 0].float(), pool["Kbar_cand"]) * a.scale
    score = content + lam * pool["ceil_cand"] * torch.erf((float(N) - pool["cen_cand"]).abs() / (2.0 * sig))
    sel = score.masked_fill(~pool["ok_cell"], -1e9).topk(4, dim=-1).indices
    cellmask = torch.zeros(B, H, C, dtype=torch.bool, device="cuda").scatter_(-1, sel, True) & pool["ok_cell"]
    tok = (pool["cand_c"][..., None] * S + torch.arange(S, device="cuda")).reshape(B, H, C * S)
    Kc = Kd.gather(2, tok[..., None].expand(B, H, C * S, D))
    Vc = Vd.gather(2, tok[..., None].expand(B, H, C * S, D))
    allow = cellmask.repeat_interleave(S, dim=-1) & (tok < N)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sc = torch.einsum("bhd,bhkd->bhk", q[:, :, 0], Kc).float() * a.scale
        w = sc.masked_fill(~allow, -1e9).softmax(-1).to(Vc.dtype)
        ob = torch.einsum("bhk,bhkd->bhd", w, Vc).unsqueeze(2)
        out = sr + a.long_range_scale.unsqueeze(0) * a.lr_norm(ob.to(q.dtype))
    out = out.transpose(1, 2).reshape(B, 1, H * D).float()
    o = outs[1].float()
    assert ((o - out).norm() / out.norm()).item() < 1e-2


def _model(d_ff=512, n_layers=2, vocab=512, serve="sparse"):
    import bench_efficiency as be
    from experiments.scaling.model import FreeSoloLM
    torch.manual_seed(17)
    m = FreeSoloLM(vocab_size=vocab, d_model=H * D, n_layers=n_layers, n_heads=H, d_ff=d_ff, max_seq_len=8192,
                   attn_type="fsattn", **be.serve_kwargs(serve)).cuda().to(torch.bfloat16).eval()
    for blk in m.blocks:                                 # the released checkpoints keep the norms in fp32
        blk.attn.fsattn.lr_norm.scale.data = blk.attn.fsattn.lr_norm.scale.data.float()
    return m


def _env(monkeypatch):
    for k, v in (("FS_FUSED_ATTN", "1"), ("FS_KV_INT8", "1"), ("FS_PRUNE_KERNEL", "1"), ("FS_STRICT_CAUSAL", "1"),
                 ("FS_FUSED_SELECT", "1"), ("FS_PRUNE_POOL", "64"), ("FS_PRUNE_WINDOW", "8"),
                 ("FS_PRUNE_BLKCHUNK", "16")):
        monkeypatch.setenv(k, v)


@pytest.mark.parametrize("L,ties,prealloc", [(4096 + 77, False, "1"), (4096 + 77, False, "0"), (3 * 1024, True, "1"),
                                            (8192, False, "1")])
@torch.no_grad()
def test_graphed_decoder_bounded_matches_full_store(monkeypatch, L, ties, prealloc):
    # teacher-forced: both runs decode the same tokens; every step's logits must be identical with the flag on and
    # off, across many block refreshes (the tied prompt makes most layer-0 cell ceilings exactly equal); without
    # KV_PREALLOC the prefill allocates the store itself
    from decode_graph import GraphedDecoder
    _env(monkeypatch)
    monkeypatch.setenv("KV_PREALLOC", prealloc)
    m = _model()
    g = torch.Generator().manual_seed(L)
    hi = 6 if ties else 512
    prompt = torch.randint(0, hi, (1, L), generator=g).cuda()
    forced = torch.randint(0, hi, (700,), generator=g).cuda()
    runs = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("FSL_KV_BOUNDED", flag)
        gd = GraphedDecoder(m, prompt, Lmax=L + len(forced) + 16, device="cuda", dtype=torch.bfloat16)
        gd.prefill()
        assert gd.bounded_kv == (flag == "1")
        assert all(c.use_int8 for c in gd.cache)
        logits = []
        for t in forced:
            gd.next_tok = t.view(1)
            gd.step()
            logits.append(gd.static_logits.clone())
        runs[flag] = torch.stack(logits)
        if flag == "1":
            assert gd.kv_faults() == (0, 0)
            for c in gd.cache:
                assert int(c._slots.held().max()) <= c._slots.n_slots < c._ncell     # cells were dropped
                assert c._qk_buf is None and c._k_buf is None
    assert torch.equal(runs["0"], runs["1"])


@torch.no_grad()
def test_bounded_store_refuses_history_reads_and_stays_off_where_unsupported(monkeypatch):
    from decode_graph import GraphedDecoder
    _env(monkeypatch)
    m = _model()
    prompt = torch.randint(0, 512, (1, 2048), generator=torch.Generator().manual_seed(5)).cuda()
    monkeypatch.setenv("FSL_KV_BOUNDED", "1")
    gd = GraphedDecoder(m, prompt, Lmax=2048 + 64, device="cuda", dtype=torch.bfloat16)
    gd.prefill()
    assert gd.bounded_kv
    c = gd.cache[0]
    for read in (c.lr_tokens, c.read_full_kv, c.completed_cells, c.rerank_keys, lambda: c.sr_window()):
        with pytest.raises(RuntimeError, match="bounded store"):
            read()
    gd.step()
    # a prompt shorter than the window, the bf16 store, an unsupported switch and FS (dense top-k over every cell)
    # keep the full store
    short = GraphedDecoder(m, prompt[:, :W - 1], Lmax=W + 64, device="cuda", dtype=torch.bfloat16)
    short.prefill()
    assert not short.bounded_kv
    for env in (("FS_KV_INT8", "0"), ("FSL_LR_DECODE_FUSED", "0")):
        monkeypatch.setenv(*env)
        gd2 = GraphedDecoder(m, prompt, Lmax=2048 + 64, device="cuda", dtype=torch.bfloat16)
        gd2.prefill()
        assert not gd2.bounded_kv
        monkeypatch.delenv(env[0])
    monkeypatch.setenv("FS_PRUNE_KERNEL", "0")
    dense = GraphedDecoder(_model(serve="dense"), prompt, Lmax=2048 + 64, device="cuda", dtype=torch.bfloat16)
    dense.prefill()
    assert not dense.bounded_kv and dense.cache[0]._qk_buf is not None


@torch.no_grad()
def test_graphed_generate_raises_on_a_store_fault(monkeypatch):
    # a fault counter above zero means a read missed the store: graphed_generate must not return those tokens
    from decode_graph import graphed_generate
    _env(monkeypatch)
    monkeypatch.setenv("FSL_KV_BOUNDED", "1")
    m = _model()
    prompt = torch.randint(0, 512, (1, 1024), generator=torch.Generator().manual_seed(3)).cuda()
    out = graphed_generate(m, prompt, 8)
    assert out.shape == (1, 1032)
    orig = kvb.SlotStore.prefill

    def faulty(self, *a, **k):
        orig(self, *a, **k)
        self.flags[1] += 1

    monkeypatch.setattr(kvb.SlotStore, "prefill", faulty)
    with pytest.raises(RuntimeError, match="bounded K / V store fault"):
        graphed_generate(m, prompt, 8)
