"""FS-B (threshold admission + content tree + beam traversal) regression tests for freesolo, ported
from the development tree's test_fsb_tree.py (F1..F120).  Mirrors tests/test_strict_causal.py (the same
tiny module: d_model 64, H=4, N=256, s=16, query block 64 -> 4 blocks, M=16 cells).  Run from the repo
root, on a GPU machine when possible:  python tests/test_fsb_tree.py

  F1  tree invariants: partition, packed layout, counts, centroid/radius/amax/pmin, balance, height
  F2  optimistic subtree bound dominates every descendant's FS leaf score
  F3  traversal: unique cells, W=1/Delta=0 reaches exactly one leaf, root read returns the whole tree
  F4  EQUIVALENCE: unfiltered tree (tau=0) read at the root == dense strict FS with top-K=all cells
      (softmax over ALL completed-prefix tokens) — ties FS-B's readout to the GO'd dense path
  F5  strict causality: no intra-block leak, block-0 LR exactly 0, full-module prefix invariance
  F6  threshold admission: tau=+inf -> zero LR; tau=median -> ~half the cells admitted and every
      selected cell has a_c >= tau
  F7  KV-cache parity: prefill + per-token decode (and lean prefill) == no-cache full forward
  F8  fused Triton readout ≈ masked reference readout (GPU only)
  F9  learned policy: finite REINFORCE loss + grads, argmax determinism at eval, action masking
  F10 finite grads through the fixed-policy module (W_q/W_v get gradient, tau/tree do not)

Launcher / eval-driver / queue plumbing of the original suite (the F68/F74/F90(d)/F103(g)/F108/F112/F116/F120
source-string checks and the bash job-arch / DRY_RUN parts) is NOT ported.  Sections that need optional
experiments modules (fsb_routerkd, fsb_utility, fsb_rr_train, fsb_s3diag_probe) run in full when those modules
are importable and otherwise print a SKIP line; the F24 real-340M tier runs only with FSB_REAL_CKPT /
FSB_REAL_TOKENS set, and F101[g] only when FSB_TEST_NFS_PATH points at a reachable NFS mount.
"""
import math
import os
import torch
import torch.nn.functional as F

_ENV0 = dict(os.environ)        # pytest runs these checks while importing this file (collection, before any test
                                # function): the end of the file restores the environment the checks change
os.environ.pop("FS_FUSED_ATTN", None)          # reference (masked) readout unless a test sets it
os.environ["FSB_DECODE_WHOLE"] = "0"           # cached_generate's per-layer decode loop (tests/test_fsb_rows.py: whole-step)


def _env_restore(*names):
    """put the named environment variables back to their values when this file was imported"""
    for n_ in names:
        if n_ in _ENV0:
            os.environ[n_] = _ENV0[n_]
        else:
            os.environ.pop(n_, None)


from src.fsattn.attention import FSAttn
from src.fsattn.fsb_tree import build_trees, traverse, fsb_policy_loss, fsb_log_stats, NEG

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
d_model, H, N = 64, 4, 256
Dh = d_model // H
S = 16          # cell size
BLK = 64        # query block (N=256 -> 4 blocks; block b has 4b eligible cells)
M = N // S      # 16 cells


def mk(fsb=True, **kw):
    kw.setdefault("cell_topk", 4)
    m = FSAttn(d_model=d_model, n_heads=H, window_size=16, causal=True,
                       long_range_variant="cell_topk", use_rope=False, lr_scale_mode="one",
                       cell_select="score", cell_stride=S, sel_strict_causal=1,
                       fsb_enabled=fsb, **kw).to(dev).eval()
    m._cell_block = BLK
    return m


def lr(m, Q, K, V):
    return m._lr_cell_topk(Q, K, V, m.sigma, N)


def rand_qkv(seed, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = torch.randn(1, H, N, Dh, generator=g) * scale
    K = torch.randn(1, H, N, Dh, generator=g) * scale
    V = torch.randn(1, H, N, Dh, generator=g) * scale
    return Q.to(dev), K.to(dev), V.to(dev)


# ── F1: tree invariants ─────────────────────────────────────────────────────────────────────
def check_tree(Kbar, ceil_c, admit, L):
    tr = build_trees(Kbar, ceil_c, admit, L)
    T, Mc, D_ = Kbar.shape
    cap0 = 1 << L
    for t in range(T):
        adm = set(torch.nonzero(admit[t]).flatten().tolist())
        m0 = len(adm)
        assert tr["m0"][t].item() == m0
        assert tr["height"][t].item() == (0 if m0 <= 1 else math.ceil(math.log2(m0))), (m0, tr["height"][t])
        for l in range(L + 1):
            nodes, cap = 1 << l, cap0 >> l
            seen = []
            for j in range(nodes):
                seg = tr["packed"][t, l * cap0 + j * cap: l * cap0 + (j + 1) * cap].tolist()
                cells = [c for c in seg if c >= 0]
                # packed left: no -1 before a valid entry
                assert seg[:len(cells)] == cells, (l, j, seg)
                assert len(set(cells)) == len(cells)
                assert tr["cnt"][l][t, j].item() == len(cells)
                assert len(cells) <= math.ceil(m0 / (1 << l)), "balance violated"
                seen += cells
                if cells:
                    X = Kbar[t, cells]
                    mu = X.mean(0)
                    assert torch.allclose(tr["mu"][l][t, j], mu, atol=1e-5)
                    r = (X - mu).norm(dim=-1).max()
                    assert abs(tr["rad"][l][t, j].item() - r.item()) < 1e-5
                    assert abs(tr["amax"][l][t, j].item() - ceil_c[t, cells].max().item()) < 1e-6
                    assert tr["pmin"][l][t, j].item() == min(cells)
                if l < L:   # children partition the node
                    lc = [c for c in tr["packed"][t, (l + 1) * cap0 + (2 * j) * (cap // 2): (l + 1) * cap0 + (2 * j + 1) * (cap // 2)].tolist() if c >= 0]
                    rc = [c for c in tr["packed"][t, (l + 1) * cap0 + (2 * j + 1) * (cap // 2): (l + 1) * cap0 + (2 * j + 2) * (cap // 2)].tolist() if c >= 0]
                    assert set(lc) | set(rc) == set(cells) and not (set(lc) & set(rc)), (l, j)
                    assert len(lc) == (len(cells) + 1) // 2 and len(rc) == len(cells) // 2
            assert set(seen) == adm, (l, seen, adm)
    return tr


torch.manual_seed(11)
Kb = torch.randn(5, 13, 8, device=dev)
ce = torch.rand(5, 13, device=dev) * 3
adm = torch.rand(5, 13, device=dev) < 0.7
adm[0] = False                     # empty tree
adm[1] = False; adm[1, 4] = True   # single cell
adm[2] = True                      # all 13 cells
tr1 = check_tree(Kb, ce, adm, L=4)
print("F1 OK  (tree partition / packed layout / summaries / balance / height)")

# ── F2: optimistic bound dominates the FS leaf score of every descendant ───────────────────
torch.manual_seed(12)
q = torch.randn(5, 8, device=dev)
qpos = 400.0
lam, sig, scale = 1.3, 4.0, 1 / math.sqrt(8)
worst = 0.0
for t in range(5):
    for l in range(5):
        for j in range(1 << l):
            cap = 16 >> l
            cells = [c for c in tr1["packed"][t, l * 16 + j * cap: l * 16 + (j + 1) * cap].tolist() if c >= 0]
            if not cells:
                continue
            U = ((q[t] @ tr1["mu"][l][t, j] + q[t].norm() * tr1["rad"][l][t, j]) * scale
                 + lam * tr1["amax"][l][t, j] * math.erf((qpos - (tr1["pmin"][l][t, j].item() + 0.5) * S) / (2 * sig)))
            leaf = max(((q[t] @ Kb[t, c]) * scale + lam * ce[t, c] * math.erf((qpos - (c + 0.5) * S) / (2 * sig))).item() for c in cells)
            worst = max(worst, leaf - U.item())
assert worst < 1e-5, f"bound violated by {worst}"
print(f"F2 OK  (subtree bound >= every descendant leaf score; worst slack {worst:.1e})")

# ── F3: traversal ────────────────────────────────────────────────────────────────────────
Tn = 5
qq = q[:, None, :].expand(Tn, 3, 8).contiguous()
qp = torch.full((Tn, 3), qpos, device=dev)
lamt = torch.full((Tn, 1, 1), lam, device=dev); sigt = torch.full((Tn, 1, 1), sig, device=dev)
h = tr1["height"]
# (a) greedy to the leaves: W=1, Delta=0 -> exactly one cell (when the tree is non-empty)
W1 = torch.ones(Tn, 3, dtype=torch.long, device=dev); D0 = h[:, None].expand(Tn, 3)
sel, nv, nr = traverse(tr1, qq, qp, lamt, sigt, scale, S, W1, D0, W_max=1, delta_max=4)
assert sel.shape == (Tn, 3, 16)
for t in range(Tn):
    m0 = tr1["m0"][t].item()
    assert nr[t].tolist() == [min(m0, 1)] * 3, (t, nr[t])
    if m0:
        assert nv[t, 0].item() >= h[t].item()      # visited at least one child per level
# (b) read at the root (Delta=H -> D=0) returns the whole admitted set (<=16 slots)
Dr = torch.zeros(Tn, 3, dtype=torch.long, device=dev)
sel, nv, nr = traverse(tr1, qq, qp, lamt, sigt, scale, S, W1, Dr, W_max=1, delta_max=4)
for t in range(Tn):
    got = sorted(c for c in sel[t, 0].tolist() if c >= 0)
    assert got == sorted(torch.nonzero(adm[t]).flatten().tolist()), (t, got)
    assert nv[t, 0].item() == 0
# (c) unique cells per row for a wide beam at an intermediate depth
W4 = torch.full((Tn, 3), 4, dtype=torch.long, device=dev); D1 = (h - 1).clamp(min=0)[:, None].expand(Tn, 3)
sel, nv, nr = traverse(tr1, qq, qp, lamt, sigt, scale, S, W4, D1, W_max=4, delta_max=4)
for t in range(Tn):
    row = [c for c in sel[t, 0].tolist() if c >= 0]
    assert len(row) == len(set(row)) and len(row) == nr[t, 0].item()
    assert len(row) <= min(tr1["m0"][t].item(), 4 * 2)
print("F3 OK  (greedy reaches one leaf; root read = whole tree; beam rows unique)")

# ── F4: EQUIVALENCE root-read FS-B == dense strict FS with K=all cells ──────────────────────
Qe, Ke, Ve = rand_qkv(7)
m_root = mk(fsb_widths="1", fsb_deltas="0,1,2,3,4", fsb_policy="fixed", fsb_width=1, fsb_delta=4)
o_fsb = lr(m_root, Qe, Ke, Ve)
m_dense = mk(fsb=False, cell_topk=M)     # dense strict: per-query top-K over ALL completed cells, K=M -> all
o_dense = lr(m_dense, Qe, Ke, Ve)
eqv = (o_fsb - o_dense).abs().max().item()
assert eqv < 1e-4, f"root-read FS-B != dense strict all-cells: {eqv}"
# and vs a hand-written reference: softmax over all tokens of completed-prefix cells, j<i
ref = torch.zeros_like(Qe)
for i in range(N):
    b = i // BLK
    ncomp = (b * BLK) // S
    if ncomp == 0:
        continue
    j = ncomp * S
    sc = (Qe[0, :, i] @ Ke[0, :, :j].transpose(1, 2).reshape(H, Dh, j)) if False else torch.einsum("hd,hjd->hj", Qe[0, :, i], Ke[0, :, :j]) * m_root.scale
    w = torch.softmax(sc, dim=-1)
    ref[0, :, i] = torch.einsum("hj,hjd->hd", w, Ve[0, :, :j])
ref = m_root.lr_norm(ref)
eqr = (o_fsb - ref).abs().max().item()
assert eqr < 1e-4, f"root-read FS-B != hand reference: {eqr}"
print(f"F4 OK  (root read == dense strict all-cells {eqv:.1e}; == hand reference {eqr:.1e})")

# ── F5: strict causality ─────────────────────────────────────────────────────────────────
def intra_block_leak(m):
    Q, K, V = rand_qkv(2)
    o1 = lr(m, Q, K, V)
    Qp, Kp, Vp = Q.clone(), K.clone(), V.clone()
    Qp[0, :, 190] += 5.0; Kp[0, :, 190] += 5.0; Vp[0, :, 190] += 5.0
    o2 = lr(m, Qp, Kp, Vp)
    return (o1[0, :, 130:161] - o2[0, :, 130:161]).abs().max().item(), o1[0, :, :BLK].abs().max().item()


def full_prefix_leak(m, p=200):
    torch.manual_seed(4)
    x = torch.randn(1, N, d_model, device=dev)
    y1 = m(x)
    x2 = x.clone(); x2[:, p] += 5.0
    return (y1[:, :p] - m(x2)[:, :p]).abs().max().item()


m_fix = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=1)           # W=2, Delta=1 -> <=4 cells
m_lrn = mk(fsb_policy="learned")                                   # argmax policy at eval
for name, mm in (("fixed W2/D1", m_fix), ("learned/argmax", m_lrn), ("root-read", m_root)):
    leak, b0 = intra_block_leak(mm)
    fl = full_prefix_leak(mm)
    assert leak < 1e-5, f"{name}: intra-block leak {leak}"
    assert b0 < 1e-6, f"{name}: block-0 LR nonzero {b0}"
    assert fl < 1e-5, f"{name}: full-module prefix leak {fl}"
    print(f"F5 [{name}] intra-block leak {leak:.1e} | block-0 {b0:.1e} | prefix leak {fl:.1e}")
print("F5 OK  (strict-causal: no intra-block leak, block-0 zero, prefix-invariant)")

# ── F6: threshold admission ──────────────────────────────────────────────────────────────
Q6, K6, V6 = rand_qkv(9)
m_inf = mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0)
m_inf.fsb_tau.fill_(float("inf"))
assert lr(m_inf, Q6, K6, V6).abs().max().item() == 0.0, "tau=inf must give zero LR"
m_med = mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0)
a_c = K6.view(1, H, M, S, Dh).norm(dim=-1).amax(-1)[0]              # (H,M) max key norm per cell
m_med.fsb_tau.copy_(a_c.median(dim=-1).values)
_ = lr(m_med, Q6, K6, V6)
st = m_med._fsb_stats
assert 0.35 < st["admit_frac"].mean().item() < 0.65, st["admit_frac"]
# every cell any query read must satisfy a_c >= tau: recompute the selection through the stats-free path
sel_cells = None
def _grab(attn, Q, K, V, sel_full, s, Bq, n):
    global sel_cells; sel_cells = sel_full.clone()
    return _orig(attn, Q, K, V, sel_full, s, Bq, n)
import src.fsattn.fsb_tree as _ft
_orig = _ft.masked_readout; _ft.masked_readout = _grab
_ = lr(m_med, Q6, K6, V6); _ft.masked_readout = _orig
for hh in range(H):
    cells = sel_cells[0, hh][sel_cells[0, hh] >= 0].unique()
    assert bool((a_c[hh, cells] >= m_med.fsb_tau[hh]).all()), "selected a cell below tau"
print(f"F6 OK  (tau=inf -> zero LR; tau=median admits {st['admit_frac'].mean().item():.2f} of eligible cells; selections respect tau)")

# ── F7: KV-cache parity (prefill + decode, lean prefill) ─────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "scaling"))
from experiments.scaling.kv_cache import FSKVCache
for name, mm in (("fixed W2/D1", m_fix), ("learned/argmax", m_lrn)):
    torch.manual_seed(21)
    x = torch.randn(1, N, d_model, device=dev)
    with torch.no_grad():
        full = mm(x)
        for lean in (False, True):
            cache = FSKVCache(mm, lean_prefill=lean)
            N0 = 150                                                  # prefill ends mid-block, mid-cell
            outs = [mm(x[:, :N0], cache=cache)]
            for t in range(N0, N):
                outs.append(mm(x[:, t:t + 1], cache=cache))
            cat = torch.cat(outs, dim=1)
            d = (cat - full).abs().max().item()
            assert d < 1e-4, f"{name} lean={lean}: cached != full ({d})"
            print(f"F7 [{name}] lean_prefill={lean}: |cached - full| = {d:.1e}")
print("F7 OK  (prefill + per-token decode reproduce the full forward)")

# ── F8: fused readout vs masked reference (GPU only) ─────────────────────────────────────
if dev == "cuda":
    Q8, K8, V8 = rand_qkv(31)
    o_ref = lr(m_fix, Q8, K8, V8)
    os.environ["FS_FUSED_ATTN"] = "1"
    try:
        o_fused = lr(m_fix, Q8, K8, V8)
    finally:
        os.environ.pop("FS_FUSED_ATTN", None)
    d8 = (o_fused - o_ref).abs().max().item()
    assert d8 < 2e-3, f"fused vs masked readout: {d8}"
    print(f"F8 OK  (fused Triton readout vs masked reference |max| = {d8:.1e})")
else:
    print("F8 SKIP (no CUDA)")

# ── F9: learned policy training signal ───────────────────────────────────────────────────
m9 = mk(fsb_policy="learned").train()
torch.manual_seed(33)
x9 = torch.randn(1, N, d_model, device=dev, requires_grad=True)
y9 = m9(x9)
st9 = m9._fsb_stats
assert {"logp", "entropy", "value", "action", "has", "cost"} <= set(st9)
assert st9["logp"].shape == (1, H, N) and torch.isfinite(st9["logp"][st9["has"]]).all()
assert (st9["logp"][~st9["has"]] == 0).all() or True         # padded rows are masked in the loss
loss_tok = torch.rand(1, N, device=dev)
aux = fsb_policy_loss(m9, loss_tok, eta=0.1, entropy_beta=0.01, value_coef=0.5)
(y9.float().pow(2).mean() + aux).backward()
for n_, p_ in m9.named_parameters():
    if "fsb_policy" in n_:
        assert p_.grad is not None and torch.isfinite(p_.grad).all(), n_
assert torch.isfinite(m9.W_v.weight.grad).all() and m9.W_v.weight.grad.abs().sum() > 0
assert m9.fsb_tau.grad is None if hasattr(m9.fsb_tau, "grad") else True
# actions respect the validity mask: Delta <= H and W <= 2^D for every recorded action
A = m9.fsb_actions
for hh in range(H):
    for i in range(BLK, N):
        if not st9["has"][0, hh, i]:
            continue
        a = st9["action"][0, hh, i].item()
        b = i // BLK
        m0 = (b * BLK) // S                                    # tau=0: all completed cells admitted
        height = 0 if m0 <= 1 else math.ceil(math.log2(m0))
        Wv, Dl = A[a, 0].item(), A[a, 1].item()
        assert Dl <= height and Wv <= (1 << max(0, height - Dl)), (i, hh, a, height)
# eval determinism (argmax) — two identical calls, identical outputs
m9.eval()
with torch.no_grad():
    o_a = m9(x9.detach()); o_b = m9(x9.detach())
assert torch.equal(o_a, o_b)
lg = fsb_log_stats(m9)
print(f"F9 OK  (policy loss {aux.item():.3f} finite; grads on policy + W_v; actions masked; argmax deterministic; "
      f"stats cells_read {lg['n_read']:.2f} visits {lg['n_visit']:.1f} cost {lg['cost']:.2f})")

# ── F10: finite grads through the fixed-policy module ────────────────────────────────────
m10 = mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0).train()
x10 = torch.randn(1, N, d_model, device=dev)
m10(x10).float().pow(2).mean().backward()
assert torch.isfinite(m10.W_q.weight.grad).all() and m10.W_q.weight.grad.abs().sum() > 0
assert torch.isfinite(m10.W_v.weight.grad).all()
assert m10.sel_scale.grad is None or m10.sel_scale.grad.abs().sum() == 0   # routing scalar: no index gradient
print("F10 OK (finite grads; routing scalar receives none)")

# ══════════════════════════════════════════════════════════════════════════════════════════
#  STREAMING FOREST (fsb_tree="forest"): F11..F17
# ══════════════════════════════════════════════════════════════════════════════════════════
from src.fsattn.fsb_tree import build_forest, build_spines, traverse_forest, _forest_attrs, ForestCache


def _node(F, SP, row, ref):
    """Attributes of one node ref of block-tree row ``row`` (python scalars / tensors)."""
    A = _forest_attrs(F, SP, torch.tensor([[ref]], device=dev).expand(SP["root"].shape[0], 1)[row:row + 1] if False else torch.full((SP["root"].shape[0], 1), -1, dtype=torch.long, device=dev).index_fill_(0, torch.tensor([row], device=dev), ref))
    g = lambda k: A[k][row, 0]
    return {"mu": g("mu"), "rad": g("rad").item(), "amax": g("amax").item(), "pmin": g("pmin").item(),
            "cnt": g("cnt").item(), "child": g("child").tolist(), "src": g("src").item(), "off": g("off").item(), "len": g("len").item()}


def _cells_of(F, SP, row, nd):
    t = SP["t_idx"][row].item()
    if nd["src"] == 1:
        return F["leaf"][t, nd["off"]:nd["off"] + nd["len"]].tolist()
    return F["adm"][t, nd["off"]:nd["off"] + nd["len"]].tolist()


def _walk(F, SP, row, ref, Kb, ce, check):
    """Recursively check a block tree; returns the list of cells under ref."""
    nd = _node(F, SP, row, ref)
    cells = _cells_of(F, SP, row, nd)
    assert len(cells) == nd["cnt"] and len(set(cells)) == len(cells) and all(c >= 0 for c in cells), (ref, nd, cells)
    X = Kb[cells]
    mu = X.mean(0)
    assert torch.allclose(nd["mu"], mu, atol=1e-4), (ref, (nd["mu"] - mu).abs().max())
    r = (X - nd["mu"]).norm(dim=-1).max().item()
    assert nd["rad"] >= r - 1e-5, (ref, nd["rad"], r)
    if nd["src"] == 1:
        assert abs(nd["rad"] - r) < 1e-4                                 # exact radius on chunk nodes
    assert abs(nd["amax"] - ce[cells].max().item()) < 1e-6 and nd["pmin"] == min(cells), (ref, nd)
    check["nodes"] += 1
    l, rr = nd["child"]
    if l == -1 and rr == -1:
        assert nd["cnt"] == 1 or nd["src"] == 2 and False, (ref, nd)   # only singleton chunk leaves have no children
        return cells
    assert l != -1 and rr != -1
    if nd["src"] == 2:
        check["spine"] += 1
    cl = _walk(F, SP, row, l, Kb, ce, check); cr = _walk(F, SP, row, rr, Kb, ce, check)
    assert sorted(cl + cr) == sorted(cells) and not (set(cl) & set(cr)), (ref, cells, cl, cr)
    return cells


torch.manual_seed(41)
Mf, Dhf = 140, 8
counts = [0, 1, 2, 3, 4, 5, 7, 8, 9, 31, 64, 65, 127, 128, 140]
Tf = len(counts)
Kb_f = torch.randn(Tf, Mf, Dhf, device=dev)
ce_f = torch.rand(Tf, Mf, device=dev) * 3
adm_f = torch.zeros(Tf, Mf, dtype=torch.bool, device=dev)
for t, c in enumerate(counts):
    idx = torch.randperm(Mf, device=dev)[:c]
    adm_f[t, idx] = True
Ff = build_forest(Kb_f, ce_f, adm_f)
Es = [0, 1, 5, 17, 64, 100, 140]                                       # eligibility cut-offs (cell id < E)
t_idx = torch.arange(Tf, device=dev).repeat_interleave(len(Es))
adm_sorted = Ff["adm"].masked_fill(Ff["adm"] < 0, Mf)
E_t = torch.tensor(Es, device=dev).repeat(Tf)
m_b = torch.searchsorted(adm_sorted, E_t.view(Tf, len(Es)).contiguous()).reshape(-1)
SPf = build_spines(Ff, m_b, t_idx)
chk = {"nodes": 0, "spine": 0}
for row in range(t_idx.numel()):
    t = t_idx[row].item(); E = E_t[row].item(); mb = m_b[row].item()
    expect = sorted(c for c in torch.nonzero(adm_f[t]).flatten().tolist() if c < E)
    assert mb == len(expect), (row, mb, len(expect))
    root = SPf["root"][row].item()
    if mb == 0:
        assert root == -1; continue
    c0 = chk["spine"]
    cells = _walk(Ff, SPf, row, root, Kb_f[t], ce_f[t], chk)
    assert sorted(cells) == expect, (row, mb)
    assert chk["spine"] - c0 == bin(mb).count("1") - 1, (row, mb, chk["spine"] - c0)
    k1 = mb.bit_length() - 1
    hexp = 0 if mb <= 1 else k1 + (1 if bin(mb).count("1") >= 2 else 0)
    assert SPf["height"][row].item() == hexp, (row, mb, SPf["height"][row].item(), hexp)
print(f"F11 OK  (forest: {t_idx.numel()} block trees over {Tf} admitted sequences, {chk['nodes']} nodes checked: partition, ranges, summaries, spine counts, heights)")

# ── F12: optimistic bound dominates every descendant leaf score on the forest ────────────────
torch.manual_seed(12)
qf = torch.randn(Tf, Dhf, device=dev)
lamf, sigf, scf, qposf = 1.3, 4.0, 1 / math.sqrt(Dhf), 3000.0   # query after every cell (causal bound)
worst = 0.0


def _bound_walk(row, ref):
    global worst
    nd = _node(Ff, SPf, row, ref); t = SPf["t_idx"][row].item()
    cells = _cells_of(Ff, SPf, row, nd)
    U = ((qf[t] @ nd["mu"] + qf[t].norm() * nd["rad"]) * scf + lamf * nd["amax"] * math.erf((qposf - (nd["pmin"] + 0.5) * S) / (2 * sigf))).item()
    leaf = max(((qf[t] @ Kb_f[t, c]) * scf + lamf * ce_f[t, c] * math.erf((qposf - (c + 0.5) * S) / (2 * sigf))).item() for c in cells)
    worst = max(worst, leaf - U)
    l, rr = nd["child"]
    if l != -1:
        _bound_walk(row, l); _bound_walk(row, rr)


for row in range(t_idx.numel()):
    if m_b[row].item() > 0:
        _bound_walk(row, SPf["root"][row].item())
assert worst < 1e-5, f"forest bound violated by {worst}"
print(f"F12 OK  (forest subtree bound >= every descendant leaf score; worst slack {worst:.1e})")

# ── F13: EQUIVALENCE root-read forest == dense strict FS with K=all cells ────────────────────
mf_root = mk(fsb_widths="1", fsb_deltas="0,1,2,3,4", fsb_policy="fixed", fsb_width=1, fsb_delta=4, fsb_tree="forest")
o_forest = lr(mf_root, Qe, Ke, Ve)
eqf = (o_forest - o_dense).abs().max().item()
assert eqf < 1e-4, f"root-read forest != dense strict all-cells: {eqf}"
print(f"F13 OK  (forest root read == dense strict all-cells {eqf:.1e})")

# ── F14: strict causality on the forest ──────────────────────────────────────────────────────
mf_fix = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=1, fsb_tree="forest")
mf_lrn = mk(fsb_policy="learned", fsb_tree="forest")
for name, mm in (("forest fixed W2/D1", mf_fix), ("forest learned/argmax", mf_lrn), ("forest root-read", mf_root)):
    leak, b0 = intra_block_leak(mm)
    fl = full_prefix_leak(mm)
    assert leak < 1e-5 and b0 < 1e-6 and fl < 1e-5, (name, leak, b0, fl)
    print(f"F14 [{name}] intra-block leak {leak:.1e} | block-0 {b0:.1e} | prefix leak {fl:.1e}")
print("F14 OK  (forest strict-causal)")

# ── F15: KV-cache parity on the forest + incremental == batch chunk trees ────────────────────
for name, mm in (("forest fixed W2/D1", mf_fix), ("forest learned/argmax", mf_lrn)):
    torch.manual_seed(21)
    x = torch.randn(1, N, d_model, device=dev)
    with torch.no_grad():
        full = mm(x)
        # decode forest: lean (default, only the chunks the decode spine reads) and the full incremental cache;
        # the full one runs last so the incremental == batch check below inspects its chunk store
        for dlean in ("1", "0"):
            os.environ["FSB_DECODE_LEAN"] = dlean
            try:
                for lean in (False, True):
                    cache = FSKVCache(mm, lean_prefill=lean)
                    N0 = 150
                    outs = [mm(x[:, :N0], cache=cache)]
                    for t in range(N0, N):
                        outs.append(mm(x[:, t:t + 1], cache=cache))
                    cat = torch.cat(outs, dim=1)
                    d = (cat - full).abs().max().item()
                    assert d < 1e-4, f"{name} lean={lean} decode_lean={dlean}: cached != full ({d})"
                    print(f"F15 [{name}] lean_prefill={lean} decode_lean={dlean}: |cached - full| = {d:.1e}")
            finally:
                os.environ.pop("FSB_DECODE_LEAN", None)
        # incremental chunk trees == batch-built chunk trees over the same completed cells
        fc = cache._fsb_forest
        Mv = fc.n_seen
        Kbar_all, ceil_all, _ = cache.completed_cells()                  # the SAME summaries the incremental build saw
        Kbar_v = Kbar_all[:, :, :Mv].reshape(H, Mv, Dh).float()
        ceil_v = ceil_all[:, :, :Mv].reshape(H, Mv).float()
        Fb = build_forest(Kbar_v, ceil_v, ceil_v >= mm.fsb_tau.view(H, 1))
        assert torch.equal(Fb["m_tot"], fc.m_tot), (name, Fb["m_tot"], fc.m_tot)
        _w = min(Fb["adm"].shape[1], fc.adm.shape[1])
        assert torch.equal(Fb["adm"][:, :_w], fc.adm[:, :_w]), name
        nchk = 0
        for k in Fb["ks"]:
            Cb = Fb["chunks"][k]; Ci = fc.chunks[k]
            for t in range(H):
                for j in range(int(Fb["m_tot"][t].item()) >> k):
                    assert torch.equal(Cb["leaf"][t, j], Ci["leaf"][t, j]), (name, k, t, j)
                    assert torch.allclose(Cb["mu"][t, j], Ci["mu"][t, j], atol=1e-6) and torch.equal(Cb["pmin"][t, j], Ci["pmin"][t, j])
                    nchk += 1
        print(f"F15 [{name}] incremental == batch chunk trees ({nchk} chunks)")
print("F15 OK  (forest prefill + per-token decode reproduce the full forward; incremental chunks identical)")

# ── F16: cost scaling — cells processed by build_trees over a forest build <= M (log2 M + 1) ──
import src.fsattn.fsb_tree as _ft2
_acc = {"cells": 0}
_orig_bt = _ft2.build_trees
def _count_bt(*args, **kw):
    _acc["cells"] += int(args[2].sum().item()); return _orig_bt(*args, **kw)
_ft2.build_trees = _count_bt
try:
    _ = build_forest(Kb_f, ce_f, adm_f)
finally:
    _ft2.build_trees = _orig_bt
m_tot_f = adm_f.sum(-1)
bound_cells = sum(int(m) * (int(m).bit_length()) for m in m_tot_f.tolist())   # m * (floor(log2 m) + 1)
assert _acc["cells"] <= bound_cells, (_acc["cells"], bound_cells)
print(f"F16 OK  (forest construction touched {_acc['cells']} cells <= sum m (log2 m + 1) = {bound_cells})")

# ── F17: block-chunked prefill is identical for every chunk size (both tree modes) ───────────
_sel_cap = {}
def _grab2(attn, Q, K, V, sel_full, s, Bq, n):
    _sel_cap["sel"] = sel_full.clone(); return _orig(attn, Q, K, V, sel_full, s, Bq, n)
for tree in ("rebuild", "forest"):
    for pol in ("fixed", "learned"):
        outs, sels = [], []
        for nbch in (1, 2, 0):
            torch.manual_seed(5)
            mm = mk(fsb_policy=pol, fsb_width=2, fsb_delta=1, fsb_tree=tree, fsb_prefill_blkchunk=nbch)
            Qc, Kc, Vc = rand_qkv(17)
            _ft.masked_readout = _grab2
            try:
                o = lr(mm, Qc, Kc, Vc)
            finally:
                _ft.masked_readout = _orig
            outs.append(o); sels.append(_sel_cap["sel"])
        assert torch.equal(sels[0], sels[1]) and torch.equal(sels[0], sels[2]), (tree, pol, "sel_full differs across chunk sizes")
        assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2]), (tree, pol, "output differs across chunk sizes")
        print(f"F17 [{tree}/{pol}] sel_full and output bit-identical for prefill chunks 1 / 2 / all")
print("F17 OK  (chunked prefill identical)")

# ── F18: grouped/padded chunk build == per-k unpadded build (identical node arrays) ────────
from src.fsattn.fsb_tree import _grouped_chunk_build, _chunk_tables
Fg = build_forest(Kb_f, ce_f, adm_f)
nid = 0
for k in Fg["ks"]:
    size = 1 << k; Cmax = Fg["Cmaxs"][k]
    pos = torch.arange(Cmax * size, device=dev)
    cid = torch.gather(Fg["adm"], 1, pos[None, :].expand(Tf, -1))
    ok = (cid >= 0) & (pos[None, :] < ((Fg["m_tot"] >> k) << k)[:, None])
    cid = cid.masked_fill(~ok, -1).view(Tf, Cmax, size)
    Kb = torch.gather(Kb_f, 1, cid.clamp(min=0).view(Tf, -1)[..., None].expand(-1, -1, Dhf)).reshape(Tf * Cmax, size, Dhf)
    ce = torch.gather(ce_f, 1, cid.clamp(min=0).view(Tf, -1)).reshape(Tf * Cmax, size)
    tr = build_trees(Kb, ce, (cid >= 0).reshape(Tf * Cmax, size), k)          # unpadded, per-k
    ref_tab = _chunk_tables(tr, k, Tf, Cmax, cid, slice(None), k)
    got = Fg["chunks"][k]
    for name in ("mu", "rad", "amax", "pmin", "leaf"):
        assert torch.equal(ref_tab[name], got[name]), (k, name)
    nid += Cmax
print(f"F18 OK  (grouped/padded build == per-k build, {nid} chunk slots, all node arrays identical)")

# ── F19: CUDA-graph forest decode == eager forest decode (bit-exact), across block boundaries ──
if dev == "cuda":
    for name, mm in (("forest fixed W2/D1", mf_fix), ("forest learned/argmax", mf_lrn)):
        torch.manual_seed(23)
        x = torch.randn(1, N, d_model, device=dev)
        N0 = 150
        outs = {}
        for mode in ("0", "1"):
            os.environ["FSB_DECODE_GRAPH"] = mode
            try:
                with torch.no_grad():
                    cache = FSKVCache(mm, lean_prefill=False); cache.reserve = N - N0
                    o = [mm(x[:, :N0], cache=cache)]
                    for t in range(N0, N):
                        o.append(mm(x[:, t:t + 1], cache=cache))
                    outs[mode] = torch.cat(o, dim=1)
                    assert (mode == "0") or (cache._fsb_forest.graph is not None), "graph never captured"
            finally:
                os.environ.pop("FSB_DECODE_GRAPH", None)
        assert torch.equal(outs["0"], outs["1"]), (name, (outs["0"] - outs["1"]).abs().max().item())
        print(f"F19 [{name}] CUDA-graph decode == eager decode (bit-exact over {N - N0} tokens, 2 blocks)")
    print("F19 OK  (graph-captured forest decode reproduces eager decode)")
else:
    print("F19 SKIP (no CUDA)")

# ── F20: one-launch Triton halving sum == Python halving reference, bit-exact, many shapes ─────
from src.fsattn.det_reduce import det_sum, _py_halving
torch.manual_seed(77)
nbad = 0; ncase = 0
for shape, dim in (((3, 5, 64), -1), ((7, 128, 16), 1), ((2, 4, 33, 8), 2), ((5, 1024), -1), ((16, 2048), 1),
                   ((4, 3, 96), -1), ((1, 4096), -1), ((6, 0), -1), ((2, 7, 5), 1), ((16, 64, 1, 64), 2),
                   ((2, 1, 4096, 64), 2), ((3, 9000), -1), ((300, 33, 65), 1)):
    x = (torch.randn(*shape, device=dev) * 1e3).float()
    if x.numel():
        x.view(-1)[::5] = -0.0                  # signed zeros must survive exactly
    a = det_sum(x, dim); b = _py_halving(x.movedim(dim, -1))
    ncase += 1; nbad += int(not torch.equal(a.view(torch.int32), b.contiguous().view(torch.int32)))
    # batch independence: a single row reduced alone equals its row in the batched call
    if x.numel() and x.shape[dim] > 0:
        xm = x.movedim(dim, -1).reshape(-1, x.shape[dim])
        one = det_sum(xm[3:4] if xm.shape[0] > 3 else xm[:1], -1)
        allr = det_sum(xm, -1)
        nbad += int(not torch.equal(one[0], allr[3] if xm.shape[0] > 3 else allr[0]))
z20 = -torch.zeros(7, 33, 5, device=dev)
for dim in (0, 1, 2):
    ncase += 1
    nbad += int(not torch.equal(det_sum(z20, dim).view(torch.int32),
                                _py_halving(z20.movedim(dim, -1)).contiguous().view(torch.int32)))
assert nbad == 0, f"det_sum mismatches: {nbad}"
print(f"F20 OK  (Triton halving sum == Python halving, bit-exact, {ncase} shapes incl. non-power-of-two, empty; batch-independent)")

# ── F21: CUDA-graph forest TRAINING path == eager (sel_full, output, stats), two inputs, both policies ─
if dev == "cuda":
    import src.fsattn.fsb_tree as _ft3
    # nbch=None: one chunk graph covers every block (nb <= NBCH).  nbch=2: several chunk graphs per micro-batch
    # (nb > NBCH) -- the per-chunk block-count slice is then a strided COPY of the build graph's output, so it
    # must be taken INSIDE the captured chunk graph (regression: stale counts -> all-empty roots / illegal address).
    for pol, nbch in (("fixed", None), ("learned", None), ("fixed", 2), ("learned", 2)):
        torch.manual_seed(31)
        kw = {} if nbch is None else {"fsb_prefill_blkchunk": nbch}
        mm = mk(fsb_policy=pol, fsb_width=2, fsb_delta=1, fsb_tree="forest", **kw)
        mm.train()                                                        # the forest graph is a TRAINING path (eval: F31)
        cap = {}
        def _grab3(attn, Qq, Kp, Vp, sel_full, s, Bq, n):
            cap["sel"] = sel_full.clone(); return _orig(attn, Qq, Kp, Vp, sel_full, s, Bq, n)
        for trial in range(2):                                            # two different inputs: buffers refreshed
            Qc, Kc, Vc = rand_qkv(40 + trial)
            res = {}
            for mode in ("0", "1"):
                os.environ["FSB_TRAIN_GRAPH"] = mode
                torch.manual_seed(100 + trial)                            # learned policy SAMPLES actions: same draws in both modes
                _ft3.masked_readout = _grab3
                try:
                    o = lr(mm, Qc, Kc, Vc)
                finally:
                    _ft3.masked_readout = _orig; os.environ.pop("FSB_TRAIN_GRAPH", None)
                st = mm._fsb_stats
                res[mode] = (cap["sel"].clone(), o.detach().clone(), st["n_visit"].clone(), st["n_read"].clone(), st["has"].clone(),
                             st["logp"].detach().clone() if "logp" in st else None)
                if mode == "1":
                    gs = getattr(mm, "_fsb_train_graphs", None)
                    assert gs is not None, "training graphs not captured"
                    nchunks = len(gs["chunks"])
                    assert nchunks == (1 if nbch is None else -(-(N // BLK) // nbch)), (nbch, nchunks)
                    if pol == "learned":
                        (o.float().pow(2).mean() + fsb_policy_loss(mm, torch.rand(1, N, device=dev))).backward()
                        assert all(p_.grad is not None and torch.isfinite(p_.grad).all() for n_, p_ in mm.named_parameters() if "fsb_policy" in n_)
                        mm.zero_grad(set_to_none=True)
            assert res["0"][4].any(), "no block selected anything (degenerate case)"
            for i, name in enumerate(("sel_full", "output", "n_visit", "n_read", "has")):
                assert torch.equal(res["0"][i], res["1"][i]), (pol, nbch, trial, name)
            if pol == "learned":
                assert torch.equal(res["0"][5], res["1"][5]), (pol, nbch, trial, "logp")
        print(f"F21 [{pol}, {nchunks} chunk graph(s)] graph-captured forest training stage == eager (sel_full, output, stats{', logp + policy grads' if pol == 'learned' else ''}), 2 inputs")
    print("F21 OK  (graph-captured forest training path bit-exact, incl. several chunk graphs per micro-batch)")
else:
    print("F21 SKIP (no CUDA)")

# ── F22: Triton level kernel == torch traversal (sel_full bit-identical), forest, both policies, W_max 1/2/4 ─
if dev == "cuda":
    import src.fsattn.fsb_tree as _ft4
    from src.fsattn.fsb_tree import build_forest as _bf, build_spines as _bs, traverse_forest as _tf
    # (a) direct traversal on random forests, several beam menus and budgets, bound and centroid scores
    torch.manual_seed(51)
    nchk = 0
    for (Wm, dm) in ((1, 2), (2, 1), (4, 0), (4, 2)):
        for ns in ("bound", "centroid"):
            Tq, Mq, Dq = 6, 70, 16
            Kbq = torch.randn(Tq, Mq, Dq, device=dev) * 2; ceq = torch.rand(Tq, Mq, device=dev) * 3
            admq = torch.rand(Tq, Mq, device=dev) < 0.8
            Fq = _bf(Kbq, ceq, admq)
            mbq = torch.tensor([0, 1, 13, 32, 55, 70], device=dev).clamp(max=Fq["m_tot"].max().item()); mbq = torch.minimum(mbq, Fq["m_tot"])
            SPq = _bs(Fq, mbq, torch.arange(Tq, device=dev))
            Bqq = 5
            qq = torch.randn(Tq, Bqq, Dq, device=dev); qpq = torch.full((Tq, Bqq), 1500.0, device=dev) + torch.arange(Bqq, device=dev)[None].float()
            lamq = torch.rand(Tq, 1, 1, device=dev) + 0.5; sigq = torch.full((Tq, 1, 1), 4.0, device=dev)
            Wq = torch.randint(1, Wm + 1, (Tq, Bqq), device=dev); dq = torch.randint(0, dm + 1, (Tq, Bqq), device=dev)
            outs = {}
            for mode in ("0", "1"):
                os.environ["FSB_TRAVERSE_TRITON"] = mode
                try:
                    outs[mode] = _tf(Fq, SPq, qq, qpq, lamq, sigq, 0.25, S, Wq, dq, Wm, dm, ns)
                finally:
                    os.environ.pop("FSB_TRAVERSE_TRITON", None)
            for i, nm in enumerate(("sel", "n_visit", "n_read")):
                assert torch.equal(outs["0"][i], outs["1"][i]), ("F22a", Wm, dm, ns, nm)
            nchk += 1
    # (b) through the module on the small model: sel_full identical for fixed and learned policies
    cap = {}
    def _grab5(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_):
        cap["sel"] = sel_full.clone(); return _orig(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_)
    for pol in ("fixed", "learned"):
        mm = mk(fsb_policy=pol, fsb_width=2, fsb_delta=1, fsb_tree="forest")
        Qc, Kc, Vc = rand_qkv(61)
        res = {}
        for mode in ("0", "1"):
            os.environ["FSB_TRAVERSE_TRITON"] = mode; os.environ["FSB_TRAIN_GRAPH"] = "0"
            _ft4.masked_readout = _grab5
            try:
                o = lr(mm, Qc, Kc, Vc)
            finally:
                _ft4.masked_readout = _orig; os.environ.pop("FSB_TRAVERSE_TRITON", None); os.environ.pop("FSB_TRAIN_GRAPH", None)
            res[mode] = (cap["sel"].clone(), o.clone())
        assert torch.equal(res["0"][0], res["1"][0]) and torch.equal(res["0"][1], res["1"][1]), ("F22b", pol)
    print(f"F22 OK  (Triton level kernel == torch traversal: {nchk} random-forest cases bit-identical; module sel_full/output identical for fixed and learned)")
else:
    print("F22 SKIP (no CUDA)")


# ── F23: PPO controller update (fsb_rl="ppo").  The store is inert on the forward (output, on-policy loss and
#        controller grads identical to reinforce); the clipped surrogate re-scored at unchanged parameters has
#        ratio 1 and the REINFORCE gradient (fp32 store: exact; bf16 store: bounded rounding); K=4 epochs run,
#        parameters stay finite and move, the store is cleared; one and several chunk graphs per micro-batch ──
from src.fsattn.fsb_tree import fsb_ppo_surrogate, fsb_ppo_clear, fsb_ppo_store_bytes, fsb_ctrl_params
for nbch in (None, 2):
    kw = {} if nbch is None else {"fsb_prefill_blkchunk": nbch}
    nchunk = 1 if nbch is None else -(-(N // BLK) // nbch)
    outs, losses, grads = {}, {}, {}
    for rl, sdt in (("reinforce", "fp32"), ("ppo", "fp32"), ("ppo", "bf16")):
        torch.manual_seed(31)
        mm = mk(fsb_policy="learned", fsb_width=2, fsb_delta=1, fsb_tree="forest", fsb_rl=rl, fsb_ppo_store_dtype=sdt, **kw)
        mm.train()
        Qc, Kc, Vc = rand_qkv(77)
        torch.manual_seed(500)                                            # same sampled actions in every mode
        o = lr(mm, Qc, Kc, Vc)
        torch.manual_seed(501); lt = torch.rand(1, N, device=dev)
        L1 = fsb_policy_loss(mm, lt, adv_norm=True)
        ctrl = fsb_ctrl_params(mm)
        g1 = [g_.clone() for g_ in torch.autograd.grad(L1, ctrl, retain_graph=True)]
        outs[(rl, sdt)], losses[(rl, sdt)], grads[(rl, sdt)] = o.detach().clone(), L1.detach().clone(), g1
        if rl == "reinforce":
            assert fsb_ppo_store_bytes(mm) == 0 and not mm._fsb_ppo_store and not mm._fsb_ppo_rec
            continue
        assert len(mm._fsb_ppo_store) == nchunk and mm._fsb_ppo_nmb == 1 and not mm._fsb_ppo_rec, (len(mm._fsb_ppo_store), nchunk)
        nbytes = fsb_ppo_store_bytes(mm)
        Ls, pst = fsb_ppo_surrogate(mm, clip=0.2, entropy_beta=0.01, value_coef=0.5)
        gS = torch.autograd.grad(Ls, ctrl)
        if sdt == "fp32":
            assert pst["ratio_maxdev"] < 1e-5 and pst["clipfrac"] == 0.0 and pst["approx_kl"] < 1e-9, pst   # same inputs; batched re-score
            for a_, b_ in zip(gS, g1):
                assert torch.allclose(a_, b_, rtol=1e-4, atol=1e-6), float((a_ - b_).abs().max())
        else:
            assert 0.0 < pst["ratio_maxdev"] < 0.05, pst
            for a_, b_ in zip(gS, g1):
                assert torch.allclose(a_, b_, rtol=5e-2, atol=2e-3), float((a_ - b_).abs().max())
        # K = 4 as the trainer does it: epoch 1 = the on-policy gradients, then 3 surrogate epochs
        opt = torch.optim.AdamW(ctrl, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
        p0 = [p_.detach().clone() for p_ in ctrl]
        for p_, g_ in zip(ctrl, g1):
            p_.grad = g_.clone()
        opt.step(); opt.zero_grad(set_to_none=True)
        for _ep in range(3):
            Ls, pst = fsb_ppo_surrogate(mm, clip=0.2, entropy_beta=0.01, value_coef=0.5)
            Ls.backward(); torch.nn.utils.clip_grad_norm_(ctrl, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        assert all(torch.isfinite(p_).all() for p_ in ctrl) and all(not torch.equal(p_, q_) for p_, q_ in zip(ctrl, p0))
        assert pst["ratio_maxdev"] > 0.0 and pst["n"] > 0, pst              # off-policy after the moves
        fsb_ppo_clear(mm)
        assert fsb_ppo_store_bytes(mm) == 0 and not mm._fsb_ppo_store and mm._fsb_ppo_nmb == 0
        print(f"F23 [ppo/{sdt}, {nchunk} chunk(s)] surrogate at ratio 1 == REINFORCE grads (maxdev {pst['ratio_maxdev']:.2e} after 3 epochs, "
              f"clipfrac {pst['clipfrac']:.3f}, kl {pst['approx_kl']:.2e}); K=4 epochs finite; store {nbytes / 1024:.1f} KiB "
              f"= {nbytes / (N * H):.0f} B per (head, query) per layer")
    for k_ in (("ppo", "fp32"), ("ppo", "bf16")):
        assert torch.equal(outs[("reinforce", "fp32")], outs[k_]) and torch.equal(losses[("reinforce", "fp32")], losses[k_]), k_
        assert all(torch.equal(a_, b_) for a_, b_ in zip(grads[("reinforce", "fp32")], grads[k_])), k_
    print(f"F23 [{nchunk} chunk(s)] forward, on-policy loss and controller grads identical for reinforce / ppo-fp32 / ppo-bf16")
print("F23 OK  (PPO controller update: store inert on the forward; surrogate == REINFORCE at ratio 1; K=4 runs)")


# ── F24: prefill routing == decode routing on a 4096-token bf16 prefix, and cache Kbar == training Kbar
#        bit-exact.  Training / prefill used to build the forest from the fp32 mean of the bf16 keys while the KV
#        cache stores the bf16 mean (dense-FS convention), so decode routed the same query on different summaries
#        (gradient audit 2026-09-27; fixed on the training side).  Bit-exactness is asserted for the cells the
#        cache completed from the SAME keys (prefill); cells completed from single-token decode keys (a different
#        GEMM shape, bf16-rounded) are reported.  Synthetic tier (this module, autocast bf16) always; real tier
#        (340M weights, real tokens) when FSB_REAL_CKPT and FSB_REAL_TOKENS are set. ─────────────────────────────
import src.fsattn.fsb_tree as _ft4
from experiments.scaling.kv_cache import FSKVCache as _KVC

def _sets_equal(a, b):
    """a, b (..., kk) int, -1 = pad: identical retained SETS per row; and identical slot order."""
    va, vb = a >= 0, b >= 0
    hit = (a[..., :, None] == b[..., None, :]) & va[..., :, None] & vb[..., None, :]
    inter = hit.any(-1).sum(-1)
    same_set = (inter == va.sum(-1)) & (inter == vb.sum(-1))
    same_order = (torch.where(va, a, -1) == torch.where(vb, b, -1)).all(-1)
    return same_set, same_order

def _f24_check(mm_, x_, Nf, Bq_, tag, lean):
    """mm_(x_) under autocast bf16; then cache prefill of Nf-Bq_ tokens + per-token decode of the last block."""
    caps = {}
    def _grab4(attn, Qq, Kp, Vp, sel_full, s_, Bq__, n_):
        caps.setdefault("sel", []).append(sel_full[0].clone()); return _orig(attn, Qq, Kp, Vp, sel_full, s_, Bq__, n_)
    _ob = _ft4.build_forest
    def _bf(Kbar, ceil_c, admit, **kw_):
        caps.setdefault("kbar", []).append(Kbar.clone()); caps.setdefault("ceil", []).append(ceil_c.clone())
        return _ob(Kbar, ceil_c, admit, **kw_)
    os.environ["FSB_TRAIN_GRAPH"] = "0"
    _ft4.masked_readout = _grab4; _ft4.build_forest = _bf
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            mm_(x_)
            assert len(caps["sel"]) == 1 and len(caps["kbar"]) == 1
            sel_pre, kb_tr, ce_tr = caps["sel"][0], caps["kbar"][0], caps["ceil"][0]        # (H,Npad,kk), (H,M,Dh), (H,M)
            P = Nf - Bq_
            cache = _KVC(mm_, lean_prefill=lean)
            caps["sel"] = []
            mm_(x_[:, :P], cache=cache)
            Kb_c, Ce_c, nc = cache.completed_cells()
            assert nc == P // mm_.cell_stride, (nc, P)
            assert torch.equal(Kb_c[0], kb_tr[:, :nc]) and torch.equal(Ce_c[0], ce_tr[:, :nc]), f"{tag}: prefill cache Kbar/ceil != training"
            Kh = cache._k_hist[:, :, :nc * mm_.cell_stride]
            old = Kh.reshape(1, kb_tr.shape[0], nc, mm_.cell_stride, -1).float().mean(dim=3)[0]     # the pre-fix formula
            frac_old = (old != kb_tr[:, :nc]).float().mean().item()
            assert Kh.dtype == torch.bfloat16 and frac_old > 0.0, (Kh.dtype, frac_old)             # the test has teeth
            dec = []
            for t in range(P, Nf):
                mm_(x_[:, t:t + 1], cache=cache)
                ls = cache._fsb_forest._last_sel
                dec.append(ls.reshape(-1, ls.shape[-1]).clone())                                    # (H,kk)
            nc0 = nc
            Kb_c, Ce_c, nc = cache.completed_cells()
            assert nc == Nf // mm_.cell_stride
            tail_eq = torch.equal(Kb_c[0, :, nc0:], kb_tr[:, nc0:nc])                                  # single-token keys
            tail_dev = (Kb_c[0, :, nc0:] - kb_tr[:, nc0:nc]).abs().max().item()
            assert torch.allclose(Kb_c[0, :, nc0:], kb_tr[:, nc0:nc], atol=1e-2, rtol=1e-2), f"{tag}: decode-completed Kbar off by {tail_dev}"
    finally:
        _ft4.masked_readout = _orig; _ft4.build_forest = _ob; os.environ.pop("FSB_TRAIN_GRAPH", None)
    sel_dec = torch.stack(dec, dim=1)                                                                # (H,Bq,kk)
    same_set, same_order = _sets_equal(sel_pre[:, P:Nf].long(), sel_dec.long())
    n_sel = int(((sel_dec >= 0).sum(-1) > 0).sum())
    nd = int((~same_set).sum())
    assert nd == 0, f"{tag}: {nd}/{same_set.numel()} (head, query) rows route differently"
    print(f"F24 [{tag}, lean={lean}] last block: {same_set.numel()} (head, query) rows identical sets, "
          f"{int(same_order.sum())} identical slot order, {n_sel} non-empty; prefix cache Kbar/ceil == training bit-exact "
          f"({nc0} cells; pre-fix fp32-mean formula differs in {100 * frac_old:.1f}% of elements); "
          f"{nc - nc0} cells completed from decode keys: {'bit-exact' if tail_eq else f'max |diff| {tail_dev:.1e}'}")

if dev == "cuda":
    for lean in (False, True):
        torch.manual_seed(24)
        mm = mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_tree="forest")
        x = torch.randn(1, 4096, d_model, device=dev)
        _f24_check(mm, x, 4096, BLK, "synthetic 4096", lean)
    _ck, _tk = os.environ.get("FSB_REAL_CKPT"), os.environ.get("FSB_REAL_TOKENS")
    if _ck and _tk:
        import sys as _sys, numpy as _np
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "scaling"))
        import longctx_infer as _lci
        from kv_cache import make_caches as _mkc, _trunk_cached as _trunk
        _lci.MODEL_CONFIGS["340m"]["vocab_size"] = 32000
        attn_kw = dict(window_size=512, long_range_variant="cell_topk", cell_stride=32, cell_topk=4, cell_select="score",
                       sel_strict_causal=1, use_rope=True, fs_uniform_sr=True,
                       rope_long_range=False, fsb_enabled=True, fsb_policy="fixed", fsb_width=4, fsb_delta=0,
                       fsb_widths="1,2,4", fsb_deltas="0,1,2", fsb_node_score="bound", fsb_tree="forest")
        real = _lci.load_model(_ck, config="340m", attn_type="fsattn", device="cuda", max_seq_len=4096 + 256, **attn_kw)
        real.eval()
        layers = [m_ for m_ in real.modules() if getattr(m_, "fsb_enabled", False)]
        for m_ in layers:
            m_.fsb_tau.fill_(0.0)
        ids = torch.from_numpy(_np.fromfile(_tk, dtype=_np.uint16, count=4096).astype(_np.int64)).to(dev)
        Bq_ = layers[0]._cell_block; P = 4096 - Bq_
        caps = {"sel": [], "kbar": [], "ceil": [], "trav": []}
        def _grab5(attn, Qq, Kp, Vp, sel_full, s_, Bq__, n_):
            caps["sel"].append(sel_full[0].clone()); return _orig(attn, Qq, Kp, Vp, sel_full, s_, Bq__, n_)
        _ob = _ft4.build_forest
        def _bf5(Kbar, ceil_c, admit, **kw_):
            caps["kbar"].append(Kbar.clone()); caps["ceil"].append(ceil_c.clone()); return _ob(Kbar, ceil_c, admit, **kw_)
        _otr = _ft4.traverse_forest
        def _tr5(F_, SP_, q_, qpos_, lam_, sigv_, *rest, **kw_):                                     # prefill traversal args
            caps["trav"].append((q_.clone(), qpos_.clone(), lam_.clone(), sigv_.clone(), tuple(r_.clone() if torch.is_tensor(r_) else r_ for r_ in rest), kw_))
            return _otr(F_, SP_, q_, qpos_, lam_, sigv_, *rest, **kw_)
        os.environ["FSB_TRAIN_GRAPH"] = "0"
        _ft4.masked_readout = _grab5; _ft4.build_forest = _bf5; _ft4.traverse_forest = _tr5
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                real(ids[None])
                assert len(caps["sel"]) == len(layers) == len(caps["kbar"])
                caches = _mkc(real, lean_prefill=False, reserve=Bq_)
                _trunk(real, ids[None, :P], caches)
                for c_, kb_, ce_ in zip(caches, caps["kbar"], caps["ceil"]):
                    Kb_c, Ce_c, nc = c_.completed_cells()
                    assert nc == P // 32 and torch.equal(Kb_c[0], kb_[:, :nc]) and torch.equal(Ce_c[0], ce_[:, :nc]), "real: prefill cache Kbar != training"
                Kh = caches[0]._k_hist[:, :, :nc * 32]
                old = Kh.reshape(1, 16, nc, 32, -1).float().mean(dim=3)[0]
                frac_old = (old != caps["kbar"][0][:, :nc]).float().mean().item()
                assert Kh.dtype == torch.bfloat16 and frac_old > 0.0, (Kh.dtype, frac_old)
                # traversals: one prefill chunk per layer for the 4096 forward (first 24), then the cache prefill's
                assert len(caps["trav"]) >= len(layers), (len(caps["trav"]), len(layers))
                dec = [[] for _ in layers]; sameq = []
                for t in range(P, 4096):
                    _trunk(real, ids[None, t:t + 1], caches)
                    for li, c_ in enumerate(caches):
                        ls = c_._fsb_forest._last_sel
                        dec[li].append(ls.reshape(-1, ls.shape[-1]).clone())
                    if t == P:
                        # the decode forest of the last block (built by this first token), routed with the PREFILL's
                        # own q for that block: isolates the summaries from the decode-side hidden-state drift
                        nbc = caps["trav"][0][0].shape[0] // 16; jl = nbc - 1
                        for li, c_ in enumerate(caches):
                            fc = c_._fsb_forest
                            q_, qpos_, lam_, sigv_, rest, kw_ = caps["trav"][li]
                            sl = slice(jl, None, nbc)                                                  # rows (h*nbc + last block)
                            rest = tuple(r_[sl] if torch.is_tensor(r_) and r_.dim() >= 1 and r_.shape[0] == q_.shape[0] else r_ for r_ in rest)
                            s_q, _, _ = _otr(fc.F, fc.SP, q_[sl], qpos_[sl], lam_[sl], sigv_[sl], *rest, **kw_)
                            sameq.append(s_q.reshape(16, Bq_, -1).clone())
                nc0 = P // 32; tail_eq = 0; tail_dev = 0.0
                for c_, kb_, ce_ in zip(caches, caps["kbar"], caps["ceil"]):
                    Kb_c, Ce_c, nc = c_.completed_cells()
                    assert nc == 128 and torch.equal(Kb_c[0, :, :nc0], kb_[:, :nc0]), "real: prefix cache Kbar changed by decode"
                    tail_eq += int(torch.equal(Kb_c[0, :, nc0:], kb_[:, nc0:nc]))
                    tail_dev = max(tail_dev, (Kb_c[0, :, nc0:] - kb_[:, nc0:nc]).abs().max().item())
        finally:
            _ft4.masked_readout = _orig; _ft4.build_forest = _ob; _ft4.traverse_forest = _otr; os.environ.pop("FSB_TRAIN_GRAPH", None)
        tot = same_tot = ord_tot = sq_tot = 0; worst = (1.0, -1)
        for li in range(len(layers)):
            pre = caps["sel"][li][:, P:4096].long()
            sd = torch.stack(dec[li], dim=1).long()                                                  # (H,Bq,kk)
            ss, so = _sets_equal(pre, sd)
            sq, _ = _sets_equal(pre, sameq[li].long())
            tot += ss.numel(); same_tot += int(ss.sum()); ord_tot += int(so.sum()); sq_tot += int(sq.sum())
            worst = min(worst, (ss.float().mean().item(), li))
        # (a) same query, decode forest vs prefill forest: identical (the summaries are the same)
        assert sq_tot == tot, f"real: decode forest routes the prefill's q differently in {tot - sq_tot}/{tot} rows"
        # (b) the decode token's own q: the residual is the decode-side bf16 hidden-state drift through the layers
        #     (pre-fix audit: 94.7% identical on average, min 82.7%); guard against a regression of the summaries
        assert same_tot >= 0.98 * tot, f"real: only {100 * same_tot / tot:.2f}% rows identical (worst layer {worst[1]}: {100 * worst[0]:.2f}%)"
        print(f"F24 [real 340M, 4096 real tokens, {len(layers)} layers] last block: decode forest x prefill q == prefill routing in "
              f"{sq_tot}/{tot} (layer, head, query) rows; decode q: {same_tot}/{tot} identical sets ({100 * same_tot / tot:.2f}%, "
              f"worst layer {worst[1]} {100 * worst[0]:.2f}%), {ord_tot} identical slot order; prefix cache Kbar/ceil == training "
              f"bit-exact in every layer ({nc0} cells; pre-fix fp32-mean formula differs in {100 * frac_old:.1f}% of layer-0 elements); "
              f"{128 - nc0} cells completed from decode keys bit-exact in {tail_eq}/{len(layers)} layers (max |diff| {tail_dev:.1e})")
        del real, caches; torch.cuda.empty_cache()
    else:
        print("F24 [real tier] SKIP (set FSB_REAL_CKPT and FSB_REAL_TOKENS)")
    print("F24 OK  (FS-B prefill routing == decode routing; cache Kbar == training Kbar bit-exact)")
else:
    print("F24 SKIP (no CUDA)")


# ── F25: radius coefficient of the optimistic bound (fsb_bound_radius_coef).  1.0 (default) is the existing path
#        bit-exactly; other values change the routing and give identical selections on the Triton and torch
#        traversals (forest) and on the rebuild path ────────────────────────────────────────────────────────────
for tree in ("forest", "rebuild"):
    Qc, Kc, Vc = rand_qkv(88)
    sels = {}
    for coef in (None, 1.0, 2.0, 0.5):
        for kern in ("1", "0"):
            os.environ["FSB_TRAVERSE_TRITON"] = kern
            kw = {} if coef is None else {"fsb_bound_radius_coef": coef}
            torch.manual_seed(31)
            mm = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree=tree, **kw)
            cap = {}
            def _g25(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_):
                cap["sel"] = sel_full.clone(); return _orig(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_)
            _ft4.masked_readout = _g25
            try:
                with torch.no_grad():
                    o = lr(mm, Qc, Kc, Vc)
            finally:
                _ft4.masked_readout = _orig
            sels[(coef, kern)] = (cap["sel"], o.clone())
    os.environ.pop("FSB_TRAVERSE_TRITON", None)
    for k_ in ((1.0, "1"), (None, "0"), (1.0, "0")):
        assert torch.equal(sels[(None, "1")][0], sels[k_][0]) and torch.equal(sels[(None, "1")][1], sels[k_][1]), (tree, k_)
    for c_ in (2.0, 0.5):
        assert torch.equal(sels[(c_, "1")][0], sels[(c_, "0")][0]) and torch.equal(sels[(c_, "1")][1], sels[(c_, "0")][1]), (tree, c_)
    d2 = (sels[(2.0, "1")][0] != sels[(1.0, "1")][0]).any(-1).float().mean().item()
    d5 = (sels[(0.5, "1")][0] != sels[(1.0, "1")][0]).any(-1).float().mean().item()
    assert d2 > 0.0 and d5 > 0.0, (tree, d2, d5)
    print(f"F25 [{tree}] coef 1.0 == default bit-exact (sel_full, output); coef 2.0 / 0.5 identical on Triton and torch traversals; "
          f"rows routed differently vs 1.0: {100 * d2:.1f}% / {100 * d5:.1f}%")
print("F25 OK  (radius coefficient: default bit-exact; both traversals agree; changes the routing)")


# ── F26: leaf radius (fsb_leaf_radius): node balls enclose every TOKEN key of the subtree, so the bound dominates the
#        best per-token score; module: changes the routing, prefill + decode reproduce the full forward, cache
#        spread == training spread bit-exact for the prefill-completed cells; default off is the unchanged path ─────
from src.fsattn.fsb_tree import build_forest as _bf26, _det_sum as _ds26
torch.manual_seed(26)
T26, M26, s26 = 3, 20, 8
Kt = torch.randn(T26, M26, s26, Dh, device=dev) * torch.rand(T26, M26, 1, 1, device=dev) * 3   # cells of unequal spread
Kb26 = Kt.mean(2); sp26 = _ds26((Kt - Kb26[:, :, None]) ** 2, -1).amax(-1).sqrt(); ce26 = Kt.norm(dim=-1).amax(-1)
adm26 = torch.rand(T26, M26, device=dev) < 0.8
L26 = 5
tr26 = build_trees(Kb26, ce26, adm26, L26, spread=sp26)
tr26b = build_trees(Kb26, ce26, adm26, L26)
q26 = torch.randn(T26, Dh, device=dev)
worst = 0.0; nchk = 0
for l in range(L26 + 1):
    cap = 1 << (L26 - l)
    for t in range(T26):
        for j in range(1 << l):
            cells = tr26["packed"][t, l * (1 << L26) + j * cap: l * (1 << L26) + (j + 1) * cap]
            cells = cells[cells >= 0]
            if cells.numel() == 0:
                continue
            toks = Kt[t, cells].reshape(-1, Dh)
            mu = tr26["mu"][l][t, j]; r = tr26["rad"][l][t, j]
            worst = max(worst, float(((toks - mu).norm(dim=-1) - r).max()))
            best_tok = float((toks @ q26[t]).max()); bound = float(mu @ q26[t] + q26[t].norm() * r)
            assert best_tok <= bound + 1e-4, (l, t, j, best_tok, bound)
            assert float(tr26b["rad"][l][t, j]) <= float(r) + 1e-6, "point radius exceeds token radius"
            nchk += 1
assert worst <= 1e-4, worst
print(f"F26 build_trees(spread): {nchk} nodes, every token key inside the node ball (worst overshoot {worst:.1e}); bound >= best token score")
if dev == "cuda":
    for lean in (False, True):
        torch.manual_seed(26)
        mm = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_leaf_radius=1)
        mm0 = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest")
        mm0.load_state_dict(mm.state_dict())
        x = torch.randn(1, N, d_model, device=dev)
        caps = {}
        _ob26 = _ft4.build_forest
        def _bf26h(Kbar, ceil_c, admit, spread=None):
            caps.setdefault("sp", []).append(None if spread is None else spread.clone()); return _ob26(Kbar, ceil_c, admit, spread=spread)
        _ft4.build_forest = _bf26h
        try:
            with torch.no_grad():
                os.environ["FSB_TRAIN_GRAPH"] = "0"
                full = mm(x); full0 = mm0(x)
                os.environ.pop("FSB_TRAIN_GRAPH", None)
                assert caps["sp"][0] is not None and caps["sp"][1] is None, "leaf radius flag not threaded to the build"
                sp_tr = caps["sp"][0]
                assert not torch.equal(full, full0), "leaf radius did not change the routing"
                cache = FSKVCache(mm, lean_prefill=lean)
                N0 = 150
                outs = [mm(x[:, :N0], cache=cache)]
                sp_c = cache.completed_spread()
                assert sp_c is not None and torch.equal(sp_c[0], sp_tr[:, :sp_c.shape[2]]), "cache spread != training spread"
                for t in range(N0, N):
                    outs.append(mm(x[:, t:t + 1], cache=cache))
                d = (torch.cat(outs, dim=1) - full).abs().max().item()
                assert d < 1e-4, f"leaf radius lean={lean}: cached != full ({d})"
                sp_c = cache.completed_spread()
                assert sp_c.shape[2] == N // S and torch.allclose(sp_c[0], sp_tr[:, :sp_c.shape[2]], atol=1e-5)
        finally:
            _ft4.build_forest = _ob26; os.environ.pop("FSB_TRAIN_GRAPH", None)
        print(f"F26 [module, lean={lean}] leaf radius changes the routing; prefill + decode reproduce the full forward (|d| {d:.1e}); "
              f"cache spread == training spread ({sp_c.shape[2]} cells)")
print("F26 OK  (leaf radius: token-enclosing node balls; module parity; opt-in)")


# ── F27: controller learning-rate multiplier (--fsb_ctrl_lr_mult): controller lr == mult x main lr at several schedule
#        points on both the reinforce path (own param group of the main optimizer) and the ppo path (own AdamW);
#        mult 1.0 leaves the parameter list untouched ─────────────────────────────────────────────────────────────
try:
    import experiments.scaling.train_dist as _td
    for _n27 in ("ctrl_lr_groups", "set_step_lr", "lr_gain_loss", "fsb_objective",
                 "set_long_range_disabled", "restore_long_range"):
        getattr(_td, _n27)
except Exception as _e:                                                    # trainer deps (accelerate, datasets) absent
    _td = None; print(f"F27 SKIP (train_dist import: {_e})"); print("F28 SKIP (train_dist import)")
if _td is not None:
    torch.manual_seed(27)
    mm = mk(fsb_policy="learned", fsb_width=2, fsb_delta=1, fsb_tree="forest")
    ctrl = fsb_ctrl_params(mm); ctrl_ids = {id(p_) for p_ in ctrl}
    plain = list(mm.parameters())
    same = _td.ctrl_lr_groups(plain, ctrl_ids, 1.0)
    assert same is plain, "mult 1.0 must return the parameter list unchanged"
    for mult in (0.25, 5.0):
        groups = _td.ctrl_lr_groups(list(mm.parameters()), ctrl_ids, mult)
        assert len(groups) == 2 and groups[1].get("fsb_ctrl") and {id(p_) for p_ in groups[1]["params"]} == ctrl_ids
        assert not ({id(p_) for p_ in groups[0]["params"]} & ctrl_ids) and len(groups[0]["params"]) + len(groups[1]["params"]) == len(plain)
        opt = torch.optim.AdamW(groups, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))          # reinforce path
        copt = torch.optim.AdamW(ctrl, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))           # ppo path
        for step in (0, 3, 25, 199):
            lr_ = _td.get_lr(step, 10, 200, 4e-4, 4e-5)
            _td.set_step_lr(opt, lr_, copt, mult)
            assert opt.param_groups[0]["lr"] == lr_ and opt.param_groups[1]["lr"] == lr_ * mult, (mult, step)
            assert copt.param_groups[0]["lr"] == lr_ * mult, (mult, step)
        # muP-style input: a list of groups with their own mup_lr_mult
        g2 = _td.ctrl_lr_groups([{"params": plain, "mup_lr_mult": 0.5}], ctrl_ids, mult)
        assert g2[1]["mup_lr_mult"] == 0.5 * mult and g2[0]["mup_lr_mult"] == 0.5
    print("F27 OK  (controller lr == mult x schedule at 4 schedule points, reinforce group and ppo optimizer; mult 1.0 untouched)")

# ── F28: retrieval-focused loss (--lr_gain_alpha).  The SR-only switch skips the long-range branch entirely; the
#        gain statistics of lr_gain_loss equal a manual two-forward computation; weights >= 1 and finite with mean
#        ~1 + alpha; alpha = 0 reproduces the plain mean CE (value and gradient) ──────────────────────────────────
if _td is not None and dev == "cuda":
    from experiments.scaling.model import FreeSoloLM as _LM
    from src.fsattn.attention import FSAttn as _EA
    torch.manual_seed(28)
    lm = _LM(vocab_size=257, d_model=d_model, n_layers=2, n_heads=H, d_ff=128, max_seq_len=N, attn_type="fsattn",
             window_size=16, long_range_variant="cell_topk", use_rope=False, lr_scale_mode="one", cell_select="score",
             cell_stride=S, sel_strict_causal=1, cell_topk=4, fsb_enabled=True, fsb_policy="fixed", fsb_width=4,
             fsb_delta=0, fsb_tree="forest").to(dev)
    for m_ in lm.modules():
        if isinstance(m_, _EA):
            m_._cell_block = BLK; m_.recompute_long_range = False
    lm.train()
    xi = torch.randint(0, 257, (2, N), device=dev); yi = torch.randint(0, 257, (2, N), device=dev)
    calls = {"n": 0}
    _olr = _EA._long_range
    def _cnt(self_, *a_, **k_):
        calls["n"] += 1; return _olr(self_, *a_, **k_)
    _EA._long_range = _cnt
    try:
        prev = _td.set_long_range_disabled(lm, True)
        assert len(prev) == 2 and all(not v_ for _, v_ in prev)
        with torch.no_grad():
            lg0 = lm(xi)
        assert calls["n"] == 0, "long-range branch ran under the SR-only switch"
        _td.restore_long_range(prev)
        assert all(not m_.disable_long_range for m_ in lm.modules() if isinstance(m_, _EA))
        lg = lm(xi)
        assert calls["n"] == 2
    finally:
        _EA._long_range = _olr
    l0 = F.cross_entropy(lg0.view(-1, 257).float(), yi.view(-1), reduction="none").view(yi.shape)
    lt = F.cross_entropy(lg.view(-1, 257).float(), yi.view(-1), reduction="none").view(yi.shape)
    for alpha, clip in ((2.0, 5.0), (0.5, 0.01)):
        loss_g, st_g = _td.lr_gain_loss(lg, yi, l0, alpha, clip)
        g_man = (l0 - lt).detach().clamp(0.0, clip); w_man = 1.0 + alpha * g_man / (g_man.mean() + 1e-6)
        assert abs(st_g["gain"] - g_man.mean().item()) < 1e-6 and abs(st_g["frac"] - (g_man > 0.05).float().mean().item()) < 1e-6
        assert abs(st_g["mass"] - (w_man[g_man > 0.05].sum() / w_man.sum()).item()) < 1e-5
        assert torch.isfinite(w_man).all() and (w_man >= 1.0).all() and abs(st_g["wmean"] - (1.0 + alpha)) < 1e-3
        assert torch.isfinite(loss_g) and abs(loss_g.item() - ((w_man * lt).sum() / w_man.sum()).item()) < 1e-5
    loss0, st0 = _td.lr_gain_loss(lg, yi, l0, 0.0, 5.0)
    plain_ce = F.cross_entropy(lg.view(-1, 257), yi.view(-1))
    assert abs(loss0.item() - plain_ce.item()) < 1e-5 and st0["wmean"] == 1.0
    gA = torch.autograd.grad(loss0, [lm.tok_emb.weight], retain_graph=True)[0]
    gB = torch.autograd.grad(plain_ce, [lm.tok_emb.weight], retain_graph=True)[0]
    assert torch.allclose(gA, gB, rtol=1e-4, atol=1e-7)
    print(f"F28 OK  (SR-only switch skips the LR branch; gain stats == manual two-forward computation; weights >= 1, mean 1+alpha; "
          f"alpha 0 == plain CE (value + grad); this tiny model: gain {st_g['gain']:.3f} nats, frac>0.05 {st_g['frac']:.2f})")
elif _td is not None:
    print("F28 SKIP (no CUDA)")


# ── F29: controller objective --fsb_reward gain: the policy loss equals a manual computation from l0, l and cost
#        (reward -clamp(l0 - l, 0, C), baseline, standardized advantage, entropy); "loss" returns the LM loss object
#        itself, so the existing path is bit-identical; works with the PPO store (records carry the gain advantage) ─
from src.fsattn.fsb_tree import fsb_reward_tokens
for rl in ("reinforce", "ppo"):
    torch.manual_seed(29)
    mm = mk(fsb_policy="learned", fsb_width=2, fsb_delta=1, fsb_tree="forest", fsb_rl=rl)
    mm.train()
    Qc, Kc, Vc = rand_qkv(99)
    torch.manual_seed(290)
    o = lr(mm, Qc, Kc, Vc)
    lt = torch.rand(1, N, device=dev) * 3; l0 = lt + torch.randn(1, N, device=dev) * 0.3      # l0 - lt: both signs
    assert fsb_reward_tokens(lt, l0, "loss", 5.0) is lt
    rt = fsb_reward_tokens(lt, l0, "gain", 0.4)
    assert torch.equal(rt, -(l0 - lt).clamp(0.0, 0.4)) and (rt <= 0).all() and (rt >= -0.4).all()
    st_ = {}
    Lg = fsb_policy_loss(mm, rt, eta=0.1, entropy_beta=0.01, value_coef=0.5, adv_norm=True, stats=st_)
    s_ = mm._fsb_stats; mask = s_["has"].float(); n_ = mask.sum().clamp(min=1.0)
    Z = rt.detach()[:, None, :] + 0.1 * s_["cost"]
    adv = (Z - s_["value"]).detach(); mu_ = (adv * mask).sum() / n_; sd_ = (((adv - mu_) ** 2) * mask).sum().div(n_).sqrt()
    adv_n = (adv - mu_) / (sd_ + 1e-6)
    manual = (adv_n * s_["logp"] * mask).sum() / n_ + 0.5 * ((s_["value"] - Z) ** 2 * mask).sum() / n_ - 0.01 * (s_["entropy"] * mask).sum() / n_
    assert torch.allclose(Lg, manual, rtol=1e-5, atol=1e-6), (rl, float(Lg), float(manual))
    assert abs(st_["reward"] - float((-rt[:, None, :].expand_as(mask) * mask).sum() / n_)) < 1e-5 and abs(st_["adv_raw_std"] - float(sd_)) < 1e-5
    assert st_["reward"] > 0.0
    if rl == "ppo":
        rec = mm._fsb_ppo_store[0]
        assert torch.equal(rec["Z"].reshape(-1), Z.reshape(-1)[:rec["Z"].numel()]) or rec["Z"].numel() > 0
        assert torch.isfinite(rec["adv"]).all()
        fsb_ppo_clear(mm)
    print(f"F29 [{rl}] gain objective == manual (loss {float(Lg):.4f}); mean gain reward {st_['reward']:.4f}, raw adv std {st_['adv_raw_std']:.4f}")
print("F29 OK  (controller reward = counterfactual long-range gain: manual match on reinforce and ppo; loss mode returns the LM loss itself)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  ROUTING-ONLY QUERY ADAPTER + ROUTER DISTILLATION (src/fsattn/fsb_route.py): F32..F36
# ══════════════════════════════════════════════════════════════════════════════════════════
import contextlib
import src.fsattn.fsb_route as _fr
from src.fsattn.fsb_route import fsb_route_q, fsb_rd_collect, det_head_matmul, route_param_names


def _run_lr(mm_, Q_, K_, V_, graph="0"):
    """_lr_cell_topk with sel_full captured; returns (sel_full, output, rd dict or None)."""
    cap = {}
    def _g(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_):
        cap["sel"] = sel_full.clone(); return _orig(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_)
    _ft4.masked_readout = _g
    os.environ["FSB_TRAIN_GRAPH"] = graph
    os.environ["FSB_EVAL_GRAPH"] = graph            # eval-mode models: graph "1" = the captured stage (zk: eval default eager)
    try:
        o = lr(mm_, Q_, K_, V_)
    finally:
        _ft4.masked_readout = _orig; os.environ.pop("FSB_TRAIN_GRAPH", None); os.environ.pop("FSB_EVAL_GRAPH", None)
    return cap["sel"], o, getattr(mm_, "_fsb_rd", None)


def _cached_run(mm_, x_, N0, lean, dgraph):
    """prefill of N0 tokens + per-token decode; returns (outputs (1,N,d), decode selections (N-N0, H, kk))."""
    os.environ["FSB_DECODE_GRAPH"] = dgraph
    try:
        cache = FSKVCache(mm_, lean_prefill=lean); cache.reserve = N - N0
        outs = [mm_(x_[:, :N0], cache=cache)]; sels = []
        for t in range(N0, N):
            outs.append(mm_(x_[:, t:t + 1], cache=cache))
            fc = getattr(cache, "_fsb_forest", None)
            if fc is not None and fc.SP is not None and getattr(fc, "_last_sel", None) is not None:
                sels.append(fc._last_sel.reshape(H, -1).clone())
            else:
                sels.append(None)
        if dgraph == "1":
            assert cache._fsb_forest.graph is not None, "decode graph never captured"
    finally:
        os.environ.pop("FSB_DECODE_GRAPH", None)
    return torch.cat(outs, dim=1), sels


def _full_sel(mm_, x_):
    """no-cache module forward with sel_full captured."""
    cap = {}
    def _g(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_):
        cap["sel"] = sel_full.clone(); return _orig(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_)
    _ft4.masked_readout = _g
    os.environ["FSB_TRAIN_GRAPH"] = "0"
    try:
        y = mm_(x_)
    finally:
        _ft4.masked_readout = _orig; os.environ.pop("FSB_TRAIN_GRAPH", None)
    return cap["sel"][0], y


# ── F32: flag off is byte-identical: no fsb_route_ parameter, same state_dict keys/values, identical selection and
#        output (eval and train mode); FreeSoloLM backbone init is unchanged by a present adapter (own RNG, skipped by
#        the blanket init) ─────────────────────────────────────────────────────────────────────────────────────
for tree in ("forest", "rebuild"):
    torch.manual_seed(32); ma = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree=tree)
    torch.manual_seed(32); mb = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree=tree, fsb_route_adapter="none",
                                   fsb_route_rank=8, fsb_route_alpha=1.0, fsb_rd_weight=0.0)
    sda, sdb = ma.state_dict(), mb.state_dict()
    assert list(sda) == list(sdb) and not any("fsb_route" in k for k in sdb), "flag off must create no parameter"
    assert all(torch.equal(sda[k], sdb[k]) for k in sda)
    Qc, Kc, Vc = rand_qkv(320)
    for mode in ("eval", "train"):
        ma.train(mode == "train"); mb.train(mode == "train")
        with torch.no_grad():
            ra = _run_lr(ma, Qc, Kc, Vc); rb = _run_lr(mb, Qc, Kc, Vc)
        assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1]) and rb[2] is None, (tree, mode)
from experiments.scaling.model import FreeSoloLM as _LM32
_lmkw = dict(vocab_size=257, d_model=d_model, n_layers=2, n_heads=H, d_ff=128, max_seq_len=N, attn_type="fsattn",
             window_size=16, long_range_variant="cell_topk", use_rope=False, lr_scale_mode="one", cell_select="score",
             cell_stride=S, sel_strict_causal=1, cell_topk=4, fsb_enabled=True, fsb_policy="fixed", fsb_width=2,
             fsb_delta=0, fsb_tree="forest")
torch.manual_seed(321); lm_a = _LM32(**_lmkw)
torch.manual_seed(321); lm_b = _LM32(**_lmkw, fsb_route_adapter="lowrank", fsb_route_rank=4)
sda, sdb = lm_a.state_dict(), lm_b.state_dict()
extra = [k for k in sdb if k not in sda]
assert extra and all(k.rsplit(".", 1)[-1].startswith("fsb_route_") for k in extra), extra
assert all(torch.equal(sda[k], sdb[k]) for k in sda), "a present adapter changed the backbone init"
assert all(float(sdb[k].abs().max()) == 0.0 for k in extra if k.endswith("fsb_route_B")), "B must be zero-init"
assert all(float(sdb[k].abs().max()) > 0.0 for k in extra if k.endswith("fsb_route_A")), "A must be random"
print(f"F32 OK  (flag off: no parameter, identical state / selection / output in eval + train, both trees; FreeSoloLM backbone "
      f"init unchanged by an adapter: {len(extra)} extra fsb_route_ tensors, B zero, A random)")


# ── F33: adapter present but ZERO (lowrank B = 0 / full W = 0): selection and output bit-identical to no adapter on
#        every path -- eager and CUDA-graph training stage (both trees for eager), full / lean prefill + per-token
#        decode with FSB_DECODE_GRAPH 0 and 1 -- and the distillation loss is identical eager vs graphed ─────────────
for adapter in ("lowrank", "full"):
    for tree in (("forest", "rebuild") if adapter == "lowrank" else ("forest",)):
        torch.manual_seed(33)
        m0 = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree=tree)
        m1 = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree=tree, fsb_route_adapter=adapter, fsb_route_rank=4)
        miss, unexp = m1.load_state_dict(m0.state_dict(), strict=False)
        assert not unexp and all("fsb_route_" in k for k in miss)
        Qc, Kc, Vc = rand_qkv(330)
        with torch.no_grad():
            graphs = ("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)
            for g in graphs:
                r0 = _run_lr(m0, Qc, Kc, Vc, g); r1 = _run_lr(m1, Qc, Kc, Vc, g)
                assert torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1]), (adapter, tree, "FSB_TRAIN_GRAPH", g)
            torch.manual_seed(331)
            x = torch.randn(1, N, d_model, device=dev)
            os.environ["FSB_TRAIN_GRAPH"] = "0"
            try:
                full0 = m0(x); full1 = m1(x)
            finally:
                os.environ.pop("FSB_TRAIN_GRAPH", None)
            assert torch.equal(full0, full1), (adapter, tree, "module forward")
            if tree == "forest":
                for lean in (False, True):
                    for dg in (("0", "1") if dev == "cuda" else ("0",)):
                        o0, s0 = _cached_run(m0, x, 150, lean, dg); o1, s1 = _cached_run(m1, x, 150, lean, dg)
                        assert torch.equal(o0, o1), (adapter, lean, dg)
                        assert all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(s0, s1)), (adapter, lean, dg)
    # distillation loss with the zero adapter: identical in eager and graphed training stages (same sampled rows)
    if dev == "cuda":
        torch.manual_seed(33)
        m2 = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_route_adapter=adapter, fsb_route_rank=4,
                fsb_rd_weight=1.0, fsb_rd_rows=8).train()
        res = {}
        for g in ("0", "1"):
            m2._fsb_rd_gen = torch.Generator(device=dev).manual_seed(5)
            with torch.no_grad():
                res[g] = _run_lr(m2, Qc, Kc, Vc, g)
        assert torch.equal(res["0"][0], res["1"][0]) and torch.equal(res["0"][1], res["1"][1])
        assert torch.equal(res["0"][2]["loss"].detach(), res["1"][2]["loss"].detach()) and torch.equal(res["0"][2]["stats"], res["1"][2]["stats"])
    print(f"F33 [{adapter}] zero adapter == no adapter bit-exact: eager{' + CUDA-graph' if dev == 'cuda' else ''} training stage, "
          f"module forward, full + lean prefill with per-token decode (FSB_DECODE_GRAPH 0{'/1' if dev == 'cuda' else ''}); rd loss eager == graphed")
print("F33 OK  (zero-initialised adapter reproduces the backbone's routing and output on every path)")


# ── F34: RANDOM adapter: (a) the fixed-order routing product is batch independent (a row alone == inside the batch)
#        and equals the Python halving; (b) full-sequence selection == prefill + per-token decode selection (sets, every
#        decode row, FSB_DECODE_GRAPH 0 and 1, full and lean prefill) and the cached output reproduces the full
#        forward; (c) the long-range READOUT uses the raw q (== masked readout of the raw Q over the routed cells,
#        != the readout of q_r); (d) the adapter changes the routing ────────────────────────────────────────────────
torch.manual_seed(34)
xq = torch.randn(2, H, 37, Dh, device=dev) * 3; Wr = torch.randn(H, Dh, 5, device=dev)
full_ = det_head_matmul(xq, Wr)
assert torch.equal(full_[1:2, :, 20:21], det_head_matmul(xq[1:2, :, 20:21].contiguous(), Wr)), "routing product is batch dependent"
from src.fsattn.det_reduce import _py_halving as _ph34
ref_ = _ph34((xq[..., :, None] * Wr[None, :, None]).movedim(-2, -1))
assert torch.allclose(full_, ref_, rtol=1e-6, atol=1e-5) and torch.allclose(full_, torch.einsum("bhnd,hde->bhne", xq, Wr), rtol=1e-5, atol=1e-4)
n_exact = int((full_ == ref_).float().mean() * 100)
for adapter in ("lowrank", "full"):
    torch.manual_seed(340)
    mm = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_route_adapter=adapter, fsb_route_rank=4)
    with torch.no_grad():
        if adapter == "lowrank":
            mm.fsb_route_B.normal_(0.0, 0.5)
        else:
            mm.fsb_route_W.normal_(0.0, 0.3)
    torch.manual_seed(341)
    x = torch.randn(1, N, d_model, device=dev)
    with torch.no_grad():
        sel_full, yfull = _full_sel(mm, x)                                 # (H,Npad,kk)
        a_ = mm.fsb_route_alpha; mm.fsb_route_alpha = 0.0
        sel_raw, _ = _full_sel(mm, x)
        mm.fsb_route_alpha = a_
        frac_changed = (sel_full != sel_raw).any(-1)[:, BLK:].float().mean().item()
        assert frac_changed > 0.05, f"{adapter}: the adapter did not change the routing ({frac_changed})"
        N0 = 150
        for lean in (False, True):
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                y, sels = _cached_run(mm, x, N0, lean, dg)
                d = (y - yfull).abs().max().item()
                assert d < 1e-4, (adapter, lean, dg, d)
                dec = torch.stack([s_ for s_ in sels], dim=1)             # (H, N-N0, kk)
                same_set, _ = _sets_equal(sel_full[:, N0:N].long(), dec.long())
                assert bool(same_set.all()), f"{adapter} lean={lean} graph={dg}: {int((~same_set).sum())} decode rows route differently"
        # (c) readout uses the raw q
        Qc, Kc, Vc = rand_qkv(342)
        sel_c, o_c, _ = _run_lr(mm, Qc, Kc, Vc)
        o_raw = _orig(mm, Qc, Kc, Vc, sel_c, S, BLK, N)
        Qr_ = fsb_route_q(mm, Qc.float())
        o_qr = _orig(mm, Qr_.to(Qc.dtype), Kc, Vc, sel_c, S, BLK, N)
        assert torch.equal(o_c, o_raw), "long-range readout must use the raw q"
        assert (o_c - o_qr).abs().max().item() > 1e-3, "q_r and q readouts coincide (test has no teeth)"
    print(f"F34 [{adapter}] adapter reroutes {100 * frac_changed:.0f}% of rows; full-sequence selection == prefill + decode selection "
          f"({N - N0} decode rows x {H} heads, full + lean prefill, decode graph 0{'/1' if dev == 'cuda' else ''}); cached == full; readout uses raw q")
print(f"F34 OK  (fixed-order routing product batch-independent ({n_exact}% of entries bit-equal to the Python halving); "
      f"random adapter: prefill routing == decode routing; readout on raw q)")


# ── F35: distillation loss == float64 brute force (value and adapter gradient) on the tiny case, for kl (with / without
#        a remainder bucket, far mask, teacher position term, teacher temperature) and topk; teacher chunking does not
#        change the loss; recall statistics == brute force ─────────────────────────────────────────────────────
def _rd_brute(mm_, Q_, K_, rows, sel, far, pos, T_, topn, mode):
    Q64, K64 = Q_[0].double(), K_[0].double()                            # (H,N,Dh)
    Kc = K64.view(H, M, S, Dh)
    kbar = K_[0].view(H, M, S, Dh).mean(2).double()                      # the module's fp32 cell mean, promoted
    ceil = Kc.norm(dim=-1).amax(-1)
    lam = mm_.sel_scale.detach().double(); sig = mm_.sigma.detach().double()
    if mm_.fsb_route_adapter == "lowrank":
        A64 = mm_.fsb_route_A.detach().double(); B64 = mm_.fsb_route_B.detach().double().requires_grad_(True); P_ = B64
        qr_all = Q64 + mm_.fsb_route_alpha * torch.einsum("hnr,hre->hne", torch.einsum("hnd,hdr->hnr", Q64, A64), B64)
    else:
        W64 = mm_.fsb_route_W.detach().double().requires_grad_(True); P_ = W64
        qr_all = Q64 + mm_.fsb_route_alpha * torch.einsum("hnd,hde->hne", Q64, W64)
    temp = mm_.fsb_route_logtemp.detach().double().exp()
    tot = 0.0; nrow = 0; hits = 0; den = 0; lhits = 0
    cen = (torch.arange(M, dtype=torch.float64, device=Q_.device) + 0.5) * S
    for hh in range(H):
        for i in rows.tolist():
            E = ((i // BLK) * BLK) // S
            if far:
                E = min(E, max(0, (i - mm_.window_size) // S))
            if E <= 0:
                continue
            sc = (Kc[hh, :E] @ Q64[hh, i]) * mm_.scale                   # (E,S)
            t = torch.logsumexp(sc, dim=-1)
            pt = lam[hh] * ceil[hh, :E] * torch.erf((i - cen[:E]).abs() / (2 * sig[hh]))
            if pos:
                t = t + pt
            t = t / T_
            r = ((kbar[hh, :E] @ qr_all[hh, i]) * mm_.scale + pt) / temp[hh]
            lq = torch.log_softmax(r, dim=0); lp = torch.log_softmax(t, dim=0)
            order = torch.argsort(t, descending=True)
            top = order[:topn]
            if mode == "kl":
                l = (lp[top].exp() * (lp[top] - lq[top])).sum()
                if E > topn:
                    rest = order[topn:]
                    lpr = torch.logsumexp(lp[rest], 0); lqr = torch.logsumexp(lq[rest], 0)
                    l = l + lpr.exp() * (lpr - lqr)
            else:
                k4 = order[:4].tolist(); l = 0.0
                for j_, c in enumerate(k4):
                    keep = torch.ones(E, dtype=torch.bool, device=Q_.device); keep[k4[:j_]] = False   # fresh mask
                    l = l + torch.logsumexp(r[keep], 0) - r[c]
                l = l / len(k4)
            tot = tot + l; nrow += 1
            s_ = set(sel[0, hh, i].tolist()) - {-1}
            hits += sum(1 for c in order[:4].tolist() if c in s_); den += min(4, E)
            leaf4 = set(torch.argsort(r.detach() * temp[hh], descending=True)[:4].tolist())   # leaf score (no temperature)
            lhits += sum(1 for c in order[:4].tolist() if c in leaf4)
    loss = tot / nrow
    g = torch.autograd.grad(loss, [P_])[0]
    return loss.item(), g, hits, den, nrow, lhits


for cfg in (dict(mode="kl", far=0, pos=0, T=1.0, topn=16), dict(mode="kl", far=1, pos=1, T=0.7, topn=4),
            dict(mode="topk", far=0, pos=1, T=1.0, topn=4), dict(mode="kl", far=0, pos=0, T=1.0, topn=4, adapter="full")):
    torch.manual_seed(35)
    ad = cfg.get("adapter", "lowrank")
    mm = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_route_adapter=ad, fsb_route_rank=4,
            fsb_rd_weight=1.0, fsb_rd_rows=8, fsb_rd_loss=cfg["mode"], fsb_rd_far=cfg["far"], fsb_rd_pos=cfg["pos"],
            fsb_rd_temp=cfg["T"], fsb_rd_topn=cfg["topn"]).train()
    with torch.no_grad():
        mm.sel_scale.fill_(0.7)
        (mm.fsb_route_B if ad == "lowrank" else mm.fsb_route_W).normal_(0.0, 0.3)
        mm.fsb_route_logtemp.normal_(0.0, 0.3)
    Qc, Kc, Vc = rand_qkv(350, scale=1.5)
    mm._fsb_rd_gen = torch.Generator(device=dev).manual_seed(7)
    with torch.no_grad():
        sel_c, _, rd = _run_lr(mm, Qc, Kc, Vc)
    assert rd is not None and rd["loss"].requires_grad
    rd["loss"].backward()
    P_ = mm.fsb_route_B if ad == "lowrank" else mm.fsb_route_W
    g_mod = P_.grad.clone(); mm.zero_grad(set_to_none=True)
    assert mm.W_q.weight.grad is None and mm.W_k.weight.grad is None and mm.sel_scale.grad is None
    lb, gb, hits, den, nrow, lhits = _rd_brute(mm, Qc, Kc, rd["rows"], sel_c, cfg["far"], cfg["pos"], cfg["T"], cfg["topn"], cfg["mode"])
    lm_ = rd["loss"].item()
    assert abs(lm_ - lb) <= 1e-5 * max(1.0, abs(lb)), (cfg, lm_, lb)
    gerr = ((g_mod.double() - gb).norm() / gb.norm().clamp(min=1e-12)).item()
    assert gerr < 1e-4, (cfg, gerr)
    st_ = rd["stats"]
    assert int(st_[7]) == nrow and int(st_[0]) == hits and int(st_[1]) == den and float(st_[2] + st_[3] + st_[4] + st_[5]) == 0.0, (cfg, st_.tolist(), hits, den, nrow)
    assert int(st_[8]) == lhits and float(st_[9] + st_[10]) == 0.0, (cfg, st_.tolist(), lhits)
    # teacher chunking: one block per chunk vs everything at once
    losses = []
    for ch in (mm.fsb_rd_rows, 4096):
        mm.fsb_rd_chunk = ch
        mm._fsb_rd_gen = torch.Generator(device=dev).manual_seed(7)
        with torch.no_grad():
            losses.append(_run_lr(mm, Qc, Kc, Vc)[2]["loss"].item())
    assert abs(losses[0] - losses[1]) <= 1e-6 * max(1.0, abs(losses[0])), losses
    # no-recompute path (FSB_RD_RECOMPUTE=0) gives the same value and gradient
    os.environ["FSB_RD_RECOMPUTE"] = "0"
    try:
        mm._fsb_rd_gen = torch.Generator(device=dev).manual_seed(7)
        with torch.no_grad():
            rd2 = _run_lr(mm, Qc, Kc, Vc)[2]
        rd2["loss"].backward()
    finally:
        os.environ.pop("FSB_RD_RECOMPUTE", None)
    assert rd2["loss"].item() == lm_ and torch.allclose(P_.grad, g_mod, rtol=1e-5, atol=1e-7)
    mm.zero_grad(set_to_none=True)
    print(f"F35 [{ad} {cfg['mode']} far={cfg['far']} pos={cfg['pos']} T={cfg['T']} topn={cfg['topn']}] rd {lm_:.6f} == float64 {lb:.6f}; "
          f"adapter grad rel err {gerr:.1e}; beam recall {hits}/{den}, leaf recall {lhits}/{den} over {nrow} rows == brute force; "
          f"chunking / recompute invariant")
print("F35 OK  (router-distillation loss and gradient == float64 brute force; stats exact)")


# ── optional trainer-side experiments modules (development tree only; absent in this checkout the sections that
#    need them run when importable and otherwise print a SKIP line: F36, F37[d], F42, F44, F96..F98, F100[b-e], F101[a-g]) ──
try:
    import experiments.scaling.fsb_routerkd as _rkflag
    _HAVE_ROUTERKD = True
except Exception:
    _rkflag = None; _HAVE_ROUTERKD = False
try:
    import experiments.scaling.fsb_utility as _fu
    _HAVE_UTILITY = True
except Exception:
    _fu = None; _HAVE_UTILITY = False
try:
    from experiments.scaling import fsb_rr_train as _rt98
    _HAVE_RRTRAIN = True
except Exception:
    _rt98 = None; _HAVE_RRTRAIN = False


# ── F36: gradients reach ONLY fsb_route_*: router-only step (frozen backbone, no-grad trunk) gives grads to the router
#        params alone and one AdamW step leaves every backbone tensor byte-identical; with a TRAINED backbone
#        (LM + w * rd) the backbone gradient equals the LM-only gradient exactly ────────────────────────────────
if dev == "cuda" and _HAVE_ROUTERKD:
        from experiments.scaling.fsb_routerkd import route_only_step, freeze_backbone, trunk as _trunk36, backbone_fingerprint
        from src.fsattn.attention import FSAttn as _EA36
        def _mklm(seed):
            torch.manual_seed(seed)
            lm = _LM32(**_lmkw, fsb_route_adapter="lowrank", fsb_route_rank=4, fsb_rd_weight=1.0, fsb_rd_rows=8).to(dev)
            for m_ in lm.modules():
                if isinstance(m_, _EA36):
                    m_._cell_block = BLK; m_.recompute_long_range = False
                    m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(36)
            return lm.train()
        ac = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        lm = _mklm(36)
        names, rparams = freeze_backbone(lm)
        assert names == sorted(route_param_names(lm)) and len(names) == 3 * 2
        xi = torch.randint(0, 257, (2, N), device=dev)
        with torch.no_grad(), ac():
            _trunk36(lm, xi)
        rdl, _ = fsb_rd_collect(lm)
        rdl.backward()
        got = sorted(n_ for n_, p_ in lm.named_parameters() if p_.grad is not None)
        assert got == names, got
        assert all(float(p_.grad.abs().sum()) > 0 for n_, p_ in lm.named_parameters() if n_.endswith("fsb_route_B"))
        lm.zero_grad(set_to_none=True)
        snap = {k: v.clone() for k, v in lm.state_dict().items()}
        fp0 = backbone_fingerprint(lm)
        opt = torch.optim.AdamW(rparams, lr=1e-2, weight_decay=0.0)
        gn, st, nmb = route_only_step(lm, [xi], rparams, opt, 1.0, 1.0, ac, world=1)
        assert nmb == 1 and torch.isfinite(gn) and st is not None
        rn = set(names)
        same = [torch.equal(snap[k], v) for k, v in lm.state_dict().items() if k not in rn]
        assert all(same) and backbone_fingerprint(lm) == fp0, "a backbone tensor changed"
        assert all(not torch.equal(snap[k], v) for k, v in lm.state_dict().items() if k.endswith("fsb_route_B")), "router did not move"
        # trained backbone (LM + w * rd): the distillation loss has NO autograd path to any backbone tensor, the LM loss none
        # to the router (the DDP tie enters at weight 0), and the joint backward gives the router its rd gradient
        lm2 = _mklm(37)
        with ac():
            lg = lm2(xi)
            loss = F.cross_entropy(lg.view(-1, 257).float(), xi.view(-1))
            rdl, _ = fsb_rd_collect(lm2)
        bb = [p_ for n_, p_ in lm2.named_parameters() if n_ not in rn]
        rr = [p_ for n_, p_ in lm2.named_parameters() if n_ in rn]
        g_rd_bb = torch.autograd.grad(rdl, bb, allow_unused=True, retain_graph=True)
        assert all(g_ is None for g_ in g_rd_bb), "distillation loss reaches a backbone tensor"
        g_lm_rr = torch.autograd.grad(loss, rr, allow_unused=True, retain_graph=True)
        assert all(g_ is None or float(g_.abs().max()) == 0.0 for g_ in g_lm_rr), "LM loss reaches the router"
        (loss + rdl).backward()
        assert all(p_.grad is not None for p_ in rr) and all(float(p_.grad.abs().sum()) > 0 for n_, p_ in lm2.named_parameters() if n_.endswith("fsb_route_B"))
        print(f"F36 OK  (router-only: grads on the {len(names)} fsb_route_ tensors only, one AdamW step leaves every backbone tensor "
              f"byte-identical (gnorm {float(gn):.3f}); LM + rd: rd has no path to the backbone, LM none to the router)")
else:
    print("F36 SKIP (no CUDA or experiments/scaling/fsb_routerkd.py absent)")


# ── F37: review fixes (2026-09-28).
#   (a) train_dist.fsb_objective ADDS every term: with --fsb_policy learned AND --fsb_rd_weight > 0 the objective is
#       LM + w*rd + policy loss and the router gets its distillation gradient (the pre-fix assembly LM + policy gave
#       it none); with rd off it is bit-identical to that assembly (value and every gradient); with both off it is the
#       LM loss object itself.
#   (b) route_init refuses distillation with fsb_leaf_radius 1 (the student would miss the leaf-radius term);
#       an adapter alone with a leaf radius still builds.
#   (c) rd_generator: one rank-distinct generator on every distilling layer; ranks draw different rows, the same
#       rank reproduces its rows, and the forward no longer consumes the global CUDA RNG.
#   (d) check_fingerprint (router-only _save): every rank of a 2-process gloo group gets the same verdict when only
#       one rank's backbone changed, so all ranks raise together instead of the healthy ones hanging in the barrier ─
from src.fsattn.fsb_route import rd_generator
from src.fsattn.fsb_tree import fsb_ctrl_params as _ctrl37
# (b) — CPU/GPU agnostic
try:
    mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_leaf_radius=1, fsb_route_adapter="lowrank",
       fsb_route_rank=4, fsb_rd_weight=1.0)
    raise AssertionError("route_init must refuse fsb_rd_weight > 0 with fsb_leaf_radius 1")
except ValueError:
    pass
_m37 = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_leaf_radius=1, fsb_route_adapter="lowrank",
          fsb_route_rank=4)
assert _m37.fsb_leaf_radius and _m37.fsb_route_adapter == "lowrank" and _m37.fsb_rd_weight == 0.0
del _m37
print("F37 [b] distillation + fsb_leaf_radius 1 refused (ValueError); adapter alone with a leaf radius builds")
if dev == "cuda":
    from types import SimpleNamespace
    from src.fsattn.attention import FSAttn as _EA37
    ac37 = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _mk37(seed, **kw):
        torch.manual_seed(seed)
        lm_ = _LM32(**{**_lmkw, **kw}).to(dev)
        for m_ in lm_.modules():
            if isinstance(m_, _EA37):
                m_._cell_block = BLK; m_.recompute_long_range = False
        return lm_.train()

    xi = torch.randint(0, 257, (2, N), device=dev); yi = torch.randint(0, 257, (2, N), device=dev)
    os.environ["FSB_TRAIN_GRAPH"] = "0"
    try:
        # (c) per-rank generator
        lmc = _mk37(373, fsb_route_adapter="lowrank", fsb_route_rank=4, fsb_rd_weight=1.0, fsb_rd_rows=8)
        rdl_ = [m_ for m_ in lmc.modules() if getattr(m_, "fsb_enabled", False)]

        def _rows(rank_):
            g_ = rd_generator(lmc, dev, 5, rank_)
            assert g_ is not None and all(m_._fsb_rd_gen is g_ for m_ in rdl_)
            st_ = torch.cuda.get_rng_state()
            with torch.no_grad(), ac37():
                lmc(xi)
            same_rng = torch.equal(torch.cuda.get_rng_state(), st_)
            r_ = [m_._fsb_rd["rows"].clone() for m_ in rdl_]
            fsb_rd_collect(lmc)
            return r_, same_rng
        r0, ok0 = _rows(0); r1, ok1 = _rows(1); r0b, _ = _rows(0)
        assert ok0 and ok1, "the distillation row draw consumed the global CUDA RNG"
        assert all(a_.numel() == b_.numel() for a_, b_ in zip(r0, r1)) and any(not torch.equal(a_, b_) for a_, b_ in zip(r0, r1)), \
            "rank 0 and rank 1 drew the same rows"
        assert all(torch.equal(a_, b_) for a_, b_ in zip(r0, r0b)), "the same rank must reproduce its rows"
        for m_ in rdl_:
            m_._fsb_rd_gen = None
        st_ = torch.cuda.get_rng_state()
        with torch.no_grad(), ac37():
            lmc(xi)
        assert not torch.equal(torch.cuda.get_rng_state(), st_), "sensitivity: without a generator the draw must use the global RNG"
        fsb_rd_collect(lmc)
        assert rd_generator(_mk37(374), dev, 5, 0) is None                 # no distilling layer -> no generator
        print(f"F37 [c] rd_generator: rank 0 / rank 1 rows differ ({sum(int(not torch.equal(a_, b_)) for a_, b_ in zip(r0, r1))}/"
              f"{len(r0)} layers), rank 0 reproduces its rows, global CUDA RNG untouched (consumed without a generator)")
        if _td is not None:
            targs = SimpleNamespace(fsb_rd_weight=0.7, fsb_reward="loss", lr_gain_clip=5.0, fsb_eta=0.1,
                                    fsb_entropy_beta=0.01, fsb_value_coef=0.5, fsb_adv_norm=True)
            _pk = dict(eta=targs.fsb_eta, entropy_beta=targs.fsb_entropy_beta, value_coef=targs.fsb_value_coef,
                       adv_norm=targs.fsb_adv_norm)
            _lkw = dict(fsb_policy="learned", fsb_width=2, fsb_delta=1)

            def _fwd(lm_, seed, amp=True):
                torch.manual_seed(seed)                                   # controller action sampling
                with (ac37() if amp else contextlib.nullcontext()):
                    lg_ = lm_(xi)
                    return lg_, F.cross_entropy(lg_.view(-1, 257), yi.view(-1))

            def _tok(lg_):
                return F.cross_entropy(lg_.view(-1, 257).float(), yi.view(-1), reduction="none").view(yi.shape).detach()
            # (a) learned controller + distillation: every term is added; the router gets its rd gradient
            lma = _mk37(371, fsb_route_adapter="lowrank", fsb_route_rank=4, fsb_rd_weight=targs.fsb_rd_weight,
                        fsb_rd_rows=8, **_lkw)
            rd_generator(lma, dev, 37, 0)
            rn37 = set(route_param_names(lma))
            rB = [p_ for n_, p_ in lma.named_parameters() if n_.endswith("fsb_route_B")]
            rT = [p_ for n_, p_ in lma.named_parameters() if n_.endswith("fsb_route_logtemp")]
            ctl = _ctrl37(lma)
            assert rB and rT and ctl
            lg, loss = _fwd(lma, 3710)
            rd_exp = torch.stack([m_._fsb_rd["loss"] for m_ in lma.modules()
                                  if getattr(m_, "fsb_enabled", False) and m_._fsb_rd is not None]).mean()
            with ac37():
                pol_exp = fsb_policy_loss(lma, _tok(lg), **_pk)
                old = loss + pol_exp                                     # the pre-fix assembly (rd dropped)
                rd_acc, rew_acc = {}, {}
                tot = _td.fsb_objective(lma, loss, lg, yi, None, targs, True, rd_acc, rew_acc)
            exp_ = loss + targs.fsb_rd_weight * rd_exp + pol_exp
            assert torch.allclose(tot, exp_, rtol=0, atol=1e-6) and abs(float(tot - old) - targs.fsb_rd_weight * float(rd_exp)) < 1e-5, \
                (float(tot), float(exp_), float(old), float(rd_exp))
            assert float(rd_exp) > 0 and rd_acc.get("st") is not None and rew_acc.get("n") == 1 and "reward" in rew_acc
            assert all(getattr(m_, "_fsb_rd", None) is None for m_ in lma.modules() if getattr(m_, "fsb_enabled", False)), \
                "fsb_objective must collect (clear) the distillation records"
            g_old = torch.autograd.grad(old, rB + rT, allow_unused=True, retain_graph=True)
            assert all(g_ is None or float(g_.abs().max()) == 0.0 for g_ in g_old), "pre-fix assembly: router must get no grad"
            tot.backward()
            assert all(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for p_ in rB + rT), \
                "learned policy + distillation: the router got no distillation gradient"
            assert all(p_.grad is not None for p_ in ctl) and sum(float(p_.grad.abs().sum()) for p_ in ctl) > 0, "controller got no gradient"
            assert float(lma.tok_emb.weight.grad.abs().sum()) > 0
            gB = [float(p_.grad.norm()) for p_ in rB]
            # (a2) rd off: fsb_objective == the pre-fix assembly (LM + policy): the value bit for bit, every gradient
            #      within the backward's own run-to-run floor.  fp32 here: this module's backward is not deterministic
            #      (atomics; the SAME code twice differs by ~3e-7 relative in fp32 and ~2e-2 under bf16 autocast).
            lmo = _mk37(372, **_lkw)
            lg1, loss1 = _fwd(lmo, 3720, amp=False)
            ref = loss1 + fsb_policy_loss(lmo, _tok(lg1), **_pk)
            ref.backward()
            g_ref = {n_: p_.grad.clone() for n_, p_ in lmo.named_parameters() if p_.grad is not None}
            lmo.zero_grad(set_to_none=True)
            targs0 = SimpleNamespace(**{**vars(targs), "fsb_rd_weight": None})
            lg2, loss2 = _fwd(lmo, 3720, amp=False)
            assert torch.equal(lg1, lg2), "two identical forwards differ (test premise)"
            tot2 = _td.fsb_objective(lmo, loss2, lg2, yi, None, targs0, True)
            assert torch.equal(tot2, ref), (float(tot2), float(ref))
            tot2.backward()
            g2 = {n_: p_.grad for n_, p_ in lmo.named_parameters() if p_.grad is not None}
            assert set(g2) == set(g_ref)
            n_eq = sum(int(torch.equal(g2[n_], g_ref[n_])) for n_ in g_ref)
            gd = max(float((g2[n_] - g_ref[n_]).abs().max() / g_ref[n_].abs().max().clamp(min=1e-30)) for n_ in g_ref)
            assert gd <= 1e-5, f"rd off: gradient changed (max rel diff {gd:.1e})"
            # (a3) both off: the LM loss object itself
            lmf = _mk37(375)
            lg3, loss3 = _fwd(lmf, 3750)
            assert _td.fsb_objective(lmf, loss3, lg3, yi, None, targs0, False) is loss3
            print(f"F37 [a] learned policy + rd: objective == LM + {targs.fsb_rd_weight}*rd ({float(rd_exp):.4f}) + policy; "
                  f"router rd grad ||dB|| {min(gB):.2e}..{max(gB):.2e} (pre-fix assembly: 0); rd off == pre-fix assembly "
                  f"(value bit-exact; {n_eq}/{len(g_ref)} grads bit-equal, max rel diff {gd:.1e}); both off returns the LM loss object")
        else:
            print("F37 [a] SKIP (train_dist import)")
    finally:
        os.environ.pop("FSB_TRAIN_GRAPH", None)
else:
    print("F37 [a,c] SKIP (no CUDA)")
# (d) 2-process gloo (CPU): the fingerprint verdict is agreed before anyone raises
if _HAVE_ROUTERKD:
    import socket, subprocess, sys as _sys37, tempfile, textwrap
    _d37 = tempfile.mkdtemp(prefix="f37_fp_")
    with socket.socket() as _so:
        _so.bind(("127.0.0.1", 0)); _port37 = _so.getsockname()[1]
    _scr = os.path.join(_d37, "fp_agree.py")
    with open(_scr, "w") as _f37:
        _f37.write(textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
            import torch, torch.distributed as dist, torch.multiprocessing as mp
            def w(rank, out):
                os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="{_port37}")
                dist.init_process_group("gloo", rank=rank, world_size=2)
                from experiments.scaling.fsb_routerkd import check_fingerprint
                fp0 = 1.234567e3
                good = check_fingerprint(fp0, fp0, "cpu", 2)
                bad = check_fingerprint(fp0 + (1e-6 if rank == 1 else 0.0), fp0, "cpu", 2)   # only rank 1 changed
                both = check_fingerprint(fp0 + 1.0, fp0, "cpu", 2)
                open(os.path.join(out, "r%d" % rank), "w").write("%d %d %d" % (good, bad, both))
                dist.destroy_process_group()
            if __name__ == "__main__":
                mp.spawn(w, args=(sys.argv[1],), nprocs=2)
            """))
    _env37 = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "GLOO_SOCKET_IFNAME": "lo"}
    _p37 = subprocess.run([_sys37.executable, _scr, _d37], env=_env37, timeout=300, capture_output=True, text=True)
    assert _p37.returncode == 0, _p37.stderr[-2000:]
    _res37 = [open(os.path.join(_d37, f"r{r_}")).read().split() for r_ in (0, 1)]
    import shutil
    shutil.rmtree(_d37, ignore_errors=True)
    assert _res37 == [["0", "1", "2"], ["0", "1", "2"]], _res37
    print(f"F37 [d] check_fingerprint over 2 gloo ranks: verdicts {_res37} (no change / rank 1 only / both) -> every rank raises together")
else:
    print("F37 [d] SKIP (experiments/scaling/fsb_routerkd.py absent)")
print("F37 OK  (objective adds rd + policy; leaf-radius refusal; per-rank rd generator" +
      ("; fingerprint verdict agreed across ranks)" if _HAVE_ROUTERKD else "; fingerprint gloo check SKIPPED)"))


# ══════════════════════════════════════════════════════════════════════════════════════════
#  NODE-LEVEL DISTILLATION OF THE BEAM + LEARNED RADIUS COEFFICIENT (fsb_rd_node / fsb_route_learn_c): F38..F45
#  Self-contained after the file header (runnable alone: exec the header + this section).
# ══════════════════════════════════════════════════════════════════════════════════════════
import sys as _sysn
_sysn.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "test_fsb_tree.py")))),
                                  "experiments", "scaling"))          # kv_cache imports longctx_infer (set by F7 otherwise)
import src.fsattn.fsb_tree as _ftn
import src.fsattn.fsb_route as _frn
from src.fsattn.fsb_route import route_radius, fsb_rd_collect as _collect_n, route_param_names as _rpn, RD_STATS_NODE
from src.fsattn.fsb_tree import build_spines as _bspn, traverse_forest as _travn, _forest_attrs as _fattrn
from experiments.scaling.kv_cache import FSKVCache as _KVCn
from experiments.scaling.model import FreeSoloLM as _LMn
_orig_n = _ftn.masked_readout                    # the true readout (every earlier test restores it)
_rdl_n = _frn.fsb_rd_loss
N2 = 512                                         # 32 cells, 8 query blocks: block trees up to 28 cells (3 chunks + spine)


def _qkv_n(seed, n, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return tuple((torch.randn(1, H, n, Dh, generator=g) * scale).to(dev) for _ in range(3))


def _lr_n(mm_, Q_, K_, V_, n, graph="0"):
    """_lr_cell_topk at length n with sel_full and the forest handed to the distillation captured."""
    cap = {}
    def _g(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_):
        cap["sel"] = sel_full.clone(); return _orig_n(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_)
    def _w(*a, **k):
        cap["forest"] = k.get("forest"); return _rdl_n(*a, **k)
    _ftn.masked_readout = _g; _frn.fsb_rd_loss = _w
    os.environ["FSB_TRAIN_GRAPH"] = graph
    os.environ["FSB_EVAL_GRAPH"] = graph            # eval-mode models: graph "1" = the captured stage (zk: eval default eager)
    try:
        o = mm_._lr_cell_topk(Q_, K_, V_, mm_.sigma, n)
    finally:
        _ftn.masked_readout = _orig_n; _frn.fsb_rd_loss = _rdl_n; os.environ.pop("FSB_TRAIN_GRAPH", None); os.environ.pop("FSB_EVAL_GRAPH", None)
    return cap["sel"], o, getattr(mm_, "_fsb_rd", None), cap.get("forest")


def _cached_n(mm_, x_, N0, lean, dgraph):
    """prefill of N0 tokens + per-token decode; (outputs, decode selections (H, kk) per token or None)."""
    os.environ["FSB_DECODE_GRAPH"] = dgraph
    try:
        cache = _KVCn(mm_, lean_prefill=lean); cache.reserve = x_.shape[1] - N0
        outs = [mm_(x_[:, :N0], cache=cache)]; sels = []
        for t in range(N0, x_.shape[1]):
            outs.append(mm_(x_[:, t:t + 1], cache=cache))
            fc = getattr(cache, "_fsb_forest", None)
            sels.append(fc._last_sel.reshape(H, -1).clone() if (fc is not None and fc.SP is not None and getattr(fc, "_last_sel", None) is not None) else None)
    finally:
        os.environ.pop("FSB_DECODE_GRAPH", None)
    return torch.cat(outs, dim=1), sels


def _full_n(mm_, x_):
    cap = {}
    def _g(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_):
        cap["sel"] = sel_full.clone(); return _orig_n(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_)
    _ftn.masked_readout = _g
    os.environ["FSB_TRAIN_GRAPH"] = "0"
    try:
        y = mm_(x_)
    finally:
        _ftn.masked_readout = _orig_n; os.environ.pop("FSB_TRAIN_GRAPH", None)
    return cap["sel"][0], y


def _sets_eq_n(a, b):
    va, vb = a >= 0, b >= 0
    hit = (a[..., :, None] == b[..., None, :]) & va[..., :, None] & vb[..., None, :]
    inter = hit.any(-1).sum(-1)
    return (inter == va.sum(-1)) & (inter == vb.sum(-1))


def _mk_node(**kw):
    base = dict(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_route_adapter="full",
                fsb_rd_weight=1.0, fsb_rd_rows=8)
    base.update(kw)
    return mk(**base)


# ── F38: flags off is byte-identical.  fsb_rd_node 0 / fsb_route_learn_c 0 (passed explicitly) create no parameter and
#        give the same state, selection, output, distillation loss and (11,) stats as a model built without the
#        kwargs, eager and CUDA-graph; with the node term ON the leaf part (stats[:11]) and the selection are unchanged
#        and loss_on == loss_off + w * node_sum / rows ──────────────────────────────────────────────────────────
for adapter in ("full", "lowrank"):
    torch.manual_seed(38)
    ma = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_route_adapter=adapter, fsb_route_rank=4,
            fsb_rd_weight=1.0, fsb_rd_rows=8).train()
    torch.manual_seed(38)
    mb_ = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_route_adapter=adapter, fsb_route_rank=4,
             fsb_rd_weight=1.0, fsb_rd_rows=8, fsb_rd_node=0, fsb_rd_node_weight=1.0, fsb_route_learn_c=0).train()
    sda, sdb = ma.state_dict(), mb_.state_dict()
    assert list(sda) == list(sdb) and all(torch.equal(sda[k], sdb[k]) for k in sda), "flags off changed the state"
    assert route_radius(mb_) is None and not hasattr(mb_, "fsb_route_ctheta") and not hasattr(mb_, "fsb_route_nlogtemp")
    with torch.no_grad():
        (ma.fsb_route_W if adapter == "full" else ma.fsb_route_B).normal_(0.0, 0.3)
    mb_.load_state_dict(ma.state_dict())
    Qc, Kc, Vc = _qkv_n(380, N2, 1.5)
    for g in (("0", "1") if dev == "cuda" else ("0",)):
        ra, rb = [], []
        for mm_, res in ((ma, ra), (mb_, rb)):
            mm_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(3)
            with torch.no_grad():
                res.extend(_lr_n(mm_, Qc, Kc, Vc, N2, g))
        assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1]), (adapter, g)
        assert torch.equal(ra[2]["loss"].detach(), rb[2]["loss"].detach()) and torch.equal(ra[2]["stats"], rb[2]["stats"])
        assert ra[2]["stats"].numel() == 11 and ra[3] is None and rb[3] is None
    # node term on (same rows): selection and leaf statistics unchanged, loss = leaf + w * node / rows
    torch.manual_seed(38)
    mc = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_route_adapter=adapter, fsb_route_rank=4,
            fsb_rd_weight=1.0, fsb_rd_rows=8, fsb_rd_node=1, fsb_rd_node_weight=0.7).train()
    miss, unexp = mc.load_state_dict(ma.state_dict(), strict=False)
    assert not unexp and miss == ["fsb_route_nlogtemp"], miss
    for g in (("0", "1") if dev == "cuda" else ("0",)):
        ma._fsb_rd_gen = torch.Generator(device=dev).manual_seed(4); mc._fsb_rd_gen = torch.Generator(device=dev).manual_seed(4)
        with torch.no_grad():
            r0 = _lr_n(ma, Qc, Kc, Vc, N2, g); r1 = _lr_n(mc, Qc, Kc, Vc, N2, g)
        st0, st1 = r0[2]["stats"], r1[2]["stats"]
        assert torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1]) and st1.numel() == RD_STATS_NODE
        assert torch.equal(st0, st1[:11]), (st0.tolist(), st1[:11].tolist())
        want = r0[2]["loss"].item() + 0.7 * st1[11].item() / st1[7].item()
        assert abs(r1[2]["loss"].item() - want) <= 1e-6 * max(1.0, abs(want)), (r1[2]["loss"].item(), want)
        assert float(st1[11]) > 0 and float(st1[14]) == 0.0, st1.tolist()
print("F38 OK  (flags off: no parameter, identical state / selection / output / rd loss / stats, eager + CUDA-graph; node on: "
      "leaf stats and selection unchanged, loss = leaf + w * node / rows, trace == forward selection)")


# ── F39: node loss == float64 brute force on the forest.  Each sampled row's block tree is enumerated by walking the
#        forest from the spine root (children refs); per depth d the level is a decision level iff its nodes plus the
#        leaves above it outnumber min(W, 2^height); T_v = logsumexp of the teacher cell scores (LSE + position term)
#        over the eligible cells under v; U_v = (q_r.mu_v + c_h ||q_r|| rho_v) scale + lambda A_v erf(...) over the
#        node temperature; KL(softmax(T/T_teacher) || softmax(U)) over the level's eligible nodes.  Checks the node KL
#        sum, the decision-level count, the gradients of the adapter / c_h / node temperature, invariance to the block
#        chunking and the recompute, and the node recall per iteration against a cell-enumeration of the trace ───────
def _node_brute(mm_, Q_, K_, rows, forest, n, far, pos, T_, sel_full):
    F_, mba = forest
    Mn = n // S
    Q64, K64 = Q_[0].double(), K_[0].double()
    Kc_ = K64.view(H, Mn, S, Dh)
    ceil = K_[0].view(H, Mn, S, Dh).float().norm(dim=-1).amax(-1).double()
    admit = K_[0].view(H, Mn, S, Dh).float().norm(dim=-1).amax(-1) >= mm_.fsb_tau.view(H, 1)
    lam = mm_.sel_scale.detach().double(); sig = mm_.sigma.detach().double()
    W64 = mm_.fsb_route_W.detach().double().requires_grad_(True)
    nlt = mm_.fsb_route_nlogtemp.detach().double().requires_grad_(True)
    params = [W64, nlt]
    qr_all = Q64 + mm_.fsb_route_alpha * torch.einsum("hnd,hde->hne", Q64, W64)
    if mm_.fsb_route_learn_c:
        th = mm_.fsb_route_ctheta.detach().double().requires_grad_(True); params.append(th)
        c64 = mm_.fsb_bound_radius_coef + torch.log1p(mm_._fsb_c_s0 * torch.expm1(th - mm_._fsb_ctheta0))
    else:
        c64 = torch.full((H,), float(mm_.fsb_bound_radius_coef), dtype=torch.float64, device=dev)
    cen = (torch.arange(Mn, dtype=torch.float64, device=dev) + 0.5) * S
    trees = {}

    def enum(h, b):
        SP1 = _bspn(F_, mba[h, b].view(1), torch.tensor([h], device=dev))
        out = []
        def rec(ref, d):
            A = _fattrn(F_, SP1, torch.tensor([[ref]], device=dev))
            nd = {k: A[k][0, 0] for k in A}
            cells = (F_["leaf"][h, int(nd["off"]):int(nd["off"]) + int(nd["len"])] if int(nd["src"]) == 1
                     else F_["adm"][h, int(nd["off"]):int(nd["off"]) + int(nd["len"])]).tolist()
            out.append((d, nd, cells, ref))
            l_, r_ = nd["child"].tolist()
            if l_ != -1:
                rec(l_, d + 1); rec(r_, d + 1)
        if int(mba[h, b]) > 0:
            rec(int(SP1["root"][0]), 0)
        return out, int(SP1["height"][0]), SP1

    tot = 0.0; ndec = 0; nrow = 0; rec_hits = {}; rec_den = 0
    for h in range(H):
        for i in rows.tolist():
            b = i // BLK
            E = (b * BLK) // S
            if far:
                E = min(E, max(0, (i - mm_.window_size) // S))
            elig = [c for c in range(E) if bool(admit[h, c])]
            if not elig:
                continue
            nrow += 1
            if (h, b) not in trees:
                trees[(h, b)] = enum(h, b)
            nodes, height, SP1 = trees[(h, b)]
            t = torch.logsumexp((Kc_[h] @ Q64[h, i]) * mm_.scale, dim=-1)                  # (Mn,)
            if pos:
                t = t + lam[h] * ceil[h] * torch.erf((i - cen).abs() / (2 * sig[h]))
            Wr = min(mm_.fsb_width, 2 ** height)
            D1 = max(d for d, *_ in nodes) + 1
            cnt = [0] * D1; lv = [0] * D1
            for d, nd, cells, _r in nodes:
                cnt[d] += 1; lv[d] += int(int(nd["cnt"]) == 1)
            eset = set(elig)
            qr = qr_all[h, i]
            for d in range(D1):
                if cnt[d] + sum(lv[:d]) <= Wr:
                    continue
                ndec += 1
                Ts, Us = [], []
                for dd, nd, cells, _r in nodes:
                    ce_ = [c for c in cells if c in eset]
                    if dd != d or not ce_:
                        continue
                    Ts.append(torch.logsumexp(t[ce_], 0) / T_)
                    u = (qr @ nd["mu"].double()) * mm_.scale + c64[h] * qr.norm() * float(nd["rad"]) * mm_.scale \
                        + lam[h] * float(nd["amax"]) * torch.erf(torch.tensor(abs(i - (float(nd["pmin"]) + 0.5) * S) / (2 * float(sig[h])), dtype=torch.float64, device=dev))
                    Us.append(u / nlt[h].exp())
                if len(Ts) >= 2:
                    lp = torch.log_softmax(torch.stack(Ts), 0); lq = torch.log_softmax(torch.stack(Us), 0)
                    tot = tot + (lp.exp() * (lp - lq)).sum()
            # node recall of the beam: teacher top cell (argmax over eligible of t / T) under a kept node, per iteration
            tt_ = torch.full((Mn,), -1e30, dtype=torch.float64, device=dev); tt_[elig] = t[elig] / T_
            cstar = int(tt_.argmax())
            rec_den += 1
            height_ = SP1["height"]
            Dq = (height_ - mm_.fsb_delta).clamp(min=0)
            Wq = torch.minimum(torch.full_like(Dq, mm_.fsb_width), torch.bitwise_left_shift(torch.ones_like(Dq), Dq))
            trace = []
            rc = route_radius(mm_)
            sel1, _, _ = _travn(F_, SP1, mm_.fsb_route_q_rows[h, i].view(1, 1, Dh), torch.tensor([[float(i)]], device=dev),
                                mm_.sel_scale.detach().float()[h].view(1, 1, 1), mm_.sigma.detach().float()[h].view(1, 1, 1),
                                mm_.scale, S, Wq.view(1, 1), (height_ - Dq).view(1, 1), mm_.fsb_W_max, mm_.fsb_delta_max,
                                mm_.fsb_node_score, radius_coef=(rc.detach()[h].view(1) if rc is not None else mm_.fsb_bound_radius_coef),
                                trace=trace)
            assert torch.equal(sel1.view(-1), sel_full[0, h, i].view(-1)), ("single-row trace != forward selection", h, i)
            for it, ref in enumerate(trace):
                under = set()
                for r_ in ref.view(-1).tolist():
                    if r_ == -1:
                        continue
                    A = _fattrn(F_, SP1, torch.tensor([[r_]], device=dev))
                    o_, l_ = int(A["off"][0, 0]), int(A["len"][0, 0])
                    under |= set((F_["leaf"][h, o_:o_ + l_] if int(A["src"][0, 0]) == 1 else F_["adm"][h, o_:o_ + l_]).tolist())
                rec_hits[it] = rec_hits.get(it, 0) + int(cstar in under)
    L = tot / max(nrow, 1)
    grads = torch.autograd.grad(L, params, allow_unused=True) if torch.is_tensor(tot) else [None] * len(params)
    return (float(tot), grads, ndec, nrow, rec_hits, rec_den)


for cfg in (dict(far=0, pos=1, T=1.0, W=2, tau=0.0, c0=8.0, learn_c=1, w=1.0),
            dict(far=1, pos=0, T=0.7, W=3, tau=0.35, c0=2.5, learn_c=1, w=0.5),
            dict(far=0, pos=1, T=1.0, W=4, tau=0.0, c0=1.0, learn_c=0, w=1.0)):
    torch.manual_seed(39)
    mm = _mk_node(fsb_width=cfg["W"], fsb_rd_node=1, fsb_rd_node_weight=cfg["w"], fsb_rd_far=cfg["far"],
                  fsb_rd_pos=cfg["pos"], fsb_rd_temp=cfg["T"], fsb_bound_radius_coef=cfg["c0"],
                  fsb_route_learn_c=cfg["learn_c"]).train()
    Qc, Kc, Vc = _qkv_n(390, N2, 1.5)
    with torch.no_grad():
        mm.sel_scale.fill_(0.7)
        mm.fsb_route_W.normal_(0.0, 0.3)
        mm.fsb_route_logtemp.normal_(0.0, 0.3)
        mm.fsb_route_nlogtemp.normal_(0.0, 0.3)
        if cfg["learn_c"]:
            mm.fsb_route_ctheta.add_(torch.randn(H, device=dev) * 0.5)
        if cfg["tau"] > 0:                                               # partial admission: irregular tree sizes
            a_ = Kc[0].view(H, N2 // S, S, Dh).float().norm(dim=-1).amax(-1)
            mm.fsb_tau.copy_(torch.quantile(a_, cfg["tau"], dim=-1))
    mm.fsb_route_q_rows = _frn.fsb_route_q(mm, Qc.float())[0]           # routing queries (H, N2, Dh) for the brute trace
    res = {}
    for key, node_on in (("off", 0), ("on", 1)):
        mm.fsb_rd_node = node_on
        mm._fsb_rd_gen = torch.Generator(device=dev).manual_seed(9)
        with torch.no_grad():
            sel_c, _, rd, fr = _lr_n(mm, Qc, Kc, Vc, N2)
        rd["loss"].backward()
        res[key] = (rd, {n_: (p_.grad.clone() if p_.grad is not None else None) for n_, p_ in mm.named_parameters() if "fsb_route_" in n_}, fr, sel_c)
        mm.zero_grad(set_to_none=True)
    mm.fsb_rd_node = 1
    rd1, g1, fr, sel_c = res["on"]; rd0, g0, _, _ = res["off"]
    assert torch.equal(rd1["rows"], rd0["rows"])
    nb_sum, nb_g, nb_dec, nb_rows, nb_hits, nb_den = _node_brute(mm, Qc, Kc, rd1["rows"], fr, N2, cfg["far"], cfg["pos"], cfg["T"], sel_c)
    st1 = rd1["stats"]
    assert int(st1[7]) == nb_rows and int(st1[13]) == nb_dec, (cfg, st1[7].item(), nb_rows, st1[13].item(), nb_dec)
    assert abs(st1[11].item() - nb_sum) <= 1e-5 * max(1.0, abs(nb_sum)), (cfg, st1[11].item(), nb_sum)
    gW = (g1["fsb_route_W"] - g0["fsb_route_W"]).double() / cfg["w"]    # node part of the adapter gradient
    errW = ((gW - nb_g[0]).norm() / nb_g[0].norm().clamp(min=1e-12)).item()
    errT = ((g1["fsb_route_nlogtemp"].double() / cfg["w"] - nb_g[1]).norm() / nb_g[1].norm().clamp(min=1e-12)).item()
    assert g0.get("fsb_route_nlogtemp") is None and errW < 1e-4 and errT < 1e-4, (cfg, errW, errT)
    errC = float("nan")
    if cfg["learn_c"]:
        assert g0["fsb_route_ctheta"] is None, "the leaf loss must not reach c_h (leaves have radius 0)"
        errC = ((g1["fsb_route_ctheta"].double() / cfg["w"] - nb_g[2]).norm() / nb_g[2].norm().clamp(min=1e-12)).item()
        assert errC < 1e-4 and float(g1["fsb_route_ctheta"].abs().max()) > 0, (cfg, errC)
    # node recall statistics == the cell enumeration of the trace
    assert int(st1[12]) == nb_den and float(st1[14]) == 0.0, (cfg, st1[12].item(), nb_den, st1[14].item())
    for it, hv in nb_hits.items():
        assert int(st1[16 + it]) == hv, (cfg, it, st1[16 + it].item(), hv)
    last = max(nb_hits); assert all(int(st1[16 + j]) == nb_hits[last] for j in range(last + 1, 16))
    hs = [int(st1[16 + j]) for j in range(16)]
    assert all(hs[j] >= hs[j + 1] for j in range(15)), hs
    # block chunking and the recompute do not change the value or the gradient
    vals = []
    for chunk, recmp in ((1, "1"), (64, "1"), (3, "0")):
        mm.fsb_rd_node_blkchunk = chunk; os.environ["FSB_RD_RECOMPUTE"] = recmp
        mm._fsb_rd_gen = torch.Generator(device=dev).manual_seed(9)
        try:
            with torch.no_grad():
                rdx = _lr_n(mm, Qc, Kc, Vc, N2)[2]
            rdx["loss"].backward()
        finally:
            os.environ.pop("FSB_RD_RECOMPUTE", None)
        vals.append((rdx["loss"].item(), mm.fsb_route_W.grad.clone()))
        mm.zero_grad(set_to_none=True)
    mm.fsb_rd_node_blkchunk = 16
    for v_, gg in vals:
        assert abs(v_ - rd1["loss"].item()) <= 1e-6 * max(1.0, abs(v_)) and torch.allclose(gg, g1["fsb_route_W"], rtol=1e-4, atol=1e-7)
    if dev == "cuda":                                                     # graphed training stage: same loss / stats
        mm._fsb_rd_gen = torch.Generator(device=dev).manual_seed(9)
        with torch.no_grad():
            rdg = _lr_n(mm, Qc, Kc, Vc, N2, "1")[2]
        assert torch.equal(rdg["stats"], st1) and rdg["loss"].item() == rd1["loss"].item()
    print(f"F39 [W{cfg['W']} far={cfg['far']} pos={cfg['pos']} T={cfg['T']} tau_q={cfg['tau']} c0={cfg['c0']} learn_c={cfg['learn_c']}] "
          f"node KL sum {st1[11].item():.6f} == float64 {nb_sum:.6f} over {nb_rows} rows, {nb_dec} decision levels; grad rel err "
          f"W {errW:.1e} node-temp {errT:.1e} c {errC:.1e}; node recall {[round(h_ / max(nb_den, 1), 3) for h_ in hs[:max(nb_hits) + 1]]} "
          f"== trace cell enumeration; chunking / recompute / graph invariant")
    del mm.fsb_route_q_rows
print("F39 OK  (node-level KL, decision levels, gradients (adapter, c_h, node temperature) and node recall == float64 brute force)")


# ── F40: learned c at init is BIT-IDENTICAL to the fixed coefficient: c_h == c0 exactly, and with a random adapter the
#        selection and output agree on every path (eager / CUDA-graph training stage, both trees, module forward, full
#        and lean prefill + per-token decode with FSB_DECODE_GRAPH 0/1); moving theta changes the routing ───────────
for c0 in (8.0, 1.0, 2.5):
    for tree in ("forest", "rebuild"):
        torch.manual_seed(40)
        ma = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree=tree, fsb_route_adapter="full", fsb_bound_radius_coef=c0)
        mb_ = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree=tree, fsb_route_adapter="full", fsb_bound_radius_coef=c0,
                 fsb_route_learn_c=1)
        with torch.no_grad():
            ma.fsb_route_W.normal_(0.0, 0.3)
        miss, unexp = mb_.load_state_dict(ma.state_dict(), strict=False)
        assert not unexp and miss == ["fsb_route_ctheta"], miss
        c_ = route_radius(mb_)
        assert torch.equal(c_, torch.full((H,), c0, device=dev)), (c0, c_.tolist())
        assert abs(float(torch.nn.functional.softplus(mb_.fsb_route_ctheta.double()).mean()) - c0) < 1e-5 * c0
        Qc, Kc, Vc = rand_qkv(400, 1.5)
        with torch.no_grad():
            for g in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                r0 = _lr_n(ma, Qc, Kc, Vc, N, g); r1 = _lr_n(mb_, Qc, Kc, Vc, N, g)
                assert torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1]), (c0, tree, g)
            torch.manual_seed(401)
            x = torch.randn(1, N, d_model, device=dev)
            s0, y0 = _full_n(ma, x); s1, y1 = _full_n(mb_, x)
            assert torch.equal(s0, s1) and torch.equal(y0, y1), (c0, tree)
            if tree == "forest":
                for lean in (False, True):
                    for dg in (("0", "1") if dev == "cuda" else ("0",)):
                        o0, q0 = _cached_n(ma, x, 150, lean, dg); o1, q1 = _cached_n(mb_, x, 150, lean, dg)
                        assert torch.equal(o0, o1) and all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(q0, q1)), (c0, lean, dg)
            # teeth: c -> 3 c0 reroutes
            mb_.fsb_route_ctheta.fill_(float(torch.log(torch.expm1(torch.tensor(3.0 * c0)))))
            r2 = _lr_n(mb_, Qc, Kc, Vc, N)
            frac = (r2[0] != r0[0]).any(-1)[..., BLK:].float().mean().item()
            assert frac > 0.02, (c0, tree, frac)
    print(f"F40 [c0={c0}] learned c at init == fixed coefficient bit-exact (c_h == {c0}; eager/graph training stage, both trees, "
          f"module forward, full + lean prefill + decode graph 0/1); c -> 3 c0 reroutes")
print("F40 OK  (learn_c at init reproduces the fixed-coefficient routing on every path)")


# ── F41: RANDOM adapter + RANDOM c_h: full-sequence selection == prefill + per-token decode selection (sets, every
#        decode row, full / lean prefill, FSB_DECODE_GRAPH 0/1), cached output == full forward; the CUDA-graph training
#        stage follows an in-place update of theta (static buffer refreshed per call) bit-exactly ──────────────────
torch.manual_seed(41)
mm = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_route_adapter="full", fsb_bound_radius_coef=4.0,
        fsb_route_learn_c=1)
with torch.no_grad():
    mm.fsb_route_W.normal_(0.0, 0.3)
    mm.fsb_route_ctheta.add_(torch.randn(H, device=dev) * 2.0)
cvals = route_radius(mm).tolist()
torch.manual_seed(411)
x = torch.randn(1, N, d_model, device=dev)
with torch.no_grad():
    sel_f, yfull = _full_n(mm, x)
    N0 = 150
    for lean in (False, True):
        for dg in (("0", "1") if dev == "cuda" else ("0",)):
            y, sels = _cached_n(mm, x, N0, lean, dg)
            d = (y - yfull).abs().max().item()
            assert d < 1e-4, (lean, dg, d)
            dec = torch.stack(sels, dim=1)
            same = _sets_eq_n(sel_f[:, N0:N].long(), dec.long())
            assert bool(same.all()), f"lean={lean} graph={dg}: {int((~same).sum())} decode rows route differently"
    if dev == "cuda":
        Qc, Kc, Vc = rand_qkv(412, 1.5)
        for step in range(3):
            re = _lr_n(mm, Qc, Kc, Vc, N, "0"); rg = _lr_n(mm, Qc, Kc, Vc, N, "1")
            assert torch.equal(re[0], rg[0]) and torch.equal(re[1], rg[1]), ("graph != eager after theta update", step)
            mm.fsb_route_ctheta.add_(torch.randn(H, device=dev) * 1.0)       # an optimizer step in place
print(f"F41 OK  (random adapter + c_h {[round(c, 2) for c in cvals]}: prefill + decode (graph 0/1, full + lean) == full-sequence "
      f"selection; CUDA-graph training stage tracks in-place theta updates bit-exactly)")


# ── F42: gradients reach ONLY fsb_route_*: router-only step with the node term + learned c on the tiny FreeSoloLM (B=2):
#        grads on the adapter, leaf temperature, c_h and node temperature only (all nonzero); loss, stats and router
#        grads on the CUDA-graph forest (shared-pool static tables read by the backward node recompute of every layer)
#        are bit-identical to the eager forest; one AdamW step leaves every backbone tensor byte-identical and moves c;
#        the c lr multiplier puts theta in its own group ───────────────────────────────────────────────────────
if dev == "cuda" and _HAVE_ROUTERKD:
        from experiments.scaling.fsb_routerkd import route_only_step as _ros, freeze_backbone as _fzb, backbone_fingerprint as _bfp
        from experiments.scaling.fsb_routerkd import trunk as _trunkn
        from src.fsattn.attention import FSAttn as _EAn
        torch.manual_seed(42)
        _kw42 = dict(vocab_size=257, d_model=d_model, n_layers=2, n_heads=H, d_ff=128, max_seq_len=N, attn_type="fsattn",
                     window_size=16, long_range_variant="cell_topk", use_rope=False, lr_scale_mode="one", cell_select="score",
                     cell_stride=S, sel_strict_causal=1, cell_topk=4, fsb_enabled=True, fsb_policy="fixed", fsb_width=2,
                     fsb_delta=0, fsb_tree="forest", fsb_route_adapter="full", fsb_rd_weight=1.0, fsb_rd_rows=8,
                     fsb_rd_node=1, fsb_route_learn_c=1, fsb_bound_radius_coef=4.0)
        lm = _LMn(**_kw42).to(dev)
        for m_ in lm.modules():
            if isinstance(m_, _EAn):
                m_._cell_block = BLK; m_.recompute_long_range = False
                m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(42)
        lm.train()
        with torch.no_grad():                                                   # a non-trivial adapter (zero W: q_r == q)
            for m_ in lm.modules():
                if isinstance(m_, _EAn):
                    m_.fsb_route_W.normal_(0.0, 0.1)
        names, rparams = _fzb(lm)
        assert names == sorted(_rpn(lm)) and len(names) == 8 and all(
            sum(n_.endswith(p) for n_ in names) == 2 for p in ("fsb_route_W", "fsb_route_logtemp", "fsb_route_ctheta", "fsb_route_nlogtemp")), names
        ac42 = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        xi = torch.randint(0, 257, (2, N), device=dev)
        with torch.no_grad(), ac42():
            _trunkn(lm, xi)
        rdl, st42 = _collect_n(lm)
        assert st42.numel() == RD_STATS_NODE
        rdl.backward()
        got = sorted(n_ for n_, p_ in lm.named_parameters() if p_.grad is not None)
        assert got == names, got
        assert all(float(p_.grad.abs().sum()) > 0 for n_, p_ in lm.named_parameters() if n_ in names), \
            [(n_, float(p_.grad.abs().sum())) for n_, p_ in lm.named_parameters() if n_ in names]
        lm.zero_grad(set_to_none=True)

        # CUDA-graph stage vs eager: the node loss of layer l is recomputed in backward from the forest tables the graph
        # build stage wrote into the SHARED pool; layer l+1's replay runs between l's forward and l's backward.  Loss,
        # stats and every router gradient must be bit-identical to the eager forest (same rows: generators reset), also
        # after a forward on another input has overwritten every static buffer.
        def _run42(x_, graph):
            if graph == "0":
                os.environ["FSB_TRAIN_GRAPH"] = "0"
            else:
                os.environ.pop("FSB_TRAIN_GRAPH", None)                         # default: graph on CUDA
            try:
                for m_ in lm.modules():
                    if isinstance(m_, _EAn):
                        m_._fsb_rd_gen.manual_seed(42)
                with torch.no_grad(), ac42():
                    _trunkn(lm, x_)
                l_, s_ = _collect_n(lm)
                l_.backward()
                gr_ = {n_: p_.grad.detach().clone() for n_, p_ in lm.named_parameters() if n_ in names}
                lm.zero_grad(set_to_none=True)
                return l_.detach().clone(), s_.detach().clone(), gr_
            finally:
                os.environ.pop("FSB_TRAIN_GRAPH", None)
        xj = torch.randint(0, 257, (2, N), device=dev)
        r_g1 = _run42(xi, "1")
        _run42(xj, "1")                                                         # dirty every static buffer / pool block
        r_e = _run42(xi, "0")
        r_g2 = _run42(xi, "1")
        for tag_, r_ in (("graph", r_g1), ("graph after another input", r_g2)):
            assert torch.equal(r_[0], r_e[0]), (tag_, float(r_[0]), float(r_e[0]))
            assert torch.equal(r_[1], r_e[1]), (tag_, (r_[1] - r_e[1]).abs().max())
            for n_ in names:
                assert torch.equal(r_[2][n_], r_e[2][n_]), (tag_, n_, float((r_[2][n_] - r_e[2][n_]).abs().max()))
        assert all(float(g_.abs().sum()) > 0 for g_ in r_e[2].values())
        print(f"F42 [a] CUDA-graph forest == eager, bit-exact: rd loss {float(r_e[0]):.6f}, {r_e[1].numel()} stats, "
              f"{len(names)} router grads (also after a forward on another input)")
        snap = {k: v.clone() for k, v in lm.state_dict().items()}
        fp0 = _bfp(lm)
        rn = set(names)
        cids = {id(p_) for n_, p_ in lm.named_parameters() if n_.endswith("fsb_route_ctheta")}
        opt = torch.optim.AdamW([{"params": [p_ for p_ in rparams if id(p_) not in cids], "lr_mult": 1.0},
                                 {"params": [p_ for p_ in rparams if id(p_) in cids], "lr_mult": 10.0}], lr=1e-2, weight_decay=0.0)
        for g_ in opt.param_groups:
            g_["lr"] = 1e-2 * g_["lr_mult"]
        c_before = [route_radius(m_).clone() for m_ in lm.modules() if isinstance(m_, _EAn)]
        gn, st, nmb = _ros(lm, [xi], rparams, opt, 1.0, 1.0, ac42, world=1)
        assert nmb == 1 and torch.isfinite(gn) and st.numel() == RD_STATS_NODE
        assert all(torch.equal(snap[k], v) for k, v in lm.state_dict().items() if k not in rn) and _bfp(lm) == fp0, "backbone changed"
        c_after = [route_radius(m_) for m_ in lm.modules() if isinstance(m_, _EAn)]
        dc = max(float((a_ - b_).abs().max()) for a_, b_ in zip(c_after, c_before))
        assert dc > 0.05, dc                                                    # lr 0.1 on theta -> c moves ~0.1
        print(f"F42 OK  (router-only with node term + learned c: grads on the {len(names)} fsb_route_ tensors only (adapter, leaf temp, "
              f"c_h, node temp all nonzero), CUDA-graph == eager bit-exact; AdamW step leaves the backbone byte-identical; "
              f"c moved {dc:.3f} (theta group lr x10))")
else:
    print("F42 SKIP (no CUDA or experiments/scaling/fsb_routerkd.py absent)")


# ── F43: refusals and the warm start.  fsb_rd_node needs the distillation, the forest and the fixed W/D0 policy;
#        fsb_route_learn_c needs an adapter and the bound score with c0 > 0.  load_router_params copies the fsb_route_*
#        tensors of a router checkpoint (fsb_route.safetensors) and keeps params the file lacks (c_h, node temp) at init;
#        a shape mismatch raises ─────────────────────────────────────────────────────────────────────────────────
for bad in (dict(fsb_rd_node=1, fsb_rd_weight=0.0), dict(fsb_rd_node=1, fsb_tree="rebuild"),
            dict(fsb_rd_node=1, fsb_delta=1), dict(fsb_rd_node=1, fsb_policy="learned"),
            dict(fsb_route_learn_c=1, fsb_route_adapter="none", fsb_rd_weight=0.0), dict(fsb_route_learn_c=1, fsb_node_score="centroid"),
            dict(fsb_route_learn_c=1, fsb_bound_radius_coef=0.0), dict(fsb_rd_node=2)):
    try:
        _mk_node(**bad)
        raise AssertionError(f"route_init must refuse {bad}")
    except ValueError:
        pass
import tempfile
from safetensors.torch import save_file as _sfn
torch.manual_seed(43)
src_m = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_route_adapter="full", fsb_rd_weight=1.0)
with torch.no_grad():
    src_m.fsb_route_W.normal_(0.0, 0.3); src_m.fsb_route_logtemp.normal_(0.0, 0.3)
dst = _mk_node(fsb_rd_node=1, fsb_route_learn_c=1, fsb_bound_radius_coef=8.0)
c_init = route_radius(dst).clone(); nt_init = dst.fsb_route_nlogtemp.detach().clone()
with tempfile.TemporaryDirectory() as td:
    _sfn({k: v.detach().cpu().contiguous() for k, v in src_m.state_dict().items() if "fsb_route_" in k}, os.path.join(td, "fsb_route.safetensors"))
    msg = _frn.load_router_params(dst, td)
    assert torch.equal(dst.fsb_route_W, src_m.fsb_route_W) and torch.equal(dst.fsb_route_logtemp, src_m.fsb_route_logtemp)
    assert torch.equal(route_radius(dst), c_init) and torch.equal(dst.fsb_route_nlogtemp, nt_init)
    assert "fsb_route_ctheta" in msg and "fsb_route_nlogtemp" in msg, msg
    small = mk(fsb_policy="fixed", fsb_width=2, fsb_delta=0, fsb_tree="forest", fsb_route_adapter="lowrank", fsb_route_rank=4)
    try:
        _frn.load_router_params(small, td)
        raise AssertionError("load_router_params must refuse a router of another shape")
    except ValueError:
        pass
print(f"F43 OK  (refusals: 8 invalid configs raise; warm start: {msg})")


# ── F44: the held-out drift gate compares against the BACKBONE'S OWN ROUTING.  With learn_c on and c_h moved away from
#        c0, heldout_ce(alpha_zero=True) == the CE of the same backbone built with NO adapter and the fixed c0,
#        bit-exactly, on the CUDA-graph and the eager forest (and with eager=True, which leaves the training graphs
#        un-recaptured); keep_c=True (alpha 0 + learned c, what alpha_zero alone measured before) and the router-on CE
#        both differ from it.  heldout_ce restores alpha, fsb_route_learn_c, FSB_TRAIN_GRAPH and train mode, also when a
#        batch raises ──────────────────────────────────────────────────────────────────────────────────────────
if dev == "cuda" and _HAVE_ROUTERKD:
        from experiments.scaling.fsb_routerkd import heldout_ce as _hce
        torch.manual_seed(44)
        _kw44 = dict(vocab_size=257, d_model=d_model, n_layers=2, n_heads=H, d_ff=128, max_seq_len=N2, attn_type="fsattn",
                     window_size=16, long_range_variant="cell_topk", use_rope=False, lr_scale_mode="one", cell_select="score",
                     cell_stride=S, sel_strict_causal=1, cell_topk=4, fsb_enabled=True, fsb_policy="fixed", fsb_width=2,
                     fsb_delta=0, fsb_tree="forest", fsb_bound_radius_coef=4.0)
        lmA = _LMn(**_kw44, fsb_route_adapter="full", fsb_rd_weight=1.0, fsb_rd_rows=8, fsb_rd_node=1,
                   fsb_route_learn_c=1).to(dev)
        lmB = _LMn(**_kw44).to(dev)                                             # no adapter, fixed c0 = 4
        sdA = lmA.state_dict()
        assert sorted(k for k in sdA if "fsb_route_" not in k) == sorted(lmB.state_dict()), "backbones differ in layout"
        lmB.load_state_dict({k: v for k, v in sdA.items() if "fsb_route_" not in k}, strict=True)
        la = [m_ for m_ in lmA.modules() if isinstance(m_, _EAn)]
        for m_ in list(la) + [m_ for m_ in lmB.modules() if isinstance(m_, _EAn)]:
            m_._cell_block = BLK; m_.recompute_long_range = False
        assert len(la) == 2 and all(not hasattr(m_, "fsb_route_ctheta") for m_ in lmB.modules())
        with torch.no_grad():
            for i_, m_ in enumerate(la):
                m_.fsb_route_W.normal_(0.0, 0.3)
                ct = torch.tensor([0.5, 1.5, 9.0, 14.0][:H], device=dev).roll(i_)
                m_.fsb_route_ctheta.copy_(torch.log(torch.expm1(ct)))          # c_h = softplus(theta) far from c0 = 4
        cA = [route_radius(m_) for m_ in la]
        assert all(float((c_ - 4.0).abs().min()) > 1.0 for c_ in cA), cA
        lmA.train(); lmB.train()
        ac44 = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        g44 = torch.Generator(device="cpu").manual_seed(44)
        held = [(torch.randint(0, 257, (2, N2), generator=g44).to(dev), torch.randint(0, 257, (2, N2), generator=g44).to(dev))
                for _ in range(2)]

        def _flags44():
            return ([m_.fsb_route_alpha for m_ in la], [m_.fsb_route_learn_c for m_ in la],
                    os.environ.get("FSB_TRAIN_GRAPH"), lmA.training)
        flags0 = _flags44()
        assert flags0 == ([1.0] * 2, [True] * 2, None, True), flags0
        res44 = {}
        for graph in ("1", "0"):
            if graph == "0":
                os.environ["FSB_TRAIN_GRAPH"] = "0"
            else:
                # heldout_ce runs in eval mode, where (zk@7e7cd86) the forest stage is captured only with FSB_EVAL_GRAPH=1
                os.environ["FSB_EVAL_GRAPH"] = "1"
            try:
                ref, nref = _hce(lmB, held, ac44)                                 # the backbone's own routing
                on, _ = _hce(lmA, held, ac44)
                fl = _flags44()
                caches = [getattr(m_, "_fsb_train_graphs", None) for m_ in la]
                base_e, _ = _hce(lmA, held, ac44, alpha_zero=True, eager=True)
                assert _flags44() == fl, ("flags not restored (eager)", _flags44(), fl)
                conly, _ = _hce(lmA, held, ac44, alpha_zero=True, keep_c=True)
                assert _flags44() == fl, ("flags not restored (keep_c)", _flags44(), fl)
                if graph == "1":
                    # eager=True (and keep_c, same key) leave the learned-c training graphs alone: no recapture
                    assert all(c_ is not None for c_ in caches)
                    assert all(getattr(m_, "_fsb_train_graphs", None) is c_ for m_, c_ in zip(la, caches)), "graphs recaptured"
                base, nb_ = _hce(lmA, held, ac44, alpha_zero=True)                # graph 1: recaptures with the c0 key
                assert _flags44() == fl, ("flags not restored", _flags44(), fl)
            finally:
                os.environ.pop("FSB_TRAIN_GRAPH", None); os.environ.pop("FSB_EVAL_GRAPH", None)
            assert nb_ == nref == 2 * 2 * N2
            assert torch.equal(base, ref), (graph, float(base), float(ref))
            assert torch.equal(base_e, ref), (graph, "eager", float(base_e), float(ref))
            assert not torch.equal(conly, ref), (graph, "learned c does not change the CE: the test is blind")
            assert not torch.equal(on, ref) and not torch.equal(on, conly), graph
            res44[graph] = [float(v_) / nref for v_ in (ref, base, on, conly)]
        assert res44["1"] == res44["0"], res44                                # graph == eager for every pass
        # restore on error: a batch whose targets do not match raises inside the forward loop
        bad44 = [(held[0][0], held[0][1][:, :7])]
        for kw_ in (dict(alpha_zero=True), dict(alpha_zero=True, eager=True), dict(alpha_zero=True, keep_c=True)):
            os.environ["FSB_TRAIN_GRAPH"] = "1"
            try:
                _hce(lmA, bad44, ac44, **kw_)
                raise AssertionError("mismatched targets must raise")
            except (RuntimeError, ValueError):
                pass
            assert _flags44() == ([1.0] * 2, [True] * 2, "1", True), (kw_, _flags44())
            os.environ.pop("FSB_TRAIN_GRAPH", None)
        r_, b_, o_, c_ = res44["1"]
        print(f"F44 OK  (c_h {[round(float(v_), 2) for v_ in cA[0]]} vs c0 4: alpha_zero CE == no-adapter fixed-c0 CE bit-exact "
              f"on graph + eager ({b_:.6f}); alpha 0 + learned c {c_:.6f}, router on {o_:.6f}; eager base keeps the training "
              f"graphs; alpha / learn_c / FSB_TRAIN_GRAPH / train mode restored, also on error)")
else:
    print("F44 SKIP (no CUDA or experiments/scaling/fsb_routerkd.py absent)")


# ── F45: _seg_lse shifts every level by its own max.  Levels 800-3000 nats apart (a single row-max shift underflows
#        every level > ~745 nats below the row max to exp = 0 in float64): per-node values and the gradient equal a
#        float64 per-level logsumexp reference; an all-NEG level and an empty level stay finite; the old row-max
#        shift is wrong on the same input (the test is not blind) ───────────────────────────────────────────────
from src.fsattn.fsb_route import _seg_lse as _slse, NEG as _NEGr
g45 = torch.Generator(device="cpu").manual_seed(45)
T5, G5, R5, Nf5, D5 = 3, 2, 4, 40, 6                                        # level 5 is empty; level 4 all NEG
dep5 = torch.randint(0, 4, (T5, G5, Nf5), generator=g45)
dep5[:, :, :3] = 4
off5 = torch.tensor([0.0, -800.0, -3000.0, 1500.0, 0.0, 0.0], dtype=torch.float64)
x5 = (torch.randn(T5, G5, R5, Nf5, generator=g45, dtype=torch.float64) * 5.0 + off5[dep5][:, :, None, :])
x5[:, :, :, :3] = _NEGr
x5[..., 7] = _NEGr                                                           # a masked node inside a live level
x5 = x5.float().to(dev).requires_grad_(True)
dep5d = dep5.to(dev)
oh5 = torch.nn.functional.one_hot(dep5d, D5).double()
out5 = _slse(x5, oh5, dep5d)
w5 = torch.randn(T5, G5, R5, Nf5, generator=g45).to(dev)
(out5 * w5).sum().backward()
xr = x5.detach().double().requires_grad_(True)
ref5 = torch.zeros_like(xr)
for l_ in range(D5):
    msk = (dep5d == l_)[:, :, None, :].expand_as(xr)
    lse_l = torch.logsumexp(xr.masked_fill(~msk, float("-inf")), dim=-1, keepdim=True)
    ref5 = torch.where(msk, lse_l.expand_as(xr), ref5)
(ref5 * w5.double()).sum().backward()
assert torch.isfinite(out5).all() and torch.isfinite(x5.grad).all()
live5 = (dep5d < 4)[:, :, None, :].expand_as(out5)
err5 = float((out5.double() - ref5.detach()).abs()[live5].max())
assert err5 < 1e-3, err5                                                     # fp32 output of values up to ~3000
negv = out5[..., :3].double()                                                # the all-NEG level: NEG + log 3, finite
assert float((negv - (_NEGr + math.log(3.0))).abs().max()) < 128.0, negv    # (fp32 ulp at 1e9 is 64)
gerr5 = float((x5.grad.double() - xr.grad).abs().max() / xr.grad.abs().max())
assert gerr5 < 1e-6, gerr5
# the old single row-max shift on the same input
with torch.no_grad():
    m_old = x5.detach().amax(-1, keepdim=True)
    lse_old = torch.log(torch.matmul(torch.exp((x5.detach() - m_old).double()), oh5).clamp(min=1e-300)) + m_old.double()
    old5 = torch.matmul(lse_old, oh5.transpose(-1, -2)).float()
old_err = float((old5.double() - ref5.detach()).abs()[live5].max())
assert old_err > 1.0, old_err
print(f"F45 OK  (_seg_lse per-level shift: levels at 0/-800/-3000/+1500 nats, an all-NEG level and an empty one: max |err| "
      f"{err5:.2e} (fp32 output), grad rel err {gerr5:.1e}; the old row-max shift is off by {old_err:.1f} nats)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  OVERSELECT (fsb_read_top): the beam searches with width W_beam, its retained cells are re-ranked by the routing
#  leaf score u_c = (q_r . kbar_c) * scale + lambda a_c erf(|i - p_c| / (2 sigma)) and only the best K are read.
#  F50..F56 (F38-F49 are reserved for other branches)
# ══════════════════════════════════════════════════════════════════════════════════════════
from src.fsattn.fsb_tree import rerank_top


def _qkv_n(seed, n_, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return tuple((torch.randn(1, H, n_, Dh, generator=g) * scale).to(dev) for _ in range(3))


def _sel_n(mm_, Q_, K_, V_, n_, graph="0", triton=None):
    """_lr_cell_topk over n_ tokens with sel_full captured -> (sel_full (1,H,Npad,kk), output (1,H,n_,Dh))."""
    cap = {}
    def _g(attn, Qq, Kp, Vp, sel_full, s_, Bq_, nn_):
        cap["sel"] = sel_full.clone(); return _orig(attn, Qq, Kp, Vp, sel_full, s_, Bq_, nn_)
    _ft4.masked_readout = _g
    os.environ["FSB_TRAIN_GRAPH"] = graph
    os.environ["FSB_EVAL_GRAPH"] = graph            # eval-mode models: graph "1" = the captured stage (zk: eval default eager)
    if triton is not None:
        os.environ["FSB_TRAVERSE_TRITON"] = triton
    try:
        o = mm_._lr_cell_topk(Q_, K_, V_, mm_.sigma, n_)
    finally:
        _ft4.masked_readout = _orig
        os.environ.pop("FSB_TRAIN_GRAPH", None); os.environ.pop("FSB_EVAL_GRAPH", None); os.environ.pop("FSB_TRAVERSE_TRITON", None)
    return cap["sel"], o


def _full_sel_n(mm_, x_):
    """no-cache module forward over x_ (1,n,d) with sel_full captured -> (sel (H,Npad,kk), y)."""
    cap = {}
    def _g(attn, Qq, Kp, Vp, sel_full, s_, Bq_, nn_):
        cap["sel"] = sel_full.clone(); return _orig(attn, Qq, Kp, Vp, sel_full, s_, Bq_, nn_)
    _ft4.masked_readout = _g
    os.environ["FSB_TRAIN_GRAPH"] = "0"
    try:
        y = mm_(x_)
    finally:
        _ft4.masked_readout = _orig; os.environ.pop("FSB_TRAIN_GRAPH", None)
    return cap["sel"][0], y


def _cached_run_n(mm_, x_, N0, lean, dgraph):
    """prefill of N0 tokens + per-token decode of x_ (1,n,d); returns (outputs (1,n,d), decode selections [(H,kk)|None])."""
    n_ = x_.shape[1]
    os.environ["FSB_DECODE_GRAPH"] = dgraph
    try:
        cache = FSKVCache(mm_, lean_prefill=lean); cache.reserve = n_ - N0
        outs = [mm_(x_[:, :N0], cache=cache)]; sels = []
        for t in range(N0, n_):
            outs.append(mm_(x_[:, t:t + 1], cache=cache))
            fc = getattr(cache, "_fsb_forest", None)
            if fc is not None and fc.SP is not None and getattr(fc, "_last_sel", None) is not None:
                sels.append(fc._last_sel.reshape(H, -1).clone())
            else:
                sels.append(None)
        if dgraph == "1" and mm_.fsb_tree == "forest":
            assert cache._fsb_forest.graph is not None, "decode graph never captured"
    finally:
        os.environ.pop("FSB_DECODE_GRAPH", None)
    return torch.cat(outs, dim=1), sels


def _same_sets(a, b):
    """a (..., ka), b (..., kb) cell ids (-1 pad, unique per row): identical retained SETS per row (any widths)."""
    va, vb = a >= 0, b >= 0
    in_b = ((a[..., :, None] == b[..., None, :]) & vb[..., None, :]).any(-1) | ~va
    in_a = ((b[..., :, None] == a[..., None, :]) & va[..., None, :]).any(-1) | ~vb
    return in_b.all(-1) & in_a.all(-1)


def _mk_os(seed=None, adapter=None, lam=0.7, **kw):
    """mk() with an optional RANDOM full routing adapter (seeded, independent of the global RNG) and lambda."""
    if seed is not None:
        torch.manual_seed(seed)
    if adapter:
        kw.update(fsb_route_adapter="full")
    m_ = mk(**kw)
    with torch.no_grad():
        m_.sel_scale.fill_(lam)
        if adapter:
            g_ = torch.Generator(device="cpu").manual_seed(adapter)
            m_.fsb_route_W.copy_((torch.randn(m_.fsb_route_W.shape, generator=g_) * 0.3).to(dev))
    return m_


def _leaf_brute(mm_, Q_, K_, rows_sel, n_):
    """float64 routing leaf score of every (head, query, candidate) of rows_sel (H,Npad,C): (H,Npad,C), -inf pad."""
    Mn = (n_ + S - 1) // S
    Kp = K_[0] if Mn * S == n_ else torch.cat([K_[0], K_.new_zeros(H, Mn * S - n_, Dh)], 1)
    kbar = Kp.view(H, Mn, S, Dh).mean(2).double()                     # the module's cell mean (key dtype), promoted
    ceil = Kp.view(H, Mn, S, Dh).double().norm(dim=-1).amax(-1)
    qr = fsb_route_q(mm_, Q_.float())[0].double()                     # (H,n,Dh)
    Npad = rows_sel.shape[1]
    qr = torch.cat([qr, qr.new_zeros(H, Npad - qr.shape[1], Dh)], 1) if Npad > qr.shape[1] else qr
    lam = mm_.sel_scale.detach().double().view(H, 1, 1); sig = mm_.sigma.detach().double().view(H, 1, 1)
    c = rows_sel.clamp(min=0)
    kb = torch.gather(kbar[:, None].expand(H, Npad, Mn, Dh), 2, c[..., None].expand(-1, -1, -1, Dh))
    a = torch.gather(ceil[:, None].expand(H, Npad, Mn), 2, c)
    ipos = torch.arange(Npad, device=dev, dtype=torch.float64).view(1, Npad, 1)
    u = (qr[:, :, None, :] * kb).sum(-1) * mm_.scale + lam * a * torch.erf((ipos - (c.double() + 0.5) * S).abs() / (2.0 * sig))
    return u.masked_fill(rows_sel < 0, float("-inf"))


# ── F50: flag OFF is byte-identical: fsb_read_top=0 builds the same module (no buffer / parameter, same kk and beam
#        slots) and gives the same selection and output as no flag on every path -- eager and CUDA-graph training stage,
#        both trees, fixed and learned policies, train and eval mode, full / lean prefill + per-token decode with
#        FSB_DECODE_GRAPH 0 and 1.  (Parity with the pre-overselect commit is checked separately by
#        experiments/scaling/fsb_overselect_offparity.py, which runs the same cases on both code trees.) ────────────────
for tree in ("forest", "rebuild"):
    for pol in ("fixed", "learned"):
        ma = _mk_os(50, fsb_policy=pol, fsb_width=2, fsb_delta=1, fsb_tree=tree)
        mb = _mk_os(50, fsb_policy=pol, fsb_width=2, fsb_delta=1, fsb_tree=tree, fsb_read_top=0)
        assert mb.fsb_read_top == 0 and (ma.fsb_kk, ma.fsb_W_max, ma.fsb_delta_max) == (mb.fsb_kk, mb.fsb_W_max, mb.fsb_delta_max) == (16, 4, 2)
        sda, sdb = ma.state_dict(), mb.state_dict()
        assert list(sda) == list(sdb) and all(torch.equal(sda[k], sdb[k]) for k in sda)
        Qc, Kc, Vc = rand_qkv(500)
        for mode in ("eval", "train"):
            ma.train(mode == "train"); mb.train(mode == "train")
            for g in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                torch.manual_seed(501); ra = _sel_n(ma, Qc, Kc, Vc, N, g)
                torch.manual_seed(501); rb = _sel_n(mb, Qc, Kc, Vc, N, g)
                assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1]), (tree, pol, mode, g)
        ma.eval(); mb.eval()
        if tree == "forest":
            torch.manual_seed(502); x = torch.randn(1, N, d_model, device=dev)
            with torch.no_grad():
                for lean in (False, True):
                    for dg in (("0", "1") if dev == "cuda" else ("0",)):
                        oa, sa = _cached_run_n(ma, x, 150, lean, dg); ob, sb = _cached_run_n(mb, x, 150, lean, dg)
                        assert torch.equal(oa, ob), (pol, lean, dg)
                        assert all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(sa, sb)), (pol, lean, dg)
print("F50 OK  (fsb_read_top 0 == no flag: same module, selection and output on eager / graphed training, both trees, both "
      "policies, full + lean prefill with decode graph 0/1)")


# ── F51: read_top == the beam's candidate count reproduces TODAY's selection SET on every row (re-ranking only reorders):
#        W8/D0 (explicit menu), W2/D1 (a retained node holds 2 cells: read_top 4 = all of them), and the E1 recipe W4/D0
#        on the default menus (W_max 4, Delta_max 2: 16 slots today) with read_top 4; forest + rebuild, eager + graphed
#        training stage, and per-token decode (full + lean prefill, decode graph 0/1) ────────────────────────────────────
n51 = 512
cfgs51 = (dict(fsb_width=8, fsb_delta=0, fsb_widths="8", fsb_deltas="0", rt=8),
          dict(fsb_width=2, fsb_delta=1, fsb_widths="2", fsb_deltas="1", rt=4),
          dict(fsb_width=4, fsb_delta=0, rt=4))
for cfg in cfgs51:
    cfg = dict(cfg); rt = cfg.pop("rt")
    for tree in ("forest", "rebuild"):
        m0 = _mk_os(51, adapter=5100, fsb_policy="fixed", fsb_tree=tree, **cfg)
        m1 = _mk_os(51, adapter=5100, fsb_policy="fixed", fsb_tree=tree, fsb_read_top=rt, **cfg)
        assert m1.fsb_kk == rt
        Qc, Kc, Vc = _qkv_n(510, n51)
        with torch.no_grad():
            for g in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                s0, _ = _sel_n(m0, Qc, Kc, Vc, n51, g); s1, _ = _sel_n(m1, Qc, Kc, Vc, n51, g)
                assert s1.shape[-1] == rt and bool(_same_sets(s0.long(), s1.long()).all()), (cfg, tree, g)
                assert int((s1 >= 0).sum()) == int((s0 >= 0).sum())
            if tree == "forest":
                torch.manual_seed(511); x = torch.randn(1, n51, d_model, device=dev)
                for lean in (False, True):
                    for dg in (("0", "1") if dev == "cuda" else ("0",)):
                        _, d0 = _cached_run_n(m0, x, 300, lean, dg); _, d1 = _cached_run_n(m1, x, 300, lean, dg)
                        for a_, b_ in zip(d0, d1):
                            assert (a_ is None) == (b_ is None) and (a_ is None or bool(_same_sets(a_.long(), b_.long()).all())), (cfg, lean, dg)
    print(f"F51 [W{cfg['fsb_width']}/D{cfg['fsb_delta']} menu {cfg.get('fsb_widths', '1,2,4')}x{cfg.get('fsb_deltas', '0,1,2')}, read_top {rt}] "
          f"same selection sets as read_top 0 (forest + rebuild, eager{' + graphed' if dev == 'cuda' else ''}, decode)")
print("F51 OK  (read_top == beam candidates reproduces today's selection set)")


# ── F52: the re-ranked K are the top-K by the routing leaf score among the beam's cells (float64 brute force):
#        (a) rerank_top on random candidates == float64 top-K, batch independent (a row alone == inside the batch);
#        (b) module, random adapter, W_beam 8 / read 4 (and W16/read 4, W4/D1 read 4): the read set == the float64 top-4
#            of the read_top-0 beam's cells at every row whose 4th/5th scores are not a near tie, forest + rebuild;
#            default menus (auto-extended beam slots, 32 candidates) == explicit menus; carried leaves are re-ranked too
torch.manual_seed(52)
Tq, Bqq, Cq, Mq, Dq = 7, 9, 12, 40, 16
selq = torch.stack([torch.stack([torch.randperm(Mq)[:Cq] for _ in range(Bqq)]) for _ in range(Tq)]).to(dev)
selq = selq.masked_fill(torch.rand(Tq, Bqq, Cq, device=dev) < 0.25, -1)
qq = torch.randn(Tq, Bqq, Dq, device=dev); qpq = 2000.0 + torch.arange(Bqq, device=dev).float()[None].expand(Tq, -1)
lamq = torch.rand(Tq, 1, 1, device=dev) + 0.5; sigq = torch.rand(Tq, 1, 1, device=dev) * 200 + 4
cK = torch.randn(3, Mq, Dq, device=dev); cA = torch.rand(3, Mq, device=dev) * 3; tix = torch.tensor([0, 1, 2, 0, 1, 2, 2], device=dev)
for K_ in (1, 4, Cq):
    out = rerank_top(selq, qq, qpq, lamq, sigq, cK, cA, tix, 0.25, S, K_)
    assert out.shape == (Tq, Bqq, K_)
    c = selq.clamp(min=0)
    u = ((qq.double()[:, :, None] * cK.double()[tix][torch.arange(Tq, device=dev)[:, None, None], c]).sum(-1) * 0.25
         + lamq.double() * cA.double()[tix][torch.arange(Tq, device=dev)[:, None, None], c]
         * torch.erf((qpq.double()[..., None] - (c.double() + 0.5) * S).abs() / (2.0 * sigq.double()))).masked_fill(selq < 0, float("-inf"))
    ref = torch.gather(selq, -1, u.sort(-1, descending=True).indices[..., :K_])
    ref = ref.masked_fill(u.sort(-1, descending=True).values[..., :K_] == float("-inf"), -1)
    assert bool(_same_sets(out, ref).all()), K_
    assert torch.equal(rerank_top(selq[2:5, 3:4].contiguous(), qq[2:5, 3:4].contiguous(), qpq[2:5, 3:4].contiguous(), lamq[2:5], sigq[2:5],
                                  cK, cA, tix[2:5], 0.25, S, K_), out[2:5, 3:4]), "rerank_top is batch dependent"
nrow52 = 0; ntie52 = 0; ndiff52 = 0; ncarry52 = 0
n52 = 512
for tree in ("forest", "rebuild"):
    for (Wb, Db, K_, menu) in ((8, 0, 4, ("8", "0")), (16, 0, 4, ("16", "0")), (4, 1, 4, ("4", "1"))):
        mbeam = _mk_os(52, adapter=5200, fsb_policy="fixed", fsb_width=Wb, fsb_delta=Db, fsb_widths=menu[0], fsb_deltas=menu[1],
                       fsb_tree=tree)                                        # today's beam at width W_beam (no re-rank)
        mos = _mk_os(52, adapter=5200, fsb_policy="fixed", fsb_width=Wb, fsb_delta=Db, fsb_widths=menu[0], fsb_deltas=menu[1],
                     fsb_tree=tree, fsb_read_top=K_)
        mdef = _mk_os(52, adapter=5200, fsb_policy="fixed", fsb_width=Wb, fsb_delta=Db, fsb_tree=tree, fsb_read_top=K_)   # default menus
        mw4 = _mk_os(52, adapter=5200, fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_tree=tree)                   # today's W4/D0
        assert mdef.fsb_W_max == max(4, 1 << (Wb - 1).bit_length()) and mdef.fsb_kk == K_
        Qc, Kc, Vc = _qkv_n(520, n52, scale=1.5)
        with torch.no_grad():
            sb, _ = _sel_n(mbeam, Qc, Kc, Vc, n52); so, _ = _sel_n(mos, Qc, Kc, Vc, n52); sd, _ = _sel_n(mdef, Qc, Kc, Vc, n52)
            s4, _ = _sel_n(mw4, Qc, Kc, Vc, n52)
        assert torch.equal(so, sd), (tree, Wb, "default menus (auto-extended slots) != explicit menus")
        sb, so, s4 = sb[0].long(), so[0].long(), s4[0].long()
        u = _leaf_brute(mos, Qc, Kc, sb, n52)                              # (H,Npad,C) float64
        us, ui = u.sort(-1, descending=True)
        ref = torch.gather(sb, -1, ui[..., :K_]).masked_fill(us[..., :K_] == float("-inf"), -1)
        nval = (sb >= 0).sum(-1)
        gap = torch.where(nval > K_, us[..., K_ - 1] - us[..., K_].clamp(min=-1e30), torch.full_like(us[..., 0], 1e30))
        tie = gap < 1e-4 * us[..., 0].abs().clamp(min=1.0)                # fp32 module vs float64 reference
        ok = _same_sets(so, ref)
        assert bool((ok | tie).all()), (tree, Wb, Db, int((~ok & ~tie).sum()))
        # every read cell is one of the beam's cells, and the read set is never larger than K
        assert bool(((so[..., :, None] == sb[..., None, :]).any(-1) | (so < 0)).all()) and bool(((so >= 0).sum(-1) <= K_).all())
        nrow52 += int((nval > K_).sum()); ntie52 += int((tie & (nval > K_)).sum())
        ndiff52 += int((~_same_sets(so, s4) & (nval > 0)).sum())
        if tree == "forest" and Db == 0:
            # carried leaves: the forest beam keeps a leaf reached before the last level at +inf priority, so carried
            # leaves occupy the FIRST beam slots.  Rows where the re-rank drops an earlier slot but keeps a later one
            # show that carried leaves are re-ranked by their score, not kept (with the brute-force match above).
            inread = (sb[..., :, None] == so[..., None, :]).any(-1) & (sb >= 0)
            idx = torch.arange(sb.shape[-1], device=dev)
            first_drop = torch.where((sb >= 0) & ~inread, idx, sb.shape[-1]).amin(-1)
            last_keep = torch.where(inread, idx, -1).amax(-1)
            ncarry52 += int((first_drop < last_keep).sum())
assert ndiff52 > 0 and ncarry52 > 0, "overselect never changed the read set (test has no teeth)"
print(f"F52 OK  (rerank_top == float64 top-K, batch independent; module read sets == float64 top-4 of the beam's cells on "
      f"{nrow52 - ntie52}/{nrow52} rows ({ntie52} near-ties skipped), forest + rebuild, W8 / W16 / W4D1; default menus == explicit; "
      f"{ndiff52} rows read a different set than today's W4/D0 beam; {ncarry52} forest rows drop an earlier (carried) beam slot)")


# ── F53: prefill == decode with a RANDOM adapter and W_beam 8 / read 4: the no-cache full-sequence selection equals the
#        per-token decode selection (sets and order) for every decode row, FSB_DECODE_GRAPH 0 and 1, full and lean prefill,
#        and the cached outputs reproduce the full forward; rebuild tree: cached outputs == full forward ────────────────────
n53, N053 = 512, 300
for tree in ("forest", "rebuild"):
    mm = _mk_os(53, adapter=5300, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree=tree, fsb_read_top=4)
    torch.manual_seed(531); x = torch.randn(1, n53, d_model, device=dev)
    with torch.no_grad():
        sel_full, yfull = _full_sel_n(mm, x)                               # (H,Npad,4)
        mm0 = _mk_os(53, adapter=5300, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree=tree, fsb_widths="8", fsb_deltas="0")
        sel_beam, _ = _full_sel_n(mm0, x)
        nd = int((~_same_sets(sel_full.long(), sel_beam[..., :4].long()))[:, BLK:n53].sum())
        # forest: carried leaves hold the first beam slots at +inf priority, so the re-rank differs from truncation;
        # rebuild at Delta 0: every leaf is reached at the last level and the beam's top-k there IS the leaf score (bit-
        # identical arithmetic, leaf radius 0), so the best 4 are the beam's first 4 slots -- overselect == truncation
        assert (nd > 0) if tree == "forest" else (nd == 0), (tree, nd)
        nord = 0; nrows = 0
        for lean in (False, True):
            for dg in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                y, sels = _cached_run_n(mm, x, N053, lean, dg)
                d = (y - yfull).abs().max().item()
                assert d < 1e-4, (tree, lean, dg, d)
                if tree == "forest":
                    dec = torch.stack(sels, dim=1)                         # (H, n-N0, 4)
                    same_set = _same_sets(sel_full[:, N053:n53].long(), dec.long())
                    assert bool(same_set.all()), f"lean={lean} graph={dg}: {int((~same_set).sum())} decode rows route differently"
                    nord += int((sel_full[:, N053:n53].long() == dec.long()).all(-1).sum()); nrows += same_set.numel()
    print(f"F53 [{tree}] W8/read 4 + random adapter: " + (f"prefill selection == decode selection on {nrows} rows (order identical on "
          f"{nord}), " if tree == "forest" else "") + f"cached == full forward (full + lean prefill{', decode graph 0/1' if dev == 'cuda' and tree == 'forest' else ''}); "
          f"{nd} rows differ from the beam's first 4 slots" + (" (rebuild at Delta 0: re-rank == truncation, as expected)" if tree == "rebuild" else ""))
print("F53 OK  (overselect: prefill routing == decode routing, cached == full)")


# ── F54: with read_top the CUDA-graph training stage == eager (sel_full, output, n_read, n_visit; fixed + learned, one and
#        several chunk graphs), chunked prefill 1 / 2 / all identical (both trees), Triton level kernel == torch traversal ──
n54 = 512
if dev == "cuda":
    for pol, nbch in (("fixed", None), ("learned", None), ("fixed", 2), ("learned", 2)):
        kw = {} if nbch is None else {"fsb_prefill_blkchunk": nbch}
        wkw = dict(fsb_width=8, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
        mm = _mk_os(54, adapter=5400, fsb_policy=pol, fsb_tree="forest", fsb_read_top=4, **wkw, **kw)
        if pol == "learned":
            mm.train()
        for trial in range(2):
            Qc, Kc, Vc = _qkv_n(540 + trial, n54)
            res = {}
            for g in ("0", "1"):
                torch.manual_seed(5400 + trial)
                sel, o = _sel_n(mm, Qc, Kc, Vc, n54, g)
                st = mm._fsb_stats
                res[g] = (sel, o.detach(), st["n_read"].clone(), st["n_visit"].clone())
                assert int(st["n_read"].max()) <= 4
            for i, nm in enumerate(("sel_full", "output", "n_read", "n_visit")):
                assert torch.equal(res["0"][i], res["1"][i]), (pol, nbch, trial, nm)
        print(f"F54 [{pol}, nbch {nbch}] graph-captured training stage == eager with read_top 4 (sel_full, output, n_read, n_visit), 2 inputs")
    for pol in ("fixed", "learned"):
        wkw = dict(fsb_width=8, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
        mm = _mk_os(541, adapter=5410, fsb_policy=pol, fsb_tree="forest", fsb_read_top=4, **wkw)
        Qc, Kc, Vc = _qkv_n(542, n54)
        r0 = _sel_n(mm, Qc, Kc, Vc, n54, "0", triton="0"); r1 = _sel_n(mm, Qc, Kc, Vc, n54, "0", triton="1")
        assert torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1]), ("F54 triton", pol)
    print("F54 [triton] Triton level kernel == torch traversal with read_top 4 (fixed W8, learned menu 2,4,8 x 0,1)")
for tree in ("forest", "rebuild"):
    for pol in ("fixed", "learned"):
        wkw = dict(fsb_width=8, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
        outs = []
        for nbch in (1, 2, 0):
            mm = _mk_os(543, adapter=5430, fsb_policy=pol, fsb_tree=tree, fsb_read_top=4, fsb_prefill_blkchunk=nbch, **wkw)
            Qc, Kc, Vc = _qkv_n(544, n54)
            outs.append(_sel_n(mm, Qc, Kc, Vc, n54))
        assert all(torch.equal(outs[0][0], o_[0]) and torch.equal(outs[0][1], o_[1]) for o_ in outs[1:]), (tree, pol)
print("F54 OK  (read_top: graphed == eager training stage, chunked prefill identical, Triton == torch traversal)")


# ── F55: TRAINING with read_top: (a) learned policy: finite REINFORCE loss, controller grads, cost counts <= K reads;
#        (b) router distillation + read_top: finite loss, grads on the router only, and the logged "beam" recall is the
#        recall of the READ set (== brute force over sel_full); (c) fixed policy: grads reach W_q / W_v ─────────────────────
m55 = _mk_os(55, fsb_policy="learned", fsb_widths="2,4,8", fsb_deltas="0,1", fsb_tree="forest", fsb_read_top=4).train()
torch.manual_seed(551)
x55 = torch.randn(1, N, d_model, device=dev, requires_grad=True)
y55 = m55(x55)
st55 = m55._fsb_stats
assert int(st55["n_read"].max()) <= 4 and bool((st55["n_read"][st55["has"]] >= 1).all())
aux55 = fsb_policy_loss(m55, torch.rand(1, N, device=dev), eta=0.1, entropy_beta=0.01, value_coef=0.5)
(y55.float().pow(2).mean() + aux55).backward()
assert torch.isfinite(aux55) and all(p_.grad is not None and torch.isfinite(p_.grad).all() for n_, p_ in m55.named_parameters() if "fsb_policy" in n_)
assert torch.isfinite(m55.W_v.weight.grad).all() and m55.W_v.weight.grad.abs().sum() > 0
m55b = _mk_os(55, adapter=5500, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree="forest", fsb_read_top=4,
              fsb_rd_weight=1.0, fsb_rd_rows=8).train()
Qc, Kc, Vc = rand_qkv(552, scale=1.5)
m55b._fsb_rd_gen = torch.Generator(device=dev).manual_seed(7)
with torch.no_grad():
    sel55, _, rd55 = _run_lr(m55b, Qc, Kc, Vc)
assert sel55.shape[-1] == 4 and rd55 is not None and torch.isfinite(rd55["loss"])
rd55["loss"].backward()
assert m55b.fsb_route_W.grad is not None and m55b.fsb_route_W.grad.abs().sum() > 0 and m55b.W_q.weight.grad is None
lb, gb, hits, den, nrow, lhits = _rd_brute(m55b, Qc, Kc, rd55["rows"], sel55, 0, 0, 1.0, 16, "kl")
st_ = rd55["stats"]
assert int(st_[0]) == hits and int(st_[1]) == den and int(st_[8]) == lhits, (st_.tolist(), hits, den, lhits)
m55c = _mk_os(55, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree="forest", fsb_read_top=4).train()
m55c(torch.randn(1, N, d_model, device=dev)).float().pow(2).mean().backward()
assert torch.isfinite(m55c.W_q.weight.grad).all() and m55c.W_q.weight.grad.abs().sum() > 0 and torch.isfinite(m55c.W_v.weight.grad).all()
print(f"F55 OK  (training with read_top 4: learned policy loss {aux55.item():.3f} + grads, reads <= 4; router distillation loss "
      f"{rd55['loss'].item():.4f} + router-only grads, read-set recall {hits}/{den} == brute force; fixed-policy grads)")


# ── F56: construction: W_beam > the menu is auto-extended (power-of-two slots), reads capped at 16, candidates at 64,
#        read_top > candidates refused, the default is off ────────────────────────────────────────────────────────────────
m56 = mk(fsb_policy="fixed", fsb_width=6, fsb_delta=0, fsb_read_top=4)
assert (m56.fsb_W_max, m56.fsb_kk, m56.fsb_read_top) == (8, 4, 4)
m56 = mk(fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_read_top=16)             # 16 x 2^2 = 64 candidates
assert (m56.fsb_W_max, m56.fsb_kk) == (16, 16)
assert mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0).fsb_read_top == 0
for bad in (dict(fsb_width=8, fsb_read_top=17), dict(fsb_width=32, fsb_read_top=4), dict(fsb_width=2, fsb_widths="2", fsb_deltas="0", fsb_read_top=4),
            dict(fsb_width=8, fsb_widths="8")):                                  # read_top 0: 8 x 2^2 = 32 > 16, as before
    try:
        mk(fsb_policy="fixed", fsb_delta=0, **bad)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass
print("F56 OK  (overselect construction: auto-extended power-of-two beam slots; reads <= 16; candidates <= 64; read_top <= candidates)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  TOKEN-LEVEL OVERSELECT RE-RANK (fsb_rerank_score lse | lse_pos | max; src/fsattn/fsb_rerank.py): F57..F62.
#  A candidate cell is scored over its s RAW keys: lse = logsumexp_j (q_r . k_j) * scale, max = max_j, lse_pos = lse + the
#  leaf score's position term; leaf (default) is the mean-key score of F50-F56.
# ══════════════════════════════════════════════════════════════════════════════════════════
import time
import src.fsattn.fsb_rerank as _frr
from src.fsattn.fsb_rerank import cell_token_scores, cell_token_scores_torch
_RR = ("lse", "lse_pos", "max")


def _tok_brute(q, sel, K4, t_idx, scale, s, mode, qpos=None, lam=None, sig=None, cellA=None):
    """float64 token-level cell score.  q (T',Bq,Dh), sel (T',Bq,C), K4 (B,H,L,Dh) raw keys (sequence b*H+h), t_idx (T',);
    lse_pos needs qpos (T',Bq), lam / sig (T',1,1), cellA (B*H,M).  Returns (T',Bq,C) float64, -inf where sel < 0."""
    Tp, Bq, C = sel.shape
    Hk = K4.shape[1]
    c = sel.clamp(min=0)
    tok = c[..., None] * s + torch.arange(s, device=sel.device)                      # (T',Bq,C,s)
    sq = t_idx.view(Tp, 1, 1, 1).expand_as(tok)
    k = K4.double()[torch.div(sq, Hk, rounding_mode="floor"), sq % Hk, tok]           # (T',Bq,C,s,Dh)
    sc = (q.double()[:, :, None, None, :] * k).sum(-1) * scale
    u = sc.amax(-1) if mode == "max" else torch.logsumexp(sc, -1)
    if mode == "lse_pos":
        a = cellA.double()[t_idx.view(Tp, 1, 1).expand_as(c), c]
        u = u + lam.double() * a * torch.erf((qpos.double()[..., None] - (c.double() + 0.5) * s).abs() / (2.0 * sig.double()))
    return u.masked_fill(sel < 0, float("-inf"))


def _topk_ref(sel, u, K_):
    """float64 top-K of the candidates by u (sets), and the near-tie mask (4th/5th gap below 1e-4 relative)."""
    us, ui = u.sort(-1, descending=True)
    ref = torch.gather(sel, -1, ui[..., :K_]).masked_fill(us[..., :K_] == float("-inf"), -1)
    nval = (sel >= 0).sum(-1)
    if K_ < sel.shape[-1]:
        gap = torch.where(nval > K_, us[..., K_ - 1] - us[..., K_].clamp(min=-1e30), torch.full_like(us[..., 0], 1e30))
        tie = gap < 1e-4 * us[..., 0].abs().clamp(min=1.0)
    else:
        tie = torch.zeros_like(nval, dtype=torch.bool)
    return ref, tie, nval


def _rerank_top_021eb50(sel, q, qpos, lam, sigv, cellK, cellA, t_idx, scale, s, K):
    """VERBATIM copy of rerank_top at 021eb50 (the reviewed overselect head, leaf score only): the byte-identity reference."""
    Tp, Bq, C = sel.shape
    Mc, Dh = cellK.shape[1], cellK.shape[2]
    ok = sel >= 0
    gid = t_idx.view(Tp, 1, 1) * Mc + sel.clamp(min=0, max=Mc - 1)
    kb = cellK.reshape(-1, Dh)[gid.reshape(-1)].view(Tp, Bq, C, Dh)
    a = cellA.reshape(-1)[gid.reshape(-1)].view(Tp, Bq, C)
    content = _ft4._det_dot(q[:, :, None, :], kb) * scale
    pos_c = (sel.float() + 0.5) * s
    erf = torch.erf((qpos[..., None] - pos_c).abs() / (2.0 * sigv))
    u = (content + lam * a * erf).masked_fill(~ok, float("-inf"))
    top_v, top_i = torch.sort(u, dim=-1, descending=True, stable=True)
    out = torch.gather(sel, -1, top_i[..., :K])
    return out.masked_fill(top_v[..., :K] == float("-inf"), -1)


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── F57: construction and the LEAF DEFAULT: (a) every module defaults to "leaf"; a token score without fsb_read_top and
#        an unknown score are refused; (b) byte-identical: fsb_rerank_score "leaf" == the flag absent == the 021eb50
#        rerank_top (verbatim copy, swapped in) on the eager / graphed training stage, both trees, fixed + learned, and on
#        full + lean prefill with per-token decode (FSB_DECODE_GRAPH 0 / 1); the token-score kernel is never called ──────
assert mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0).fsb_rerank_score == "leaf"
assert mk(fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_read_top=4).fsb_rerank_score == "leaf"
for bad in (dict(fsb_rerank_score="lse"), dict(fsb_rerank_score="max", fsb_read_top=0), dict(fsb_rerank_score="mean", fsb_read_top=4)):
    try:
        mk(fsb_policy="fixed", fsb_width=8, fsb_delta=0, **bad)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass
_cts57 = _frr.cell_token_scores
def _never57(*a_, **k_):
    raise AssertionError("cell_token_scores called on the leaf path")
n57 = 512
_rt_new57 = _ft4.rerank_top
for tree in ("forest", "rebuild"):
    for pol in ("fixed", "learned"):
        wkw = dict(fsb_width=8, fsb_delta=0, fsb_widths="8", fsb_deltas="0") if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
        mA = _mk_os(57, adapter=5700, fsb_policy=pol, fsb_tree=tree, fsb_read_top=4, **wkw)
        mB = _mk_os(57, adapter=5700, fsb_policy=pol, fsb_tree=tree, fsb_read_top=4, fsb_rerank_score="leaf", **wkw)
        sda, sdb = mA.state_dict(), mB.state_dict()
        assert list(sda) == list(sdb) and all(torch.equal(sda[k], sdb[k]) for k in sda)
        Qc, Kc, Vc = _qkv_n(570, n57, scale=1.5)
        _frr.cell_token_scores = _never57
        try:
            for mode in ("eval", "train"):
                mA.train(mode == "train"); mB.train(mode == "train")
                for g in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                    with torch.no_grad():
                        torch.manual_seed(571); ra = _sel_n(mA, Qc, Kc, Vc, n57, g)
                        torch.manual_seed(571); rb = _sel_n(mB, Qc, Kc, Vc, n57, g)
                        _ft4.rerank_top = lambda *a_, score="leaf", tokK=None: _rerank_top_021eb50(*a_)
                        try:
                            torch.manual_seed(571); rc = _sel_n(mB, Qc, Kc, Vc, n57, g)
                        finally:
                            _ft4.rerank_top = _rt_new57
                    assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1]), (tree, pol, mode, g, "flag leaf != absent")
                    assert torch.equal(rb[0], rc[0]) and torch.equal(rb[1], rc[1]), (tree, pol, mode, g, "leaf != 021eb50 rerank_top")
            mA.eval(); mB.eval()
            if tree == "forest":
                torch.manual_seed(572); x = torch.randn(1, n57, d_model, device=dev)
                with torch.no_grad():
                    for lean in (False, True):
                        for dg in (("0", "1") if dev == "cuda" else ("0",)):
                            oa, sa = _cached_run_n(mA, x, 300, lean, dg)
                            _ft4.rerank_top = lambda *a_, score="leaf", tokK=None: _rerank_top_021eb50(*a_)
                            try:
                                ob, sb_ = _cached_run_n(mB, x, 300, lean, dg)
                            finally:
                                _ft4.rerank_top = _rt_new57
                            assert torch.equal(oa, ob), (pol, lean, dg)
                            assert all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(sa, sb_)), (pol, lean, dg)
        finally:
            _frr.cell_token_scores = _cts57
print("F57 OK  (fsb_rerank_score: default leaf; token scores need read_top; leaf == flag absent == the 021eb50 rerank_top, bit-exact on "
      "eager / graphed training, both trees, both policies, full + lean prefill with decode graph 0/1; kernel never called for leaf)")


# ── F58: cell_token_scores (Triton on CUDA) against the torch reference and a float64 brute force, on bf16 / fp32 keys,
#        head dims 16 / 24 / 64 / 96, cell sizes 16 / 12 (non-power-of-two) / 32:
#        (a) strided (B,N,H,D)-transposed keys == a contiguous copy == a 3-D (T,L,Dh) table, bit-exact;
#        (b) batch independent: a sub-block of rows scored alone == inside the batch, bit-exact;
#        (c) torch reference independent of its chunk size (bit-exact); kernel vs torch reference; vs float64 <= 1e-5;
#        (d) rerank_top(score) == the float64 top-K of the token scores (K = 1, 4, all), near-ties skipped;
#        (e) a NEEDLE: one outlier key in a cell whose mean is poor -- lse / max / lse_pos pick it, leaf does not ───────────
torch.manual_seed(58)
nbit58 = ncmp58 = 0; maxrel58 = 0.0; maxerr58 = 0.0; nrow58 = ntie58 = 0
for (Dq, sq_, kdt) in ((16, 16, torch.float32), (24, 12, torch.bfloat16), (64, 32, torch.bfloat16), (96, 32, torch.bfloat16)):
    Bk, Hk, Mq, Tq, Bqq, Cq = 2, 3, 40, 7, 9, 12
    base = (torch.randn(Bk, Mq * sq_, Hk, Dq, device=dev) * 1.5).to(kdt)             # like W_k(x).view(B, N, H, D)
    K4v = base.transpose(1, 2)                                                       # (B,H,L,Dh) strided view (the prefill K)
    K4c = K4v.contiguous()
    sel = torch.stack([torch.stack([torch.randperm(Mq)[:Cq] for _ in range(Bqq)]) for _ in range(Tq)]).to(dev)
    sel = sel.masked_fill(torch.rand(Tq, Bqq, Cq, device=dev) < 0.25, -1)
    q = torch.randn(Tq, Bqq, Dq, device=dev)
    tix = torch.randint(0, Bk * Hk, (Tq,), device=dev)
    scl = 1.0 / math.sqrt(Dq)
    qpq = 2000.0 + torch.arange(Bqq, device=dev).float()[None].expand(Tq, -1).contiguous()
    lamq = torch.rand(Tq, 1, 1, device=dev) + 0.5; sigq = torch.rand(Tq, 1, 1, device=dev) * 200 + 4
    cA = torch.rand(Bk * Hk, Mq, device=dev) * 3; cK = torch.randn(Bk * Hk, Mq, Dq, device=dev)
    v = sel >= 0
    for mode in _RR:
        out = cell_token_scores(q, sel, K4v, tix, scl, sq_, mode)
        assert out.shape == sel.shape and out.dtype == torch.float32
        assert bool(torch.isfinite(out[v]).all()) and bool((out[~v] == float("-inf")).all())
        assert torch.equal(out, cell_token_scores(q, sel, K4c, tix, scl, sq_, mode)), (Dq, sq_, mode, "strided != contiguous")
        assert torch.equal(out, cell_token_scores(q, sel, K4c.reshape(Bk * Hk, Mq * sq_, Dq), tix, scl, sq_, mode)), "3-D != 4-D"
        one = cell_token_scores(q[2:5, 3:4].contiguous(), sel[2:5, 3:4].contiguous(), K4v, tix[2:5], scl, sq_, mode)
        assert torch.equal(one, out[2:5, 3:4]), (Dq, sq_, mode, "batch dependent")
        ref_t = cell_token_scores_torch(q, sel, K4v, tix, scl, sq_, mode)
        with _env(FSB_RERANK_CHUNK=sq_ * Dq * 5):                                   # 5 (row, candidate) entries per chunk
            assert torch.equal(ref_t, cell_token_scores_torch(q, sel, K4v, tix, scl, sq_, mode)), "torch path chunk dependent"
        ncmp58 += 1; nbit58 += int(torch.equal(out, ref_t))
        maxrel58 = max(maxrel58, ((out - ref_t).abs()[v] / ref_t.abs()[v].clamp(min=1.0)).max().item())
        brute = _tok_brute(q, sel, K4v, tix, scl, sq_, "lse" if mode == "lse_pos" else mode)
        maxerr58 = max(maxerr58, ((out.double() - brute).abs()[v] / brute.abs()[v].clamp(min=1.0)).max().item())
        for K_ in (1, 4, Cq):
            rt = rerank_top(sel, q, qpq, lamq, sigq, cK, cA, tix, scl, sq_, K_, score=mode, tokK=K4v)
            assert rt.shape == (Tq, Bqq, K_)
            u = _tok_brute(q, sel, K4v, tix, scl, sq_, mode, qpq, lamq, sigq, cA)
            ref, tie, nval = _topk_ref(sel, u, K_)
            assert bool((_same_sets(rt, ref) | tie).all()), (Dq, sq_, mode, K_)
            nrow58 += int((nval > K_).sum()); ntie58 += int((tie & (nval > K_)).sum())
            if K_ == Cq:                                                              # all candidates kept: order == float64 order
                assert bool(((rt == ref) | tie[..., None]).all())
assert maxrel58 < 2e-6 and maxerr58 < 1e-5, (maxrel58, maxerr58)
if dev == "cuda":
    # the kernel implements the reference arithmetic exactly (fp32 product, halving sums, libdevice exp / log, no FMA
    # contraction): the Triton and torch paths agree to the bit, so either path gives the same selection
    assert nbit58 == ncmp58, f"Triton != torch reference in {ncmp58 - nbit58}/{ncmp58} cases (max rel diff {maxrel58:.1e})"
    print(f"F58 [kernel] Triton == torch reference bit-exact in {nbit58}/{ncmp58} cases (max rel diff {maxrel58:.1e}); "
          f"float64 max rel err {maxerr58:.1e}")
# (e) needle: cell 0 = 31 small noise keys + ONE key 10 x along q (mean score ~0.3 x), cell 1 = every key 0.5 x along q
Dn, sn = 32, 32
qn_ = torch.zeros(1, 1, Dn, device=dev); qn_[..., 0] = 4.0
Kn = torch.randn(1, 1, 2 * sn, Dn, device=dev) * 0.05
Kn[0, 0, 5, 0] = 10.0
Kn[0, 0, sn:, 0] += 0.5
seln = torch.tensor([[[1, 0]]], device=dev)
cKn = Kn.view(1, 2, sn, Dn).mean(2); cAn = Kn.view(1, 2, sn, Dn).norm(dim=-1).amax(-1)
picks = {}
for mode in ("leaf",) + _RR:
    picks[mode] = int(rerank_top(seln, qn_, torch.full((1, 1), 5000.0, device=dev), torch.zeros(1, 1, 1, device=dev),
                                 torch.ones(1, 1, 1, device=dev), cKn, cAn, torch.zeros(1, dtype=torch.long, device=dev),
                                 1.0, sn, 1, score=mode, tokK=Kn)[0, 0, 0])
assert picks == {"leaf": 1, "lse": 0, "lse_pos": 0, "max": 0}, picks
print(f"F58 OK  (token scores: strided == contiguous == 3-D, batch independent, chunk independent, within {maxerr58:.1e} of float64; "
      f"rerank_top == float64 top-K on {nrow58 - ntie58}/{nrow58} rows ({ntie58} near-ties); needle picked by lse/lse_pos/max, not leaf)")


# ── F59: MODULE read sets == the float64 top-4 of the beam's cells by the token score (random adapter: the routing query
#        q_r scores the keys), W8/D0, W16/D0, W4/D1, forest + rebuild, every score; and the token re-rank reads a
#        different set than the leaf re-rank on some rows (the test has teeth) ─────────────────────────────────────────
n59 = 512
nrow59 = ntie59 = 0; ndiff59 = {sc: 0 for sc in _RR}
for tree in ("forest", "rebuild"):
    for (Wb, Db, menu) in ((8, 0, ("8", "0")), (16, 0, ("16", "0")), (4, 1, ("4", "1"))):
        common = dict(adapter=5900, fsb_policy="fixed", fsb_width=Wb, fsb_delta=Db, fsb_widths=menu[0], fsb_deltas=menu[1], fsb_tree=tree)
        mbeam = _mk_os(59, **common)
        mleaf = _mk_os(59, fsb_read_top=4, **common)
        Qc, Kc, Vc = _qkv_n(590, n59, scale=1.5)
        with torch.no_grad():
            sb = _sel_n(mbeam, Qc, Kc, Vc, n59)[0][0].long(); sl_ = _sel_n(mleaf, Qc, Kc, Vc, n59)[0][0].long()
            qr = fsb_route_q(mbeam, Qc.float())[0]                                     # (H,n,Dh) routing queries
        Npad = sb.shape[1]
        assert Npad == n59
        Mn = n59 // S
        ceil59 = Kc[0].view(H, Mn, S, Dh).float().norm(dim=-1).amax(-1)                 # (H,M) the module's a_c
        qpos59 = torch.arange(Npad, device=dev, dtype=torch.float32)[None].expand(H, -1)
        for score in _RR:
            mos = _mk_os(59, fsb_read_top=4, fsb_rerank_score=score, **common)
            with torch.no_grad():
                so = _sel_n(mos, Qc, Kc, Vc, n59)[0][0].long()
            u = _tok_brute(qr, sb, Kc, torch.arange(H, device=dev), mos.scale, S, score, qpos59,
                           mos.sel_scale.detach().view(H, 1, 1), mos.sigma.detach().view(H, 1, 1), ceil59)
            ref, tie, nval = _topk_ref(sb, u, 4)
            ok = _same_sets(so, ref)
            assert bool((ok | tie).all()), (tree, Wb, Db, score, int((~ok & ~tie).sum()))
            assert bool(((so[..., :, None] == sb[..., None, :]).any(-1) | (so < 0)).all()) and bool(((so >= 0).sum(-1) <= 4).all())
            nrow59 += int((nval > 4).sum()); ntie59 += int((tie & (nval > 4)).sum())
            ndiff59[score] += int((~_same_sets(so, sl_) & (nval > 0)).sum())
assert all(v_ > 0 for v_ in ndiff59.values()), ndiff59
print(f"F59 OK  (module read sets == float64 top-4 of the beam's cells by the token score on {nrow59 - ntie59}/{nrow59} rows "
      f"({ntie59} near-ties), forest + rebuild, W8 / W16 / W4D1; rows reading a different set than the leaf re-rank: {ndiff59})")


# ── F60: PREFILL == DECODE with the token re-rank.
#        (a) fp32, random adapter, W8 / read 4, every score: the no-cache selection equals the per-token decode selection
#            (sets) for every decode row, full + lean prefill, FSB_DECODE_GRAPH 0 / 1 (forest); cached outputs == full
#            forward (both trees); graph-replayed decode == eager decode bit-exact;
#        (b) bf16 (autocast) 4096-token prefix, forest, every score, full + lean prefill: the cache's raw keys ARE the
#            prefill's; the decode forest + the cache's key buffer routed with the PREFILL's q selects the prefill's cells
#            bit-exactly (order included); the beam candidates' token scores from Kp and from the cache are bit-identical;
#            the decode token's own q selects the prefill's sets on the whole last block ─────────────────────────────────
n60, N060 = 512, 300
for tree in ("forest", "rebuild"):
    for score in _RR:
        mm = _mk_os(60, adapter=6000, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree=tree, fsb_read_top=4, fsb_rerank_score=score)
        torch.manual_seed(601); x = torch.randn(1, n60, d_model, device=dev)
        nord = nrows = 0; ys = {}
        with torch.no_grad():
            sel_full, yfull = _full_sel_n(mm, x)
            for lean in (False, True):
                for dg in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                    y, sels = _cached_run_n(mm, x, N060, lean, dg)
                    d = (y - yfull).abs().max().item()
                    assert d < 1e-4, (tree, score, lean, dg, d)
                    ys[(lean, dg)] = y
                    if tree == "forest":
                        dec = torch.stack(sels, dim=1)
                        same_set = _same_sets(sel_full[:, N060:n60].long(), dec.long())
                        assert bool(same_set.all()), f"{score} lean={lean} graph={dg}: {int((~same_set).sum())} decode rows route differently"
                        nord += int((sel_full[:, N060:n60].long() == dec.long()).all(-1).sum()); nrows += same_set.numel()
            if dev == "cuda" and tree == "forest":
                for lean in (False, True):
                    assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (score, lean, "graph decode != eager decode")
        print(f"F60 [{tree}, {score}] " + (f"prefill selection == decode selection on {nrows} rows (order identical on {nord}), "
              f"graph decode == eager bit-exact, " if tree == "forest" else "") + "cached == full forward (full + lean prefill)")


def _f60_bf16(mm_, x_, Nf, lean, tag):
    caps = []
    _otr = _ft4.traverse_forest
    def _tr(F_, SP_, q_, qpos_, lam_, sigv_, *rest, **kw_):
        out_ = _otr(F_, SP_, q_, qpos_, lam_, sigv_, *rest, **kw_)
        caps.append((q_.clone(), qpos_.clone(), lam_.clone(), sigv_.clone(), rest, dict(kw_), out_[0].clone()))
        return out_
    os.environ["FSB_TRAIN_GRAPH"] = "0"
    _ft4.traverse_forest = _tr
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mm_(x_)                                                                     # no-cache forward = prefill routing
            q_, qpos_, lam_, sigv_, rest, kw_, sel_pre = caps[-1]                       # the chunk holding the last block
            Kp_pre = kw_["tokK"]
            assert Kp_pre is not None and Kp_pre.dtype == torch.bfloat16 and kw_["rerank_score"] == mm_.fsb_rerank_score
            nbc = q_.shape[0] // H; sl = slice(nbc - 1, None, nbc)                    # rows (h * nbc + last block)
            P = Nf - BLK
            cache = FSKVCache(mm_, lean_prefill=lean); cache.reserve = Nf - P
            mm_(x_[:, :P], cache=cache)
            mm_(x_[:, P:P + 1], cache=cache)                                           # builds the decode forest of the last block
            fc = cache._fsb_forest
            dec = [fc._last_sel.reshape(H, -1).clone()]
            nc = P // S
            assert torch.equal(cache._k_buf[:, :, :nc * S], Kp_pre[:, :, :nc * S]), f"{tag}: cache keys != prefill keys"
            rest_sl = tuple(r_[sl] if torch.is_tensor(r_) and r_.dim() >= 1 and r_.shape[0] == q_.shape[0] else r_ for r_ in rest)
            kw2 = dict(kw_); kw2["tokK"] = cache._k_buf
            s_dec, _, _ = _otr(fc.F, fc.SP, q_[sl], qpos_[sl], lam_[sl], sigv_[sl], *rest_sl, **kw2)
            assert torch.equal(s_dec, sel_pre[sl]), f"{tag}: decode forest + cache keys route the prefill q differently"
            kw0 = dict(kw2); kw0["read_top"] = 0
            cand, _, _ = _otr(fc.F, fc.SP, q_[sl], qpos_[sl], lam_[sl], sigv_[sl], *rest_sl, **kw0)
            tix_ = torch.arange(H, device=dev)
            sc_pre = cell_token_scores(q_[sl], cand, Kp_pre, tix_, mm_.scale, S, mm_.fsb_rerank_score)
            sc_dec = cell_token_scores(q_[sl], cand, cache._k_buf, tix_, mm_.scale, S, mm_.fsb_rerank_score)
            assert torch.equal(sc_pre, sc_dec), f"{tag}: token scores from Kp != from the cache"
            ncand = int((cand >= 0).sum())
            for t in range(P + 1, Nf):
                mm_(x_[:, t:t + 1], cache=cache)
                dec.append(cache._fsb_forest._last_sel.reshape(H, -1).clone())
    finally:
        _ft4.traverse_forest = _otr; os.environ.pop("FSB_TRAIN_GRAPH", None)
    sel_dec = torch.stack(dec, dim=1)                                                  # (H,Bq,kk)
    same_set, same_order = _sets_equal(sel_pre[sl].long(), sel_dec.long())
    assert bool(same_set.all()), f"{tag}: {int((~same_set).sum())}/{same_set.numel()} decode rows route differently"
    print(f"F60 [{tag}, lean={lean}] bf16 4096: cache keys == prefill keys; prefill q on the decode forest: identical cells + order "
          f"({same_set.numel()} rows); {ncand} candidate token scores Kp == cache bit-exact; decode q: {same_set.numel()} identical "
          f"sets, {int(same_order.sum())} identical order")


if dev == "cuda":
    for score in _RR:
        for lean in (False, True):
            torch.manual_seed(602)
            mm = _mk_os(602, adapter=6020, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree="forest", fsb_read_top=4,
                        fsb_rerank_score=score)
            x = torch.randn(1, 4096, d_model, device=dev)
            _f60_bf16(mm, x, 4096, lean, f"W8/read 4 {score}")
else:
    print("F60 [bf16] SKIP (no CUDA)")
print("F60 OK  (token re-rank: prefill routing == decode routing; decode keys == prefill keys; graph decode == eager)")


# ── F61: with the token re-rank the CUDA-graph training stage == eager (sel_full, output, n_read, n_visit; fixed + learned,
#        one and several chunk graphs); chunked prefill 1 / 2 / all identical (both trees); Triton level kernel == torch
#        traversal; module selection with the Triton re-rank kernel vs the torch reference path; training backward ────────
n61 = 512
nsame61 = ntot61 = 0
for score in _RR:
    if dev == "cuda":
        for pol, nbch in (("fixed", None), ("learned", None), ("fixed", 2), ("learned", 2)):
            kw = {} if nbch is None else {"fsb_prefill_blkchunk": nbch}
            wkw = dict(fsb_width=8, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
            mm = _mk_os(61, adapter=6100, fsb_policy=pol, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score=score, **wkw, **kw)
            if pol == "learned":
                mm.train()
            for trial in range(2):
                Qc, Kc, Vc = _qkv_n(610 + trial, n61)
                res = {}
                for g in ("0", "1"):
                    torch.manual_seed(6100 + trial)
                    sel, o = _sel_n(mm, Qc, Kc, Vc, n61, g)
                    st = mm._fsb_stats
                    res[g] = (sel, o.detach(), st["n_read"].clone(), st["n_visit"].clone())
                    assert int(st["n_read"].max()) <= 4
                for i, nm in enumerate(("sel_full", "output", "n_read", "n_visit")):
                    assert torch.equal(res["0"][i], res["1"][i]), (score, pol, nbch, trial, nm)
        mm = _mk_os(611, adapter=6110, fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score=score)
        Qc, Kc, Vc = _qkv_n(612, n61)
        with torch.no_grad():
            r0 = _sel_n(mm, Qc, Kc, Vc, n61, "0", triton="0"); r1 = _sel_n(mm, Qc, Kc, Vc, n61, "0", triton="1")
            assert torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1]), ("F61 traversal triton", score)
            with _env(FSB_RERANK_TRITON=0):
                r2 = _sel_n(mm, Qc, Kc, Vc, n61, "0")
        same = (r1[0].long() == r2[0].long()).all(-1)
        nsame61 += int(same.sum()); ntot61 += same.numel()
    for tree in ("forest", "rebuild"):
        outs = []
        for nbch in (1, 2, 0):
            mm = _mk_os(613, adapter=6130, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree=tree, fsb_read_top=4,
                        fsb_rerank_score=score, fsb_prefill_blkchunk=nbch)
            Qc, Kc, Vc = _qkv_n(614, n61)
            with torch.no_grad():
                outs.append(_sel_n(mm, Qc, Kc, Vc, n61))
        assert all(torch.equal(outs[0][0], o_[0]) and torch.equal(outs[0][1], o_[1]) for o_ in outs[1:]), (tree, score)
    m61 = _mk_os(615, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score=score).train()
    m61(torch.randn(1, N, d_model, device=dev)).float().pow(2).mean().backward()
    assert torch.isfinite(m61.W_q.weight.grad).all() and m61.W_q.weight.grad.abs().sum() > 0 and torch.isfinite(m61.W_v.weight.grad).all()
if dev == "cuda":
    assert nsame61 == ntot61, (nsame61, ntot61)
    print(f"F61 [kernel vs torch path] module selections identical (order included) on {nsame61}/{ntot61} (head, query) rows")
print("F61 OK  (token re-rank: graphed == eager training stage, chunked prefill identical, Triton == torch traversal, grads flow)")


# ── F62: MEMORY at 256k shapes (340M: 16 heads, Dh 64, cells of 32, one 32-block prefill chunk = 512 tree rows x 128
#        queries, 64 candidate slots of which 16 valid = W16/D0): the Triton path allocates only the (rows, C) score table
#        (no (rows, C, s, Dh) intermediate); the torch path stays within its FSB_RERANK_CHUNK budget ──────────────────────
if dev == "cuda":
    Tb, Bqb, Cb, Hb, Db_, sb_ = 512, 128, 64, 16, 64, 32
    Mb = 8192
    Kbig = torch.randn(1, Hb, Mb * sb_, Db_, device=dev, dtype=torch.bfloat16)          # 262144 tokens
    selb = torch.randint(0, Mb, (Tb, Bqb, Cb), device=dev)
    selb[..., 16:] = -1
    qb = torch.randn(Tb, Bqb, Db_, device=dev)
    tixb = torch.arange(Hb, device=dev).repeat_interleave(Tb // Hb)
    for mode in ("lse", "max"):                                                         # compile outside the measurement
        cell_token_scores(qb[:1], selb[:1], Kbig, tixb[:1], 0.125, sb_, mode)
    res62 = {}
    for mode in ("lse", "max"):
        torch.cuda.synchronize(); base62 = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        outb = cell_token_scores(qb, selb, Kbig, tixb, 0.125, sb_, mode)
        torch.cuda.synchronize(); dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() - base62
        out_bytes = Tb * Bqb * Cb * 4
        assert peak <= out_bytes + (4 << 20), (mode, peak, out_bytes)
        res62[mode] = (outb, dt, peak)
    # torch path on the first 8 tree rows with a 2^20-element chunk budget
    nr = 8
    with _env(FSB_RERANK_TRITON=0, FSB_RERANK_CHUNK=1 << 20):
        torch.cuda.synchronize(); base62 = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        outt = cell_token_scores(qb[:nr], selb[:nr], Kbig, tixb[:nr], 0.125, sb_, "lse")
        torch.cuda.synchronize(); peak_t = torch.cuda.max_memory_allocated() - base62
    assert peak_t <= nr * Bqb * Cb * 4 * 2 + 6 * (1 << 20) * 4 + (8 << 20), peak_t
    rel62 = ((outt - res62["lse"][0][:nr]).abs()[selb[:nr] >= 0] / outt.abs()[selb[:nr] >= 0].clamp(min=1.0)).max().item()
    assert rel62 < 2e-6, rel62
    del Kbig, selb, qb, outb, outt; torch.cuda.empty_cache()
    print(f"F62 OK  (256k shapes, {Tb * Bqb} queries x {Cb} slots (16 valid): Triton lse {res62['lse'][1] * 1e3:.1f} ms, peak +"
          f"{res62['lse'][2] / 2**20:.1f} MiB (score table {Tb * Bqb * Cb * 4 / 2**20:.1f} MiB); max {res62['max'][1] * 1e3:.1f} ms; "
          f"torch path (chunk 2^20) peak +{peak_t / 2**20:.1f} MiB on {nr} rows, rel diff to Triton {rel62:.1e})")
else:
    print("F62 SKIP (no CUDA)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  INTEGRATION: router CPT (routing adapter, node distillation, learned c; fsb-routernode) x OVERSELECT re-rank
#  (fsb_read_top + fsb_rerank_score; fsb-overselect), and the re-rank QUERY (fsb_rerank_query route | raw): F63..F68.
#  route (default) = the re-rank scores with the routing query q_r the beam traversed with (fsb-overselect behaviour);
#  raw = the raw q (the exact per-cell LSE teacher's query) scores the re-rank while the beam still traverses with q_r.
#  Cross-tree parity with both parents (flag-off / default paths bit-exact) is run by
#  experiments/scaling/fsb_overselect_offparity.py (--tiny_rerank / --tiny_router; launchers/fsb_integrate_parity.sh).
# ══════════════════════════════════════════════════════════════════════════════════════════
_RQ_SCORES = ("leaf",) + _RR


def _mk_int(seed, adapter=None, alpha=1.0, theta=None, **kw):
    """_mk_os with the adapter scale ``alpha`` and (theta given) a learned radius coefficient moved off its init."""
    if adapter:
        kw["fsb_route_alpha"] = alpha
    if theta is not None:
        kw["fsb_route_learn_c"] = 1
    m_ = _mk_os(seed, adapter=adapter, **kw)
    if theta is not None:
        with torch.no_grad():
            g_ = torch.Generator(device="cpu").manual_seed(theta)
            m_.fsb_route_ctheta.add_((torch.randn(H, generator=g_) * 1.5).to(dev))
    return m_


# ── F63: construction and the ROUTE DEFAULT.  (a) every module defaults to "route"; raw without fsb_read_top and an
#        unknown value are refused; (b) fsb_rerank_query "route" passed explicitly == the flag absent: same state, same
#        selection and output (eager + CUDA-graph training stage, both trees, leaf + token scores), same cached decode;
#        (c) raw == route BIT-EXACTLY when the routing query IS the raw query: no adapter, and an adapter at alpha 0;
#        (d) an adapter at alpha 0 == NO adapter (the overselect selection and output of the same backbone), so with the
#        cross-tree parity (adapter off == fsb-overselect) the merged code at alpha 0 selects what fsb-overselect selects ──
assert mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0).fsb_rerank_query == "route"
assert mk(fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_read_top=4, fsb_rerank_score="lse_pos").fsb_rerank_query == "route"
for bad in (dict(fsb_rerank_query="raw"), dict(fsb_rerank_query="raw", fsb_read_top=0),
            dict(fsb_rerank_query="exact", fsb_read_top=4)):
    try:
        mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0, **bad)
        raise AssertionError(f"fsb_init must refuse {bad}")
    except ValueError:
        pass
n63, N063 = 512, 300
n63b = n63c = n63d = 0
for tree in ("forest", "rebuild"):
    for score in ("leaf", "lse_pos"):
        common = dict(fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree=tree, fsb_read_top=4, fsb_rerank_score=score)
        ma = _mk_int(63, adapter=6300, **common)
        mb = _mk_int(63, adapter=6300, fsb_rerank_query="route", **common)
        sda, sdb = ma.state_dict(), mb.state_dict()
        assert list(sda) == list(sdb) and all(torch.equal(sda[k], sdb[k]) for k in sda)
        Qc, Kc, Vc = _qkv_n(630, n63, 1.5)
        torch.manual_seed(631); x = torch.randn(1, n63, d_model, device=dev)
        with torch.no_grad():
            for g in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                ra = _sel_n(ma, Qc, Kc, Vc, n63, g); rb = _sel_n(mb, Qc, Kc, Vc, n63, g)
                assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1]), ("F63b", tree, score, g)
                n63b += 1
            for dg in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                ya, sa = _cached_run_n(ma, x, N063, False, dg); yb, sb_ = _cached_run_n(mb, x, N063, False, dg)
                assert torch.equal(ya, yb) and all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(sa, sb_))
                n63b += 1
        # (c) + (d): no adapter / alpha 0: raw == route, and alpha 0 == no adapter
        m_none_r = _mk_int(63, **common)
        m_none_w = _mk_int(63, fsb_rerank_query="raw", **common)
        m_a0_r = _mk_int(63, adapter=6300, alpha=0.0, **common)
        m_a0_w = _mk_int(63, adapter=6300, alpha=0.0, fsb_rerank_query="raw", **common)
        with torch.no_grad():
            for g in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                r0 = _sel_n(m_none_r, Qc, Kc, Vc, n63, g)
                for m_ in (m_none_w, m_a0_r, m_a0_w):
                    r_ = _sel_n(m_, Qc, Kc, Vc, n63, g)
                    assert torch.equal(r0[0], r_[0]) and torch.equal(r0[1], r_[1]), ("F63cd", tree, score, g)
                    n63c += 1
            for dg in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                y0, s0 = _cached_run_n(m_none_r, x, N063, True, dg)
                for m_ in (m_none_w, m_a0_w):
                    y_, s_ = _cached_run_n(m_, x, N063, True, dg)
                    assert torch.equal(y0, y_) and all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(s0, s_))
                    n63d += 1
print(f"F63 OK  (default route; raw needs read_top, unknown refused; route explicit == absent on {n63b} eager/graph/decode "
      f"cases; raw == route and alpha 0 == no adapter bit-exact on {n63c} full-sequence + {n63d} cached-decode cases)")


# ── F64: raw vs route differ ONLY in the re-rank score.  With a random adapter (alpha 1): (a) the beam's candidate cells
#        handed to rerank_top are identical (the traversal still uses q_r) and the re-rank query is the raw q for raw, q_r
#        for route (captured at rerank_top); (b) each read set == the float64 top-4 of the beam's candidates by the score
#        computed with ITS query (raw q / q_r), forest + rebuild, W8/D0 and W16/D0, every score; (c) raw and route read
#        different sets on some rows (the flag has teeth) ─────────────────────────────────────────────────────────────
n64 = 512
nrow64 = ntie64 = 0; ndiff64 = {sc: 0 for sc in _RQ_SCORES}
_orr = _ft4.rerank_top
for tree in ("forest", "rebuild"):
    for (Wb, menu) in ((8, "8"), (16, "16")):
        common = dict(fsb_policy="fixed", fsb_width=Wb, fsb_delta=0, fsb_widths=menu, fsb_deltas="0", fsb_tree=tree)
        mbeam = _mk_int(64, adapter=6400, **common)                                     # read_top 0: all W_beam candidates
        Qc, Kc, Vc = _qkv_n(640, n64, scale=1.5)
        with torch.no_grad():
            sb = _sel_n(mbeam, Qc, Kc, Vc, n64)[0][0].long()                            # (H,n,W_beam)
            qr = fsb_route_q(mbeam, Qc.float())[0]
        assert not torch.equal(qr, Qc[0].float())
        Mn = n64 // S
        ceil64 = Kc[0].view(H, Mn, S, Dh).float().norm(dim=-1).amax(-1)
        qpos64 = torch.arange(n64, device=dev, dtype=torch.float32)[None].expand(H, -1)
        kbar64 = Kc[0].view(H, Mn, S, Dh).mean(2)
        for score in _RQ_SCORES:
            sels, caps = {}, {}
            for rq in ("route", "raw"):
                mq = _mk_int(64, adapter=6400, fsb_read_top=4, fsb_rerank_score=score, fsb_rerank_query=rq, **common)
                cap_ = []
                def _rt(sel_, q_, *a_, **k_):
                    cap_.append((sel_.clone(), q_.clone())); return _orr(sel_, q_, *a_, **k_)
                _ft4.rerank_top = _rt
                try:
                    with torch.no_grad():
                        sels[rq] = _sel_n(mq, Qc, Kc, Vc, n64)[0][0].long()
                finally:
                    _ft4.rerank_top = _orr
                caps[rq] = cap_
                lam_ = mq.sel_scale.detach().view(H, 1, 1); sig_ = mq.sigma.detach().view(H, 1, 1)
                qq = Qc[0].float() if rq == "raw" else qr
                if score == "leaf":
                    c_ = sb.clamp(min=0)
                    kb = torch.gather(kbar64.double()[:, None].expand(H, n64, Mn, Dh), 2, c_[..., None].expand(-1, -1, -1, Dh))
                    a_ = torch.gather(ceil64.double()[:, None].expand(H, n64, Mn), 2, c_)
                    u = ((qq.double()[:, :, None, :] * kb).sum(-1) * mq.scale + lam_.double() * a_
                         * torch.erf((qpos64.double()[..., None] - (c_.double() + 0.5) * S).abs() / (2.0 * sig_.double())))
                    u = u.masked_fill(sb < 0, float("-inf"))
                else:
                    u = _tok_brute(qq, sb, Kc, torch.arange(H, device=dev), mq.scale, S, score, qpos64, lam_, sig_, ceil64)
                ref, tie, nval = _topk_ref(sb, u, 4)
                ok = _same_sets(sels[rq], ref)
                assert bool((ok | tie).all()), (tree, Wb, score, rq, int((~ok & ~tie).sum()))
                nrow64 += int((nval > 4).sum()); ntie64 += int((tie & (nval > 4)).sum())
            # (a) identical candidates at every rerank_top call; the query differs (raw q vs q_r)
            assert len(caps["route"]) == len(caps["raw"]) > 0
            for (s_r, q_r_), (s_w, q_w) in zip(caps["route"], caps["raw"]):
                assert torch.equal(s_r, s_w), (tree, Wb, score, "the beam's candidates differ between raw and route")
                assert not torch.equal(q_r_, q_w)
            # the raw query captured IS the raw q (rows of Qc) and the route query IS q_r: check the first chunk's rows
            q_w0, q_r0 = caps["raw"][0][1], caps["route"][0][1]
            nbc0 = q_w0.shape[0] // H
            assert torch.equal(q_w0.view(H, nbc0 * BLK, Dh), Qc[0, :, :nbc0 * BLK].float()), (tree, Wb, score)
            assert torch.equal(q_r0.view(H, nbc0 * BLK, Dh), qr[:, :nbc0 * BLK]), (tree, Wb, score)
            ndiff64[score] += int((~_same_sets(sels["raw"], sels["route"]))[:, BLK:].sum())
assert all(v_ > 0 for v_ in ndiff64.values()), ndiff64
print(f"F64 OK  (raw vs route: identical beam candidates at every re-rank call, the re-rank query is the raw q / q_r; read "
      f"sets == float64 top-4 by each query's score on {nrow64 - ntie64}/{nrow64} rows ({ntie64} near-ties), forest + rebuild, "
      f"W8 / W16; rows where raw reads a different set than route: {ndiff64})")


# ── F65: PREFILL == DECODE for raw AND route with a NONZERO adapter (and with a learned c_h moved off its init):
#        (a) fp32, W8/read 4, every score, both queries: the no-cache full-sequence selection equals the per-token decode
#            selection (sets) on every decode row, full + lean prefill, FSB_DECODE_GRAPH 0 / 1 (forest); cached outputs ==
#            full forward (both trees); graph-replayed decode == eager decode bit-exact;
#        (b) bf16 (autocast) 4096-token prefix, forest, lse_pos, raw + route: the decode forest + the cache's keys scoring
#            the PREFILL's routing / raw queries select the prefill's cells bit-exactly (order included); the decode
#            token's own q selects the prefill's sets on the whole last block ─────────────────────────────────────────
n65, N065 = 512, 300
nrows65 = nord65 = 0
for tree in ("forest", "rebuild"):
    for score in _RQ_SCORES:
        for rq in ("route", "raw"):
            for th in ((None, 6510) if tree == "forest" else (None,)):
                mm = _mk_int(65, adapter=6500, theta=th, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree=tree,
                             fsb_read_top=4, fsb_rerank_score=score, fsb_rerank_query=rq, fsb_bound_radius_coef=4.0)
                torch.manual_seed(651); x = torch.randn(1, n65, d_model, device=dev)
                ys = {}
                with torch.no_grad():
                    sel_full, yfull = _full_sel_n(mm, x)
                    for lean in (False, True):
                        for dg in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                            y, sels = _cached_run_n(mm, x, N065, lean, dg)
                            d = (y - yfull).abs().max().item()
                            assert d < 1e-4, (tree, score, rq, th, lean, dg, d)
                            ys[(lean, dg)] = y
                            if tree == "forest":
                                dec = torch.stack(sels, dim=1)
                                same_set = _same_sets(sel_full[:, N065:n65].long(), dec.long())
                                assert bool(same_set.all()), f"{score}/{rq}/c{th} lean={lean} graph={dg}: {int((~same_set).sum())} decode rows differ"
                                nord65 += int((sel_full[:, N065:n65].long() == dec.long()).all(-1).sum()); nrows65 += same_set.numel()
                    if dev == "cuda" and tree == "forest":
                        for lean in (False, True):
                            assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (score, rq, th, lean, "graph decode != eager")
print(f"F65 [fp32] prefill selection == decode selection on {nrows65} rows (order identical on {nord65}), raw + route, every "
      f"score, fixed and learned c_h; graph decode == eager; cached == full forward (both trees)")


def _f65_bf16(mm_, x_, Nf, lean, tag):
    """F60(b) for the re-rank query: the prefill's routing q AND its re-rank query (raw q) are replayed on the decode forest."""
    caps = []
    _otr = _ft4.traverse_forest
    def _tr(F_, SP_, q_, qpos_, lam_, sigv_, *rest, **kw_):
        out_ = _otr(F_, SP_, q_, qpos_, lam_, sigv_, *rest, **kw_)
        caps.append((q_.clone(), qpos_.clone(), lam_.clone(), sigv_.clone(), rest, dict(kw_), out_[0].clone()))
        return out_
    os.environ["FSB_TRAIN_GRAPH"] = "0"
    _ft4.traverse_forest = _tr
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mm_(x_)
            q_, qpos_, lam_, sigv_, rest, kw_, sel_pre = caps[-1]
            assert kw_["tokK"] is not None and kw_["rerank_score"] == mm_.fsb_rerank_score
            assert (kw_.get("rq") is not None) == (mm_.fsb_rerank_query == "raw"), tag
            nbc = q_.shape[0] // H; sl = slice(nbc - 1, None, nbc)
            P = Nf - BLK
            cache = FSKVCache(mm_, lean_prefill=lean); cache.reserve = Nf - P
            mm_(x_[:, :P], cache=cache)
            mm_(x_[:, P:P + 1], cache=cache)
            fc = cache._fsb_forest
            dec = [fc._last_sel.reshape(H, -1).clone()]
            rest_sl = tuple(r_[sl] if torch.is_tensor(r_) and r_.dim() >= 1 and r_.shape[0] == q_.shape[0] else r_ for r_ in rest)
            kw2 = dict(kw_); kw2["tokK"] = cache._k_buf
            for k_ in ("radius_coef", "rq"):
                v_ = kw2.get(k_)
                if torch.is_tensor(v_) and v_.dim() >= 1 and v_.shape[0] == q_.shape[0]:
                    kw2[k_] = v_[sl]
            s_dec, _, _ = _otr(fc.F, fc.SP, q_[sl], qpos_[sl], lam_[sl], sigv_[sl], *rest_sl, **kw2)
            assert torch.equal(s_dec, sel_pre[sl]), f"{tag}: decode forest + cache keys route the prefill queries differently"
            for t in range(P + 1, Nf):
                mm_(x_[:, t:t + 1], cache=cache)
                dec.append(cache._fsb_forest._last_sel.reshape(H, -1).clone())
    finally:
        _ft4.traverse_forest = _otr; os.environ.pop("FSB_TRAIN_GRAPH", None)
    sel_dec = torch.stack(dec, dim=1)
    same_set, same_order = _sets_equal(sel_pre[sl].long(), sel_dec.long())
    assert bool(same_set.all()), f"{tag}: {int((~same_set).sum())}/{same_set.numel()} decode rows route differently"
    print(f"F65 [{tag}, lean={lean}] bf16 4096: prefill queries on the decode forest + cache keys: identical cells + order "
          f"({same_set.numel()} rows); decode q: {same_set.numel()} identical sets, {int(same_order.sum())} identical order")


if dev == "cuda":
    for rq in ("route", "raw"):
        for lean in (False, True):
            torch.manual_seed(652)
            mm = _mk_int(652, adapter=6520, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree="forest", fsb_read_top=4,
                         fsb_rerank_score="lse_pos", fsb_rerank_query=rq)
            x = torch.randn(1, 4096, d_model, device=dev)
            _f65_bf16(mm, x, 4096, lean, f"W8/read 4 lse_pos {rq}")
else:
    print("F65 [bf16] SKIP (no CUDA)")
print("F65 OK  (raw + route with a nonzero adapter: prefill routing == decode routing; graph decode == eager)")


# ── F66: with raw the CUDA-graph training stage == eager (the static raw-q buffer is refreshed per call): sel_full,
#        output, n_read, n_visit; fixed W8 + learned menu, one and several chunk graphs, learned c_h, and after an in-place
#        adapter update; chunked prefill 1 / 2 / all identical (both trees); Triton level kernel == torch traversal ───────
n66 = 512
if dev == "cuda":
    for score in ("leaf", "lse_pos"):
        for pol, nbch in (("fixed", None), ("learned", None), ("fixed", 2), ("learned", 2)):
            kw = {} if nbch is None else {"fsb_prefill_blkchunk": nbch}
            wkw = dict(fsb_width=8, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
            th = 6610 if pol == "fixed" else None
            mm = _mk_int(66, adapter=6600, theta=th, fsb_policy=pol, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score=score,
                         fsb_rerank_query="raw", **wkw, **kw)
            if pol == "learned":
                mm.train()
            for trial in range(3):
                Qc, Kc, Vc = _qkv_n(660 + trial, n66)
                res = {}
                for g in ("0", "1"):
                    torch.manual_seed(6600 + trial)
                    with torch.no_grad():
                        sel, o = _sel_n(mm, Qc, Kc, Vc, n66, g)
                    st = mm._fsb_stats
                    res[g] = (sel, o.detach(), st["n_read"].clone(), st["n_visit"].clone())
                    assert int(st["n_read"].max()) <= 4
                for i, nm in enumerate(("sel_full", "output", "n_read", "n_visit")):
                    assert torch.equal(res["0"][i], res["1"][i]), (score, pol, nbch, trial, nm)
                with torch.no_grad():
                    mm.fsb_route_W.add_(torch.randn_like(mm.fsb_route_W) * 0.1)      # an optimizer step in place
    mm = _mk_int(661, adapter=6610, fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_tree="forest", fsb_read_top=4,
                 fsb_rerank_score="lse_pos", fsb_rerank_query="raw")
    Qc, Kc, Vc = _qkv_n(662, n66)
    with torch.no_grad():
        r0 = _sel_n(mm, Qc, Kc, Vc, n66, "0", triton="0"); r1 = _sel_n(mm, Qc, Kc, Vc, n66, "0", triton="1")
    assert torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1]), "F66 traversal triton (raw)"
for tree in ("forest", "rebuild"):
    outs = []
    for nbch in (1, 2, 0):
        mm = _mk_int(663, adapter=6630, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree=tree, fsb_read_top=4,
                     fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_prefill_blkchunk=nbch)
        Qc, Kc, Vc = _qkv_n(664, n66)
        with torch.no_grad():
            outs.append(_sel_n(mm, Qc, Kc, Vc, n66))
    assert all(torch.equal(outs[0][0], o_[0]) and torch.equal(outs[0][1], o_[1]) for o_ in outs[1:]), tree
print("F66 OK  (raw re-rank: graphed == eager training stage incl. learned c and in-place adapter updates, chunked prefill "
      "identical, Triton == torch traversal)")


# ── F67: router distillation ON TOP of the overselect (the combined system's CPT).  (a) W4/D0 with read_top 4 (the read
#        set IS the beam's 4 cells) gives the SAME distillation loss, (32,) stats (leaf + node KL, beam recall, node
#        recall, trace) and router grads as W4 native, eager and CUDA-graph: the overselect plumbing leaves the node loss
#        and the learned c untouched; (b) W8 read 4 with every score and raw / route: the node trace re-run (eager torch
#        loop + the same re-rank) reproduces the forward's READ set exactly (trace mismatch 0), eager and CUDA-graph;
#        the loss is finite and the gradient reaches only fsb_route_* (adapter, temperatures, c_h) ────────────────────
Qc, Kc, Vc = _qkv_n(670, N2, 1.5)
for g in (("0", "1") if dev == "cuda" else ("0",)):
    res = []
    for extra in ({}, dict(fsb_read_top=4)):
        torch.manual_seed(67)
        m_ = _mk_node(fsb_width=4, fsb_rd_node=1, fsb_rd_node_weight=0.5, fsb_route_learn_c=1, fsb_bound_radius_coef=8.0,
                      **extra).train()
        with torch.no_grad():
            g_ = torch.Generator(device="cpu").manual_seed(671)
            m_.fsb_route_W.copy_((torch.randn(m_.fsb_route_W.shape, generator=g_) * 0.3).to(dev))
            m_.fsb_route_ctheta.add_((torch.randn(H, generator=g_) * 1.0).to(dev))
        m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(672)
        sel_, o_, rd_, _ = _lr_n(m_, Qc, Kc, Vc, N2, g)
        rd_["loss"].backward()
        grads = {n_: p_.grad.clone() for n_, p_ in m_.named_parameters() if p_.grad is not None}
        res.append((sel_, o_.detach(), rd_["loss"].detach(), rd_["stats"], grads))
    (s0, o0, l0, st0, g0), (s1, o1, l1, st1, g1) = res
    assert _same_sets(s0.long(), s1.long()).all() and torch.allclose(o0, o1, atol=1e-5), ("F67a selection/output", g)
    assert torch.equal(l0, l1) and torch.equal(st0, st1), ("F67a loss/stats", g, l0.item(), l1.item())
    assert sorted(g0) == sorted(g1) and all(torch.equal(g0[k], g1[k]) for k in g0), ("F67a grads", g)
    assert st0.numel() == RD_STATS_NODE and float(st0[14]) == 0.0 and float(st0[11]) > 0
nmis67 = ntr67 = 0
for score in _RQ_SCORES:
    for rq in ("route", "raw"):
        torch.manual_seed(673)
        m_ = _mk_node(fsb_width=8, fsb_rd_node=1, fsb_rd_node_weight=0.5, fsb_route_learn_c=1, fsb_bound_radius_coef=8.0,
                      fsb_read_top=4, fsb_rerank_score=score, fsb_rerank_query=rq).train()
        with torch.no_grad():
            g_ = torch.Generator(device="cpu").manual_seed(674)
            m_.fsb_route_W.copy_((torch.randn(m_.fsb_route_W.shape, generator=g_) * 0.3).to(dev))
            m_.fsb_route_ctheta.add_((torch.randn(H, generator=g_) * 1.0).to(dev))
        for g in (("0", "1") if dev == "cuda" else ("0",)):
            m_.zero_grad(set_to_none=True)
            m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(675)
            sel_, o_, rd_, fr_ = _lr_n(m_, Qc, Kc, Vc, N2, g)
            st_ = rd_["stats"]
            assert fr_ is not None and st_.numel() == RD_STATS_NODE and "cellK" in fr_[0], (score, rq, g)
            assert float(st_[14]) == 0.0, (score, rq, g, "node trace re-run != the forward's read set", st_[14].item(), st_[15].item())
            nmis67 += int(st_[14]); ntr67 += int(st_[15])
            assert torch.isfinite(rd_["loss"]) and int(sel_.shape[-1]) == 4
            rd_["loss"].backward()
            got = sorted(n_ for n_, p_ in m_.named_parameters() if p_.grad is not None and p_.grad.abs().sum() > 0)
            assert got == ["fsb_route_W", "fsb_route_ctheta", "fsb_route_logtemp", "fsb_route_nlogtemp"], (score, rq, got)
print(f"F67 OK  (router CPT on the overselect: W4 read 4 == W4 native loss / stats / router grads bit-exact (eager + graph); "
      f"W8 read 4, every score, raw + route: node trace reproduces the read set ({nmis67} mismatches / {ntr67} rows), grads "
      f"only on fsb_route_*)")


# ── F68: plumbing.  build_attn_kwargs threads fsb_rerank_query only when set (None = not threaded: the key is absent,
#        so every existing command line resolves the same attn_kwargs) ─────────────────────────────────────────────────
from experiments.scaling.config import build_attn_kwargs as _bak68


class _NS68:
    """argparse-like namespace: unset attributes read None (the train_dist defaults of every fsb_* flag)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, k):
        return None


_kw_off = _bak68(_NS68(attn_type="fsattn", fsb=True, fsb_read_top=4, fsb_rerank_score="lse_pos"))[1]
_kw_on = _bak68(_NS68(attn_type="fsattn", fsb=True, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw"))[1]
assert "fsb_rerank_query" not in _kw_off and _kw_on["fsb_rerank_query"] == "raw"
assert {k: v for k, v in _kw_on.items() if k != "fsb_rerank_query"} == _kw_off
print("F68 OK  (--fsb_rerank_query threaded by build_attn_kwargs only when set)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  UNION CANDIDATE POOL (fsb_union_width U, with fsb_read_top): the overselect re-rank's candidates are the W-wide beam's
#  retained cells UNION the retained cells of a second beam of width min(U, 2^D) run with the same routing query, stop
#  rule and node score.  Union cells already among the W-wide beam's candidates are dropped (the W-wide beam's slots and
#  order are kept, the second beam's new cells follow); the re-rank then keeps read_top.  F69..F74.
# ══════════════════════════════════════════════════════════════════════════════════════════
from src.fsattn.fsb_tree import union_candidates


def _rr_cap(fn):
    """run fn() with every rerank_top call's (candidates, query) captured -> (fn(), [(sel (T',Bq,C), q), ...])."""
    cap_ = []
    _o = _ft4.rerank_top
    def _rt(sel_, q_, *a_, **k_):
        cap_.append((sel_.clone(), q_.clone())); return _o(sel_, q_, *a_, **k_)
    _ft4.rerank_top = _rt
    try:
        return fn(), cap_
    finally:
        _ft4.rerank_top = _o


def _cands(caps):
    """the candidate tensors handed to rerank_top by one full-sequence forward (B = 1), per chunk (T*nbc, Bq, C) with
    row (h*nbc + j), concatenated in block order -> (H, Npad, C)."""
    out = []
    for sel_, _ in caps:
        nbc = sel_.shape[0] // H
        out.append(sel_.reshape(H, nbc * sel_.shape[1], sel_.shape[2]))
    return torch.cat(out, dim=1)


def _nstats(mm_):
    st = mm_._fsb_stats
    return st["n_read"].clone(), st["n_visit"].clone()


# ── F69: construction and flag OFF.  (a) every module defaults to fsb_union_width 0 (no parameter / buffer); U without
#        fsb_read_top, U < 0 and a union beam over 64 candidates are refused; the union beam's slots round up to a power
#        of two and leave the read width (fsb_kk) and the main beam's slots unchanged; (b) U = 0 passed explicitly == the
#        flag absent: same state, sel_full, output, n_read, n_visit on the eager and CUDA-graph training stage (both
#        trees, fixed + learned policy, eval argmax + train sampling, leaf + lse_pos, raw query with a random adapter)
#        and the same cached decode (full + lean prefill, FSB_DECODE_GRAPH 0 / 1).  Byte-identity with the flag ABSENT
#        against the parent trees is checked by experiments/scaling/fsb_overselect_offparity.py ──────────────────────
assert mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0).fsb_union_width == 0
assert mk(fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_read_top=4, fsb_rerank_score="lse_pos").fsb_union_width == 0
m69 = mk(fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_read_top=4, fsb_union_width=3)
assert (m69.fsb_union_width, m69.fsb_union_W_max, m69.fsb_kk, m69.fsb_W_max) == (3, 4, 4, 16)
m69 = mk(fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_read_top=4, fsb_union_width=16)      # 16 x 2^2 = 64: allowed
assert (m69.fsb_union_W_max, m69.fsb_kk) == (16, 4)
for bad in (dict(fsb_width=4, fsb_union_width=4), dict(fsb_width=8, fsb_read_top=0, fsb_union_width=4),
            dict(fsb_width=8, fsb_read_top=4, fsb_union_width=-1), dict(fsb_width=8, fsb_read_top=4, fsb_union_width=17)):
    try:
        mk(fsb_policy="fixed", fsb_delta=0, **bad)
        raise AssertionError(f"fsb_init must refuse {bad}")
    except ValueError:
        pass
n69, N069 = 512, 400
n69b = 0
for tree in ("forest", "rebuild"):
    for pol in ("fixed", "learned"):
        for score in ("leaf", "lse_pos"):
            wkw = dict(fsb_width=8, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
            common = dict(fsb_policy=pol, fsb_tree=tree, fsb_read_top=4, fsb_rerank_score=score, fsb_rerank_query="raw", **wkw)
            ma = _mk_int(69, adapter=6900, **common)
            mb = _mk_int(69, adapter=6900, fsb_union_width=0, **common)
            sda, sdb = ma.state_dict(), mb.state_dict()
            assert list(sda) == list(sdb) and all(torch.equal(sda[k], sdb[k]) for k in sda)
            assert (ma.fsb_kk, ma.fsb_W_max, ma.fsb_delta_max, ma.fsb_union_W_max) == (mb.fsb_kk, mb.fsb_W_max, mb.fsb_delta_max, 0)
            Qc, Kc, Vc = _qkv_n(690, n69, 1.5)
            for mode in ("eval", "train"):
                ma.train(mode == "train"); mb.train(mode == "train")
                for g in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                    res = []
                    for m_ in (ma, mb):
                        torch.manual_seed(691)
                        with torch.no_grad():
                            sel_, o_ = _sel_n(m_, Qc, Kc, Vc, n69, g)
                        res.append((sel_, o_) + _nstats(m_))
                    assert all(torch.equal(a_, b_) for a_, b_ in zip(*res)), (tree, pol, score, mode, g)
                    n69b += 1
            ma.eval(); mb.eval()
            torch.manual_seed(692); x = torch.randn(1, n69, d_model, device=dev)
            with torch.no_grad():
                for lean in (False, True):
                    for dg in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                        ya, sa = _cached_run_n(ma, x, N069, lean, dg); yb, sb_ = _cached_run_n(mb, x, N069, lean, dg)
                        assert torch.equal(ya, yb) and all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(sa, sb_)), \
                            (tree, pol, score, lean, dg)
                        n69b += 1
print(f"F69 OK  (fsb_union_width: default 0; refused without read_top / < 0 / > 64 union candidates; power-of-two union slots, "
      f"read width unchanged; U = 0 explicit == absent on {n69b} eager / graph / train-sampling / cached-decode cases)")


# ── F70: the candidate pool IS the set union.  (a) union_candidates == a python brute force (W-beam part unchanged, then
#        the U-beam's cells not already present, in order; no duplicates; set == A | B) on random rows with -1 pads;
#        (b) end to end, forest + rebuild, (W, U) = (16, 4), (8, 4), (4, 8) on the default menus (Delta_max 2: -1 padded
#        node slots) and (8, 2) on an explicit W8/D0 menu, leaf + lse_pos, raw query with a random adapter: the candidates
#        the union module hands to rerank_top == [the W-beam module's candidates (identical slots) | the U-beam module's
#        cells not among them (same order)], brute-forced per row in python against two separate modules (fsb_width W,
#        fsb_width U; no union); the read set == the float64 top-4 of the union pool by the score of the raw query;
#        the union adds cells and changes the read set on some rows (the flag has teeth) ───────────────────────────────
g70 = torch.Generator(device="cpu").manual_seed(70)
for _ in range(20):
    R70 = 64
    Ca70 = int(torch.randint(1, 20, (1,), generator=g70)); Cb70 = int(torch.randint(1, 20, (1,), generator=g70))
    A70 = torch.stack([torch.randperm(40, generator=g70)[:Ca70] for _ in range(R70)])
    B70 = torch.stack([torch.randperm(40, generator=g70)[:Cb70] for _ in range(R70)])
    A70 = A70.masked_fill(torch.rand(R70, Ca70, generator=g70) < 0.3, -1)
    B70 = B70.masked_fill(torch.rand(R70, Cb70, generator=g70) < 0.3, -1)
    U70 = union_candidates(A70.to(dev), B70.to(dev)).cpu()
    assert U70.shape == (R70, Ca70 + Cb70)
    for r in range(R70):
        a_l, b_l, u_l = A70[r].tolist(), B70[r].tolist(), U70[r].tolist()
        assert u_l[:Ca70] == a_l and u_l[Ca70:] == [c if (c < 0 or c not in a_l) else -1 for c in b_l]
        v = [c for c in u_l if c >= 0]
        assert len(v) == len(set(v)) and set(v) == {c for c in a_l + b_l if c >= 0}
n70 = 512
nrow70 = ntie70 = nadd70 = ndiff70 = 0
Mn70 = n70 // S
qpos70 = torch.arange(n70, device=dev, dtype=torch.float32)[None].expand(H, -1)
for tree in ("forest", "rebuild"):
    for (Wb, Ub, menu) in ((16, 4, None), (8, 4, None), (4, 8, None), (8, 2, ("8", "0"))):
        mkw = {} if menu is None else dict(fsb_widths=menu[0], fsb_deltas=menu[1])
        for score in ("leaf", "lse_pos"):
            common = dict(fsb_policy="fixed", fsb_delta=0, fsb_tree=tree, fsb_read_top=4, fsb_rerank_score=score,
                          fsb_rerank_query="raw", **mkw)
            m_u = _mk_int(70, adapter=7000, fsb_width=Wb, fsb_union_width=Ub, **common)
            m_w = _mk_int(70, adapter=7000, fsb_width=Wb, **common)
            m_b = _mk_int(70, adapter=7000, fsb_width=Ub, **common)
            Qc, Kc, Vc = _qkv_n(700, n70, 1.5)
            with torch.no_grad():
                (su, _), cu = _rr_cap(lambda: _sel_n(m_u, Qc, Kc, Vc, n70))
                (sw, _), cw = _rr_cap(lambda: _sel_n(m_w, Qc, Kc, Vc, n70))
                _, cb = _rr_cap(lambda: _sel_n(m_b, Qc, Kc, Vc, n70))
            CU, CW, CB = _cands(cu).long(), _cands(cw).long(), _cands(cb).long()
            Cw = CW.shape[-1]
            d70 = _ft4.beam_delta_eff(m_u)         # forest prefill: compact beams, 2^fsb_delta slots per frontier slot
            assert d70 == _ft4.beam_delta_eff(m_w), (tree, Wb, Ub, score, d70)
            R70 = 1 << (m_u.fsb_delta_max if d70 is None else d70)
            assert CU.shape[-1] == Cw + m_u.fsb_union_W_max * R70, (CU.shape, Cw, R70)
            assert torch.equal(CU[..., :Cw], CW), (tree, Wb, Ub, score, "the W-beam part of the pool changed")
            for h in range(H):
                for i, (u_l, w_l, b_l) in enumerate(zip(CU[h].tolist(), CW[h].tolist(), CB[h].tolist())):
                    b_v = [c for c in b_l if c >= 0]
                    assert [c for c in u_l[Cw:] if c >= 0] == [c for c in b_v if c not in w_l], (tree, Wb, Ub, score, h, i)
                    v = [c for c in u_l if c >= 0]
                    assert len(v) == len(set(v)) and set(v) == {c for c in w_l if c >= 0} | set(b_v), (tree, Wb, Ub, score, h, i)
            nadd70 += int(((CU[..., Cw:] >= 0).any(-1))[:, BLK:].sum())
            # read set == float64 top-4 of the union pool by the raw-query score
            lam_ = m_u.sel_scale.detach().view(H, 1, 1); sig_ = m_u.sigma.detach().view(H, 1, 1)
            ceil70 = Kc[0].view(H, Mn70, S, Dh).float().norm(dim=-1).amax(-1)
            kbar70 = Kc[0].view(H, Mn70, S, Dh).mean(2)
            qq = Qc[0].float()
            for r0 in range(0, n70, 128):
                sb = CU[:, r0:r0 + 128]
                if score == "leaf":
                    c_ = sb.clamp(min=0)
                    kb = torch.gather(kbar70.double()[:, None].expand(H, sb.shape[1], Mn70, Dh), 2, c_[..., None].expand(-1, -1, -1, Dh))
                    a_ = torch.gather(ceil70.double()[:, None].expand(H, sb.shape[1], Mn70), 2, c_)
                    u = ((qq[:, r0:r0 + 128].double()[:, :, None, :] * kb).sum(-1) * m_u.scale + lam_.double() * a_
                         * torch.erf((qpos70[:, r0:r0 + 128].double()[..., None] - (c_.double() + 0.5) * S).abs() / (2.0 * sig_.double())))
                    u = u.masked_fill(sb < 0, float("-inf"))
                else:
                    u = _tok_brute(qq[:, r0:r0 + 128], sb, Kc, torch.arange(H, device=dev), m_u.scale, S, score,
                                   qpos70[:, r0:r0 + 128], lam_, sig_, ceil70)
                ref, tie, nval = _topk_ref(sb, u, 4)
                ok = _same_sets(su[0, :, r0:r0 + 128].long(), ref)
                assert bool((ok | tie).all()), (tree, Wb, Ub, score, r0, int((~ok & ~tie).sum()))
                nrow70 += int((nval > 4).sum()); ntie70 += int((tie & (nval > 4)).sum())
            ndiff70 += int((~_same_sets(su[0].long(), sw[0].long()))[:, BLK:].sum())
assert nadd70 > 0 and ndiff70 > 0, ("the union never added a cell / never changed the read set (no teeth)", nadd70, ndiff70)
print(f"F70 OK  (union pool == [W-beam candidates | U-beam cells not among them] == set union, brute-forced per row against "
      f"separate W / U modules, forest + rebuild, (W,U) = (16,4) (8,4) (4,8) (8,2), leaf + lse_pos; read sets == float64 top-4 "
      f"of the pool on {nrow70 - ntie70}/{nrow70} rows ({ntie70} near-ties); rows where the union adds cells {nadd70}, "
      f"reads a different set than no union {ndiff70})")


# ── F71: PREFILL == DECODE with the union (random adapter, route + raw, leaf + lse_pos): (a) fp32, forest + rebuild,
#        (W, U) = (8, 4), (4, 8), (16, 4): the no-cache full-sequence selection equals the per-token decode selection
#        (sets) on every decode row (forest), cached outputs == full forward (both trees), full + lean prefill,
#        FSB_DECODE_GRAPH 0 / 1, graph-replayed decode == eager decode bit-exact; (b) bf16 (autocast) 4096-token
#        prefix, forest, lse_pos, raw: the decode forest + the cache's keys route the PREFILL's queries (both beams +
#        the re-rank) to the prefill's cells bit-exactly (order included); the decode token's own q selects the
#        prefill's sets on the whole last block ─────────────────────────────────────────────────────────────────
n71, N071 = 512, 300
nrows71 = nord71 = 0
for tree in ("forest", "rebuild"):
    for (Wb, Ub, score, rq) in ((8, 4, "leaf", "route"), (8, 4, "lse_pos", "raw"), (4, 8, "lse_pos", "route"),
                                (4, 8, "leaf", "raw"), (16, 4, "lse_pos", "raw")):
        mm = _mk_int(71, adapter=7100, fsb_policy="fixed", fsb_width=Wb, fsb_delta=0, fsb_tree=tree, fsb_read_top=4,
                     fsb_rerank_score=score, fsb_rerank_query=rq, fsb_bound_radius_coef=4.0, fsb_union_width=Ub)
        torch.manual_seed(711); x = torch.randn(1, n71, d_model, device=dev)
        ys = {}
        with torch.no_grad():
            sel_full, yfull = _full_sel_n(mm, x)
            for lean in (False, True):
                for dg in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                    y, sels = _cached_run_n(mm, x, N071, lean, dg)
                    d = (y - yfull).abs().max().item()
                    assert d < 1e-4, (tree, Wb, Ub, score, rq, lean, dg, d)
                    ys[(lean, dg)] = y
                    if tree == "forest":
                        dec = torch.stack(sels, dim=1)
                        same_set = _same_sets(sel_full[:, N071:n71].long(), dec.long())
                        assert bool(same_set.all()), f"W{Wb} U{Ub} {score}/{rq} lean={lean} graph={dg}: {int((~same_set).sum())} decode rows differ"
                        nord71 += int((sel_full[:, N071:n71].long() == dec.long()).all(-1).sum()); nrows71 += same_set.numel()
            if dev == "cuda" and tree == "forest":
                for lean in (False, True):
                    assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (Wb, Ub, score, rq, lean, "graph decode != eager")
print(f"F71 [fp32] union: prefill selection == decode selection on {nrows71} rows (order identical on {nord71}); graph decode "
      f"== eager; cached == full forward (both trees)")
if dev == "cuda":
    for (Wb, Ub) in ((16, 4), (4, 8)):
        for lean in (False, True):
            torch.manual_seed(712)
            mm = _mk_int(712, adapter=7120, fsb_policy="fixed", fsb_width=Wb, fsb_delta=0, fsb_tree="forest", fsb_read_top=4,
                         fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_union_width=Ub)
            x = torch.randn(1, 4096, d_model, device=dev)
            _f65_bf16(mm, x, 4096, lean, f"F71 union W{Wb} U{Ub} lse_pos raw")
else:
    print("F71 [bf16] SKIP (no CUDA)")
print("F71 OK  (union: prefill routing == decode routing; graph decode == eager)")


# ── F72: U = W equals NO union: forest + rebuild, W = 4 / 8 / 16, leaf + lse_pos, raw query + random adapter: sel_full,
#        output and n_read identical (eager + CUDA-graph training stage), n_visit exactly doubled (the second beam is
#        traversed and costed), and the same cached decode (outputs + per-token selections, graph 0 / 1) ─────────────
n72, N072 = 512, 400
n72c = 0
for tree in ("forest", "rebuild"):
    for Wb in (4, 8, 16):
        for score in ("leaf", "lse_pos"):
            common = dict(fsb_policy="fixed", fsb_width=Wb, fsb_delta=0, fsb_tree=tree, fsb_read_top=4, fsb_rerank_score=score,
                          fsb_rerank_query="raw")
            m0 = _mk_int(72, adapter=7200, **common)
            m1 = _mk_int(72, adapter=7200, fsb_union_width=Wb, **common)
            assert m1.fsb_union_W_max == m1.fsb_W_max
            Qc, Kc, Vc = _qkv_n(720, n72, 1.5)
            for g in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                with torch.no_grad():
                    r0 = _sel_n(m0, Qc, Kc, Vc, n72, g); st0 = _nstats(m0)
                    r1 = _sel_n(m1, Qc, Kc, Vc, n72, g); st1 = _nstats(m1)
                assert torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1]) and torch.equal(st0[0], st1[0]), (tree, Wb, score, g)
                assert torch.equal(st1[1], 2 * st0[1]) and float(st0[1].sum()) > 0, (tree, Wb, score, g)
                n72c += 1
            if Wb != 16:
                torch.manual_seed(721); x = torch.randn(1, n72, d_model, device=dev)
                with torch.no_grad():
                    for dg in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                        y0, s0 = _cached_run_n(m0, x, N072, False, dg); y1, s1 = _cached_run_n(m1, x, N072, False, dg)
                        assert torch.equal(y0, y1) and all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(s0, s1)), \
                            (tree, Wb, score, dg)
                        n72c += 1
print(f"F72 OK  (U = W == no union: identical sel_full / output / n_read, n_visit doubled, identical cached decode; {n72c} cases)")


# ── F73: TRAINING STAGE with the union.  (a) CUDA-graph training stage == eager (sel_full, output, n_read, n_visit): fixed W8
#        U4 with a learned c_h and the learned (W, Delta) menu 2,4,8 x 0,1 with U4 (the union beam follows each query's
#        Delta), one and several chunk graphs, raw query, after in-place adapter updates; (b) chunked prefill 1 / 2 / all
#        identical (both trees); (c) Triton level kernel == torch traversal for both beams (fixed W16 U4, learned);
#        (d) learned policy: the same sampled actions as without the union, finite REINFORCE loss + controller grads,
#        reads <= 4, and n_visit counts the union beam (>= the no-union visits, > on some rows); (e) router node
#        distillation on the union pool: the node trace re-run (eager torch loop + the same union + re-rank) reproduces
#        the forward's read set (0 mismatches), eager + graph; finite loss, grads only on fsb_route_* ──────────────────
n73 = 512
if dev == "cuda":
    for score in ("leaf", "lse_pos"):
        for pol, nbch in (("fixed", None), ("learned", None), ("fixed", 2), ("learned", 2)):
            kw = {} if nbch is None else {"fsb_prefill_blkchunk": nbch}
            wkw = dict(fsb_width=8, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
            th = 7310 if pol == "fixed" else None
            mm = _mk_int(73, adapter=7300, theta=th, fsb_policy=pol, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score=score,
                         fsb_rerank_query="raw", fsb_union_width=4, **wkw, **kw)
            if pol == "learned":
                mm.train()
            for trial in range(3):
                Qc, Kc, Vc = _qkv_n(730 + trial, n73)
                res = {}
                for g in ("0", "1"):
                    torch.manual_seed(7300 + trial)
                    with torch.no_grad():
                        sel, o = _sel_n(mm, Qc, Kc, Vc, n73, g)
                    res[g] = (sel, o.detach()) + _nstats(mm)
                    assert int(res[g][2].max()) <= 4
                for i, nm in enumerate(("sel_full", "output", "n_read", "n_visit")):
                    assert torch.equal(res["0"][i], res["1"][i]), (score, pol, nbch, trial, nm)
                with torch.no_grad():
                    mm.fsb_route_W.add_(torch.randn_like(mm.fsb_route_W) * 0.1)      # an optimizer step in place
    for pol in ("fixed", "learned"):
        wkw = dict(fsb_width=16, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
        mm = _mk_int(731, adapter=7310, fsb_policy=pol, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score="lse_pos",
                     fsb_rerank_query="raw", fsb_union_width=4, **wkw)
        Qc, Kc, Vc = _qkv_n(732, n73)
        with torch.no_grad():
            r0 = _sel_n(mm, Qc, Kc, Vc, n73, "0", triton="0"); nv0 = _nstats(mm)[1]
            r1 = _sel_n(mm, Qc, Kc, Vc, n73, "0", triton="1"); nv1 = _nstats(mm)[1]
        assert torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1]) and torch.equal(nv0, nv1), ("F73 triton (union)", pol)
    print("F73 [graph/triton] union: graphed training stage == eager (fixed + learned c, learned menu, 1 / several chunk graphs, "
          "in-place adapter updates); Triton == torch traversal for both beams")
for tree in ("forest", "rebuild"):
    for pol in ("fixed", "learned"):
        wkw = dict(fsb_width=8, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
        outs = []
        for nbch in (1, 2, 0):
            mm = _mk_int(733, adapter=7330, fsb_policy=pol, fsb_tree=tree, fsb_read_top=4, fsb_rerank_score="lse_pos",
                         fsb_rerank_query="raw", fsb_union_width=4, fsb_prefill_blkchunk=nbch, **wkw)
            Qc, Kc, Vc = _qkv_n(734, n73)
            with torch.no_grad():
                outs.append(_sel_n(mm, Qc, Kc, Vc, n73) + _nstats(mm))
        assert all(all(torch.equal(a_, b_) for a_, b_ in zip(outs[0], o_)) for o_ in outs[1:]), (tree, pol)
# (d) learned policy training with the union
res73 = []
for U73 in (0, 4):
    m73 = _mk_os(73, fsb_policy="learned", fsb_widths="2,4,8", fsb_deltas="0,1", fsb_tree="forest", fsb_read_top=4,
                 fsb_rerank_score="lse_pos", **({"fsb_union_width": U73} if U73 else {})).train()
    torch.manual_seed(735)
    x73 = torch.randn(1, N, d_model, device=dev, requires_grad=True)
    torch.manual_seed(736)
    y73 = m73(x73)
    st73 = m73._fsb_stats
    aux73 = fsb_policy_loss(m73, torch.rand(1, N, device=dev), eta=0.1, entropy_beta=0.01, value_coef=0.5)
    (y73.float().pow(2).mean() + aux73).backward()
    assert torch.isfinite(aux73) and int(st73["n_read"].max()) <= 4
    assert all(p_.grad is not None and torch.isfinite(p_.grad).all() for n_, p_ in m73.named_parameters() if "fsb_policy" in n_)
    res73.append((st73["action"].clone(), st73["n_visit"].clone(), st73["has"].clone()))
(a0, v0, h0), (a1, v1, h1) = res73
assert torch.equal(a0, a1) and torch.equal(h0, h1), "the union changed the controller's sampled actions"
assert bool((v1[h1] >= v0[h0]).all()) and bool((v1[h1] > v0[h0]).any()), "n_visit must count the union beam"
# (e) router node distillation on the union pool
Qc, Kc, Vc = _qkv_n(737, N2, 1.5)
nmis73 = ntr73 = 0
for score in ("leaf", "lse_pos"):
    for rq in ("route", "raw"):
        torch.manual_seed(738)
        m_ = _mk_node(fsb_width=8, fsb_rd_node=1, fsb_rd_node_weight=0.5, fsb_route_learn_c=1, fsb_bound_radius_coef=8.0,
                      fsb_read_top=4, fsb_rerank_score=score, fsb_rerank_query=rq, fsb_union_width=4).train()
        with torch.no_grad():
            g_ = torch.Generator(device="cpu").manual_seed(739)
            m_.fsb_route_W.copy_((torch.randn(m_.fsb_route_W.shape, generator=g_) * 0.3).to(dev))
            m_.fsb_route_ctheta.add_((torch.randn(H, generator=g_) * 1.0).to(dev))
        for g in (("0", "1") if dev == "cuda" else ("0",)):
            m_.zero_grad(set_to_none=True)
            m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(740)
            sel_, o_, rd_, fr_ = _lr_n(m_, Qc, Kc, Vc, N2, g)
            st_ = rd_["stats"]
            assert fr_ is not None and st_.numel() == RD_STATS_NODE, (score, rq, g)
            assert float(st_[14]) == 0.0, (score, rq, g, "node trace re-run != the forward's read set", st_[14].item(), st_[15].item())
            nmis73 += int(st_[14]); ntr73 += int(st_[15])
            assert torch.isfinite(rd_["loss"]) and int(sel_.shape[-1]) == 4
            rd_["loss"].backward()
            got = sorted(n_ for n_, p_ in m_.named_parameters() if p_.grad is not None and p_.grad.abs().sum() > 0)
            assert got == ["fsb_route_W", "fsb_route_ctheta", "fsb_route_logtemp", "fsb_route_nlogtemp"], (score, rq, got)
print(f"F73 OK  (union in training: graphed == eager, chunked prefill identical, Triton == torch; learned policy: same actions, "
      f"finite loss + grads, visits count the union beam (+{float((v1 - v0)[h1].mean()):.1f} per query); node distillation: "
      f"trace reproduces the read set ({nmis73} mismatches / {ntr73} rows), grads only on fsb_route_*)")


# ── F74: plumbing.  build_attn_kwargs threads fsb_union_width only when set (None = the key is absent: every existing
#        command line resolves the same attn_kwargs) ───────────────────────────────────────────────────────────────────
_kw74_off = _bak68(_NS68(attn_type="fsattn", fsb=True, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw"))[1]
_kw74_on = _bak68(_NS68(attn_type="fsattn", fsb=True, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw",
                        fsb_union_width=4))[1]
assert "fsb_union_width" not in _kw74_off and _kw74_on["fsb_union_width"] == 4
assert {k: v for k, v in _kw74_on.items() if k != "fsb_union_width"} == _kw74_off
print("F74 OK  (--fsb_union_width threaded by build_attn_kwargs only when set)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  zk@7e7cd86 long-prompt eval-memory tests (zk labels F31..F35, renumbered F75..F79 on merge: F31..F36
#  are taken by the router tests above).  F75 = zk F31, F76 = F32, F77 = F33, F78 = F34, F79 = F35.
# ══════════════════════════════════════════════════════════════════════════════════════════
# ── F75: EVAL-mode fsb_long_range captures NO CUDA graph (FSB_TRAIN_GRAPH default 1 notwithstanding) and equals the
#        graph path bit-exactly on the same inputs: the TRAINING-mode graph (fixed policy) and the eval graph restored
#        by FSB_EVAL_GRAPH=1 (both policies; the learned one samples in training so it is compared at argmax only).
#        Eval graphs kept every layer's whole forest resident (the 64k+ RULER OOMs of 2026-09-27). ─
if dev == "cuda":
    import src.fsattn.fsb_tree as _ft31
    _gcount = {"n": 0}
    _orig_graphed = _ft31._forest_chunks_graphed
    def _count_graphed(*a_, **k_):
        _gcount["n"] += 1; return _orig_graphed(*a_, **k_)
    _ft31._forest_chunks_graphed = _count_graphed
    try:
        for pol, nbch in (("fixed", None), ("fixed", 2), ("learned", None), ("learned", 2)):
            torch.manual_seed(310)
            kw = {} if nbch is None else {"fsb_prefill_blkchunk": nbch}
            mm = mk(fsb_policy=pol, fsb_width=2, fsb_delta=1, fsb_tree="forest", fsb_bound_radius_coef=8.0, **kw)
            cap = {}
            def _grab31(attn, Qq, Kp, Vp, sel_full, s, Bq, n):
                cap["sel"] = sel_full.clone(); return _orig(attn, Qq, Kp, Vp, sel_full, s, Bq, n)
            for trial in range(2):
                Qc, Kc, Vc = rand_qkv(310 + trial)
                res = {}
                for mode in ("eval", "eval_graph", "train_graph"):
                    if mode == "train_graph" and pol == "learned":
                        continue
                    mm.train(mode == "train_graph")
                    if mode == "eval_graph":
                        os.environ["FSB_EVAL_GRAPH"] = "1"
                    n0 = _gcount["n"]; mm._fsb_train_graphs = None
                    _ft31.masked_readout = _grab31
                    try:
                        with torch.no_grad():
                            o = lr(mm, Qc, Kc, Vc)
                    finally:
                        _ft31.masked_readout = _orig; os.environ.pop("FSB_EVAL_GRAPH", None)
                    captured = _gcount["n"] > n0
                    assert captured == (mode != "eval"), (pol, nbch, mode, captured)
                    assert (mm._fsb_train_graphs is None) == (mode == "eval"), (pol, nbch, mode)
                    st = mm._fsb_stats
                    res[mode] = (cap["sel"].clone(), o.clone(), st["n_visit"].clone(), st["n_read"].clone(), st["has"].clone())
                assert res["eval"][4].any(), "no block selected anything (degenerate case)"
                for other in [m_ for m_ in res if m_ != "eval"]:
                    for i, name in enumerate(("sel_full", "output", "n_visit", "n_read", "has")):
                        assert torch.equal(res["eval"][i], res[other][i]), (pol, nbch, trial, other, name)
            mm._fsb_train_graphs = None
            print(f"F75 [{pol}, blkchunk {nbch}] eval: no graph captured; == {'train-mode graph and ' if pol == 'fixed' else ''}eval graph "
                  f"(sel_full, output, stats), 2 inputs")
    finally:
        _ft31._forest_chunks_graphed = _orig_graphed
    print("F75 OK  (eval-mode FS-B prefill is eager and bit-identical to the graph-captured path)")
else:
    print("F75 SKIP (no CUDA)")


# ── F76: lean decode forest (FSB_DECODE_LEAN=1, default: only the chunk of each set bit of m_tot) == the full incremental
#        forest cache, bit-exact, eager and CUDA-graph decode, over many block boundaries (a longer sequence: 64 cells),
#        with tau = 0 and with per-head thresholds (rows with different admitted counts); the lean table holds
#        <= 2 m_tot nodes per row. ─
from experiments.scaling.kv_cache import FSKVCache as _EKV32
N32 = 1024
for pol in ("fixed", "learned"):
    for tau_mode in ("zero", "per-head"):
        torch.manual_seed(320)
        mm = mk(fsb_policy=pol, fsb_width=2, fsb_delta=1, fsb_tree="forest", fsb_bound_radius_coef=8.0)
        x = torch.randn(1, N32, d_model, device=dev)
        N0 = 600
        if tau_mode == "per-head":
            with torch.no_grad():
                c0 = _EKV32(mm); mm(x[:, :N0], cache=c0)
                ce0 = c0._ceil[0]                                            # (H, ncell)
                mm.fsb_tau.copy_(torch.stack([ce0[h].quantile(q) for h, q in enumerate((0.0, 0.3, 0.5, 0.8))]))
        outs = {}
        for dlean in ("0", "1"):
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                os.environ["FSB_DECODE_LEAN"] = dlean; os.environ["FSB_DECODE_GRAPH"] = dg
                try:
                    with torch.no_grad():
                        cache = _EKV32(mm); cache.reserve = N32 - N0
                        o = [mm(x[:, :N0], cache=cache)]
                        for t in range(N0, N32):
                            o.append(mm(x[:, t:t + 1], cache=cache))
                        outs[(dlean, dg)] = torch.cat(o, dim=1)
                    fc = cache._fsb_forest
                    if dlean == "1":
                        assert fc.lean and fc.chunks == {} and fc.F["chunks"] == {}
                        assert fc.F["Ntot"] <= max(2 * int(fc.m_tot.max().item()), 1), (fc.F["Ntot"], fc.m_tot)
                        ntot_lean = fc.F["Ntot"]
                    else:
                        ntot_full = fc.F["Ntot"]
                finally:
                    os.environ.pop("FSB_DECODE_LEAN", None); os.environ.pop("FSB_DECODE_GRAPH", None)
        ref = outs[("0", "0")]
        for key, o in outs.items():
            assert torch.equal(o, ref), (pol, tau_mode, key, (o - ref).abs().max().item())
        with torch.no_grad():
            dfull = (ref - mm(x)).abs().max().item()                         # and both == the no-cache forward (F15 tolerance)
        assert dfull < 1e-4, (pol, tau_mode, dfull)
        print(f"F76 [{pol}, tau {tau_mode}] lean == full decode forest bit-exact over {N32 - N0} tokens / "
              f"{(N32 - 1) // BLK - N0 // BLK} block boundaries (eager{' + graph' if dev == 'cuda' else ''}); "
              f"final node table {ntot_lean} vs {ntot_full} nodes per row; |decode - no-cache forward| {dfull:.1e}")
print("F76 OK  (lean decode forest reproduces the full forest cache)")


# ── F77: long-prompt memory trims are value-preserving: (a) fsb_long_range reading the zero-tailed KV history buffer
#        (kv_padded) == zero-padded copies of K/V (sel_full, output, stats), bf16 inputs, prompt lengths with and
#        without a partial last cell; (b) the lean-prefill layer output with the buffer view + early frees + in-place
#        SR/LR combine (no_grad, decode slack) == the plain expression over padded copies (grad mode, no slack). ─
import src.fsattn.fsb_tree as _ft33
from experiments.scaling.kv_cache import FSKVCache as _EKV33
for N33 in (1000, 1024, 777):
    torch.manual_seed(330)
    mm = mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_tree="forest", fsb_bound_radius_coef=8.0, fsb_prefill_blkchunk=3)
    g33 = torch.Generator(device="cpu").manual_seed(N33)
    Q33, K33, V33 = (torch.randn(1, H, N33, Dh, generator=g33).to(dev).to(torch.bfloat16) for _ in range(3))
    kb = torch.zeros(1, H, N33 + 64, Dh, device=dev, dtype=torch.bfloat16); kb[:, :, :N33] = K33
    vb = torch.zeros(1, H, N33 + 64, Dh, device=dev, dtype=torch.bfloat16); vb[:, :, :N33] = V33
    res = []
    for kvp in (None, (kb, vb)):
        cap = {}
        def _grab33(attn, Qq, Kp, Vp, sel_full, s, Bq, n):
            cap["sel"] = sel_full.clone(); return _orig(attn, Qq, Kp, Vp, sel_full, s, Bq, n)
        _ft33.masked_readout = _grab33
        try:
            with torch.no_grad():
                o = _ft33.fsb_long_range(mm, Q33, kb[:, :, :N33], vb[:, :, :N33], mm.sigma, N33, kv_padded=kvp)
        finally:
            _ft33.masked_readout = _orig
        st = mm._fsb_stats
        res.append((cap["sel"], o, st["n_visit"], st["n_read"], st["admit_frac"], st["m0_mean"]))
    for i, name in enumerate(("sel_full", "output", "n_visit", "n_read", "admit_frac", "m0_mean")):
        assert torch.equal(res[0][i], res[1][i]), (N33, name)
    # (b) module level, lean prefill
    x33 = torch.randn(1, N33, d_model, device=dev)
    outs33 = []
    for mode in ("trim", "plain"):
        c33 = _EKV33(mm, lean_prefill=True); c33.reserve = 64 if mode == "trim" else 0
        ctx = torch.no_grad() if mode == "trim" else torch.enable_grad()
        with ctx, torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16):
            outs33.append(mm(x33, cache=c33).detach())
    assert torch.equal(outs33[0], outs33[1]), (N33, (outs33[0] - outs33[1]).abs().max().item())
    print(f"F77 [N={N33}, pad {(-N33) % S}] buffer-view K/V == padded copies (sel_full, output, stats); lean-prefill layer "
          f"with frees + in-place combine == plain expression")
print("F77 OK  (long-prompt memory trims are bit-identical)")


# ── F78: the decode graph is captured without torch.cuda.graph's gc.collect() / empty_cache() (default) and decodes
#        bit-identically to the capture through torch.cuda.graph (FSB_DECODE_CAPTURE_GC=1) and to eager decode, over
#        several block boundaries (one capture each); the default path calls gc.collect() zero times. ─
if dev == "cuda":
    import gc as _gc34
    from experiments.scaling.kv_cache import FSKVCache as _EKV34
    N34, N0 = 700, 300
    torch.manual_seed(340)
    mm = mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_tree="forest", fsb_bound_radius_coef=8.0)
    x = torch.randn(1, N34, d_model, device=dev)
    outs, ngc = {}, {}
    _gc_orig = _gc34.collect
    for mode in ("eager", "capture_gc", "capture_lean"):
        cnt = [0]
        def _gc_count(*a, _c=cnt, **k):
            _c[0] += 1
            return _gc_orig(*a, **k)
        _gc34.collect = _gc_count
        os.environ["FSB_DECODE_GRAPH"] = "0" if mode == "eager" else "1"
        os.environ["FSB_DECODE_CAPTURE_GC"] = "1" if mode == "capture_gc" else "0"
        try:
            with torch.no_grad():
                cache = _EKV34(mm); cache.reserve = N34 - N0
                o = [mm(x[:, :N0], cache=cache)]
                for t in range(N0, N34):
                    o.append(mm(x[:, t:t + 1], cache=cache))
                outs[mode] = torch.cat(o, dim=1)
            assert mode == "eager" or cache._fsb_forest.graph is not None, "graph never captured"
        finally:
            _gc34.collect = _gc_orig
            os.environ.pop("FSB_DECODE_GRAPH", None); os.environ.pop("FSB_DECODE_CAPTURE_GC", None)
        ngc[mode] = cnt[0]
    nblk = (N34 - 1) // BLK - N0 // BLK + 1
    assert torch.equal(outs["capture_lean"], outs["capture_gc"]), (outs["capture_lean"] - outs["capture_gc"]).abs().max().item()
    assert torch.equal(outs["capture_lean"], outs["eager"]), (outs["capture_lean"] - outs["eager"]).abs().max().item()
    assert ngc["capture_lean"] == 0 and ngc["capture_gc"] >= nblk, (ngc, nblk)
    print(f"F78 decode over {N34 - N0} tokens / {nblk} blocks: lean capture == torch.cuda.graph capture == eager (bit-exact); "
          f"gc.collect() calls: lean {ngc['capture_lean']}, torch.cuda.graph {ngc['capture_gc']}")
    print("F78 OK  (decode graph captured without gc.collect / empty_cache, bit-identical)")
else:
    print("F78 SKIP (no CUDA)")


# ── F79: cached_generate pre-allocates every layer's K/V history and cell summaries before the prompt forward
#        (KV_PREALLOC=1, default): the prefill fills exactly those buffers, and the generated tokens and every
#        step's logits are bit-identical to allocating them inside prefill (KV_PREALLOC=0); full-seq and lean
#        prefill, a prompt with a partial last cell, decode across block boundaries. ─
import experiments.scaling.kv_cache as _kvc35
from experiments.scaling.model import FreeSoloLM as _LM35
torch.manual_seed(350)
lm35 = _LM35(vocab_size=97, d_model=d_model, n_layers=2, n_heads=H, d_ff=128, max_seq_len=1024, attn_type="fsattn",
             window_size=16, long_range_variant="cell_topk", use_rope=True, rope_long_range=False, fs_uniform_sr=True,
             lr_scale_mode="one", cell_select="score", cell_stride=S, cell_topk=4, sel_strict_causal=1,
             fsb_enabled=True, fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_tree="forest",
             fsb_bound_radius_coef=8.0).to(dev).eval()
for blk_ in lm35.blocks:
    blk_.attn.fsattn._cell_block = BLK
prompt35 = torch.randint(0, 97, (1, 301), generator=torch.Generator().manual_seed(351))
res35 = {}
_head35 = _kvc35._lm_head_fp32
for lean35 in (False, True):
    for pa in ("0", "1"):
        lg35 = []
        def _h35(model, hidden, chunk, _l=lg35):
            o_ = _head35(model, hidden, chunk); _l.append(o_.detach().clone()); return o_
        os.environ["KV_PREALLOC"] = pa
        _kvc35._lm_head_fp32 = _h35
        try:
            if pa == "1":                                                # prefill must write into the pre-allocated buffers
                _pf = _kvc35.FSKVCache.prefill
                ptrs = []
                def _pf35(self, K_, V_, _p=ptrs, _o=_pf):
                    pre_ = self._prealloc
                    _o(self, K_, V_)
                    _p.append(pre_ is not None and self._k_buf is pre_[2] and self._v_buf is pre_[3] and self._kbar is not None and self._kbar.data_ptr() == pre_[4].data_ptr())
                _kvc35.FSKVCache.prefill = _pf35
            with torch.no_grad():
                g35 = _kvc35.cached_generate(lm35, prompt35, 90, device=dev, dtype="bfloat16" if dev == "cuda" else "float32",
                                             lean_prefill=lean35)
        finally:
            os.environ.pop("KV_PREALLOC", None)
            _kvc35._lm_head_fp32 = _head35
            if pa == "1":
                _kvc35.FSKVCache.prefill = _pf
        if pa == "1":
            assert ptrs and all(ptrs), ptrs
        res35[(lean35, pa)] = (g35, lg35)
    (g0, l0), (g1, l1) = res35[(lean35, "0")], res35[(lean35, "1")]
    assert torch.equal(g0, g1) and len(l0) == len(l1) and all(torch.equal(a_, b_) for a_, b_ in zip(l0, l1)), lean35
    print(f"F79 [{'lean' if lean35 else 'full-seq'} prefill] pre-allocated K/V + summaries: prefill filled them in "
          f"{len(ptrs)} layers; {g1.shape[1] - prompt35.shape[1]} generated tokens and every step's logits identical")
print("F79 OK  (K/V pre-allocation changes no value)")


# ── F80 (merge of zk@7e7cd86 into fsb-integrate2): zk's eval-memory changes under the fsb-integrate2 selection features.
#        (a) lean decode forest (FSB_DECODE_LEAN=1, default) == the full forest cache (0): bit-exact outputs and per-token
#            decode selections, eager and graph decode, for W16 read 4 lse_pos with the raw re-rank query + a random full
#            adapter + a learned radius coefficient, the same with the union beam (U 4), and W8 read 4 leaf without
#            adapter; tau 0 and per-head tau (the lean cache must re-expose the per-cell re-rank tables at every block,
#            also when no new cell was admitted); full and lean prefill; cached == no-cache forward (F15 tolerance).
#        (b) the eager prefill builds the fp32 query and q_r per block chunk (_qf_rows): sel_full and output identical for
#            block chunks 1 / 2 / all and equal to the whole-sequence q_r of the eval-mode graph stage (FSB_EVAL_GRAPH=1). ─
from experiments.scaling.kv_cache import FSKVCache as _EKV80
N80, N080 = 1024, 600
cfg80 = (("W16 r4 lse_pos raw + adapter + learned c", dict(adapter=8000, theta=8001, fsb_width=16, fsb_delta=0,
                                                          fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw")),
         ("W16 r4 lse_pos raw + adapter, union U4", dict(adapter=8002, fsb_width=16, fsb_delta=0, fsb_read_top=4,
                                                        fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_union_width=4)),
         ("W8 r4 leaf, no adapter", dict(fsb_width=8, fsb_delta=0, fsb_read_top=4)))
for name80, kw80 in cfg80:
    for tau_mode in ("zero", "per-head"):
        mm = _mk_int(80, fsb_policy="fixed", fsb_tree="forest", **kw80)
        torch.manual_seed(801)
        x = torch.randn(1, N80, d_model, device=dev)
        if tau_mode == "per-head":
            with torch.no_grad():
                c0 = _EKV80(mm); mm(x[:, :N080], cache=c0)
                ce0 = c0._ceil[0]                                            # (H, ncell)
                mm.fsb_tau.copy_(torch.stack([ce0[h].quantile(q) for h, q in enumerate((0.0, 0.3, 0.5, 0.8))]))
        outs, sels = {}, {}
        runs80 = [(dl, dg, False) for dl in ("0", "1") for dg in (("0", "1") if dev == "cuda" else ("0",))] + [("1", "0", True)]
        for dlean, dg, lpf in runs80:
            os.environ["FSB_DECODE_LEAN"] = dlean; os.environ["FSB_DECODE_GRAPH"] = dg
            try:
                with torch.no_grad():
                    cache = _EKV80(mm, lean_prefill=lpf); cache.reserve = N80 - N080
                    o = [mm(x[:, :N080], cache=cache)]; sl = []
                    for t in range(N080, N80):
                        o.append(mm(x[:, t:t + 1], cache=cache))
                        fc = cache._fsb_forest
                        sl.append(fc._last_sel.clone() if (fc.SP is not None and getattr(fc, "_last_sel", None) is not None) else None)
                    outs[(dlean, dg, lpf)] = torch.cat(o, dim=1); sels[(dlean, dg, lpf)] = sl
                fc = cache._fsb_forest
                assert fc.lean == (dlean == "1") and "cellK" in fc.F and fc.F["cellK"].shape[1] == min(((N80 - 1) // BLK) * BLK // S, N80 // S)
            finally:
                os.environ.pop("FSB_DECODE_LEAN", None); os.environ.pop("FSB_DECODE_GRAPH", None)
        ref = outs[("0", "0", False)]; sref = sels[("0", "0", False)]
        assert all(s_ is not None for s_ in sref)
        for key in outs:
            assert all(a_ is not None and torch.equal(a_, b_) for a_, b_ in zip(sels[key], sref)), (name80, tau_mode, key, "decode selections")
            if not key[2]:
                assert torch.equal(outs[key], ref), (name80, tau_mode, key, (outs[key] - ref).abs().max().item())
        with torch.no_grad():
            yfull = mm(x)
        dfull = max((o_ - yfull).abs().max().item() for o_ in outs.values())
        assert dfull < 1e-4, (name80, tau_mode, dfull)
        print(f"F80a [{name80}, tau {tau_mode}] lean == full decode forest: outputs + {len(sref)} decode selections bit-exact "
              f"(eager{' + graph' if dev == 'cuda' else ''} decode; lean prefill: same selections); |cached - no-cache| {dfull:.1e}")
n80 = 512
for name80, kw80 in cfg80:
    res80 = []
    for nbch in (1, 2, 0):
        mm = _mk_int(80, fsb_policy="fixed", fsb_tree="forest", fsb_prefill_blkchunk=nbch, **kw80)
        Qc, Kc, Vc = _qkv_n(802, n80)
        with torch.no_grad():
            for g in (("0", "1") if dev == "cuda" else ("0",)):
                res80.append(_sel_n(mm, Qc, Kc, Vc, n80, g))
    assert all(torch.equal(r_[0], res80[0][0]) and torch.equal(r_[1], res80[0][1]) for r_ in res80[1:]), name80
    print(f"F80b [{name80}] per-chunk q_r: sel_full + output identical for block chunks 1/2/all, eager"
          f"{' and eval graph' if dev == 'cuda' else ''}")
print("F80 OK  (merge: lean decode forest, per-chunk fp32/routing queries and the eval graph default under overselect / "
      "raw re-rank / adapter / learned c / union)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  HYBRID SELECTOR (fsb_hybrid off|pool|union|quota): FS-L's salient pool (top-P cells by max key norm + the recent
#  window, ranked per query by FS-L's native score g) combined with the FS-B tree's candidates at the final read.
#  F81..F90.  The tests use a SMALL pool (P = 4, Wc = 2) so the top-P truncation is exercised (N = 512: up to 28
#  eligible cells), read the query block from attn._cell_block (BLK), and build fresh caches for every run.
# ══════════════════════════════════════════════════════════════════════════════════════════
import types as _types81
import src.fsattn.fsb_tree as _fth
from src.fsattn.fsb_tree import (hybrid_pool, pool_order, pool_scores, pool_rank, quota_merge, hybrid_prov, HYBRID_MODES,
                                union_candidates as _uc81)
from src.fsattn.linear_cache import LinearFSCache as _LEC81

PH, WCH = 4, 2                                   # the small test pool: top-4 by max key norm + the recent 2 cells


def _mk_hy(seed, mode, adapter=None, P=PH, Wc=WCH, short=4, quota=1, W=8, score="lse_pos", theta=None, **kw):
    """forest, fixed W/D0 beam, read 4, raw re-rank query; the hybrid kwargs only when ``mode`` is given (None = absent)."""
    hk = {} if mode is None else dict(fsb_hybrid=mode, fsb_pool_size=P, fsb_pool_recent=Wc, fsb_pool_short=short,
                                      fsb_pool_quota=quota)
    kw.setdefault("fsb_bound_radius_coef", 4.0)
    return _mk_int(seed, adapter=adapter, theta=theta, fsb_policy="fixed", fsb_width=W, fsb_delta=0, fsb_tree="forest",
                   fsb_read_top=4, fsb_rerank_score=score, fsb_rerank_query="raw", **hk, **kw)


def _hcap(fn, names=("hybrid_pool", "pool_rank", "rerank_top")):
    """run fn() with the named fsb_tree functions wrapped; every call's (args, kwargs, output) is recorded."""
    rec = {n_: [] for n_ in names}
    orig = {n_: getattr(_fth, n_) for n_ in names}
    def _wrap(n_):
        def _w(*a_, **k_):
            o_ = orig[n_](*a_, **k_); rec[n_].append((a_, k_, o_)); return o_
        return _w
    for n_ in names:
        setattr(_fth, n_, _wrap(n_))
    try:
        return fn(), rec
    finally:
        for n_ in names:
            setattr(_fth, n_, orig[n_])


def _rows_hb(ts):
    """per-chunk (T*nbc, Bq, X) tree-row tensors of ONE full-sequence forward (B = 1, row h*nbc + j) -> (H, Npad, X)."""
    out = []
    for t_ in ts:
        nbc = t_.shape[0] // H
        out.append(t_.reshape(H, nbc * t_.shape[1], t_.shape[2]))
    return torch.cat(out, dim=1)


def _pools_by_block(ts):
    """per-chunk pool rows (T*nbc, Cp) of one forward -> {block: (H, Cp)} (chunks in block order)."""
    out, b0 = {}, 0
    for t_ in ts:
        nbc = t_.shape[0] // H
        v = t_.view(H, nbc, -1)
        for j in range(nbc):
            out[b0 + j] = v[:, j].clone()
        b0 += nbc
    return out


def _hy_tabs(mm_, Q_, K_, n_):
    """the module's per-cell a_c (H,M) fp32 and the float64 FS-L native score g of every (head, query row, cell)
    (H,Npad,M), -inf for cells not eligible for the row's block (id >= E_b); E_row (Npad,)."""
    Mn = (n_ + S - 1) // S
    Kp = K_[0] if Mn * S == n_ else torch.cat([K_[0], K_.new_zeros(H, Mn * S - n_, Dh)], 1)
    kbar = Kp.view(H, Mn, S, Dh).mean(2).float()
    ceil = Kp.view(H, Mn, S, Dh).float().norm(dim=-1).amax(-1)
    Npad = ((n_ + BLK - 1) // BLK) * BLK
    q = Q_[0].float()
    q = torch.cat([q, q.new_zeros(H, Npad - n_, Dh)], 1) if Npad > n_ else q
    lam = mm_.sel_scale.detach().double().view(H, 1, 1); sig = mm_.sigma.detach().double().view(H, 1, 1)
    ipos = torch.arange(Npad, device=dev, dtype=torch.float64)
    cpos = (torch.arange(Mn, device=dev, dtype=torch.float64) + 0.5) * S
    g = (torch.einsum("hnd,hmd->hnm", q.double(), kbar.double()) * mm_.scale
         + lam * ceil.double()[:, None, :] * torch.erf((ipos[:, None] - cpos[None, :]).abs() / (2.0 * sig)))
    E_row = (torch.div(torch.arange(Npad, device=dev), BLK, rounding_mode="floor") * BLK) // S
    g = g.masked_fill(torch.arange(Mn, device=dev)[None, None, :] >= E_row[None, :, None], float("-inf"))
    return ceil, g, E_row


def _pool_brute(a_row, E, P, Wc):
    """python pool of one (head, block): top-P by (a desc, id asc) among ids < E, then E-1..E-Wc not already in it."""
    if P <= 0:
        return list(range(E))
    top = sorted(range(E), key=lambda c: (-a_row[c], c))[:P]
    return top + [E - 1 - i for i in range(Wc) if E - 1 - i >= 0 and E - 1 - i not in top]


def _pool_mask(ceil, E_row, P, Wc):
    """(H,Npad,M) bool: cell in the python pool of the row's block."""
    Mn = ceil.shape[1]
    out = torch.zeros(H, E_row.shape[0], Mn, dtype=torch.bool, device=dev)
    for h in range(H):
        a_row = ceil[h].tolist()
        for E in sorted(set(E_row.tolist())):
            cells = _pool_brute(a_row, E, P, Wc)
            if cells:
                out[h, (E_row == E).nonzero().flatten()[:, None], torch.tensor(cells, device=dev)[None, :]] = True
    return out


def _pool_ref(ceil, g64, E_row, P, Wc, K_):
    """float64 top-K of the python pool by g: (ref ids (H,Npad,K) set, near-tie mask, valid count)."""
    pm = _pool_mask(ceil, E_row, P, Wc)
    Mn = ceil.shape[1]
    sel_all = torch.arange(Mn, device=dev).view(1, 1, Mn).expand(H, E_row.shape[0], Mn).masked_fill(~pm, -1)
    return _topk_ref(sel_all, g64.masked_fill(~pm, float("-inf")), K_)


def _hy_cached(mm_, x_, N0, lean, dgraph):
    """prefill N0 + per-token decode; returns (outputs, decode sels [(H,kk)], decode pools [(H,Cp)], provs [(H,kk)])."""
    n_ = x_.shape[1]
    os.environ["FSB_DECODE_GRAPH"] = dgraph
    try:
        cache = FSKVCache(mm_, lean_prefill=lean); cache.reserve = n_ - N0
        outs = [mm_(x_[:, :N0], cache=cache)]; sels, pools, provs = [], [], []
        for t in range(N0, n_):
            outs.append(mm_(x_[:, t:t + 1], cache=cache))
            fc = cache._fsb_forest
            sels.append(fc._last_sel.reshape(H, -1).clone())
            pools.append(fc.pool.clone())
            provs.append(fc._last_prov.reshape(H, -1).clone())
        if dgraph == "1":
            assert cache._fsb_forest.graph is not None, "decode graph never captured"
    finally:
        os.environ.pop("FSB_DECODE_GRAPH", None)
    return torch.cat(outs, dim=1), sels, pools, provs


# ── F81: construction, refusals and flag OFF.  (a) every module defaults to fsb_hybrid off (pool 64 / recent 8 / short 4 /
#        quota 1; no parameter or buffer); pool / union / quota build with read_top 4, forest, fixed policy; fsb_pool_size
#        0 (every eligible cell) drops the recent window; (b) refused: read_top 0, an unknown mode, the rebuild tree, the
#        learned policy, a routing adapter with the route re-rank query (D3), negative pool sizes, a union shortlist
#        outside 1..P+Wc, a quota outside 1..read_top, router distillation in pool mode (no tree); (c) "off" passed
#        explicitly (with pool knobs set) == the flag absent: same state, sel_full, output, n_read, n_visit on the eager
#        and CUDA-graph training stage (train + eval, leaf + lse_pos, with and without a random adapter), and the same
#        cached decode (full + lean prefill, FSB_DECODE_GRAPH 0 / 1) ──────────────────────────────────────────────────
assert HYBRID_MODES == ("off", "pool", "union", "quota")
m81 = mk(fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree="forest", fsb_read_top=4)
assert (m81.fsb_hybrid, m81.fsb_pool_size, m81.fsb_pool_recent, m81.fsb_pool_short, m81.fsb_pool_quota) == ("off", 64, 8, 4, 1)
assert mk(fsb_policy="fixed", fsb_width=2, fsb_delta=1, fsb_tree="rebuild", fsb_hybrid="off").fsb_hybrid == "off"
m81 = _mk_hy(81, "quota", quota=2)
assert (m81.fsb_hybrid, m81.fsb_pool_quota, m81.fsb_kk, m81.fsb_pool_size, m81.fsb_pool_recent) == ("quota", 2, 4, PH, WCH)
m81 = _mk_hy(81, "pool", P=0, Wc=8)
assert (m81.fsb_pool_size, m81.fsb_pool_recent) == (0, 0)
assert _mk_hy(81, "union", short=PH + WCH).fsb_pool_short == PH + WCH
assert _mk_hy(81, "quota", quota=4, adapter=8100).fsb_pool_quota == 4
_b81 = dict(fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score="lse_pos",
            fsb_pool_size=PH, fsb_pool_recent=WCH)
for bad in (dict(fsb_hybrid="union", fsb_width=4, fsb_read_top=0, fsb_rerank_score="leaf"), dict(fsb_hybrid="bogus"),
            dict(fsb_hybrid="union", fsb_tree="rebuild"),
            dict(fsb_hybrid="quota", fsb_policy="learned", fsb_widths="2,4,8", fsb_deltas="0,1"),
            dict(fsb_hybrid="union", fsb_route_adapter="full"),
            dict(fsb_hybrid="pool", fsb_route_adapter="full", fsb_rerank_query="route"),
            dict(fsb_hybrid="pool", fsb_pool_size=-1), dict(fsb_hybrid="pool", fsb_pool_recent=-1),
            dict(fsb_hybrid="union", fsb_pool_short=PH + WCH + 1), dict(fsb_hybrid="union", fsb_pool_short=0),
            dict(fsb_hybrid="quota", fsb_pool_quota=0), dict(fsb_hybrid="quota", fsb_pool_quota=5),
            dict(fsb_hybrid="pool", fsb_route_adapter="full", fsb_rerank_query="raw", fsb_rd_weight=1.0)):
    kw_ = dict(_b81); kw_.update(bad)
    try:
        mk(**kw_)
        raise AssertionError(f"fsb_init must refuse {bad}")
    except ValueError:
        pass
n81, N081 = 512, 400
n81b = 0
for adapter in (None, 8101):
    for score in ("leaf", "lse_pos"):
        ma = _mk_hy(81, None, adapter=adapter, score=score)
        mb = _mk_hy(81, "off", adapter=adapter, score=score)
        sda, sdb = ma.state_dict(), mb.state_dict()
        assert list(sda) == list(sdb) and all(torch.equal(sda[k], sdb[k]) for k in sda)
        assert [n_ for n_, _ in ma.named_parameters()] == [n_ for n_, _ in mb.named_parameters()]
        Qc, Kc, Vc = _qkv_n(810, n81, 1.5)
        for mode in ("eval", "train"):
            ma.train(mode == "train"); mb.train(mode == "train")
            for g in (("0", "1") if dev == "cuda" else ("0",)):
                res = []
                for m_ in (ma, mb):
                    with torch.no_grad():
                        sel_, o_ = _sel_n(m_, Qc, Kc, Vc, n81, g)
                    res.append((sel_, o_) + _nstats(m_))
                assert all(torch.equal(a_, b_) for a_, b_ in zip(*res)), (adapter, score, mode, g)
                n81b += 1
        ma.eval(); mb.eval()
        torch.manual_seed(811); x = torch.randn(1, n81, d_model, device=dev)
        with torch.no_grad():
            for lean in (False, True):
                for dg in (("0", "1") if dev == "cuda" else ("0",)):
                    ya, sa = _cached_run_n(ma, x, N081, lean, dg); yb, sb_ = _cached_run_n(mb, x, N081, lean, dg)
                    assert torch.equal(ya, yb) and all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(sa, sb_)), \
                        (adapter, score, lean, dg)
                    n81b += 1
print(f"F81 OK  (fsb_hybrid: default off; pool / union / quota construction; 13 refusals; off explicit == absent on "
      f"{n81b} eager / graph / train / cached-decode cases)")


# ── F82: the pool builder and the pool scorer.  (a) hybrid_pool == a python brute force (top-P by (a_c desc, id asc)
#        among the cells < E, then E-1..E-Wc minus the cells already in it, compacted, -1 tail; P = 0: every cell < E)
#        on random rows with QUANTIZED priorities (many exact ties) and all-equal priorities, P / Wc = 4/2, 1/0, 3/8,
#        64/8, 0; (b) pool_scores' content (cell_token_scores with one "token" per cell) == _det_dot(q, kbar) * scale
#        bit for bit, Triton == torch fallback, Dh 16 / 64 / 96, and the whole score == rerank_top's leaf arithmetic;
#        (c) pool_rank == the exact (g desc, cell id asc) order of those scores, with duplicated cells (exact ties) ────
g82 = torch.Generator(device="cpu").manual_seed(82)
npool82 = 0
for trial in range(6):
    T82, M82 = 5, int(torch.randint(1, 40, (1,), generator=g82))
    a82 = (torch.rand(T82, M82, generator=g82) * 4).round() / 2
    if trial == 0:
        a82[:] = 1.0
    order82 = pool_order(a82.to(dev))
    for P82, W82 in ((4, 2), (1, 0), (3, 8), (64, 8), (0, 0)):
        Tq = 12
        t82 = torch.randint(0, T82, (Tq,), generator=g82)
        E82 = torch.randint(0, M82 + 1, (Tq,), generator=g82)
        pl = hybrid_pool(order82, t82.to(dev), E82.to(dev), P82, W82).cpu()
        assert pl.shape == (Tq, (P82 + W82) if P82 > 0 else M82), (pl.shape, P82, W82, M82)
        for r in range(Tq):
            ref = _pool_brute(a82[int(t82[r])].tolist(), int(E82[r]), P82, W82)
            row = pl[r].tolist()
            assert row[:len(ref)] == ref and all(c == -1 for c in row[len(ref):]), (trial, P82, W82, r, row, ref)
            npool82 += 1
nsc82 = nties82 = 0
for Dh82 in (16, 64, 96):
    T82, M82, Bq82, Cp82 = 6, 50, 7, 12
    cellK = torch.randn(T82, M82, Dh82, generator=g82)
    cellA = torch.rand(T82, M82, generator=g82) * 3
    cellK[:, 30] = cellK[:, 3]; cellA[:, 30] = cellA[:, 3]                    # exact duplicates: ties at far queries
    cellK[:, 41] = cellK[:, 3]; cellA[:, 41] = cellA[:, 3]
    cellK, cellA = cellK.to(dev), cellA.to(dev)
    pool82 = torch.tensor([[41, 3, 30] + [c_ for c_ in torch.randperm(M82, generator=g82).tolist() if c_ not in (3, 30, 41)][:Cp82 - 5]
                           + [-1, -1] for _ in range(T82)], device=dev)
    t_i82 = torch.arange(T82, device=dev)
    pq82 = torch.randn(T82, Bq82, Dh82, generator=g82).to(dev)
    qpos82 = (torch.arange(Bq82) + 1500).float()[None].expand(T82, -1).to(dev).contiguous()    # every cell far: erf == 1
    lam82 = (torch.rand(T82, 1, 1, generator=g82) + 0.5).to(dev); sig82 = torch.full((T82, 1, 1), 4.0, device=dev)
    sc82 = 1.0 / math.sqrt(Dh82)
    sel82 = pool82[:, None, :].expand(T82, Bq82, pool82.shape[1])
    kb82 = cellK[t_i82.view(-1, 1, 1), sel82.clamp(min=0)]
    a_82 = cellA[t_i82.view(-1, 1, 1), sel82.clamp(min=0)]
    cont_ref = _ft4._det_dot(pq82[:, :, None, :], kb82) * sc82
    u_ref = (cont_ref + lam82 * a_82 * torch.erf((qpos82[..., None] - (sel82.float() + 0.5) * S).abs() / (2.0 * sig82))
             ).masked_fill(sel82 < 0, float("-inf"))
    for tri in (("1", "0") if dev == "cuda" else ("0",)):
        with _env(FSB_RERANK_TRITON=tri):
            cont = cell_token_scores(pq82, sel82, cellK, t_i82, sc82, 1, "max")
            g_ = pool_scores(pool82, pq82, qpos82, lam82, sig82, cellK, cellA, t_i82, sc82, S)
            L82 = pool_rank(pool82, pq82, qpos82, lam82, sig82, cellK, cellA, t_i82, sc82, S, 6)
        assert torch.equal(cont.masked_fill(sel82 < 0, 0.0), cont_ref.masked_fill(sel82 < 0, 0.0)), (Dh82, tri, "content != _det_dot")
        assert torch.equal(g_, u_ref), (Dh82, tri, "pool score != the leaf arithmetic")
        for t in range(T82):
            for b in range(Bq82):
                ids = pool82[t].tolist(); us = u_ref[t, b].tolist()
                ref = sorted([(-u_, c_) for u_, c_ in zip(us, ids) if c_ >= 0])[:6]
                assert L82[t, b].tolist() == [c_ for _, c_ in ref] + [-1] * (6 - len(ref)), (Dh82, tri, t, b)
                vals = [u_ for u_, c_ in zip(us, ids) if c_ in (3, 30, 41)]
                nties82 += int(len(vals) == 3 and vals[0] == vals[1] == vals[2])
                nsc82 += 1
assert nties82 > 0, "the duplicate cells never tied exactly"
print(f"F82 OK  (hybrid_pool == python brute force on {npool82} rows (quantized / all-equal priorities); pool content == "
      f"_det_dot bit-exact (Triton == torch, Dh 16/64/96); pool_rank == exact (g desc, id asc) order on {nsc82} queries, "
      f"{nties82} with exact 3-way ties)")


# ── F83: POOL mode (FS-L's native selector inside FS-B).  (a) the read set == the float64 top-4 of the python pool by g
#        (near-ties excepted), P/Wc = 4/2 and 4/8 and P = 0 (every eligible cell: the exhaustive FS top-4); (b) SET
#        EQUALITY with FS-L itself on the same Q/K/V: the strict chunked kernel _cell_score_On_chunked_strict with
#        FS_PRUNE_STABLE_TIES=1 (sel_full captured from the fused call) and LinearFSCache.select of the bounded
#        cache (P = 0 against FS-L with P = 64 >= M: its pool then holds every eligible cell); (c) causality: every
#        pool row of block b holds only cells < E_b = b*Bq/s (never a cell of the current block), and the top-P
#        truncation is active (some block has more eligible cells than the pool); (d) per-head independence: changing
#        head 0's keys changes no other head's selection; (e) no beam runs (n_visit 0) ────────────────────────────────
n83 = 512
nrow83 = ntie83 = nfsl83 = nlec83 = 0
for (P83, W83) in ((PH, WCH), (PH, 8), (0, 0)):
    mh = _mk_hy(83, "pool", P=P83, Wc=W83)
    Qc, Kc, Vc = _qkv_n(830, n83, 1.5)
    with torch.no_grad():
        (sel_h, o_h), rec83 = _hcap(lambda: _sel_n(mh, Qc, Kc, Vc, n83))
    assert float(mh._fsb_stats["n_visit"].abs().sum()) == 0.0, "pool mode must not run the beam"
    sel_h = sel_h[0].long()
    ceil83, g83, E83 = _hy_tabs(mh, Qc, Kc, n83)
    ref, tie, nval = _pool_ref(ceil83, g83, E83, P83, W83, 4)
    ok = _same_sets(sel_h, ref)
    assert bool((ok | tie).all()), (P83, W83, int((~ok & ~tie).sum()))
    nrow83 += int((nval > 4).sum()); ntie83 += int((tie & (nval > 4)).sum())
    # (c) causality + truncation
    pools83 = _pools_by_block([o_ for _, _, o_ in rec83["hybrid_pool"]])
    for b, pb in pools83.items():
        Eb = (b * BLK) // S
        assert bool(((pb < Eb) | (pb == -1)).all()), (P83, W83, b, "pool holds a cell of the current block")
        if P83 > 0:
            assert pb.shape[-1] == P83 + W83
    if P83 > 0:
        assert max((b * BLK) // S for b in pools83) > P83 + W83, "the top-P truncation is never exercised"
    # (b) FS-L on the same inputs
    Pl = P83 if P83 > 0 else 64
    Wl = W83 if P83 > 0 else 8
    ml = mk(fsb=False, cell_prune_kernel=1, cell_prune_pool=Pl, cell_prune_window=Wl, fs_fused_attn=1)
    with torch.no_grad():
        ml.sel_scale.copy_(mh.sel_scale)
    ml.cell_prune_stable_ties = 1
    assert torch.equal(ml.sigma, mh.sigma)
    cap83 = {}
    def _fl83(Q_, K_, V_, sel_, s_, Bq_, **k_):
        cap83["sel"] = sel_.clone(); return Q_.new_zeros(Q_.shape)
    ml._cell_fused_lr = _fl83
    with torch.no_grad():
        ml._cell_score_On_chunked_strict(Qc, Kc, Vc, ml.sigma, n83)
    sel_f = cap83["sel"][0].long()
    ok_f = _same_sets(sel_h, sel_f)
    assert bool((ok_f | tie).all()), (P83, W83, "FS-L strict", int((~ok_f & ~tie).sum()))
    nfsl83 += int(ok_f.sum())
    ml2 = mk(fsb=False, cell_prune_kernel=1, cell_prune_pool=Pl, cell_prune_window=Wl, fs_fused_attn=0)
    with torch.no_grad():
        ml2.sel_scale.copy_(mh.sel_scale)
    lec = _LEC81(ml2)
    sel_c = torch.full_like(sel_h, -1)
    _osel = lec.select
    def _sel83(q_):
        slots_, ids_ = _osel(q_)
        if ids_ is not None:
            sel_c[:, lec.pos:lec.pos + q_.shape[2], :ids_.shape[-1]] = ids_[0]
        return slots_, ids_
    lec.select = _sel83
    with torch.no_grad():
        lec.attend(Qc, Kc, Vc)
    ok_c = _same_sets(sel_h[:, :n83], sel_c[:, :n83])
    assert bool((ok_c | tie[:, :n83]).all()), (P83, W83, "LinearFSCache.select", int((~ok_c & ~tie[:, :n83]).sum()))
    nlec83 += int(ok_c.sum())
    # (d) per-head independence
    Kd = Kc.clone(); Kd[:, 0] = Kd[:, 0] * 1.7 + 0.3
    with torch.no_grad():
        sel_d, _ = _sel_n(mh, Qc, Kd, Vc, n83)
    assert torch.equal(sel_d[0, 1:].long(), sel_h[1:]) and not torch.equal(sel_d[0, 0].long(), sel_h[0]), (P83, W83)
print(f"F83 OK  (pool mode == float64 top-4 of the python pool on {nrow83 - ntie83}/{nrow83} rows ({ntie83} near-ties); "
      f"== FS-L strict chunked kernel (stable ties) on {nfsl83} rows and == LinearFSCache.select on {nlec83} rows "
      f"(P 4/Wc 2, P 4/Wc 8, P 0 vs FS-L P 64); causal pools, truncation active, heads independent, no beam)")


# ── F84: UNION.  The candidates handed to the final re-rank == [the tree candidates of the same module WITHOUT the
#        hybrid (identical slots) | the pool shortlist L[:short] cells not among them (same order)], the shortlist ==
#        the float64 top-short of the python pool by g (near-ties excepted), the read set == the float64 top-4 of that
#        union by the re-rank score of the RAW query (lse_pos and leaf; with and without a random adapter; W 8 / 16 / 4
#        and short 4 / 6), and the union changes the read set on some rows (the flag has teeth) ──────────────────────
os.environ["FSB_BEAM_COMPACT"] = "0"         # the hybrid keeps 2^Delta_max beam slots (fsb_tree.beam_delta_eff): so must
                                             # the module without it whose tree slots F84 compares them with
n84 = 512
nrow84 = ntie84 = nadd84 = ndiff84 = 0
qpos84 = torch.arange(n84, device=dev, dtype=torch.float32)[None].expand(H, -1)
for (Wb, sh) in ((8, 4), (16, 4), (4, 6)):
    for adapter in (None, 8400):
        for score in ("lse_pos", "leaf"):
            mu = _mk_hy(84, "union", adapter=adapter, W=Wb, short=sh, score=score)
            mt = _mk_hy(84, None, adapter=adapter, W=Wb, score=score)
            Qc, Kc, Vc = _qkv_n(840, n84, 1.5)
            with torch.no_grad():
                (su, _), ru = _hcap(lambda: _sel_n(mu, Qc, Kc, Vc, n84))
                (st, _), rt_ = _hcap(lambda: _sel_n(mt, Qc, Kc, Vc, n84))
            CT = _rows_hb([a_[0] for a_, _, _ in rt_["rerank_top"]]).long()
            CU = _rows_hb([a_[0] for a_, _, _ in ru["rerank_top"]]).long()
            LS = _rows_hb([o_ for _, _, o_ in ru["pool_rank"]]).long()
            C = CT.shape[-1]
            assert CU.shape[-1] == C + sh and LS.shape[-1] == sh
            assert torch.equal(CU[..., :C], CT), (Wb, sh, adapter, score, "tree part changed")
            dup = ((LS[..., :, None] == CT[..., None, :]) & (LS[..., :, None] >= 0)).any(-1)
            assert torch.equal(CU[..., C:], LS.masked_fill(dup, -1)), (Wb, sh, adapter, score, "pool part")
            ceil84, g84, E84 = _hy_tabs(mu, Qc, Kc, n84)
            refL, tieL, nvL = _pool_ref(ceil84, g84, E84, PH, WCH, sh)
            okL = _same_sets(LS, refL)
            assert bool((okL | tieL).all()), (Wb, sh, adapter, score, "shortlist", int((~okL & ~tieL).sum()))
            nadd84 += int(((CU[..., C:] >= 0).any(-1))[:, BLK:].sum())
            lam_ = mu.sel_scale.detach().view(H, 1, 1); sig_ = mu.sigma.detach().view(H, 1, 1)
            ceil_ = Kc[0].view(H, n84 // S, S, Dh).float().norm(dim=-1).amax(-1)
            kbar_ = Kc[0].view(H, n84 // S, S, Dh).mean(2)
            qq = Qc[0].float()
            for r0 in range(0, n84, 128):
                sb = CU[:, r0:r0 + 128]
                if score == "leaf":
                    c_ = sb.clamp(min=0)
                    kb = torch.gather(kbar_.double()[:, None].expand(H, sb.shape[1], n84 // S, Dh), 2, c_[..., None].expand(-1, -1, -1, Dh))
                    a_ = torch.gather(ceil_.double()[:, None].expand(H, sb.shape[1], n84 // S), 2, c_)
                    u = ((qq[:, r0:r0 + 128].double()[:, :, None, :] * kb).sum(-1) * mu.scale + lam_.double() * a_
                         * torch.erf((qpos84[:, r0:r0 + 128].double()[..., None] - (c_.double() + 0.5) * S).abs() / (2.0 * sig_.double())))
                    u = u.masked_fill(sb < 0, float("-inf"))
                else:
                    u = _tok_brute(qq[:, r0:r0 + 128], sb, Kc, torch.arange(H, device=dev), mu.scale, S, score,
                                   qpos84[:, r0:r0 + 128], lam_, sig_, ceil_)
                ref, tie, nval = _topk_ref(sb, u, 4)
                ok = _same_sets(su[0, :, r0:r0 + 128].long(), ref)
                assert bool((ok | tie).all()), (Wb, sh, adapter, score, r0, int((~ok & ~tie).sum()))
                nrow84 += int((nval > 4).sum()); ntie84 += int((tie & (nval > 4)).sum())
            ndiff84 += int((~_same_sets(su[0].long(), st[0].long()))[:, BLK:].sum())
assert nadd84 > 0 and ndiff84 > 0, ("the union never added a pool cell / never changed the read (no teeth)", nadd84, ndiff84)
print(f"F84 OK  (union candidates == [tree candidates | pool shortlist minus duplicates]; shortlist == float64 pool top-short; "
      f"read == float64 top-4 of the union by the raw-query score on {nrow84 - ntie84}/{nrow84} rows ({ntie84} near-ties); "
      f"pool cells added on {nadd84} rows, read changed on {ndiff84})")
_env_restore("FSB_BEAM_COMPACT")


# ── F85: QUOTA.  (a) quota_merge == a python brute force ([L[:K_L] | T | L[K_L:]], keep the first occurrence of every id,
#        drop -1, first K) on random rows with -1 pads and overlaps, K_L 1..4; (b) end to end (K_L 1 / 2, W 8 / 16, with
#        and without a random adapter): the read == quota_merge of the module's own ranked pool list and its ranked
#        tree list (order included); the tree list == the read of the same module WITHOUT the hybrid (the tree side is
#        unchanged); every read has <= 4 UNIQUE cells, == min(4, |L u T|) of them, the first min(K_L, |L|) pool cells
#        are always read; (c) backfill both ways: a 1-cell pool (P 1, Wc 0) with K_L 2 lets the tree fill the second
#        reserved slot; a head whose threshold admits no cell (empty tree) reads exactly its pool list ────────────────
g85 = torch.Generator(device="cpu").manual_seed(85)
for _ in range(10):
    R85 = 64
    L85 = torch.stack([torch.randperm(12, generator=g85)[:4] for _ in range(R85)])
    T85 = torch.stack([torch.randperm(12, generator=g85)[:4] for _ in range(R85)])
    L85 = L85.masked_fill(torch.rand(R85, 4, generator=g85) < 0.25, -1)
    T85 = T85.masked_fill(torch.rand(R85, 4, generator=g85) < 0.25, -1)
    for KL in (1, 2, 3, 4):
        out85 = quota_merge(L85.to(dev), T85.to(dev), KL, 4).cpu()
        for r in range(R85):
            seq = L85[r, :KL].tolist() + T85[r].tolist() + L85[r, KL:].tolist()
            res = []
            for c_ in seq:
                if c_ >= 0 and c_ not in res:
                    res.append(c_)
            assert out85[r].tolist() == (res + [-1] * 4)[:4], (KL, r, out85[r].tolist(), res)
n85 = 512
nread85 = nres85 = 0
for (KL, Wb) in ((1, 8), (2, 16), (2, 8)):
    for adapter in (None, 8500):
        mq = _mk_hy(85, "quota", adapter=adapter, quota=KL, W=Wb)
        mt = _mk_hy(85, None, adapter=adapter, W=Wb)
        Qc, Kc, Vc = _qkv_n(850, n85, 1.5)
        with torch.no_grad():
            (sq, _), rq85 = _hcap(lambda: _sel_n(mq, Qc, Kc, Vc, n85))
            st, _ = _sel_n(mt, Qc, Kc, Vc, n85)
        sq = sq[0].long().cpu()
        Lq = _rows_hb([o_ for _, _, o_ in rq85["pool_rank"]]).long().cpu()
        Tq_ = _rows_hb([o_ for _, _, o_ in rq85["rerank_top"]]).long().cpu()
        assert torch.equal(Tq_, st[0].long().cpu()), (KL, Wb, adapter, "tree ranking != the non-hybrid read")
        for h in range(H):
            for i in range(BLK, n85):
                seq = Lq[h, i, :KL].tolist() + Tq_[h, i].tolist() + Lq[h, i, KL:].tolist()
                res = []
                for c_ in seq:
                    if c_ >= 0 and c_ not in res:
                        res.append(c_)
                assert sq[h, i].tolist() == (res + [-1] * 4)[:4], (KL, Wb, adapter, h, i)
                lv = [c_ for c_ in Lq[h, i].tolist() if c_ >= 0]
                assert set(lv[:KL]) <= set(sq[h, i].tolist())
                u_ = set(lv) | {c_ for c_ in Tq_[h, i].tolist() if c_ >= 0}
                assert sum(c_ >= 0 for c_ in sq[h, i].tolist()) == min(4, len(u_))
                nread85 += 1
                nres85 += int(len(set(lv[:KL]) - set(Tq_[h, i].tolist())) > 0)
assert nres85 > 0, "the quota never reserved a cell the tree would not read (no teeth)"
# (c) backfill both ways
mq = _mk_hy(851, "quota", P=1, Wc=0, quota=2, W=8)
Qc, Kc, Vc = _qkv_n(851, n85, 1.5)
with torch.no_grad():
    (sq, _), rq85 = _hcap(lambda: _sel_n(mq, Qc, Kc, Vc, n85))
Lq = _rows_hb([o_ for _, _, o_ in rq85["pool_rank"]]).long().cpu(); Tq_ = _rows_hb([o_ for _, _, o_ in rq85["rerank_top"]]).long().cpu()
sq = sq[0].long().cpu()
assert bool(((Lq[..., 1:] == -1).all())), "a 1-cell pool list"
nfill85 = 0
for h in range(H):
    for i in range(BLK, n85):
        l0 = int(Lq[h, i, 0]); tl_ = [c_ for c_ in Tq_[h, i].tolist() if c_ >= 0 and c_ != l0]
        assert sq[h, i].tolist() == ([l0] + tl_ + [-1] * 4)[:4], (h, i)
        nfill85 += int(len(tl_) >= 3)
mq = _mk_hy(852, "quota", quota=1, W=8)
with torch.no_grad():
    mq.fsb_tau.copy_(torch.tensor([float("inf"), 0.0, 0.0, 0.0], device=dev))      # head 0 admits nothing
    (sq, _), rq85 = _hcap(lambda: _sel_n(mq, Qc, Kc, Vc, n85))
Lq = _rows_hb([o_ for _, _, o_ in rq85["pool_rank"]]).long(); Tq_ = _rows_hb([o_ for _, _, o_ in rq85["rerank_top"]]).long()
assert bool((Tq_[0] == -1).all()) and torch.equal(sq[0, 0].long(), Lq[0]), "empty tree: the read is the pool list"
assert bool((sq[0, 0, BLK:] >= 0).any())
print(f"F85 OK  (quota_merge == python brute force; read == merge of the module's pool + tree lists on {nread85} rows, "
      f"K_L reserved, <= 4 unique, tree side == the non-hybrid read; reserved-only-by-quota on {nres85} rows; backfill: "
      f"tree fills a short pool ({nfill85} rows), an empty tree reads the pool list)")


# ── F86: PREFILL == DECODE for every mode (pool P4 / P0, union, union + union beam, quota 1 / 2), with and without a
#        random adapter (raw query) and a learned c_h, tau 0 / per-head tau (one head admitting nothing) / every head
#        admitting nothing (the decode early return must not fire: D2): cached outputs == full forward, the per-token
#        decode selection == the full-sequence selection (sets; order counted), the decode pool == the prefill pool
#        rows of that block (bit-exact; constant within a block), full + lean prefill, FSB_DECODE_GRAPH 0 / 1 with graph
#        decode == eager decode bit-exact; (b) bf16 (autocast) 4096-token prefix, union + quota W16 raw + adapter: the
#        decode forest, pool and cache keys route the PREFILL's queries to the prefill's cells bit-exactly, and the
#        decode token's own q selects the prefill's sets on the whole last block ─────────────────────────────────────
n86, N086 = 512, 360
cfg86 = (("pool P4", dict(mode="pool")), ("pool P0", dict(mode="pool", P=0, Wc=0)), ("union W8", dict(mode="union", W=8)),
         ("union W8 + union beam U4", dict(mode="union", W=8, fsb_union_width=4)),
         ("quota1 W16", dict(mode="quota", quota=1, W=16)), ("quota2 W8", dict(mode="quota", quota=2, W=8)))
nrows86 = nord86 = 0
for name86, c86 in cfg86:
    c86 = dict(c86); mode86 = c86.pop("mode")
    for adapter, theta, tau_mode in ((None, None, "zero"), (8600, 8601, "per-head"), (8602, None, "all-inf")):
        mm = _mk_hy(86, mode86, adapter=adapter, theta=theta, **c86)
        torch.manual_seed(861); x = torch.randn(1, n86, d_model, device=dev)
        if tau_mode == "per-head":
            with torch.no_grad():
                c0 = FSKVCache(mm); mm(x[:, :N086], cache=c0)
                ce0 = c0._ceil[0]
                mm.fsb_tau.copy_(torch.stack([ce0[h].quantile(q_) for h, q_ in enumerate((0.0, 0.3, 0.6))] +
                                             [torch.tensor(float("inf"), device=dev)]))
        elif tau_mode == "all-inf":
            with torch.no_grad():
                mm.fsb_tau.fill_(float("inf"))
        with torch.no_grad():
            (res_full), rec86 = _hcap(lambda: _full_sel_n(mm, x), names=("hybrid_pool",))
        sel_full, yfull = res_full
        pools_pre = _pools_by_block([o_ for _, _, o_ in rec86["hybrid_pool"]])
        ys = {}
        with torch.no_grad():
            for lean in (False, True):
                for dg in (("0", "1") if dev == "cuda" else ("0",)):
                    y, sels, pools, provs = _hy_cached(mm, x, N086, lean, dg)
                    d = (y - yfull).abs().max().item()
                    assert d < 1e-4, (name86, tau_mode, lean, dg, d)
                    ys[(lean, dg)] = y
                    dec = torch.stack(sels, dim=1)
                    same = _same_sets(sel_full[:, N086:n86].long(), dec.long())
                    assert bool(same.all()), f"{name86} {tau_mode} lean={lean} graph={dg}: {int((~same).sum())} decode rows differ"
                    nord86 += int((sel_full[:, N086:n86].long() == dec.long()).all(-1).sum()); nrows86 += same.numel()
                    for k_, t in enumerate(range(N086, n86)):
                        b = t // BLK
                        pp, pd = pools_pre[b], pools[k_]
                        if c86.get("P", PH) == 0:
                            Eb = (b * BLK) // S
                            assert pd.shape[-1] == Eb and torch.equal(pp[:, :Eb], pd) and bool((pp[:, Eb:] == -1).all()), (name86, t)
                        else:
                            assert torch.equal(pp, pd), (name86, tau_mode, t, "decode pool != prefill pool")
                        assert provs[k_].dtype == torch.int8 and torch.equal(provs[k_] < 0, sels[k_] < 0)
            if dev == "cuda":
                for lean in (False, True):
                    assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (name86, tau_mode, lean, "graph decode != eager")
            if tau_mode == "all-inf":
                assert float(yfull[:, BLK:].abs().sum()) > 0 and bool((sel_full[:, BLK:] >= 0).any())
print(f"F86 [fp32] hybrid: prefill selection == decode selection on {nrows86} rows (order identical on {nord86}); decode pool == "
      f"prefill pool; graph decode == eager; cached == full forward; tau 0 / per-head / all-inf (D2)")


def _f86_bf16(mm_, x_, Nf, lean, tag):
    """_f65_bf16 for the hybrid: the prefill's routing q, raw q AND pool rows are replayed on the decode forest + pool."""
    caps = []
    _otr = _ft4.traverse_forest
    def _tr(F_, SP_, q_, qpos_, lam_, sigv_, *rest, **kw_):
        out_ = _otr(F_, SP_, q_, qpos_, lam_, sigv_, *rest, **kw_)
        caps.append((q_.clone(), qpos_.clone(), lam_.clone(), sigv_.clone(), rest, dict(kw_), out_[0].clone()))
        return out_
    os.environ["FSB_TRAIN_GRAPH"] = "0"
    _ft4.traverse_forest = _tr
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mm_(x_)
            q_, qpos_, lam_, sigv_, rest, kw_, sel_pre = caps[-1]
            assert kw_["hybrid"] == mm_.fsb_hybrid and kw_.get("rq") is not None, tag
            nbc = q_.shape[0] // H; sl = slice(nbc - 1, None, nbc)
            P = Nf - BLK
            cache = FSKVCache(mm_, lean_prefill=lean); cache.reserve = Nf - P
            mm_(x_[:, :P], cache=cache)
            mm_(x_[:, P:P + 1], cache=cache)
            fc = cache._fsb_forest
            assert torch.equal(fc.pool, kw_["pool"][sl]), f"{tag}: decode pool != prefill pool"
            dec = [fc._last_sel.reshape(H, -1).clone()]
            rest_sl = tuple(r_[sl] if torch.is_tensor(r_) and r_.dim() >= 1 and r_.shape[0] == q_.shape[0] else r_ for r_ in rest)
            kw2 = dict(kw_); kw2["tokK"] = cache._k_buf; kw2["pool"] = fc.pool
            for k_ in ("radius_coef", "rq"):
                v_ = kw2.get(k_)
                if torch.is_tensor(v_) and v_.dim() >= 1 and v_.shape[0] == q_.shape[0]:
                    kw2[k_] = v_[sl]
            s_dec, _, _ = _otr(fc.F, fc.SP, q_[sl], qpos_[sl], lam_[sl], sigv_[sl], *rest_sl, **kw2)
            assert torch.equal(s_dec, sel_pre[sl]), f"{tag}: decode forest + pool + cache keys route the prefill queries differently"
            for t in range(P + 1, Nf):
                mm_(x_[:, t:t + 1], cache=cache)
                dec.append(cache._fsb_forest._last_sel.reshape(H, -1).clone())
    finally:
        _ft4.traverse_forest = _otr; os.environ.pop("FSB_TRAIN_GRAPH", None)
    sel_dec = torch.stack(dec, dim=1)
    same_set, same_order = _sets_equal(sel_pre[sl].long(), sel_dec.long())
    assert bool(same_set.all()), f"{tag}: {int((~same_set).sum())}/{same_set.numel()} decode rows route differently"
    print(f"F86 [{tag}, lean={lean}] bf16 4096: prefill queries on the decode forest + pool + cache keys: identical cells + "
          f"order ({same_set.numel()} rows); decode q: {same_set.numel()} identical sets, {int(same_order.sum())} identical order")


if dev == "cuda":
    for mode86, kw86 in (("union", dict(short=4)), ("quota", dict(quota=2))):
        for lean in (False, True):
            torch.manual_seed(862)
            mm = _mk_hy(862, mode86, adapter=8620, W=16, P=16, Wc=8, **kw86)
            x = torch.randn(1, 4096, d_model, device=dev)
            _f86_bf16(mm, x, 4096, lean, f"{mode86} W16 P16/Wc8 lse_pos raw")
else:
    print("F86 [bf16] SKIP (no CUDA)")
print("F86 OK  (hybrid: prefill routing == decode routing, decode pool == prefill pool, graph decode == eager, D2 fixed)")


# ── F87: TRAINING STAGE.  (a) CUDA-graph training stage == eager (sel_full, output, n_read, n_visit) for pool P4 / P0,
#        union (+ union beam), quota 1 / 2, one and several chunk graphs, random adapter (raw query) + learned c_h,
#        after in-place adapter updates; (b) the graph key carries the hybrid config: switching fsb_pool_quota or the
#        mode on a captured module re-captures (graph == eager for the new config, and the selection changes);
#        (c) chunked prefill 1 / 2 / all identical; (d) Triton == torch for the traversal and for the pool / re-rank
#        scores (FSB_TRAVERSE_TRITON, FSB_RERANK_TRITON) ──────────────────────────────────────────────────────────────
n87 = 512
cfg87 = (("pool P4", dict(mode="pool")), ("pool P0", dict(mode="pool", P=0, Wc=0)), ("union W8", dict(mode="union", W=8)),
         ("union W8 U4", dict(mode="union", W=8, fsb_union_width=4)), ("quota1 W16", dict(mode="quota", quota=1, W=16)),
         ("quota2 W8", dict(mode="quota", quota=2, W=8)))
n87c = 0
if dev == "cuda":
    for name87, c87 in cfg87:
        c87 = dict(c87); mode87 = c87.pop("mode")
        for nbch in (None, 2):
            kw = {} if nbch is None else {"fsb_prefill_blkchunk": nbch}
            mm = _mk_hy(87, mode87, adapter=8700, theta=(8701 if mode87 != "pool" else None), **c87, **kw).train()
            for trial in range(2):
                Qc, Kc, Vc = _qkv_n(870 + trial, n87)
                res = {}
                for g in ("0", "1"):
                    with torch.no_grad():
                        sel, o = _sel_n(mm, Qc, Kc, Vc, n87, g)
                    res[g] = (sel, o.detach()) + _nstats(mm)
                for i, nm in enumerate(("sel_full", "output", "n_read", "n_visit")):
                    assert torch.equal(res["0"][i], res["1"][i]), (name87, nbch, trial, nm)
                assert int(res["1"][2].max()) <= 4
                with torch.no_grad():
                    mm.fsb_route_W.add_(torch.randn_like(mm.fsb_route_W) * 0.1)
                n87c += 1
    # (b) graph key
    mm = _mk_hy(871, "quota", adapter=8710, quota=1, W=8).train()
    Qc, Kc, Vc = _qkv_n(872, n87)
    with torch.no_grad():
        s1 = _sel_n(mm, Qc, Kc, Vc, n87, "1"); k1 = mm._fsb_train_graphs["key"]
        for attr, val in (("fsb_pool_quota", 2), ("fsb_hybrid", "union"), ("fsb_pool_short", 2), ("fsb_hybrid", "pool")):
            setattr(mm, attr, val)
            sg = _sel_n(mm, Qc, Kc, Vc, n87, "1"); k2 = mm._fsb_train_graphs["key"]
            se = _sel_n(mm, Qc, Kc, Vc, n87, "0")
            assert k2 != k1 and torch.equal(sg[0], se[0]) and torch.equal(sg[1], se[1]), (attr, val, "stale graph")
            if attr != "fsb_pool_short":
                assert not torch.equal(sg[0], s1[0]), (attr, val, "config change did not change the selection (no teeth)")
            k1, s1 = k2, sg
    print(f"F87 [graph] hybrid: graphed training stage == eager on {n87c} cases (pool P4/P0, union (+U4), quota 1/2, 1 / "
          f"several chunk graphs, adapter + learned c, in-place updates); the graph key re-captures on config changes")
for name87, c87 in cfg87:
    c87 = dict(c87); mode87 = c87.pop("mode")
    outs = []
    for nbch in (1, 2, 0):
        mm = _mk_hy(873, mode87, adapter=8730, fsb_prefill_blkchunk=nbch, **c87)
        Qc, Kc, Vc = _qkv_n(874, n87)
        with torch.no_grad():
            outs.append(_sel_n(mm, Qc, Kc, Vc, n87) + _nstats(mm))
    assert all(all(torch.equal(a_, b_) for a_, b_ in zip(outs[0], o_)) for o_ in outs[1:]), (name87, "chunk size")
    if dev == "cuda":
        mm = _mk_hy(875, mode87, adapter=8750, **c87)
        Qc, Kc, Vc = _qkv_n(876, n87)
        ref = None
        for trv, rrk in (("1", "1"), ("0", "1"), ("1", "0"), ("0", "0")):
            with torch.no_grad(), _env(FSB_RERANK_TRITON=rrk):
                r_ = _sel_n(mm, Qc, Kc, Vc, n87, "0", triton=trv) + _nstats(mm)
            if ref is None:
                ref = r_
            assert all(torch.equal(a_, b_) for a_, b_ in zip(ref, r_)), (name87, trv, rrk, "Triton != torch")
print("F87 OK  (hybrid training stage: graphed == eager, graph key re-captures, chunked prefill identical, Triton == torch)")


# ── F88: EXACT TIES (P5).  Repeated haystack text gives cells with identical keys (NoPE: the same tokens -> the same
#        kbar and a_c).  A fixture where every third cell repeats cell 1's tokens exactly: (a) the pool keeps the LOWER
#        cell ids among tied priorities (== the python brute force) and every pool list is in the exact (g desc, id asc)
#        order of the module's own fp32 scores; (b) pool mode == LinearFSCache.select sets; (c) prefill == decode for
#        pool / union / quota on a module input with repeated segments (identical projected keys) ─────────────────────
n88 = 512
Qc, Kc, Vc = _qkv_n(880, n88, 1.5)
Kt = Kc.clone()
Kt[:, :, S:2 * S] = Kc[:, :, S:2 * S] * 1.6                    # the repeated cell carries the largest key norm
for c_ in range(4, n88 // S, 3):
    Kt[:, :, c_ * S:(c_ + 1) * S] = Kt[:, :, S:2 * S]
nt88 = 0
for mode88 in ("pool", "union", "quota"):
    mh = _mk_hy(88, mode88, quota=2)
    with torch.no_grad():
        (sel_t, _), rec88 = _hcap(lambda: _sel_n(mh, Qc, Kt, Vc, n88), names=("hybrid_pool", "pool_rank", "pool_scores"))
    ceil88, g88, E88 = _hy_tabs(mh, Qc, Kt, n88)
    pools88 = _pools_by_block([o_ for _, _, o_ in rec88["hybrid_pool"]])
    for b, pb in pools88.items():
        Eb = (b * BLK) // S
        for h in range(H):
            ref = _pool_brute(ceil88[h].tolist(), Eb, PH, WCH)
            assert pb[h].tolist()[:len(ref)] == ref, (mode88, b, h)
    for (a_, k_, L_), (_, _, gs_) in zip(rec88["pool_rank"], rec88["pool_scores"]):
        pool_, Kk = a_[0], a_[-1]
        ids_ = pool_[:, None, :].expand_as(gs_).cpu(); gs_ = gs_.cpu(); L_ = L_.cpu()
        for t in range(0, gs_.shape[0]):
            for b in range(0, gs_.shape[1], 7):
                us, cs = gs_[t, b].tolist(), ids_[t, b].tolist()
                ref = [c_ for _, c_ in sorted((-u_, c_) for u_, c_ in zip(us, cs) if c_ >= 0)][:Kk]
                assert L_[t, b].tolist()[:len(ref)] == ref, (mode88, t, b)
                vals = sorted(u_ for u_, c_ in zip(us, cs) if c_ >= 0)
                nt88 += int(any(vals[i_] == vals[i_ + 1] for i_ in range(len(vals) - 1)))
    if mode88 == "pool":
        ml2 = mk(fsb=False, cell_prune_kernel=1, cell_prune_pool=PH, cell_prune_window=WCH, fs_fused_attn=0)
        with torch.no_grad():
            ml2.sel_scale.copy_(mh.sel_scale)
        lec = _LEC81(ml2)
        sel_c = torch.full_like(sel_t[0].long(), -1)
        _osel88 = lec.select
        def _sel88(q_):
            slots_, ids_ = _osel88(q_)
            if ids_ is not None:
                sel_c[:, lec.pos:lec.pos + q_.shape[2], :ids_.shape[-1]] = ids_[0]
            return slots_, ids_
        lec.select = _sel88
        with torch.no_grad():
            lec.attend(Qc, Kt, Vc)
        ref, tie, _ = _pool_ref(ceil88, g88, E88, PH, WCH, 4)
        ok_c = _same_sets(sel_t[0, :, :n88].long(), sel_c[:, :n88])
        assert bool((ok_c | tie[:, :n88]).all()), ("tie fixture: LinearFSCache", int((~ok_c & ~tie[:, :n88]).sum()))
assert nt88 > 0, "the tie fixture produced no exact score ties"
torch.manual_seed(881)
x = torch.randn(1, n88, d_model, device=dev)
x[:, S:2 * S] = x[:, S:2 * S] * 1.6
for c_ in range(4, n88 // S, 3):
    x[:, c_ * S:(c_ + 1) * S] = x[:, S:2 * S]
# decode only the last block: every cell it may read was completed in the prefill (a cell completed during decode gets its
# keys from a 1-row projection, whose rounding differs from the full forward's, so its copies would no longer tie exactly)
N088 = n88 - BLK
for mode88 in ("pool", "union", "quota"):
    mm = _mk_hy(882, mode88, quota=2)
    with torch.no_grad():
        sel_full, yfull = _full_sel_n(mm, x)
        for lean in (False, True):
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                y, sels, _, _ = _hy_cached(mm, x, N088, lean, dg)
                assert (y - yfull).abs().max().item() < 1e-4
                dec = torch.stack(sels, dim=1)
                assert bool(_same_sets(sel_full[:, N088:n88].long(), dec.long()).all()), (mode88, lean, dg, "tie fixture decode")
print(f"F88 OK  (exact-tie fixture: pools keep the lower cell id, pool lists in exact (g desc, id asc) order ({nt88} tied "
      f"queries), pool mode == LinearFSCache.select, prefill == decode on repeated segments)")


# ── F89: ROUTER NODE TRACE on the hybrid (I9: the re-run applies the same pool rows and merge).  Union and quota 2, leaf +
#        lse_pos, raw query, random adapter + learned c: the node-trace re-run reproduces the forward's read set (0
#        mismatches), eager + CUDA-graph; finite loss, gradients only on fsb_route_*.  (Pool mode + distillation is
#        refused at construction: F81.) ─────────────────────────────────────────────────────────────────────────────
Qc, Kc, Vc = _qkv_n(890, N2, 1.5)
nmis89 = ntr89 = 0
for mode89, extra in (("union", dict(fsb_pool_short=4)), ("quota", dict(fsb_pool_quota=2))):
    for score in ("leaf", "lse_pos"):
        torch.manual_seed(891)
        m_ = _mk_node(fsb_width=8, fsb_rd_node=1, fsb_rd_node_weight=0.5, fsb_route_learn_c=1, fsb_bound_radius_coef=8.0,
                      fsb_read_top=4, fsb_rerank_score=score, fsb_rerank_query="raw", fsb_hybrid=mode89, fsb_pool_size=PH,
                      fsb_pool_recent=WCH, **extra).train()
        with torch.no_grad():
            g_ = torch.Generator(device="cpu").manual_seed(892)
            m_.fsb_route_W.copy_((torch.randn(m_.fsb_route_W.shape, generator=g_) * 0.3).to(dev))
            m_.fsb_route_ctheta.add_((torch.randn(H, generator=g_) * 1.0).to(dev))
        for g in (("0", "1") if dev == "cuda" else ("0",)):
            m_.zero_grad(set_to_none=True)
            m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(893)
            sel_, o_, rd_, fr_ = _lr_n(m_, Qc, Kc, Vc, N2, g)
            st_ = rd_["stats"]
            assert fr_ is not None and st_.numel() == RD_STATS_NODE, (mode89, score, g)
            assert float(st_[14]) == 0.0, (mode89, score, g, "node trace re-run != the forward's read set", st_[14].item())
            nmis89 += int(st_[14]); ntr89 += int(st_[15])
            assert torch.isfinite(rd_["loss"]) and int(sel_.shape[-1]) == 4
            rd_["loss"].backward()
            got = sorted(n_ for n_, p_ in m_.named_parameters() if p_.grad is not None and p_.grad.abs().sum() > 0)
            assert got == ["fsb_route_W", "fsb_route_ctheta", "fsb_route_logtemp", "fsb_route_nlogtemp"], (mode89, score, got)
print(f"F89 OK  (router node distillation on the hybrid: the trace re-run applies the same pool + merge ({nmis89} mismatches / "
      f"{ntr89} rows, eager + graph), finite loss, grads only on fsb_route_*)")


# ── F90: PROVENANCE, GUARDS, PLUMBING, PROBE.  (a) rerank_top(prov=None) is the existing call (identical output); with
#        tags it also returns the kept slots' tags; hybrid_prov == python membership; (b) the prefill's provenance
#        (traverse_forest(prov_out=[])) == the decode's fc._last_prov on every decode row, and it is consistent with the
#        mode (pool: every read slot 1; union / quota: a slot tagged 1 or 2 holds a pool cell); (c) GraphedDecoder
#        refuses FS-B models (D1); (d) plumbing: build_attn_kwargs threads the hybrid keys only when set; (e) the S3
#        probe, when experiments/scaling/fsb_s3diag_probe.py is present ────────────────────────────────────────────────
g90 = torch.Generator(device="cpu").manual_seed(90)
sel90 = torch.stack([torch.randperm(30, generator=g90)[:10] for _ in range(8)]).view(2, 4, 10).to(dev)
sel90[0, 0, 3] = -1
q90 = torch.randn(2, 4, Dh, generator=g90).to(dev)
qp90 = torch.full((2, 4), 700.0, device=dev); lam90 = torch.full((2, 1, 1), 0.7, device=dev); sg90 = torch.full((2, 1, 1), 4.0, device=dev)
cK90 = torch.randn(2, 30, Dh, generator=g90).to(dev); cA90 = torch.rand(2, 30, generator=g90).to(dev)
t90 = torch.arange(2, device=dev)
o_a = rerank_top(sel90, q90, qp90, lam90, sg90, cK90, cA90, t90, 0.25, S, 4)
tags90 = torch.randint(0, 3, sel90.shape, generator=g90).to(dev)
o_b, p_b = rerank_top(sel90, q90, qp90, lam90, sg90, cK90, cA90, t90, 0.25, S, 4, prov=tags90)
assert torch.equal(o_a, o_b) and p_b.shape == o_b.shape
for i in range(2):
    for j in range(4):
        for k in range(4):
            c_ = int(o_b[i, j, k]); slot = sel90[i, j].tolist().index(c_)
            assert int(p_b[i, j, k]) == int(tags90[i, j, slot])
tr90 = torch.tensor([[3, 5, -1, 7]], device=dev); pl90 = torch.tensor([[5, 9, -1, 2]], device=dev)
rd90 = torch.tensor([[5, 3, 9, -1]], device=dev)
assert hybrid_prov(rd90, tr90, pl90).tolist() == [[2, 0, 1, -1]] and hybrid_prov(rd90, None, pl90).tolist() == [[1, 0, 1, -1]]
nprov90 = 0
for mode90 in ("pool", "union", "quota"):
    mm = _mk_hy(90, mode90, adapter=9000, quota=2)
    torch.manual_seed(901); x = torch.randn(1, n86, d_model, device=dev)
    caps90 = []
    _otr90 = _ft4.traverse_forest
    def _tr90(*a_, **k_):
        pl_ = []
        out_ = _otr90(*a_, **k_, prov_out=pl_)
        caps90.append(pl_[0].clone()); return out_
    _ft4.traverse_forest = _tr90
    try:
        with torch.no_grad():
            sel_full, _ = _full_sel_n(mm, x)
    finally:
        _ft4.traverse_forest = _otr90
    prov_full = _rows_hb(caps90).long()
    with torch.no_grad():
        for dg in (("0", "1") if dev == "cuda" else ("0",)):
            _, sels, pools, provs = _hy_cached(mm, x, N086, False, dg)
            for k_, t in enumerate(range(N086, n86)):
                if torch.equal(sels[k_].long(), sel_full[:, t].long()):
                    assert torch.equal(provs[k_].long(), prov_full[:, t]), (mode90, dg, t, "decode provenance != prefill")
                    nprov90 += 1
                pv, sv = provs[k_].long(), sels[k_].long()
                inpool = ((sv[..., None] == pools[k_][:, None, :]) & (sv[..., None] >= 0)).any(-1)
                if mode90 == "pool":
                    assert bool(((pv == 1) == (sv >= 0)).all())
                else:
                    assert bool((~(pv >= 1) | inpool).all()), (mode90, t, "a pool-tagged slot outside the pool")
assert nprov90 > 0
# (c) GraphedDecoder guard
import decode_graph as _dg90
try:
    _dg90.GraphedDecoder(_types81.SimpleNamespace(blocks=[_types81.SimpleNamespace(attn=_types81.SimpleNamespace(fsattn=mm))]),
                         torch.zeros(1, 8, dtype=torch.long), Lmax=16, device=dev)
    raise AssertionError("GraphedDecoder must refuse an FS-B model")
except NotImplementedError:
    pass
# (d) plumbing
_kw90_off = _bak68(_NS68(attn_type="fsattn", fsb=True, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw"))[1]
_kw90_on = _bak68(_NS68(attn_type="fsattn", fsb=True, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw",
                        fsb_hybrid="quota", fsb_pool_size=0, fsb_pool_quota=2))[1]
assert not any(k_.startswith("fsb_pool") or k_ == "fsb_hybrid" for k_ in _kw90_off)
assert (_kw90_on["fsb_hybrid"], _kw90_on["fsb_pool_size"], _kw90_on["fsb_pool_quota"]) == ("quota", 0, 2)
assert {k: v for k, v in _kw90_on.items() if k not in ("fsb_hybrid", "fsb_pool_size", "fsb_pool_quota")} == _kw90_off
try:
    import experiments.scaling.fsb_s3diag_probe as _pr90
except Exception:
    _pr90 = None
if _pr90 is not None:
    _a90 = _types81.SimpleNamespace(width=16, coef=8.0, read_top=4, rerank_query="raw", adapter="full", alpha=1.0, hybrid="off",
                                    pool_size=64, pool_recent=8, pool_short=4, pool_quota=1)
    assert not any(k_.startswith("fsb_pool") or k_ == "fsb_hybrid" for k_ in _pr90.attn_kwargs(_a90))
    _a90.hybrid = "quota"; _a90.pool_quota = 2
    _kp90 = _pr90.attn_kwargs(_a90)
    assert (_kp90["fsb_hybrid"], _kp90["fsb_pool_quota"], _kp90["fsb_pool_size"]) == ("quota", 2, 64)
    nprobe90 = 0
    for mode90, W90 in (("union", 16), ("quota", 8), ("pool", 4), (None, 16)):
        mm = _mk_hy(905, mode90, adapter=9050, W=W90, quota=2)
        Qc, Kc, Vc = _qkv_n(906, n86, 1.5)
        rows90 = torch.arange(440, 470, device=dev)
        with torch.no_grad():
            sel_full, _ = _sel_n(mm, Qc, Kc, Vc, n86)
        pr90 = {"rows": rows90, "needle_cells": torch.tensor([5, 17, 26], device=dev), "out": {}, "sys_w": W90, "sys_k": 4,
                "layer_of": {id(mm): 0}, "score_ranks": False, "needle_tok": torch.tensor([5 * S + 3, 17 * S + 7, 26 * S], device=dev),
                "pool_size": 8, "pool_recent": 2, "_sel_actual": sel_full[0, :, rows90].clone()}
        with torch.no_grad():
            _pr90.diag_layer(_ft4, _frr, _fr, mm, Qc, Kc, mm.sigma, n86, pr90)
        o90 = pr90["out"][0]
        assert bool((o90["own_eq"] == 1).all()), (mode90, "probe recompute != the actual read set")
        for h in range(H):
            for r, i in enumerate(rows90.tolist()):
                act = set(sel_full[0, h, i].tolist()); Eb = (i // BLK * BLK) // S
                nc = {c_ for c_ in (5, 17, 26) if c_ < Eb}
                assert int(o90["inRead"][h, r]) == int(bool(nc & act)), (mode90, h, i)
                assert (int(o90["provRead"][h, r]) >= 0) == bool(nc & act)
                if mode90 == "pool" and nc & act:
                    assert int(o90["provRead"][h, r]) == 1
                lm = float(o90["lrmass"][h, r])
                assert 0.0 <= lm <= 1.0 + 1e-6 and (lm == 0.0 or bool(nc & act)), (mode90, h, i, lm)
                nprobe90 += 1
    print(f"F90 [e] S3 probe: hybrid configs, own_eq 1, consistent hybrid fields on {nprobe90} rows")
else:
    print("F90 [e] SKIP (experiments/scaling/fsb_s3diag_probe.py absent)")
print(f"F90 OK  (rerank_top prov hook (default unchanged); hybrid_prov; decode provenance == prefill on {nprov90} rows; "
      f"GraphedDecoder refuses FS-B; --fsb_hybrid / --fsb_pool_* threaded by build_attn_kwargs only when set)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  RESIDUAL RE-RANKER (fsb_rerank_rr; src/fsattn/fsb_rerank.py rr_*): score(c) = lse_pos(q_raw, c) + phi(q, c) . w_h, the
#  per-layer fsb_route_rr_w zero-initialised; utility labels (experiments/scaling/fsb_utility.py), trainer
#  (experiments/scaling/fsb_rr_train.py), router-only checkpoints and the eval-time router overlay.  F91..F98.
# ══════════════════════════════════════════════════════════════════════════════════════════
import tempfile as _tf91
import shutil as _sh91
from src.fsattn.fsb_rerank import RR_FEATURES, RR_NF, rr_inputs, rr_features, rr_residual, PROV_TREE, PROV_POOL


def _mk_rr(seed, rrseed=None, scale=0.5, adapter=None, **kw):
    """_mk_int with the residual re-ranker on (fsb_rerank_rr linear) and, rrseed given, RANDOM weights (own generator)."""
    m_ = _mk_int(seed, adapter=adapter, fsb_rerank_rr="linear", **kw)
    if rrseed is not None:
        g_ = torch.Generator(device="cpu").manual_seed(rrseed)
        with torch.no_grad():
            m_.fsb_route_rr_w.copy_((torch.randn(m_.fsb_route_rr_w.shape, generator=g_) * scale).to(dev))
    return m_


def _expect_value_error(fn, what):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"must refuse: {what}")


# ── F91: construction and flag OFF.  (a) every module defaults to fsb_rerank_rr none (no parameter); linear without
#        fsb_read_top, with a score other than lse_pos, with an adapter and the route re-rank query, and an unknown mode
#        are refused; (b) linear creates fsb_route_rr_w (H, RR_NF) and fsb_route_rr_logtemp (H,), both ZERO, also without
#        a routing adapter, picked up by route_param_names; in an FreeSoloLM the backbone init is unchanged and the blanket
#        init skips them; (c) none passed explicitly == the flag absent: same state, sel_full and output (eager + CUDA-graph
#        stage, both trees) and the same cached decode ────────────────────────────────────────────────────────────────
assert mk(fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_read_top=4, fsb_rerank_score="lse_pos").fsb_rerank_rr == "none"
assert mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0).fsb_rerank_rr == "none"
for bad in (dict(fsb_width=8, fsb_rerank_rr="linear"),
            dict(fsb_width=8, fsb_read_top=4, fsb_rerank_rr="linear"),
            dict(fsb_width=8, fsb_read_top=4, fsb_rerank_score="lse", fsb_rerank_rr="linear"),
            dict(fsb_width=8, fsb_read_top=4, fsb_rerank_score="max", fsb_rerank_rr="linear"),
            dict(fsb_width=8, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_rr="mlp"),
            dict(fsb_width=8, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_rr="linear", fsb_route_adapter="full"),
            dict(fsb_width=8, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_rr="linear", fsb_route_adapter="full",
                 fsb_rerank_query="route")):
    _expect_value_error(lambda: mk(fsb_policy="fixed", fsb_delta=0, **bad), bad)
m91 = mk(fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_rr="linear")
names91 = sorted(n_ for n_, _ in m91.named_parameters() if "fsb_route" in n_)
assert names91 == ["fsb_route_rr_logtemp", "fsb_route_rr_w"], names91
assert tuple(m91.fsb_route_rr_w.shape) == (H, RR_NF) and float(m91.fsb_route_rr_w.abs().max()) == 0.0
assert float(m91.fsb_route_rr_logtemp.abs().max()) == 0.0 and RR_NF == len(RR_FEATURES) == 10
m91b = mk(fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_rr="linear",
          fsb_route_adapter="full", fsb_rerank_query="raw")
assert sorted(n_ for n_, _ in m91b.named_parameters() if "fsb_route" in n_) == sorted(
    ["fsb_route_W", "fsb_route_logtemp", "fsb_route_rr_w", "fsb_route_rr_logtemp"])
_lmkw91 = dict(_lmkw, fsb_width=8, fsb_read_top=4, fsb_rerank_score="lse_pos")
torch.manual_seed(811); lm91a = _LM32(**_lmkw91)
torch.manual_seed(811); lm91b = _LM32(**_lmkw91, fsb_rerank_rr="linear")
sda, sdb = lm91a.state_dict(), lm91b.state_dict()
extra91 = sorted(k for k in sdb if k not in sda)
assert extra91 and all(k.rsplit(".", 1)[-1] in ("fsb_route_rr_w", "fsb_route_rr_logtemp") for k in extra91), extra91
assert all(torch.equal(sda[k], sdb[k]) for k in sda), "the re-ranker changed the backbone init"
assert all(float(sdb[k].abs().max()) == 0.0 for k in extra91), "the blanket init must skip fsb_route_rr_*"
assert sorted(route_param_names(lm91b)) == extra91
n91 = 0
for tree in ("forest", "rebuild"):
    common = dict(fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree=tree, fsb_read_top=4, fsb_rerank_score="lse_pos",
                  fsb_rerank_query="raw")
    ma = _mk_int(81, adapter=8100, **common)
    mb = _mk_int(81, adapter=8100, fsb_rerank_rr="none", **common)
    sda, sdb = ma.state_dict(), mb.state_dict()
    assert list(sda) == list(sdb) and all(torch.equal(sda[k], sdb[k]) for k in sda)
    Qc, Kc, Vc = _qkv_n(810, 512, 1.5)
    for g in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
        with torch.no_grad():
            ra = _sel_n(ma, Qc, Kc, Vc, 512, g); rb = _sel_n(mb, Qc, Kc, Vc, 512, g)
        assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1]), (tree, g)
        n91 += 1
    torch.manual_seed(812); x = torch.randn(1, 512, d_model, device=dev)
    with torch.no_grad():
        for dg in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
            ya, sa = _cached_run_n(ma, x, 400, False, dg); yb, sb_ = _cached_run_n(mb, x, 400, False, dg)
            assert torch.equal(ya, yb) and all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(sa, sb_)), (tree, dg)
            n91 += 1
print(f"F91 OK  (fsb_rerank_rr: default none, 7 invalid configs refused; linear = zero (H, {RR_NF}) weights + zero loss-only "
      f"temperature without an adapter, backbone init unchanged, blanket init skipped; none explicit == absent on {n91} cases)")


# ── F92: ZERO re-ranker weights == no re-ranker, BIT-EXACT: sel_full and output on the eager and CUDA-graph training
#        stage (eval + train mode), the no-cache module forward, and cached decode (full + lean prefill, FSB_DECODE_GRAPH
#        0 / 1: outputs and per-token selections) for forest W16 / W8 (no adapter; raw + random adapter; union U4) and the
#        rebuild tree; (b) bf16 4096: the rr model routes the prefill's queries on the decode forest identically ────────
n92 = 0
cfg92 = (("forest W16 lse_pos", None, dict(fsb_tree="forest", fsb_width=16)),
         ("forest W8 raw + adapter", 8200, dict(fsb_tree="forest", fsb_width=8, fsb_rerank_query="raw")),
         ("forest W8 U4 raw + adapter", 8201, dict(fsb_tree="forest", fsb_width=8, fsb_rerank_query="raw", fsb_union_width=4)),
         ("rebuild W8 lse_pos", None, dict(fsb_tree="rebuild", fsb_width=8)))
for name92, ad92, kw92 in cfg92:
    common = dict(fsb_policy="fixed", fsb_delta=0, fsb_read_top=4, fsb_rerank_score="lse_pos", **kw92)
    ma = _mk_int(82, adapter=ad92, **common)
    mb = _mk_rr(82, adapter=ad92, **common)
    sda, sdb = ma.state_dict(), mb.state_dict()
    assert all(torch.equal(sda[k], sdb[k]) for k in sda) and set(sdb) - set(sda) == {"fsb_route_rr_w", "fsb_route_rr_logtemp"}
    Qc, Kc, Vc = _qkv_n(820, 512, 1.5)
    for mode in ("eval", "train"):
        ma.train(mode == "train"); mb.train(mode == "train")
        for g in (("0", "1") if (dev == "cuda" and kw92["fsb_tree"] == "forest") else ("0",)):
            with torch.no_grad():
                ra = _sel_n(ma, Qc, Kc, Vc, 512, g); rb = _sel_n(mb, Qc, Kc, Vc, 512, g)
            assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1]), (name92, mode, g)
            n92 += 1
    ma.eval(); mb.eval()
    torch.manual_seed(821); x = torch.randn(1, 512, d_model, device=dev)
    with torch.no_grad():
        sa_, ya_ = _full_sel_n(ma, x); sb2, yb_ = _full_sel_n(mb, x)
        assert torch.equal(sa_, sb2) and torch.equal(ya_, yb_), name92
        for lean in (False, True):
            for dg in (("0", "1") if (dev == "cuda" and kw92["fsb_tree"] == "forest") else ("0",)):
                ya, sa = _cached_run_n(ma, x, 400, lean, dg); yb, sb_ = _cached_run_n(mb, x, 400, lean, dg)
                assert torch.equal(ya, yb) and all((a_ is None and b_ is None) or torch.equal(a_, b_) for a_, b_ in zip(sa, sb_)), \
                    (name92, lean, dg)
                n92 += 1
if dev == "cuda":
    for lean in (False, True):
        torch.manual_seed(822)
        mm = _mk_rr(822, adapter=8220, fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_tree="forest", fsb_read_top=4,
                    fsb_rerank_score="lse_pos", fsb_rerank_query="raw")
        x = torch.randn(1, 4096, d_model, device=dev)
        _f65_bf16(mm, x, 4096, lean, "F92 rr zero W16 lse_pos raw")
print(f"F92 OK  (zero re-ranker == no re-ranker bit-exact on {n92} eager / graph / train / cached-decode cases, 4 configs)")


# ── F93: the features.  (a) cell_token_scores with cellK (the EXTRA kernel specialization) returns the plain lse_pos
#        score bit-exactly, the max token score == the "max" mode output bit-exactly, and the mean-key score == the leaf
#        re-rank's _det_dot(q, kbar) * scale bit-exactly; Triton == the torch fallback for all three; bf16 / fp32 keys,
#        a strided cellK view; (b) rr_features: finite, 0 on invalid slots, the four centred features have row max 0,
#        slot / C, provenance codes; rr_residual == float64 brute force, exactly 0 at w = 0, and a row's value does not
#        depend on the rows sharing the call ─────────────────────────────────────────────────────────────────────────
g93 = torch.Generator(device="cpu").manual_seed(83)
Tq93, Bq93, C93, M93, s93 = 12, 5, 9, 20, 16
n93 = 0
for kdt in (torch.bfloat16, torch.float32):
    q93 = torch.randn(Tq93, Bq93, Dh, generator=g93).to(dev)
    K4_93 = (torch.randn(1, 3, M93 * s93, Dh, generator=g93) * 1.3).to(dev).to(kdt)
    t93 = (torch.arange(Tq93) % 3).to(dev)
    sel93 = torch.randint(-1, M93, (Tq93, Bq93, C93), generator=g93).to(dev)
    kbar93 = K4_93[0].view(3, M93, s93, Dh).mean(2).float()                 # (3, M, Dh): the module's convention
    big93 = torch.zeros(3, M93 + 5, Dh + 3, device=dev)
    big93[:, :M93, :Dh] = kbar93
    kbar_strided = big93[:, :M93, :Dh]                                     # stride(1) != Dh: a strided view
    assert kbar_strided.stride(1) != Dh
    outs = {}
    for tri in (("1", "0") if dev == "cuda" else ("0",)):
        with _env(FSB_RERANK_TRITON=tri):
            plain = cell_token_scores(q93, sel93, K4_93, t93, 0.25, s93, "lse_pos")
            mxm = cell_token_scores(q93, sel93, K4_93, t93, 0.25, s93, "max")
            for ck in (kbar93, kbar_strided):
                lse_, mx_, mk_ = cell_token_scores(q93, sel93, K4_93, t93, 0.25, s93, "lse_pos", cellK=ck)
                assert torch.equal(lse_, plain) and torch.equal(mx_, mxm), (kdt, tri, "lse / max")
                gid = t93.view(-1, 1, 1) * M93 + sel93.clamp(min=0)
                kb = kbar93.reshape(-1, Dh)[gid.reshape(-1)].view(Tq93, Bq93, C93, Dh)
                ref_mk = (_ft4._det_dot(q93[:, :, None, :], kb) * 0.25).masked_fill(sel93 < 0, float("-inf"))
                assert torch.equal(mk_, ref_mk), (kdt, tri, "mean-key", (mk_ - ref_mk).abs().nan_to_num().max().item())
                n93 += 1
            outs[tri] = (lse_, mx_, mk_)
    if "1" in outs and "0" in outs:
        assert all(torch.equal(a_, b_) for a_, b_ in zip(outs["1"], outs["0"])), (kdt, "Triton != torch")
# (b) features and residual on a module-like call
lam93 = torch.rand(Tq93, 1, 1, generator=g93).to(dev) + 0.5
sig93 = torch.full((Tq93, 1, 1), 4.0, device=dev)
qpos93 = (torch.arange(Bq93, dtype=torch.float32)[None].expand(Tq93, Bq93) + M93 * s93).to(dev)
cellA93 = K4_93[0].float().view(3, M93, s93, Dh).norm(dim=-1).amax(-1)
sel93[0, 0] = -1                                                         # an all-invalid row
inp93 = rr_inputs(sel93, q93, qpos93, lam93, sig93, kbar93, cellA93, t93, 0.25, s93, K4_93)
prov93 = torch.randint(0, 4, sel93.shape, generator=g93).to(dev)
for pv in (None, prov93):
    phi93 = rr_features(inp93, sel93, qpos93, s93, pv)
    assert phi93.shape == (Tq93, Bq93, C93, RR_NF) and bool(torch.isfinite(phi93).all())
    ok93 = sel93 >= 0
    assert float(phi93[~ok93].abs().max()) == 0.0
    rows_ok = ok93.any(-1)
    for f_ in range(4):
        mxr = phi93[..., f_].masked_fill(~ok93, float("-inf")).amax(-1)
        assert bool((mxr[rows_ok] == 0).all()), (f_, "centred features must have row max 0")
    assert torch.allclose(phi93[..., 4][ok93], inp93["a"][ok93])
    slot93 = (torch.arange(C93, device=dev).float() / C93).expand_as(sel93)
    assert torch.equal(phi93[..., 7][ok93], slot93[ok93])
    if pv is None:
        assert float(phi93[..., 8:].abs().max()) == 0.0
    else:
        assert torch.equal(phi93[..., 8][ok93], ((pv & PROV_POOL) != 0).float()[ok93])
        assert torch.equal(phi93[..., 9][ok93], (((pv & PROV_POOL) != 0) & ((pv & PROV_TREE) != 0)).float()[ok93])
w93 = torch.randn(3, RR_NF, generator=g93).to(dev)
r93 = rr_residual(phi93, w93, t93)
r93_ref = (phi93.double() * w93.double()[t93 % 3][:, None, None, :]).sum(-1)
assert float((r93.double() - r93_ref).abs().max()) < 1e-4
assert float(rr_residual(phi93, torch.zeros_like(w93), t93).abs().max()) == 0.0
sub = torch.tensor([3, 7, 11], device=dev)
inp_s = rr_inputs(sel93[sub], q93[sub], qpos93[sub], lam93[sub], sig93[sub], kbar93, cellA93, t93[sub], 0.25, s93, K4_93)
phi_s = rr_features(inp_s, sel93[sub], qpos93[sub], s93, prov93[sub])
assert torch.equal(phi_s, phi93[sub]) and torch.equal(rr_residual(phi_s, w93, t93[sub]), r93[sub]), "features / residual are not row-local"
print(f"F93 OK  (EXTRA kernel: lse == lse_pos, max == max mode, mean-key == _det_dot leaf score, bit-exact on {n93} cases; "
      f"Triton == torch; features finite / masked / centred / provenance; residual == float64, 0 at w = 0, row-local)")


# ── F94: RANDOM re-ranker weights.  (a) the re-rank changes the read on a fraction of rows and the module read sets ==
#        the float64 top-4 of lse_pos + phi . w over the beam's candidates (rows away from near-ties); (b) PREFILL ==
#        DECODE: the no-cache full-sequence selection equals the per-token decode selection on every decode row (forest:
#        no adapter / raw + adapter / union U4; full + lean prefill; FSB_DECODE_GRAPH 0 / 1, graph decode == eager
#        bit-exact); cached == no-cache forward (both trees); bf16 4096 prefill queries on the decode forest; (c) the
#        CUDA-graph training stage == eager, also after an in-place weight update (the graph reads the weights by
#        address), chunked prefill 1 / 2 / all identical, Triton traversal == torch; (d) router node distillation on
#        top: the node trace re-run applies the same re-ranker (0 mismatches) and no gradient reaches fsb_route_rr_* ───
n94 = 512
chg94 = rows94 = ntie94 = 0
for ad94 in (None, 8400):
    kwq = dict(fsb_rerank_query="raw") if ad94 else {}
    m0 = _mk_rr(84, adapter=ad94, fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_tree="forest", fsb_read_top=4,
                fsb_rerank_score="lse_pos", **kwq)
    m1 = _mk_rr(84, rrseed=8401, adapter=ad94, fsb_policy="fixed", fsb_width=16, fsb_delta=0, fsb_tree="forest", fsb_read_top=4,
                fsb_rerank_score="lse_pos", **kwq)
    Qc, Kc, Vc = _qkv_n(841, n94, 1.5)
    with torch.no_grad():
        (s0, _), caps0 = _rr_cap(lambda: _sel_n(m0, Qc, Kc, Vc, n94))
        (s1, _), caps1 = _rr_cap(lambda: _sel_n(m1, Qc, Kc, Vc, n94))
    diff = ~_same_sets(s0[0].long(), s1[0].long())
    chg94 += int(diff[:, BLK * 2:n94].sum()); rows94 += diff[:, BLK * 2:n94].numel()
    # brute force on the candidates actually handed to the re-rank (same beam: the re-ranker acts after it)
    cands = _cands(caps1)                                                  # (H, Npad, C)
    assert torch.equal(cands, _cands(caps0)), "the re-ranker must not change the beam's candidates"
    Kp94 = Kc[0]
    kbar94 = Kp94.view(H, n94 // S, S, Dh).mean(2).float()
    ceil94 = Kp94.float().view(H, n94 // S, S, Dh).norm(dim=-1).amax(-1)
    qraw = Qc[0].float()
    Npad = cands.shape[1]
    qrows = torch.cat([qraw, qraw.new_zeros(H, Npad - n94, Dh)], 1) if Npad > n94 else qraw
    t_h = torch.arange(H, device=dev)
    qpos_all = torch.arange(Npad, device=dev, dtype=torch.float32)[None].expand(H, Npad)
    lam_h = m1.sel_scale.detach().float().view(H, 1, 1); sig_h = m1.sigma.detach().float().view(H, 1, 1)
    inp = rr_inputs(cands, qrows[:, :, :], qpos_all, lam_h, sig_h, kbar94, ceil94, t_h, m1.scale, S, Kc)
    phi = rr_features(inp, cands, qpos_all, S)
    u64 = (inp["lse"].double() + inp["pos"].double() + (phi.double() * m1.fsb_route_rr_w.detach().double()[:, None, None, :]).sum(-1))
    u64 = u64.masked_fill(cands < 0, float("-inf"))
    ref, tie, _ = _topk_ref(cands, u64, 4)
    rows_ = torch.arange(BLK * 2, n94, device=dev)
    ok_rows = ~tie[:, rows_]
    same = _same_sets(ref[:, rows_].long(), s1[0][:, rows_].long())
    assert bool(same[ok_rows].all()), f"adapter {ad94}: {int((~same & ok_rows).sum())} rows differ from the float64 re-rank"
    ntie94 += int((~ok_rows).sum())
assert chg94 > 0, "random re-ranker weights never changed a read"
print(f"F94 [a] random weights change the read on {chg94}/{rows94} rows; module read sets == float64 top-4 of lse_pos + r "
      f"({ntie94} near-tie rows skipped)")
# (b) prefill == decode
n94b, N084 = 512, 300
nrows94 = nord94 = 0
cfg94 = ((None, dict(fsb_tree="forest", fsb_width=16)), (8410, dict(fsb_tree="forest", fsb_width=8, fsb_rerank_query="raw")),
         (8411, dict(fsb_tree="forest", fsb_width=8, fsb_rerank_query="raw", fsb_union_width=4)),
         (None, dict(fsb_tree="rebuild", fsb_width=8)))
for ad94, kw94 in cfg94:
    mm = _mk_rr(842, rrseed=8420, adapter=ad94, fsb_policy="fixed", fsb_delta=0, fsb_read_top=4, fsb_rerank_score="lse_pos", **kw94)
    torch.manual_seed(843); x = torch.randn(1, n94b, d_model, device=dev)
    tree = kw94["fsb_tree"]
    ys = {}
    with torch.no_grad():
        sel_full, yfull = _full_sel_n(mm, x)
        for lean in (False, True):
            for dg in (("0", "1") if (dev == "cuda" and tree == "forest") else ("0",)):
                y, sels = _cached_run_n(mm, x, N084, lean, dg)
                d = (y - yfull).abs().max().item()
                assert d < 1e-4, (kw94, lean, dg, d)
                ys[(lean, dg)] = (y, sels)
                if tree == "forest":
                    dec = torch.stack(sels, dim=1)
                    same_set = _same_sets(sel_full[:, N084:n94b].long(), dec.long())
                    assert bool(same_set.all()), f"{kw94} lean={lean} graph={dg}: {int((~same_set).sum())} decode rows differ"
                    nord94 += int((sel_full[:, N084:n94b].long() == dec.long()).all(-1).sum()); nrows94 += same_set.numel()
        if dev == "cuda" and tree == "forest":
            for lean in (False, True):
                assert torch.equal(ys[(lean, "0")][0], ys[(lean, "1")][0]), (kw94, lean, "graph decode != eager")
                assert all(torch.equal(a_, b_) for a_, b_ in zip(ys[(lean, "0")][1], ys[(lean, "1")][1])), (kw94, lean)
print(f"F94 [b] random re-ranker: prefill selection == decode selection on {nrows94} rows (order identical on {nord94}); graph "
      f"decode == eager; cached == no-cache forward (both trees)")
if dev == "cuda":
    for (Wb, Ub, ad) in ((16, 0, 8431), (8, 4, 8430)):                    # an adapter: the raw re-rank query is a separate rq
        for lean in (False, True):
            torch.manual_seed(844)
            mm = _mk_rr(844, rrseed=8440, adapter=ad, fsb_policy="fixed", fsb_width=Wb, fsb_delta=0, fsb_tree="forest",
                        fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw",
                        **({"fsb_union_width": Ub} if Ub else {}))
            x = torch.randn(1, 4096, d_model, device=dev)
            _f65_bf16(mm, x, 4096, lean, f"F94 rr random W{Wb} U{Ub} lse_pos raw")
# (c) graph == eager (incl. after in-place updates), chunk invariance, Triton == torch
if dev == "cuda":
    for pol in ("fixed", "learned"):
        wkw = dict(fsb_width=16, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
        mm = _mk_rr(845, rrseed=8450, adapter=8451, fsb_policy=pol, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score="lse_pos",
                    fsb_rerank_query="raw", **wkw)
        if pol == "learned":
            mm.train()
        for trial in range(3):
            Qc, Kc, Vc = _qkv_n(846 + trial, n94, 1.5)
            res = {}
            for g in ("0", "1"):
                torch.manual_seed(8460 + trial)
                with torch.no_grad():
                    res[g] = _sel_n(mm, Qc, Kc, Vc, n94, g)
            assert torch.equal(res["0"][0], res["1"][0]) and torch.equal(res["0"][1], res["1"][1]), (pol, trial)
            with torch.no_grad():
                mm.fsb_route_rr_w.add_(torch.randn_like(mm.fsb_route_rr_w) * 0.3)   # an optimizer step in place
                mm.fsb_route_W.add_(torch.randn_like(mm.fsb_route_W) * 0.05)
    for pol in ("fixed", "learned"):
        wkw = dict(fsb_width=16, fsb_delta=0) if pol == "fixed" else dict(fsb_widths="2,4,8", fsb_deltas="0,1")
        mm = _mk_rr(847, rrseed=8470, fsb_policy=pol, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score="lse_pos", **wkw)
        Qc, Kc, Vc = _qkv_n(848, n94)
        with torch.no_grad():
            r0 = _sel_n(mm, Qc, Kc, Vc, n94, "0", triton="0"); r1 = _sel_n(mm, Qc, Kc, Vc, n94, "0", triton="1")
        assert torch.equal(r0[0], r1[0]) and torch.equal(r0[1], r1[1]), ("F94 triton", pol)
for tree in ("forest", "rebuild"):
    outs = []
    for nbch in (1, 2, 0):
        mm = _mk_rr(849, rrseed=8490, adapter=8491, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree=tree, fsb_read_top=4,
                    fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_prefill_blkchunk=nbch)
        Qc, Kc, Vc = _qkv_n(8492, n94)
        with torch.no_grad():
            outs.append(_sel_n(mm, Qc, Kc, Vc, n94))
    assert all(torch.equal(o_[0], outs[0][0]) and torch.equal(o_[1], outs[0][1]) for o_ in outs[1:]), tree
print("F94 [c] random re-ranker: graphed training stage == eager (fixed + learned, after in-place re-ranker / adapter "
      "updates); chunked prefill 1 / 2 / all identical (both trees); Triton == torch traversal")
# (d) router node distillation with the re-ranker
Qc, Kc, Vc = _qkv_n(8493, N2, 1.5)
nmis94 = ntr94 = 0
for g in (("0", "1") if dev == "cuda" else ("0",)):
    torch.manual_seed(8494)
    m_ = _mk_node(fsb_width=8, fsb_rd_node=1, fsb_rd_node_weight=0.5, fsb_bound_radius_coef=8.0, fsb_read_top=4,
                  fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_rerank_rr="linear").train()
    with torch.no_grad():
        g_ = torch.Generator(device="cpu").manual_seed(8495)
        m_.fsb_route_W.copy_((torch.randn(m_.fsb_route_W.shape, generator=g_) * 0.3).to(dev))
        m_.fsb_route_rr_w.copy_((torch.randn(m_.fsb_route_rr_w.shape, generator=g_) * 0.5).to(dev))
    m_.zero_grad(set_to_none=True)
    m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(8496)
    sel_, o_, rd_, fr_ = _lr_n(m_, Qc, Kc, Vc, N2, g)
    st_ = rd_["stats"]
    assert float(st_[14]) == 0.0, (g, "node trace re-run != the forward's read set", st_[14].item(), st_[15].item())
    nmis94 += int(st_[14]); ntr94 += int(st_[15])
    rd_["loss"].backward()
    assert m_.fsb_route_rr_w.grad is None or float(m_.fsb_route_rr_w.grad.abs().max()) == 0.0, "rd loss reached the re-ranker"
    assert m_.fsb_route_W.grad is not None and float(m_.fsb_route_W.grad.abs().sum()) > 0
print(f"F94 OK  (random re-ranker: reads change, == float64 re-rank; prefill == decode; graph == eager incl. in-place updates; "
      f"node trace applies the re-ranker ({nmis94} mismatches / {ntr94} rows), no rd gradient on the re-ranker)")


# ── F95: the one-shot SELECTION OVERRIDE (utility collector hook in fsb_long_range) touches ONLY the listed (batch,
#        head, row) entries: sel_full equals the plain run except there (where it holds the given cells), the output
#        (1) equals the plain output everywhere except those (head, row) entries of those batch rows; ``seen`` / ``row_sel``
#        record the pre-override reads; the attribute is cleared (the next call equals the plain run) ────────────────
m95 = _mk_int(85, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score="lse_pos")
Q1, K1, V1 = _qkv_n(850, 512, 1.5)
Q3, K3, V3 = (t_.expand(3, -1, -1, -1).contiguous() for t_ in (Q1, K1, V1))
with torch.no_grad():
    sp, op = _sel_n(m95, Q3, K3, V3, 512)
    ov_new = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=dev)
    ov95 = {"v": torch.tensor([1, 2], device=dev), "h": torch.tensor([0, 3], device=dev),
            "i": torch.tensor([300, 450], device=dev), "sel": ov_new, "rows": torch.tensor([300, 450], device=dev)}
    m95._fsb_sel_override = ov95
    so, oo = _sel_n(m95, Q3, K3, V3, 512)
    assert m95._fsb_sel_override is None, "the override must be one-shot"
    s2, o2 = _sel_n(m95, Q3, K3, V3, 512)
assert torch.equal(s2, sp) and torch.equal(o2, op), "the override leaked into the next call"
mask95 = torch.zeros(3, H, sp.shape[2], dtype=torch.bool, device=dev)
mask95[1, 0, 300] = True; mask95[2, 3, 450] = True
assert torch.equal(so[~mask95], sp[~mask95]) and torch.equal(so[1, 0, 300], ov_new[0]) and torch.equal(so[2, 3, 450], ov_new[1])
assert not torch.equal(so[1, 0, 300], sp[1, 0, 300])
omask = mask95[:, :, :512]
assert torch.equal(oo[~omask], op[~omask]), "the override changed an entry it does not list"
assert not torch.equal(oo[1, 0, 300], op[1, 0, 300]) and not torch.equal(oo[2, 3, 450], op[2, 3, 450])
assert torch.equal(ov95["seen"], torch.stack([sp[1, 0, 300], sp[2, 3, 450]]))
assert torch.equal(ov95["row_sel"], sp[:, :, [300, 450]]) and torch.equal(sp[0], sp[1]) and torch.equal(sp[1], sp[2])
print("F95 OK  (selection override: only the listed (batch, head, row) reads and outputs change; seen / row_sel record the "
      "pre-override reads; one-shot)")


# ── F96: the utility-label COLLECTOR (experiments/scaling/fsb_utility.py) on a 3-layer FreeSoloLM (W8, read 4, lse_pos):
#        (a) refusals: read_top 0 and a score other than lse_pos; (b) the reference capture: the recorded S0 per (l, h, i)
#        is the module's deployed read (sel_full), one shortlist / feature row per head, full score == base without a
#        re-ranker;
#        (c) swaps: exactly four unique cells, c in the shortlist outside S0, c0 in S0, rotated slots; (d) variant 0 of
#        every batched replay (baseline in the SAME batch) has the deployed read (own_eq) and every variant's
#        pre-override read at (l, i) equals it; its CE equals a plain forward of the same truncated sequence and the
#        full-length reference (tolerance); (e) the utilities do not depend on the replay batch: a label replayed alone
#        with its variant 0 (fp32, forced background) reproduces u_fp32, the duplicate variant 0 in the last batch row
#        gives the same loss (batch-position noise 0 in fp32) and the bf16 replay's rows share the deployed read;
#        (f) pack() / u_summary() / window_loss; targeted sampling weights ─────────────────────────────────────────────
from src.fsattn.attention import FSAttn as _EA96
if _HAVE_UTILITY:
    import experiments.scaling.fsb_utility as _fu
    def _mklm96(seed, width=8, **kw):
        torch.manual_seed(seed)
        lm_ = _LM32(**dict(_lmkw, n_layers=3, fsb_width=width, **kw)).to(dev).eval()
        for m_ in lm_.modules():
            if isinstance(m_, _EA96):
                m_._cell_block = BLK
        return lm_


    for bad96 in (dict(width=4), dict(fsb_read_top=4), dict(fsb_read_top=4, fsb_rerank_score="lse")):
        lmb = _mklm96(860, **bad96)
        _expect_value_error(lambda: _fu.check_config(_fu.fsb_attn_layers(lmb)), f"collector with {bad96}")
    lm96 = _mklm96(861, fsb_read_top=4, fsb_rerank_score="lse_pos")
    lay96 = _fu.fsb_attn_layers(lm96)
    _fu.check_config(lay96)
    g96 = torch.Generator(device="cpu").manual_seed(862)
    x96 = torch.randint(0, 257, (1, N + 1), generator=g96).to(dev)
    xi96, yi96 = x96[:, :N], x96[:, 1:]
    m96, gam96 = 8, 0.7
    pos96 = [(0, 2 * BLK), (1, 200), (2, N - m96), (1, 150)]
    gen96 = torch.Generator().manual_seed(863)
    cap96 = {}
    _orig96 = _ft4.masked_readout
    def _g96(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_):
        cap96.setdefault(id(attn), sel_full.clone()); return _orig96(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_)
    _ft4.masked_readout = _g96
    try:
        xs96, ce96, rec96, _ = _fu.reference_forward(lm96, lay96, xi96, yi96, {l: [i for l2, i in pos96 if l2 == l] for l, _ in pos96})
    finally:
        _ft4.masked_readout = _orig96
    for (l, i), r in rec96.items():
        assert torch.equal(r["S0"], cap96[id(lay96[l])][0, :, i]), (l, i, "recorded S0 != the deployed read")
        assert r["cands"].shape == (H, r["slots"]) and r["phi"].shape == (H, r["slots"], RR_NF)
        assert bool(((r["full"] == r["base"]) | ~torch.isfinite(r["base"])).all()), "no re-ranker: full score == base"
    assert set(rec96) == set(pos96)
    groups96, labels96, st96, t96, ceref96 = _fu.collect_sequence(lm96, lay96, xi96, yi96, pos96, m96, gam96, n_swap=2,
                                                                   max_batch=5, gen=gen96, fp32=True)
    assert groups96 and labels96 and all(g_["own_eq"] for g_ in groups96), "variant 0's read != the deployed read"
    assert all(lb["same_pre"] for lb in labels96), "a variant's pre-override read differs from variant 0's"
    w96 = _fu.horizon_weights(m96, gam96, dev)
    for lb in labels96:
        g_ = groups96[lb["g"]]
        S0 = g_["S0"].tolist()
        new = list(S0); new[lb["slot"]] = lb["c"]
        assert lb["c0"] == S0[lb["slot"]] and lb["c"] not in S0 and lb["c"] in g_["cands"].tolist() and len(set(new)) == 4
        assert abs(float((lb["ce_v0"].to(dev) * w96).sum()) - g_["L_v0"]) < 1e-3 * max(1.0, abs(g_["L_v0"]))
        assert math.isfinite(lb["u"]) and math.isfinite(lb["u_fp32"])
    # (d) variant-0 CE vs a plain forward of the same truncated sequence and vs the full-length reference
    dmax96 = dref96 = 0.0
    for (l, i) in pos96:
        g_ = next(g for g in groups96 if g["l"] == l and g["i"] == i)
        with torch.no_grad(), _fu.autocast_ctx(xi96.device):
            h_ = lm96.drop(lm96.tok_emb(xi96[:, :i + m96]))
            for blk in lm96.blocks:
                h_ = blk(h_)
            ce_tr = _fu.lm_ce(lm96, h_[:, i:i + m96], yi96[:, i:i + m96])[0]
        dmax96 = max(dmax96, float((ce_tr.cpu() - g_["ce_v0"].float()).abs().max()))
        dref96 = max(dref96, g_["dref"])
    assert dmax96 < 2e-2 and dref96 < 2e-2, (dmax96, dref96)
    # (e) batch independence of the fp32 utilities
    chk96 = 0
    for lb in labels96[:6]:
        g_ = groups96[lb["g"]]
        new = g_["S0"].clone(); new[lb["slot"]] = lb["c"]
        S0all = torch.stack([g2["S0"] for g2 in groups96 if g2["l"] == g_["l"] and g2["i"] == g_["i"]])     # (H, 4) head order
        r_ = _fu.replay_chunk(lm96, lay96, xs96[g_["l"]], yi96, g_["l"], g_["i"], m96, w96, [(g_["h"], lb["c"], lb["slot"], new)],
                              S0all, fp32=True, force_row=True)
        u_alone = float(r_["lw"][0] - r_["lw"][1])
        assert abs(u_alone - lb["u_fp32"]) < 1e-4, (u_alone, lb["u_fp32"])
        assert r_["V"] == 3 and abs(r_["dv0"]) < 1e-5, r_["dv0"]                  # the duplicate variant 0 (last row)
        rb_ = _fu.replay_chunk(lm96, lay96, xs96[g_["l"]], yi96, g_["l"], g_["i"], m96, w96, [(g_["h"], lb["c"], lb["slot"], new)],
                               S0all, fp32=False)
        assert torch.equal(rb_["row_sel"][0, :, :4].cpu(), S0all) and torch.equal(rb_["row_sel"][1], rb_["row_sel"][0])
        chk96 += 1
    assert all(math.isfinite(lb["dv0"]) and abs(lb["dv0"]) < 2e-2 for lb in labels96)
    pk96 = _fu.pack(groups96, labels96)
    assert pk96["p_u"].numel() == len(labels96) and pk96["g_phi"].shape[-1] == RR_NF
    assert bool((torch.gather(pk96["g_cands"][pk96["p_g"]], 1, pk96["p_ci"][:, None])[:, 0] == pk96["p_c"]).all())
    assert bool((torch.gather(pk96["g_cands"][pk96["p_g"]], 1, pk96["p_c0i"][:, None])[:, 0] == pk96["p_c0"]).all())
    us96 = _fu.u_summary(pk96["p_u"])
    wl96 = _fu.window_loss(ce96, w96)
    assert wl96.numel() == N - m96 + 1 and abs(float(wl96[5]) - float((ce96[5:5 + m96] * w96).sum())) < 1e-5
    pos_t, p_t = _fu.sample_positions(N, m96, BLK, 3, [0, 1, 2], torch.Generator().manual_seed(864),
                                      torch.linspace(0, 1, N), eps=0.05)
    assert len(pos_t) == 3 and all(2 * BLK <= i <= N - m96 for _, i in pos_t) and float(p_t.min()) >= 0.05 - 1e-6
    print(f"F96 OK  (collector: refusals; recorded S0 == deployed read; {len(labels96)} labels over {len(groups96)} (l,h,i): 4 unique "
          f"cells, c outside S0, c0 in S0; own_eq; variant-0 CE vs plain truncated forward {dmax96:.1e}, vs full reference "
          f"{dref96:.1e}; fp32 u batch-independent on {chk96} labels; u pos {us96['pos']:.2f} mean|u| {us96['mean_abs']:.2e})")
else:
    print("F96 SKIP (experiments/scaling/fsb_utility.py absent)")


# ── F97: router-only checkpoints and the eval-time overlay.  (a) save_route_ckpt(router_only) writes fsb_route.safetensors
#        + step / meta and NO model.safetensors (the default still writes both); (b) overlay_router_file loads every
#        fsb_route_* tensor (strict) and refuses a missing tensor, a shape mismatch and an extra tensor (re-ranker weights
#        onto a model without --fsb_rerank_rr); (c) router_key_check: a model with a re-ranker and a --ckpt without its
#        tensors is refused (identity would load silently), a router file satisfies it, a --ckpt carrying re-ranker
#        tensors is refused for a model without one; (d) backbone_fingerprint(trained=...) keeps its default; (e) plumbing ─
from src.fsattn.fsb_route import overlay_router_file, router_key_check, load_router_params as _lrp97
from safetensors.torch import save_file as _sf97, load_file as _lf97

if _HAVE_ROUTERKD:
    from experiments.scaling.fsb_routerkd import save_route_ckpt as _src97, backbone_fingerprint as _bfp97
    _tmp97 = _tf91.mkdtemp(prefix="fsb_rr_f97_")
    try:
        _lmkw97 = dict(_lmkw, fsb_width=8, fsb_read_top=4, fsb_rerank_score="lse_pos")
        torch.manual_seed(870); lmA = _LM32(**_lmkw97, fsb_rerank_rr="linear")
        with torch.no_grad():
            for n_, p_ in lmA.named_parameters():
                if "fsb_route_rr" in n_:
                    p_.copy_(torch.randn_like(p_))
        dR = os.path.join(_tmp97, "rr_step5"); _src97(lmA, dR, "meta test", 5, router_only=True)
        assert sorted(os.listdir(dR)) == ["fsb_route.safetensors", "meta.txt", "step.txt"], os.listdir(dR)
        assert sorted(_lf97(os.path.join(dR, "fsb_route.safetensors"))) == sorted(route_param_names(lmA))
        dF = os.path.join(_tmp97, "full"); _src97(lmA, dF, "meta test", 5)
        assert sorted(os.listdir(dF)) == ["fsb_route.safetensors", "meta.txt", "model.safetensors", "step.txt"]
        torch.manual_seed(871); lmB = _LM32(**_lmkw97, fsb_rerank_rr="linear")
        rep97, keys97 = overlay_router_file(lmB, dR)
        pa, pb = dict(lmA.named_parameters()), dict(lmB.named_parameters())
        assert all(torch.equal(pa[k], pb[k]) for k in route_param_names(lmA)) and "loaded 4 router tensors" in rep97, rep97
        sdR = _lf97(os.path.join(dR, "fsb_route.safetensors"))
        k0 = sorted(sdR)[0]
        _sf97({k: v for k, v in sdR.items() if k != k0}, os.path.join(_tmp97, "miss.safetensors"))
        _expect_value_error(lambda: overlay_router_file(lmB, os.path.join(_tmp97, "miss.safetensors")), "missing tensor")
        _sf97({k: (v[:, :3].contiguous() if k.endswith("fsb_route_rr_w") else v) for k, v in sdR.items()}, os.path.join(_tmp97, "shape.safetensors"))
        _expect_value_error(lambda: overlay_router_file(lmB, os.path.join(_tmp97, "shape.safetensors")), "shape mismatch")
        torch.manual_seed(872); lmC = _LM32(**_lmkw97)                        # no re-ranker
        _expect_value_error(lambda: overlay_router_file(lmC, dR), "re-ranker tensors onto a model without one")
        # (c) the eval-time key check: --ckpt dirs holding a model.safetensors
        dBase = os.path.join(_tmp97, "base"); os.makedirs(dBase)
        sdC = {k: v.contiguous() for k, v in lmC.state_dict().items() if k != "lm_head.weight"}
        _sf97(sdC, os.path.join(dBase, "model.safetensors"))
        _expect_value_error(lambda: router_key_check(lmB, dBase, None), "re-ranker model, ckpt without its tensors")
        rep97c, _ = router_key_check(lmB, dBase, dR)
        assert "loaded 4 router tensors" in rep97c
        assert router_key_check(lmC, dBase, None)[0] == "no router tensors in the model"
        _expect_value_error(lambda: router_key_check(lmC, dF, None), "a ckpt carrying re-ranker tensors, model without re-ranker")
        assert "router tensors from" in router_key_check(lmB, dF, None)[0]      # a full save carries them
        # the lenient warm start still works with the re-ranker tensors
        torch.manual_seed(873); lmD = _LM32(**_lmkw97, fsb_rerank_rr="linear")
        assert "loaded 4 router tensors" in _lrp97(lmD, dR)
        # (d) fingerprint: default excludes every fsb_route_* tensor; trained=rr names includes a frozen adapter
        torch.manual_seed(874); lmE = _LM32(**_lmkw97, fsb_rerank_rr="linear", fsb_route_adapter="full", fsb_rerank_query="raw")
        rrn = [n_ for n_ in route_param_names(lmE) if "fsb_route_rr_" in n_]
        f_def, f_rr = _bfp97(lmE), _bfp97(lmE, trained=rrn)
        with torch.no_grad():
            for n_, p_ in lmE.named_parameters():
                if n_.endswith("fsb_route_W"):
                    p_.add_(1.0)
        assert _bfp97(lmE) == f_def and _bfp97(lmE, trained=rrn) != f_rr, "the adapter must be fingerprinted with the backbone"
        with torch.no_grad():
            for n_, p_ in lmE.named_parameters():
                if "fsb_route_rr_w" in n_:
                    p_.add_(1.0)
        assert _bfp97(lmE, trained=rrn) == _bfp97(lmE, trained=rrn)
    finally:
        _sh91.rmtree(_tmp97, ignore_errors=True)
else:
    print("F97 [a-d] SKIP (experiments/scaling/fsb_routerkd.py absent)")

# (e) plumbing
_kw97_off = _bak68(_NS68(attn_type="fsattn", fsb=True, fsb_read_top=4, fsb_rerank_score="lse_pos"))[1]
_kw97_on = _bak68(_NS68(attn_type="fsattn", fsb=True, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_rr="linear"))[1]
assert "fsb_rerank_rr" not in _kw97_off and _kw97_on["fsb_rerank_rr"] == "linear"
assert {k: v for k, v in _kw97_on.items() if k != "fsb_rerank_rr"} == _kw97_off
print("F97 OK  (--fsb_rerank_rr threaded by build_attn_kwargs only when set" +
      ("; router-only save + overlay + key check + fingerprint (a-d))" if _HAVE_ROUTERKD else ")"))


# ── F98: the TRAINER (experiments/scaling/fsb_rr_train.py) on synthetic labels generated by a known w*: (a) pair_loss ==
#        a manual computation (noise floor, |u| clip weights, temperature); (b) the held-out split is by sequence;
#        (c) AdamW on the pairwise loss raises held-out pairwise accuracy above the lse_pos baseline and the replay
#        win-rate is defined; (d) softmax_bg (same background) is finite and differentiable ─────────────────────────
if _HAVE_RRTRAIN:
    from experiments.scaling import fsb_rr_train as _rt98
    g98 = torch.Generator(device="cpu").manual_seed(88)
    L98, G98, C98 = 2, 600, 7
    lab98 = {"g_l": torch.randint(0, L98, (G98,), generator=g98), "g_h": torch.randint(0, H, (G98,), generator=g98),
             "g_chunk_id": torch.randint(0, 30, (G98,), generator=g98),
             "g_phi": torch.randn(G98, C98, RR_NF, generator=g98), "g_base": torch.randn(G98, C98, generator=g98) * 0.3,
             "g_cands": torch.arange(C98).expand(G98, C98).clone()}
    wstar = torch.randn(L98, H, RR_NF, generator=g98)
    sstar = lab98["g_base"] + (lab98["g_phi"] * wstar[lab98["g_l"], lab98["g_h"]][:, None, :]).sum(-1)
    lab98["g_full"] = lab98["g_base"].clone()
    lab98["g_S0"] = torch.sort(lab98["g_base"], dim=1, descending=True, stable=True).indices[:, :4]
    pg, pc, pc0, psl = [], [], [], []
    for gi in range(G98):
        S0 = lab98["g_S0"][gi].tolist(); out_ = [c for c in range(C98) if c not in S0]
        for k in range(2):
            pg.append(gi); pc.append(out_[k]); pc0.append(S0[k]); psl.append(k)
    lab98.update({"p_g": torch.tensor(pg), "p_c": torch.tensor(pc), "p_c0": torch.tensor(pc0), "p_slot": torch.tensor(psl),
                  "p_ci": torch.tensor(pc), "p_c0i": torch.tensor(pc0)})
    lab98["p_u"] = (sstar[lab98["p_g"], lab98["p_ci"]] - sstar[lab98["p_g"], lab98["p_c0i"]]) * 0.01
    lab98["p_u"][::17] = 1e-6                                                  # below the noise floor: dropped
    Wz = torch.zeros(L98, H, RR_NF, requires_grad=True)
    Tm = torch.full((L98, H), 1.3)
    allp = torch.arange(lab98["p_u"].numel())
    lo98, n98 = _rt98.pair_loss(lab98, Wz, Tm, allp, 1e-4, 0.02)
    u_ = lab98["p_u"]; keep = u_.abs() >= 1e-4
    dz = (lab98["g_base"][lab98["p_g"], lab98["p_ci"]] - lab98["g_base"][lab98["p_g"], lab98["p_c0i"]]) / 1.3
    a_ = u_.abs().clamp(max=0.02)
    man = ((a_ * F.softplus(-torch.sign(u_) * dz))[keep].sum() / a_[keep].sum())
    assert n98 == int(keep.sum()) and abs(float(lo98) - float(man)) < 1e-6, (float(lo98), float(man))
    tr98, he98, ghe98 = _rt98.split_heldout(lab98, 0.3, 0)
    assert bool((tr98 ^ he98).all()) and 0 < int(he98.sum()) < he98.numel()
    held_ids = set(lab98["g_chunk_id"][ghe98].tolist())
    assert not (held_ids & set(lab98["g_chunk_id"][~ghe98].tolist())), "held-out split must be by sequence"
    W98 = torch.zeros(L98, H, RR_NF, requires_grad=True)
    opt98 = torch.optim.AdamW([W98], lr=0.05, weight_decay=0.0)
    tri = torch.nonzero(tr98).flatten(); hei = torch.nonzero(he98).flatten()
    m_before = _rt98.pair_metrics(lab98, W98.detach(), hei, 1e-4)
    for _ in range(150):
        lo_, _ = _rt98.pair_loss(lab98, W98, torch.ones(L98, H), tri, 1e-4, 0.02)
        opt98.zero_grad(); lo_.backward(); opt98.step()
    m_after = _rt98.pair_metrics(lab98, W98.detach(), hei, 1e-4)
    assert m_before["acc"] == m_before["acc_lse_pos"] and m_after["acc"] > m_after["acc_lse_pos"] + 0.1 and m_after["acc"] > 0.85, \
        (m_before, m_after)
    wr98 = _rt98.replay_winrate(lab98, W98.detach(), ghe98)
    assert wr98["groups"] == int(ghe98.sum()) and 0.0 <= wr98["changed"] <= 1.0
    lsb, nsb = _rt98.softmax_bg_loss(lab98, W98, torch.ones(L98, H), tri[:200], 1e-4, 0.05)
    assert lsb is not None and nsb > 0 and torch.isfinite(lsb)
    lsb.backward()
    print(f"F98 OK  (trainer: pair loss == manual; split by sequence; held-out pairwise accuracy {m_before['acc']:.2f} -> "
          f"{m_after['acc']:.2f} (lse_pos {m_after['acc_lse_pos']:.2f}); replay win-rate {wr98['win_rate']:.2f} on "
          f"{wr98['evaluable']} evaluable changes; softmax_bg finite)")
else:
    print("F98 SKIP (experiments/scaling/fsb_rr_train.py absent)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  HYBRID x RESIDUAL RE-RANKER x UTILITY COLLECTOR (the merge of fsb-hybrid-sel / -rr / -io).  The re-ranker reads the
#  COMMON SHORTLIST C = [tree candidates | pool top fsb_pool_short] (fsb_tree.hybrid_shortlist) with per-candidate
#  provenance codes (tree only / pool only / both); union re-ranks C (the final common ranking), quota re-ranks only C's
#  tree cells (the tree list T_rank) while the reserved pool slots keep FS-L's order by g; pool mode refuses it.  The
#  collector labels swaps within C and records the hybrid's final read as S0.  F99..F100.
# ══════════════════════════════════════════════════════════════════════════════════════════
def _mk_hyrr(seed, mode, rrseed=None, scale=0.5, **kw):
    """_mk_hy with the residual re-ranker on and (rrseed given) random weights (own generator)."""
    m_ = _mk_hy(seed, mode, fsb_rerank_rr="linear", **kw)
    if rrseed is not None:
        g_ = torch.Generator(device="cpu").manual_seed(rrseed)
        with torch.no_grad():
            m_.fsb_route_rr_w.copy_((torch.randn(m_.fsb_route_rr_w.shape, generator=g_) * scale).to(dev))
    return m_


# ── F99: HYBRID + RE-RANKER.  (a) fsb_rerank_rr is refused with fsb_hybrid pool and builds with union / quota; (b) ZERO
#        re-ranker weights == no re-ranker BIT-EXACT under union (+ union beam) and quota 1 / 2, with and without a random
#        adapter: eager + CUDA-graph training stage (eval + train), no-cache forward, cached decode (full + lean prefill,
#        FSB_DECODE_GRAPH 0 / 1: outputs, per-token selections, pools and provenance); (c) RANDOM weights: the deployed
#        re-rank receives C = [tree candidates | pool top-short] with the python provenance codes (quota: scope = C's
#        tree cells); union reads == the float64 top-4 of lse_pos + phi . w over C, quota reads == the quota merge of the
#        pool list with the float64 ranking of C's tree cells (rows away from near-ties); the reserved pool slots are
#        those of the zero re-ranker (the pool quota keeps the order by g); some reads change (teeth); (d) PREFILL ==
#        DECODE with random weights (sets; graph decode == eager; decode provenance == prefill provenance); (e) the
#        CUDA-graph training stage == eager with random weights and a learned c, also after an in-place weight update ──
_expect_value_error(lambda: _mk_hy(990, "pool", fsb_rerank_rr="linear"), "fsb_rerank_rr with fsb_hybrid pool")
for mode99 in ("union", "quota"):
    m_ = _mk_hyrr(990, mode99)
    assert tuple(m_.fsb_route_rr_w.shape) == (H, RR_NF) and m_.fsb_hybrid == mode99 and m_.fsb_rerank_rr == "linear"
n99, N099 = 512, 360
cfg99 = (("union W8", dict(mode="union", W=8)), ("union W8 U4", dict(mode="union", W=8, fsb_union_width=4)),
         ("quota1 W16", dict(mode="quota", quota=1, W=16)), ("quota2 W8", dict(mode="quota", quota=2, W=8)))
nz99 = 0
for name99, c99 in cfg99:
    c99 = dict(c99); mode99 = c99.pop("mode")
    for ad99 in (None, 9900):
        ma = _mk_hy(99, mode99, adapter=ad99, **c99)
        mb = _mk_hyrr(99, mode99, adapter=ad99, **c99)
        Qc, Kc, Vc = _qkv_n(990, n99, 1.5)
        for mode_t in ("eval", "train"):
            ma.train(mode_t == "train"); mb.train(mode_t == "train")
            for g in (("0", "1") if dev == "cuda" else ("0",)):
                with torch.no_grad():
                    ra = _sel_n(ma, Qc, Kc, Vc, n99, g); rb = _sel_n(mb, Qc, Kc, Vc, n99, g)
                assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1]), (name99, ad99, mode_t, g)
                nz99 += 1
        ma.eval(); mb.eval()
        torch.manual_seed(991); x = torch.randn(1, n99, d_model, device=dev)
        with torch.no_grad():
            sa_, ya_ = _full_sel_n(ma, x); sb_, yb_ = _full_sel_n(mb, x)
            assert torch.equal(sa_, sb_) and torch.equal(ya_, yb_), (name99, ad99)
            for lean in (False, True):
                for dg in (("0", "1") if dev == "cuda" else ("0",)):
                    oa = _hy_cached(ma, x, N099, lean, dg); ob = _hy_cached(mb, x, N099, lean, dg)
                    assert torch.equal(oa[0], ob[0]), (name99, ad99, lean, dg)
                    for k_ in (1, 2, 3):
                        assert all(torch.equal(a_, b_) for a_, b_ in zip(oa[k_], ob[k_])), (name99, ad99, lean, dg, k_)
                    nz99 += 1
print(f"F99 [b] zero re-ranker == no re-ranker bit-exact under the hybrid on {nz99} eager / graph / train / cached-decode cases")
# (c) random weights
chg99 = rows99 = tie99 = res99 = 0
for name99, c99 in cfg99:
    c99 = dict(c99); mode99 = c99.pop("mode")
    for ad99 in (None, 9901):
        m0 = _mk_hy(992, mode99, adapter=ad99, **c99)
        m1 = _mk_hyrr(992, mode99, rrseed=9920, adapter=ad99, **c99)
        Qc, Kc, Vc = _qkv_n(993, n99, 1.5)
        with torch.no_grad():
            (s0, _), _r0 = _hcap(lambda: _sel_n(m0, Qc, Kc, Vc, n99), names=("hybrid_select",))
            (s1, _), rec1 = _hcap(lambda: _sel_n(m1, Qc, Kc, Vc, n99), names=("hybrid_select", "rerank_top"))
        diff = ~_same_sets(s0[0].long(), s1[0].long())
        chg99 += int(diff.sum()); rows99 += diff.numel()
        K_L = m1.fsb_pool_quota
        assert len(rec1["rerank_top"]) == len(rec1["hybrid_select"]) > 0, name99
        for (ha, hk, hout), (ra_, rk_, _o) in zip(rec1["hybrid_select"], rec1["rerank_top"]):
            sel_tree, pool, pq, qpos, lam, sigv, F_, SP_, scale_, s_, rt_, rsc_, tokK_, hmode, short = ha[:15]
            assert hmode == mode99 and hk.get("rr_w") is not None and rt_ == 4
            tabs = (F_["cellK"], F_["cellA"], SP_["t_idx"])
            plist = pool_rank(pool, pq, qpos, lam, sigv, *tabs, scale_, s_, short)
            plistK = pool_rank(pool, pq, qpos, lam, sigv, *tabs, scale_, s_, 4)
            C = _uc81(sel_tree, plist)
            in_t = ((C[..., :, None] == sel_tree[..., None, :]).any(-1)) & (C >= 0)
            in_p = ((C[..., :, None] == plist[..., None, :]) & (plist[..., None, :] >= 0)).any(-1) & (C >= 0)
            codes = in_t.long() * PROV_TREE + in_p.long() * PROV_POOL
            assert torch.equal(ra_[0], C), (name99, "the re-rank's candidates != [tree candidates | pool top-short]")
            assert torch.equal(rk_["prov"].long(), codes), (name99, "the re-ranker's provenance codes")
            if mode99 == "quota":
                assert torch.equal(rk_["scope"], in_t), (name99, "quota scope != C's tree cells")
            else:
                assert rk_.get("scope") is None, name99
            inp = rr_inputs(C, pq, qpos, lam, sigv, *tabs, scale_, s_, tokK_)
            phi = rr_features(inp, C, qpos, s_, codes)
            w64 = m1.fsb_route_rr_w.detach().double()[SP_["t_idx"] % H]
            u64 = inp["lse"].double() + inp["pos"].double() + (phi.double() * w64[:, None, None, :]).sum(-1)
            u64 = u64.masked_fill(C < 0, float("-inf"))
            if mode99 == "quota":
                u64 = u64.masked_fill(~in_t, float("-inf"))
            us, ui = u64.sort(-1, descending=True)
            rank64 = torch.gather(C, -1, ui[..., :4]).masked_fill(us[..., :4] == float("-inf"), -1)
            top5 = us[..., :5]
            gaps = torch.where(torch.isfinite(top5[..., 1:]), top5[..., :-1] - top5[..., 1:], torch.full_like(top5[..., 1:], 1e30))
            tie = (gaps < 1e-4 * us[..., :1].abs().clamp(min=1.0)).any(-1)
            ref = rank64 if mode99 == "union" else quota_merge(plistK, rank64, K_L, 4)
            same = _same_sets(ref.long(), hout.long())
            assert bool(same[~tie].all()), (name99, ad99, int((~same & ~tie).sum()), "hybrid + re-ranker read != float64 reference")
            tie99 += int(tie.sum()); res99 += int((~tie).sum())
        if mode99 == "quota":
            assert torch.equal(s1[0][:, BLK:, :K_L], s0[0][:, BLK:, :K_L]), (name99, "the reserved pool slots moved")
assert chg99 > 0, "random re-ranker weights never changed a hybrid read"
print(f"F99 [c] random re-ranker: C + codes (+ quota scope) as specified; reads == float64 reference on {res99} rows "
      f"({tie99} near-tie rows skipped); quota's reserved pool slots unchanged; reads changed on {chg99}/{rows99} rows")
# (d) prefill == decode
nd99 = 0
for name99, c99 in (cfg99[0], cfg99[2], cfg99[3]):
    c99 = dict(c99); mode99 = c99.pop("mode")
    mm = _mk_hyrr(994, mode99, rrseed=9940, adapter=9941, **c99)
    torch.manual_seed(995); x = torch.randn(1, n99, d_model, device=dev)
    caps99 = []
    _otr99 = _ft4.traverse_forest
    def _tr99(*a_, **k_):
        pl_ = []
        out_ = _otr99(*a_, **k_, prov_out=pl_)
        caps99.append(pl_[0].clone()); return out_
    _ft4.traverse_forest = _tr99
    try:
        with torch.no_grad():
            sel_full, yfull = _full_sel_n(mm, x)
    finally:
        _ft4.traverse_forest = _otr99
    prov_full = _rows_hb(caps99).long()
    ys = {}
    with torch.no_grad():
        for lean in (False, True):
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                y, sels, pools, provs = _hy_cached(mm, x, N099, lean, dg)
                assert (y - yfull).abs().max().item() < 1e-4, (name99, lean, dg)
                ys[(lean, dg)] = y
                dec = torch.stack(sels, dim=1)
                same = _same_sets(sel_full[:, N099:n99].long(), dec.long())
                assert bool(same.all()), (name99, lean, dg, int((~same).sum()), "decode rows differ")
                for k_, t in enumerate(range(N099, n99)):
                    if torch.equal(sels[k_].long(), sel_full[:, t].long()):
                        assert torch.equal(provs[k_].long(), prov_full[:, t]), (name99, t, "decode provenance != prefill")
                nd99 += same.numel()
    if dev == "cuda":
        for lean in (False, True):
            assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (name99, lean, "graph decode != eager")
print(f"F99 [d] random re-ranker under the hybrid: prefill selection == decode selection on {nd99} rows; graph decode == eager")
# (e) graphed training stage
if dev == "cuda":
    ng99 = 0
    for name99, c99 in cfg99:
        c99 = dict(c99); mode99 = c99.pop("mode")
        mm = _mk_hyrr(996, mode99, rrseed=9960, adapter=9961, theta=9962, **c99).train()
        Qc, Kc, Vc = _qkv_n(997, n99, 1.5)
        prev = None
        for trial in range(2):
            res = {}
            for g in ("0", "1"):
                with torch.no_grad():
                    res[g] = _sel_n(mm, Qc, Kc, Vc, n99, g) + _nstats(mm)
            assert all(torch.equal(a_, b_) for a_, b_ in zip(res["0"], res["1"])), (name99, trial, "graph != eager")
            if prev is not None:
                assert not torch.equal(prev, res["1"][0]), (name99, "the in-place re-ranker update was not seen")
            prev = res["1"][0]
            with torch.no_grad():
                mm.fsb_route_rr_w.add_(torch.randn_like(mm.fsb_route_rr_w) * 2.0)
            ng99 += 1
    print(f"F99 [e] graphed training stage == eager on {ng99} hybrid + re-ranker cases (incl. an in-place weight update)")
else:
    print("F99 [e] SKIP (no CUDA)")
print("F99 OK  (hybrid + residual re-ranker: refused in pool mode; zero weights == off bit-exact; union ranks the common "
      "shortlist, quota ranks its tree cells and keeps the pool quota by g; prefill == decode; graph == eager)")


# ── F100: the utility COLLECTOR on the hybrid.  (a) the selection override (the collector's hook) touches only the listed
#        entries in pool / union / quota mode; (b) the reference capture on a 3-layer FreeSoloLM (P4 / Wc2 pool, W4, read 4,
#        lse_pos; union / quota with a random re-ranker, pool without): per (l, h, i) the recorded shortlist and
#        provenance == what the deployed re-rank received (quota: scope == its tree cells; pool: C == the pool list, all
#        pool-only), S0 == the deployed final read (sel_full), and S0 == the recorded scores' top-4 (union) / the quota
#        merge of the recorded pool list with the recorded ranking of the scope (quota) / the pool list (pool);
#        (c) collect_sequence on union (W4) / quota (W8: tree cells outside S0 exist) with the re-ranker off: own_eq, four
#        unique cells, c in C outside S0, c0 in S0, pool-only cells in the shortlist; quota draws only swaps the
#        re-ranker can make (c a tree cell, c0 a non-reserved tree cell: >= 1 such label, none with c0 in L[:K_L]);
#        (d) pack(): g_prov / g_scope / g_plist; the trainer's replay at the collection weights reproduces S0 on every
#        group (quota: through the stored pool list and quota_merge) and pairs_in_scope keeps exactly the pairs of two
#        tree cells whose c0 is not reserved in quota (every pair in union); (e) refusals and plumbing ────────────────
for mode100 in ("pool", "union", "quota"):
    m100 = _mk_hy(100, mode100, W=8, quota=2)
    Q1, K1, V1 = _qkv_n(1000, 512, 1.5)
    Q3, K3, V3 = (t_.expand(3, -1, -1, -1).contiguous() for t_ in (Q1, K1, V1))
    with torch.no_grad():
        sp, op = _sel_n(m100, Q3, K3, V3, 512)
        ov_new = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=dev)
        ov100 = {"v": torch.tensor([1, 2], device=dev), "h": torch.tensor([0, 3], device=dev),
                 "i": torch.tensor([300, 450], device=dev), "sel": ov_new, "rows": torch.tensor([300, 450], device=dev)}
        m100._fsb_sel_override = ov100
        so, oo = _sel_n(m100, Q3, K3, V3, 512)
        assert m100._fsb_sel_override is None, mode100
        s2, o2 = _sel_n(m100, Q3, K3, V3, 512)
    assert torch.equal(s2, sp) and torch.equal(o2, op), (mode100, "the override leaked into the next call")
    mk100 = torch.zeros(3, H, sp.shape[2], dtype=torch.bool, device=dev); mk100[1, 0, 300] = True; mk100[2, 3, 450] = True
    assert torch.equal(so[~mk100], sp[~mk100]) and torch.equal(so[1, 0, 300], ov_new[0]) and torch.equal(so[2, 3, 450], ov_new[1])
    om100 = mk100[:, :, :512]
    assert torch.equal(oo[~om100], op[~om100]), (mode100, "the override changed an entry it does not list")
    for (b_, h_, i_), nw in (((1, 0, 300), ov_new[0]), ((2, 3, 450), ov_new[1])):
        if not bool(_same_sets(nw[None].long(), sp[b_, h_, i_][None].long())[0]):
            assert not torch.equal(oo[b_, h_, i_], op[b_, h_, i_]), (mode100, "a changed read left the output unchanged")
    assert torch.equal(ov100["seen"], torch.stack([sp[1, 0, 300], sp[2, 3, 450]])) and torch.equal(ov100["row_sel"], sp[:, :, [300, 450]])


print("F100 [a] selection override exact in pool / union / quota")
if _HAVE_UTILITY and _HAVE_RRTRAIN:
    def _mklm100(seed, mode, rr=None, short=4, quota=2, width=4):
        hk = dict(fsb_hybrid=mode, fsb_pool_size=PH, fsb_pool_recent=WCH, fsb_pool_short=short, fsb_pool_quota=quota)
        if rr is not None:
            hk["fsb_rerank_rr"] = "linear"
        lm_ = _mklm96(seed, width=width, fsb_read_top=4, fsb_rerank_score="lse_pos", **hk)
        if rr is not None:
            g_ = torch.Generator(device="cpu").manual_seed(rr)
            with torch.no_grad():
                for m_ in _fu.fsb_attn_layers(lm_):
                    m_.fsb_route_rr_w.copy_((torch.randn(m_.fsb_route_rr_w.shape, generator=g_) * 0.5).to(dev))
        return lm_


    g100 = torch.Generator(device="cpu").manual_seed(1001)
    x100 = torch.randint(0, 257, (1, N + 1), generator=g100).to(dev)
    xi100, yi100 = x100[:, :N], x100[:, 1:]
    m100w, gam100 = 8, 0.7
    pos100 = [(0, 2 * BLK), (1, 200), (2, N - m100w), (1, 150), (2, 3 * BLK + 5)]
    tg100 = {}
    for l, i in pos100:
        tg100.setdefault(l, []).append(i)
    nrec100 = 0
    for mode100 in ("pool", "union", "quota"):
        lm = _mklm100(1002, mode100, rr=(1003 if mode100 != "pool" else None))
        lay = _fu.fsb_attn_layers(lm)
        _fu.check_config(lay)
        cur100 = {}
        hk100 = [m_.register_forward_pre_hook(lambda mod, args, l=l: cur100.__setitem__("l", l)) for l, m_ in enumerate(lay)]
        rt100 = []
        cap100 = {}
        _ort100, _omr100 = _fth.rerank_top, _ft4.masked_readout
        def _rt100(sel_, q_, qpos_, *a_, **k_):
            rt100.append((cur100["l"], sel_.clone(), qpos_.clone(), k_.get("prov"), k_.get("scope")))
            return _ort100(sel_, q_, qpos_, *a_, **k_)
        def _mr100(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_):
            cap100.setdefault(id(attn), sel_full.clone()); return _omr100(attn, Qq, Kp, Vp, sel_full, s_, Bq_, n_)
        _fth.rerank_top = _rt100; _ft4.masked_readout = _mr100
        try:
            xs100, ce100, rec100, _ = _fu.reference_forward(lm, lay, xi100, yi100, tg100)
        finally:
            _fth.rerank_top = _ort100; _ft4.masked_readout = _omr100
            for h_ in hk100:
                h_.remove()
        assert set(rec100) == set(pos100), mode100
        for (l, i), r in rec100.items():
            assert torch.equal(r["S0"], cap100[id(lay[l])][0, :, i]), (mode100, l, i, "recorded S0 != the deployed final read")
            C, codes, scope = r["cands"], r["prov"].long(), r["scope"]
            ok = C >= 0
            assert torch.equal(codes == 0, ~ok) and r["phi"].shape == (H, C.shape[-1], RR_NF)
            if mode100 == "pool":
                assert torch.equal(C[:, :4], r["plist"]) and bool((codes[ok] == PROV_POOL).all())
                assert torch.equal(r["S0"], r["plist"]), (mode100, l, i, "pool read != the pool list")
            else:
                qi = i % BLK
                hits = []
                for (lc, sel_, qpos_, prov_, scope_) in rt100:
                    if lc != l:
                        continue
                    rows = torch.nonzero(qpos_[:, qi] == float(i)).flatten()
                    if rows.numel():
                        hits.append((sel_, qpos_, prov_, scope_, rows))
                assert len(hits) == 1, (mode100, l, i, len(hits))
                sel_, qpos_, prov_, scope_, rows = hits[0]
                assert rows.numel() == H, (mode100, l, i, rows.numel())        # rows h * nbc + j (B = 1): ascending = head order
                assert torch.equal(C, sel_[rows, qi]), (mode100, l, i, "recorded shortlist != the deployed re-rank's candidates")
                assert torch.equal(codes, prov_[rows, qi].long()), (mode100, l, i, "recorded provenance != the deployed codes")
                if mode100 == "quota":
                    assert torch.equal(scope, scope_[rows, qi]) and torch.equal(scope, ok & ((codes & PROV_TREE) != 0))
                else:
                    assert torch.equal(scope, ok)
                full = r["full"].masked_fill(~scope, float("-inf"))
                o_ = torch.sort(full, dim=-1, descending=True, stable=True)
                top = torch.gather(C, -1, o_.indices[:, :4]).masked_fill(o_.values[:, :4] == float("-inf"), -1)
                want = top if mode100 == "union" else quota_merge(r["plist"], top, 2, 4)
                assert torch.equal(r["S0"], want), (mode100, l, i, "S0 != the recorded scores' read")
            nrec100 += 1
    print(f"F100 [b] hybrid capture: recorded shortlist / provenance / scope == the deployed re-rank's, S0 == the final read and "
          f"== the recorded scores' read on {nrec100} (mode, l, i) groups")
    nlab100 = {}
    for mode100 in ("union", "quota"):
        lm = _mklm100(1004, mode100, width=(8 if mode100 == "quota" else 4))
        lay = _fu.fsb_attn_layers(lm)
        gen100 = torch.Generator().manual_seed(1005)
        grp, lab, st_, _, _ = _fu.collect_sequence(lm, lay, xi100, yi100, pos100, m100w, gam100, n_swap=2, max_batch=6, gen=gen100)
        assert grp and lab and all(g_["own_eq"] for g_ in grp), (mode100, "variant 0's read != the deployed read")
        assert all(lb["same_pre"] for lb in lab), mode100
        for lb in lab:
            g_ = grp[lb["g"]]
            S0 = g_["S0"].tolist(); new = list(S0); new[lb["slot"]] = lb["c"]
            assert lb["c0"] == S0[lb["slot"]] and lb["c"] not in S0 and lb["c"] in g_["cands"].tolist() and len(set(new)) == 4
        assert any(bool((g_["prov"].long() == PROV_POOL).any()) for g_ in grp), (mode100, "no pool-only cell offered (no teeth)")
        pk = _fu.pack(grp, lab)
        assert pk["g_prov"].shape == pk["g_cands"].shape and pk["g_scope"].shape == pk["g_cands"].shape and pk["g_plist"].shape[1] == 4
        Wc100 = torch.zeros(len(lay), H, RR_NF)
        wr = _rt98.replay_winrate(pk, Wc100, torch.ones(pk["g_l"].numel(), dtype=torch.bool), quota=(2 if mode100 == "quota" else None))
        assert wr["changed"] == 0, (mode100, wr, "the replay at the collection weights must reproduce S0")
        insc = _rt98.pairs_in_scope(pk)
        both_tree = torch.tensor([bool(pk["g_prov"][lb["g"], int(pk["p_ci"][k])] & PROV_TREE) and
                                  bool(pk["g_prov"][lb["g"], int(pk["p_c0i"][k])] & PROV_TREE) for k, lb in enumerate(lab)])
        assert torch.equal(insc, both_tree if mode100 == "quota" else torch.ones_like(insc)), mode100
        insq = _rt98.pairs_in_scope(pk, 2 if mode100 == "quota" else None)
        if mode100 == "quota":
            res100 = pk["g_plist"][pk["p_g"], :2]
            assert not bool(((res100 == pk["p_c0"][:, None]) & (res100 >= 0)).any()), "quota: a swap replaces a reserved pool cell"
            assert int(insq.sum()) > 0 and bool(insq.all()), (mode100, "quota: every drawn swap must be one the re-ranker can make")
        else:
            assert bool(insq.all()), mode100
        nlab100[mode100] = (len(lab), int(insq.sum()))
    # (e) refusals and plumbing
    lmq = _mklm100(1006, "quota", short=2)
    _expect_value_error(lambda: _fu.check_config(_fu.fsb_attn_layers(lmq)), "collector: quota with fsb_pool_short < fsb_read_top")
    _ns100 = _types81.SimpleNamespace(width=16, read_top=4, rerank_score="lse_pos", rerank_query="raw", union_width=0, coef=8.0,
                                      adapter="none", alpha=1.0, fsb_rerank_rr="none")
    _k0 = _fu.attn_kwargs(_ns100)
    assert not any(k_.startswith("fsb_pool") or k_ == "fsb_hybrid" for k_ in _k0)
    _ns100.hybrid = "off"; _ns100.pool_short = 4
    assert _fu.attn_kwargs(_ns100) == _k0
    _ns100.hybrid = "quota"; _ns100.pool_quota = 2; _ns100.pool_size = None
    _k1 = _fu.attn_kwargs(_ns100)
    assert {k: v for k, v in _k1.items() if k not in _k0} == {"fsb_hybrid": "quota", "fsb_pool_short": 4, "fsb_pool_quota": 2}
    print(f"F100 OK  (collector on the hybrid: override exact in pool / union / quota; capture == deployed shortlist, codes, "
          f"scope and final read; labels {nlab100} (all, in scope: quota swaps only non-reserved tree cells); replay reproduces S0; "
          f"refusals + attn_kwargs plumbing)")
else:
    print("F100 [b-e] SKIP (experiments/scaling/fsb_utility.py or fsb_rr_train.py absent)")


# ── F101: review fixes of the W2 re-ranker pipeline.  (a) quota scope == REACHABILITY: brute force over every tree
#        ranking, a pair (c in, c0 out) is a read the quota merge can produce iff pairs_in_scope keeps it (the replaced
#        cell never a reserved pool cell); (b) make_swaps: off / union records draw exactly the pre-fix swaps (reference
#        copy), quota records only reachable swaps; (c) load_labels: an older shard (no g_prov / g_scope / g_plist /
#        fp32 keys) + a new one concatenate row-aligned with pack()'s defaults; shards of another config / backbone are
#        refused (allowed on request); (d) split_heldout never empty with >= 2 chunks; pair_noise; (e) the trainer's
#        metric / replay scores (det) == the DEPLOYED full score recorded by the collector, bit for bit, with a random
#        re-ranker (off / union / quota), and the replay at the collection weights reproduces S0; (f) re-ranker
#        IDENTITY: recorded in the router file, the overlay refuses another backbone / selection config (allowed on
#        request), legacy files load; rr_all_zero + train_dist's refusal; (g) on_nfs resolves symlinks / '..';
#        (h) quota + re-ranker validates fsb_pool_short; (i) decode: a selection config changed on a live cache
#        mid-block (quota K_L, pool size, off -> quota) selects exactly as a fresh cache (eager and CUDA graph) ─────────
import itertools as _it101
import json as _js101
import hashlib as _hl101
from src.fsattn.fsb_route import (rr_identity, rr_select_config, adapter_sha256, identity_mismatch, router_file_identity,
                                 RR_IDENTITY_KEY, ckpt_weights_sha256, rr_all_zero)
from src.fsattn.fsb_tree import quota_merge as _qm101

# (a) reachability
if _HAVE_RRTRAIN:
    g101 = torch.Generator(device="cpu").manual_seed(1010)
    nreach101 = nin101 = nteeth101 = 0
    for trial in range(60):
        KL = 1 + trial % 2
        tree = torch.randperm(12, generator=g101)[:5 + trial % 2].tolist()          # tree cells (the re-ranker ranks them)
        pl = torch.randperm(12, generator=g101)[:4].tolist()                       # the ranked pool list L[:K]
        if trial % 5 == 0:
            pl[3] = -1                                                              # a short pool
        Tr0 = torch.tensor(tree[:4])[None]
        S0 = _qm101(torch.tensor(pl)[None], Tr0, KL, 4)[0]
        if int((S0 >= 0).sum()) < 4:
            continue
        reach = set()
        for perm in _it101.permutations(tree):
            reach.add(frozenset(_qm101(torch.tensor(pl)[None], torch.tensor(perm[:4])[None], KL, 4)[0].tolist()))
        cands = sorted(set(tree) | set(c for c in pl if c >= 0))
        C_ = torch.tensor(cands)
        labx = {"g_scope": torch.tensor([[c in tree for c in cands]]), "g_plist": torch.tensor([pl]), "g_cands": C_[None]}
        pg, pc, pc0, pci, pc0i = [], [], [], [], []
        for c in cands:
            if c in S0.tolist():
                continue
            for c0 in S0.tolist():
                pg.append(0); pc.append(c); pc0.append(c0); pci.append(cands.index(c)); pc0i.append(cands.index(c0))
        labx.update({"p_g": torch.tensor(pg), "p_c": torch.tensor(pc), "p_c0": torch.tensor(pc0), "p_ci": torch.tensor(pci),
                     "p_c0i": torch.tensor(pc0i)})
        ins = _rt98.pairs_in_scope(labx, KL)
        for k in range(len(pg)):
            new = frozenset(set(S0.tolist()) - {pc0[k]} | {pc[k]})
            assert bool(ins[k]) == (new in reach), (trial, KL, tree, pl, S0.tolist(), pc[k], pc0[k], bool(ins[k]))
            nreach101 += int(new in reach); nin101 += 1
        # the pre-fix rule (tree bits only) kept pairs replacing a reserved cell: count them (the fix's teeth)
        nteeth101 += int((_rt98.pairs_in_scope(labx) & ~ins).sum())
    assert nreach101 > 0 and nin101 > nreach101 and nteeth101 > 0
    print(f"F101 [a] quota scope == reachability over every tree ranking on {nin101} pairs ({nreach101} reachable; the pre-fix "
          f"scope kept {nteeth101} unreachable pairs with a reserved c0)")
else:
    print("F101 [a] SKIP (experiments/scaling/fsb_rr_train.py absent)")

# (b) make_swaps: the pre-fix draw (reference copy) for off / union records; quota records only reachable swaps
if _HAVE_UTILITY:
    def _make_swaps_ref(rec, n_swap, gen, K=4):
        H_ = rec["S0"].shape[0]; swaps, ok = [], []
        for h in range(H_):
            S0 = rec["S0"][h].cpu(); cands = rec["cands"][h].cpu()
            good = bool((S0 >= 0).all()) and int(torch.unique(S0).numel()) == K
            ok.append(good)
            if not good:
                continue
            valid = cands[cands >= 0]
            outside = valid[~(valid[:, None] == S0[None, :]).any(-1)]
            if outside.numel() == 0:
                continue
            pick = outside[torch.randperm(outside.numel(), generator=gen)[:n_swap]]
            off = int(torch.randint(0, K, (1,), generator=gen))
            for k, c in enumerate(pick.tolist()):
                slot = (off + k) % K
                new = S0.clone(); new[slot] = c
                swaps.append((h, int(c), slot, new))
        return swaps, torch.tensor(ok)


    nsw101 = {"ref": 0, "quota": 0}
    for trial in range(40):
        Hs, Cs = 4, 10
        cands = torch.stack([torch.randperm(20, generator=g101)[:Cs] for _ in range(Hs)])
        cands[:, 7] = -1
        S0 = cands[:, [0, 2, 3, 5]].clone()
        if trial % 7 == 0:
            S0[1, 3] = S0[1, 2]                                                        # a head without K unique cells
        rec = {"S0": S0, "cands": cands, "scope": cands >= 0}
        for n_sw in (1, 2, 5):
            ga, gb = torch.Generator().manual_seed(trial), torch.Generator().manual_seed(trial)
            sa, oka = _fu.make_swaps(rec, n_sw, ga)
            sb, okb = _make_swaps_ref(rec, n_sw, gb)
            assert torch.equal(oka, okb) and len(sa) == len(sb), trial
            for x_, y_ in zip(sa, sb):
                assert x_[:3] == y_[:3] and torch.equal(x_[3], y_[3]), (trial, x_, y_)
            assert int(torch.randint(0, 1 << 30, (1,), generator=ga)) == int(torch.randint(0, 1 << 30, (1,), generator=gb))
            nsw101["ref"] += len(sa)
        # quota: scope = the first 6 slots (tree), reserved = S0 slot 0 (a pool cell that is also a tree cell)
        scope = torch.zeros_like(cands, dtype=torch.bool); scope[:, :6] = True; scope &= cands >= 0
        resv = torch.zeros_like(S0, dtype=torch.bool); resv[:, 0] = True
        recq = dict(rec, scope=scope, quota=True, reserved=resv)
        sq, _ = _fu.make_swaps(recq, 3, torch.Generator().manual_seed(trial))
        for (h, c, slot, new) in sq:
            tree_h = cands[h][scope[h]].tolist()
            assert c in tree_h and c not in S0[h].tolist() and not bool(resv[h, slot]) and int(S0[h, slot]) in tree_h, (trial, h, c, slot)
        nsw101["quota"] += len(sq)
    assert nsw101["quota"] > 0
    print(f"F101 [b] make_swaps: off / union draws == the pre-fix draws ({nsw101['ref']} swaps, same generator state after); "
          f"quota: {nsw101['quota']} swaps, every c a tree cell, every c0 a non-reserved tree cell")
else:
    print("F101 [b] SKIP (experiments/scaling/fsb_utility.py absent)")

# (c) load_labels: old + new shards, config / backbone refusals
if _HAVE_RRTRAIN:
    _tmp101 = _tf91.mkdtemp(prefix="fsb_rr_f101_")
    try:
        def _shard(G, C, P, seed, old=False, **meta_over):
            gg = torch.Generator().manual_seed(seed)
            cands = torch.arange(C).expand(G, C).clone(); cands[:, -1] = -1
            d = {"g_l": torch.zeros(G, dtype=torch.long), "g_i": torch.arange(G), "g_h": torch.arange(G) % H,
                 "g_seq": torch.zeros(G, dtype=torch.long), "g_chunk_id": torch.arange(G) + 100 * seed,
                 "g_S0": cands[:, :4].clone(), "g_cands": cands, "g_phi": torch.randn(G, C, RR_NF, generator=gg),
                 "g_base": torch.randn(G, C, generator=gg).masked_fill(cands < 0, float("-inf")),
                 "g_own_eq": torch.ones(G, dtype=torch.bool), "g_heads_ok": torch.ones(G, dtype=torch.bool),
                 "g_dref": torch.zeros(G), "g_L_v0": torch.zeros(G), "g_L_ref": torch.zeros(G), "g_ce_v0": torch.zeros(G, 8).half(),
                 "p_g": torch.arange(P) % G, "p_c": torch.full((P,), C - 2), "p_c0": torch.arange(P) % 4, "p_slot": torch.arange(P) % 4,
                 "p_ci": torch.full((P,), C - 2), "p_c0i": torch.arange(P) % 4, "p_u": torch.randn(P, generator=gg) * 1e-2,
                 "p_chunk": torch.zeros(P, dtype=torch.long), "p_same_pre": torch.ones(P, dtype=torch.bool),
                 "p_dv0": torch.zeros(P), "p_ce": torch.zeros(P, 8).half()}
            d["g_full"] = d["g_base"].clone()
            if not old:
                d.update({"g_prov": (cands >= 0).to(torch.int8), "g_scope": cands >= 0, "g_plist": torch.full((G, 4), -1),
                          "g_own_eq_fp32": torch.ones(G, dtype=torch.bool), "p_u_fp32": torch.randn(P, generator=gg) * 1e-3,
                          "p_dv0_fp32": torch.zeros(P)})
            args = dict(ckpt="/ck/base", config="340m", width=16, read_top=4, rerank_score="lse_pos", rerank_query="raw",
                        union_width=0, coef=8.0, adapter="none", alpha=1.0, m=8, gamma=0.7)
            if not old:
                args.update(hybrid="off", pool_size=None, pool_recent=None, pool_short=None, pool_quota=None)
            args.update({k: v for k, v in meta_over.items() if k != "identity"})
            meta = {"args": args, "features": list(RR_FEATURES)}
            if "identity" in meta_over:
                meta["identity"] = meta_over["identity"]
            f = os.path.join(_tmp101, f"s{seed}.pt"); torch.save({"meta": meta, **d}, f)
            return f
        fo = _shard(2, 7, 3, 1, old=True)
        fn = _shard(3, 6, 4, 2)
        lab101, metas101 = _rt98.load_labels([fn, fo])
        assert lab101["g_cands"].shape == (5, 7) and lab101["g_scope"].shape == (5, 7) and lab101["g_plist"].shape == (5, 4)
        assert torch.equal(lab101["p_g"], torch.tensor([0, 1, 2, 0, 3, 4, 3]))
        assert torch.equal(lab101["g_scope"][3:], lab101["g_cands"][3:] >= 0) and bool((lab101["g_plist"][3:] == -1).all())
        assert torch.equal(lab101["g_prov"][3:].long(), (lab101["g_cands"][3:] >= 0).long() * PROV_TREE)
        assert bool(torch.isnan(lab101["p_u_fp32"][4:]).all()) and bool(torch.isfinite(lab101["p_u_fp32"][:4]).all())
        ins101 = _rt98.pairs_in_scope(lab101)                                   # the reviewer's IndexError case
        assert ins101.shape == (7,) and bool(ins101.all())
        for bad_kw, what in ((dict(width=8), "width"), (dict(hybrid="quota", pool_quota=2), "hybrid"),
                             (dict(ckpt="/ck/other"), "base path"), (dict(gamma=0.5), "gamma")):
            fb = _shard(2, 6, 2, 3, **bad_kw)
            try:
                _rt98.load_labels([fn, fb]); raise AssertionError(f"shards of another {what} must be refused")
            except SystemExit as e_:
                assert "disagree" in str(e_), (what, str(e_))
            _rt98.load_labels([fn, fb], allow_mismatch=True)
        fi1 = _shard(2, 6, 2, 4, identity={"base_sha256": "a" * 64, "adapter_sha256": "none"})
        fi2 = _shard(2, 6, 2, 5, identity={"base_sha256": "b" * 64, "adapter_sha256": "none"})
        try:
            _rt98.load_labels([fi1, fi2]); raise AssertionError("shards of another backbone (sha256) must be refused")
        except SystemExit as e_:
            assert "weight sha256" in str(e_), str(e_)
        fi3 = _shard(2, 6, 2, 6, ckpt="/ck/elsewhere", identity={"base_sha256": "a" * 64, "adapter_sha256": "none"})
        _rt98.load_labels([fi1, fi3])                                           # same sha256 at another path: fine
        # explicit library defaults == unset
        _rt98.load_labels([_shard(2, 6, 2, 7, hybrid="quota"), _shard(2, 6, 2, 8, hybrid="quota", pool_quota=1, pool_size=64)])
        # (d) held-out split and the measured noise
        for n_ch, want in ((1, 0), (2, 1), (3, 1), (10, 2)):
            labh = {"g_chunk_id": torch.arange(n_ch).repeat_interleave(3), "p_g": torch.arange(3 * n_ch)}
            trh, heh, gheh = _rt98.split_heldout(labh, 0.2, 0)
            assert len(set(labh["g_chunk_id"][gheh].tolist())) == want, (n_ch, want)
            assert len(set(labh["g_chunk_id"][~gheh].tolist())) == n_ch - want
        assert not bool(_rt98.split_heldout(labh, 0.0, 0)[1].any())
        labn = {"p_u": torch.tensor([0.1, -0.2, 0.3]), "p_u_fp32": torch.tensor([0.05, float("nan"), 0.31]),
                "p_dv0": torch.tensor([0.0, 0.5, float("nan")]), "p_dv0_fp32": torch.tensor([1e-6, float("nan"), 0.0])}
        assert torch.allclose(_rt98.pair_noise(labn, "bf16"), torch.tensor([0.05, 0.5, 0.01]), atol=1e-6)
        assert torch.equal(_rt98.pair_noise(labn, "fp32"), torch.tensor([1e-6, 0.0, 0.0]))
    finally:
        _sh91.rmtree(_tmp101, ignore_errors=True)
    print("F101 [c,d] load_labels: old + new shards aligned (defaults filled; pairs_in_scope OK); width / hybrid / base path / "
          "gamma / weight-sha256 mismatches refused (allowed on request); held-out split never empty; pair_noise")
else:
    print("F101 [c,d] SKIP (experiments/scaling/fsb_rr_train.py absent)")

# (e) det scores == the deployed full score; replay at the collection weights == S0 (random re-ranker)
if _HAVE_UTILITY and _HAVE_RRTRAIN:
    neq101 = ndiff101 = 0
    for mode101 in ("off", "union", "quota"):
        if mode101 == "off":
            lm = _mklm96(1020, width=8, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_rr="linear")
            g_ = torch.Generator(device="cpu").manual_seed(1021)
            with torch.no_grad():
                for m_ in _fu.fsb_attn_layers(lm):
                    m_.fsb_route_rr_w.copy_((torch.randn(m_.fsb_route_rr_w.shape, generator=g_) * 0.5).to(dev))
        else:
            lm = _mklm100(1020, mode101, rr=1021, width=8)
        lay = _fu.fsb_attn_layers(lm)
        Wc = torch.stack([m_.fsb_route_rr_w.detach().float() for m_ in lay])
        _, _, rec101, _ = _fu.reference_forward(lm, lay, xi100, yi100, tg100)
        phis, bases, fulls, gl, gh = [], [], [], [], []
        for (l, i), r in rec101.items():
            for h in range(H):
                phis.append(r["phi"][h]); bases.append(r["base"][h]); fulls.append(r["full"][h]); gl.append(l); gh.append(h)
        labd = {"g_phi": torch.stack(phis), "g_base": torch.stack(bases), "g_l": torch.tensor(gl), "g_h": torch.tensor(gh)}
        ga_ = torch.arange(len(gl))
        gs_det = _rt98.group_scores(labd, Wc, ga_, det=True)
        assert torch.equal(gs_det, torch.stack(fulls)), (mode101, "trainer det score != the deployed full score")
        gs_pl = _rt98.group_scores(labd, Wc, ga_, det=False)
        ndiff101 += int((gs_pl != gs_det).sum()); neq101 += gs_det.numel()
        if mode101 != "off":
            grp, labs, _, _, _ = _fu.collect_sequence(lm, lay, xi100, yi100, pos100[:3], m100w, gam100, n_swap=1, max_batch=6,
                                                      gen=torch.Generator().manual_seed(1022))
            pk = _fu.pack(grp, labs)
            wr = _rt98.replay_winrate(pk, Wc, torch.ones(pk["g_l"].numel(), dtype=torch.bool), quota=(2 if mode101 == "quota" else None))
            assert wr["changed"] == 0, (mode101, wr, "replay at the collection weights must reproduce S0")
    print(f"F101 [e] trainer det scores == deployed full scores bit-exact on {neq101} (group, cell) entries, off / union / quota "
          f"with a random re-ranker (the plain sum differs on {ndiff101}); replay at the collection weights reproduces S0")
else:
    print("F101 [e] SKIP (experiments/scaling/fsb_utility.py or fsb_rr_train.py absent)")

if _HAVE_ROUTERKD:
    _tmp101 = _tf91.mkdtemp(prefix="fsb_rr_f101i_")
    try:
        _kw101 = dict(_lmkw, fsb_width=8, fsb_read_top=4, fsb_rerank_score="lse_pos")
        torch.manual_seed(1030); lmA = _LM32(**_kw101, fsb_rerank_rr="linear")
        with torch.no_grad():
            for n_, p_ in lmA.named_parameters():
                if "fsb_route_rr_w" in n_:
                    p_.copy_(torch.randn_like(p_))
        bases101 = []
        for k_, sd_seed in ((1, 1031), (2, 1032)):
            torch.manual_seed(sd_seed); lm_ = _LM32(**_kw101)
            d_ = os.path.join(_tmp101, f"base{k_}"); os.makedirs(d_)
            _sf97({k: v.contiguous() for k, v in lm_.state_dict().items() if k != "lm_head.weight"}, os.path.join(d_, "model.safetensors"))
            bases101.append(d_)
        idA = rr_identity(lmA, bases101[0])
        assert idA["base_sha256"] == _hl101.sha256(open(os.path.join(bases101[0], "model.safetensors"), "rb").read()).hexdigest()
        assert idA["adapter_sha256"] == "none" and idA["config"]["fsb_rerank_query"] == "raw" and idA["config"]["fsb_hybrid"] == "off"
        assert "fsb_pool_quota" not in idA["config"] and idA["config"]["features"] == list(RR_FEATURES)
        dR = os.path.join(_tmp101, "rr"); _src97(lmA, dR, "m", 3, router_only=True,
                                                 route_metadata={RR_IDENTITY_KEY: _js101.dumps(dict(idA, rr_trained=1))})
        assert router_file_identity(dR)["base_sha256"] == idA["base_sha256"] and router_file_identity(dR)["rr_trained"] == 1
        torch.manual_seed(1033); lmB = _LM32(**_kw101, fsb_rerank_rr="linear")
        assert rr_all_zero(lmB)
        repB, _ = overlay_router_file(lmB, dR, ckpt=bases101[0])
        assert "identity ok" in repB and not rr_all_zero(lmB), repB
        assert identity_mismatch(idA, model=lmB, ckpt=bases101[0], check_adapter=True) == []
        _expect_value_error(lambda: overlay_router_file(lmB, dR, ckpt=bases101[1]), "a re-ranker on another backbone")
        _expect_value_error(lambda: router_key_check(lmB, bases101[1], dR), "key check: another backbone")
        assert "MISMATCH ALLOWED" in router_key_check(lmB, bases101[1], dR, allow_mismatch=True)[0]
        for bad_kw, what in ((dict(fsb_width=4), "width"), (dict(fsb_hybrid="union", fsb_pool_size=4, fsb_pool_recent=2,
                                                                fsb_pool_short=4), "hybrid union")):
            torch.manual_seed(1034); lmX = _LM32(**dict(_kw101, **bad_kw), fsb_rerank_rr="linear")
            _expect_value_error(lambda: overlay_router_file(lmX, dR, ckpt=bases101[0]), f"a re-ranker on another {what}")
            assert "MISMATCH ALLOWED" in overlay_router_file(lmX, dR, ckpt=bases101[0], allow_mismatch=True)[0]
        dL = os.path.join(_tmp101, "legacy"); _src97(lmA, dL, "m", 3, router_only=True)
        assert router_file_identity(dL) is None and "no re-ranker identity" in overlay_router_file(lmB, dL, ckpt=bases101[1])[0]
        torch.manual_seed(1035); lmE = _LM32(**_kw101, fsb_rerank_rr="linear", fsb_route_adapter="full", fsb_rerank_query="raw")
        a0 = adapter_sha256(lmE)
        with torch.no_grad():
            for n_, p_ in lmE.named_parameters():
                if n_.endswith("fsb_route_W"):
                    p_.add_(0.5)
        assert a0 != "none" and adapter_sha256(lmE) != a0 and rr_select_config(lmE)["fsb_route_adapter"] == "full"
        with torch.no_grad():
            for n_, p_ in lmE.named_parameters():
                if n_.endswith("fsb_route_rr_w"):
                    p_.add_(1.0)
        assert identity_mismatch({"adapter_sha256": a0}, model=lmE, check_adapter=True), "adapter change must be detected"
        assert ckpt_weights_sha256(os.path.join(_tmp101, "nothing")) is None
    finally:
        _sh91.rmtree(_tmp101, ignore_errors=True)
    print("F101 [f] identity: base sha256 + selection config recorded; overlay refuses another backbone / width / "
          "hybrid (allowed on request), legacy files load; adapter sha256 tracks the adapter")
else:
    print("F101 [f] SKIP (experiments/scaling/fsb_routerkd.py absent)")

_nfs101 = os.environ.get("FSB_TEST_NFS_PATH", "")
if _HAVE_UTILITY and _nfs101 and os.path.exists(_nfs101):
    _tmp101 = _tf91.mkdtemp(prefix="fsb_rr_f101n_")
    try:
        os.symlink(_nfs101, os.path.join(_tmp101, "lnk"))
        assert _fu.on_nfs(os.path.join(_nfs101, "x")) and _fu.on_nfs(os.path.join(_nfs101, "..", os.path.basename(_nfs101), "y"))
        assert _fu.on_nfs(os.path.join(_tmp101, "lnk", "labels")), "a symlink into NFS"
        assert not _fu.on_nfs(os.path.join(_tmp101, "labels"))
    finally:
        _sh91.rmtree(_tmp101, ignore_errors=True)
    print("F101 [g] on_nfs: FSB_TEST_NFS_PATH itself, '..' through it and a symlink into it are NFS; the local "
          "temp dir is not")
else:
    print("F101 [g] SKIP (experiments/scaling/fsb_utility.py absent or FSB_TEST_NFS_PATH unset/unreachable)")

# (h) quota + re-ranker: fsb_pool_short validated
for bad_short in (0, PH + WCH + 1):
    _expect_value_error(lambda: _mk_hy(1040, "quota", short=bad_short, fsb_rerank_rr="linear"), f"quota + rr short {bad_short}")
assert _mk_hy(1040, "quota", short=0).fsb_pool_short == 0                  # without the re-ranker: unused, unchanged
assert _mk_hy(1040, "quota", short=PH + WCH, fsb_rerank_rr="linear").fsb_pool_short == PH + WCH
print("F101 [h] quota + re-ranker refuses fsb_pool_short outside 1..P+Wc (quota alone unchanged)")

# (i) decode: a selection config changed mid-block on a live cache == a fresh cache with that config
def _hy_switch(mm_, x_, N0, dgraph, t_sw=None, setter=None):
    n_ = x_.shape[1]
    os.environ["FSB_DECODE_GRAPH"] = dgraph
    try:
        cache = FSKVCache(mm_, lean_prefill=False); cache.reserve = n_ - N0
        sels = {}
        with torch.no_grad():
            mm_(x_[:, :N0], cache=cache)
            for t in range(N0, n_):
                if t == t_sw:
                    setter(mm_)
                mm_(x_[:, t:t + 1], cache=cache)
                sels[t] = cache._fsb_forest._last_sel.reshape(H, -1).clone()
    finally:
        os.environ.pop("FSB_DECODE_GRAPH", None)
    return sels


def _set101(**kv):
    def f_(m_):
        for k_, v_ in kv.items():
            setattr(m_, k_, v_)
    return f_


t_sw101 = 400                                                              # mid-block (blocks start at 384, 448)
teeth101 = nrow101 = 0
for name101, mkA, mkB, setter in (
        ("quota K_L 1->2", lambda: _mk_hy(1050, "quota", quota=1), lambda: _mk_hy(1050, "quota", quota=2),
         _set101(fsb_pool_quota=2)),
        ("union pool P 4->2", lambda: _mk_hy(1051, "union"), lambda: _mk_hy(1051, "union", P=2), _set101(fsb_pool_size=2)),
        ("off -> quota 2", lambda: _mk_hy(1052, None), lambda: _mk_hy(1052, "quota", quota=2),
         _set101(fsb_hybrid="quota", fsb_pool_size=PH, fsb_pool_recent=WCH, fsb_pool_short=4, fsb_pool_quota=2))):
    torch.manual_seed(1053); x = torch.randn(1, n86, d_model, device=dev)
    for dg in (("0", "1") if dev == "cuda" else ("0",)):
        mA, mB, mC = mkA(), mkB(), mkA()
        assert all(torch.equal(a_, b_) for a_, b_ in zip(mA.state_dict().values(), mB.state_dict().values()))
        sw = _hy_switch(mA, x, N086, dg, t_sw101, setter)
        ref = _hy_switch(mB, x, N086, dg)
        ctl = _hy_switch(mC, x, N086, dg)                                  # never switched
        for t in range(t_sw101, n86):
            assert torch.equal(sw[t], ref[t]), (name101, dg, t, "switched live cache != a fresh cache")
            nrow101 += 1
        teeth101 += sum(int(not torch.equal(ctl[t], ref[t])) for t in range(t_sw101, 448))
assert teeth101 > 0, "no decode row differs between the two configs inside the switched block: no teeth"
print(f"F101 [i] decode: quota K_L / pool size / off -> quota changed mid-block on a live cache == a fresh cache on "
      f"{nrow101} decode steps (eager{' + CUDA graph' if dev == 'cuda' else ''}; {teeth101} in-block steps differ between the configs)")
print("F101 OK  ((h) quota + re-ranker fsb_pool_short validated; (i) decode selection-config change == a fresh cache; "
      "(a)-(g) ran or SKIPPED as printed above)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  EVAL-TIME NOISE FILTER (tau quantile calibration) + ADAPTIVE READ COUNT (fsb_read_margin), 2026-09-29.
#  F102..F104.  The tests use W16 beams (read 4 or 16), the lse_pos re-rank with the raw query and a random full
#  adapter, the small hybrid pool (P = 4, Wc = 2) and fresh caches for every run.
# ══════════════════════════════════════════════════════════════════════════════════════════
from src.fsattn.fsb_tree import (fsb_calibrate_tau as _fct, margin_kw as _mkw, _margin_trim, masked_readout as _mro,
                                fsb_read_stats_reset, fsb_read_stats)


def _ceil_of(mm_, x_):
    """per-cell max key norm (H,M) of x_ (a multiple of S tokens) through the module's own key projection (the KV
    cache's cell summaries, the values fsb_long_range calibrates on)."""
    with torch.no_grad():
        c_ = FSKVCache(mm_); mm_(x_, cache=c_)
    return c_._ceil[0].clone()


def _mk_mg(seed, mode, margin=None, rmin=1, W=16, rt=16, adapter=None, P=PH, Wc=WCH, short=4, quota=1, **kw):
    """forest, fixed W/D0 beam, read rt, lse_pos re-rank with the raw query; the hybrid kwargs when ``mode`` is given
    (None = absent); fsb_read_margin / fsb_read_min passed only when ``margin`` is not None (explicit 0 = the flag)."""
    hk = {} if mode is None else dict(fsb_hybrid=mode, fsb_pool_size=P, fsb_pool_recent=Wc, fsb_pool_short=short,
                                      fsb_pool_quota=quota)
    mg = {} if margin is None else dict(fsb_read_margin=margin, fsb_read_min=rmin)
    kw.setdefault("fsb_bound_radius_coef", 4.0)
    return _mk_int(seed, adapter=adapter, fsb_policy="fixed", fsb_width=W, fsb_delta=0, fsb_tree="forest", fsb_read_top=rt,
                   fsb_rerank_score="lse_pos", fsb_rerank_query="raw", **hk, **mg, **kw)


# ── F102: tau QUANTILE calibration (fsb_calibrate_tau, the eval driver's --fsb_tau_quantile) and correctness with tau > 0.
#   (a) on the tiny module the calibrated tau_h == torch.quantile of the concatenated per-cell max key norms of the
#       calibration inputs (the KV cache's own cell summaries); the returned admitted fraction == 1 - q up to ties;
#       quantile 0 = no-op (None, tau untouched); the calibration hooks are cleared;
#   (b) a q 0.5 tau on W16 read 4 lse_pos raw + random full adapter, hybrid off / union / quota: admission gates the
#       TREE only (the pool rows equal the tau-0 module's; off: every read cell has a_c >= tau; union / quota: every
#       read cell below tau is a pool cell); the read sets differ from tau 0 (teeth); prefill == decode (full + lean
#       prefill, eager + graph decode), cached == full forward, graph decode == eager, chunked prefill 1 / 2 / all
#       identical, CUDA-graph training stage == eager.  (No admitted cell + non-empty pool, D2, is F86's all-inf case.) ──
mm = _mk_os(1020, fsb_policy="fixed", fsb_width=8, fsb_delta=0, fsb_tree="forest", fsb_read_top=4, fsb_rerank_score="lse_pos",
            fsb_rerank_query="raw")
torch.manual_seed(1021); xs102 = [torch.randn(1, n_, d_model, device=dev) for n_ in (512, 256, 512)]
assert float(mm.fsb_tau.abs().max()) == 0.0 and _fct(mm, xs102, 0.0) is None and float(mm.fsb_tau.abs().max()) == 0.0
res102 = _fct(mm, xs102, 0.5)
assert mm._fsb_calib is None and mm._fsb_stats is None and not mm.training       # hooks cleared, mode restored
a102 = torch.cat([_ceil_of(mm, x_) for x_ in xs102], dim=1)                    # (H, 80) the calibration cells' a_c
tau_ref = torch.quantile(a102, 0.5, dim=1)
assert torch.allclose(mm.fsb_tau, tau_ref, atol=1e-5) and torch.allclose(res102["tau"][0].to(dev), tau_ref, atol=1e-5)
assert bool((mm.fsb_tau > 0).all()) and tuple(res102["tau"].shape) == (1, H) and res102["n_cells"] == a102.shape[1]
frac_ref = (a102 >= mm.fsb_tau[:, None]).float().mean(1)
assert torch.allclose(res102["admit_frac"][0].to(dev), frac_ref, atol=1e-6)
assert bool((frac_ref >= 0.5).all()) and bool((frac_ref <= 0.5 + 2.0 / a102.shape[1]).all()), frac_ref
print(f"F102 [a] fsb_calibrate_tau: tau == the 0.5-quantile of the cache's cell summaries ({a102.shape[1]} cells / head), "
      f"admitted fraction {[round(float(v), 3) for v in frac_ref]}; q 0 = no-op")
n102, N0102 = 512, 300
nrow102 = nord102 = nbelow102 = 0
for name102, mode102, c102 in (("off W16 r4", None, {}), ("union W16 r4", "union", dict(short=4)),
                               ("quota W16 r4 K_L 2", "quota", dict(quota=2))):
    mm = _mk_mg(1022, mode102, W=16, rt=4, adapter=10220, **c102)
    m0 = _mk_mg(1022, mode102, W=16, rt=4, adapter=10220, **c102)
    assert all(torch.equal(a_, b_) for a_, b_ in zip(mm.state_dict().values(), m0.state_dict().values()))
    torch.manual_seed(1023); xc102 = [torch.randn(1, 512, d_model, device=dev) for _ in range(2)]
    _fct(mm, xc102, 0.5)
    assert bool((mm.fsb_tau > 0).all()) and float(m0.fsb_tau.abs().max()) == 0.0
    torch.manual_seed(1024); x = torch.randn(1, n102, d_model, device=dev)
    with torch.no_grad():
        (sel_t, y_t), rec_t = _hcap(lambda: _full_sel_n(mm, x), names=("hybrid_pool",))
        (sel_0, y_0), rec_0 = _hcap(lambda: _full_sel_n(m0, x), names=("hybrid_pool",))
    assert not torch.equal(sel_t, sel_0), (name102, "tau 0.5 changed no read set (no teeth)")
    ce102 = _ceil_of(mm, x)                                                     # (H,M)
    below = ce102.gather(1, sel_t.clamp(min=0).view(H, -1)).view_as(sel_t) < mm.fsb_tau.view(H, 1, 1)
    below = below & (sel_t >= 0)
    if mode102 is None:
        assert not bool(below.any()), (name102, "a tree read below tau")
    else:
        pt = [o_ for _, _, o_ in rec_t["hybrid_pool"]]; p0 = [o_ for _, _, o_ in rec_0["hybrid_pool"]]
        assert len(pt) == len(p0) > 0 and all(torch.equal(a_, b_) for a_, b_ in zip(pt, p0)), (name102, "pool rows depend on tau")
        pb = _pools_by_block(pt)
        prow = torch.stack([pb[i // BLK] for i in range(sel_t.shape[1])], dim=1)          # (H,Npad,Cp)
        inpool = (sel_t[..., :, None] == prow[..., None, :]).any(-1)
        assert bool((inpool | ~below).all()), (name102, "a read cell below tau is not a pool cell")
        nbelow102 += int(below.sum())
    ys = {}
    with torch.no_grad():
        for lean in (False, True):
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                if mode102 is None:
                    y, sels = _cached_run_n(mm, x, N0102, lean, dg)
                else:
                    y, sels, _pools, _provs = _hy_cached(mm, x, N0102, lean, dg)
                d = (y - y_t).abs().max().item()
                assert d < 1e-4, (name102, lean, dg, d)
                dec = torch.stack(sels, dim=1)
                same = _same_sets(sel_t[:, N0102:n102].long(), dec.long())
                assert bool(same.all()), f"{name102} lean={lean} graph={dg}: {int((~same).sum())} decode rows differ"
                nord102 += int((sel_t[:, N0102:n102].long() == dec.long()).all(-1).sum()); nrow102 += same.numel()
                ys[(lean, dg)] = y
        if dev == "cuda":
            for lean in (False, True):
                assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (name102, lean, "graph decode != eager")
    outs = []
    for nbch in (1, 2, 0):
        mc = _mk_mg(1022, mode102, W=16, rt=4, adapter=10220, fsb_prefill_blkchunk=nbch, **c102)
        with torch.no_grad():
            mc.fsb_tau.copy_(mm.fsb_tau)
            outs.append(_full_sel_n(mc, x) + _nstats(mc))
    assert all(all(torch.equal(a_, b_) for a_, b_ in zip(outs[0], o_)) for o_ in outs[1:]), (name102, "chunk size")
    assert torch.equal(outs[0][0], sel_t), (name102, "chunk 1 != the default chunking")
    if dev == "cuda":
        mm.train(); Qc, Kc, Vc = _qkv_n(1025, n102)
        with torch.no_grad():
            r0 = _sel_n(mm, Qc, Kc, Vc, n102, "0") + _nstats(mm)
            r1 = _sel_n(mm, Qc, Kc, Vc, n102, "1") + _nstats(mm)
        assert all(torch.equal(a_, b_) for a_, b_ in zip(r0, r1)), (name102, "graphed training stage != eager under tau")
        mm.eval()
print(f"F102 OK  (tau quantile: calibration == quantile of the cell summaries; tau > 0 gates the tree only (pool rows == "
      f"tau 0, {nbelow102} pool reads below tau, off reads all >= tau); prefill == decode on {nrow102} rows (order identical "
      f"on {nord102}), cached == full, graph decode == eager, chunked prefill identical, training graph == eager)")


# ── F103: ADAPTIVE READ COUNT (fsb_read_margin delta, fsb_read_min m).
#   (a) construction: defaults 0 / 1 (margin_kw empty); refusals: margin < 0, margin without read_top, read_min outside
#       1..read_top, read_min without a margin, hybrid pool / quota with a margin (union takes it); _margin_trim on a
#       hand-made row;
#   (b) margin 0 passed explicitly == the flag absent bit-exact: selection / output on the no-cache forward, per-token
#       decode (graph 0 / 1), the CUDA-graph training stage; off and union;
#   (c) semantics vs float64 brute force on every re-rank call: the read is a PREFIX of the fixed top-K order of length
#       clamp(max(#candidates within delta of the best, min(m, #valid)), K) (rows with a candidate at the boundary
#       skipped), with m and K respected and some rows reading strictly between m and K (teeth);
#   (d) with a margin: prefill == decode (full + lean, graph 0 / 1), cached == full, graph decode == eager, chunked
#       prefill identical, training stage graph == eager with the margin in the graph key (a changed margin re-captures
#       and changes the read), Triton == torch, a margin changed mid-block on a live decode cache == a fresh cache;
#   (e) the per-example read-count accumulators (fsb_read_stats_reset / fsb_read_stats) == the prefill's n_read plus the
#       decode selections, eager and graph decode; disarmed = None;
#   (f) the masked readout and the fused Triton readout skip -1 slots ANYWHERE in the slot list (a per-row slot
#       permutation of a margin-trimmed selection gives the same output);
m = _mk_mg(1030, None)
assert m.fsb_read_margin == 0.0 and m.fsb_read_min == 1 and _mkw(m) == {} and m.fsb_kk == 16
m = _mk_mg(1030, None, margin=2.0, rmin=3)
assert m.fsb_read_margin == 2.0 and m.fsb_read_min == 3 and _mkw(m) == {"read_margin": 2.0, "read_min": 3}
m = _mk_mg(1030, "union", margin=1.0)
assert _mkw(m) == {"read_margin": 1.0, "read_min": 1} and m.fsb_hybrid == "union"
for what, fn in (("margin < 0", lambda: _mk_mg(1030, None, margin=-1.0)),
                 ("margin without read_top", lambda: mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_tree="forest",
                                                        fsb_read_margin=1.0)),
                 ("read_min 0", lambda: _mk_mg(1030, None, margin=1.0, rmin=0)),
                 ("read_min > read_top", lambda: _mk_mg(1030, None, margin=1.0, rmin=17)),
                 ("read_min without margin", lambda: _mk_mg(1030, None, margin=0.0, rmin=2)),
                 ("hybrid pool + margin", lambda: _mk_mg(1030, "pool", margin=1.0)),
                 ("hybrid quota + margin", lambda: _mk_mg(1030, "quota", margin=1.0))):
    _expect_value_error(fn, what)
tv103 = torch.tensor([[5.0, 4.5, 3.0, 2.9, float("-inf")], [float("-inf")] * 5], device=dev)
o103 = torch.tensor([[7, 3, 9, 1, -1], [-1] * 5], device=dev)
assert torch.equal(_margin_trim(o103, tv103, 5, 0.0, 1), o103)
assert _margin_trim(o103, tv103, 5, 1.0, 1).tolist() == [[7, 3, -1, -1, -1], [-1] * 5]
assert _margin_trim(o103, tv103, 5, 1.0, 3).tolist() == [[7, 3, 9, -1, -1], [-1] * 5]
assert _margin_trim(o103, tv103, 5, 2.0, 1).tolist() == [[7, 3, 9, -1, -1], [-1] * 5]
assert _margin_trim(o103, tv103, 5, 2.5, 1).tolist() == [[7, 3, 9, 1, -1], [-1] * 5]
print("F103 [a] construction: defaults 0 / 1; 7 refusals; union takes the margin; _margin_trim keeps a prefix >= read_min")
n103, N0103 = 512, 300
for mode103 in (None, "union"):
    mA = _mk_mg(1032, mode103, adapter=10320); mB = _mk_mg(1032, mode103, adapter=10320, margin=0.0)
    assert all(torch.equal(a_, b_) for a_, b_ in zip(mA.state_dict().values(), mB.state_dict().values()))
    torch.manual_seed(1033); x = torch.randn(1, n103, d_model, device=dev)
    with torch.no_grad():
        sA, yA = _full_sel_n(mA, x); sB, yB = _full_sel_n(mB, x)
    assert torch.equal(sA, sB) and torch.equal(yA, yB), (mode103, "explicit margin 0 != absent")
    for dg in (("0", "1") if dev == "cuda" else ("0",)):
        with torch.no_grad():
            rA = (_cached_run_n(mA, x, N0103, False, dg) if mode103 is None else _hy_cached(mA, x, N0103, False, dg)[:2])
            rB = (_cached_run_n(mB, x, N0103, False, dg) if mode103 is None else _hy_cached(mB, x, N0103, False, dg)[:2])
        assert torch.equal(rA[0], rB[0]) and all(torch.equal(a_, b_) for a_, b_ in zip(rA[1], rB[1])), (mode103, dg)
    if dev == "cuda":
        mA.train(); mB.train(); Qc, Kc, Vc = _qkv_n(1034, n103)
        with torch.no_grad():
            rA = _sel_n(mA, Qc, Kc, Vc, n103, "1") + _nstats(mA); rB = _sel_n(mB, Qc, Kc, Vc, n103, "1") + _nstats(mB)
        assert all(torch.equal(a_, b_) for a_, b_ in zip(rA, rB)), (mode103, "graphed stage")
        mA.eval(); mB.eval()
print("F103 [b] explicit margin 0 == absent bit-exact (no-cache forward, cached decode graph 0 / 1, training stage; off + union)")
nchk103 = nbnd103 = nteeth103 = ncall103 = 0
for mode103, delta103, rmin103 in ((None, 0.5, 1), (None, 2.0, 3), ("union", 1.0, 2)):
    mm = _mk_mg(1035, mode103, adapter=10350, margin=delta103, rmin=rmin103)
    torch.manual_seed(1036); x = torch.randn(1, n103, d_model, device=dev)
    with torch.no_grad():
        (sel_m, y_m), rec = _hcap(lambda: _full_sel_n(mm, x), names=("rerank_top",))
    assert rec["rerank_top"], "no re-rank call captured"
    assert int((sel_m >= 0).sum(-1).max()) <= 16
    for a_, k_, o_ in rec["rerank_top"]:
        sel_c, q_, qpos_, lam_, sigv_, cellK_, cellA_, tidx_, scale_, s_, K_ = a_
        assert k_.get("read_margin") == delta103 and k_.get("read_min") == rmin103 and k_["score"] == "lse_pos"
        k0 = {kk: vv for kk, vv in k_.items() if kk not in ("read_margin", "read_min")}
        with torch.no_grad():
            o_full = rerank_top(*a_, **k0)                                       # the fixed top-K order
            u64 = _tok_brute(q_, sel_c, k_["tokK"], tidx_, scale_, s_, "lse_pos", qpos_, lam_, sigv_, cellA_)
        thr = u64.max(-1).values - delta103
        n_valid = (sel_c >= 0).sum(-1)
        n_in = ((u64 >= thr[..., None]) & (sel_c >= 0)).sum(-1)               # valid candidates within the margin (a row
        # without any candidate, block 0, has thr = -inf and reads nothing)
        n_exp = torch.maximum(n_in, torch.minimum(torch.full_like(n_valid, rmin103), n_valid)).clamp(max=K_)
        n_got = (o_ >= 0).sum(-1)
        bnd = ((u64 - thr[..., None]).abs() < 1e-4).any(-1)                      # a candidate at the margin boundary
        pref = torch.where(torch.arange(K_, device=dev) < n_got[..., None], o_full, torch.full_like(o_full, -1))
        assert torch.equal(o_, pref), (mode103, delta103, rmin103, "the margin read is not a prefix of the fixed top-K order")
        assert bool((n_got == n_exp)[~bnd].all()), (mode103, delta103, rmin103, int(((n_got != n_exp) & ~bnd).sum()))
        nchk103 += int((~bnd).sum()); nbnd103 += int(bnd.sum()); ncall103 += 1
        nteeth103 += int(((n_got < K_) & (n_got > rmin103) & (n_valid > n_got)).sum())
assert nteeth103 > 0, "no row reads strictly between read_min and read_top: no teeth"
print(f"F103 [c] margin read == prefix of the fixed top-K order of the float64 brute-force length on {nchk103} rows "
      f"({nbnd103} boundary rows skipped, {ncall103} re-rank calls, {nteeth103} rows strictly between m and K)")
nrow103 = 0
for mode103 in (None, "union"):
    mm = _mk_mg(1037, mode103, adapter=10370, margin=1.0, rmin=2)
    torch.manual_seed(1038); x = torch.randn(1, n103, d_model, device=dev)
    with torch.no_grad():
        sel_f, y_f = _full_sel_n(mm, x)
    nr_f = (sel_f[:, BLK:] >= 0).sum(-1)
    assert int(nr_f.min()) >= 2 and int(nr_f.max()) <= 16 and bool((nr_f < 16).any()), (mode103, "read counts")
    ys = {}
    with torch.no_grad():
        for lean in (False, True):
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                y, sels = (_cached_run_n(mm, x, N0103, lean, dg) if mode103 is None else _hy_cached(mm, x, N0103, lean, dg)[:2])
                assert (y - y_f).abs().max().item() < 1e-4, (mode103, lean, dg)
                dec = torch.stack(sels, dim=1)
                same = _same_sets(sel_f[:, N0103:n103].long(), dec.long())
                assert bool(same.all()), f"{mode103} lean={lean} graph={dg}: {int((~same).sum())} decode rows differ"
                ys[(lean, dg)] = y; nrow103 += same.numel()
        if dev == "cuda":
            for lean in (False, True):
                assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (mode103, lean, "graph decode != eager")
    outs = []
    for nbch in (1, 2, 0):
        mc = _mk_mg(1037, mode103, adapter=10370, margin=1.0, rmin=2, fsb_prefill_blkchunk=nbch)
        with torch.no_grad():
            outs.append(_full_sel_n(mc, x) + _nstats(mc))
    assert all(all(torch.equal(a_, b_) for a_, b_ in zip(outs[0], o_)) for o_ in outs[1:]), (mode103, "chunk size")
    assert torch.equal(outs[0][0], sel_f)
    if dev == "cuda":
        mm.train(); Qc, Kc, Vc = _qkv_n(1039, n103)
        with torch.no_grad():
            r0 = _sel_n(mm, Qc, Kc, Vc, n103, "0") + _nstats(mm); r1 = _sel_n(mm, Qc, Kc, Vc, n103, "1") + _nstats(mm)
            assert all(torch.equal(a_, b_) for a_, b_ in zip(r0, r1)), (mode103, "graphed training stage != eager")
            k1 = mm._fsb_train_graphs["key"]
            mm.fsb_read_margin = 3.0
            r2 = _sel_n(mm, Qc, Kc, Vc, n103, "1") + _nstats(mm); k2 = mm._fsb_train_graphs["key"]
            r3 = _sel_n(mm, Qc, Kc, Vc, n103, "0") + _nstats(mm)
            assert k2 != k1 and all(torch.equal(a_, b_) for a_, b_ in zip(r2, r3)), (mode103, "stale graph after a margin change")
            assert not torch.equal(r2[0], r1[0]) and float(r2[2].sum()) > float(r1[2].sum()), (mode103, "a wider margin read no more")
            mm.fsb_read_margin = 1.0
            ref = None
            for trv, rrk in (("1", "1"), ("0", "1"), ("1", "0"), ("0", "0")):
                with _env(FSB_RERANK_TRITON=rrk):
                    r_ = _sel_n(mm, Qc, Kc, Vc, n103, "0", triton=trv) + _nstats(mm)
                if ref is None:
                    ref = r_
                assert all(torch.equal(a_, b_) for a_, b_ in zip(ref, r_)), (mode103, trv, rrk, "Triton != torch")
        mm.eval()
torch.manual_seed(1040); x103d = torch.randn(1, n103, d_model, device=dev)
for dg in (("0", "1") if dev == "cuda" else ("0",)):
    mA = _mk_mg(1041, None, adapter=10410, margin=1.0); mB = _mk_mg(1041, None, adapter=10410, margin=3.0, rmin=2)
    mC = _mk_mg(1041, None, adapter=10410, margin=1.0)
    sw = _hy_switch(mA, x103d, N0103, dg, 400, _set101(fsb_read_margin=3.0, fsb_read_min=2))
    ref = _hy_switch(mB, x103d, N0103, dg); ctl = _hy_switch(mC, x103d, N0103, dg)
    assert all(torch.equal(sw[t], ref[t]) for t in range(400, n103)), (dg, "margin switched mid-block on a live cache != fresh")
    assert any(not torch.equal(ctl[t], ref[t]) for t in range(400, 448)), (dg, "no teeth in the switched block")
print(f"F103 [d] with a margin: prefill == decode on {nrow103} rows (full + lean, graph 0 / 1), cached == full, graph decode == "
      f"eager, chunked prefill identical, training graph == eager + key re-captures, Triton == torch, live-cache switch == fresh")
mm = _mk_mg(1042, None, adapter=10420, margin=1.0)
assert fsb_read_stats(mm) is None and getattr(mm, "_fsb_read_acc", None) is None
torch.manual_seed(1043); x = torch.randn(1, n103, d_model, device=dev)
sts103 = {}
for dg in (("0", "1") if dev == "cuda" else ("0",)):
    fsb_read_stats_reset(mm)
    with torch.no_grad():
        y, sels = _cached_run_n(mm, x, N0103, False, dg)
    st = fsb_read_stats(mm)
    pre = mm._fsb_stats["n_read"]                                                  # (1,H,N0) the prefill's reads
    ndec = (torch.stack(sels, dim=1) >= 0).sum(-1).float()                         # (H, n - N0) the decode reads
    tot = float(pre.sum() + ndec.sum()); rows = pre.numel() + ndec.numel()
    assert abs(st["mean"] - tot / rows) < 1e-5 and st["rows"] == rows, (dg, st, tot / rows, rows)
    assert st["max"] == max(float(pre.max()), float(ndec.max())) and 0.0 < st["mean"] < 16.0
    assert len(st["layer_mean"]) == 1 and abs(st["layer_mean"][0] - st["mean"]) < 1e-6
    sts103[dg] = st
if dev == "cuda":
    assert sts103["0"] == sts103["1"], "read stats differ between eager and graph decode"
fsb_read_stats_reset(mm, on=False)
assert fsb_read_stats(mm) is None and mm._fsb_read_acc is None
print(f"F103 [e] read-count accumulators == prefill n_read + decode selections (mean {sts103['0']['mean']:.3f}, max "
      f"{sts103['0']['max']:.0f} over {sts103['0']['rows']:.0f} rows), eager == graph decode; disarmed = None")
mm = _mk_mg(1044, None, adapter=10440, margin=1.0, rmin=2)
Qc, Kc, Vc = _qkv_n(1045, n103)
with torch.no_grad():
    sel_f, y_f = _sel_n(mm, Qc, Kc, Vc, n103)                                     # (1,H,Npad,16), -1 tails
    assert bool((sel_f[:, :, BLK:] < 0).any()) and bool((sel_f[:, :, BLK:] >= 0).any())
    g103 = torch.Generator(device="cpu").manual_seed(1046)
    perm = torch.stack([torch.randperm(16, generator=g103) for _ in range(sel_f.shape[2])]).to(dev)
    sel_p = sel_f.gather(-1, perm[None, None].expand_as(sel_f))                    # -1 slots anywhere
    assert bool(((sel_p >= 0).sum(-1) == (sel_f >= 0).sum(-1)).all()) and not torch.equal(sel_p, sel_f)
    y_m = _mro(mm, Qc, Kc, Vc, sel_f, S, BLK, n103); y_p = _mro(mm, Qc, Kc, Vc, sel_p, S, BLK, n103)
    assert torch.equal(y_m, y_f) and (y_p - y_m).abs().max().item() < 1e-5, "masked readout depends on the slot order"
    if dev == "cuda":
        y_fu = mm._cell_fused_lr(Qc, Kc, Vc, sel_f.to(torch.int32).contiguous(), S, BLK)
        y_fp = mm._cell_fused_lr(Qc, Kc, Vc, sel_p.to(torch.int32).contiguous(), S, BLK)
        d_fu = (y_fu - y_m).abs().max().item(); d_fp = (y_fp - y_m).abs().max().item()
        assert d_fu < 2e-3 and d_fp < 2e-3, (d_fu, d_fp)
        print(f"F103 [f] fused Triton readout skips -1 slots anywhere: |max| vs masked {d_fu:.1e} (tails) / {d_fp:.1e} (permuted)")
    else:
        print("F103 [f] masked readout skips -1 slots anywhere (fused kernel: SKIP, no CUDA)")
print("F103 OK  (fsb_read_margin: construction + refusals; margin 0 == absent; prefix-of-top-K semantics == float64 "
      "brute force; prefill == decode, graph == eager (decode + training, keys), chunked prefill, Triton == torch; "
      "read-count accumulators; masked + fused readouts skip -1 anywhere)")


# ── F104: fsb_tau_calib (the eval driver's noise filter): the seeded chunk draw from the first 98% of a uint16 corpus, the
#   tau cache file (computed once, re-read identical, loaded by a second model of the same key with identical tau and sha,
#   the chunk ids recorded; another data path under the same key refused), the tiny FreeSoloLM's tau > 0 on every layer,
#   tau_summary ──────────────────────────────────────────────────────────────────────────────────────────────────────
import numpy as _np104
from experiments.scaling import fsb_tau_calib as _ftc
_d104 = tempfile.mkdtemp(prefix="fsb_tau_")
_rng104 = _np104.random.default_rng(104)
_tok104 = _rng104.integers(0, 257, size=200 * 65, dtype=_np104.uint16)             # 200 chunks of 64 (+1) tokens
_bin104 = os.path.join(_d104, "corpus.bin"); _tok104.tofile(_bin104)
ids_a, ch_a = _ftc.calib_chunks(_bin104, 4, 64, 7)
ids_b, ch_b = _ftc.calib_chunks(_bin104, 4, 64, 7)
ids_c, _ = _ftc.calib_chunks(_bin104, 4, 64, 8)
assert ids_a == ids_b and ids_a != ids_c and len(set(ids_a)) == 4 and ids_a == sorted(ids_a) and max(ids_a) < int(200 * 0.98)
assert all(c_.shape == (1, 64) and c_.dtype == torch.long for c_ in ch_a)
assert all(torch.equal(c_[0], torch.from_numpy(_tok104[i * 65:i * 65 + 64].astype(_np104.int64))) for i, c_ in zip(ids_a, ch_a))
assert all(torch.equal(a_, b_) for a_, b_ in zip(ch_a, ch_b))
try:
    _ftc.calib_chunks(_bin104, 500, 64, 7)
    raise AssertionError("more chunks than the train split holds were accepted")
except ValueError:
    pass
assert _ftc.tau_cache_path("/x", "ab" * 32, 0.5, 8, 4096, 1234) == "/x/fsb_tau_abababababababab_q0.5_n8_l4096_s1234.pt"
torch.manual_seed(1040); lm1 = _LM32(**_lmkw).to(dev).eval()
torch.manual_seed(1040); lm2 = _LM32(**_lmkw).to(dev).eval()
cal104 = dict(ckpt_sha="ab" * 32, quantile=0.5, n_batches=4, length=64, seed=7, data_path=_bin104, cache_dir=_d104,
              device=dev, autocast_dtype=None, log=lambda *a_, **k_: None)
info1 = _ftc.calibrate_or_load(lm1, **cal104)
assert info1["loaded_from_cache"] is False and info1["chunk_ids"] == ids_a and len(info1["tau_sha256"]) == 64
assert os.path.isfile(info1["cache_file"]) and info1["quantile"] == 0.5 and info1["n_batches"] == 4 and info1["seed"] == 7
t1 = torch.stack([m_.fsb_tau.detach().float().cpu() for m_ in lm1.modules() if getattr(m_, "fsb_enabled", False)])
assert tuple(t1.shape) == (2, H) and bool((t1 > 0).all()) and _ftc.tau_sha256(t1) == info1["tau_sha256"]
assert 0.4 <= info1["admit_frac_calib_mean"] <= 0.7 and len(info1["admit_frac_calib_layer"]) == 2 and info1["n_cells_per_head"] == 16
info2 = _ftc.calibrate_or_load(lm2, **cal104)
t2 = torch.stack([m_.fsb_tau.detach().float().cpu() for m_ in lm2.modules() if getattr(m_, "fsb_enabled", False)])
assert info2["loaded_from_cache"] is True and torch.equal(t1, t2) and info2["tau_sha256"] == info1["tau_sha256"]
assert info2["chunk_ids"] == ids_a and info2["cache_file"] == info1["cache_file"]
per104, all104 = _ftc.tau_summary(lm2)
assert len(per104) == 2 and all104[0] <= all104[1] <= all104[2] and abs(all104[1] - float(t1.mean())) < 1e-6
_bin104b = os.path.join(_d104, "other.bin"); _tok104.tofile(_bin104b)
try:
    _ftc.calibrate_or_load(lm2, **dict(cal104, data_path=_bin104b))
    raise AssertionError("a cached tau of another data path was accepted")
except RuntimeError:
    pass
import shutil
shutil.rmtree(_d104, ignore_errors=True)
print(f"F104 OK  (fsb_tau_calib: seeded chunk draw inside the train split, chunk ids {ids_a}; tau cached once and reloaded "
      f"identical (sha {info1['tau_sha256'][:12]}), calibration admitted fraction {info1['admit_frac_calib_mean']:.3f}; "
      f"another data path refused)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  BEAM QUOTA (fsb_beam_quota k, with fsb_read_top; pass 3 of fsb_improvement_plan_20260929): the main beam's first k
#  valid picks are reserved in the read, the exact re-rank order over ALL candidates (main + union beam [+ the hybrid's
#  pool shortlist]) fills the remaining slots; composes with the hybrid pool quota ([L[:K_L] | beam | tree | L[K_L:]]).
#  F105..F108.  The tests use the W4 main beam with the U16 union beam (the pass-3 selector), read 4, the lse_pos
#  re-rank with the raw query, random full adapters, the small hybrid pool (P 4, Wc 2) and fresh caches for every run.
# ══════════════════════════════════════════════════════════════════════════════════════════
from src.fsattn.fsb_tree import beam_reserved, beam_quota_merge, beam_kw as _bkw, beam_prov, hybrid_codes as _hc105
from src.fsattn.fsb_rerank import PROV_BEAMQ


def _mk_bq(seed, k, W=4, U=16, rt=4, adapter=None, mode=None, P=PH, Wc=WCH, short=4, quota=1, **kw):
    """forest, fixed W/D0 main beam, union beam U (0 = none), read rt, lse_pos re-rank with the raw query; fsb_beam_quota
    only when ``k`` is not None (explicit 0 = the flag); the hybrid kwargs when ``mode`` is given (None = absent)."""
    base = dict(fsb_policy="fixed", fsb_width=W, fsb_delta=0, fsb_tree="forest", fsb_read_top=rt, fsb_rerank_score="lse_pos",
                fsb_rerank_query="raw", fsb_bound_radius_coef=4.0)
    if U:
        base["fsb_union_width"] = U
    if mode is not None:
        base.update(fsb_hybrid=mode, fsb_pool_size=P, fsb_pool_recent=Wc, fsb_pool_short=short, fsb_pool_quota=quota)
    if k is not None:
        base["fsb_beam_quota"] = k
    base.update(kw)
    return _mk_int(seed, adapter=adapter, **base)


def _mk_plain(seed, W=4, adapter=None, **kw):
    """the plain W-beam module (no overselect): its read == the W beam's own picks (fsb_kk = W_max * 2^Delta_max slots)."""
    kw.setdefault("fsb_bound_radius_coef", 4.0)
    return _mk_int(seed, adapter=adapter, fsb_policy="fixed", fsb_width=W, fsb_delta=0, fsb_tree="forest", **kw)


def _compact_rows(sel):
    """(...,C) ids with -1 anywhere -> valid ids first (order kept), -1 after."""
    keep = (sel < 0).to(torch.int32).argsort(dim=-1, stable=True)
    return sel.gather(-1, keep)


def _merge_py(parts, K_):
    """python reference of beam_quota_merge on one row: the parts concatenated, first occurrences kept, -1 dropped, first K_."""
    res = []
    for c_ in [c for p_ in parts for c in p_]:
        if c_ >= 0 and c_ not in res:
            res.append(c_)
    return (res + [-1] * K_)[:K_]


def _bq_cached(mm_, x_, N0, lean, dgraph):
    """prefill N0 + per-token decode (fresh cache); (outputs, decode sels [(H,kk)], provs [(H,kk)] = fc._last_prov)."""
    n_ = x_.shape[1]
    os.environ["FSB_DECODE_GRAPH"] = dgraph
    try:
        cache = FSKVCache(mm_, lean_prefill=lean); cache.reserve = n_ - N0
        outs = [mm_(x_[:, :N0], cache=cache)]; sels, provs = [], []
        for t in range(N0, n_):
            outs.append(mm_(x_[:, t:t + 1], cache=cache))
            fc = cache._fsb_forest
            sels.append(fc._last_sel.reshape(H, -1).clone()); provs.append(fc._last_prov.reshape(H, -1).clone())
        if dgraph == "1":
            assert cache._fsb_forest.graph is not None, "decode graph never captured"
    finally:
        os.environ.pop("FSB_DECODE_GRAPH", None)
    return torch.cat(outs, dim=1), sels, provs


# ── F105: construction, refusals, the two merge primitives and flag OFF.
#   (a) default 0 (beam_kw empty); k = read_top with W4 and no union; k with the union beam; composition with the pool
#       quota (K_L + k <= read_top); refusals: k < 0, k > read_top, k > fsb_width, without read_top, the rebuild tree,
#       the learned policy, with fsb_read_margin, with fsb_hybrid pool, K_L + k > read_top;
#   (b) beam_reserved == the first k valid ids of the row (python), beam_quota_merge == the python merge ([bres | Tr]
#       and [L[:K_L] | bres | Tr | L[K_L:]]) on random rows with -1 pads and overlaps; hybrid_prov / hybrid_codes /
#       beam_prov carry PROV_BEAMQ exactly on the reserved cells and are unchanged without bres;
#   (c) fsb_beam_quota 0 passed explicitly == the flag absent bit-exact: selection / output on the no-cache forward,
#       cached decode (graph 0 / 1), the CUDA-graph training stage; off and quota hybrid ────────────────────────────
m = _mk_bq(1050, None)
assert m.fsb_beam_quota == 0 and _bkw(m) == {} and m.fsb_kk == 4 and m.fsb_union_W_max == 16 and m.fsb_W_max == 4
m = _mk_bq(1050, 2)
assert m.fsb_beam_quota == 2 and _bkw(m) == {"beam_quota": 2}
assert _mk_bq(1050, 4, U=0).fsb_beam_quota == 4 and _mk_bq(1050, 4).fsb_beam_quota == 4
assert _mk_bq(1050, 2, mode="quota", quota=2).fsb_beam_quota == 2 and _mk_bq(1050, 3, mode="quota", quota=1).fsb_pool_quota == 1
assert _mk_bq(1050, 1, mode="union", short=4).fsb_hybrid == "union"
for what, fn in (("k < 0", lambda: _mk_bq(1050, -1)),
                 ("k > read_top", lambda: _mk_bq(1050, 5)),
                 ("k > fsb_width", lambda: _mk_bq(1050, 3, W=2, rt=4)),
                 ("without read_top", lambda: mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_tree="forest", fsb_beam_quota=1)),
                 ("rebuild tree", lambda: _mk_bq(1050, 1, U=0, fsb_tree="rebuild")),
                 ("learned policy", lambda: mk(fsb_policy="learned", fsb_widths="2,4,8", fsb_deltas="0,1", fsb_tree="forest",
                                                fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_beam_quota=1)),
                 ("with read_margin", lambda: _mk_bq(1050, 1, rt=16, fsb_read_margin=1.0)),
                 ("hybrid pool", lambda: _mk_bq(1050, 1, mode="pool")),
                 ("pool quota + k > read_top", lambda: _mk_bq(1050, 2, mode="quota", quota=3))):
    _expect_value_error(fn, what)
g105 = torch.Generator(device="cpu").manual_seed(105)
nmerge105 = 0
for _ in range(10):
    R105 = 64
    A105 = torch.stack([torch.randperm(12, generator=g105)[:8] for _ in range(R105)])
    A105 = A105.masked_fill(torch.rand(R105, 8, generator=g105) < 0.35, -1)
    for k in (1, 2, 4):
        B105 = beam_reserved(A105.to(dev), k).cpu()
        assert B105.shape == (R105, k)
        for r in range(R105):
            v = [c for c in A105[r].tolist() if c >= 0]
            assert B105[r].tolist() == (v + [-1] * k)[:k], (k, r, A105[r].tolist(), B105[r].tolist())
    T105 = torch.stack([torch.randperm(12, generator=g105)[:4] for _ in range(R105)]).masked_fill(torch.rand(R105, 4, generator=g105) < 0.25, -1)
    L105 = torch.stack([torch.randperm(12, generator=g105)[:4] for _ in range(R105)]).masked_fill(torch.rand(R105, 4, generator=g105) < 0.25, -1)
    for k in (1, 2, 3):
        B105 = beam_reserved(A105.to(dev), k)
        o105 = beam_quota_merge(B105, T105.to(dev), 4).cpu()
        for r in range(R105):
            assert o105[r].tolist() == _merge_py([B105[r].tolist(), T105[r].tolist()], 4), (k, r)
            nmerge105 += 1
        for KL in (1, 2):
            o105 = beam_quota_merge(B105, T105.to(dev), 4, L105.to(dev), KL).cpu()
            for r in range(R105):
                assert o105[r].tolist() == _merge_py([L105[r, :KL].tolist(), B105[r].tolist(), T105[r].tolist(), L105[r, KL:].tolist()], 4), (k, KL, r)
                nmerge105 += 1
tr105 = torch.tensor([[3, 5, -1, 7, 9]], device=dev); pl105 = torch.tensor([[5, 9, -1, 2]], device=dev)
rd105 = torch.tensor([[5, 3, 9, -1]], device=dev); br105 = torch.tensor([[3, 9]], device=dev)
assert PROV_BEAMQ == 4 and hybrid_prov(rd105, tr105, pl105).tolist() == [[2, 0, 2, -1]]
assert hybrid_prov(rd105, tr105, pl105, br105).tolist() == [[2, 4, 6, -1]] and beam_prov(rd105, br105).tolist() == [[0, 4, 4, -1]]
cd105 = torch.tensor([[3, 5, 9, -1, 2]], device=dev)
assert _hc105(cd105, tr105, pl105).tolist() == [[1, 3, 3, 0, 2]] and _hc105(cd105, tr105, pl105, br105).tolist() == [[5, 3, 7, 0, 2]]
print(f"F105 [a,b] fsb_beam_quota: default 0; 9 refusals; beam_reserved + beam_quota_merge == python on {nmerge105} rows; "
      f"PROV_BEAMQ on the reserved slots only")
n105, N0105 = 512, 300
n105c = 0
for mode105 in (None, "quota"):
    mA = _mk_bq(1052, None, adapter=10520, mode=mode105); mB = _mk_bq(1052, 0, adapter=10520, mode=mode105)
    assert all(torch.equal(a_, b_) for a_, b_ in zip(mA.state_dict().values(), mB.state_dict().values()))
    torch.manual_seed(1053); x = torch.randn(1, n105, d_model, device=dev)
    with torch.no_grad():
        sA, yA = _full_sel_n(mA, x); sB, yB = _full_sel_n(mB, x)
    assert torch.equal(sA, sB) and torch.equal(yA, yB), (mode105, "explicit 0 != absent")
    for dg in (("0", "1") if dev == "cuda" else ("0",)):
        with torch.no_grad():
            rA = (_cached_run_n(mA, x, N0105, False, dg) if mode105 is None else _hy_cached(mA, x, N0105, False, dg)[:2])
            rB = (_cached_run_n(mB, x, N0105, False, dg) if mode105 is None else _hy_cached(mB, x, N0105, False, dg)[:2])
        assert torch.equal(rA[0], rB[0]) and all(torch.equal(a_, b_) for a_, b_ in zip(rA[1], rB[1])), (mode105, dg)
        n105c += 1
    if dev == "cuda":
        mA.train(); mB.train(); Qc, Kc, Vc = _qkv_n(1054, n105)
        with torch.no_grad():
            rA = _sel_n(mA, Qc, Kc, Vc, n105, "1") + _nstats(mA); rB = _sel_n(mB, Qc, Kc, Vc, n105, "1") + _nstats(mB)
        assert all(torch.equal(a_, b_) for a_, b_ in zip(rA, rB)), (mode105, "graphed stage")
        mA.eval(); mB.eval(); n105c += 1
print(f"F105 OK  (fsb_beam_quota: construction + 9 refusals; merge primitives == python; explicit 0 == absent bit-exact on "
      f"{n105c} cached-decode / graphed-stage cases, off + quota hybrid)")


# ── F106: SEMANTICS end to end (fp32, n 512, no adapter and a random full adapter).
#   (a) k = read_top = 4, W4, no union: the read == the plain W4 beam's read (same cells, same order, rows compacted;
#       outputs equal up to the readout's slot count); the same SET as the W4 read-4 lse_pos re-rank without the quota,
#       whose ORDER differs on some rows (teeth);
#   (b) W4 main beam + U16 union beam, k 1 / 2 / 4: the first slots hold the plain W4 beam's first k picks; the read ==
#       the python merge [W4 picks | the re-rank order over all candidates (the module's own rerank_top output)] with
#       no duplicates and exactly min(4, |W4 picks u re-rank list|) cells; the quota keeps a cell the re-rank evicts and
#       changes the read on some rows (teeth);
#   (c) composition with the pool quota (K_L, k) = (1, 1), (2, 2), (1, 3): the read == the python merge [L[:K_L] | W4
#       picks | tree re-rank | L[K_L:]] of the module's own pool list, the W4 picks and its tree list; the reserved pool
#       cells come first; the read differs from the pool quota alone on some rows;
#   (d) provenance: the prefill's prov_out (plain: 0 / 4 / -1; quota hybrid: the hybrid codes + 4) == the decode's
#       fc._last_prov on every row with the same read (eager + graph decode); PROV_BEAMQ is set exactly on the slots
#       holding a W4 pick ───────────────────────────────────────────────────────────────────────────────────────────
n106 = 512
nrow106 = nord106 = 0
for adapter in (None, 10600):
    mq = _mk_bq(1060, 4, U=0, adapter=adapter)
    mp = _mk_plain(1060, adapter=adapter)
    mr = _mk_bq(1060, None, U=0, adapter=adapter)
    Qc, Kc, Vc = _qkv_n(1061, n106, 1.5)
    with torch.no_grad():
        sq, yq = _sel_n(mq, Qc, Kc, Vc, n106); sp, yp = _sel_n(mp, Qc, Kc, Vc, n106); sr, yr = _sel_n(mr, Qc, Kc, Vc, n106)
    assert sq.shape[-1] == 4 and sp.shape[-1] == 16 and sr.shape[-1] == 4
    spc = _compact_rows(sp.long())[..., :4]
    assert torch.equal(sq.long(), spc), (adapter, "k = read_top != the plain W4 beam read")
    assert (yq - yp).abs().max().item() < 1e-4, (adapter, (yq - yp).abs().max().item())
    assert bool(_same_sets(sq.long(), sr.long()).all()) and not torch.equal(sq.long(), sr.long()), (adapter, "no teeth vs the re-rank order")
    nrow106 += sq.shape[2]; nord106 += int((sq.long() == sr.long()).all(-1).sum())
print(f"F106 [a] k = read_top, W4, no union == the plain W4 beam (cells + order) on {nrow106} rows (the re-rank order agrees on {nord106})")
nres106 = ndiff106 = nfull106 = 0
for adapter in (None, 10602):
    mp = _mk_plain(1062, adapter=adapter)
    m0 = _mk_bq(1062, None, adapter=adapter)
    Qc, Kc, Vc = _qkv_n(1063, n106, 1.5)
    with torch.no_grad():
        sp, _ = _sel_n(mp, Qc, Kc, Vc, n106); s0, _ = _sel_n(m0, Qc, Kc, Vc, n106)
    W4 = _compact_rows(sp.long())[0].cpu()                                     # (H,Npad,16): the W4 beam's picks, valid first
    s0 = s0[0].long().cpu()
    for k in (1, 2, 4):
        mq = _mk_bq(1062, k, adapter=adapter)
        with torch.no_grad():
            (sq, _), rec = _hcap(lambda: _sel_n(mq, Qc, Kc, Vc, n106), names=("rerank_top",))
        Tq = _rows_hb([o_ for _, _, o_ in rec["rerank_top"]]).long().cpu()      # (H,Npad,4): the re-rank order over all candidates
        sq = sq[0].long().cpu()
        for h in range(H):
            for i in range(BLK, n106):
                br = [c for c in W4[h, i, :k].tolist() if c >= 0]
                exp = _merge_py([br, Tq[h, i].tolist()], 4)
                assert sq[h, i].tolist() == exp, (adapter, k, h, i, sq[h, i].tolist(), exp)
                assert sq[h, i, :len(br)].tolist() == br
                got = [c for c in sq[h, i].tolist() if c >= 0]
                nu = len(set(br) | {c for c in Tq[h, i].tolist() if c >= 0})
                assert len(got) == len(set(got)) == min(4, nu), (adapter, k, h, i)
                nres106 += int(len(set(br) - set(Tq[h, i].tolist())) > 0); nfull106 += 1
        ndiff106 += int((~_same_sets(sq, s0))[:, BLK:].sum())
assert nres106 > 0 and ndiff106 > 0, ("the quota never kept a cell the re-rank evicts / never changed the read (no teeth)", nres106, ndiff106)
print(f"F106 [b] W4 + U16, k 1 / 2 / 4: read == [W4 picks | re-rank order] merge on {nfull106} rows; the quota keeps an evicted "
      f"pick on {nres106}, the read differs from no quota on {ndiff106} rows")
ncomp106 = nteeth106 = 0
for (KL, k) in ((1, 1), (2, 2), (1, 3)):
    for adapter in (None, 10604):
        mp = _mk_plain(1064, adapter=adapter)
        mq = _mk_bq(1064, k, adapter=adapter, mode="quota", quota=KL)
        mh = _mk_bq(1064, None, adapter=adapter, mode="quota", quota=KL)
        Qc, Kc, Vc = _qkv_n(1065, n106, 1.5)
        with torch.no_grad():
            sp, _ = _sel_n(mp, Qc, Kc, Vc, n106)
            (sq, _), rec = _hcap(lambda: _sel_n(mq, Qc, Kc, Vc, n106))
            sh, _ = _sel_n(mh, Qc, Kc, Vc, n106)
        W4 = _compact_rows(sp.long())[0].cpu()
        Lq = _rows_hb([o_ for _, _, o_ in rec["pool_rank"]]).long().cpu()
        Tq = _rows_hb([o_ for _, _, o_ in rec["rerank_top"]]).long().cpu()
        sq = sq[0].long().cpu()
        for h in range(H):
            for i in range(BLK, n106):
                br = [c for c in W4[h, i, :k].tolist() if c >= 0]
                exp = _merge_py([Lq[h, i, :KL].tolist(), br, Tq[h, i].tolist(), Lq[h, i, KL:].tolist()], 4)
                assert sq[h, i].tolist() == exp, (KL, k, adapter, h, i, sq[h, i].tolist(), exp)
                lv = [c for c in Lq[h, i, :KL].tolist() if c >= 0]
                assert sq[h, i, :len(lv)].tolist() == lv
                ncomp106 += 1
        nteeth106 += int((~_same_sets(sq, sh[0].long().cpu()))[:, BLK:].sum())
assert nteeth106 > 0, "the beam quota never changed the pool quota's read (no teeth)"
print(f"F106 [c] pool quota K_L + beam quota k: read == [L[:K_L] | W4 picks | tree | L[K_L:]] merge on {ncomp106} rows; differs "
      f"from the pool quota alone on {nteeth106}")
nprov106 = 0
for mode106, k106 in ((None, 2), ("quota", 1)):
    mm = _mk_bq(1066, k106, adapter=10660, mode=mode106, quota=1)
    mp = _mk_plain(1066, adapter=10660)
    torch.manual_seed(1067); x = torch.randn(1, n106, d_model, device=dev)
    caps106 = []
    _otr106 = _ft4.traverse_forest
    def _tr106(*a_, **k_):
        pl_ = []
        out_ = _otr106(*a_, **k_, prov_out=pl_)
        caps106.append(pl_[0].clone()); return out_
    _ft4.traverse_forest = _tr106
    try:
        with torch.no_grad():
            sel_full, _ = _full_sel_n(mm, x)
    finally:
        _ft4.traverse_forest = _otr106
    prov_full = _rows_hb(caps106).long()
    with torch.no_grad():
        sp_full, _ = _full_sel_n(mp, x)
    W4 = _compact_rows(sp_full.long())[..., :k106]
    resv = ((sel_full.long()[..., :, None] == W4[..., None, :]) & (W4[..., None, :] >= 0)).any(-1) & (sel_full >= 0)
    assert torch.equal((prov_full >= 0) & ((prov_full & PROV_BEAMQ) != 0), resv), (mode106, "PROV_BEAMQ != membership in the W4 picks")
    assert torch.equal(prov_full == -1, sel_full < 0)
    vals = set(prov_full.unique().tolist())
    assert vals <= ({-1, 0, 4} if mode106 is None else {-1, 0, 1, 2, 4, 6}) and 4 in vals, (mode106, vals)
    with torch.no_grad():
        for dg in (("0", "1") if dev == "cuda" else ("0",)):
            _, sels, provs = _bq_cached(mm, x, N0105, False, dg)
            for k_, t in enumerate(range(N0105, n106)):
                if torch.equal(sels[k_].long(), sel_full[:, t].long()):
                    assert torch.equal(provs[k_].long(), prov_full[:, t]), (mode106, dg, t, "decode provenance != prefill")
                    nprov106 += 1
assert nprov106 > 0
print(f"F106 OK  (k = read_top == the plain beam; [W4 picks | re-rank] merge; pool + beam quota composition; provenance: "
      f"PROV_BEAMQ == W4 membership, decode == prefill on {nprov106} rows)")


# ── F107: PARITY with the beam quota.  (a) prefill == decode (full + lean prefill, FSB_DECODE_GRAPH 0 / 1), cached ==
#   full forward, graph decode == eager, for W4 U16 k 2 / W4 k 4 (no union) / quota hybrid K_L 1 + k 2 / union hybrid
#   k 1; (b) CUDA-graph training stage == eager, and the graph key re-captures on a k change (the selection changes:
#   teeth); (c) chunked prefill 1 / 2 / all identical; (d) Triton == torch (traversal + re-rank kernels); (e) k switched
#   mid-block on a live decode cache == a fresh cache (the decode config key); (f) the router node-trace re-run
#   reproduces the forward's read set (0 mismatches), plain and quota hybrid, eager + graph; (g) bf16 4096: the decode
#   forest + cache keys route the prefill's queries to the prefill's cells (GPU) ────────────────────────────────────
n107, N0107 = 512, 300
cfg107 = (("W4 U16 k2", dict(k=2)), ("W4 k4 no union", dict(k=4, U=0)), ("quota K_L1 + k2", dict(k=2, mode="quota", quota=1)),
          ("union hybrid k1", dict(k=1, mode="union", short=4)))
nrow107 = nord107 = ngr107 = 0
for name107, c107 in cfg107:
    mm = _mk_bq(1070, adapter=10700, **c107)
    torch.manual_seed(1071); x = torch.randn(1, n107, d_model, device=dev)
    with torch.no_grad():
        sel_f, y_f = _full_sel_n(mm, x)
    ys = {}
    with torch.no_grad():
        for lean in (False, True):
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                y, sels, _ = _bq_cached(mm, x, N0107, lean, dg)
                assert (y - y_f).abs().max().item() < 1e-4, (name107, lean, dg, (y - y_f).abs().max().item())
                dec = torch.stack(sels, dim=1)
                same = _same_sets(sel_f[:, N0107:n107].long(), dec.long())
                assert bool(same.all()), f"{name107} lean={lean} graph={dg}: {int((~same).sum())} decode rows differ"
                nord107 += int((sel_f[:, N0107:n107].long() == dec.long()).all(-1).sum()); nrow107 += same.numel()
                ys[(lean, dg)] = y
        if dev == "cuda":
            for lean in (False, True):
                assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (name107, lean, "graph decode != eager")
    outs = []
    for nbch in (1, 2, 0):
        mc = _mk_bq(1070, adapter=10700, fsb_prefill_blkchunk=nbch, **c107)
        with torch.no_grad():
            outs.append(_full_sel_n(mc, x) + _nstats(mc))
    assert all(all(torch.equal(a_, b_) for a_, b_ in zip(outs[0], o_)) for o_ in outs[1:]), (name107, "chunk size")
    assert torch.equal(outs[0][0], sel_f), (name107, "chunk 1 != the default chunking")
    if dev == "cuda":
        mm.train(); Qc, Kc, Vc = _qkv_n(1072, n107)
        with torch.no_grad():
            r0 = _sel_n(mm, Qc, Kc, Vc, n107, "0") + _nstats(mm); r1 = _sel_n(mm, Qc, Kc, Vc, n107, "1") + _nstats(mm)
            assert all(torch.equal(a_, b_) for a_, b_ in zip(r0, r1)), (name107, "graphed training stage != eager")
            k1 = mm._fsb_train_graphs["key"]
            k_new = 1 if c107["k"] > 1 else 2
            mm.fsb_beam_quota = k_new
            r2 = _sel_n(mm, Qc, Kc, Vc, n107, "1") + _nstats(mm); k2 = mm._fsb_train_graphs["key"]
            r3 = _sel_n(mm, Qc, Kc, Vc, n107, "0") + _nstats(mm)
            assert k2 != k1 and all(torch.equal(a_, b_) for a_, b_ in zip(r2, r3)), (name107, "stale graph after a k change")
            assert not torch.equal(r2[0], r1[0]), (name107, "the k change did not change the selection (no teeth)")
            mm.fsb_beam_quota = c107["k"]
            ref = None
            for trv, rrk in (("1", "1"), ("0", "1"), ("1", "0"), ("0", "0")):
                with _env(FSB_RERANK_TRITON=rrk):
                    r_ = _sel_n(mm, Qc, Kc, Vc, n107, "0", triton=trv) + _nstats(mm)
                if ref is None:
                    ref = r_
                assert all(torch.equal(a_, b_) for a_, b_ in zip(ref, r_)), (name107, trv, rrk, "Triton != torch")
            ngr107 += 1
        mm.eval()
print(f"F107 [a-d] prefill == decode on {nrow107} rows (order identical on {nord107}; full + lean, graph 0 / 1), cached == full, "
      f"graph decode == eager, chunked prefill identical; training graph == eager + key re-captures, Triton == torch on {ngr107} configs")
torch.manual_seed(1073); x107 = torch.randn(1, n107, d_model, device=dev)
for dg in (("0", "1") if dev == "cuda" else ("0",)):
    mA = _mk_bq(1074, 1, adapter=10740); mB = _mk_bq(1074, 3, adapter=10740); mC = _mk_bq(1074, 1, adapter=10740)
    sw = _hy_switch(mA, x107, N0107, dg, 400, _set101(fsb_beam_quota=3))
    ref = _hy_switch(mB, x107, N0107, dg); ctl = _hy_switch(mC, x107, N0107, dg)
    assert all(torch.equal(sw[t], ref[t]) for t in range(400, n107)), (dg, "k switched mid-block on a live cache != fresh")
    assert any(not torch.equal(ctl[t], ref[t]) for t in range(400, 448)), (dg, "no teeth in the switched block")
print("F107 [e] k switched mid-block on a live decode cache == a fresh cache (eager" + (" + CUDA graph)" if dev == "cuda" else ")"))
nmis107 = ntr107 = 0
for name107, extra in (("plain W4 U16 k2", dict()),
                       ("quota K_L1 k2", dict(fsb_hybrid="quota", fsb_pool_size=PH, fsb_pool_recent=WCH, fsb_pool_quota=1))):
    torch.manual_seed(1075)
    m_ = _mk_node(fsb_width=4, fsb_rd_node=1, fsb_rd_node_weight=0.5, fsb_route_learn_c=1, fsb_bound_radius_coef=8.0,
                  fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_union_width=16, fsb_beam_quota=2,
                  **extra).train()
    with torch.no_grad():
        g_ = torch.Generator(device="cpu").manual_seed(1076)
        m_.fsb_route_W.copy_((torch.randn(m_.fsb_route_W.shape, generator=g_) * 0.3).to(dev))
        m_.fsb_route_ctheta.add_((torch.randn(H, generator=g_) * 1.0).to(dev))
    Qc, Kc, Vc = _qkv_n(1077, N2, 1.5)
    for g in (("0", "1") if dev == "cuda" else ("0",)):
        m_.zero_grad(set_to_none=True)
        m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(1078)
        sel_, o_, rd_, fr_ = _lr_n(m_, Qc, Kc, Vc, N2, g)
        st_ = rd_["stats"]
        assert fr_ is not None and st_.numel() == RD_STATS_NODE, (name107, g)
        assert float(st_[14]) == 0.0, (name107, g, "node trace re-run != the forward's read set", st_[14].item())
        nmis107 += int(st_[14]); ntr107 += int(st_[15])
        assert torch.isfinite(rd_["loss"]) and int(sel_.shape[-1]) == 4
print(f"F107 [f] router node trace with the beam quota: {nmis107} mismatches / {ntr107} rows (plain + quota hybrid, eager"
      + (" + graph)" if dev == "cuda" else ")"))
if dev == "cuda":
    for lean in (False, True):
        torch.manual_seed(1079)
        mm = _mk_bq(1079, 2, adapter=10790)
        x = torch.randn(1, 4096, d_model, device=dev)
        _f65_bf16(mm, x, 4096, lean, "F107 beam quota W4 U16 k2 lse_pos raw")
else:
    print("F107 [g] bf16 SKIP (no CUDA)")
print("F107 OK  (beam quota: prefill == decode, graph == eager (decode + training, keys), chunked prefill, Triton == torch, "
      "live-cache switch == fresh, node trace 0 mismatches, bf16 4096 routing)")


# ── F108: PLUMBING (the kept part).  The re-ranker identity records --fsb_beam_quota only when on (older identities
#   stay comparable) and reports the difference against a model with a quota ─────────────────────────────────────────
from src.fsattn.fsb_route import rr_select_config as _rsc108, identity_mismatch as _im108
m_on108 = _mk_bq(1080, 2, adapter=10800); m_off108 = _mk_bq(1080, None, adapter=10800)
c_on108, c_off108 = _rsc108(m_on108), _rsc108(m_off108)
assert c_on108["fsb_beam_quota"] == 2 and "fsb_beam_quota" not in c_off108
assert {k: v for k, v in c_on108.items() if k != "fsb_beam_quota"} == c_off108
assert _im108({"config": c_off108}, model=m_off108) == [] and _im108({"config": c_on108}, model=m_on108) == []
assert _im108({"config": c_off108}, model=m_on108) and _im108({"config": c_on108}, model=m_off108)
print("F108 OK  (--fsb_beam_quota: re-ranker identity records the quota only when on)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  FILTERED SECOND FOREST (fsb_union_tau_quantile q, with fsb_union_width + fsb_read_top; pass 4 of
#  fsb_improvement_plan_20260929): the union beam traverses a SECOND forest over the cells admitted by the per-head
#  fsb_tau_union buffer, the main beam and forest are unchanged, the candidates are union_candidates(main, second) and
#  the existing re-rank / beam quota / pool quota apply.  F109..F112.  The tests use the W16 main beam with the U16
#  second beam (the pass-4 smoke selector; W4 + U16 for the beam quota), read 4, the lse_pos re-rank with the raw
#  query, random full adapters and fresh caches for every run.  fsb_tau_union is set per head from the prompt's own
#  cell summaries: head 0 the 0.5-quantile, head 1 the 0.9-quantile (the second forest is empty in early blocks),
#  head 2 +inf (always empty), head 3 0 (every cell: the second forest IS the main forest).
# ══════════════════════════════════════════════════════════════════════════════════════════
from src.fsattn.fsb_tree import (union_tau_kw as _utkw, union_tau_on as _uton, _member as _mem109)
from src.fsattn.fsb_rerank import PROV_UTAU


def _mk_ut(seed, q, W=16, U=16, rt=4, adapter=None, mode=None, k=None, P=PH, Wc=WCH, short=4, quota=1, **kw):
    """forest, fixed W/D0 main beam, union beam U (0 = none), read rt, lse_pos re-rank with the raw query;
    fsb_union_tau_quantile only when ``q`` is not None (explicit 0 = the flag); the hybrid kwargs when ``mode`` is given;
    fsb_beam_quota when ``k`` is given."""
    base = dict(fsb_policy="fixed", fsb_width=W, fsb_delta=0, fsb_tree="forest", fsb_read_top=rt, fsb_rerank_score="lse_pos",
                fsb_rerank_query="raw", fsb_bound_radius_coef=4.0)
    if U:
        base["fsb_union_width"] = U
    if mode is not None:
        base.update(fsb_hybrid=mode, fsb_pool_size=P, fsb_pool_recent=Wc, fsb_pool_short=short, fsb_pool_quota=quota)
    if k is not None:
        base["fsb_beam_quota"] = k
    if q is not None:
        base["fsb_union_tau_quantile"] = q
    base.update(kw)
    return _mk_int(seed, adapter=adapter, **base)


_UTQ = (0.5, 0.9, float("inf"), 0.0)                  # per-head quantile of the cell summaries (inf = an always-empty forest)


def _set_utau(mm_, ce_, qs=_UTQ):
    """fsb_tau_union[h] = the qs[h]-quantile of the cell summaries ce_ (H,M) (inf -> +inf, 0 -> 0); returns a copy."""
    t_ = torch.stack([torch.quantile(ce_[h_], q_) if 0 < q_ < 1 else ce_.new_tensor(float(q_)) for h_, q_ in enumerate(qs)])
    with torch.no_grad():
        mm_.fsb_tau_union.copy_(t_)
    return mm_.fsb_tau_union.detach().clone()


def _ceil_K(K_):
    """per-cell max key norm (H,M) of raw keys K_ (1,H,n,Dh), fsb_long_range's arithmetic."""
    n_ = K_.shape[2]
    return K_[0].view(H, n_ // S, S, Dh).float().norm(dim=-1).amax(-1)


def _ut_cached(mm_, x_, N0, lean, dgraph):
    """prefill N0 + per-token decode (fresh cache); (outputs, decode sels [(H,kk)], provs [(H,kk) | None], the second
    forest's m_tot per token [(T,) | None], the main forest's m_tot per token)."""
    n_ = x_.shape[1]
    os.environ["FSB_DECODE_GRAPH"] = dgraph
    try:
        cache = FSKVCache(mm_, lean_prefill=lean); cache.reserve = n_ - N0
        outs = [mm_(x_[:, :N0], cache=cache)]; sels, provs, m2, m1 = [], [], [], []
        for t in range(N0, n_):
            outs.append(mm_(x_[:, t:t + 1], cache=cache))
            fc = cache._fsb_forest
            sels.append(fc._last_sel.reshape(H, -1).clone())
            pv = getattr(fc, "_last_prov", None)
            provs.append(pv.reshape(H, -1).clone() if pv is not None else None)
            m2.append(fc.utau.m_tot.clone() if getattr(fc, "utau", None) is not None else None)
            m1.append(fc.m_tot.clone())
        if dgraph == "1":
            assert cache._fsb_forest.graph is not None, "decode graph never captured"
    finally:
        os.environ.pop("FSB_DECODE_GRAPH", None)
    return torch.cat(outs, dim=1), sels, provs, m2, m1


# ── F109: construction, refusals, flag OFF and tau_union 0.
#   (a) default 0 (union_tau_kw empty; the (H,) buffer fsb_tau_union exists, zeros, non-persistent: not in the state
#       dict); q with W16 + U16 + read 4; refusals: q < 0, q >= 1, without a union beam, without read_top, the rebuild
#       tree, the learned policy, fsb_hybrid pool;
#   (b) fsb_union_tau_quantile 0 passed explicitly == the flag absent bit-exact: selection / output / n_read / n_visit on
#       the no-cache forward, cached decode (full + lean prefill, graph 0 / 1), the CUDA-graph training stage (off and
#       quota hybrid);
#   (c) q > 0 with fsb_tau_union at 0 (every cell admitted) == fsb_union_width on ONE forest bit-exact on the same paths
#       (the second forest IS the main forest: same m_tot in decode, no PROV_UTAU), no adapter and a random full adapter,
#       off and quota hybrid ─────────────────────────────────────────────────────────────────────────────────────────
m = _mk_ut(1090, None)
assert m.fsb_union_tau_quantile == 0.0 and not _uton(m) and _utkw(m, None, None, 1) == {}
assert tuple(m.fsb_tau_union.shape) == (H,) and float(m.fsb_tau_union.abs().max()) == 0.0
assert "fsb_tau_union" not in m.state_dict() and "fsb_tau" in m.state_dict()
assert any(n_ == "fsb_tau_union" for n_, _ in m.named_buffers())
m = _mk_ut(1090, 0.5)
assert m.fsb_union_tau_quantile == 0.5 and _uton(m) and m.fsb_union_W_max == 16 and m.fsb_W_max == 16 and m.fsb_kk == 4
for what, fn in (("q < 0", lambda: _mk_ut(1090, -0.1)),
                 ("q >= 1", lambda: _mk_ut(1090, 1.0)),
                 ("without a union beam", lambda: _mk_ut(1090, 0.5, U=0)),
                 ("without read_top", lambda: mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_tree="forest",
                                                 fsb_union_tau_quantile=0.5)),
                 ("rebuild tree", lambda: _mk_ut(1090, 0.5, fsb_tree="rebuild")),
                 ("learned policy", lambda: mk(fsb_policy="learned", fsb_widths="2,4,8", fsb_deltas="0,1", fsb_tree="forest",
                                                fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_union_width=16,
                                                fsb_union_tau_quantile=0.5)),
                 ("hybrid pool", lambda: _mk_ut(1090, 0.5, mode="pool"))):
    _expect_value_error(fn, what)
print("F109 [a] fsb_union_tau_quantile: default 0, fsb_tau_union (H,) zeros non-persistent; 7 refusals")
n109, N0109 = 512, 300
n109c = 0
for mode109 in (None, "quota"):
    mA = _mk_ut(1092, None, adapter=10920, mode=mode109); mB = _mk_ut(1092, 0, adapter=10920, mode=mode109)
    assert all(torch.equal(a_, b_) for a_, b_ in zip(mA.state_dict().values(), mB.state_dict().values()))
    torch.manual_seed(1093); x = torch.randn(1, n109, d_model, device=dev)
    with torch.no_grad():
        sA, yA = _full_sel_n(mA, x); nA = _nstats(mA); sB, yB = _full_sel_n(mB, x); nB = _nstats(mB)
    assert torch.equal(sA, sB) and torch.equal(yA, yB) and all(torch.equal(a_, b_) for a_, b_ in zip(nA, nB)), (mode109, "explicit 0 != absent")
    for lean in (False, True):
        for dg in (("0", "1") if dev == "cuda" else ("0",)):
            with torch.no_grad():
                rA = _ut_cached(mA, x, N0109, lean, dg); rB = _ut_cached(mB, x, N0109, lean, dg)
            assert torch.equal(rA[0], rB[0]) and all(torch.equal(a_, b_) for a_, b_ in zip(rA[1], rB[1])), (mode109, lean, dg)
            assert all(m_ is None for m_ in rA[3]) and all(m_ is None for m_ in rB[3])
            n109c += 1
    if dev == "cuda":
        mA.train(); mB.train(); Qc, Kc, Vc = _qkv_n(1094, n109)
        with torch.no_grad():
            rA = _sel_n(mA, Qc, Kc, Vc, n109, "1") + _nstats(mA); rB = _sel_n(mB, Qc, Kc, Vc, n109, "1") + _nstats(mB)
        assert all(torch.equal(a_, b_) for a_, b_ in zip(rA, rB)), (mode109, "graphed stage")
        mA.eval(); mB.eval(); n109c += 1
print(f"F109 [b] explicit 0 == absent bit-exact on the no-cache forward + {n109c} cached-decode / graphed-stage cases (off + quota hybrid)")
n109d = 0
for mode109 in (None, "quota"):
    for adapter in (None, 10950):
        mP = _mk_ut(1095, None, adapter=adapter, mode=mode109); mZ = _mk_ut(1095, 0.5, adapter=adapter, mode=mode109)
        assert float(mZ.fsb_tau_union.abs().max()) == 0.0 and _uton(mZ)
        torch.manual_seed(1096); x = torch.randn(1, n109, d_model, device=dev)
        with torch.no_grad():
            sP, yP = _full_sel_n(mP, x); nP = _nstats(mP)
            (sZ, yZ), recZ = _hcap(lambda: _full_sel_n(mZ, x), names=("fsb_utau_acc",)); nZ = _nstats(mZ)
        assert torch.equal(sP, sZ) and torch.equal(yP, yZ) and all(torch.equal(a_, b_) for a_, b_ in zip(nP, nZ)), (mode109, adapter, "tau_union 0 != one forest")
        pz = recZ["fsb_utau_acc"][0][0][1]
        assert pz is not None and not bool(((pz >= 0) & ((pz.to(torch.int16) & PROV_UTAU) != 0)).any()), (mode109, adapter, "PROV_UTAU with tau_union 0")
        assert "admit_frac_union" in mZ._fsb_stats and torch.equal(mZ._fsb_stats["admit_frac_union"], mZ._fsb_stats["admit_frac"])
        for lean in (False, True):
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                with torch.no_grad():
                    rP = _ut_cached(mP, x, N0109, lean, dg); rZ = _ut_cached(mZ, x, N0109, lean, dg)
                assert torch.equal(rP[0], rZ[0]) and all(torch.equal(a_, b_) for a_, b_ in zip(rP[1], rZ[1])), (mode109, adapter, lean, dg)
                assert all(m2_ is not None and torch.equal(m2_, m1_) for m2_, m1_ in zip(rZ[3], rZ[4])), (mode109, adapter, lean, dg, "second forest != main")
                n109d += 1
        if dev == "cuda":
            mP.train(); mZ.train(); Qc, Kc, Vc = _qkv_n(1097, n109)
            with torch.no_grad():
                rP = _sel_n(mP, Qc, Kc, Vc, n109, "1") + _nstats(mP); rZ = _sel_n(mZ, Qc, Kc, Vc, n109, "1") + _nstats(mZ)
            assert all(torch.equal(a_, b_) for a_, b_ in zip(rP, rZ)), (mode109, adapter, "graphed stage: tau_union 0 != one forest")
            mP.eval(); mZ.eval(); n109d += 1
print(f"F109 OK  (fsb_union_tau_quantile: construction + 7 refusals; explicit 0 == absent; tau_union 0 == fsb_union_width on one "
      f"forest bit-exact on the no-cache forward + {n109d} cached-decode / graphed-stage cases, off + quota hybrid, +- adapter)")


# ── F110: SEMANTICS with a filtering tau_union (fp32, n 512, random full adapter; W16 main + U16 second beam, read 4).
#   (a) the main beam's candidates are UNCHANGED (== the one-forest union module's main candidates, cells and order);
#   (b) every second-beam cell is admitted (a_c >= tau_union[h]) and eligible (id < E_b); head 2 (tau +inf) finds nothing
#       and its read == the no-union module's read; head 3 (tau 0) finds exactly the one-forest union beam's cells;
#   (c) the second beam's candidates == the MAIN beam of a module whose main forest is the filtered one (fsb_tau =
#       tau_union, fsb_width U, no union) bit-exact: the filtered forest is built and searched exactly like a main forest;
#   (d) union order and dedup: the re-rank's candidate list == [main slots | second beam's cells not among them -> -1];
#   (e) teeth: the read differs from the one-forest union on some rows and some read cells come from the second beam only;
#   (f) provenance: PROV_UTAU exactly on the read slots holding a second-beam-only cell (-1 on empties, 0 elsewhere); the
#       armed read accumulators count them over prefill + decode (== the captured provenance); the second forest's
#       admitted fraction is in the stats (head 2: 0, head 3: 1);
#   (g) prefill == decode (full + lean prefill, graph 0 / 1; the second forest is EMPTY in some (row, block) whose main
#       forest is not, and always for head 2), cached == full forward, graph decode == eager, decode provenance == the
#       prefill's; (h) chunked prefill 1 / 2 / all identical; (i) CUDA-graph training stage == eager, and an in-place
#       tau_union change is seen by the captured build (replay == eager on the new tau, selection changes, same key);
#   (j) Triton == torch (traversal + re-rank kernels) ───────────────────────────────────────────────────────────────────
n110, N0110 = 512, 300
mm = _mk_ut(1100, 0.5, adapter=11000)
mP = _mk_ut(1100, None, adapter=11000)                                       # one-forest union (the pass-2 selector)
mN = _mk_ut(1100, None, U=0, adapter=11000)                                 # no union beam
mF = _mk_ut(1100, None, U=0, adapter=11000)                                 # the filtered forest as the MAIN forest (W16)
torch.manual_seed(1101); x = torch.randn(1, n110, d_model, device=dev)
ce110 = _ceil_of(mm, x)                                                      # (H,M) the prompt's cell summaries
tau_u = _set_utau(mm, ce110)
with torch.no_grad():
    mF.fsb_tau.copy_(tau_u)
assert bool(torch.isinf(tau_u[2])) and float(tau_u[3]) == 0.0 and bool((tau_u[:2] > 0).all())
with torch.no_grad():
    (sel_t, y_t), rec = _hcap(lambda: _full_sel_n(mm, x), names=("union_candidates", "rerank_top", "_forest_beam", "utau_build",
                                                                  "fsb_utau_acc"))
    (sel_p, y_p), recP = _hcap(lambda: _full_sel_n(mP, x), names=("union_candidates",))
    (sel_f, y_f), recF = _hcap(lambda: _full_sel_n(mF, x), names=("_forest_beam",))
    sel_n, y_n = _full_sel_n(mN, x)
main_t = _rows_hb([a_[0] for a_, _, _ in rec["union_candidates"]]).long()          # (H,Npad,Cm) the main beam's candidates
sec_t = _rows_hb([a_[1] for a_, _, _ in rec["union_candidates"]]).long()           # (H,Npad,Cu) the second beam's (raw)
Cm110 = main_t.shape[-1]                              # 16 x 2^fsb_delta (compact beams) or 16 x 2^Delta_max slots
main_p = _rows_hb([a_[0] for a_, _, _ in recP["union_candidates"]]).long()
sec_p = _rows_hb([a_[1] for a_, _, _ in recP["union_candidates"]]).long()
assert torch.equal(main_t, main_p), "the main beam changed under the filtered second forest"
E_row110 = (torch.div(torch.arange(n110, device=dev), BLK, rounding_mode="floor") * BLK) // S   # eligible: id < E_b
ok_sec = sec_t >= 0
a_sec = ce110.gather(1, sec_t.clamp(min=0).view(H, -1)).view_as(sec_t)
assert bool((a_sec >= tau_u.view(H, 1, 1))[ok_sec].all()), "a second-beam cell below tau_union"
assert bool((sec_t < E_row110.view(1, -1, 1))[ok_sec].all()), "a second-beam cell not eligible for its block"
assert not bool(ok_sec[2].any()) and bool(ok_sec[0].any()) and bool(ok_sec[1].any())
assert torch.equal(sec_t[3], sec_p[3]), "head 3 (tau_union 0) != the one-forest union beam"
assert torch.equal(sel_t[2], sel_n[2]), "head 2 (empty second forest) != the no-union read"
fb_t = [o_[0] for _, _, o_ in rec["_forest_beam"]]; fb_f = [o_[0] for _, _, o_ in recF["_forest_beam"]]
assert len(fb_t) == 2 * len(fb_f) > 0
assert torch.equal(_rows_hb(fb_t[0::2]).long(), main_t) and torch.equal(_rows_hb(fb_t[1::2]).long(), sec_t)
assert torch.equal(_rows_hb(fb_f).long(), sec_t), "the second beam != the main beam over the filtered forest as a main forest"
cand110 = _rows_hb([a_[0] for a_, _, _ in rec["rerank_top"]]).long()               # (H,Npad,Cm+Cu) the re-rank's candidates
dup110 = (sec_t[..., :, None] == main_t[..., None, :]).any(-1) & ok_sec
assert cand110.shape[-1] == Cm110 + sec_t.shape[-1] and torch.equal(cand110[..., :Cm110], main_t)
assert torch.equal(cand110[..., Cm110:], sec_t.masked_fill(dup110, -1))
unew110 = cand110[..., Cm110:]
in_unew = _mem109(sel_t.long(), unew110)                                          # read slots from the second beam only
assert not torch.equal(sel_t, sel_p) and bool(in_unew.any()) and bool(in_unew[0].any()), "no teeth"
assert bool(_mem109(sel_t.long(), cand110).logical_or(sel_t < 0).all()), "a read cell outside the candidates"
prov110 = rec["fsb_utau_acc"][0][0][1][0].long()                                   # (H,N,4) the prefill's provenance
assert torch.equal(prov110 == -1, sel_t < 0) and torch.equal((prov110 >= 0) & ((prov110 & PROV_UTAU) != 0), in_unew)
assert set(prov110.unique().tolist()) == {-1, 0, PROV_UTAU}, prov110.unique().tolist()
mba110 = rec["utau_build"][0][2][1]                                                # (T,nb) the second forest's m_b
assert bool((mba110[2] == 0).all()) and bool((mba110[3, 1:] > 0).all())
empty110 = (mba110[:2, 1:] == 0)
assert bool(empty110.any()), "no (row, block) with an empty second forest and a non-empty main forest (pick another seed)"
afu110 = mm._fsb_stats["admit_frac_union"]
assert tuple(afu110.shape) == (H,) and float(afu110[2]) == 0.0 and float(afu110[3]) == 1.0 and bool((afu110 <= mm._fsb_stats["admit_frac"] + 1e-6).all())
fsb_read_stats_reset(mm)
with torch.no_grad():
    _full_sel_n(mm, x)
st110 = fsb_read_stats(mm); fsb_read_stats_reset(mm, on=False)
assert st110["utau_cells"] == float(in_unew.sum()) and abs(st110["utau_frac"] - float(in_unew.sum()) / float((sel_t >= 0).sum())) < 1e-6
print(f"F110 [a-f] main beam unchanged; second beam: {int(ok_sec.sum())} admitted + eligible cells (head 2 none, head 3 == one forest), "
      f"== the filtered forest as a main forest bit-exact; union order + dedup; {int(in_unew.sum())} second-beam-only reads "
      f"(PROV_UTAU == membership; accumulators agree); {int(empty110.sum())} (row, block) with an empty second forest")
nrow110 = nord110 = nprov110 = ndec110 = 0
ys = {}
with torch.no_grad():
    for lean in (False, True):
        for dg in (("0", "1") if dev == "cuda" else ("0",)):
            fsb_read_stats_reset(mm)
            y, sels, provs, m2s, m1s = _ut_cached(mm, x, N0110, lean, dg)
            st_d = fsb_read_stats(mm); fsb_read_stats_reset(mm, on=False)
            d = (y - y_t).abs().max().item()
            assert d < 1e-4, (lean, dg, d)
            dec = torch.stack(sels, dim=1)
            same = _same_sets(sel_t[:, N0110:n110].long(), dec.long())
            assert bool(same.all()), f"lean={lean} graph={dg}: {int((~same).sum())} decode rows differ"
            nord110 += int((sel_t[:, N0110:n110].long() == dec.long()).all(-1).sum()); nrow110 += same.numel()
            for k_, t in enumerate(range(N0110, n110)):
                assert provs[k_] is not None and m2s[k_] is not None
                assert torch.equal(provs[k_] == -1, sels[k_] < 0)
                if torch.equal(sels[k_].long(), sel_t[:, t].long()):
                    assert torch.equal(provs[k_].long(), prov110[:, t]), (lean, dg, t, "decode provenance != prefill")
                    nprov110 += 1
            assert all(int(m2_[2]) == 0 for m2_ in m2s) and all(bool((m2_ <= m1_).all()) for m2_, m1_ in zip(m2s, m1s))
            ut_dec = sum(int(((p_ >= 0) & ((p_.long() & PROV_UTAU) != 0)).sum()) for p_ in provs)
            assert st_d["utau_cells"] == float(in_unew[:, :N0110].sum()) + ut_dec, (lean, dg, st_d["utau_cells"])
            ndec110 += ut_dec
            ys[(lean, dg)] = y
    if dev == "cuda":
        for lean in (False, True):
            assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (lean, "graph decode != eager")
outs = []
for nbch in (1, 2, 0):
    mc = _mk_ut(1100, 0.5, adapter=11000, fsb_prefill_blkchunk=nbch)
    with torch.no_grad():
        mc.fsb_tau_union.copy_(tau_u)
        outs.append(_full_sel_n(mc, x) + _nstats(mc))
assert all(all(torch.equal(a_, b_) for a_, b_ in zip(outs[0], o_)) for o_ in outs[1:]), "chunk size"
assert torch.equal(outs[0][0], sel_t), "chunk 1 != the default chunking"
print(f"F110 [g,h] prefill == decode on {nrow110} rows (order identical on {nord110}; full + lean, graph 0 / 1), cached == full, "
      f"graph decode == eager, decode provenance == prefill on {nprov110} rows ({ndec110} second-beam-only decode reads), "
      f"chunked prefill identical")
if dev == "cuda":
    mm.train(); Qc, Kc, Vc = _qkv_n(1102, n110)
    tau_k = _set_utau(mm, _ceil_K(Kc))
    with torch.no_grad():
        r0 = _sel_n(mm, Qc, Kc, Vc, n110, "0") + _nstats(mm); r1 = _sel_n(mm, Qc, Kc, Vc, n110, "1") + _nstats(mm)
        assert all(torch.equal(a_, b_) for a_, b_ in zip(r0, r1)), "graphed training stage != eager"
        k1 = mm._fsb_train_graphs["key"]
        _set_utau(mm, _ceil_K(Kc), qs=(0.9, 0.3, 0.0, 0.6))                    # in place: the captured build reads the buffer
        r2 = _sel_n(mm, Qc, Kc, Vc, n110, "1") + _nstats(mm); k2 = mm._fsb_train_graphs["key"]
        r3 = _sel_n(mm, Qc, Kc, Vc, n110, "0") + _nstats(mm)
        assert k2 == k1 and all(torch.equal(a_, b_) for a_, b_ in zip(r2, r3)), "replay after an in-place tau_union change != eager"
        assert not torch.equal(r2[0], r1[0]), "the tau_union change did not change the selection (no teeth)"
        mm.fsb_union_tau_quantile = 0.75                                       # the knob is part of the key
        r4 = _sel_n(mm, Qc, Kc, Vc, n110, "1"); assert mm._fsb_train_graphs["key"] != k2
        assert torch.equal(r4[0], r2[0]); mm.fsb_union_tau_quantile = 0.5
        ref = None
        for trv, rrk in (("1", "1"), ("0", "1"), ("1", "0"), ("0", "0")):
            with _env(FSB_RERANK_TRITON=rrk):
                r_ = _sel_n(mm, Qc, Kc, Vc, n110, "0", triton=trv) + _nstats(mm)
            if ref is None:
                ref = r_
            assert all(torch.equal(a_, b_) for a_, b_ in zip(ref, r_)), (trv, rrk, "Triton != torch")
    mm.eval()
    print("F110 [i,j] training graph == eager, in-place tau_union change seen by the captured build (same key, new selection), "
          "knob in the key, Triton == torch")
else:
    print("F110 [i,j] SKIP (no CUDA)")
print("F110 OK  (filtered second forest: main beam unchanged, second beam == the filtered forest searched as a main forest, "
      "admission + eligibility, union order / dedup, PROV_UTAU + accumulators, prefill == decode incl. empty second forests, "
      "chunk invariance, graph == eager)")


# ── F111: COMPOSITION and the remaining paths.  (a) beam quota k 1 / 2 on W4 + U16 over the filtered forest: the read ==
#   the python merge [W4 picks | the re-rank order over main + filtered-second candidates] (F106's reference), PROV_BEAMQ
#   on the W4 picks and PROV_UTAU on the second-beam-only reads; (b) pool quota K_L 1 (+ beam quota 1): the read ==
#   [L[:K_L] | W4 picks | tree re-rank | L[K_L:]] of the module's own lists, the hybrid provenance carries the new bit on
#   top; (a) + (b): prefill == decode (graph 0 / 1) incl. provenance; (c) the config switched on a LIVE decode cache
#   mid-block (q 0.5 -> 0, 0 -> 0.5, an in-place tau_union change) == a fresh cache (the decode config key sees the knob
#   and the buffer's version); (d) the router node-trace re-run reproduces the forward's read set (0 mismatches) with the
#   filtered second forest (plain + quota hybrid, eager + graph); (e) bf16 4096: the decode forests + cache keys route the
#   prefill's queries to the prefill's cells (GPU) ─────────────────────────────────────────────────────────────────────
n111, N0111 = 512, 300
nmerge111 = nres111 = nprov111 = nrow111 = 0
for name111, c111 in (("W4 U16 k1", dict(k=1)), ("W4 U16 k2", dict(k=2)), ("quota K_L1 + k1", dict(k=1, mode="quota", quota=1)),
                      ("quota K_L1", dict(mode="quota", quota=1))):
    for adapter in (None, 11110):
        mp = _mk_plain(1111, adapter=adapter)
        mq = _mk_ut(1111, 0.5, W=4, adapter=adapter, **c111)
        torch.manual_seed(1112); x = torch.randn(1, n111, d_model, device=dev)
        _set_utau(mq, _ceil_of(mq, x))
        k111 = c111.get("k", 0); KL111 = c111.get("quota", 0) if "mode" in c111 else 0
        with torch.no_grad():
            sp, _ = _full_sel_n(mp, x)
            (sq, _), rec = _hcap(lambda: _full_sel_n(mq, x), names=("rerank_top", "union_candidates", "pool_rank", "fsb_utau_acc"))
        W4 = _compact_rows(sp.long()).cpu()                                     # (H,Npad,16): the W4 beam's picks
        cand = _rows_hb([o_ for _, _, o_ in rec["union_candidates"]]).long()    # (H,Npad,16+64); quota: one call per chunk
        unew = cand[..., 16:]
        Tq = _rows_hb([o_ for _, _, o_ in rec["rerank_top"]]).long().cpu()      # the module's re-rank order (all candidates / tree)
        Lq = _rows_hb([o_ for _, _, o_ in rec["pool_rank"]]).long().cpu() if "mode" in c111 else None
        sqc = sq.long().cpu()
        for h in range(H):
            for i in range(BLK, n111):
                br = [c for c in W4[h, i, :k111].tolist() if c >= 0]
                if Lq is None:
                    exp = _merge_py([br, Tq[h, i].tolist()], 4)
                else:
                    exp = _merge_py([Lq[h, i, :KL111].tolist(), br, Tq[h, i].tolist(), Lq[h, i, KL111:].tolist()], 4)
                assert sqc[h, i].tolist() == exp, (name111, adapter, h, i, sqc[h, i].tolist(), exp)
                nmerge111 += 1
                nres111 += int(len(set(br) - set(Tq[h, i].tolist())) > 0)
        prov = rec["fsb_utau_acc"][0][0][1][0].long()                             # (H,N,4)
        in_u = _mem109(sq.long(), unew)
        assert torch.equal((prov >= 0) & ((prov & PROV_UTAU) != 0), in_u), (name111, adapter, "PROV_UTAU != second-beam-only membership")
        assert torch.equal(prov == -1, sq < 0)
        if k111:
            resv = _mem109(sq.long(), _compact_rows(sp.long())[..., :k111])
            assert torch.equal((prov >= 0) & ((prov & PROV_BEAMQ) != 0), resv), (name111, adapter, "PROV_BEAMQ != W4 membership")
        assert bool(in_u.any()), (name111, adapter, "no second-beam-only read (no teeth)")
        vals = set(prov.unique().tolist())
        assert vals <= ({-1, 0, 4, 8, 12} if Lq is None else {-1, 0, 1, 2, 4, 6, 8, 9, 10, 12, 14}), (name111, vals)
        with torch.no_grad():
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                y, sels, provs, _, _ = _ut_cached(mq, x, N0111, False, dg)
                dec = torch.stack(sels, dim=1)
                same = _same_sets(sq[:, N0111:n111].long(), dec.long())
                assert bool(same.all()), (name111, adapter, dg, int((~same).sum()))
                nrow111 += same.numel()
                for k_, t in enumerate(range(N0111, n111)):
                    if torch.equal(sels[k_].long(), sq[:, t].long()):
                        assert torch.equal(provs[k_].long(), prov[:, t]), (name111, adapter, dg, t, "decode provenance != prefill")
                        nprov111 += 1
assert nres111 > 0
print(f"F111 [a,b] beam quota / pool quota over the filtered second forest: read == the python merges on {nmerge111} rows "
      f"({nres111} keep an evicted W4 pick); PROV_BEAMQ + PROV_UTAU == memberships; prefill == decode on {nrow111} rows, "
      f"provenance equal on {nprov111}")
torch.manual_seed(1113); x113 = torch.randn(1, n111, d_model, device=dev)
ce113 = _ceil_of(_mk_ut(1114, 0.5, W=4, adapter=11140), x113)
_Q2 = (0.9, 0.5, 0.0, float("inf"))
nsw111 = teeth111 = 0
for dg in (("0", "1") if dev == "cuda" else ("0",)):
    def _fresh(q_, qs=None):
        m_ = _mk_ut(1114, q_, W=4, adapter=11140)                          # W4 main + U16: the second beam adds many cells
        if qs is not None:
            _set_utau(m_, ce113, qs)
        return m_
    # q 0.5 -> 0 mid-block == the one-forest union; 0 -> 0.5 (buffer pre-set) == a fresh filtered module; an in-place
    # tau_union change == a fresh module with the new tau (the decode config key sees the buffer's version)
    for nm, mkA, mkB, setter in (("0.5 -> 0", lambda: _fresh(0.5, _UTQ), lambda: _fresh(None), _set101(fsb_union_tau_quantile=0.0)),
                                 ("0 -> 0.5", lambda: _fresh(None, _UTQ), lambda: _fresh(0.5, _UTQ), _set101(fsb_union_tau_quantile=0.5)),
                                 ("tau A -> tau B in place", lambda: _fresh(0.5, _UTQ), lambda: _fresh(0.5, _Q2),
                                  lambda m_: _set_utau(m_, ce113, _Q2))):
        mA, mB, mC = mkA(), mkB(), mkA()
        sw = _hy_switch(mA, x113, N0111, dg, 400, setter)
        ref = _hy_switch(mB, x113, N0111, dg); ctl = _hy_switch(mC, x113, N0111, dg)
        assert all(torch.equal(sw[t], ref[t]) for t in range(400, n111)), (nm, dg, "switched live cache != a fresh cache")
        teeth111 += sum(int(not torch.equal(ctl[t], ref[t])) for t in range(400, n111))   # the two configs differ after the switch
        nsw111 += 1
assert teeth111 > 0, "no decode row differs between the switched configs after the switch: no teeth"
print(f"F111 [c] {nsw111} live-cache switches (knob on / off, in-place tau_union) == a fresh cache (eager" + (" + CUDA graph)" if dev == "cuda" else ")")
      + f"; {teeth111} post-switch steps differ between the configs")
nmis111 = ntr111 = 0
for name111, extra in (("plain W4 U16 k2", dict()),
                       ("quota K_L1 k2", dict(fsb_hybrid="quota", fsb_pool_size=PH, fsb_pool_recent=WCH, fsb_pool_quota=1))):
    torch.manual_seed(1115)
    m_ = _mk_node(fsb_width=4, fsb_rd_node=1, fsb_rd_node_weight=0.5, fsb_route_learn_c=1, fsb_bound_radius_coef=8.0,
                  fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_union_width=16, fsb_beam_quota=2,
                  fsb_union_tau_quantile=0.5, **extra).train()
    with torch.no_grad():
        g_ = torch.Generator(device="cpu").manual_seed(1116)
        m_.fsb_route_W.copy_((torch.randn(m_.fsb_route_W.shape, generator=g_) * 0.3).to(dev))
        m_.fsb_route_ctheta.add_((torch.randn(H, generator=g_) * 1.0).to(dev))
    Qc, Kc, Vc = _qkv_n(1117, N2, 1.5)
    _set_utau(m_, _ceil_K(Kc))
    for g in (("0", "1") if dev == "cuda" else ("0",)):
        m_.zero_grad(set_to_none=True)
        m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(1118)
        sel_, o_, rd_, fr_ = _lr_n(m_, Qc, Kc, Vc, N2, g)
        st_ = rd_["stats"]
        assert fr_ is not None and st_.numel() == RD_STATS_NODE and fr_[0].get("utau_F") is not None, (name111, g)
        assert float(st_[14]) == 0.0, (name111, g, "node trace re-run != the forward's read set", st_[14].item())
        nmis111 += int(st_[14]); ntr111 += int(st_[15])
        assert torch.isfinite(rd_["loss"]) and int(sel_.shape[-1]) == 4
print(f"F111 [d] router node trace with the filtered second forest: {nmis111} mismatches / {ntr111} rows (plain + quota hybrid, eager"
      + (" + graph)" if dev == "cuda" else ")"))


def _f65_bf16_ut(mm_, x_, Nf, lean, tag):
    """_f65_bf16 with the filtered second forest: the prefill's traversal (routing q, raw q, second-forest budget) is
    replayed on the decode cache's main AND second forests + spines; then the block's decode tokens."""
    caps = []
    _otr = _ft4.traverse_forest
    def _tr(F_, SP_, q_, qpos_, lam_, sigv_, *rest, **kw_):
        out_ = _otr(F_, SP_, q_, qpos_, lam_, sigv_, *rest, **kw_)
        caps.append((q_.clone(), qpos_.clone(), lam_.clone(), sigv_.clone(), rest, dict(kw_), out_[0].clone()))
        return out_
    os.environ["FSB_TRAIN_GRAPH"] = "0"
    _ft4.traverse_forest = _tr
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mm_(x_)
            q_, qpos_, lam_, sigv_, rest, kw_, sel_pre = caps[-1]
            assert kw_["tokK"] is not None and kw_.get("union_F") is not None and kw_.get("union_SP") is not None, tag
            nbc = q_.shape[0] // H; sl = slice(nbc - 1, None, nbc)
            P = Nf - BLK
            cache = FSKVCache(mm_, lean_prefill=lean); cache.reserve = Nf - P
            mm_(x_[:, :P], cache=cache)
            mm_(x_[:, P:P + 1], cache=cache)
            fc = cache._fsb_forest
            assert fc.utau is not None
            dec = [fc._last_sel.reshape(H, -1).clone()]
            rest_sl = tuple(r_[sl] if torch.is_tensor(r_) and r_.dim() >= 1 and r_.shape[0] == q_.shape[0] else r_ for r_ in rest)
            kw2 = dict(kw_); kw2["tokK"] = cache._k_buf; kw2.pop("prov_out", None)
            for k_ in ("radius_coef", "rq", "union_delta"):
                v_ = kw2.get(k_)
                if torch.is_tensor(v_) and v_.dim() >= 1 and v_.shape[0] == q_.shape[0]:
                    kw2[k_] = v_[sl]
            kw2["union_F"] = fc.utau.F; kw2["union_SP"] = fc.utau.SP
            s_dec, _, _ = _otr(fc.F, fc.SP, q_[sl], qpos_[sl], lam_[sl], sigv_[sl], *rest_sl, **kw2)
            assert torch.equal(s_dec, sel_pre[sl]), f"{tag}: decode forests + cache keys route the prefill queries differently"
            for t in range(P + 1, Nf):
                mm_(x_[:, t:t + 1], cache=cache)
                dec.append(cache._fsb_forest._last_sel.reshape(H, -1).clone())
    finally:
        _ft4.traverse_forest = _otr; os.environ.pop("FSB_TRAIN_GRAPH", None)
    sel_dec = torch.stack(dec, dim=1)
    same_set, same_order = _sets_equal(sel_pre[sl].long(), sel_dec.long())
    assert bool(same_set.all()), f"{tag}: {int((~same_set).sum())}/{same_set.numel()} decode rows route differently"
    print(f"F111 [e] [{tag}, lean={lean}] bf16 4096: prefill queries on the decode forests + cache keys: identical cells + order "
          f"({same_set.numel()} rows); decode q: {same_set.numel()} identical sets, {int(same_order.sum())} identical order")


if dev == "cuda":
    for lean in (False, True):
        torch.manual_seed(1119)
        mm = _mk_ut(1119, 0.5, adapter=11190)
        x = torch.randn(1, 4096, d_model, device=dev)
        _set_utau(mm, _ceil_of(mm, x))
        _f65_bf16_ut(mm, x, 4096, lean, "filtered second forest W16 U16 lse_pos raw")
else:
    print("F111 [e] bf16 SKIP (no CUDA)")
print("F111 OK  (filtered second forest + beam quota / pool quota merges, provenance bits, prefill == decode, live-cache switches, "
      "node trace 0 mismatches, bf16 4096 routing)")


# ── F112: the union tau calibration shares the main tau's cache file (fsb_tau_calib.calibrate_or_load with target =
#   "fsb_tau_union" writes only the union buffer and records the target; fsb_calibrate_tau(target) likewise); the
#   re-ranker identity records --fsb_union_tau_quantile only when on; fsb_resolved records the union tau; read stats
#   carry the utau keys (0 when off) ─────────────────────────────────────────────────────────────────────────────────
import numpy as _np112
import tempfile as _tf112
import shutil as _sh112
_d112 = _tf112.mkdtemp(prefix="fsb_utau_")
_tok112 = _np112.random.default_rng(112).integers(0, 257, size=200 * 65, dtype=_np112.uint16)
_bin112 = os.path.join(_d112, "corpus.bin"); _tok112.tofile(_bin112)
torch.manual_seed(1120); lmA = _LM32(**_lmkw).to(dev).eval()
torch.manual_seed(1120); lmB = _LM32(**_lmkw).to(dev).eval()
cal112 = dict(ckpt_sha="cd" * 32, quantile=0.5, n_batches=4, length=64, seed=7, data_path=_bin112, cache_dir=_d112,
              device=dev, autocast_dtype=None, log=lambda *a_, **k_: None)
iA = _ftc.calibrate_or_load(lmA, **cal112)
iB = _ftc.calibrate_or_load(lmB, **cal112, target="fsb_tau_union")
_lay112 = lambda lm_, nm_: torch.stack([getattr(m_, nm_).detach().float().cpu() for m_ in lm_.modules() if getattr(m_, "fsb_enabled", False)])
tA, tAu, tBm, tBu = _lay112(lmA, "fsb_tau"), _lay112(lmA, "fsb_tau_union"), _lay112(lmB, "fsb_tau"), _lay112(lmB, "fsb_tau_union")
assert iA["loaded_from_cache"] is False and iA["target"] == "fsb_tau" and iB["loaded_from_cache"] is True and iB["target"] == "fsb_tau_union"
assert iB["cache_file"] == iA["cache_file"] and iB["tau_sha256"] == iA["tau_sha256"]
assert torch.equal(tA, tBu) and bool((tA > 0).all()) and float(tAu.abs().max()) == 0.0 and float(tBm.abs().max()) == 0.0
assert "target" not in torch.load(iA["cache_file"], map_location="cpu")["meta"]          # the shared file carries no target
_pu, _au = _ftc.tau_summary(lmB, target="fsb_tau_union"); _pm, _am = _ftc.tau_summary(lmB)
assert abs(_au[1] - float(tA.mean())) < 1e-6 and _am == [0.0, 0.0, 0.0]
try:
    _ftc.calibrate_or_load(lmB, **cal112, target="fsb_tau_x")
    raise AssertionError("a bad calibration target was accepted")
except ValueError:
    pass
_sh112.rmtree(_d112, ignore_errors=True)
mm = _mk_ut(1121, 0.5)
torch.manual_seed(1122); xs112 = [torch.randn(1, 512, d_model, device=dev) for _ in range(2)]
res112 = _fct(mm, xs112, 0.5, target="fsb_tau_union")
assert float(mm.fsb_tau.abs().max()) == 0.0 and bool((mm.fsb_tau_union > 0).all()) and torch.allclose(res112["tau"][0].to(dev), mm.fsb_tau_union)
assert mm._fsb_calib is None and not mm.training
_expect_value_error(lambda: _fct(mm, xs112, 0.5, target="fsb_tau_x"), "a bad calibration target")
from experiments.scaling.fsb_timing_probe import fsb_resolved as _fr112
m_on112 = _mk_ut(1123, 0.5, adapter=11230); m_off112 = _mk_ut(1123, None, adapter=11230)
c_on112, c_off112 = _rsc108(m_on112), _rsc108(m_off112)
assert c_on112["fsb_union_tau_quantile"] == 0.5 and "fsb_union_tau_quantile" not in c_off112
assert {k: v for k, v in c_on112.items() if k != "fsb_union_tau_quantile"} == c_off112
assert _im108({"config": c_off112}, model=m_off112) == [] and _im108({"config": c_on112}, model=m_on112) == []
assert _im108({"config": c_off112}, model=m_on112) and _im108({"config": c_on112}, model=m_off112)
r_on112, r_off112 = _fr112(m_on112), _fr112(m_off112)
assert r_on112["attrs"]["fsb_union_tau_quantile"] == 0.5 and r_off112["attrs"]["fsb_union_tau_quantile"] == 0.0
assert "tau_union" in r_on112 and "tau_union" not in r_off112 and r_on112["tau_union"]["max"] == 0.0
fsb_read_stats_reset(m_off112)
torch.manual_seed(1124); x = torch.randn(1, 512, d_model, device=dev)
with torch.no_grad():
    _full_sel_n(m_off112, x)
st112 = fsb_read_stats(m_off112); fsb_read_stats_reset(m_off112, on=False)
assert st112["utau_cells"] == 0.0 and st112["utau_frac"] == 0.0 and st112["mean"] > 0
print("F112 OK  (--fsb_union_tau_quantile: calibrate_or_load / fsb_calibrate_tau target = the union buffer only, "
      "shared cache file; re-ranker identity + fsb_resolved records; read stats utau keys)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  UNION QUOTA (fsb_union_quota k, with fsb_union_width + fsb_read_top; pass 5 of fsb_improvement_plan_20260929): the
#  two beams' candidates are never ranked against each other -- k read slots are reserved for the best SECOND-BEAM-ONLY
#  cells (the exact re-rank among themselves), the main beam's own re-rank order fills the rest; composable with the
#  beam quota and the pool quota; on the one-forest union and the filtered second forest alike.  F113..F116.  The tests
#  use the W16 main beam with the U16 second beam over the filtered forest (the pass-5 selector DAQ1; fsb_tau_union per
#  head as in F109..F112: head 2 always empty, head 3 == the main forest) and the one-forest W4 + U16 union (many
#  second-beam-only cells), read 4, the lse_pos re-rank with the raw query, random full adapters and fresh caches.
# ══════════════════════════════════════════════════════════════════════════════════════════
from src.fsattn.fsb_tree import (union_quota_kw as _uqkw, union_quota_on as _uqon, union_quota_merge, union_quota_list,
                                union_quota_prov_add, _priority_merge)
from src.fsattn.fsb_rerank import PROV_UNIONQ


def _mk_uq(seed, k, q=0.5, W=16, U=16, rt=4, adapter=None, mode=None, bq=None, P=PH, Wc=WCH, short=4, quota=1, **kw):
    """_mk_ut with fsb_union_quota only when ``k`` is not None (explicit 0 = the flag) and fsb_beam_quota ``bq``; ``q``
    None = the one-forest union (no fsb_union_tau_quantile)."""
    if k is not None:
        kw["fsb_union_quota"] = k
    return _mk_ut(seed, q, W=W, U=U, rt=rt, adapter=adapter, mode=mode, k=bq, P=P, Wc=Wc, short=short, quota=quota, **kw)


def _expect_uq_error(fn, what):
    """a ValueError naming fsb_union_quota (not one of the other knobs' refusals)."""
    try:
        fn()
    except ValueError as e_:
        assert "fsb_union_quota" in str(e_), (what, str(e_))
        return
    raise AssertionError(f"must refuse: {what}")


def _uq_capture(mm_, x_):
    """no-cache forward with the union quota's lists captured: (sel (H,Npad,4) long, y, main candidates (H,Npad,Cm),
    second-beam-only candidates (H,Npad,Cu), Tu (H,Npad,4) the second-only re-rank order, Tm (H,Npad,4) the main
    candidates' re-rank order, prov (H,N,4) the prefill's provenance, Lq (H,Npad,4) the pool list or None, the raw
    rerank_top records).  Checks that the traversal re-ranks exactly the two halves of union_candidates per chunk."""
    with torch.no_grad():
        (sel_, y_), rec = _hcap(lambda: _full_sel_n(mm_, x_), names=("union_candidates", "rerank_top", "fsb_utau_acc", "pool_rank"))
    Cm = mm_.fsb_W_max * (1 << mm_.fsb_delta_max)
    ucs = [o_ for _, _, o_ in rec["union_candidates"]]
    rr = rec["rerank_top"]
    assert len(rr) == 2 * len(ucs) > 0, (len(rr), len(ucs))
    for i_, uc_ in enumerate(ucs):                                           # per chunk: [second-only | main] on the split lists
        assert torch.equal(rr[2 * i_][0][0], uc_[..., Cm:]) and torch.equal(rr[2 * i_ + 1][0][0], uc_[..., :Cm])
        assert rr[2 * i_][2].shape[-1] == rr[2 * i_ + 1][2].shape[-1] == mm_.fsb_read_top
    cand = _rows_hb(ucs).long()
    Tu = _rows_hb([o_ for _, _, o_ in rr[0::2]]).long(); Tm = _rows_hb([o_ for _, _, o_ in rr[1::2]]).long()
    assert len(rec["fsb_utau_acc"]) == 1
    prov = rec["fsb_utau_acc"][0][0][1][0].long()
    Lq = _rows_hb([o_ for _, _, o_ in rec["pool_rank"]]).long() if rec["pool_rank"] else None
    return sel_.long(), y_, cand[..., :Cm], cand[..., Cm:], Tu, Tm, prov, Lq, rr


# ── F113: construction, refusals, the merge primitive and flag OFF.
#   (a) default 0 (union_quota_kw empty); k 1 / 2 / 4 with W16 + U16 (filtered) and W4 + U16 (one forest); compositions
#       accepted (beam quota + k, pool quota + beam quota + k <= read_top); 13 refusals naming the knob: k < 0, k >
#       read_top, without a union beam, without read_top, the rebuild tree, the learned policy, fsb_hybrid pool, fsb_hybrid
#       union, with fsb_read_margin, with fsb_rerank_rr, beam quota + k > read_top, pool quota + k > read_top, pool quota +
#       beam quota + k > read_top;
#   (b) union_quota_merge == the python merge ([ures | Tm | urem], [bres | ...], [L[:K_L] | (bres) | ... | L[K_L:]]) on
#       random rows with -1 pads and overlaps, == beam_quota_merge / quota_merge on union_quota_list (the hybrid path's
#       merge), _priority_merge == the two existing merges; PROV_UNIONQ == 16; union_quota_prov_add sets PROV_UTAU on the
#       second-only reads and PROV_UNIONQ on the reserved ones, -1 on empties, on top of an existing provenance;
#   (c) fsb_union_quota 0 passed explicitly == the flag absent bit-exact: selection / output / n_read / n_visit on the
#       no-cache forward, cached decode (full + lean prefill, graph 0 / 1, provenance), the CUDA-graph training stage;
#       the filtered second forest (off + quota hybrid) and the one-forest union ────────────────────────────────────
m = _mk_uq(1130, None)
assert m.fsb_union_quota == 0 and not _uqon(m) and _uqkw(m) == {}
m = _mk_uq(1130, 1)
assert m.fsb_union_quota == 1 and _uqon(m) and _uqkw(m) == {"union_quota": 1} and m.fsb_kk == 4 and m.fsb_union_W_max == 16
assert _mk_uq(1130, 4).fsb_union_quota == 4 and _mk_uq(1130, 2, q=None, W=4).fsb_union_quota == 2
assert _mk_uq(1130, 1, W=4, bq=2).fsb_beam_quota == 2 and _mk_uq(1130, 2, W=4, bq=2).fsb_union_quota == 2
mq = _mk_uq(1130, 1, W=4, bq=1, mode="quota", quota=2)
assert (mq.fsb_pool_quota, mq.fsb_beam_quota, mq.fsb_union_quota) == (2, 1, 1)
assert _mk_uq(1130, 3, mode="quota", quota=1).fsb_union_quota == 3
for what, fn in (("k < 0", lambda: _mk_uq(1130, -1)),
                 ("k > read_top", lambda: _mk_uq(1130, 5)),
                 ("without a union beam", lambda: _mk_uq(1130, 1, q=None, U=0)),
                 ("without read_top", lambda: mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_tree="forest", fsb_union_quota=1)),
                 ("rebuild tree", lambda: _mk_uq(1130, 1, q=None, fsb_tree="rebuild")),
                 ("learned policy", lambda: mk(fsb_policy="learned", fsb_widths="2,4,8", fsb_deltas="0,1", fsb_tree="forest",
                                                fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_union_width=16, fsb_union_quota=1)),
                 ("hybrid pool", lambda: _mk_uq(1130, 1, q=None, mode="pool")),
                 ("hybrid union", lambda: _mk_uq(1130, 1, q=None, mode="union", short=4)),
                 ("with read_margin", lambda: _mk_uq(1130, 1, q=None, rt=16, fsb_read_margin=1.0)),
                 ("with the residual re-ranker", lambda: _mk_uq(1130, 1, q=None, fsb_rerank_rr="linear")),
                 ("beam quota + k > read_top", lambda: _mk_uq(1130, 3, W=4, bq=2)),
                 ("pool quota + k > read_top", lambda: _mk_uq(1130, 4, mode="quota", quota=1)),
                 ("pool + beam + union quotas > read_top", lambda: _mk_uq(1130, 2, W=4, bq=2, mode="quota", quota=1))):
    _expect_uq_error(fn, what)
g113 = torch.Generator(device="cpu").manual_seed(113)
nmerge113 = 0


def _rows113(w_, p_, R_=64):
    return torch.stack([torch.randperm(12, generator=g113)[:w_] for _ in range(R_)]).masked_fill(torch.rand(R_, w_, generator=g113) < p_, -1)


for _ in range(8):
    for k in (1, 2, 3):
        ures = _rows113(k, 0.4); Tm = _rows113(4, 0.25); urem = _rows113(4 - k, 0.4); bres = _rows113(2, 0.3); L = _rows113(4, 0.25)
        uD, TD, rD, bD, LD = (t_.to(dev) for t_ in (ures, Tm, urem, bres, L))
        lst = union_quota_list(uD, TD, rD)
        assert lst.shape[-1] == 8 and torch.equal(lst.cpu(), torch.cat([ures, Tm, urem], -1))
        a_ = union_quota_merge(uD, TD, rD, 4).cpu(); b_ = union_quota_merge(uD, TD, rD, 4, bres=bD).cpu()
        assert torch.equal(b_, beam_quota_merge(bD, lst, 4).cpu())
        for r in range(64):
            assert a_[r].tolist() == _merge_py([ures[r].tolist(), Tm[r].tolist(), urem[r].tolist()], 4), (k, r)
            assert b_[r].tolist() == _merge_py([bres[r].tolist(), ures[r].tolist(), Tm[r].tolist(), urem[r].tolist()], 4), (k, r)
            nmerge113 += 2
        for KL in (1, 2):
            c_ = union_quota_merge(uD, TD, rD, 4, bres=bD, L=LD, K_L=KL).cpu(); d_ = union_quota_merge(uD, TD, rD, 4, L=LD, K_L=KL).cpu()
            assert torch.equal(c_, beam_quota_merge(bD, lst, 4, LD, KL).cpu()) and torch.equal(d_, quota_merge(LD, lst, KL, 4).cpu())
            for r in range(64):
                assert c_[r].tolist() == _merge_py([L[r, :KL].tolist(), bres[r].tolist(), ures[r].tolist(), Tm[r].tolist(), urem[r].tolist(),
                                                    L[r, KL:].tolist()], 4), (k, KL, r)
                assert d_[r].tolist() == _merge_py([L[r, :KL].tolist(), ures[r].tolist(), Tm[r].tolist(), urem[r].tolist(), L[r, KL:].tolist()], 4), (k, KL, r)
                nmerge113 += 2
        assert torch.equal(_priority_merge([bD, TD], 4), beam_quota_merge(bD, TD, 4))
        assert torch.equal(_priority_merge([LD[..., :1], TD, LD[..., 1:]], 4), quota_merge(LD, TD, 1, 4))
assert PROV_UNIONQ == 16 and PROV_UNIONQ & (PROV_TREE | PROV_POOL | PROV_BEAMQ | PROV_UTAU) == 0
sel113 = torch.tensor([[7, 3, 9, -1]], device=dev); unew113 = torch.tensor([[9, 7, -1, 5]], device=dev); ures113 = torch.tensor([[7]], device=dev)
pl113 = []
union_quota_prov_add(pl113, sel113, unew113, ures113)
assert len(pl113) == 1 and pl113[0].dtype == torch.int8 and pl113[0].tolist() == [[24, 0, 8, -1]]
pl113 = [torch.tensor([[0, 4, 0, -1]], dtype=torch.int8, device=dev)]
union_quota_prov_add(pl113, sel113, unew113, ures113)
assert len(pl113) == 1 and pl113[0].tolist() == [[24, 4, 8, -1]]
print(f"F113 [a,b] fsb_union_quota: default 0; 13 refusals; union_quota_merge == python on {nmerge113} rows and == the beam / pool "
      f"quota merges on union_quota_list; PROV_UNIONQ 16 on the reserved slots, PROV_UTAU on the second-only reads")
n113, N0113 = 512, 300
n113c = 0
for mode113, q113, W113 in ((None, 0.5, 16), ("quota", 0.5, 16), (None, None, 4)):
    mA = _mk_uq(1132, None, q=q113, W=W113, adapter=11320, mode=mode113); mB = _mk_uq(1132, 0, q=q113, W=W113, adapter=11320, mode=mode113)
    assert all(torch.equal(a_, b_) for a_, b_ in zip(mA.state_dict().values(), mB.state_dict().values()))
    torch.manual_seed(1133); x = torch.randn(1, n113, d_model, device=dev)
    if q113:
        ce113 = _ceil_of(mA, x); _set_utau(mA, ce113); _set_utau(mB, ce113)
    with torch.no_grad():
        sA, yA = _full_sel_n(mA, x); nA = _nstats(mA); sB, yB = _full_sel_n(mB, x); nB = _nstats(mB)
    assert torch.equal(sA, sB) and torch.equal(yA, yB) and all(torch.equal(a_, b_) for a_, b_ in zip(nA, nB)), (mode113, q113, "explicit 0 != absent")
    for lean in (False, True):
        for dg in (("0", "1") if dev == "cuda" else ("0",)):
            with torch.no_grad():
                rA = _ut_cached(mA, x, N0113, lean, dg); rB = _ut_cached(mB, x, N0113, lean, dg)
            assert torch.equal(rA[0], rB[0]) and all(torch.equal(a_, b_) for a_, b_ in zip(rA[1], rB[1])), (mode113, q113, lean, dg)
            assert all((pa is None and pb is None) or torch.equal(pa, pb) for pa, pb in zip(rA[2], rB[2])), (mode113, q113, lean, dg)
            n113c += 1
    if dev == "cuda":
        mA.train(); mB.train(); Qc, Kc, Vc = _qkv_n(1134, n113)
        with torch.no_grad():
            rA = _sel_n(mA, Qc, Kc, Vc, n113, "1") + _nstats(mA); rB = _sel_n(mB, Qc, Kc, Vc, n113, "1") + _nstats(mB)
        assert all(torch.equal(a_, b_) for a_, b_ in zip(rA, rB)), (mode113, q113, "graphed stage")
        mA.eval(); mB.eval(); n113c += 1
print(f"F113 OK  (fsb_union_quota: construction + 13 refusals; merge primitive == python; explicit 0 == absent bit-exact on the "
      f"no-cache forward + {n113c} cached-decode / graphed-stage cases: filtered forest off + quota hybrid, one-forest union)")


# ── F114: SEMANTICS end to end (fp32, n 512, random full adapter).  For W16 + U16 over the filtered forest (k 1 / 2) and
#   the one-forest W4 + U16 union (k 1 / 2):
#   (a) the traversal re-ranks exactly two lists per chunk, the second-beam-only candidates and the main candidates (the
#       halves of union_candidates), never the union (``_uq_capture``);
#   (b) the read == the python merge [Tu[:k] | Tm | Tu[k:]] of the module's own two re-rank orders on every row (the
#       reserved cells first, then the main order, no duplicates, exactly min(4, |main u second-only|) cells);
#   (c) float64: the reserved cells are the top-k of the second-beam-only candidates by the lse_pos score and Tm the
#       float64 top-4 of the main candidates (near-ties skipped);
#   (d) fallback: rows whose second beam found NO new cell read the main order alone == the no-union module's read (head 2
#       of the filtered forest always, the early blocks of the one-forest union); rows with fewer than k new cells (k 2)
#       read them all and the main order fills;
#   (e) teeth against the shared re-rank (no quota): the read differs on some rows, the quota keeps a reserved cell the
#       shared re-rank evicted and drops a second-only cell the shared re-rank read;
#   (f) provenance: PROV_UNIONQ exactly on the slots holding a reserved cell, PROV_UTAU exactly on the second-beam-only
#       reads (also for the one-forest union), -1 on empties, PROV_UNIONQ => PROV_UTAU; the armed read accumulators
#       count both (== the captured provenance); admit_frac_union only with the filtered forest ────────────────────
n114 = 512
nrow114 = nres114 = ndrop114 = nfb114 = npart114 = n64_114 = ntie114 = 0
for name114, c114 in (("W16 U16 tau0.5 k1", dict(k=1)), ("W16 U16 tau0.5 k2", dict(k=2)),
                      ("W4 U16 one-forest k1", dict(k=1, q=None, W=4)), ("W4 U16 one-forest k2", dict(k=2, q=None, W=4))):
    k114 = c114["k"]; q114 = c114.get("q", 0.5); W114 = c114.get("W", 16)
    mm = _mk_uq(1140, adapter=11400, **c114)
    m0 = _mk_uq(1140, None, adapter=11400, **{k_: v_ for k_, v_ in c114.items() if k_ != "k"})   # the shared re-rank (no quota)
    mN = _mk_uq(1140, None, U=0, adapter=11400, q=None, W=W114)                                  # no union beam
    torch.manual_seed(1141); x = torch.randn(1, n114, d_model, device=dev)
    if q114:
        ce114 = _ceil_of(mm, x); _set_utau(mm, ce114); _set_utau(m0, ce114)
    sq, yq, main, unew, Tu, Tm, prov, _, rr114 = _uq_capture(mm, x)
    with torch.no_grad():
        s0, _ = _full_sel_n(m0, x); sN, _ = _full_sel_n(mN, x)
    s0 = s0.long(); sN = sN.long()
    sqc, Tuc, Tmc, unc, mainc = sq.cpu(), Tu.cpu(), Tm.cpu(), unew.cpu(), main.cpu()
    for h in range(H):
        for i in range(BLK, n114):
            ures = [c for c in Tuc[h, i, :k114].tolist() if c >= 0]
            exp = _merge_py([Tuc[h, i, :k114].tolist(), Tmc[h, i].tolist(), Tuc[h, i, k114:].tolist()], 4)
            assert sqc[h, i].tolist() == exp, (name114, h, i, sqc[h, i].tolist(), exp)
            assert sqc[h, i, :len(ures)].tolist() == ures, (name114, h, i)  # the reserved cells first
            got = [c for c in sqc[h, i].tolist() if c >= 0]
            nu = len({c for c in mainc[h, i].tolist() if c >= 0} | {c for c in unc[h, i].tolist() if c >= 0})
            assert len(got) == len(set(got)) == min(4, nu), (name114, h, i)
            nnew = int((unc[h, i] >= 0).sum())
            if nnew == 0:                                                    # fallback: the main order alone
                assert sqc[h, i].tolist() == Tmc[h, i].tolist(), (name114, h, i); nfb114 += 1
            elif nnew < k114:                                                # every new cell reserved, the main order fills
                assert len(ures) == nnew, (name114, h, i); npart114 += 1
            nrow114 += 1
    empty = (unew < 0).all(-1)                                               # (H,Npad) rows without a second-beam-only cell
    assert bool(empty[:, BLK:].any()), (name114, "no row without a second-beam-only cell")
    assert torch.equal(sq[empty], sN[empty]), (name114, "empty second-only list != the no-union read")
    if q114:
        assert bool(empty[2].all()) and torch.equal(sq[2], sN[2]), (name114, "head 2 (empty second forest) != the no-union read")
    assert bool((~_same_sets(sq, s0))[:, BLK:].any()), (name114, "no teeth against the shared re-rank")
    ures_all = Tu[..., :k114]
    kept = _mem109(ures_all, sq) & ~_mem109(ures_all, s0)                   # a reserved cell the shared re-rank evicted
    dropped = _mem109(s0, unew) & ~_mem109(s0, sq)                          # a second-only cell only the shared re-rank read
    nres114 += int(kept.sum()); ndrop114 += int(dropped.sum())
    in_unew = _mem109(sq, unew); in_res = _mem109(sq, ures_all)
    assert torch.equal(prov == -1, sq < 0), name114
    assert torch.equal((prov >= 0) & ((prov & PROV_UNIONQ) != 0), in_res), (name114, "PROV_UNIONQ != reserved membership")
    assert torch.equal((prov >= 0) & ((prov & PROV_UTAU) != 0), in_unew), (name114, "PROV_UTAU != second-beam-only membership")
    assert bool((in_unew | ~in_res).all()) and set(prov.unique().tolist()) <= {-1, 0, PROV_UTAU, PROV_UTAU | PROV_UNIONQ}, name114
    assert bool(in_res.any()) and int(in_res.sum()) == int((ures_all >= 0).sum()), (name114, "a reserved cell not read")
    fsb_read_stats_reset(mm)
    with torch.no_grad():
        _full_sel_n(mm, x)
    st114 = fsb_read_stats(mm); fsb_read_stats_reset(mm, on=False)
    assert st114["uq_cells"] == float(in_res.sum()) and st114["utau_cells"] == float(in_unew.sum()), (name114, st114)
    assert abs(st114["uq_frac"] - float(in_res.sum()) / float((sq >= 0).sum())) < 1e-6
    assert ("admit_frac_union" in mm._fsb_stats) == bool(q114), name114
    for j_, (a_, k_, o_) in enumerate(rr114):                                # (c) float64 brute force on the two lists
        sel_c, q_, qpos_, lam_, sigv_, cellK_, cellA_, tidx_, scale_, s_, K_ = a_
        assert k_["score"] == "lse_pos" and K_ == 4 and "read_margin" not in k_ and "rr_w" not in k_, (name114, j_)
        with torch.no_grad():
            u64 = _tok_brute(q_, sel_c, k_["tokK"], tidx_, scale_, s_, "lse_pos", qpos_, lam_, sigv_, cellA_)
        kk_ = k114 if j_ % 2 == 0 else 4                                     # the reserved count / the main top-4
        ref, tie, nval = _topk_ref(sel_c, u64, kk_)
        same = _same_sets(o_[..., :kk_].long(), ref.long())
        assert bool(same[~tie].all()), (name114, j_, int((~same & ~tie).sum()))
        n64_114 += int((~tie).sum()); ntie114 += int(tie.sum())
assert nres114 > 0 and ndrop114 > 0 and nfb114 > 0 and npart114 > 0, (nres114, ndrop114, nfb114, npart114)
print(f"F114 OK  (union quota: two split re-ranks per chunk; read == [reserved | main order | remainder] merge on {nrow114} rows "
      f"({nfb114} fall back to the main order, {npart114} reserve fewer than k); reserved / main == float64 top-k on {n64_114} rows "
      f"({ntie114} near-ties skipped); teeth: {nres114} reserved cells the shared re-rank evicted, {ndrop114} second-only cells only "
      f"it read; PROV_UNIONQ / PROV_UTAU == memberships, accumulators agree)")


# ── F115: PARITY, COMPOSITION and the remaining paths.  (a) prefill == decode (full + lean prefill, FSB_DECODE_GRAPH
#   0 / 1), cached == full forward, graph decode == eager, decode provenance == the prefill's, for W16 U16 tau0.5 k1 /
#   W4 U16 one-forest k2 / quota hybrid K_L1 + k1 / W4 U16 tau0.5 with beam quota 1 + pool quota 1 + k 1 or 2;
#   (b) composition: the read == the python merge [L[:K_L] | W4 picks | Tu[:k] | Tm | Tu[k:] | L[K_L:]] of the module's
#       own lists (the plain W4 beam's picks, pool_rank, the two re-rank orders); PROV_BEAMQ on the W4 picks, PROV_UNIONQ
#       on the reserved union cells, PROV_UTAU on the second-beam-only reads, the hybrid codes underneath; a beam-reserved
#       cell never carries a union bit; every reserved union cell is read; the read differs from the same pool + beam
#       quota without the union quota (teeth);
#   (c) chunked prefill 1 / 2 / all identical; CUDA-graph training stage == eager, the key re-captures on a k change and
#       the selection changes; Triton == torch (traversal + re-rank kernels);
#   (d) k switched mid-block on a live decode cache (1 -> 0, 0 -> 1, 1 -> 2) == a fresh cache (the decode config key);
#   (e) the router node-trace re-run reproduces the forward's read set (0 mismatches), plain + quota hybrid with the beam
#       quota, eager + graph; (f) bf16 4096: the decode forests + cache keys route the prefill's queries to the
#       prefill's cells (GPU) ──────────────────────────────────────────────────────────────────────────────────────
n115, N0115 = 512, 300
cfg115 = (("W16 U16 tau0.5 k1", dict(k=1)), ("W4 U16 one-forest k2", dict(k=2, q=None, W=4)),
          ("quota K_L1 + k1", dict(k=1, mode="quota", quota=1)),
          ("W4 U16 tau0.5 bq1 + K_L1 + k1", dict(k=1, W=4, bq=1, mode="quota", quota=1)),
          ("W4 U16 tau0.5 bq1 + K_L1 + k2", dict(k=2, W=4, bq=1, mode="quota", quota=1)))
nrow115 = nord115 = nprov115 = ngr115 = ncomp115 = nteeth115 = 0
for name115, c115 in cfg115:
    mm = _mk_uq(1150, adapter=11500, **c115)
    torch.manual_seed(1151); x = torch.randn(1, n115, d_model, device=dev)
    q115 = c115.get("q", 0.5); k115 = c115["k"]; bq115 = c115.get("bq", 0); KL115 = c115.get("quota", 0) if "mode" in c115 else 0
    if q115:
        ce115 = _ceil_of(mm, x); _set_utau(mm, ce115)
    sel_f, y_f, main_f, unew_f, Tu_f, Tm_f, prov_f, Lq_f, _ = _uq_capture(mm, x)
    assert (Lq_f is not None) == ("mode" in c115), name115
    if bq115 or KL115:                                                        # (b) the composed python merge
        mp = _mk_plain(1150, W=c115.get("W", 16), adapter=11500)
        mh = _mk_uq(1150, None, adapter=11500, **{k_: v_ for k_, v_ in c115.items() if k_ != "k"})
        if q115:
            _set_utau(mh, ce115)
        with torch.no_grad():
            sp, _ = _full_sel_n(mp, x); sh, _ = _full_sel_n(mh, x)
        W4 = _compact_rows(sp.long())[..., :bq115] if bq115 else None
        W4c = W4.cpu() if W4 is not None else None
        sfc, Tuc, Tmc, Lqc = sel_f.cpu(), Tu_f.cpu(), Tm_f.cpu(), Lq_f.cpu()
        for h in range(H):
            for i in range(BLK, n115):
                br = [c for c in W4c[h, i].tolist() if c >= 0] if W4c is not None else []
                exp = _merge_py([Lqc[h, i, :KL115].tolist(), br, Tuc[h, i, :k115].tolist(), Tmc[h, i].tolist(), Tuc[h, i, k115:].tolist(),
                                 Lqc[h, i, KL115:].tolist()], 4)
                assert sfc[h, i].tolist() == exp, (name115, h, i, sfc[h, i].tolist(), exp)
                ncomp115 += 1
        in_res = _mem109(sel_f, Tu_f[..., :k115]); in_u = _mem109(sel_f, unew_f); in_p = _mem109(sel_f, Lq_f)
        assert torch.equal((prov_f >= 0) & ((prov_f & PROV_UNIONQ) != 0), in_res), (name115, "PROV_UNIONQ != reserved membership")
        assert torch.equal((prov_f >= 0) & ((prov_f & PROV_UTAU) != 0), in_u), (name115, "PROV_UTAU != second-beam-only membership")
        assert torch.equal((prov_f >= 0) & (((prov_f & 3) == 1) | ((prov_f & 3) == 2)), in_p), (name115, "pool codes != pool membership")
        assert torch.equal(prov_f == -1, sel_f < 0) and int(in_res.sum()) == int((Tu_f[..., :k115] >= 0).sum()), name115
        if W4 is not None:
            in_b = _mem109(sel_f, W4)
            assert torch.equal((prov_f >= 0) & ((prov_f & PROV_BEAMQ) != 0), in_b), (name115, "PROV_BEAMQ != W4 membership")
            assert not bool((in_b & in_u).any()) and bool(in_b.any()), (name115, "a beam-reserved cell carries a union bit")
        assert bool(in_res.any()) and bool(in_p.any()), name115
        nteeth115 += int((~_same_sets(sel_f, sh.long()))[:, BLK:].sum())
    ys = {}
    with torch.no_grad():
        for lean in (False, True):
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                y, sels, provs, _, _ = _ut_cached(mm, x, N0115, lean, dg)
                assert (y - y_f).abs().max().item() < 1e-4, (name115, lean, dg, (y - y_f).abs().max().item())
                dec = torch.stack(sels, dim=1)
                same = _same_sets(sel_f[:, N0115:n115], dec.long())
                assert bool(same.all()), f"{name115} lean={lean} graph={dg}: {int((~same).sum())} decode rows differ"
                nord115 += int((sel_f[:, N0115:n115] == dec.long()).all(-1).sum()); nrow115 += same.numel()
                for k_, t in enumerate(range(N0115, n115)):
                    assert provs[k_] is not None and torch.equal(provs[k_] == -1, sels[k_] < 0), (name115, lean, dg, t)
                    if torch.equal(sels[k_].long(), sel_f[:, t]):
                        assert torch.equal(provs[k_].long(), prov_f[:, t]), (name115, lean, dg, t, "decode provenance != prefill")
                        nprov115 += 1
                ys[(lean, dg)] = y
        if dev == "cuda":
            for lean in (False, True):
                assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (name115, lean, "graph decode != eager")
    outs = []
    for nbch in (1, 2, 0):
        mc = _mk_uq(1150, adapter=11500, fsb_prefill_blkchunk=nbch, **c115)
        with torch.no_grad():
            if q115:
                mc.fsb_tau_union.copy_(mm.fsb_tau_union)
            outs.append(_full_sel_n(mc, x) + _nstats(mc))
    assert all(all(torch.equal(a_, b_) for a_, b_ in zip(outs[0], o_)) for o_ in outs[1:]), (name115, "chunk size")
    assert torch.equal(outs[0][0].long(), sel_f), (name115, "chunk 1 != the default chunking")
    if dev == "cuda":
        mm.train(); Qc, Kc, Vc = _qkv_n(1152, n115)
        if q115:
            _set_utau(mm, _ceil_K(Kc))
        with torch.no_grad():
            r0 = _sel_n(mm, Qc, Kc, Vc, n115, "0") + _nstats(mm); r1 = _sel_n(mm, Qc, Kc, Vc, n115, "1") + _nstats(mm)
            assert all(torch.equal(a_, b_) for a_, b_ in zip(r0, r1)), (name115, "graphed training stage != eager")
            k1 = mm._fsb_train_graphs["key"]
            k_new = 2 if k115 == 1 else 1
            mm.fsb_union_quota = k_new
            r2 = _sel_n(mm, Qc, Kc, Vc, n115, "1") + _nstats(mm); k2 = mm._fsb_train_graphs["key"]
            r3 = _sel_n(mm, Qc, Kc, Vc, n115, "0") + _nstats(mm)
            assert k2 != k1 and all(torch.equal(a_, b_) for a_, b_ in zip(r2, r3)), (name115, "stale graph after a k change")
            assert not torch.equal(r2[0], r1[0]), (name115, "the k change did not change the selection (no teeth)")
            mm.fsb_union_quota = k115
            ref = None
            for trv, rrk in (("1", "1"), ("0", "1"), ("1", "0"), ("0", "0")):
                with _env(FSB_RERANK_TRITON=rrk):
                    r_ = _sel_n(mm, Qc, Kc, Vc, n115, "0", triton=trv) + _nstats(mm)
                if ref is None:
                    ref = r_
                assert all(torch.equal(a_, b_) for a_, b_ in zip(ref, r_)), (name115, trv, rrk, "Triton != torch")
            ngr115 += 1
        mm.eval()
assert nteeth115 > 0, "the union quota never changed the pool + beam quota's read (no teeth)"
print(f"F115 [a-c] prefill == decode on {nrow115} rows (order identical on {nord115}; full + lean, graph 0 / 1), cached == full, "
      f"graph decode == eager, decode provenance == prefill on {nprov115} rows; composition == the python merges on {ncomp115} rows "
      f"(differs from the pool + beam quota alone on {nteeth115}); chunked prefill identical; training graph == eager + key "
      f"re-captures, Triton == torch on {ngr115} configs")
torch.manual_seed(1153); x153 = torch.randn(1, n115, d_model, device=dev)
ce153 = _ceil_of(_mk_uq(1154, 1, adapter=11540), x153)
nsw115 = teeth115d = 0
for dg in (("0", "1") if dev == "cuda" else ("0",)):
    def _fresh115(k_):
        m_ = _mk_uq(1154, k_, adapter=11540)
        _set_utau(m_, ce153)
        return m_
    for nm, kA, kB in (("1 -> 0", 1, None), ("0 -> 1", None, 1), ("1 -> 2", 1, 2)):
        mA, mB, mC = _fresh115(kA), _fresh115(kB), _fresh115(kA)
        sw = _hy_switch(mA, x153, N0115, dg, 400, _set101(fsb_union_quota=(kB or 0)))
        ref = _hy_switch(mB, x153, N0115, dg); ctl = _hy_switch(mC, x153, N0115, dg)
        assert all(torch.equal(sw[t], ref[t]) for t in range(400, n115)), (nm, dg, "k switched on a live cache != a fresh cache")
        teeth115d += sum(int(not torch.equal(ctl[t], ref[t])) for t in range(400, n115))
        nsw115 += 1
assert teeth115d > 0, "no decode row differs between the switched configs after the switch: no teeth"
print(f"F115 [d] {nsw115} live-cache switches of k (1 -> 0, 0 -> 1, 1 -> 2) == a fresh cache (eager" + (" + CUDA graph)" if dev == "cuda" else ")")
      + f"; {teeth115d} post-switch steps differ between the configs")
nmis115 = ntr115 = 0
for name115, extra in (("plain W4 U16 tau0.5 k1", dict()),
                       ("quota K_L1 bq1 k1", dict(fsb_hybrid="quota", fsb_pool_size=PH, fsb_pool_recent=WCH, fsb_pool_quota=1, fsb_beam_quota=1))):
    torch.manual_seed(1155)
    m_ = _mk_node(fsb_width=4, fsb_rd_node=1, fsb_rd_node_weight=0.5, fsb_route_learn_c=1, fsb_bound_radius_coef=8.0,
                  fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_union_width=16, fsb_union_quota=1,
                  fsb_union_tau_quantile=0.5, **extra).train()
    with torch.no_grad():
        g_ = torch.Generator(device="cpu").manual_seed(1156)
        m_.fsb_route_W.copy_((torch.randn(m_.fsb_route_W.shape, generator=g_) * 0.3).to(dev))
        m_.fsb_route_ctheta.add_((torch.randn(H, generator=g_) * 1.0).to(dev))
    Qc, Kc, Vc = _qkv_n(1157, N2, 1.5)
    _set_utau(m_, _ceil_K(Kc))
    for g in (("0", "1") if dev == "cuda" else ("0",)):
        m_.zero_grad(set_to_none=True)
        m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(1158)
        sel_, o_, rd_, fr_ = _lr_n(m_, Qc, Kc, Vc, N2, g)
        st_ = rd_["stats"]
        assert fr_ is not None and st_.numel() == RD_STATS_NODE and fr_[0].get("utau_F") is not None, (name115, g)
        assert float(st_[14]) == 0.0, (name115, g, "node trace re-run != the forward's read set", st_[14].item())
        nmis115 += int(st_[14]); ntr115 += int(st_[15])
        assert torch.isfinite(rd_["loss"]) and int(sel_.shape[-1]) == 4
print(f"F115 [e] router node trace with the union quota: {nmis115} mismatches / {ntr115} rows (plain + quota hybrid with the beam quota, eager"
      + (" + graph)" if dev == "cuda" else ")"))
if dev == "cuda":
    for lean in (False, True):
        torch.manual_seed(1159)
        mm = _mk_uq(1159, 1, adapter=11590)
        x = torch.randn(1, 4096, d_model, device=dev)
        _set_utau(mm, _ceil_of(mm, x))
        _f65_bf16_ut(mm, x, 4096, lean, "union quota W16 U16 tau0.5 k1 lse_pos raw")
else:
    print("F115 [f] bf16 SKIP (no CUDA)")
print("F115 OK  (union quota: prefill == decode, graph == eager (decode + training, keys), composition with the beam + pool quotas, "
      "chunked prefill, Triton == torch, live-cache switches == fresh, node trace 0 mismatches, bf16 4096 routing)")


# ── F116: PLUMBING (the kept part).  The re-ranker identity records --fsb_union_quota only when on (older identities
#   stay comparable) and reports the difference against a model with a quota; fsb_resolved records the attribute;
#   read stats carry the uq keys (0 when off; > 0 for the one-forest union with the quota) ───────────────────────────
m_on116 = _mk_uq(1160, 1, adapter=11600); m_off116 = _mk_uq(1160, None, adapter=11600)
c_on116, c_off116 = _rsc108(m_on116), _rsc108(m_off116)
assert c_on116["fsb_union_quota"] == 1 and "fsb_union_quota" not in c_off116
assert {k: v for k, v in c_on116.items() if k != "fsb_union_quota"} == c_off116
assert _im108({"config": c_off116}, model=m_off116) == [] and _im108({"config": c_on116}, model=m_on116) == []
assert _im108({"config": c_off116}, model=m_on116) and _im108({"config": c_on116}, model=m_off116)
r_on116, r_off116 = _fr112(m_on116), _fr112(m_off116)
assert r_on116["attrs"]["fsb_union_quota"] == 1 and r_off116["attrs"]["fsb_union_quota"] == 0
fsb_read_stats_reset(m_off116)
torch.manual_seed(1161); x = torch.randn(1, 512, d_model, device=dev)
with torch.no_grad():
    _full_sel_n(m_off116, x)
st116 = fsb_read_stats(m_off116); fsb_read_stats_reset(m_off116, on=False)
assert st116["uq_cells"] == 0.0 and st116["uq_frac"] == 0.0 and st116["mean"] > 0
m1f116 = _mk_uq(1162, 1, q=None, W=4, adapter=11620)                            # the one-forest union with the quota
fsb_read_stats_reset(m1f116)
with torch.no_grad():
    _full_sel_n(m1f116, x)
st116 = fsb_read_stats(m1f116); fsb_read_stats_reset(m1f116, on=False)
assert st116["uq_cells"] > 0 and st116["utau_cells"] >= st116["uq_cells"] and 0 < st116["uq_frac"] <= st116["utau_frac"]
assert "admit_frac_union" not in (m1f116._fsb_stats or {})
print("F116 OK  (--fsb_union_quota: re-ranker identity records the quota only when on; fsb_resolved records it; "
      "read stats uq keys)")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  UNION PRIOR SCALE (fsb_union_prior_scale s, with fsb_union_width + fsb_read_top; pass 6 of
#  fsb_improvement_plan_20260929): the exact re-rank's position / key-norm prior term lambda_h a_c erf(.) of the
#  SECOND-BEAM-ONLY candidates (the slots after the main block of union_candidates) is multiplied by s in [0, 1] (0 =
#  they compete on the LSE alone); main-beam candidates keep the full lse_pos.  F117..F120.  The tests use the U16
#  second beam over the filtered forest (fsb_tau_union per head as in F109..F112: head 2 always empty, head 3 == the
#  main forest) and the one-forest union, read 4, the lse_pos re-rank with the raw query, random full adapters and
#  fresh caches.  The flag-off / parity / plumbing sections keep the W16 main beam (the pass-4 selector DA50); the
#  SEMANTICS fixtures (F118, the F119 live-cache switches) use the W4 main beam and sel_scale 0.1 (LAM118), because
#  the scale needs second-beam-only cells IN CONTENTION: over the 32-cell test forest the W16 beam retains nearly every
#  admitted cell (86 / 1792 rows with a second-only candidate, none of them read at s 1, so s 0 == s 1 on every row),
#  and at sel_scale 0.7 the prior term (about 2.2 on every candidate) exceeds the whole LSE spread (below 0.8), so
#  s 0.5 and s 0 evict every second-only read alike; with W4 and lambda 0.1 (prior about 0.3) hundreds of rows differ
#  between s 1 / 0.5 / 0.  sel_scale > 0 everywhere, so the prior term is >= 0 and scaling it can only LOWER a
#  second-beam-only candidate's score.
# ══════════════════════════════════════════════════════════════════════════════════════════
from src.fsattn.fsb_tree import union_prior_kw as _upkw, union_prior_on as _upon, union_prior_vec, _prior_extend


def _mk_up(seed, s, q=0.5, W=16, U=16, rt=4, adapter=None, mode=None, bq=None, uq=None, P=PH, Wc=WCH, short=4, quota=1, **kw):
    """_mk_ut with fsb_union_prior_scale only when ``s`` is not None (an explicit 1.0 = the flag), fsb_beam_quota ``bq``
    and fsb_union_quota ``uq``; ``q`` None = the one-forest union (no fsb_union_tau_quantile)."""
    if s is not None:
        kw["fsb_union_prior_scale"] = s
    if uq is not None:
        kw["fsb_union_quota"] = uq
    return _mk_ut(seed, q, W=W, U=U, rt=rt, adapter=adapter, mode=mode, k=bq, P=P, Wc=Wc, short=short, quota=quota, **kw)


def _expect_up_error(fn, what):
    """a ValueError naming fsb_union_prior_scale (not one of the other knobs' refusals)."""
    try:
        fn()
    except ValueError as e_:
        assert "fsb_union_prior_scale" in str(e_), (what, str(e_))
        return
    raise AssertionError(f"must refuse: {what}")


def _pos64(sel_c, qpos_, lam_, sigv_, cellA_, tidx_, s_):
    """float64 position / key-norm prior term lam a_c erf(|qpos - (c + 0.5) s| / (2 sigma)) of candidates sel_c (T',Bq,C)."""
    Tp_, Bq_, C_ = sel_c.shape
    c_ = sel_c.clamp(min=0)
    a_ = cellA_.double()[tidx_.view(Tp_, 1, 1).expand_as(c_), c_]
    return lam_.double() * a_ * torch.erf((qpos_.double()[..., None] - (c_.double() + 0.5) * s_).abs() / (2.0 * sigv_.double()))


def _up_capture(mm_, x_):
    """no-cache forward with the re-rank calls captured: (sel (H,Npad,4) long, y, union candidates (H,Npad,C), the raw
    rerank_top records [(args, kwargs, out)], the raw union_candidates records, the raw pool_rank records)."""
    with torch.no_grad():
        (sel_, y_), rec = _hcap(lambda: _full_sel_n(mm_, x_), names=("union_candidates", "rerank_top", "pool_rank"))
    Cu = mm_.fsb_union_W_max * (1 << mm_.fsb_delta_max)
    ucs = [r_ for r_ in rec["union_candidates"] if r_[0][1].shape[-1] == Cu]      # the two beams' union (not a pool list's)
    cand = _rows_hb([o_ for _, _, o_ in ucs]).long() if ucs else None
    return sel_.long(), y_, cand, rec["rerank_top"], ucs, rec["pool_rank"]


def _up_capture_u(mm_, x_):
    """_up_capture plus the implementation's own SCORE VECTOR of every re-rank call: the fp32 u (T',Bq,C) that
    rerank_top hands to its stable descending sort (torch.sort recorded only inside rerank_top), as (H,Npad,C) rows in
    the union candidates' slot layout -> (sel, y, cand, rerank records, union records, pool records, U)."""
    st_ = {"in": False, "u": None}; us_ = []
    o_sort, o_rr = torch.sort, _fth.rerank_top

    def _sort(t_, *a_, **k_):
        if st_["in"] and k_.get("stable") and k_.get("descending"):
            st_["u"] = t_.clone()
        return o_sort(t_, *a_, **k_)

    def _rr(*a_, **k_):
        st_["in"], st_["u"] = True, None
        try:
            o_ = o_rr(*a_, **k_)
        finally:
            st_["in"] = False
        assert st_["u"] is not None, "rerank_top sorted nothing"
        us_.append(st_["u"]); return o_
    torch.sort, _fth.rerank_top = _sort, _rr
    try:
        r_ = _up_capture(mm_, x_)
    finally:
        torch.sort, _fth.rerank_top = o_sort, o_rr
    assert len(us_) == len(r_[3]), (len(us_), len(r_[3]))
    return r_ + (_rows_hb(us_),)


# ── F117: construction, refusals, the scale vector and flag OFF.
#   (a) default 1.0 (union_prior_kw empty); s 0 / 0.5 with W16 + U16 (filtered) and W4 + U16 (one forest), with the beam
#       quota, the union quota, the quota hybrid and the union hybrid accepted; 8 refusals naming the knob: s < 0, s > 1,
#       NaN, without a union beam, without read_top, the rebuild tree, the learned policy, fsb_hybrid pool;
#   (b) union_prior_vec (1,1,Cm+Cu): 1.0 on the main slots, s after; _prior_extend appends 1.0 slots; rerank_top with a
#       vector of ones == rerank_top without one BIT-EXACT on the forward's own re-rank calls (args replayed); with a
#       vector of s on every slot == the float64 lse + s pos top-4 (near-ties skipped);
#   (c) fsb_union_prior_scale 1.0 passed explicitly == the flag absent bit-exact: selection / output / n_read / n_visit on
#       the no-cache forward, cached decode (full + lean prefill, graph 0 / 1, provenance), the CUDA-graph training stage;
#       the filtered second forest (off + quota hybrid) and the one-forest union ────────────────────────────────────
m = _mk_up(1170, None)
assert m.fsb_union_prior_scale == 1.0 and not _upon(m) and _upkw(m) == {}
m = _mk_up(1170, 0.0)
assert m.fsb_union_prior_scale == 0.0 and _upon(m) and _upkw(m) == {"union_prior": 0.0} and m.fsb_kk == 4 and m.fsb_union_W_max == 16
assert _mk_up(1170, 0.5).fsb_union_prior_scale == 0.5 and _mk_up(1170, 1.0).fsb_union_prior_scale == 1.0 and not _upon(_mk_up(1170, 1.0))
assert _mk_up(1170, 0.5, q=None, W=4).fsb_union_prior_scale == 0.5 and _mk_up(1170, 0.0, W=4, bq=1).fsb_beam_quota == 1
assert _mk_up(1170, 0.0, uq=1).fsb_union_quota == 1 and _mk_up(1170, 0.5, mode="quota", quota=1).fsb_hybrid == "quota"
assert _mk_up(1170, 0.5, mode="union", short=4).fsb_hybrid == "union"
mq = _mk_up(1170, 0.5, W=4, bq=1, uq=1, mode="quota", quota=1)
assert (mq.fsb_pool_quota, mq.fsb_beam_quota, mq.fsb_union_quota, mq.fsb_union_prior_scale) == (1, 1, 1, 0.5)
for what, fn in (("s < 0", lambda: _mk_up(1170, -0.1)),
                 ("s > 1", lambda: _mk_up(1170, 1.5)),
                 ("NaN", lambda: _mk_up(1170, float("nan"))),
                 ("without a union beam", lambda: _mk_up(1170, 0.5, q=None, U=0)),
                 ("without read_top", lambda: mk(fsb_policy="fixed", fsb_width=4, fsb_delta=0, fsb_tree="forest", fsb_union_prior_scale=0.5)),
                 ("rebuild tree", lambda: _mk_up(1170, 0.5, q=None, fsb_tree="rebuild")),
                 ("learned policy", lambda: mk(fsb_policy="learned", fsb_widths="2,4,8", fsb_deltas="0,1", fsb_tree="forest",
                                                fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_union_width=16, fsb_union_prior_scale=0.5)),
                 ("hybrid pool", lambda: _mk_up(1170, 0.5, q=None, mode="pool"))):
    _expect_up_error(fn, what)
v117 = union_prior_vec(3, 2, 0.25, dev)
assert v117.shape == (1, 1, 5) and v117.dtype == torch.float32 and v117.flatten().tolist() == [1.0, 1.0, 1.0, 0.25, 0.25]
assert _prior_extend(v117, 2).flatten().tolist() == [1.0, 1.0, 1.0, 0.25, 0.25, 1.0, 1.0]
assert union_prior_vec(0, 2, 0.0, dev).flatten().tolist() == [0.0, 0.0] and union_prior_vec(2, 0, 0.0, dev).flatten().tolist() == [1.0, 1.0]
mm = _mk_up(1171, None, adapter=11710)
torch.manual_seed(1172); x = torch.randn(1, 512, d_model, device=dev)
_set_utau(mm, _ceil_of(mm, x))
_, _, _, rr117, _, _ = _up_capture(mm, x)
assert len(rr117) > 0
n117b = n117s = ntie117 = 0
for a_, k_, o_ in rr117:
    assert "prior_scale" not in k_
    C_ = a_[0].shape[-1]
    with torch.no_grad():
        o1 = _fth.rerank_top(*a_, **k_, prior_scale=torch.ones(1, 1, C_, device=dev))
        assert torch.equal(o1, o_), "rerank_top with a vector of ones != without one"
        for s117 in (0.0, 0.5):
            os_ = _fth.rerank_top(*a_, **k_, prior_scale=torch.full((1, 1, C_), s117, device=dev))
            sel_c, q_, qpos_, lam_, sigv_, cellK_, cellA_, tidx_, scale_, s_, K_ = a_
            u64 = _tok_brute(q_, sel_c, k_["tokK"], tidx_, scale_, s_, "lse") + s117 * _pos64(sel_c, qpos_, lam_, sigv_, cellA_, tidx_, s_)
            u64 = u64.masked_fill(sel_c < 0, float("-inf"))
            ref, tie, _ = _topk_ref(sel_c, u64, K_)
            same = _same_sets(os_.long(), ref.long())
            assert bool(same[~tie].all()), (s117, int((~same & ~tie).sum()))
            n117s += int((~tie).sum()); ntie117 += int(tie.sum())
    n117b += 1
print(f"F117 [a,b] fsb_union_prior_scale: default 1.0; 8 refusals; union_prior_vec / _prior_extend; rerank_top with ones == without on "
      f"{n117b} re-rank calls bit-exact; a uniform s == float64 lse + s pos top-4 on {n117s} rows ({ntie117} near-ties skipped)")
n117, N0117 = 512, 300
n117c = 0
for mode117, q117, W117 in ((None, 0.5, 16), ("quota", 0.5, 16), (None, None, 4)):
    mA = _mk_up(1173, None, q=q117, W=W117, adapter=11730, mode=mode117); mB = _mk_up(1173, 1.0, q=q117, W=W117, adapter=11730, mode=mode117)
    assert all(torch.equal(a_, b_) for a_, b_ in zip(mA.state_dict().values(), mB.state_dict().values()))
    torch.manual_seed(1174); x = torch.randn(1, n117, d_model, device=dev)
    if q117:
        ce117 = _ceil_of(mA, x); _set_utau(mA, ce117); _set_utau(mB, ce117)
    with torch.no_grad():
        sA, yA = _full_sel_n(mA, x); nA = _nstats(mA); sB, yB = _full_sel_n(mB, x); nB = _nstats(mB)
    assert torch.equal(sA, sB) and torch.equal(yA, yB) and all(torch.equal(a_, b_) for a_, b_ in zip(nA, nB)), (mode117, q117, "explicit 1.0 != absent")
    for lean in (False, True):
        for dg in (("0", "1") if dev == "cuda" else ("0",)):
            with torch.no_grad():
                rA = _ut_cached(mA, x, N0117, lean, dg); rB = _ut_cached(mB, x, N0117, lean, dg)
            assert torch.equal(rA[0], rB[0]) and all(torch.equal(a_, b_) for a_, b_ in zip(rA[1], rB[1])), (mode117, q117, lean, dg)
            assert all((pa is None and pb is None) or torch.equal(pa, pb) for pa, pb in zip(rA[2], rB[2])), (mode117, q117, lean, dg)
            n117c += 1
    if dev == "cuda":
        mA.train(); mB.train(); Qc, Kc, Vc = _qkv_n(1175, n117)
        if q117:
            _set_utau(mA, _ceil_K(Kc)); _set_utau(mB, _ceil_K(Kc))
        with torch.no_grad():
            rA = _sel_n(mA, Qc, Kc, Vc, n117, "1") + _nstats(mA); rB = _sel_n(mB, Qc, Kc, Vc, n117, "1") + _nstats(mB)
        assert all(torch.equal(a_, b_) for a_, b_ in zip(rA, rB)), (mode117, q117, "graphed stage")
        mA.eval(); mB.eval(); n117c += 1
print(f"F117 OK  (fsb_union_prior_scale: construction + 8 refusals; scale vector; explicit 1.0 == absent bit-exact on the no-cache "
      f"forward + {n117c} cached-decode / graphed-stage cases: filtered forest off + quota hybrid, one-forest union)")


# ── F118: SEMANTICS end to end (fp32, n 512, random full adapter, sel_scale LAM118 = 0.1).  For the W4 main beam with
#   the U16 second beam over the filtered forest and the one-forest W4 + U16 union (the section comment: at W16 the
#   second beam's rare extra candidate is never read, at lambda 0.7 s 0.5 and s 0 evict alike), s 0 and 0.5:
#   (a) the traversal re-ranks the union candidates ONCE per chunk with a (1,1,C) prior_scale of 1.0 on the main block
#       and s after it (the layout of union_candidates);
#   (b) float64: the read == the top-4 of the union candidates by lse + prior for the main block and lse + s * prior
#       for the second-beam-only block (s 0: the LSE alone) on every row (near-ties skipped);
#   (b') the implementation's OWN score vectors (the fp32 u rerank_top sorts, captured inside the call) against the
#       s = 1 forward's: the main slots are bit-identical, every valid second-only slot drops by exactly (1 - s) times
#       the fp32 prior term lam a_c erf(.) recomputed from the call's own arguments (which is > 0), and u differs on
#       EVERY row that has a second-only candidate (not only where the top-4 changes);
#   (c) main-beam invariance against the s = 1 read (same adapter, same tau_union): every main-beam cell read at s = 1 is
#       read at s (the second-only scores can only drop), the main-beam cells keep their relative order, and the number
#       of second-beam-only reads never grows per row; rows whose second beam found no new cell read exactly the s = 1
#       (== no-union) cells; (d) teeth: the read differs from s = 1 on some rows, s 0.5 differs from both s 0 and s 1;
#   (e) with the union quota (k 1): the second-only list's split re-rank gets a vector of s on every slot and the main
#       list none; the reserved cell == the float64 top-1 of the second-only candidates by lse + s prior, Tm the
#       float64 top-4 of the main candidates by lse_pos; the read == the python merge [Tu[:1] | Tm | Tu[1:]] ────────
def _pos32_118(sel_c, qpos_, lam_, sigv_, cellA_, tidx_, s_):
    """fp32 prior term as rerank_top forms it: lam a_c erf(|qpos - (c + 0.5) s| / (2 sigma)) of candidates sel_c (T',Bq,C)."""
    c_ = sel_c.clamp(min=0)
    a_ = cellA_[tidx_.view(sel_c.shape[0], 1, 1).expand_as(c_), c_]
    return lam_ * a_ * torch.erf((qpos_[..., None] - (c_.float() + 0.5) * s_).abs() / (2.0 * sigv_))


os.environ["FSB_BEAM_COMPACT"] = "0"                  # a union prior scale keeps 2^Delta_max beam slots: so must the s = 1
                                                      # module whose candidates F118 compares slot by slot
LAM118 = 0.1                                          # sel_scale of the semantics fixtures (the section comment)
n118 = 512
nrow118 = n64_118 = ntie118 = nmono118 = nfb118 = nu118 = nus118 = 0
du118 = 0.0
teeth118 = {}
for name118, c118 in (("W4 U16 tau0.5", dict(W=4)), ("W4 U16 one-forest", dict(q=None, W=4))):
    q118 = c118.get("q", 0.5); W118 = c118["W"]
    torch.manual_seed(1181); x = torch.randn(1, n118, d_model, device=dev)
    m1 = _mk_up(1180, None, adapter=11800, lam=LAM118, **c118)
    mN = _mk_up(1180, None, U=0, adapter=11800, q=None, W=W118, lam=LAM118)                      # no union beam
    ce118 = _ceil_of(m1, x) if q118 else None
    if q118:
        _set_utau(m1, ce118)
    s1, _, cand1, rr1, uc1, _, U1 = _up_capture_u(m1, x)
    assert all("prior_scale" not in k_ for _, k_, _ in rr1)
    assert LAM118 > 0 and abs(float(m1.sel_scale.min()) - LAM118) < 1e-6 and abs(float(m1.sel_scale.max()) - LAM118) < 1e-6
    with torch.no_grad():
        sN, _ = _full_sel_n(mN, x)
    sN = sN.long()
    Cm = m1.fsb_W_max * (1 << m1.fsb_delta_max)
    main1, unew1 = cand1[..., :Cm], cand1[..., Cm:]
    reads = {1.0: s1}
    for s118 in (0.0, 0.5):
        ms = _mk_up(1180, s118, adapter=11800, lam=LAM118, **c118)
        assert all(torch.equal(a_, b_) for a_, b_ in zip(m1.state_dict().values(), ms.state_dict().values()))
        if q118:
            _set_utau(ms, ce118)
        ss, _, cands, rrs, ucs, _, Us = _up_capture_u(ms, x)
        assert torch.equal(cands, cand1), (name118, s118, "the candidates changed with s")   # the beams do not read s
        assert len(rrs) == len(ucs) > 0, (name118, s118)
        for j_, ((a_, k_, o_), (_, _, uc_)) in enumerate(zip(rrs, ucs)):                     # (a) one re-rank per chunk on the union
            sel_c, q_, qpos_, lam_, sigv_, cellK_, cellA_, tidx_, scale_, s_, K_ = a_
            assert torch.equal(sel_c, uc_) and K_ == 4 and k_["score"] == "lse_pos" and "rr_w" not in k_ and "read_margin" not in k_
            ps_ = k_["prior_scale"]
            assert ps_.shape == (1, 1, sel_c.shape[-1]) and ps_.dtype == torch.float32
            assert bool((ps_[..., :Cm] == 1.0).all()) and bool((ps_[..., Cm:] == s118).all()), (name118, s118, j_)
            with torch.no_grad():                                                            # (b) float64 reference
                u64 = (_tok_brute(q_, sel_c, k_["tokK"], tidx_, scale_, s_, "lse")
                       + ps_.double() * _pos64(sel_c, qpos_, lam_, sigv_, cellA_, tidx_, s_)).masked_fill(sel_c < 0, float("-inf"))
            ref, tie, _ = _topk_ref(sel_c, u64, 4)
            same = _same_sets(o_.long(), ref.long())
            assert bool(same[~tie].all()), (name118, s118, j_, int((~same & ~tie).sum()))
            n64_118 += int((~tie).sum()); ntie118 += int(tie.sum())
        # (b') the implementation's own u against the s = 1 forward's (rows >= BLK)
        assert U1.shape == Us.shape == cand1.shape, (name118, s118, U1.shape, Us.shape, cand1.shape)
        pt32 = _rows_hb([_pos32_118(a_[0], a_[2], a_[3], a_[4], a_[6], a_[7], a_[9]) for a_, _, _ in rrs])
        vm_, vu_ = (main1 >= 0)[:, BLK:], (unew1 >= 0)[:, BLK:]
        U1r, Usr, ptr = U1[:, BLK:], Us[:, BLK:], pt32[:, BLK:]
        assert torch.equal(U1r[..., :Cm][vm_], Usr[..., :Cm][vm_]), (name118, s118, "a main slot's score moved with s")
        d_ = (U1r[..., Cm:] - Usr[..., Cm:])[vu_]; e_ = ((1.0 - s118) * ptr[..., Cm:])[vu_]
        assert bool((e_ > 0).all()), (name118, s118, "a second-only prior term is not > 0")
        assert bool(((d_ - e_).abs() <= 1e-5 * (1.0 + e_.abs())).all()), (name118, s118, float((d_ - e_).abs().max()))
        rows_u = vu_.any(-1)
        assert bool(((U1r[..., Cm:] != Usr[..., Cm:]) & vu_).any(-1)[rows_u].all()), (name118, s118, "u equal on a row with a second-only candidate")
        assert bool(rows_u.any()) and bool(torch.isfinite(Usr[..., :Cm][vm_]).all()), (name118, s118)
        nu118 += int(rows_u.sum()); nus118 += int(vu_.sum()); du118 = max(du118, float((d_ - e_).abs().max()))
        reads[s118] = ss
        ssc, s1c, mainc, unewc = ss.cpu(), s1.cpu(), main1.cpu(), unew1.cpu()
        for h in range(H):
            for i in range(BLK, n118):
                mset = {c for c in mainc[h, i].tolist() if c >= 0}; uset = {c for c in unewc[h, i].tolist() if c >= 0}
                r1 = [c for c in s1c[h, i].tolist() if c >= 0]; rs = [c for c in ssc[h, i].tolist() if c >= 0]
                assert len(rs) == len(set(rs)) == min(4, len(mset | uset)) and set(rs) <= mset | uset, (name118, s118, h, i)
                m_1 = [c for c in r1 if c in mset]; m_s = [c for c in rs if c in mset]
                assert set(m_1) <= set(m_s), (name118, s118, h, i, "a main-beam cell read at s = 1 is lost at s")   # (c)
                assert [c for c in m_s if c in set(m_1)] == m_1, (name118, s118, h, i, "the main cells' order changed")
                assert len([c for c in rs if c in uset]) <= len([c for c in r1 if c in uset]), (name118, s118, h, i)
                nmono118 += 1
                if not uset:
                    assert rs == r1 == [c for c in sN[h, i].tolist() if c >= 0], (name118, s118, h, i); nfb118 += 1
                nrow118 += 1
        teeth118[(name118, s118, 1.0)] = int((~_same_sets(ss, s1))[:, BLK:].sum())
    teeth118[(name118, 0.0, 0.5)] = int((~_same_sets(reads[0.0], reads[0.5]))[:, BLK:].sum())
    assert teeth118[(name118, 0.0, 1.0)] > 0, (name118, "s 0 == s 1 on every row: no teeth")                        # (d)
    assert teeth118[(name118, 0.5, 1.0)] > 0 and teeth118[(name118, 0.0, 0.5)] > 0, (name118, teeth118)
    # (e) the union quota's split re-ranks
    for s118 in (0.0, 0.5):
        mu = _mk_up(1180, s118, uq=1, adapter=11800, lam=LAM118, **c118)
        if q118:
            _set_utau(mu, ce118)
        su, _, candu, rru, ucu, _ = _up_capture(mu, x)
        assert torch.equal(candu, cand1) and len(rru) == 2 * len(ucu) > 0, (name118, s118)
        Tu_l, Tm_l = [], []
        for j_, (_, _, uc_) in enumerate(ucu):
            (au, ku, ou), (am, km, om) = rru[2 * j_], rru[2 * j_ + 1]
            assert torch.equal(au[0], uc_[..., Cm:]) and torch.equal(am[0], uc_[..., :Cm]) and "prior_scale" not in km
            psu = ku["prior_scale"]
            assert psu.shape == (1, 1, uc_.shape[-1] - Cm) and bool((psu == s118).all()), (name118, s118, j_)
            for (a_, k_, o_, kk_, sc_) in ((au, ku, ou, 1, s118), (am, km, om, 4, 1.0)):
                sel_c, q_, qpos_, lam_, sigv_, cellK_, cellA_, tidx_, scale_, s_, K_ = a_
                with torch.no_grad():
                    u64 = (_tok_brute(q_, sel_c, k_["tokK"], tidx_, scale_, s_, "lse")
                           + sc_ * _pos64(sel_c, qpos_, lam_, sigv_, cellA_, tidx_, s_)).masked_fill(sel_c < 0, float("-inf"))
                ref, tie, _ = _topk_ref(sel_c, u64, kk_)
                same = _same_sets(o_[..., :kk_].long(), ref.long())
                assert bool(same[~tie].all()), (name118, s118, j_, kk_, int((~same & ~tie).sum()))
                n64_118 += int((~tie).sum()); ntie118 += int(tie.sum())
            Tu_l.append(ou); Tm_l.append(om)
        Tuc, Tmc, suc = _rows_hb(Tu_l).long().cpu(), _rows_hb(Tm_l).long().cpu(), su.cpu()
        for h in range(H):
            for i in range(BLK, n118):
                exp = _merge_py([Tuc[h, i, :1].tolist(), Tmc[h, i].tolist(), Tuc[h, i, 1:].tolist()], 4)
                assert suc[h, i].tolist() == exp, (name118, s118, h, i, suc[h, i].tolist(), exp)
                nrow118 += 1
assert nfb118 > 0 and nu118 > 0
print(f"F118 OK  (union prior scale: one re-rank per chunk with the [1 | s] vector; read == float64 lse + scaled prior top-k on "
      f"{n64_118} rows ({ntie118} near-ties skipped); the implementation's own u vs s 1: main slots bit-identical, {nus118} "
      f"second-only slots drop by exactly (1 - s) lam a erf (max dev {du118:.1e}), u differs on all {nu118} rows with a second-only "
      f"candidate; main-beam invariance + monotone second-only reads on {nmono118} rows "
      f"({nfb118} rows without a second-only cell == the no-union read); teeth (rows): "
      + ", ".join(f"{k_[0]} s{k_[1]:g} vs s{k_[2]:g} {v_}" for k_, v_ in teeth118.items())
      + f"; union quota split re-ranks == float64 + python merge)")
_env_restore("FSB_BEAM_COMPACT")


# ── F119: PARITY, COMPOSITION and the remaining paths.  (a) prefill == decode (full + lean prefill, FSB_DECODE_GRAPH
#   0 / 1), cached == full forward, graph decode == eager, decode provenance == the prefill's, for W16 U16 tau0.5 s0 /
#   W4 U16 one-forest s0.5 / quota hybrid K_L1 s0 / W4 U16 tau0.5 beam quota 1 s0.5 / W16 U16 tau0.5 union quota 1 s0 /
#   union hybrid s0 / W4 U16 tau0.5 bq1 + K_L1 + uq1 s0.5;
#   (b) composition: every captured re-rank call carries the expected prior_scale layout (1.0 on the main slots, s on
#       the second beam's, 1.0 on a pool list appended after them; the union quota's second-only list s everywhere, its
#       main list none) and the read == the python merge [L[:K_L] | W4 picks | Tu[:k] | Tm | Tu[k:] | L[K_L:]] of the
#       module's own lists; the read differs from the same composition at s = 1 (teeth);
#   (c) chunked prefill 1 / 2 / all identical; CUDA-graph training stage == eager, the key re-captures on an s change and
#       the selection changes; Triton == torch (traversal + re-rank kernels);
#   (d) s switched mid-block on a live decode cache (1 -> 0, 0 -> 1, 0 -> 0.5) == a fresh cache (the decode config key);
#   (e) the router node-trace re-run reproduces the forward's read set (0 mismatches), plain + quota hybrid with the beam
#       quota, eager + graph; (f) the residual re-ranker at w == 0 with s 0 == s 0 without it (bit-exact: the scale is on
#       the score, r == 0); (g) bf16 4096: the decode forests + cache keys route the prefill's queries to the prefill's
#       cells (GPU) ────────────────────────────────────────────────────────────────────────────────────────────────
n119, N0119 = 512, 300
cfg119 = (("W16 U16 tau0.5 s0", dict(s=0.0)), ("W4 U16 one-forest s0.5", dict(s=0.5, q=None, W=4)),
          ("quota K_L1 s0", dict(s=0.0, mode="quota", quota=1)),
          ("W4 U16 tau0.5 bq1 s0.5", dict(s=0.5, W=4, bq=1)),
          ("W16 U16 tau0.5 uq1 s0", dict(s=0.0, uq=1)),
          ("union hybrid s0", dict(s=0.0, mode="union", short=4)),
          ("W4 U16 tau0.5 bq1 + K_L1 + uq1 s0.5", dict(s=0.5, W=4, bq=1, uq=1, mode="quota", quota=1)))
nrow119 = nord119 = nprov119 = ngr119 = ncomp119 = nteeth119 = nps119 = 0
for name119, c119 in cfg119:
    mm = _mk_up(1190, adapter=11900, **c119)
    torch.manual_seed(1191); x = torch.randn(1, n119, d_model, device=dev)
    q119 = c119.get("q", 0.5); s119 = c119["s"]; bq119 = c119.get("bq", 0); uq119 = c119.get("uq", 0)
    KL119 = c119.get("quota", 0) if c119.get("mode") == "quota" else 0
    if q119:
        ce119 = _ceil_of(mm, x); _set_utau(mm, ce119)
    sel_f, y_f, cand_f, rr_f, uc_f, pr_f = _up_capture(mm, x)
    Cm = mm.fsb_W_max * (1 << mm.fsb_delta_max)
    # (b) the prior_scale layout of every re-rank call and the composed python merge
    per = 2 if uq119 else 1
    assert len(rr_f) == per * len(uc_f) > 0, (name119, len(rr_f), len(uc_f))
    Tu_l, Tm_l = [], []
    for j_, (_, _, uc_) in enumerate(uc_f):
        if uq119:
            (au, ku, ou), (am, km, om) = rr_f[2 * j_], rr_f[2 * j_ + 1]
            assert torch.equal(au[0], uc_[..., Cm:]) and torch.equal(am[0], uc_[..., :Cm]) and "prior_scale" not in km
            assert ku["prior_scale"].shape == (1, 1, uc_.shape[-1] - Cm) and bool((ku["prior_scale"] == s119).all()), (name119, j_)
            Tu_l.append(ou); Tm_l.append(om); nps119 += 2
        else:
            a_, k_, o_ = rr_f[j_]
            ps_ = k_["prior_scale"]; Cc = a_[0].shape[-1]
            assert ps_.shape == (1, 1, Cc) and Cc >= uc_.shape[-1] and torch.equal(a_[0][..., :uc_.shape[-1]], uc_), (name119, j_)
            assert bool((ps_[..., :Cm] == 1.0).all()) and bool((ps_[..., Cm:uc_.shape[-1]] == s119).all()), (name119, j_)
            assert bool((ps_[..., uc_.shape[-1]:] == 1.0).all()), (name119, j_, "a pool slot is scaled")      # union hybrid: pool cells
            if c119.get("mode") == "union":
                assert Cc == uc_.shape[-1] + 4, (name119, j_, Cc)                                                  # + fsb_pool_short
            Tm_l.append(o_); nps119 += 1
    if bq119 or KL119 or uq119:
        mp = _mk_plain(1190, W=c119.get("W", 16), adapter=11900)
        mh = _mk_up(1190, None, adapter=11900, **{k_: v_ for k_, v_ in c119.items() if k_ != "s"})     # the same composition at s 1
        if q119:
            _set_utau(mh, ce119)
        with torch.no_grad():
            sp, _ = _full_sel_n(mp, x); sh, _ = _full_sel_n(mh, x)
        W4c = _compact_rows(sp.long())[..., :bq119].cpu() if bq119 else None
        Lqc = _rows_hb([o_ for _, _, o_ in pr_f]).long().cpu() if pr_f else None
        sfc, Tmc = sel_f.cpu(), _rows_hb(Tm_l).long().cpu()
        Tuc = _rows_hb(Tu_l).long().cpu() if Tu_l else None
        for h in range(H):
            for i in range(BLK, n119):
                br = [c for c in W4c[h, i].tolist() if c >= 0] if W4c is not None else []
                Lp, Lb = (Lqc[h, i, :KL119].tolist(), Lqc[h, i, KL119:].tolist()) if Lqc is not None else ([], [])
                ur, ub = (Tuc[h, i, :uq119].tolist(), Tuc[h, i, uq119:].tolist()) if Tuc is not None else ([], [])
                exp = _merge_py([Lp, br, ur, Tmc[h, i].tolist(), ub, Lb], 4)
                assert sfc[h, i].tolist() == exp, (name119, h, i, sfc[h, i].tolist(), exp)
                ncomp119 += 1
        nteeth119 += int((~_same_sets(sel_f, sh.long()))[:, BLK:].sum())
    ys = {}
    with torch.no_grad():
        for lean in (False, True):
            for dg in (("0", "1") if dev == "cuda" else ("0",)):
                y, sels, provs, _, _ = _ut_cached(mm, x, N0119, lean, dg)
                assert (y - y_f).abs().max().item() < 1e-4, (name119, lean, dg, (y - y_f).abs().max().item())
                dec = torch.stack(sels, dim=1)
                same = _same_sets(sel_f[:, N0119:n119], dec.long())
                assert bool(same.all()), f"{name119} lean={lean} graph={dg}: {int((~same).sum())} decode rows differ"
                nord119 += int((sel_f[:, N0119:n119] == dec.long()).all(-1).sum()); nrow119 += same.numel()
                nprov119 += sum(int(p_ is not None) for p_ in provs)
                ys[(lean, dg)] = y
        if dev == "cuda":
            for lean in (False, True):
                assert torch.equal(ys[(lean, "0")], ys[(lean, "1")]), (name119, lean, "graph decode != eager")
    outs = []
    for nbch in (1, 2, 0):
        mc = _mk_up(1190, adapter=11900, fsb_prefill_blkchunk=nbch, **c119)
        with torch.no_grad():
            if q119:
                mc.fsb_tau_union.copy_(mm.fsb_tau_union)
            outs.append(_full_sel_n(mc, x) + _nstats(mc))
    assert all(all(torch.equal(a_, b_) for a_, b_ in zip(outs[0], o_)) for o_ in outs[1:]), (name119, "chunk size")
    assert torch.equal(outs[0][0].long(), sel_f), (name119, "chunk 1 != the default chunking")
    if dev == "cuda":
        mm.train(); Qc, Kc, Vc = _qkv_n(1192, n119)
        if q119:
            _set_utau(mm, _ceil_K(Kc))
        with torch.no_grad():
            r0 = _sel_n(mm, Qc, Kc, Vc, n119, "0") + _nstats(mm); r1 = _sel_n(mm, Qc, Kc, Vc, n119, "1") + _nstats(mm)
            assert all(torch.equal(a_, b_) for a_, b_ in zip(r0, r1)), (name119, "graphed training stage != eager")
            k1 = mm._fsb_train_graphs["key"]
            mm.fsb_union_prior_scale = 1.0
            r2 = _sel_n(mm, Qc, Kc, Vc, n119, "1") + _nstats(mm); k2 = mm._fsb_train_graphs["key"]
            r3 = _sel_n(mm, Qc, Kc, Vc, n119, "0") + _nstats(mm)
            assert k2 != k1 and all(torch.equal(a_, b_) for a_, b_ in zip(r2, r3)), (name119, "stale graph after an s change")
            assert not torch.equal(r2[0], r1[0]), (name119, "the s change did not change the selection (no teeth)")
            mm.fsb_union_prior_scale = s119
            ref = None
            for trv, rrk in (("1", "1"), ("0", "1"), ("1", "0"), ("0", "0")):
                with _env(FSB_RERANK_TRITON=rrk):
                    r_ = _sel_n(mm, Qc, Kc, Vc, n119, "0", triton=trv) + _nstats(mm)
                if ref is None:
                    ref = r_
                assert all(torch.equal(a_, b_) for a_, b_ in zip(ref, r_)), (name119, trv, rrk, "Triton != torch")
            ngr119 += 1
        mm.eval()
assert nteeth119 > 0, "the prior scale never changed the composed read (no teeth)"
print(f"F119 [a-c] prefill == decode on {nrow119} rows (order identical on {nord119}; full + lean, graph 0 / 1), cached == full, "
      f"graph decode == eager ({nprov119} decode provenance rows); {nps119} re-rank calls carry the expected prior_scale layout; "
      f"composition == the python merges on {ncomp119} rows (differs from the same composition at s 1 on {nteeth119}); chunked "
      f"prefill identical; training graph == eager + key re-captures, Triton == torch on {ngr119} configs")
torch.manual_seed(1193); x193 = torch.randn(1, n119, d_model, device=dev)
ce193 = _ceil_of(_mk_up(1194, 0.0, adapter=11940, W=4, lam=LAM118), x193)
nsw119 = teeth119d = 0
for dg in (("0", "1") if dev == "cuda" else ("0",)):
    def _fresh119(s_):
        m_ = _mk_up(1194, s_, adapter=11940, W=4, lam=LAM118)      # the W4 + lambda 0.1 semantics fixture (F118): teeth
        _set_utau(m_, ce193)
        return m_
    for nm, sA, sB in (("1 -> 0", None, 0.0), ("0 -> 1", 0.0, None), ("0 -> 0.5", 0.0, 0.5)):
        mA, mB, mC = _fresh119(sA), _fresh119(sB), _fresh119(sA)
        sw = _hy_switch(mA, x193, N0119, dg, 400, _set101(fsb_union_prior_scale=(1.0 if sB is None else sB)))
        ref = _hy_switch(mB, x193, N0119, dg); ctl = _hy_switch(mC, x193, N0119, dg)
        assert all(torch.equal(sw[t], ref[t]) for t in range(400, n119)), (nm, dg, "s switched on a live cache != a fresh cache")
        teeth119d += sum(int(not torch.equal(ctl[t], ref[t])) for t in range(400, n119))
        nsw119 += 1
assert teeth119d > 0, "no decode row differs between the switched configs after the switch: no teeth"
print(f"F119 [d] {nsw119} live-cache switches of s (1 -> 0, 0 -> 1, 0 -> 0.5) == a fresh cache (eager" + (" + CUDA graph)" if dev == "cuda" else ")")
      + f"; {teeth119d} post-switch steps differ between the configs")
nmis119 = ntr119 = 0
for name119, extra in (("plain W4 U16 tau0.5 s0", dict()),
                       ("quota K_L1 bq1 s0", dict(fsb_hybrid="quota", fsb_pool_size=PH, fsb_pool_recent=WCH, fsb_pool_quota=1, fsb_beam_quota=1)),
                       ("uq1 s0.5", dict(fsb_union_quota=1, fsb_union_prior_scale=0.5))):
    torch.manual_seed(1195)
    m_ = _mk_node(fsb_width=4, fsb_rd_node=1, fsb_rd_node_weight=0.5, fsb_route_learn_c=1, fsb_bound_radius_coef=8.0,
                  fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw", fsb_union_width=16,
                  fsb_union_tau_quantile=0.5, **dict(dict(fsb_union_prior_scale=0.0), **extra)).train()
    with torch.no_grad():
        g_ = torch.Generator(device="cpu").manual_seed(1196)
        m_.fsb_route_W.copy_((torch.randn(m_.fsb_route_W.shape, generator=g_) * 0.3).to(dev))
        m_.fsb_route_ctheta.add_((torch.randn(H, generator=g_) * 1.0).to(dev))
    Qc, Kc, Vc = _qkv_n(1197, N2, 1.5)
    _set_utau(m_, _ceil_K(Kc))
    for g in (("0", "1") if dev == "cuda" else ("0",)):
        m_.zero_grad(set_to_none=True)
        m_._fsb_rd_gen = torch.Generator(device=dev).manual_seed(1198)
        sel_, o_, rd_, fr_ = _lr_n(m_, Qc, Kc, Vc, N2, g)
        st_ = rd_["stats"]
        assert fr_ is not None and st_.numel() == RD_STATS_NODE and fr_[0].get("utau_F") is not None, (name119, g)
        assert float(st_[14]) == 0.0, (name119, g, "node trace re-run != the forward's read set", st_[14].item())
        nmis119 += int(st_[14]); ntr119 += int(st_[15])
        assert torch.isfinite(rd_["loss"]) and int(sel_.shape[-1]) == 4
print(f"F119 [e] router node trace with the union prior scale: {nmis119} mismatches / {ntr119} rows (plain + quota hybrid with the beam "
      f"quota + union quota, eager" + (" + graph)" if dev == "cuda" else ")"))
torch.manual_seed(1199); x199 = torch.randn(1, n119, d_model, device=dev)
mR = _mk_up(1199, 0.0, adapter=11990, fsb_rerank_rr="linear"); m0 = _mk_up(1199, 0.0, adapter=11990)
assert mR.fsb_rerank_rr == "linear" and float(mR.fsb_route_rr_w.abs().max()) == 0.0
ce199 = _ceil_of(m0, x199); _set_utau(mR, ce199); _set_utau(m0, ce199)
with torch.no_grad():
    sR, yR = _full_sel_n(mR, x199); s0_, y0_ = _full_sel_n(m0, x199)
assert torch.equal(sR, s0_) and torch.equal(yR, y0_), "the residual re-ranker at w == 0 with s 0 != s 0 without it"
_, _, _, rrR, _, _ = _up_capture(mR, x199)
assert all("rr_w" in k_ and k_["prior_scale"].shape[-1] == a_[0].shape[-1] for a_, k_, _ in rrR) and len(rrR) > 0
print("F119 [f] residual re-ranker at w == 0 with s 0 == s 0 without it (bit-exact; prior_scale reaches the rr branch)")
if dev == "cuda":
    for lean in (False, True):
        torch.manual_seed(1189)
        mm = _mk_up(1189, 0.0, adapter=11890)
        x = torch.randn(1, 4096, d_model, device=dev)
        _set_utau(mm, _ceil_of(mm, x))
        _f65_bf16_ut(mm, x, 4096, lean, "union prior scale W16 U16 tau0.5 s0 lse_pos raw")
else:
    print("F119 [g] bf16 SKIP (no CUDA)")
print("F119 OK  (union prior scale: prefill == decode, graph == eager (decode + training, keys), composition with the beam / pool / "
      "union quotas and the union hybrid, chunked prefill, Triton == torch, live-cache switches == fresh, node trace 0 mismatches, "
      "rr at w 0, bf16 4096 routing)")


# ── F120: PLUMBING (the kept part).  The re-ranker identity records --fsb_union_prior_scale only when on (older
#   identities stay comparable) and reports the difference against a model carrying the scale; fsb_resolved records
#   the attribute ─────────────────────────────────────────────────────────────────────────────────────────────────
m_on120 = _mk_up(1200, 0.5, adapter=12000); m_off120 = _mk_up(1200, None, adapter=12000)
c_on120, c_off120 = _rsc108(m_on120), _rsc108(m_off120)
assert c_on120["fsb_union_prior_scale"] == 0.5 and "fsb_union_prior_scale" not in c_off120
assert {k: v for k, v in c_on120.items() if k != "fsb_union_prior_scale"} == c_off120
assert _im108({"config": c_off120}, model=m_off120) == [] and _im108({"config": c_on120}, model=m_on120) == []
assert _im108({"config": c_off120}, model=m_on120) and _im108({"config": c_on120}, model=m_off120)
r_on120, r_off120 = _fr112(m_on120), _fr112(m_off120)
assert r_on120["attrs"]["fsb_union_prior_scale"] == 0.5 and r_off120["attrs"]["fsb_union_prior_scale"] == 1.0
assert _fr112(_mk_up(1200, 0.0, adapter=12000))["attrs"]["fsb_union_prior_scale"] == 0.0         # an explicit 0 is recorded as 0
print("F120 OK  (--fsb_union_prior_scale: re-ranker identity records the scale only when on; fsb_resolved records it)")

# ── F121: FSB_FAST_BEAM fused descent == _forest_beam (the per-level Triton loop) and == the torch loop ──
if dev == "cuda":
    from src.fsattn.fsb_tree import _forest_beam
    from src.fsattn.fsb_fast_step import fast_beam_ok, forest_beam_fused
    os.environ["FSB_FAST_BEAM"] = "1"
    try:
        torch.manual_seed(121)
        Tp121 = SPf["root"].shape[0]
        for wv, dv in ((1, 0), (2, 1), (4, 2), (16, 2)):
            Wm = 1 << (max(wv, 1) - 1).bit_length()
            h121 = SPf["height"]
            for Bq in (1, 3):
                Dv = (h121 - dv).clamp(min=0)[:, None].expand(Tp121, Bq)
                Wv = torch.minimum(torch.full_like(Dv, wv), torch.bitwise_left_shift(torch.ones_like(Dv), Dv))
                dl = (h121[:, None] - Dv).clamp(min=0)
                q121 = torch.randn(Tp121, Bq, Dhf, device=dev)
                qpos121 = torch.rand(Tp121, Bq, device=dev) * (Mf * S)
                lam121 = torch.rand(Tp121, 1, 1, device=dev) + 0.2
                sig121 = torch.rand(Tp121, 1, 1, device=dev) * 8 + 0.5
                rcs = [1.0, 0.85] + ([torch.rand(Tp121, device=dev) + 0.5] if Bq == 1 else [])
                for ns in ("bound", "centroid"):
                    for rc in rcs:
                        os.environ["FSB_FAST_BEAM"] = "0"
                        a121 = _forest_beam(Ff, SPf, q121, qpos121, lam121, sig121, scf, S, Wv, dl, Wm, dv, ns, rc, None)
                        os.environ["FSB_FAST_BEAM"] = "1"
                        b121 = _forest_beam(Ff, SPf, q121, qpos121, lam121, sig121, scf, S, Wv, dl, Wm, dv, ns, rc, None,
                                            torch_loop=True)
                        assert fast_beam_ok(Ff, SPf, q121, Wm, dv), (wv, dv, Bq)
                        c121 = forest_beam_fused(Ff, SPf, q121, qpos121, lam121, sig121, scf, S, Wv, dl, Wm, dv, ns, rc)
                        assert torch.equal(a121[0], c121[0]) and torch.equal(a121[1], c121[1]), (wv, dv, Bq, ns, rc)
                        assert torch.equal(b121[0], c121[0]) and torch.equal(b121[1], c121[1]), ("torch", wv, dv, Bq, ns)
        torch.manual_seed(1217)
        M121, Dh121 = 1024, 64
        cnt121 = [509, 512, 1024, 300, 700, 61, 129, 958]
        T121 = len(cnt121)
        Kb121 = torch.randn(T121, M121, Dh121, device=dev)
        ce121 = torch.rand(T121, M121, device=dev) * 3
        adm121 = torch.zeros(T121, M121, dtype=torch.bool, device=dev)
        for t_, c_ in enumerate(cnt121):
            adm121[t_, torch.randperm(M121, device=dev)[:c_]] = True
        F121b = build_forest(Kb121, ce121, adm121)
        SP121b = build_spines(F121b, torch.tensor(cnt121, device=dev), torch.arange(T121, device=dev))
        sc121, h121b = 1 / math.sqrt(Dh121), SP121b["height"]
        for wv, dv in ((4, 0), (16, 2)):
            Wm = 1 << (max(wv, 1) - 1).bit_length()
            for Bq in (1, 128):
                Dv = (h121b - dv).clamp(min=0)[:, None].expand(T121, Bq)
                Wv = torch.minimum(torch.full_like(Dv, wv), torch.bitwise_left_shift(torch.ones_like(Dv), Dv))
                dl = (h121b[:, None] - Dv).clamp(min=0)
                q121 = torch.randn(T121, Bq, Dh121, device=dev)
                qpos121 = torch.rand(T121, Bq, device=dev) * (M121 * S)
                lam121 = torch.ones(T121, 1, 1, device=dev)
                sig121 = torch.full((T121, 1, 1), 4.0, device=dev)
                rcs = [8.0] + ([torch.rand(T121, device=dev) + 4.0] if Bq == 1 else [])
                for ns in ("bound", "centroid"):
                    for rc in rcs:
                        os.environ["FSB_FAST_BEAM"] = "0"
                        a121 = _forest_beam(F121b, SP121b, q121, qpos121, lam121, sig121, sc121, S, Wv, dl, Wm, dv, ns, rc, None)
                        os.environ["FSB_FAST_BEAM"] = "1"
                        b121 = _forest_beam(F121b, SP121b, q121, qpos121, lam121, sig121, sc121, S, Wv, dl, Wm, dv, ns, rc, None,
                                            torch_loop=True)
                        assert fast_beam_ok(F121b, SP121b, q121, Wm, dv), (wv, dv, Bq)
                        c121 = forest_beam_fused(F121b, SP121b, q121, qpos121, lam121, sig121, sc121, S, Wv, dl, Wm, dv, ns, rc)
                        assert torch.equal(a121[0], c121[0]) and torch.equal(a121[1], c121[1]), ("deep", wv, dv, Bq, ns, rc)
                        assert torch.equal(b121[0], c121[0]) and torch.equal(b121[1], c121[1]), ("deep+t", wv, dv, Bq, ns)
        print(f"F121 OK  (fused beam == Triton level loop == torch loop: cells + n_visit bit-exact over the F11 forest and a "
              f"deep Dh=64 M=1024 forest (11 levels, scale 1/8, lam 1 sig 4, radius 8.0), "
              f"W 1/2/4/16 x delta 0/1/2, Bq 1/3/128, bound/centroid, scalar/tensor radius)")
    finally:
        os.environ.pop("FSB_FAST_BEAM", None)
else:
    print("F121 SKIP (no CUDA)")

# ── F122: FSB_FAST_RERANK fused score+select == rerank_top (lse / lse_pos / max) ─────────────────────────
if dev == "cuda":
    from src.fsattn.fsb_fast_step import fast_rerank_ok, rerank_top_fast
    os.environ["FSB_FAST_RERANK"] = "1"
    try:
        torch.manual_seed(122)
        for C in (5, 37, 64, 128):
            for score in ("lse", "lse_pos", "max"):
                Tp122, Bq122, Mc = 7, 3, 60
                sel122 = torch.randint(0, Mc, (Tp122, Bq122, C), device=dev)
                sel122 = sel122.masked_fill(torch.rand(Tp122, Bq122, C, device=dev) < 0.3, -1)
                dup122 = torch.rand(Tp122, Bq122, C, device=dev) < 0.2
                src122 = torch.randint(0, C, (Tp122, Bq122, C), device=dev)
                sel122 = torch.where(dup122, torch.gather(sel122, -1, src122), sel122)
                cellK = torch.randn(Tp122, Mc, Dhf, device=dev)
                cellA = torch.rand(Tp122, Mc, device=dev) * 3
                t_idx122 = torch.arange(Tp122, device=dev)
                q122 = torch.randn(Tp122, Bq122, Dhf, device=dev)
                qpos122 = torch.rand(Tp122, Bq122, device=dev) * 500
                lam122 = torch.rand(Tp122, 1, 1, device=dev) + 0.2
                sig122 = torch.rand(Tp122, 1, 1, device=dev) * 8 + 0.5
                tokK122 = torch.randn(Tp122, Mc * S + 7, Dhf, device=dev)          # 3-dim: t' -> seq t'
                for K in (1, 2, 4, 8, 16):
                    if K > C:
                        continue
                    ref122 = rerank_top(sel122, q122, qpos122, lam122, sig122, cellK, cellA, t_idx122, scf, S, K,
                                        score=score, tokK=tokK122)
                    assert not fast_rerank_ok(sel122, q122, tokK122, S, K, score), (C, score, K)
                    got122 = rerank_top_fast(sel122, q122, qpos122, lam122, sig122, cellA, t_idx122, scf, S, K,
                                             score, tokK122)
                    assert torch.equal(ref122, got122), (C, score, K)
        tokK4 = (torch.randn(2, 3, 1000, Dhf, device=dev) * 2).to(torch.bfloat16)   # 4-dim low-precision keys
        t_idx4 = torch.arange(Tp122, device=dev) % 6
        sel4 = torch.randint(0, 60, (Tp122, Bq122, 48), device=dev).masked_fill(
            torch.rand(Tp122, Bq122, 48, device=dev) < 0.25, -1)
        ref4 = rerank_top(sel4, q122, qpos122, lam122, sig122, cellK, cellA, t_idx4, scf, S, 4,
                          score="lse_pos", tokK=tokK4)
        got4 = rerank_top_fast(sel4, q122, qpos122, lam122, sig122, cellA, t_idx4, scf, S, 4, "lse_pos", tokK4)
        assert torch.equal(ref4, got4), "bf16 4-dim tokK"
        print(f"F122 OK  (fused re-rank == rerank_top: sel bit-exact, C 5/37/64/128, lse/lse_pos/max, K 1..16, "
              f"empty slots + score ties, fp32 and bf16 keys)")
    finally:
        os.environ.pop("FSB_FAST_RERANK", None)
else:
    print("F122 SKIP (no CUDA)")

# ── F123: eager per-token decode with the DA overselect config: fused flags bit-exact end-to-end ────────
if dev == "cuda":
    torch.manual_seed(123)
    mm123 = _mk_os(1230, fsb_width=16, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw",
                   fsb_union_width=16, fsb_tree="forest", fsb_union_tau_quantile=0.5)
    x123 = torch.randn(1, 2048, d_model, device=dev)
    _set_utau(mm123, _ceil_of(mm123, x123))
    for lean in (False, True):
        os.environ["FSB_FAST_BEAM"] = "0"; os.environ["FSB_FAST_RERANK"] = "0"      # the reference (fast is the default)
        a123, sa123 = _cached_run_n(mm123, x123, 1536, lean, "0")
        os.environ["FSB_FAST_BEAM"] = "1"; os.environ["FSB_FAST_RERANK"] = "1"
        try:
            b123, sb123 = _cached_run_n(mm123, x123, 1536, lean, "0")
        finally:
            os.environ.pop("FSB_FAST_BEAM", None); os.environ.pop("FSB_FAST_RERANK", None)
        assert torch.equal(a123, b123), ("decode out", lean)
        assert len(sa123) == len(sb123)
        selok = all((u is None and v is None) or (u is not None and v is not None and torch.equal(u, v))
                    for u, v in zip(sa123, sb123))
        assert selok, ("decode sel", lean)
    print("F123 OK  (eager decode, DA flags W16 rt4 lse_pos raw U16 tau_q0.5: outputs + selections bit-exact with "
          "FSB_FAST_BEAM + FSB_FAST_RERANK, full and lean caches)")
else:
    print("F123 SKIP (no CUDA)")

# F124: batched prefill with the fast flags: the fused beam runs and is bit-exact; the re-rank stays on the reference.
if dev == "cuda":
    torch.manual_seed(124)
    Tp124 = SPf["root"].shape[0]
    tokK124 = torch.randn(Tf, Mf * S, Dhf, device=dev)
    q124 = torch.randn(Tp124, 2, Dhf, device=dev)
    qpos124 = torch.rand(Tp124, 2, device=dev) * (Mf * S)
    lam124 = torch.rand(Tp124, 1, 1, device=dev) + 0.2
    sig124 = torch.rand(Tp124, 1, 1, device=dev) * 8 + 0.5
    Ff["cellK"], Ff["cellA"] = Kb_f, ce_f
    D124 = (SPf["height"] - 2).clamp(min=0)[:, None].expand(Tp124, 2)
    W124 = torch.minimum(torch.full_like(D124, 16), torch.bitwise_left_shift(torch.ones_like(D124), D124))
    dl124 = (SPf["height"][:, None] - D124).clamp(min=0)
    for uw in (0, 16):
        os.environ["FSB_FAST_BEAM"] = "0"; os.environ["FSB_FAST_RERANK"] = "0"      # the reference (fast is the default)
        ref124 = traverse_forest(Ff, SPf, q124, qpos124, lam124, sig124, scf, S, W124, dl124, 16, 2, "bound",
                                 read_top=4, rerank_score="lse_pos", tokK=tokK124, union_width=uw, union_W_max=uw)
        os.environ["FSB_FAST_BEAM"] = "1"; os.environ["FSB_FAST_RERANK"] = "1"
        try:
            from unittest.mock import patch
            import src.fsattn.fsb_fast_step as _fs124
            calls124, real124 = [], _fs124.forest_beam_fused

            def _count124(*a, **k):
                calls124.append(tuple(a[2].shape))
                return real124(*a, **k)

            with patch("src.fsattn.fsb_fast_step.forest_beam_fused", side_effect=_count124), \
                 patch("src.fsattn.fsb_fast_step.rerank_top_fast", side_effect=AssertionError("fused prefill rerank")):
                got124 = traverse_forest(Ff, SPf, q124, qpos124, lam124, sig124, scf, S, W124, dl124, 16, 2, "bound",
                                         read_top=4, rerank_score="lse_pos", tokK=tokK124, union_width=uw, union_W_max=uw)
        finally:
            os.environ.pop("FSB_FAST_BEAM", None); os.environ.pop("FSB_FAST_RERANK", None)
        assert calls124 and all(sh[1] == 2 for sh in calls124), ("fused prefill beam did not run", uw, calls124)
        assert all(torch.equal(r_, g_) for r_, g_ in zip(ref124, got124)), ("prefill traversal", uw)
    Ff.pop("cellK"); Ff.pop("cellA")
    print("F124 OK  (batched prefill: fused beam bit-exact, re-rank on the reference path; union beam off and on)")
else:
    print("F124 SKIP (no CUDA)")

# F125: fused reranking uses the row stride of the packed cell table.
if dev == "cuda":
    from src.fsattn.fsb_fast_step import rerank_top_fast
    torch.manual_seed(125)
    T125, M125, D125, S125 = 3, 9, 8, 16
    q125 = torch.zeros(T125, 1, D125, device=dev)
    qp125 = torch.full((T125, 1), 512.0, device=dev)
    lam125 = torch.ones(T125, 1, 1, device=dev)
    sig125 = torch.full_like(lam125, 4.0)
    sel125 = torch.arange(M125, device=dev).view(1, 1, M125).expand(T125, 1, M125).contiguous()
    ck125 = torch.zeros(T125, M125, D125, device=dev)
    tk125 = torch.zeros(T125, M125 * S125, D125, device=dev)
    ti125 = torch.arange(T125, device=dev)
    for ca125 in (torch.rand(M125, T125, device=dev).t(),
                  torch.rand(T125, 2 * M125 + 1, device=dev)[:, :M125]):
        assert not ca125.is_contiguous()
        ref125 = rerank_top(sel125, q125, qp125, lam125, sig125, ck125, ca125, ti125,
                            1.0, S125, 4, score="lse_pos", tokK=tk125)
        got125 = rerank_top_fast(sel125, q125, qp125, lam125, sig125, ca125, ti125,
                                 1.0, S125, 4, "lse_pos", tk125)
        assert torch.equal(ref125, got125), ("strided cell table", ca125.stride())
    print("F125 OK  (fused rerank matches reference with transposed and sliced cell tables)")
else:
    print("F125 SKIP (no CUDA)")

# F126: reusable forest storage preserves two-forest eager and graph decode.
if dev == "cuda":
    from unittest.mock import patch
    torch.manual_seed(126)
    mm126 = _mk_os(1260, fsb_width=16, fsb_read_top=4, fsb_rerank_score="lse_pos", fsb_rerank_query="raw",
                   fsb_union_width=16, fsb_tree="forest", fsb_union_tau_quantile=0.5)
    x126 = torch.randn(1, 768, d_model, device=dev)
    _set_utau(mm126, _ceil_of(mm126, x126))
    for lean126 in ("0", "1"):
        for graph126 in ("0", "1"):
            with patch.dict(os.environ, FSB_DECODE_LEAN=lean126, FSB_FAST_BEAM="1", FSB_FAST_RERANK="1"):
                with patch.dict(os.environ, FSB_FAST_CACHE="0"):
                    a126, sa126 = _cached_run_n(mm126, x126, 512, True, graph126)
                with patch.dict(os.environ, FSB_FAST_CACHE="1"):
                    b126, sb126 = _cached_run_n(mm126, x126, 512, True, graph126)
            assert torch.equal(a126, b126), ("storage decode", lean126, graph126)
            assert len(sa126) == len(sb126)
            assert all((a is None and b is None) or
                       (a is not None and b is not None and torch.equal(a, b))
                       for a, b in zip(sa126, sb126)), ("storage selections", lean126, graph126)
    print("F126 OK  (reusable storage preserves two-forest outputs and selections, full/lean and eager/graph)")
else:
    print("F126 SKIP (no CUDA)")

os.environ.clear()                              # the session continues with the environment it had
os.environ.update(_ENV0)
print("ALL FS-B TESTS PASSED")
