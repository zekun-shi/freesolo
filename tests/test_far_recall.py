"""Far-recall masks and weighted CE of experiments/scaling/train_dist.py (--far_recall_*).  Run from the repo
root on a host with a CUDA GPU (skips under pytest without one):

    python tests/test_far_recall.py [--data <uint16 Llama-32k .bin>] [--n_real 64] [--device cuda]

T1 hand-built windows: a copy at distance 600 fires at exactly the expected tokens, 400 does not, the dmin edge,
   a repeated short filler never fires (a filler whose period exceeds dmin fires only in its 2nd repetition),
   a cross-EOS copy does not fire (generic and EOS-target), source switches; torch (batched) == the numpy twin
T2 beta = 0 equals plain CE in value and gradient (fp32 and bf16 under autocast); beta > 0 equals a float64 reference
T3 the numpy twin == torch masks (batched, on --device) on the real windows
T4 the design's verbatim masks (spec_far_recall_masks below) vs this implementation on the real windows; every
   difference must be the design's same-document off-by-one (an EOS at the copied source)
T5 real-window densities (rho, cont, start, sw, weight mass, mean weight, documents per window) and the mask cost
T7 --far_recall_span / --far_recall_cont_mode on an MQ-shaped window with the real Llama-32k ids (needles
   'is: ▁ d1..d7 .', answer ': ▁ v1 , ▁ v2 , ▁ v3 .'): the exact weighted targets of span 1, span 3, align and both
T8 defaults == the reference masks (_ref_far_recall_* below) bitwise; flags on: torch == numpy, real densities

Without --data the corpus-dependent sections (T3-T5 and the T8 real-window densities) are skipped.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    import pytest
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from experiments.scaling.train_dist import (FAR_RECALL_EOS, FAR_RECALL_GAP, far_recall_loss, far_recall_masks,
                                            _gram_codes, _gram_codes_np, _prev_unique, _prev_unique_np)

EOS = FAR_RECALL_EOS
L = 4097


def far_recall_masks_np(w, k=4, m=4, dmin=512, gap=FAR_RECALL_GAP, eos=FAR_RECALL_EOS, parts=False, span=1,
                        cont_mode="match"):
    """numpy twin of far_recall_masks for ONE window w (L,) (x = w[:-1], y = w[1:]) -> (far, sw) bool (L - 1,);
    parts=True (diagnostics) -> (continuation, start, sw).  span / cont_mode: see far_recall_masks."""
    w = np.asarray(w, dtype=np.int64)
    if min(k, m) <= 4 and len(w) and w.max() >= 32768:
        raise ValueError("far-recall masks: token id >= 32768 breaks the exact n-gram packing (n <= 4)")
    L = len(w)
    N = L - 1
    x, y = w[:-1], w[1:]
    t = np.arange(N)
    e = (w == eos).astype(np.int64)
    doc = np.cumsum(e) - e                      # EOS strictly before each position (an EOS belongs to its document)
    doc_tgt = doc[1:]                           # document of y_t
    p = _prev_unique_np(_gram_codes_np(w, k))[np.maximum(t - k + 1, 0)]
    pc = np.maximum(p, 0)
    src = pc + k
    cont = (t >= k - 1) & (p >= 0) & (t - src > dmin) & (w[src] == y) & (doc[pc] == doc_tgt)
    q = _prev_unique_np(_gram_codes_np(w, m))[np.minimum(t + 1, L - m)]
    qc = np.maximum(q, 0)
    start = ((t + 1 <= L - m) & (q >= 0) & (t - q > dmin) & ((q == 0) | (w[np.maximum(qc - 1, 0)] != x))
             & (doc[qc] == doc_tgt) & ~cont)
    prev = np.concatenate([[-1], np.maximum.accumulate(np.where(start, t, -1))[:-1]])   # previous start (< t)
    sw = start & (prev >= 0) & (t - prev <= gap) & (np.abs(q - q[np.maximum(prev, 0)]) > gap)
    if cont_mode == "align":                    # the continuation term without w[src] == y_t (see far_recall_masks)
        cont = (t >= k - 1) & (p >= 0) & (t - src > dmin) & (doc[pc] == doc_tgt)
    if span > 1:                                # spread each start / switch over the next span - 1 targets
        st0, sw0 = start, sw
        start, sw = st0.copy(), sw0.copy()
        for j in range(1, span):
            same = doc_tgt[j:] == doc_tgt[:-j]
            start[j:] |= st0[:-j] & same
            sw[j:] |= sw0[:-j] & same
    return (cont, start, sw) if parts else (cont | start, sw)


# ── the design's code (verbatim apart from the names; returns its parts too) ──
@torch.no_grad()
def _spec_prev_unique(h):  # (B,L) hashes -> index of the single earlier equal hash, else -1
    hs, o = torch.sort(h, dim=-1, stable=True)
    same = torch.zeros_like(o, dtype=torch.bool); same[:, 1:] = hs[:, 1:] == hs[:, :-1]
    sec = same.clone(); sec[:, 1:] &= ~same[:, :-1]
    pv = torch.full_like(o, -1); pv[:, 1:] = torch.where(sec[:, 1:], o[:, :-1], torch.full_like(o[:, 1:], -1))
    return torch.full_like(o, -1).scatter_(1, o, pv)


@torch.no_grad()
def spec_far_recall_masks(x, y, k=4, m=4, dmin=512, gap=64, eos=2):
    w = torch.cat([x, y[:, -1:]], 1); B, L = w.shape; N = L - 1
    t = torch.arange(N, device=w.device); doc = (w == eos).long().cumsum(1)
    def grams(n):
        h = torch.zeros(B, L - n + 1, dtype=torch.long, device=w.device)
        for j in range(n): h = h * 1000003 + w[:, j:L - n + 1 + j]
        return h
    p = _spec_prev_unique(grams(k))[:, (t - k + 1).clamp_min(0)]; src = (p + k).clamp_min(0)
    cont = (t >= k - 1) & (p >= 0) & (t - src > dmin) & (w.gather(1, src) == y) & (doc.gather(1, src) == doc[:, :N])
    q = _spec_prev_unique(grams(m))[:, (t + 1).clamp_max(L - m)]; qc = q.clamp_min(0)
    start = (t + 1 <= L - m) & (q >= 0) & (t - q > dmin) & ((q == 0) | (w.gather(1, (q - 1).clamp_min(0)) != x)) & (doc.gather(1, qc) == doc[:, :N]) & ~cont
    sw = torch.zeros_like(start)
    for b in range(B):
        ts = start[b].nonzero().squeeze(1)
        if ts.numel() > 1:
            qs = q[b, ts]; ok = ((ts[1:] - ts[:-1]) <= gap) & ((qs[1:] - qs[:-1]).abs() > gap)
            sw[b, ts[1:][ok]] = True
    return cont | start, sw, dict(cont=cont, start=start, src=src, q=q)


# ── the reference masks (verbatim apart from the names): the T8 bitwise reference ──
def _ref_far_recall_masks_np(w, k=4, m=4, dmin=512, gap=FAR_RECALL_GAP, eos=FAR_RECALL_EOS, parts=False):
    """numpy twin of far_recall_masks for ONE window w (L,) (x = w[:-1], y = w[1:]) -> (far, sw) bool (L - 1,);
    parts=True (diagnostics) -> (continuation, start, sw)."""
    w = np.asarray(w, dtype=np.int64)
    if min(k, m) <= 4 and len(w) and w.max() >= 32768:
        raise ValueError("far-recall masks: token id >= 32768 breaks the exact n-gram packing (n <= 4)")
    L = len(w)
    N = L - 1
    x, y = w[:-1], w[1:]
    t = np.arange(N)
    e = (w == eos).astype(np.int64)
    doc = np.cumsum(e) - e                      # EOS strictly before each position (an EOS belongs to its document)
    doc_tgt = doc[1:]                           # document of y_t
    p = _prev_unique_np(_gram_codes_np(w, k))[np.maximum(t - k + 1, 0)]
    pc = np.maximum(p, 0)
    src = pc + k
    cont = (t >= k - 1) & (p >= 0) & (t - src > dmin) & (w[src] == y) & (doc[pc] == doc_tgt)
    q = _prev_unique_np(_gram_codes_np(w, m))[np.minimum(t + 1, L - m)]
    qc = np.maximum(q, 0)
    start = ((t + 1 <= L - m) & (q >= 0) & (t - q > dmin) & ((q == 0) | (w[np.maximum(qc - 1, 0)] != x))
             & (doc[qc] == doc_tgt) & ~cont)
    prev = np.concatenate([[-1], np.maximum.accumulate(np.where(start, t, -1))[:-1]])   # previous start (< t)
    sw = start & (prev >= 0) & (t - prev <= gap) & (np.abs(q - q[np.maximum(prev, 0)]) > gap)
    return (cont, start, sw) if parts else (cont | start, sw)


@torch.no_grad()
def _ref_far_recall_masks(x, y, k=4, m=4, dmin=512, gap=FAR_RECALL_GAP, eos=FAR_RECALL_EOS):
    """P1 far-recall masks of a batch of packed windows (definitions at FAR_RECALL_EOS above): x, y (B, N) int64
    inputs / targets (y = x shifted by one) -> (far, sw) bool (B, N), far = far continuation | far recall start,
    sw = source switch (subset of the starts).  No model involved; far_recall_masks_np is the per-window twin."""
    w = torch.cat([x, y[:, -1:]], 1)
    B, L = w.shape
    N = L - 1
    t = torch.arange(N, device=w.device)
    e = (w == eos).long()
    doc = e.cumsum(1) - e                       # EOS strictly before each position (an EOS belongs to its document)
    doc_tgt = doc[:, 1:]                        # document of y_t
    p = _prev_unique(_gram_codes(w, k))[:, (t - k + 1).clamp_min(0)]   # earlier start of the k-gram ending at x_t
    pc = p.clamp_min(0)
    src = pc + k
    cont = (t >= k - 1) & (p >= 0) & (t - src > dmin) & (w.gather(1, src) == y) & (doc.gather(1, pc) == doc_tgt)
    q = _prev_unique(_gram_codes(w, m))[:, (t + 1).clamp_max(L - m)]   # earlier start of the m-gram beginning at y_t
    qc = q.clamp_min(0)
    start = ((t + 1 <= L - m) & (q >= 0) & (t - q > dmin) & ((q == 0) | (w.gather(1, (qc - 1).clamp_min(0)) != x))
             & (doc.gather(1, qc) == doc_tgt) & ~cont)
    last = torch.cummax(torch.where(start, t.expand(B, N), torch.full_like(q, -1)), dim=1).values
    prev = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], 1)                # previous start (< t)
    sw = start & (prev >= 0) & (t - prev <= gap) & ((q - q.gather(1, prev.clamp_min(0))).abs() > gap)
    return cont | start, sw


# ── helpers ──
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def background(rng, n=L):
    return rng.integers(100, 32000, size=n).astype(np.int64)   # no EOS, no repeated 4-gram in practice


def torch_masks(ws, dev, **kw):
    w = torch.as_tensor(np.stack(ws), device=dev)
    far, sw = far_recall_masks(w[:, :-1], w[:, 1:], **kw)
    return far.cpu().numpy(), sw.cpu().numpy()


def fired(mask):
    return set(np.flatnonzero(mask).tolist())


def copy_expect(c, n=16, k=4):
    """far positions t (target y_t = w[t + 1]) of a clean far copy of an n-token span placed at c: the recall start
    (target w[c]) and the continuations (targets w[c + k .. c + n - 1])."""
    return {c - 1} | {c + j - 1 for j in range(k, n)}


def t1_hand_built(dev):
    print("T1 hand-built windows", flush=True)
    rng = np.random.default_rng(7)
    a, n = 1000, 16
    cases = []
    for D in (600, 400, 513, 514):                    # copy distance (target position - source position)
        w = background(rng); w[a + D:a + D + n] = w[a:a + n]
        far, sw = far_recall_masks_np(w)
        exp = copy_expect(a + D, n) if D - 1 > 512 else set()
        check(f"copy at distance {D}: {'fires' if exp else 'silent'}", fired(far) == exp and not sw.any(),
              f"far {sorted(fired(far))[:3]}.. ({int(far.sum())} tokens, expected {len(exp)}), sw {int(sw.sum())}")
        cases.append(w)
    # repeated short filler (RULER-noise-like period 24): never fires; a needle copied 1500 back inside it does
    filler = background(rng, 24)
    w = np.tile(filler, L // 24 + 1)[:L].copy()
    far, sw = far_recall_masks_np(w)
    check("repeated filler (period 24) never fires", not far.any() and not sw.any(), f"far {int(far.sum())} sw {int(sw.sum())}")
    cases.append(w)
    w2 = w.copy(); needle = background(rng, n); w2[701:701 + n] = needle; w2[2203:2203 + n] = needle  # 1502 back, off-phase
    far, sw = far_recall_masks_np(w2)
    check("needle copy inside the filler fires exactly at the copy", fired(far) == copy_expect(2203, n) and not sw.any(),
          f"far {int(far.sum())} tokens")
    cases.append(w2)
    # a filler whose period exceeds dmin: the 2nd repetition is a unique far copy (fires: its start at t = 599 and the
    # continuations t = 603 .. 1202, the last 3 through the 4-grams straddling into the 3rd), repetitions >= 3 never
    filler = background(rng, 600)
    w = np.tile(filler, L // 600 + 1)[:L].copy()
    far, sw = far_recall_masks_np(w)
    pos = np.flatnonzero(far)
    check("long filler (period 600 > dmin): fires only on its 2nd repetition",
          fired(far) == {599} | set(range(603, 1203)) and not sw.any(), f"{len(pos)} tokens in [{pos.min()}, {pos.max()}]")
    cases.append(w)
    # cross-EOS copy: the source in the previous document
    w = background(rng); w[a + 300] = EOS; w[a + 600:a + 600 + n] = w[a:a + n]
    far, sw = far_recall_masks_np(w)
    check("cross-EOS copy (distance 600, EOS between) does not fire", not far.any(), f"far {int(far.sum())}")
    cases.append(w)
    ctl = w.copy(); ctl[a + 300] = 5          # control: the same window without the EOS fires
    check("  control: the same window without the EOS fires", fired(far_recall_masks_np(ctl)[0]) == copy_expect(a + 600, n))
    cases.append(ctl)
    # EOS-target: two consecutive documents end with the same 8 tokens; the 2nd document's EOS copies the 1st's EOS
    w = background(rng); e1, e2 = 1500, 2300
    tail = background(rng, 8)
    w[e1 - 8:e1] = tail; w[e1] = EOS; w[e2 - 8:e2] = tail; w[e2] = EOS
    far, sw = far_recall_masks_np(w)
    sfar = spec_far_recall_masks(torch.as_tensor(w[None, :-1]), torch.as_tensor(w[None, 1:]))[0][0].numpy()
    check("EOS-target cross-document copy does not fire", not far.any(),
          f"this {int(far.sum())}; the design's verbatim rule fires at {sorted(fired(sfar))} (target = EOS at {e2})")
    cases.append(w)
    # source switch: starts 30 apart copying sources 200 apart -> one switch; sources 40 apart or starts 100 apart -> none
    for (a1, a2, c1, c2, want) in ((200, 400, 1500, 1530, True), (200, 240, 1500, 1530, False),
                                   (200, 400, 1500, 1600, False)):
        w = background(rng); w[c1:c1 + n] = w[a1:a1 + n]; w[c2:c2 + n] = w[a2:a2 + n]
        far, sw = far_recall_masks_np(w)
        exp_far = copy_expect(c1, n) | copy_expect(c2, n)
        exp_sw = {c2 - 1} if want else set()
        check(f"switch: starts {c2 - c1} apart, sources {a2 - a1} apart -> {'switch' if want else 'no switch'}",
              fired(far) == exp_far and fired(sw) == exp_sw, f"far {int(far.sum())} sw {sorted(fired(sw))}")
        cases.append(w)
    # torch (batched, device) == numpy on the hand-built windows
    tf, ts = torch_masks(cases, dev)
    ok = all(np.array_equal(tf[i], far_recall_masks_np(c)[0]) and np.array_equal(ts[i], far_recall_masks_np(c)[1])
             for i, c in enumerate(cases))
    check(f"torch ({dev}, batch {len(cases)}) == numpy twin on the hand-built windows", ok)
    return cases


def t2_loss(dev):
    print("T2 weighted CE", flush=True)
    g = torch.Generator().manual_seed(0)
    B, N, V = 2, 512, 32000
    base = torch.randn(B, N, V, generator=g) * 2
    y = torch.randint(0, V, (B, N), generator=g).to(dev)
    far = (torch.rand(B, N, generator=g) < 0.05).to(dev)
    sw = far & (torch.rand(B, N, generator=g) < 0.3).to(dev)
    # fp32 logits
    la = base.clone().to(dev).requires_grad_(True)
    lb = base.clone().to(dev).requires_grad_(True)
    loss_a, _ = far_recall_loss(la, y, far, sw, 0.0, 0.0)
    loss_b = F.cross_entropy(lb.view(-1, V), y.view(-1))
    loss_a.backward(); loss_b.backward()
    dv = abs(loss_a.item() - loss_b.item()); dg = (la.grad - lb.grad).abs().max().item()
    check("beta 0 == plain CE (fp32): value and gradient", dv <= 1e-6 * loss_b.item() and dg <= 1e-6 * lb.grad.abs().max().item(),
          f"|dloss| {dv:.2e} (loss {loss_b.item():.6f}), max |dgrad| {dg:.2e} (max |grad| {lb.grad.abs().max().item():.2e})")
    # bf16 logits under bf16 autocast, the training numerics (the model's lm_head output is bf16): bitwise, in the
    # training shape (1, 4096) and in a batch of 2
    for (Bb, Nb) in ((1, 4096), (2, 512)):
        gb = torch.Generator().manual_seed(1)
        lbase = (torch.randn(Bb, Nb, V, generator=gb) * 2).to(dev, torch.bfloat16)
        yb = torch.randint(0, V, (Bb, Nb), generator=gb).to(dev)
        fb = (torch.rand(Bb, Nb, generator=gb) < 0.05).to(dev)
        la = lbase.clone().requires_grad_(True)
        lb = lbase.clone().requires_grad_(True)
        with torch.autocast(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=torch.bfloat16):
            loss_a, _ = far_recall_loss(la, yb, fb, fb, 0.0, 0.0)
            loss_b = F.cross_entropy(lb.view(-1, V), yb.view(-1))
        loss_a.backward(); loss_b.backward()
        ndiff = int((la.grad != lb.grad).sum())
        check(f"beta 0 == plain CE bitwise (bf16 logits, bf16 autocast, ({Bb}, {Nb})): value and gradient",
              torch.equal(loss_a.detach(), loss_b.detach()) and ndiff == 0,
              f"loss {loss_a.item():.7f} vs {loss_b.item():.7f}, gradient elements differing {ndiff}/{la.grad.numel()}")
    # beta 9 / 9 vs a float64 reference
    la = base.clone().to(dev).requires_grad_(True)
    l64 = base.clone().double().to(dev).requires_grad_(True)
    loss_a, st = far_recall_loss(la, y, far, sw, 9.0, 9.0)
    w = 1.0 + 9.0 * far.double() + 9.0 * sw.double()
    lt = F.cross_entropy(l64.view(-1, V), y.view(-1), reduction="none").view(B, N)
    ref = (w * lt).sum() / w.sum()
    loss_a.backward(); ref.backward()
    dv = abs(loss_a.item() - ref.item()); dg = (la.grad.double() - l64.grad).abs().max().item()
    stv = st.tolist()
    exp = [far.float().mean().item(), sw.float().mean().item(), ((w * far).sum() / w.sum()).item(), lt.mean().item(),
           w.mean().item()]
    check("beta 9 / switch 9 == float64 sum(w l) / sum(w), and the stats", dv < 1e-5 and dg < 1e-8
          and np.allclose(stv, exp, rtol=1e-5), f"|dloss| {dv:.2e}, max |dgrad| {dg:.2e}, stats {np.round(stv, 4).tolist()}")
    check("  the weighted loss differs from plain CE", abs(loss_a.item() - lt.mean().item()) > 1e-3,
          f"{loss_a.item():.4f} vs {lt.mean().item():.4f}")


def real_windows(path, n, seed=0):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    stride = L
    val_start = int((len(data) // stride) * 0.99)
    ids = np.sort(np.random.default_rng(seed).choice(val_start, n, replace=False))
    return ids, [data[c * stride:(c + 1) * stride].astype(np.int64) for c in ids]


def window_stats(ws, beta=9.0, beta_sw=9.0):
    rows = []
    for w in ws:
        c, s, sw = far_recall_masks_np(w, parts=True)
        far = c | s
        wt = 1.0 + beta * far + beta_sw * sw
        rows.append([far.mean(), c.mean(), s.mean(), sw.mean(), (wt * far).sum() / wt.sum(), wt.mean(),
                     int((w == EOS).sum()), int(far.sum()) + int(sw.sum())])
    return np.array(rows)


def fmt_stats(r, label):
    m = r.mean(0)
    return (f"{label}: rho {100 * m[0]:.3f}% (cont {100 * m[1]:.3f}% + start {100 * m[2]:.3f}%) | sw {100 * m[3]:.3f}% "
            f"({m[3] * (L - 1):.1f}/window) | weight mass {100 * m[4]:.2f}% | mean w {m[5]:.4f} | EOS/window {m[6]:.2f} | "
            f"count {m[7]:.1f}")


def t3_t4(ws, label, dev):
    # T3 twin agreement (torch batched on dev vs numpy per window)
    t0 = time.time(); npm = [far_recall_masks_np(w) for w in ws]; t_np = (time.time() - t0) / len(ws)
    tf, ts = [], []
    for i in range(0, len(ws), 16):
        f_, s_ = torch_masks(ws[i:i + 16], dev); tf.extend(f_); ts.extend(s_)
    agree = all(np.array_equal(tf[i], npm[i][0]) and np.array_equal(ts[i], npm[i][1]) for i in range(len(ws)))
    check(f"T3 numpy twin == torch ({dev}, batches of 16) on {label}", agree,
          f"far tokens {int(sum(f.sum() for f, _ in npm))}, sw tokens {int(sum(s.sum() for _, s in npm))}")
    # T4 the design's verbatim code vs this implementation
    sfar, ssw, src, q, scont = [], [], [], [], []
    for i in range(0, len(ws), 16):
        w_t = torch.as_tensor(np.stack(ws[i:i + 16]), device=dev)
        a_, b_, sp = spec_far_recall_masks(w_t[:, :-1], w_t[:, 1:])
        sfar.append(a_.cpu().numpy()); ssw.append(b_.cpu().numpy()); src.append(sp["src"].cpu().numpy())
        q.append(sp["q"].cpu().numpy()); scont.append(sp["cont"].cpu().numpy())
    sfar, ssw, src, q, scont = (np.concatenate(v) for v in (sfar, ssw, src, q, scont))
    myfar, mysw = np.stack([a for a, _ in npm]), np.stack([b for _, b in npm])
    wa = np.stack(ws)
    dfar = np.argwhere(sfar != myfar)
    explained = 0
    for b, t in dfar:
        s_pos = src[b, t] if scont[b, t] else q[b, t]
        explained += int(sfar[b, t] and not myfar[b, t] and wa[b, s_pos] == EOS and wa[b, t + 1] == EOS)
    dsw = np.argwhere(ssw != mysw)
    sw_near = sum(int(any(bb == b and abs(tt - t) <= 64 for bb, tt in dfar)) for b, t in dsw)
    check(f"T4 design's verbatim masks == this implementation except its EOS-source leak, on {label}",
          explained == len(dfar) and sw_near == len(dsw),
          f"{len(dfar)} far tokens differ, {explained} of them = the design firing on an EOS target copied from the "
          f"previous document's EOS; {len(dsw)} sw tokens differ ({sw_near} next to such a token); "
          f"rho design {100 * sfar.mean():.4f}% vs {100 * myfar.mean():.4f}%")
    return t_np


def t3_t4_t5(path, n_real, dev):
    print(f"T3/T4/T5 real windows of {path}", flush=True)
    ids, ws = real_windows(path, n_real)
    t_np = t3_t4(ws, f"{len(ws)} random train windows", dev)
    # T5 densities and cost
    r = window_stats(ws)
    print("  " + fmt_stats(r, f"{len(ws)} random train windows (rng 0)"), flush=True)
    x = torch.as_tensor(ws[0][None, :-1], device=dev); y = torch.as_tensor(ws[0][None, 1:], device=dev)
    for _ in range(3):
        far_recall_masks(x, y)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        far_recall_masks(x, y)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"  cost: numpy twin {1e3 * t_np:.2f} ms/window (data worker); torch {1e3 * (time.time() - t0) / 50:.2f} ms per "
          f"(1, 4096) micro-batch on {dev}", flush=True)


# ── T7 / T8: --far_recall_span / --far_recall_cont_mode ──
COLON, SPACE, COMMA, DOT = 28747, 28705, 28725, 28723              # Llama-32k ids (the eval's generated_ids)
DIGIT = [28734, 28740, 28750, 28770, 28781, 28782, 28784, 28787, 28783, 28774]   # '0'..'9'


def mq_window(seed=11, A=3000):
    """background + four needles '<key> is : ▁ d1..d7 .' at 400/800/1200/1600 and, at A, the answer
    'are : ▁ v1 , ▁ v2 , ▁ v3 .' (values 1874701, 3540352, 6228093; first digits all differ)."""
    rng = np.random.default_rng(seed)
    w = background(rng)
    IS, ARE = 1500, 1600
    vals = ["1874701", "3540352", "6228093", "9015566"]
    for a, v in zip((400, 800, 1200, 1600), vals):
        needle = list(background(rng, 3)) + [IS, COLON, SPACE] + [DIGIT[int(c)] for c in v] + [DOT]
        w[a:a + len(needle)] = needle
    ans = [ARE, COLON, SPACE]
    for i, v in enumerate(vals[:3]):
        ans += [DIGIT[int(c)] for c in v] + ([COMMA, SPACE] if i < 2 else [DOT])
    w[A:A + len(ans)] = ans
    return w


def t7_mq_window(dev):
    print("T7 span / align on an MQ-shaped window", flush=True)
    A = 3000
    w = mq_window(A=A)
    T = lambda js: {j - 1 for j in js}                     # window position of a target -> its t (y_t = w[t + 1])
    v1, v2, v3 = range(A + 3, A + 10), range(A + 12, A + 19), range(A + 21, A + 28)
    # default (span 1, match): ':' of v1 (start), v1's digits 3..7 (continuation: the 4-gram ':' '▁' d1 d2 matches the
    # needle), the '▁' of v2 / v3 (start + switch), their digits 4..7 (continuation), the final '.' (the needle's own
    # '.'); v1's d1 d2, v2 / v3's d1..d3 and the commas carry no extra weight
    p1_far = T({A + 1, A + 11, A + 20, A + 28} | set(v1[3:]) | set(v2[3:]) | set(v3[3:]) | {A + 5})
    p1_sw = T({A + 11, A + 20})
    span3 = T({A + 2, A + 3, A + 12, A + 13, A + 21, A + 22})        # the next 2 targets after each start
    span3_sw = T({A + 12, A + 13, A + 21, A + 22})
    align = T({A + 10, A + 19, A + 29})       # the copy exits: the two ',' and the token after the final '.'
    want = {("match", 1): (p1_far, p1_sw), ("match", 3): (p1_far | span3, p1_sw | span3_sw),
            ("align", 1): (p1_far | align, p1_sw), ("align", 3): (p1_far | span3 | align, p1_sw | span3_sw)}
    wt = torch.as_tensor(w[None], device=dev)
    for (mode, span), (ef, es) in want.items():
        far, sw = far_recall_masks_np(w, span=span, cont_mode=mode)
        tf, ts = far_recall_masks(wt[:, :-1], wt[:, 1:], span=span, cont_mode=mode)
        ok = fired(far) == ef and fired(sw) == es
        check(f"MQ window span {span} {mode}: weighted targets exactly as designed", ok,
              f"far extra {sorted(fired(far) - ef)} missing {sorted(ef - fired(far))}; sw extra {sorted(fired(sw) - es)} "
              f"missing {sorted(es - fired(sw))}")
        check(f"  torch ({dev}) == numpy, span {span} {mode}",
              np.array_equal(tf[0].cpu().numpy(), far) and np.array_equal(ts[0].cpu().numpy(), sw))
    # the per-token weights at beta 9 / switch 9 on v2 ('▁' d1 d2 d3 d4) and on v1's ',' (documentation of the design)
    for (mode, span) in (("match", 1), ("match", 3), ("align", 3)):
        far, sw = far_recall_masks_np(w, span=span, cont_mode=mode)
        wv = 1 + 9 * far.astype(int) + 9 * sw.astype(int)
        print(f"    weights span {span} {mode}: v2 '▁' d1 d2 d3 d4 = {[int(wv[j - 1]) for j in range(A + 11, A + 16)]}, "
              f"v1 ',' = {int(wv[A + 9])}, final '.' = {int(wv[A + 27])}", flush=True)
    # span must stay inside the matched m-gram: span 4 = m marks d3 as well
    far, sw = far_recall_masks_np(w, span=4)
    check("span 4 (= m) also marks d3' (the whole matched 4-gram)", (A + 14 - 1) in fired(far) and (A + 14 - 1) in fired(sw))
    return w


def t8_defaults_and_flags(cases, data, n_real, dev):
    print("T8 defaults == reference bitwise; flags on: torch == numpy, densities", flush=True)
    ws = list(cases)
    if data and os.path.exists(data):
        ws += real_windows(data, n_real, seed=1)[1]
    ok_np = all(all(np.array_equal(a, b) for a, b in zip(far_recall_masks_np(w), _ref_far_recall_masks_np(w)))
                and all(np.array_equal(a, b) for a, b in zip(far_recall_masks_np(w, parts=True),
                                                             _ref_far_recall_masks_np(w, parts=True))) for w in ws)
    check(f"numpy masks (far, sw, parts) at the defaults == the reference on {len(ws)} windows", ok_np)
    ok_t = True
    for i in range(0, len(ws), 16):
        wt = torch.as_tensor(np.stack(ws[i:i + 16]), device=dev)
        a, b = far_recall_masks(wt[:, :-1], wt[:, 1:]), _ref_far_recall_masks(wt[:, :-1], wt[:, 1:])
        c = far_recall_masks(wt[:, :-1], wt[:, 1:], span=1, cont_mode="match")
        ok_t &= all(torch.equal(x_, y_) for x_, y_ in zip(a, b)) and all(torch.equal(x_, y_) for x_, y_ in zip(c, b))
    check(f"torch masks at the defaults (implicit and explicit) == the reference on {len(ws)} windows ({dev})", ok_t)
    for (span, mode) in ((2, "match"), (3, "align"), (4, "align"), (3, "match"), (1, "align")):
        agree = True
        for i in range(0, len(ws), 16):
            wt = torch.as_tensor(np.stack(ws[i:i + 16]), device=dev)
            tf, ts = far_recall_masks(wt[:, :-1], wt[:, 1:], span=span, cont_mode=mode)
            for j, w in enumerate(ws[i:i + 16]):
                nf, ns = far_recall_masks_np(w, span=span, cont_mode=mode)
                agree &= np.array_equal(tf[j].cpu().numpy(), nf) and np.array_equal(ts[j].cpu().numpy(), ns)
        check(f"torch == numpy with span {span} {mode} on {len(ws)} windows", agree)
    # flags only ADD weight: every default far / sw target stays marked (the starts are the same in both modes)
    sup = all(np.all(far_recall_masks_np(w, span=3, cont_mode="align")[0] >= far_recall_masks_np(w)[0])
              and np.all(far_recall_masks_np(w, span=3, cont_mode="align")[1] >= far_recall_masks_np(w)[1]) for w in ws)
    check("span 3 align marks a superset of the default far and switch targets", sup)
    if data and os.path.exists(data):
        real = ws[len(cases):]
        for (span, mode) in ((1, "match"), (3, "match"), (1, "align"), (3, "align")):
            r = []
            for w in real:
                far, sw = far_recall_masks_np(w, span=span, cont_mode=mode)
                wt = 1.0 + 9.0 * far + 9.0 * sw
                r.append([far.mean(), sw.mean(), (wt * far).sum() / wt.sum(), wt.mean()])
            m = np.mean(r, 0)
            print(f"    real windows ({len(real)}), span {span} {mode}: far {100 * m[0]:.3f}% sw {100 * m[1]:.3f}% | "
                  f"weight mass on far tokens {100 * m[2]:.2f}% (beta 9 / 9) | mean w {m[3]:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--n_real", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cases = t1_hand_built(a.device)
    t2_loss(a.device)
    cases.append(t7_mq_window(a.device))
    t8_defaults_and_flags(cases, a.data, a.n_real, a.device)
    if a.data and os.path.exists(a.data):
        t3_t4_t5(a.data, a.n_real, a.device)
    else:
        print("(no --data: T3-T5 skipped)")
    if a.device.startswith("cuda"):
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
