import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed, DistributedDataParallelKwargs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.scaling.model import FreeSoloLM, MODEL_CONFIGS
from experiments.scaling.config import apply_config, build_attn_kwargs, is_config_mode


# Far-recall weighted CE: per-token weights w_t = 1 + beta * f_t + beta_sw * chi_t, where f_t (far continuation
# or far recall start) and chi_t (source switch) are exact n-gram statistics of each packed window's own tokens
# (no model in the loop). Far continuation: y_t continues the unique earlier occurrence of the k-gram ending at
# x_t, the source more than dmin positions back; far recall start: the m-gram beginning at y_t occurred exactly
# once before, more than dmin back, with a different token before that occurrence; source switch: a recall start
# within GAP tokens of the previous one, copied from a source more than GAP away from that start's source.
# Every match lies in one document: no EOS from the earlier occurrence's first token to x_t.
FAR_RECALL_EOS = 2       # Llama-32k </s>: appended after every document by tokenize_fineweb*.py
FAR_RECALL_GAP = 64      # source-switch window (tokens)


def _gram_base(n):
    # n <= 4: exact packing of token ids < 2^15 (Llama-32k ids < 32000) in 60 bits of int64, no overflow;
    # n > 4: 64-bit multiplicative hash (int64 wrap-around, identical in numpy and torch)
    return 32768 if n <= 4 else 1000003


def _gram_codes_np(w, n):
    L = len(w)
    h = np.zeros(L - n + 1, dtype=np.int64)
    for j in range(n):
        h = h * _gram_base(n) + w[j:L - n + 1 + j]
    return h


def _prev_unique_np(h):
    # codes (M,) -> (M,) index of the earlier position with the same code if exactly one earlier position has
    # it (this is its second occurrence), else -1
    o = np.argsort(h, kind="stable")
    hs = h[o]
    same = np.zeros(len(h), dtype=bool)
    same[1:] = hs[1:] == hs[:-1]
    sec = same.copy()
    sec[1:] &= ~same[:-1]
    pv = np.full(len(h), -1, dtype=np.int64)
    pv[1:] = np.where(sec[1:], o[:-1], -1)
    out = np.full(len(h), -1, dtype=np.int64)
    out[o] = pv
    return out


def _gram_codes(w, n):
    # (B, L) int64 token ids -> (B, L - n + 1) code of the n-gram starting at each position (_gram_codes_np)
    L = w.shape[1]
    h = torch.zeros_like(w[:, :L - n + 1])
    for j in range(n):
        h = h * _gram_base(n) + w[:, j:L - n + 1 + j]
    return h


def _prev_unique(h):
    # (B, M) codes -> (B, M) index of the earlier position with the same code if exactly one earlier position
    # has it (this is its second occurrence), else -1 (_prev_unique_np)
    hs, o = torch.sort(h, dim=-1, stable=True)
    same = torch.zeros_like(hs, dtype=torch.bool)
    same[:, 1:] = hs[:, 1:] == hs[:, :-1]
    sec = same.clone()
    sec[:, 1:] &= ~same[:, :-1]
    pv = torch.full_like(o, -1)
    pv[:, 1:] = torch.where(sec[:, 1:], o[:, :-1], torch.full_like(o[:, 1:], -1))
    return torch.full_like(o, -1).scatter_(1, o, pv)


@torch.no_grad()
def far_recall_masks(x, y, k=4, m=4, dmin=512, gap=FAR_RECALL_GAP, eos=FAR_RECALL_EOS, span=1, cont_mode="match"):
    # x, y (B, N) int64 inputs / targets (y = x shifted by one) -> (far, sw) bool (B, N), far = far continuation
    # | far recall start, sw = source switch (a subset of the starts).  span (1..m, default 1): a start (and a
    # switch) also marks the next span - 1 targets of the same document.  cont_mode "match" (default): the
    # continuation needs y_t == the far source's next token; "align" drops the y_t test (context-only weight).
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
    if cont_mode == "align":
        cont = (t >= k - 1) & (p >= 0) & (t - src > dmin) & (doc.gather(1, pc) == doc_tgt)
    if span > 1:
        st0, sw0 = start, sw
        start, sw = st0.clone(), sw0.clone()
        for j in range(1, span):
            same = doc_tgt[:, j:] == doc_tgt[:, :-j]
            start[:, j:] |= st0[:, :-j] & same
            sw[:, j:] |= sw0[:, :-j] & same
    return cont | start, sw


def far_recall_loss(logits, y, far, sw, beta, beta_sw):
    # w_t = 1 + beta far_t + beta_sw sw_t (no gradient through w), loss = sum w l / sum w.  The per-token CE is
    # the plain path's own op on the same logits (no fp32 pre-cast), summed in fp32, so beta = beta_sw = 0
    # reproduces the plain mean CE bitwise.  Returns (loss, stats), stats a detached fp32 tensor [far fraction,
    # sw fraction, weight mass on far tokens, plain mean CE, mean weight] (no host sync).
    lt = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none").view(y.shape).float()
    with torch.no_grad():
        w = 1.0 + beta * far.float() + beta_sw * sw.float()
        stats = torch.stack([far.float().mean(), sw.float().mean(), (w * far).sum() / w.sum(),
                             lt.detach().mean(), w.mean()])
    return (w * lt).sum() / w.sum(), stats


class DistPreTokenizedDataset(IterableDataset):
    def __init__(self, path, seq_len, rank, world_size, val_split=False, seed=42):
        self.path = path
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.val_split = val_split
        self.seed = seed

    def __iter__(self):
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        stride = self.seq_len + 1
        total_chunks = len(data) // stride
        val_start = int(total_chunks * 0.99)
        lo, hi = (val_start, total_chunks) if self.val_split else (0, val_start)

        wi = torch.utils.data.get_worker_info()
        nw = wi.num_workers if wi is not None else 1
        wid = wi.id if wi is not None else 0
        gid = self.rank * nw + wid
        gn = self.world_size * nw

        chunk_ids = np.arange(lo + gid, hi, gn)
        rng = np.random.default_rng(self.seed + gid)
        if not self.val_split:
            rng.shuffle(chunk_ids)
        for cid in chunk_ids:
            s = cid * stride
            chunk = data[s:s + stride].astype(np.int64)
            if len(chunk) < stride:
                continue
            yield torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


class DistFineWebEduDataset(IterableDataset):
    def __init__(self, seq_len, rank, world_size, subset="sample-10BT", seed=42):
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.subset = subset
        self.seed = seed
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("gpt2")

    def __iter__(self):
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node

        ds = load_dataset("HuggingFaceFW/fineweb-edu", name=self.subset,
                          split="train", streaming=True)
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)
        ds = split_dataset_by_node(ds, rank=self.rank, world_size=self.world_size)

        buf = []
        n = self.seq_len + 1
        for ex in ds:
            buf.extend(self.tok.encode(ex["text"]))
            while len(buf) >= n:
                chunk = buf[:n]
                buf = buf[self.seq_len:]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:], dtype=torch.long))


def parse_args():
    p = argparse.ArgumentParser(description="Distributed FreeSoloLM training")
    p.add_argument("--config", default="1.3b",
                   help="MODEL_CONFIGS key (debug/small/340m/1.3b/3b) for the "
                        "argparse path, OR a path to a configs/attn/<variant>.yaml "
                        "(ending in .yaml/.yml) to enable OmegaConf config mode. In "
                        "config mode, base.yaml + the variant + any 'key=value' CLI "
                        "dotlist overrides populate the same args below.")
    p.add_argument("--attn_type", default="fsattn", choices=["fsattn", "standard"])
    p.add_argument("--n_layers", type=int, default=None,
                   help="override config n_layers (iso-param)")
    p.add_argument("--vocab", type=int, default=None,
                   help="override config vocab_size (e.g. 32000 for the Llama tokenizer)")
    p.add_argument("--long_range_variant", default="cell_topk", choices=["cell_topk"])
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--sel_strict_causal", type=int, default=None,
                   help="1 = strictly-causal per-query cell routing (else block-causal). "
                        "None => env FS_STRICT_CAUSAL. Recorded in ckpt meta + printed per-rank.")
    p.add_argument("--cell_stride", type=int, default=64,
                   help="token-cell size for the long-range branch.")
    p.add_argument("--cell_topk", type=int, default=1,
                   help="# cells kept for exact within-cell retrieval.")
    p.add_argument("--cell_select", default="score", choices=["score"],
                   help="long-range cell selection rule.")
    p.add_argument("--disable_short_range", action="store_true", default=False,
                   help="LR-only ablation: train with no short-range window.")
    p.add_argument("--disable_long_range", action="store_true", default=False,
                   help="SR-only ablation: train with no long-range branch (= sliding window).")
    p.add_argument("--sel_content_only", action="store_true", default=False,
                   help="Select long-range cells by content (q.k_bar) only, dropping the "
                        "sel_scale*ceil*distance selection term.")
    p.add_argument("--fs_nope_longrange", action="store_true", default=False,
                   help="Do not apply RoPE'd Q to the long-range branch (un-rotated Q) -> "
                        "translation-invariant long range that extrapolates flat.")
    p.add_argument("--fs_uniform_sr", action="store_true", default=False,
                   help="Uniform-window short-range: a plain windowed softmax (no distance "
                        "bias), pairs with a larger --window_size (e.g. 512). Also fixes the "
                        "long-range sigma buffer at --sigma_init.")
    p.add_argument("--sigma_init", type=float, default=4.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--require_freesolo", action="store_true", default=False,
                   help="Hard guard: refuse to train unless the resolved config is FreeSolo "
                        "(fsattn + long_range_variant=cell_topk + cell_select=score + "
                        "sel_strict_causal=1 + NoPE long-range branch, i.e. --fs_nope_longrange "
                        "or --no_rope).")
    p.add_argument("--fused_adam", action="store_true", default=False,
                   help="Use fused AdamW (torch.optim.AdamW(fused=True)); ~5-10%% throughput, free.")
    p.add_argument("--tf32", action="store_true", default=False,
                   help="set_float32_matmul_precision('high') — TF32 matmuls.")
    p.add_argument("--no_long_range_recompute", action="store_true", default=False,
                   help="Disable the long-range backward recompute (bit-identical gradients; "
                        "faster when memory is ample).")
    p.add_argument("--fs_nope", action="store_true",
                   help="Disable RoPE in the short-range branch (use_rope=False).")
    p.add_argument("--no_rope", action="store_true",
                   help="Disable RoPE for any attn_type (use_rope=False on the wrapper).")
    p.add_argument("--per_gpu_batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--lr_schedule", choices=["cosine", "wsd"], default="cosine")
    p.add_argument("--cooldown_start_step", type=int, default=None,
                   help="WSD: completed updates before linear decay begins")
    p.add_argument("--checkpoint_steps", default="",
                   help="additional full-state checkpoint steps, comma separated")
    p.add_argument("--data_sha256", default=None,
                   help="WSD: checksum verified by the scaling launcher")
    p.add_argument("--validation_chunks", type=int, default=2560,
                   help="WSD: first chunks of the held-out 1 percent, shared across ranks")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=["adamw"], default="adamw")
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_subset", default="sample-10BT")
    p.add_argument("--pretokenized", default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output_dir", default="./runs/default")
    p.add_argument("--run_name", default=None)
    p.add_argument("--mlflow_uri", default="", help='empty disables MLflow')
    p.add_argument("--wandb_project", default="", help='wandb project name (empty disables wandb)')
    p.add_argument("--resume", default=None)
    p.add_argument("--max_seq_len", type=int, default=None,
                   help="Override config max_seq_len (CPT/long-context, e.g. 4096).")
    p.add_argument("--init_from", default=None,
                   help="Load model weights only from an accelerate checkpoint dir (CPT warm-start).")
    # FS-B: threshold admission + a balanced content tree with query-dependent beam traversal replacing
    # the FS / FS-L long-range selection (src/fsattn/fsb_tree.py). All off unless --fsb is given.
    p.add_argument("--fsb", action="store_true", default=False,
                   help="FS-B long-range branch: admit eligible cells with max-key-norm >= tau_h into a "
                        "per-block balanced content tree, then beam-search it per query (W, Delta).")
    p.add_argument("--fsb_widths", default="1,2,4", help="beam-width menu (comma list)")
    p.add_argument("--fsb_deltas", default="0,1,2",
                   help="height-relative stop-depth menu: D=max(0,H-Delta); a depth-D node holds <=2^Delta cells. "
                        "max(W)*2^max(Delta) must be <=16 (fused kernel slots).")
    p.add_argument("--fsb_policy", default="fixed", choices=["fixed", "learned"],
                   help="fixed: constant (--fsb_width, --fsb_delta) for every query (reference policy); "
                        "learned: per-query (W,Delta) sampled from a small controller (REINFORCE + baseline).")
    p.add_argument("--fsb_width", type=int, default=4, help="fixed-policy beam width")
    p.add_argument("--fsb_delta", type=int, default=0, help="fixed-policy height-relative depth offset")
    p.add_argument("--fsb_node_score", default="bound", choices=["bound", "centroid"],
                   help="internal-node priority: optimistic subtree bound (q.mu+|q|r)/sqrt(d)+lam*A*erf, or centroid")
    p.add_argument("--fsb_route_cost", type=float, default=0.5, help="cost of one scored node in token-read units")
    p.add_argument("--fsb_bound_radius_coef", type=float, default=1.0,
                   help="weight of the ||q|| r_v term in the optimistic node bound (1.0 = the bound; 0 = centroid)")
    p.add_argument("--fsb_leaf_radius", type=int, default=0,
                   help="1: cells carry the radius of their own token keys, so node bounds enclose every token key "
                        "(max-per-token search); 0 (default): cells are points")
    p.add_argument("--fsb_read_top", type=int, default=None,
                   help="overselect (opt-in; None/0 = off): the beam runs with its width (fixed: --fsb_width = "
                        "W_beam) and only the K best of its retained cells by the routing leaf score "
                        "(q_r.kbar*scale + position term) are read. Reads stay K <= 16 cells; the beam's "
                        "candidate set W_max*2^Delta_max may reach 64.")
    p.add_argument("--fsb_rerank_score", default=None, choices=["leaf", "lse", "lse_pos", "max"],
                   help="overselect re-rank score (opt-in; None = leaf, byte-identical): lse / max = logsumexp / "
                        "max over the candidate cell's raw token scores (q_r.k_j*scale); lse_pos = lse + the "
                        "position term")
    p.add_argument("--fsb_rerank_query", default=None, choices=["route", "raw"],
                   help="overselect re-rank query (opt-in; None = route, byte-identical): route = the routing "
                        "query q_r the beam traverses with; raw = the raw q scores the re-rank (the beam still "
                        "traverses with q_r)")
    p.add_argument("--fsb_union_width", type=int, default=None,
                   help="overselect union candidate pool (opt-in; None/0 = off, byte-identical; needs "
                        "--fsb_read_top): the re-rank's candidates are the --fsb_width beam's cells UNION the "
                        "cells of a second beam of width U run with the same routing query and stop rule "
                        "(duplicates dropped), then --fsb_read_top are read")
    p.add_argument("--fsb_union_tau_quantile", type=float, default=None,
                   help="Filtered second-forest quantile; requires --fsb_union_tau_file.")
    p.add_argument("--fsb_union_tau_file", default=None,
                   help="Portable calibration JSON for --init_from. Thresholds stay fixed during CPT; "
                        "the main forest's admission threshold is independent.")
    p.add_argument("--fsb_hybrid", default=None, choices=["off", "pool", "union", "quota"],
                   help="hybrid selector (opt-in; None/off = byte-identical; needs --fsb_read_top, forest, fixed "
                        "policy): pool = FS-L's native pool top-K only; union = pool shortlist UNION the tree "
                        "candidates, one re-rank; quota = --fsb_pool_quota reserved pool slots, the tree fills "
                        "the rest")
    p.add_argument("--fsb_pool_size", type=int, default=None, help="hybrid pool top-P (None = 64; 0 = every eligible cell)")
    p.add_argument("--fsb_pool_recent", type=int, default=None, help="hybrid pool recent-cell window (None = 8)")
    p.add_argument("--fsb_pool_short", type=int, default=None, help="hybrid union: pool shortlist size (None = 4)")
    p.add_argument("--fsb_pool_quota", type=int, default=None, help="hybrid quota: reserved pool slots K_L (None = 1)")
    p.add_argument("--fsb_rerank_rr", default=None, choices=["none", "linear"],
                   help="residual re-ranker on the overselect's lse_pos score (opt-in; None = not threaded, "
                        "byte-identical; needs --fsb_read_top, --fsb_rerank_score lse_pos): zero-initialised "
                        "fsb_route_rr_* per layer, trained offline on utility labels; this script only runs it "
                        "fixed (load trained weights with --init_router_from).")
    p.add_argument("--fsb_policy_temp", type=float, default=1.0, help="policy logit temperature")
    p.add_argument("--fsb_tree", default="rebuild", choices=["rebuild", "forest"],
                   help="rebuild: one balanced tree per query block over all eligible admitted cells (rebuilt per "
                        "block); forest: binary-counter forest of content-bisected chunk trees built once / "
                        "incrementally (streaming insertion, O(M log^2 M) construction).")
    p.add_argument("--fsb_prefill_blkchunk", type=int, default=None,
                   help="query blocks per chunk in the full-sequence FS-B path (memory bound; results identical). "
                        "None => env FSB_PREFILL_BLKCHUNK => 32; <=0 = all blocks at once. When given (flag or "
                        "env), it also caps the query blocks per stage of the fast eval-mode prefill.")
    p.add_argument("--fsb_tau_quantile", type=float, default=0.0,
                   help=">0: calibrate tau_h at start (per layer/head) to this quantile of the max-key-norm "
                        "distribution over --fsb_tau_calib_batches val batches, then freeze (saved in ckpt). "
                        "0 = admit all.")
    p.add_argument("--fsb_tau_calib_batches", type=int, default=8)
    p.add_argument("--fsb_eta", type=float, default=0.1, help="learned policy: compute-cost weight in Z=loss+eta*C")
    p.add_argument("--fsb_entropy_beta", type=float, default=0.01, help="learned policy: entropy bonus")
    p.add_argument("--fsb_value_coef", type=float, default=0.5, help="learned policy: value-baseline loss weight")
    p.add_argument("--fsb_adv_norm", action="store_true", default=False,
                   help="learned policy: standardize the advantage per layer (keeps the controller grad O(1))")
    p.add_argument("--fsb_policy_clip", type=float, default=0.0,
                   help=">0: clip the controller's grad norm separately at this value and exclude it from the "
                        "global --grad_clip (so the REINFORCE term cannot scale down the backbone update); the "
                        "logged gnorm is then the backbone norm. 0 = single global clip (default).")
    p.add_argument("--fsb_rl", default="reinforce", choices=["reinforce", "ppo"],
                   help="learned policy update: reinforce = one on-policy pass per optimizer step; ppo = the same "
                        "first pass, then --fsb_ppo_epochs-1 controller-only epochs of the clipped surrogate on "
                        "the stored controller batch, stepped by a separate AdamW over the fsb_policy params "
                        "(excluded from the main optimizer).")
    p.add_argument("--fsb_ppo_epochs", type=int, default=4, help="ppo: controller epochs per optimizer step (1 = reinforce)")
    p.add_argument("--fsb_ppo_clip", type=float, default=0.2, help="ppo: ratio clip epsilon")
    p.add_argument("--fsb_ppo_store_dtype", default="bf16", choices=["bf16", "fp32"],
                   help="ppo: storage dtype of the RMS-normalized controller input q (bf16 = d_model*2 B/token/layer)")
    p.add_argument("--fsb_ppo_check_sync", action="store_true", default=False,
                   help="ppo: after every step, all-gather the controller params and assert they are identical "
                        "on every rank")
    p.add_argument("--fsb_ctrl_lr_mult", type=float, default=1.0,
                   help="learned policy: the controller's learning rate = this x the main schedule at every step "
                        "(reinforce: its own param group of the main optimizer; ppo: the controller AdamW). "
                        "1.0 = unchanged.")
    p.add_argument("--fsb_reward", default="loss", choices=["loss", "gain"],
                   help="learned policy objective per token: loss = the LM loss (default, unchanged); gain = "
                        "-clamp(l0 - l, 0, --lr_gain_clip) from the short-range-only counterfactual forward, "
                        "i.e. the policy is rewarded for reads that improve the prediction; every layer/head "
                        "that read for token i gets token i's gain.")
    p.add_argument("--lr_gain_clip", type=float, default=5.0,
                   help="clip of the per-token long-range gain used by --fsb_reward gain (nats)")
    # FS-B routing-only query adapter + router distillation (src/fsattn/fsb_route.py). Every default is
    # None / off: nothing is threaded into the model and every existing path is byte-identical.
    p.add_argument("--fsb_route_adapter", default=None, choices=["none", "lowrank", "full"],
                   help="routing-only query adapter q_r = q + alpha*adapter_h(q) (per layer/head; drives only "
                        "the FS-B routing: node bounds, leaf scores, beam). lowrank: A (Dh x r) small random, "
                        "B (r x Dh) zero; full: Dh x Dh zero.")
    p.add_argument("--fsb_route_rank", type=int, default=None, help="lowrank adapter rank r (default 32)")
    p.add_argument("--fsb_route_alpha", type=float, default=None, help="adapter scale alpha (train/eval knob, default 1.0)")
    p.add_argument("--fsb_rd_weight", type=float, default=None,
                   help=">0: router distillation of the per-cell LSE attention mass into the routing adapter "
                        "(0/None = off)")
    p.add_argument("--fsb_rd_rows", type=int, default=None, help="distillation: query rows sampled per 128-block (default 16)")
    p.add_argument("--fsb_rd_loss", default=None, choices=["kl", "topk"],
                   help="kl: KL(teacher || student) on the teacher top-n cells + remainder bucket; topk: "
                        "Plackett-Luce on the teacher top-4")
    p.add_argument("--fsb_rd_far", type=int, default=None, help="1: teacher/student over far cells only (cell_end <= i - window)")
    p.add_argument("--fsb_rd_pos", type=int, default=None, help="1: teacher adds the FS position term lambda*a_c*erf")
    p.add_argument("--fsb_rd_temp", type=float, default=None, help="teacher temperature T (default 1.0)")
    p.add_argument("--fsb_rd_topn", type=int, default=None, help="KL support: teacher top-n cells + a remainder bucket (default 16)")
    p.add_argument("--fsb_rd_node", type=int, default=None,
                   help="1: node-level distillation of the beam's decisions -- per sampled row and per depth of "
                        "its block tree, KL(softmax(subtree teacher LSE) || softmax(node bound U_v(q_r))) over "
                        "the level's nodes, on decision levels only, added to the leaf KL (forest, fixed W/D0; "
                        "src/fsattn/fsb_route.py)")
    p.add_argument("--fsb_rd_node_weight", type=float, default=None, help="weight of the node KL (default 1.0)")
    p.add_argument("--fsb_route_learn_c", type=int, default=None,
                   help="1: per-head learnable radius coefficient c_h = softplus(theta_h), exactly "
                        "--fsb_bound_radius_coef at init, used by every traversal (training, prefill, decode) "
                        "and trained by the node loss")
    p.add_argument("--fsb_route_c_lr_mult", type=float, default=1.0,
                   help="router training: lr multiplier of the learned radius coefficient theta_h "
                        "(fsb_route_ctheta); 1.0 = one group")
    p.add_argument("--fsb_route_nt_lr_mult", type=float, default=1.0,
                   help="router training: lr multiplier of the node-level student temperature "
                        "(fsb_route_nlogtemp); 1.0 = unchanged")
    p.add_argument("--init_router_from", default=None,
                   help="warm-start the fsb_route_* tensors from a router checkpoint dir "
                        "(<dir>/fsb_route.safetensors, else the fsb_route_* keys of <dir>/model.safetensors); "
                        "router params absent there keep their init")
    p.add_argument("--far_recall_beta", type=float, default=0.0,
                   help=">0: far-recall weighted CE, w_t = 1 + beta [far continuation or far recall start] + "
                        "--far_recall_switch_beta [source switch], loss = sum w_t l_t / sum w_t (masks: "
                        "far_recall_masks; one document, sources more than --far_recall_dmin back). "
                        "0 (default, with switch beta 0) = plain CE.")
    p.add_argument("--retention_weight", type=float, default=0.0,
                   help="Optional natural-repeat cell-retention margin loss for pruned FS-L.")
    p.add_argument("--retention_rank", type=int, default=16,
                   help="Causal norm cutoff rank for the auxiliary loss; serving pool size is unchanged.")
    p.add_argument("--retention_margin", type=float, default=0.02,
                   help="Relative margin above the detached norm cutoff.")
    p.add_argument("--far_recall_switch_beta", type=float, default=0.0,
                   help="extra weight of source-switch tokens (a recall start within 64 tokens of the previous "
                        "one, copied from a source more than 64 tokens away); switch tokens get 1 + beta + switch_beta")
    p.add_argument("--far_recall_k", type=int, default=4, help="far continuation: n-gram ending at x_t")
    p.add_argument("--far_recall_m", type=int, default=4, help="far recall start: n-gram beginning at y_t")
    p.add_argument("--far_recall_dmin", type=int, default=512,
                   help="the copied source sits more than this many positions before the query x_t (the local window)")
    p.add_argument("--far_recall_span", type=int, default=1,
                   help="1..--far_recall_m: a recall start / switch also weights the next span - 1 tokens of the "
                        "copy (the tokens that pick the far source, e.g. a number's first digits); 1 (default)")
    p.add_argument("--far_recall_cont_mode", default="match", choices=["match", "align"],
                   help="continuation weight: match (default) = only where y_t equals the far source's next "
                        "token; align = wherever the k-gram ending at x_t has a unique far source (context-only "
                        "weight: the copy's exit is weighted too)")
    args, extras = p.parse_known_args()
    if is_config_mode(args.config):
        apply_config(args, extras)
    elif extras:
        p.error("unrecognized arguments: %s" % " ".join(extras))
    return args


def ctrl_lr_groups(opt_params, ctrl_ids, mult):
    # Give the FS-B controller params (ids in ``ctrl_ids``) their own param group whose ``mup_lr_mult`` is
    # multiplied by ``mult`` (the schedule sets lr = base lr x mup_lr_mult per group).  ``opt_params`` is an
    # iterable of params or a list of param-group dicts.  Unchanged (same object) when mult == 1 or no ids.
    if mult == 1.0 or not ctrl_ids:
        return opt_params
    groups = opt_params if (isinstance(opt_params, list) and opt_params and isinstance(opt_params[0], dict)) \
        else [{"params": list(opt_params)}]
    out = []
    for g in groups:
        keep = [q_ for q_ in g["params"] if id(q_) not in ctrl_ids]
        ctrl = [q_ for q_ in g["params"] if id(q_) in ctrl_ids]
        if keep:
            out.append({**g, "params": keep})
        if ctrl:
            out.append({**g, "params": ctrl, "mup_lr_mult": g.get("mup_lr_mult", 1.0) * mult, "fsb_ctrl": True})
    return out


def set_step_lr(optimizer, lr, ctrl_opt=None, ctrl_mult=1.0):
    # Schedule step: every main group gets lr x its mup_lr_mult (1.0 when absent); the separate FS-B
    # controller optimizer (ppo) gets lr x ctrl_mult.
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("mup_lr_mult", 1.0)
    if ctrl_opt is not None:
        for pg in ctrl_opt.param_groups:
            pg["lr"] = lr * ctrl_mult


def set_long_range_disabled(model, flag):
    # Flip the short-range-only switch of every long-range attention module (the LR branch, incl. the FS-B
    # forest, is then skipped).  Returns the previous values for restore.
    prev = []
    for m in model.modules():
        if hasattr(m, "disable_long_range") and hasattr(m, "_long_range"):
            prev.append((m, m.disable_long_range)); m.disable_long_range = flag
    return prev


def restore_long_range(prev):
    for m, v in prev:
        m.disable_long_range = v


def fsb_objective(model, loss, logits, y, l0, args, fsb_learned, rd_acc=None, rew_acc=None):
    # Optimizer objective of one micro-batch: the LM ``loss``, plus the router-distillation term of the last
    # forward (--fsb_rd_weight > 0; its gradient reaches only the fsb_route_* params), plus the learned
    # controller's policy loss (``fsb_learned``: --fsb_policy learned; per-token reward from the detached LM
    # loss or the counterfactual gain ``l0``).  Each term is ADDED to the running total, so distillation and
    # the learned controller combine.  With both off it returns ``loss`` itself (the same object).
    # ``rd_acc`` / ``rew_acc`` (dicts, optional) accumulate the distillation stats and the controller reward
    # stats for logging.
    loss_total = loss
    if (getattr(args, "fsb_rd_weight", None) or 0.0) > 0:
        from src.fsattn.fsb_route import fsb_rd_collect
        _rd, _rd_st = fsb_rd_collect(model)
        if _rd is not None:
            loss_total = loss_total + args.fsb_rd_weight * _rd
            if rd_acc is not None:
                rd_acc["st"] = _rd_st.detach() if rd_acc.get("st") is None else rd_acc["st"] + _rd_st.detach()
    if fsb_learned:
        from src.fsattn.fsb_tree import fsb_policy_loss, fsb_reward_tokens
        _lt = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1),
                              reduction="none").view(y.shape).detach()
        _rt = fsb_reward_tokens(_lt, l0, args.fsb_reward, args.lr_gain_clip)
        _rs = {}
        loss_total = loss_total + fsb_policy_loss(model, _rt, eta=args.fsb_eta, entropy_beta=args.fsb_entropy_beta,
                                                  value_coef=args.fsb_value_coef, adv_norm=args.fsb_adv_norm,
                                                  stats=_rs)
        if rew_acc is not None:
            for k_ in _rs:
                rew_acc[k_] = rew_acc.get(k_, 0.0) + _rs[k_]
            rew_acc["n"] = rew_acc.get("n", 0) + 1
    return loss_total


def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def estimate_val_loss(model, val_loader, accelerator, max_batches=20):
    model.eval()
    tot_loss = torch.zeros((), device=accelerator.device)
    tot_tok = torch.zeros((), device=accelerator.device)
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(accelerator.device), y.to(accelerator.device)
        with accelerator.autocast():
            logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        tot_loss += loss * y.numel()
        tot_tok += y.numel()
    tot_loss = accelerator.reduce(tot_loss, reduction="sum")
    tot_tok = accelerator.reduce(tot_tok, reduction="sum")
    model.train()
    return (tot_loss / tot_tok.clamp(min=1)).item()


def main():
    args = parse_args()
    checkpoint_steps = {int(s) for s in args.checkpoint_steps.split(",") if s}
    scaling = args.lr_schedule == "wsd"
    if scaling:
        from experiments.scaling.scaling_schedule import wsd_lr, validate_resume
        if args.cooldown_start_step is None:
            raise ValueError("WSD requires --cooldown_start_step")
        wsd_lr(0, args.warmup_steps, args.max_steps, args.lr, args.min_lr, args.cooldown_start_step)
        if (not args.pretokenized or not args.data_sha256 or len(args.data_sha256) != 64
                or args.validation_chunks < 1):
            raise ValueError("WSD requires verified pretokenized data and a positive validation slice")
        if args.init_from or args.fsb or args.retention_weight or args.far_recall_beta or args.far_recall_switch_beta:
            raise ValueError("The scaling study uses from-scratch, unweighted FS or dense training")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.tf32:
            raise ValueError("TF32 must stay off in the matched scaling study")
    elif args.cooldown_start_step is not None:
        raise ValueError("--cooldown_start_step requires --lr_schedule wsd")
    from experiments.scaling.training_calibration import validate_union_training, load_union_training
    validate_union_training(args)
    if args.retention_weight < 0 or args.retention_rank < 1 or args.retention_margin < 0:
        raise ValueError("Invalid retention loss settings")
    if args.retention_weight > 0 and (args.gradient_checkpointing or args.fsb or args.compile):
        raise ValueError("Retention supervision currently requires native FS-L without block checkpointing or compile")

    ipg = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    ddp = DistributedDataParallelKwargs(gradient_as_bucket_view=True, find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb_project else None,
        kwargs_handlers=[ipg, ddp],
    )
    set_seed(args.seed)
    rank = accelerator.process_index
    world = accelerator.num_processes

    cfg = MODEL_CONFIGS[args.config].copy()
    if getattr(args, "n_layers", None):
        cfg["n_layers"] = args.n_layers
    if getattr(args, "vocab", None):
        cfg["vocab_size"] = args.vocab
    if getattr(args, "max_seq_len", None):
        cfg["max_seq_len"] = args.max_seq_len
    seq_len = cfg["max_seq_len"]
    _far_on = args.far_recall_beta > 0 or args.far_recall_switch_beta > 0
    if args.far_recall_beta < 0 or args.far_recall_switch_beta < 0:
        raise ValueError("--far_recall_beta / --far_recall_switch_beta must be >= 0")
    if (args.far_recall_span != 1 or args.far_recall_cont_mode != "match") and not _far_on:
        raise ValueError("--far_recall_span / --far_recall_cont_mode shape the far-recall weights: they need "
                         "--far_recall_beta or --far_recall_switch_beta > 0")
    if args.far_recall_cont_mode != "match" and not args.far_recall_beta > 0:
        raise ValueError("--far_recall_cont_mode align reweights the continuation term, which only --far_recall_beta "
                         "weights: with --far_recall_beta 0 it would change nothing")
    if _far_on:
        if not (1 <= args.far_recall_k < seq_len and 1 <= args.far_recall_m < seq_len and args.far_recall_dmin >= 0):
            raise ValueError("--far_recall_k / --far_recall_m must be in [1, seq_len) and --far_recall_dmin >= 0")
        if not 1 <= args.far_recall_span <= args.far_recall_m:
            raise ValueError("--far_recall_span must be in [1, --far_recall_m] (the tokens inside the matched m-gram)")
        if min(args.far_recall_k, args.far_recall_m) <= 4 and cfg["vocab_size"] > 32768:
            raise ValueError("far-recall masks pack n <= 4 token ids exactly in 15 bits each: vocab must be <= 32768")
    run_name = args.run_name
    if run_name is None:
        run_name = f"{args.config}_{args.attn_type}"
        if args.attn_type == "fsattn":
            run_name += f"_{args.long_range_variant}_w{args.window_size}"

    if args.wandb_project:
        accelerator.init_trackers(
            args.wandb_project, config=vars(args),
            init_kwargs={"wandb": {"name": run_name}},
        )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    use_mlflow = False
    mlflow = None
    if args.mlflow_uri and accelerator.is_main_process:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("fsattn-scaling")
            mlflow.start_run(run_name=run_name)
            use_mlflow = True
        except Exception as e:
            accelerator.print(f"MLflow unavailable ({e}); stdout only")

    if scaling:
        from experiments.scaling.scaling_data import ScalingTokenDataset
        train_ds = ScalingTokenDataset(args.pretokenized, seq_len, rank, world,
                                       args.per_gpu_batch_size, args.seed)
        val_ds = ScalingTokenDataset(args.pretokenized, seq_len, rank, world,
                                     args.per_gpu_batch_size, validation_chunks=args.validation_chunks)
        required = args.max_steps * args.grad_accum_steps * args.per_gpu_batch_size * world
        if required > train_ds.train_chunks:
            raise ValueError("Scaling budget exceeds a single pass through the training split")
    elif args.pretokenized:
        train_ds = DistPreTokenizedDataset(args.pretokenized, seq_len, rank, world, seed=args.seed)
        val_ds = DistPreTokenizedDataset(args.pretokenized, seq_len, rank, world, val_split=True)
    else:
        train_ds = DistFineWebEduDataset(seq_len, rank, world, args.data_subset, seed=args.seed)
        val_ds = DistFineWebEduDataset(seq_len, rank, world, args.data_subset, seed=args.seed + 7919)
    train_loader = DataLoader(train_ds, batch_size=args.per_gpu_batch_size,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              **({"generator": torch.Generator().manual_seed(args.seed + rank)} if scaling else {}))
    val_loader = DataLoader(val_ds, batch_size=args.per_gpu_batch_size,
                            num_workers=1, pin_memory=True, drop_last=True,
                            **({"generator": torch.Generator().manual_seed(args.seed + 10000 + rank)} if scaling else {}))

    _, attn_kwargs = build_attn_kwargs(args)
    _sc_route = attn_kwargs.get("sel_strict_causal", os.environ.get("FS_STRICT_CAUSAL"))
    print(f"[rank {accelerator.process_index}] attn={args.attn_type} "
          f"cell_select={attn_kwargs.get('cell_select')} sel_strict_causal={_sc_route} "
          f"prune_kernel={attn_kwargs.get('cell_prune_kernel', os.environ.get('FS_PRUNE_KERNEL'))} "
          f"no_rope={getattr(args, 'no_rope', None)}", flush=True)

    if args.require_freesolo:
        _fs = (args.attn_type == "fsattn"
               and attn_kwargs.get("long_range_variant") == "cell_topk"
               and attn_kwargs.get("cell_select") == "score"
               and int(attn_kwargs.get("sel_strict_causal") or 0) == 1
               and (attn_kwargs.get("use_rope") is False
                    or attn_kwargs.get("rope_long_range") is False))
        assert _fs, (
            "NOT the FreeSolo config — refusing to train (require_freesolo). Need: --attn_type fsattn "
            "--long_range_variant cell_topk --cell_select score --sel_strict_causal 1 and a NoPE "
            "long-range branch (--fs_nope_longrange for the released models, or --no_rope). "
            f"Resolved attn_kwargs={attn_kwargs}")
        accelerator.print("[freesolo-guard] OK: fsattn score strict-causal NoPE-long-range confirmed.")

    if args.tf32:
        torch.set_float32_matmul_precision("high")
        accelerator.print("[accel] TF32 matmul precision = high")

    model = FreeSoloLM(attn_type=args.attn_type,
                    gradient_checkpointing=args.gradient_checkpointing,
                    **cfg, **attn_kwargs)
    n_params = model.count_params()
    if getattr(args, "init_from", None):
        from safetensors.torch import load_file as _load_sft
        _sd = _load_sft(os.path.join(args.init_from, "model.safetensors"))
        _sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in _sd.items()}
        _missing, _unexpected = model.load_state_dict(_sd, strict=False)
        # params absent from the checkpoint that warm-start from their own init instead, excluded from the
        # mismatch gate: fsb_* = FS-B policy / router params + tau buffers (new vs a non-FS-B ckpt);
        # lr_group = zero-init grouped-read gain / temperature.
        _new_ok = ("fsb_", "lr_group")
        _missing_core = [k for k in _missing if not any(t in k for t in _new_ok)]
        accelerator.print(f"init_from {args.init_from}: loaded {len(_sd)} tensors "
                          f"(missing={len(_missing)}, missing_core={len(_missing_core)}, "
                          f"unexpected={len(_unexpected)})")
        assert len(_unexpected) == 0 and len(_missing_core) <= 2, \
            f"init_from key mismatch (missing_core={len(_missing_core)}, unexpected={len(_unexpected)}) — check prefix"
    union_calibration = load_union_training(model, args)
    if union_calibration is not None:
        accelerator.print(f"[fsb] fixed union calibration: {union_calibration['tau_sha256']} "
                          f"q={union_calibration['quantile']}")
    if getattr(args, "init_router_from", None):
        # warm-start the routing adapter (and any other fsb_route_* tensor the checkpoint has) from a router run
        from src.fsattn.fsb_route import load_router_params
        accelerator.print(load_router_params(model, args.init_router_from))
    if getattr(args, "fsb_rerank_rr", None) not in (None, "none") and not getattr(args, "resume", None):
        # the residual re-ranker receives no gradient here (the traversal runs under no_grad): train_dist can
        # only run it FIXED with weights loaded above; an all-zero one is the identity and its saves would
        # carry zero re-ranker tensors that an eval loads as a valid router file
        from src.fsattn.fsb_route import rr_all_zero
        if rr_all_zero(model):
            raise ValueError("--fsb_rerank_rr: the re-ranker weights are all zero and train_dist cannot train "
                             "them (no gradient reaches them): load trained weights with --init_router_from "
                             "(they stay fixed), or drop the flag")
    if (getattr(args, "fsb_rd_weight", None) or 0.0) > 0:
        # router distillation with a trained backbone (LM + w * rd)
        if args.gradient_checkpointing:
            # the block recompute would re-run the distillation (a second row draw from the generator, which
            # checkpoint does not restore): the rd gradient would come from other rows than the logged loss
            raise ValueError("--fsb_rd_weight > 0 with a trained backbone does not support --gradient_checkpointing")
        from src.fsattn.fsb_route import rd_generator
        _g_rd = rd_generator(model, accelerator.device, args.seed, accelerator.process_index)
        accelerator.print(f"[fsb rd] router distillation weight {args.fsb_rd_weight} alongside the LM loss; row "
                          f"sampling from a per-rank generator (seed {args.seed} * 1000 + rank)"
                          + ("" if _g_rd is not None else " -- WARNING: no FS-B layer distills"))
    if args.no_long_range_recompute:
        n_set = 0
        for m in model.modules():
            if type(m).__name__ == "FSAttn":
                m.recompute_long_range = False
                n_set += 1
        accelerator.print(f"long-range recompute DISABLED on {n_set} layers")

    if args.compile and not args.resume:
        model = torch.compile(model)

    model = accelerator.prepare(model)
    # FS-B ppo: the controller (fsb_policy.*) is stepped K times per optimizer step by its own AdamW, so it
    # must be excluded from the main optimizer (which would otherwise apply the first epoch a second time).
    _fsb_learned = bool(getattr(args, "fsb", False)) and getattr(args, "fsb_policy", "fixed") == "learned"
    _ppo = _fsb_learned and getattr(args, "fsb_rl", "reinforce") == "ppo"
    ctrl_params = [q_ for n_, q_ in model.named_parameters() if "fsb_policy" in n_] if _ppo else []
    _ctrl_ids = {id(q_) for q_ in ctrl_params}
    if _ppo and not ctrl_params:
        raise RuntimeError("--fsb_rl ppo: no fsb_policy parameters found (is --fsb_policy learned?)")
    _ctrl_ids_all = {id(q_) for n_, q_ in model.named_parameters() if "fsb_policy" in n_} if _fsb_learned else set()
    _opt_params = model.parameters()
    if _ppo:
        _opt_params = [q_ for q_ in model.parameters() if id(q_) not in _ctrl_ids]
    # reinforce + --fsb_ctrl_lr_mult != 1: the controller gets its own group (lr = mult x schedule)
    _opt_params = ctrl_lr_groups(_opt_params, _ctrl_ids_all - _ctrl_ids, args.fsb_ctrl_lr_mult)
    optimizer = torch.optim.AdamW(_opt_params, lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95),
                                  fused=args.fused_adam)
    optimizer = accelerator.prepare(optimizer)
    retention = None
    if args.retention_weight > 0:
        from experiments.scaling.retention_loss import RetentionObjective
        retention = RetentionObjective(model, rank=args.retention_rank, margin=args.retention_margin)
        accelerator.print(f"[retention] weight={args.retention_weight} rank={args.retention_rank} "
                          f"margin={args.retention_margin}; serving pool unchanged")
    ctrl_opt = None
    if _ppo:
        ctrl_opt = torch.optim.AdamW(ctrl_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        # saved/restored with accelerator.save_state as custom_checkpoint_0.pkl; a --resume from a pre-ppo
        # checkpoint (no such file) starts the controller optimizer fresh instead of failing the load.
        if not args.resume or os.path.exists(os.path.join(args.resume, "custom_checkpoint_0.pkl")):
            accelerator.register_for_checkpointing(ctrl_opt)
        accelerator.print(f"[fsb ppo] controller AdamW over {sum(q_.numel() for q_ in ctrl_params)} params "
                          f"({len(ctrl_params)} tensors) excluded from the main optimizer | "
                          f"epochs={args.fsb_ppo_epochs} clip={args.fsb_ppo_clip} store={args.fsb_ppo_store_dtype} "
                          f"check_sync={args.fsb_ppo_check_sync} ctrl_lr_mult={args.fsb_ctrl_lr_mult}")
    elif _fsb_learned and args.fsb_ctrl_lr_mult != 1.0:
        accelerator.print(f"[fsb] controller param group: lr = {args.fsb_ctrl_lr_mult} x schedule "
                          f"({sum(1 for g_ in optimizer.param_groups if g_.get('fsb_ctrl'))} group(s))")

    tokens_per_step = args.per_gpu_batch_size * args.grad_accum_steps * seq_len * world
    accelerator.print(
        f"Model: {args.config}/{args.attn_type}  {n_params/1e9:.3f}B params | "
        f"world={world} | per_gpu_bs={args.per_gpu_batch_size} accum={args.grad_accum_steps} "
        f"seq={seq_len} | tokens/step={tokens_per_step:,} | strategy={accelerator.distributed_type}")

    start_step = 0
    scaling_contract = None
    scaling_schedule = None
    if scaling:
        from importlib.metadata import version
        from experiments.scaling.provenance import source_revision
        revision = source_revision()
        scaling_contract = dict(
            config=cfg, attn_type=args.attn_type, attn_kwargs=attn_kwargs,
            source_revision=revision, data_sha256=args.data_sha256,
            data_bytes=os.path.getsize(args.pretokenized), data_order="global_numpy_pcg64/1",
            numpy_version=np.__version__, torch_version=torch.__version__,
            accelerate_version=version("accelerate"), python_version=sys.version.split()[0],
            cuda_version=torch.version.cuda,
            seed=args.seed, world_size=world, per_gpu_batch_size=args.per_gpu_batch_size,
            grad_accum_steps=args.grad_accum_steps, num_workers=args.num_workers,
            validation_chunks=args.validation_chunks, warmup_steps=args.warmup_steps,
            lr=args.lr, min_lr=args.min_lr, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, fused_adam=args.fused_adam, compile=args.compile,
            gradient_checkpointing=args.gradient_checkpointing,
            execution_env={k: v for k, v in sorted(os.environ.items()) if k.startswith(("FS_", "FSB_"))},
        )
        scaling_schedule = dict(max_steps=args.max_steps, cooldown_start=args.cooldown_start_step)
    if args.resume:
        accelerator.print(f"Resuming from {args.resume}")
        meta = os.path.join(args.resume, "step.txt")
        if os.path.exists(meta):
            start_step = int(open(meta).read().strip())
        if scaling:
            with open(os.path.join(args.resume, "scaling_state.json")) as f:
                scaling_saved = json.load(f)
            validate_resume(scaling_saved, scaling_contract, scaling_schedule, start_step)
        accelerator.load_state(args.resume)
        if args.compile:
            model = torch.compile(model)

    def save_ckpt(tag, step, val_loss):
        path = os.path.join(args.output_dir, f"{run_name}_{tag}")
        if scaling:
            state_path = os.path.join(path, "scaling_state.json")
            if accelerator.is_main_process and os.path.exists(state_path):
                os.unlink(state_path)
            accelerator.wait_for_everyone()
        accelerator.save_state(path)
        if scaling:
            accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if union_calibration is not None:
                from experiments.scaling.training_calibration import save_union_training
                save_union_training(model, union_calibration, path)
            if retention is not None:
                with open(os.path.join(path, "retention_config.json"), "w") as f:
                    json.dump(dict(weight=args.retention_weight, rank=args.retention_rank,
                                   margin=args.retention_margin, source="natural exact 4-gram repeats"), f)
            with open(os.path.join(path, "step.txt"), "w") as f:
                f.write(str(step))
            with open(os.path.join(path, "meta.txt"), "w") as f:
                _mup_meta = "mup=0"
                f.write(f"step={step} val_loss={val_loss} params={n_params} "
                        f"attn={args.attn_type} cell_select={getattr(args, 'cell_select', '')} "
                        f"sel_strict_causal={attn_kwargs.get('sel_strict_causal')} "
                        f"no_rope={getattr(args, 'no_rope', None)} "
                        f"long_range_variant={attn_kwargs.get('long_range_variant')} "
                        f"{_mup_meta}"
                        # FS-B runs only (a default run's meta.txt stays byte-identical)
                        + (f" fsb=1 fsb_policy={args.fsb_policy} fsb_widths={args.fsb_widths} fsb_deltas={args.fsb_deltas} "
                           f"fsb_width={args.fsb_width} fsb_delta={args.fsb_delta} fsb_node_score={args.fsb_node_score} "
                           f"fsb_tau_quantile={args.fsb_tau_quantile} fsb_tree={args.fsb_tree}"
                           + (f" fsb_read_top={args.fsb_read_top}" if getattr(args, "fsb_read_top", None) else "")
                           + (f" fsb_rerank_score={args.fsb_rerank_score}" if getattr(args, "fsb_rerank_score", None) else "")
                           + (f" fsb_rerank_query={args.fsb_rerank_query}" if getattr(args, "fsb_rerank_query", None) else "")
                           + (f" fsb_union_width={args.fsb_union_width}" if getattr(args, "fsb_union_width", None) else "")
                           + "".join(f" {k_}={getattr(args, k_)}" for k_ in ("fsb_hybrid", "fsb_pool_size",
                                                                             "fsb_pool_recent", "fsb_pool_short",
                                                                             "fsb_pool_quota")
                                     if getattr(args, k_, None) is not None)
                           if getattr(args, "fsb", False) else "")
                        + ("".join(f" {k_}={getattr(args, k_)}" for k_ in ("far_recall_beta", "far_recall_switch_beta",
                                                                           "far_recall_k", "far_recall_m", "far_recall_dmin"))
                           + (f" far_recall_span={args.far_recall_span}" if args.far_recall_span != 1 else "")
                           + (f" far_recall_cont_mode={args.far_recall_cont_mode}"
                              if args.far_recall_cont_mode != "match" else "")
                           if _far_on else "")
                        + "\n")
            if scaling:
                state = dict(schema="freesolo_wsd/1", step=step, params=n_params,
                             tokens=step * tokens_per_step,
                             next_microbatch=step * args.grad_accum_steps,
                             contract=scaling_contract, schedule=scaling_schedule,
                             device=str(accelerator.device),
                             gpu=torch.cuda.get_device_name(accelerator.device)
                             if accelerator.device.type == "cuda" else None,
                             validation_loss=val_loss if math.isfinite(val_loss) else None)
                state_path = os.path.join(path, "scaling_state.json")
                with open(state_path + ".tmp", "w") as f:
                    json.dump(state, f, indent=2, sort_keys=True)
                os.replace(state_path + ".tmp", state_path)
        if scaling:
            accelerator.wait_for_everyone()
        accelerator.print(f"  → saved {path} (step {step}, val {val_loss:.4f})")

    # FS-B: one-shot tau calibration (fixed training-time statistics; frozen + saved in ckpt)
    _fsb_learned = bool(getattr(args, "fsb", False)) and getattr(args, "fsb_policy", "fixed") == "learned"
    if getattr(args, "fsb", False):
        from src.fsattn.fsb_tree import (fsb_log_stats, fsb_calibrate_tau, fsb_ppo_surrogate,
                                         fsb_ppo_clear, fsb_ppo_store_bytes)
        if args.fsb_tau_quantile > 0 and start_step == 0:
            _vi = iter(val_loader)
            _cal = [next(_vi)[0].to(accelerator.device) for _ in range(args.fsb_tau_calib_batches)]
            fsb_calibrate_tau(model, _cal, args.fsb_tau_quantile,
                              reduce_fn=lambda t: accelerator.reduce(t, reduction="mean"))
            _taus = torch.stack([m.fsb_tau for m in model.modules() if getattr(m, "fsb_enabled", False)])
            accelerator.print(f"[fsb] tau calibrated at q={args.fsb_tau_quantile} on {len(_cal)} val batches: "
                              f"mean {_taus.mean():.3f} min {_taus.min():.3f} max {_taus.max():.3f} "
                              f"(layers x heads {tuple(_taus.shape)})")
            del _cal
        accelerator.print(f"[fsb] policy={args.fsb_policy} widths={args.fsb_widths} deltas={args.fsb_deltas} "
                          f"fixed=(W{args.fsb_width},Delta{args.fsb_delta}) node_score={args.fsb_node_score} "
                          f"radius_coef={args.fsb_bound_radius_coef} leaf_radius={args.fsb_leaf_radius} "
                          f"tau_q={args.fsb_tau_quantile} eta={args.fsb_eta} adv_norm={args.fsb_adv_norm} "
                          f"reward={args.fsb_reward} policy_clip={args.fsb_policy_clip} tree={args.fsb_tree}"
                          + (f" overselect read_top={args.fsb_read_top} (beam W{args.fsb_width})"
                             if getattr(args, "fsb_read_top", None) else "")
                          + (f" rerank_score={args.fsb_rerank_score}" if getattr(args, "fsb_rerank_score", None) else "")
                          + (f" rerank_query={args.fsb_rerank_query}" if getattr(args, "fsb_rerank_query", None) else "")
                          + (f" union_width={args.fsb_union_width}" if getattr(args, "fsb_union_width", None) else "")
                          + "".join(f" {k_[4:]}={getattr(args, k_)}" for k_ in ("fsb_hybrid", "fsb_pool_size",
                                                                                "fsb_pool_recent", "fsb_pool_short",
                                                                                "fsb_pool_quota")
                                    if getattr(args, k_, None) is not None))

    def _fsb_ppo_update(lr_now):
        # One ppo controller update (called once per optimizer step, after the main step).  Epoch 1 applies
        # the on-policy gradients of the main backward (already averaged by DDP).  Epochs 2..K re-score the
        # stored controller batch with the current controller (clipped surrogate + value + entropy), all-reduce
        # the controller grads by hand (no DDP forward is involved), clip, step.  Then the store is cleared.
        import torch.distributed as dist
        torch.cuda.synchronize(); t_ = time.time()
        clipv = args.fsb_policy_clip if args.fsb_policy_clip > 0 else args.grad_clip
        for pg in ctrl_opt.param_groups:
            pg["lr"] = lr_now * args.fsb_ctrl_lr_mult
        store_mib = fsb_ppo_store_bytes(model) / 2 ** 20
        for q_ in ctrl_params:
            if q_.grad is None:
                q_.grad = torch.zeros_like(q_)
        gn0 = torch.nn.utils.clip_grad_norm_(ctrl_params, clipv)
        ctrl_opt.step(); ctrl_opt.zero_grad(set_to_none=True)
        st_ = {}
        for _ep in range(args.fsb_ppo_epochs - 1):
            loss_ppo, st_ = fsb_ppo_surrogate(model, clip=args.fsb_ppo_clip, entropy_beta=args.fsb_entropy_beta,
                                              value_coef=args.fsb_value_coef)
            loss_ppo.backward()
            for q_ in ctrl_params:
                if q_.grad is None:
                    q_.grad = torch.zeros_like(q_)
                if world > 1:
                    dist.all_reduce(q_.grad, op=dist.ReduceOp.SUM); q_.grad.div_(world)
            torch.nn.utils.clip_grad_norm_(ctrl_params, clipv)
            ctrl_opt.step(); ctrl_opt.zero_grad(set_to_none=True)
        fsb_ppo_clear(model)
        if args.fsb_ppo_check_sync and world > 1:
            flat = torch.cat([q_.detach().flatten().float() for q_ in ctrl_params])
            gath = [torch.empty_like(flat) for _ in range(world)]
            dist.all_gather(gath, flat)
            assert all(torch.equal(g_, gath[0]) for g_ in gath), "FS-B PPO: controller parameters differ across ranks"
            st_["sync"] = world
        torch.cuda.synchronize()
        st_["ms"] = (time.time() - t_) * 1e3; st_["store_mib"] = store_mib; st_["gn0"] = float(gn0)
        return st_

    model.train()
    step = start_step
    best_val = float("inf")
    t0 = time.time()
    total_tokens = step * tokens_per_step
    accelerator.print(f"Training {args.max_steps} steps "
                      f"({args.max_steps*tokens_per_step/1e9:.1f}B tokens)…")

    if scaling:
        train_ds.start_microbatch = start_step * args.grad_accum_steps
        accelerator.print(f"Scaling data cursor: microbatch {train_ds.start_microbatch}; single pass")
    train_iter = iter(train_loader)
    if start_step > 0 and not scaling:
        skip_mb = start_step * args.grad_accum_steps
        accelerator.print(f"Data-faithful resume: skipping {skip_mb} micro-batches (to step {start_step})...")
        for _i in range(skip_mb):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                next(train_iter)
        accelerator.print("  resume dataloader fast-forward done.")
    running = 0.0
    skipped = 0
    accum_bad = False
    ppo_info = None
    rew_acc = {}
    rd_acc = {}
    fr_acc = {}
    retention_acc = {}
    while step < args.max_steps:
        lr = (wsd_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr, args.cooldown_start_step)
              if scaling else get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr))
        # the FS-B controller: its own group of the main optimizer (reinforce) or its own optimizer (ppo),
        # stepped at --fsb_ctrl_lr_mult x the schedule
        set_step_lr(optimizer, lr, ctrl_opt, args.fsb_ctrl_lr_mult)

        with accelerator.accumulate(model):
            try:
                x, y = next(train_iter)
            except StopIteration:
                if scaling:
                    raise RuntimeError("Scaling data exhausted; refusing to repeat the training corpus") from None
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(accelerator.device), y.to(accelerator.device)
            l0 = None
            if _fsb_learned and args.fsb_reward == "gain":
                # counterfactual: the same tokens WITHOUT the long-range branch (no grad, no forest build)
                _prev = set_long_range_disabled(model, True)
                try:
                    with torch.no_grad(), accelerator.autocast():
                        _lg0 = model(x)
                        l0 = F.cross_entropy(_lg0.view(-1, _lg0.size(-1)).float(), y.view(-1),
                                             reduction="none").view(y.shape)
                    del _lg0
                finally:
                    restore_long_range(_prev)
            with accelerator.autocast():
                if retention is not None:
                    retention.begin(x, y)
                logits = model(x)
                if _far_on:
                    # far-recall weighted CE (masks from the window's own tokens; no model in the loop)
                    _far, _sw = far_recall_masks(x, y, args.far_recall_k, args.far_recall_m, args.far_recall_dmin,
                                                 span=args.far_recall_span, cont_mode=args.far_recall_cont_mode)
                    loss, _frs = far_recall_loss(logits, y, _far, _sw, args.far_recall_beta, args.far_recall_switch_beta)
                    fr_acc["st"] = _frs if fr_acc.get("st") is None else fr_acc["st"] + _frs
                    fr_acc["n"] = fr_acc.get("n", 0) + 1
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                # LM loss + router distillation (--fsb_rd_weight > 0; grad only to fsb_route_*) + the learned
                # controller's policy loss (per-token objective, detached: the LM loss (--fsb_reward loss) or
                # minus the counterfactual long-range gain (--fsb_reward gain)); terms add to the running total
                loss_total = fsb_objective(model, loss, logits, y, l0, args, _fsb_learned, rd_acc, rew_acc)
                if retention is not None:
                    retention_loss, retention_count = retention.finish()
                    loss_total = loss_total + args.retention_weight * retention_loss
                    _rs = torch.stack((retention_loss.detach(), retention_loss.new_tensor(retention_count)))
                    retention_acc["sum"] = _rs if not retention_acc else retention_acc["sum"] + _rs
                    retention_acc["n"] = retention_acc.get("n", 0) + 1
            accelerator.backward(loss_total)
            gnorm = None
            if accelerator.sync_gradients:
                if _fsb_learned and (args.fsb_policy_clip > 0 or _ppo):
                    # FS-B: clip the controller on its own so its (large) REINFORCE gradient cannot
                    # shrink the backbone update through the shared global clip; gnorm = backbone norm.
                    # (ppo: the controller is clipped + stepped in _fsb_ppo_update instead.)
                    _pol = [q_ for n_, q_ in model.named_parameters() if "fsb_policy" in n_]
                    _bb = [q_ for n_, q_ in model.named_parameters() if "fsb_policy" not in n_]
                    gnorm = accelerator.clip_grad_norm_(_bb, args.grad_clip)
                    if not _ppo:
                        accelerator.clip_grad_norm_(_pol, args.fsb_policy_clip)
                else:
                    gnorm = accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            # A single non-finite loss or gradient poisons every parameter it touches, and one
            # rank's NaN reaches the others through the allreduced gradients: decide together,
            # then skip the update so the run continues from the last finite weights. A bad
            # micro-batch marks the whole accumulation window.
            if not torch.isfinite(loss_total.detach()):
                accum_bad = True
            if accelerator.sync_gradients:
                _fp_bad = torch.zeros((), device=accelerator.device)
                if accum_bad:
                    _fp_bad += 1
                if torch.is_tensor(gnorm) and not torch.isfinite(gnorm):
                    _fp_bad += 1
                _fp_bad = accelerator.reduce(_fp_bad, reduction="max")
                if _fp_bad == 0:
                    optimizer.step()
                else:
                    skipped += 1
                    if skipped <= 5 or skipped % 50 == 0:
                        accelerator.print(f"[skip] non-finite loss/grad at step {step + 1}: "
                                          f"update skipped (skipped={skipped})")
                accum_bad = False
            optimizer.zero_grad()
            if _ppo and accelerator.sync_gradients and _fp_bad == 0:
                ppo_info = _fsb_ppo_update(lr)

        running += loss.item()
        if not accelerator.sync_gradients:
            continue

        step += 1
        current_val = float("nan")
        total_tokens += tokens_per_step
        avg_loss = running / args.grad_accum_steps
        running = 0.0

        if step % args.log_every == 0:
            tps = ((step - start_step) * tokens_per_step if scaling else total_tokens) / (time.time() - t0)
            gn = gnorm.item() if torch.is_tensor(gnorm) else (gnorm or 0.0)
            accelerator.print(
                f"step {step:7d} | loss {avg_loss:.4f} | ppl {math.exp(min(avg_loss,20)):.1f} | "
                f"lr {lr:.2e} | gnorm {gn:.2f} | {tps/1e3:.1f}K tok/s | {total_tokens/1e9:.2f}B"
                + (f" | skipped {skipped}" if skipped else ""))
            if getattr(args, "fsb", False):
                _fs = fsb_log_stats(model)
                accelerator.print(f"  [fsb] cells_read {_fs['n_read']:.2f} | node_visits {_fs['n_visit']:.1f} | "
                                  f"cost {_fs['cost']:.2f} | admit_frac {_fs['admit_frac']:.3f} | "
                                  f"tree_size {_fs['m0']:.1f}"
                                  + (f" | policy_ent {_fs['ent']:.3f} | act {['%.2f' % a for a in _fs['act_hist']]}"
                                     if 'act_hist' in _fs else ""))
                if rew_acc.get("n"):
                    _n = rew_acc["n"]
                    accelerator.print(f"  [fsb reward] {args.fsb_reward} | reward {rew_acc['reward'] / _n:.4f} | "
                                      f"Z {rew_acc['Z'] / _n:.4f} | baseline {rew_acc['value'] / _n:.4f} | "
                                      f"raw adv std {rew_acc['adv_raw_std'] / _n:.4f} | "
                                      f"|adv| used {rew_acc['adv_abs'] / _n:.4f}"
                                      + (" (standardized)" if args.fsb_adv_norm else ""))
                if rd_acc.get("st") is not None:
                    from src.fsattn.fsb_route import rd_stats_str
                    accelerator.print(f"  [fsb rd] weight {args.fsb_rd_weight} | {rd_stats_str(rd_acc['st'])}")
                    rd_acc = {}
                if ppo_info:
                    accelerator.print(f"  [fsb ppo] K={args.fsb_ppo_epochs} | update {ppo_info['ms']:.0f} ms | "
                                      f"store {ppo_info['store_mib']:.0f} MiB | ctrl gnorm(ep1) {ppo_info['gn0']:.3f}"
                                      + (f" | ratio {ppo_info['ratio_mean']:.4f} maxdev {ppo_info['ratio_maxdev']:.4f}"
                                         f" clipfrac {ppo_info['clipfrac']:.4f} kl {ppo_info['approx_kl']:.2e}"
                                         if 'ratio_mean' in ppo_info else "")
                                      + (f" | ranks in sync ({ppo_info['sync']})" if 'sync' in ppo_info else ""))
            if fr_acc.get("n"):
                # mean over the log window's micro-batches and over the ranks (each rank ran the same count);
                # the step line's loss is the weighted loss, 'plain CE' the unweighted mean CE of the same tokens
                _v = accelerator.reduce(fr_acc["st"] / fr_acc["n"], reduction="mean").tolist()
                accelerator.print(f"  [far_recall] beta {args.far_recall_beta:g} switch_beta {args.far_recall_switch_beta:g} "
                                  f"(k {args.far_recall_k} m {args.far_recall_m} dmin {args.far_recall_dmin}"
                                  + (f", span {args.far_recall_span}" if args.far_recall_span != 1 else "")
                                  + (f", cont {args.far_recall_cont_mode}" if args.far_recall_cont_mode != "match" else "")
                                  + f") | rho {_v[0]:.5f} | sw {_v[1]:.5f} | weight mass {_v[2]:.4f} | "
                                  f"mean w {_v[4]:.4f} | plain CE {_v[3]:.4f}")
                fr_acc = {}
            if retention_acc:
                _rs = accelerator.reduce(retention_acc["sum"] / retention_acc["n"], reduction="mean").tolist()
                accelerator.print(f"  [retention] unweighted margin {_rs[0]:.6f} | "
                                  f"layer/head/source targets per batch {_rs[1]:.1f} | weight {args.retention_weight:g}")
                retention_acc = {}
            if use_mlflow:
                mlflow.log_metrics({"train/loss": avg_loss, "train/lr": lr,
                                    "train/tok_per_s": tps,
                                    "train/total_tokens_B": total_tokens/1e9}, step=step)
            rew_acc = {}
            accelerator.log({"train/loss": avg_loss, "train/ppl": math.exp(min(avg_loss, 20)),
                             "train/lr": lr, "train/grad_norm": gn,
                             "train/tok_per_s": tps, "train/total_tokens_B": total_tokens/1e9},
                            step=step)

        if step % args.eval_every == 0 or (scaling and step == args.max_steps):
            val_batches = (args.validation_chunks // (world * args.per_gpu_batch_size)) if scaling else 20
            val = estimate_val_loss(model, val_loader, accelerator, max_batches=val_batches)
            current_val = val
            accelerator.print(f"  → val_loss {val:.4f} | val_ppl {math.exp(min(val,20)):.1f}")
            if use_mlflow:
                mlflow.log_metrics({"val/loss": val, "val/ppl": math.exp(min(val,20))}, step=step)
            accelerator.log({"val/loss": val, "val/ppl": math.exp(min(val, 20))}, step=step)
            if val < best_val:
                best_val = val
                save_ckpt("best", step, val)

        if step % args.save_every == 0 or step in checkpoint_steps or (scaling and step == args.max_steps):
            save_ckpt(f"step{step}", step, current_val if scaling else best_val)

    accelerator.print(f"\nDone. best_val={best_val:.4f} | "
                      f"{total_tokens/1e9:.2f}B tokens in {(time.time()-t0)/3600:.2f}h")
    if use_mlflow:
        mlflow.end_run()
    accelerator.end_training()


if __name__ == "__main__":
    main()
