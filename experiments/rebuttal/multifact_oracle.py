"""Causal source-cell interventions around the existing FS-B evaluator.

No kernel or model parameter changes. Native kernels still compute the candidate
sets and ranks. The experimental interventions run only at question/decode
positions, and every mode starts from a fresh complete forward/cache.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import runpy
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_spans(prefix, offsets, fact, stride=32):
    result = {}
    for label, value in (("key", fact["key_desc"]), ("value", fact["value"])):
        matches = list(re.finditer(re.escape(value), prefix))
        if len(matches) != 1:
            raise ValueError(f"Expected one source {label}, got {len(matches)}")
        lo, hi = matches[0].span()
        tokens = [i for i, (a, b) in enumerate(offsets) if b > a and a < hi and b > lo]
        if not tokens:
            raise ValueError(f"Unmapped {label}")
        result[label + "_tokens"] = tokens
        result[label + "_cells"] = sorted({i // stride for i in tokens})
        result[label + "_span"] = [lo, hi]
    result["cells"] = sorted(set(result["key_cells"] + result["value_cells"]))
    if len(result["cells"]) > 4:
        raise ValueError("The complete source does not fit in K=4")
    return result


def replace_required(selected, required):
    """Preserve present targets; replace rightmost nongold slots, at fixed K.

    Rows may have different eligible targets, represented by -1 padding. This
    deliberately does not grow a selection or change unaffected rows.
    """
    import torch

    if selected.ndim != 2 or required.ndim != 2 or len(selected) != len(required):
        raise ValueError("Expected two row matrices")
    if required.shape[1] > selected.shape[1]:
        raise ValueError("Targets exceed the read budget")
    out = selected.clone()
    changed = torch.zeros(len(out), dtype=torch.long, device=out.device)
    slots = torch.arange(out.shape[1], device=out.device)[None, :]
    for col in range(required.shape[1]):
        target = required[:, col]
        missing = (target >= 0) & ~(out == target[:, None]).any(-1)
        protected = ((out[..., None] == required[:, None, :]) & (required[:, None, :] >= 0)).any(-1)
        replace = slots.expand_as(out).masked_fill(protected, -1).amax(-1)
        if bool((missing & (replace < 0)).any()):
            raise ValueError("No nongold slot available")
        rows = missing.nonzero().flatten()
        out[rows, replace[rows]] = target[rows]
        changed += missing.long()
    return out, changed


class OracleTap:
    def __init__(self, model, emit):
        from src.fsattn import fsb_tree, fsb_fast_step, fsb_rows

        self.ft = fsb_tree
        self.emit = emit
        self.layers = {m: i for i, m in enumerate(m for m in model.modules() if getattr(m, "fsb_enabled", False))}
        assert len(self.layers) == 24
        self.enabled = False
        self.attn = None
        self.restores = []
        self.rerank_depth = 0
        self.calls = 0
        self.reads = 0

        def context(original):
            def call(attn, *args, **kwargs):
                previous = self.attn
                self.attn = attn
                try:
                    return original(attn, *args, **kwargs)
                finally:
                    self.attn = previous
            return call

        self.patch(fsb_tree, "fsb_long_range", context(fsb_tree.fsb_long_range))
        self.patch(fsb_tree, "_forest_token", context(fsb_tree._forest_token))
        for module, name in ((fsb_rows, "union_rerank_rows"), (fsb_fast_step, "union_rerank_fast"),
                             (fsb_fast_step, "rerank_top_fast"), (fsb_tree, "rerank_top")):
            self.patch(module, name, self.wrap_rerank(getattr(module, name), name))
        old_prefill = fsb_tree.masked_readout

        def prefill_read(attn, Q, Kp, Vp, selected, s, Bq, N):
            if self.enabled:
                positions = list(range(self.question_start, N))
                if positions:
                    import torch
                    idx = torch.tensor(positions, device=Q.device)
                    self.observe_read(attn, Q[:, :, idx], Kp, selected[:, :, idx], idx)
            return old_prefill(attn, Q, Kp, Vp, selected, s, Bq, N)

        self.patch(fsb_tree, "masked_readout", prefill_read)
        old_decode = fsb_tree._fsb_decode_readout

        def decode_read(attn, q_t, cache, selected, pos, full_buffer=False):
            if self.enabled:
                import torch
                kp = cache._k_buf if full_buffer and cache._k_buf is not None else cache.lr_tokens()[0]
                self.observe_read(attn, q_t, kp, selected.view(q_t.shape[0], q_t.shape[1], 1, -1),
                                  torch.tensor([int(pos)], device=q_t.device))
            return old_decode(attn, q_t, cache, selected, pos, full_buffer)

        self.patch(fsb_tree, "_fsb_decode_readout", decode_read)

    def patch(self, module, name, value):
        self.restores.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def close(self):
        for module, name, original in reversed(self.restores):
            setattr(module, name, original)

    def wrap_rerank(self, original, name):
        signature = inspect.signature(original)

        def run(*args, **kwargs):
            if not self.enabled or self.attn is None or self.rerank_depth:
                return original(*args, **kwargs)
            import torch

            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            assert a["K"] == 4 and a["s"] == 32 and a["score"] == "lse_pos"
            if "sel_a" in a:
                candidates = a["sel_a"] if a["sel_b"] is None else torch.cat((a["sel_a"], a["sel_b"]), -1)
            else:
                candidates = a["sel"]
            shape = candidates.shape
            pos = a["qpos"].expand(shape[:2])
            active = (pos >= self.question_start) & (pos < self.prompt_length if shape[1] > 1 else True)
            if not bool(active.any()):
                return original(*args, **kwargs)
            # Query blocks see only cells completed before the block began.
            eligible = torch.div(pos.long(), self.attn._cell_block, rounding_mode="floor") * (self.attn._cell_block // 32)
            cells = self.random_cells if self.mode == "random_readout" else self.sources[self.fact_idx]["cells"]
            targets = torch.tensor(cells, device=pos.device)[None, None, :].expand(*shape[:2], -1)
            targets = targets.masked_fill(~active[..., None] | (targets >= eligible[..., None]), -1)
            native_candidates = candidates
            if self.mode == "gold_candidates":
                injected, changed = replace_required(candidates.reshape(-1, shape[-1]), targets.reshape(-1, len(cells)))
                injected = injected.reshape(shape).contiguous()
                if "sel_a" in a:
                    ca = a["sel_a"].shape[-1]
                    a["sel_a"] = injected[..., :ca].contiguous()
                    if a["sel_b"] is not None:
                        a["sel_b"] = injected[..., ca:].contiguous()
                else:
                    a["sel"] = injected
            else:
                changed = torch.zeros(pos.numel(), dtype=torch.long, device=pos.device)
            self.rerank_depth += 1
            try:
                result = original(*bound.args, **bound.kwargs)
            finally:
                self.rerank_depth -= 1
            if result is None:
                return None  # Native rows fallback; the subsequent re-rank is tapped.
            selected = result[0] if isinstance(result, tuple) else result
            native_selected = selected.clone()
            if self.mode in ("gold_readout", "random_readout"):
                forced, changed = replace_required(selected.reshape(-1, 4), targets.reshape(-1, len(cells)))
                forced = forced.reshape_as(selected)
                selected.copy_(forced)
                if isinstance(result, tuple) and result[1] is not None:
                    result[1].masked_fill_(native_selected != selected, -1)
            self.calls += 1
            # Transfer only question/decode rows, not the thousands of prefix rows.
            ij = active.nonzero()
            r, c = ij[:, 0], ij[:, 1]
            heads = a["t_idx"][r] % self.attn.n_heads
            before = native_candidates[r, c]
            gold = self.sources[self.fact_idx]["cells"]
            gt = torch.tensor(gold, device=pos.device)
            available = a["cellA"][a["t_idx"][r, None], gt[None, :]]
            threshold = torch.minimum(self.attn.fsb_tau[heads], self.attn.fsb_tau_union[heads])
            availability = (available >= threshold[:, None]) & (gt[None, :] < eligible[r, c, None])
            info = {
                "positions": pos[r, c].long().tolist(), "heads": heads.tolist(),
                "eligible": eligible[r, c].tolist(), "source_available": availability.tolist(),
                "source_in_candidates": (before[..., None] == gt).any(1).tolist(),
                "candidate_cells": before.tolist(), "native_selected": native_selected[r, c].tolist(),
                "selected": selected[r, c].tolist(), "changed": changed.reshape(shape[:2])[r, c].tolist(),
            }
            self.emit(dict(kind="routing", index=self.index, mode=self.mode, layer=self.layers[self.attn], kernel=name, **info))
            return result
        return run

    def observe_read(self, attn, Q, Kp, selected, positions):
        import torch

        assert Q.shape[0] == 1 and selected.shape[-1] == 4
        H, nq, dh = Q.shape[1:]
        cells = selected[0].long()
        tok = (cells.clamp(min=0)[..., None] * 32 + torch.arange(32, device=Q.device)).flatten(-2)
        keys = Kp[0].gather(1, tok.reshape(H, -1)[..., None].expand(-1, -1, dh)).reshape(H, nq, 128, dh)
        scores = torch.einsum("hqd,hqkd->hqk", Q[0], keys).float() * attn.scale
        allowed = (cells >= 0).repeat_interleave(32, -1) & (tok < positions[None, :, None])
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1).nan_to_num()
        masses = []
        for source in self.sources:
            token_ids = torch.tensor(source["value_tokens"], device=Q.device)
            source_ids = torch.tensor(source["cells"], device=Q.device)
            value_mask = (tok[..., None] == token_ids).any(-1)
            cell_mask = (cells[..., None] == source_ids).any(-1).repeat_interleave(32, -1)
            masses.append(dict(value_mass=(weights * value_mask).sum(-1).tolist(),
                               cell_mass=(weights * cell_mask).sum(-1).tolist()))
        self.reads += 1
        self.emit(dict(kind="attention", index=self.index, mode=self.mode, layer=self.layers[attn],
                       positions=positions.tolist(), source_mass=masses,
                       convention="Native autocast QK and FP32 softmax, before value-dtype cast"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    forwarded = a.evaluator_args[1:] if a.evaluator_args[:1] == ["--"] else a.evaluator_args
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "experiments/scaling")]
    # Disable graph capture and the monolithic decode kernel so each intervention
    # is applied anew to the actual query. Keep native traversal/row/rerank kernels.
    os.environ.update(FSB_DECODE_GRAPH="0", FSB_DECODE_WHOLE="0", FSB_DECODE_V2="0")
    import torch
    import kv_cache
    from transformers import AutoTokenizer

    refs = [json.loads(x) for x in a.reference.read_text().splitlines()]
    header = refs[0]
    references = [x for x in refs if x["kind"] == "example"]
    data = Path(forwarded[forwarded.index("--data") + 1])
    assert digest(data.read_bytes()) == header["data_sha256"]
    records = [json.loads(x) for x in data.read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(header["tokenizer"], local_files_only=True)
    assert tok.is_fast
    original = kv_cache.cached_generate
    a.trace.parent.mkdir(parents=True, exist_ok=True)
    with a.trace.open("x") as out:
        def emit(row):
            out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()

        emit(dict(kind="manifest", diagnostic_only=True, argv=sys.argv,
                  source_commit=(root / "SOURCE_COMMIT").read_text().strip() if (root / "SOURCE_COMMIT").exists() else None,
                  driver_sha256=digest(Path(__file__).read_bytes()), reference=str(a.reference),
                  reference_sha256=digest(a.reference.read_bytes()), reference_manifest=header,
                  data_sha256=digest(data.read_bytes()),
                  modes=["native", "gold_readout", "random_readout", "gold_candidates"],
                  intervention="all layers, question tokens and generated tokens; fresh prefix forward per mode",
                  candidate_policy="replace nongold candidate slots, preserve original candidate budget",
                  env={k:v for k,v in os.environ.items() if k.startswith(("FS", "CUDA_VISIBLE", "OMP_", "MKL_"))}))
        calls = 0
        tap = None

        def generate(model, prompt_ids, *args, **kwargs):
            nonlocal calls, tap
            index = calls
            calls += 1
            ref = references[index]
            pi, qi = ref["prefix_index"], ref["question_index"]
            rec = records[pi]
            assert len(rec["facts"]) == 2 and len(rec["questions"]) == 2
            enc = tok(rec["prefix"], return_offsets_mapping=True)
            ids = enc["input_ids"] + tok.encode(rec["questions"][qi]["q"], add_special_tokens=False)
            assert prompt_ids.tolist() == ids
            assert digest(json.dumps(ids, separators=(",", ":")).encode()) == ref["prompt_tokens_sha256"]
            if tap is None:
                tap = OracleTap(model, emit)
            tap.enabled = False
            control = original(model, prompt_ids, *args, **kwargs)
            tap.index, tap.fact_idx = index, ref["fact_idx"]
            tap.question_start, tap.prompt_length = len(enc["input_ids"]), len(ids)
            tap.sources = [source_spans(rec["prefix"], enc["offset_mapping"], f) for f in rec["facts"]]
            eligible = (tap.question_start // 128) * 4
            assert all(c < eligible for src in tap.sources for c in src["cells"])
            excluded = {c for src in tap.sources for c in src["cells"]}
            count = len(tap.sources[tap.fact_idx]["cells"])
            starts = [c for c in range(eligible - count + 1) if not excluded.intersection(range(c, c + count))]
            start = random.Random(42 + index).choice(starts)
            tap.random_cells = list(range(start, start + count))
            emit(dict(kind="prompt", index=index, prefix_index=pi, question_index=qi, fact_idx=tap.fact_idx,
                      sources=tap.sources, random_cells=tap.random_cells, gold=ref["gold"],
                      prompt_tokens=len(ids), question_start=tap.question_start,
                      prompt_tokens_sha256=ref["prompt_tokens_sha256"]))
            for mode in ("native", "gold_readout", "random_readout", "gold_candidates"):
                tap.mode, tap.enabled = mode, True
                tap.calls = tap.reads = 0
                generated = original(model, prompt_ids, *args, **kwargs)
                tap.enabled = False
                assert tap.calls and tap.reads, "No instrumented routing/readout observed"
                if mode == "native":
                    assert torch.equal(control, generated), "Instrumentation changed native generation"
                genids = generated[0, len(ids):].tolist()
                pred = tok.decode(genids, skip_special_tokens=True)
                values = [f["value"] for f in rec["facts"]]
                ordinary_values = [v for v in values if v in ref["pred"]]
                current_values = [v for v in values if v in pred]
                emit(dict(kind="answer", index=index, prefix_index=pi, question_index=qi, mode=mode,
                          generated_ids=genids, pred=pred, correct=ref["gold"].lower() in pred.lower(),
                          copies_other_value=any(v != ref["gold"] and v in pred for v in values),
                          instrumentation_exact=(True if mode == "native" else None),
                          ordinary_tokens_exact=(genids == ref["generated_ids"] if mode == "native" else None),
                          ordinary_answer_values_exact=(current_values == ordinary_values if mode == "native" else None),
                          rerank_calls=tap.calls, readout_calls=tap.reads))
                del generated
            print("ORACLE_PROMPT_COMPLETE", index, flush=True)
            return control

        kv_cache.cached_generate = generate
        try:
            sys.argv = [str(root / "eval_multifact.py"), *forwarded]
            runpy.run_path(sys.argv[0], run_name="__main__")
            produced = [json.loads(x) for x in Path(forwarded[forwarded.index("--output") + 1]).read_text().splitlines()]
            actual = produced[0]
            assert actual["ckpt_sha256"] == header["ckpt_sha256"]
            assert actual["union_tau_calib"]["tau_sha256"] == header["union_tau_calib"]["tau_sha256"]
            assert actual["attn_kwargs"] == header["attn_kwargs"]
            emit(dict(kind="complete", prompts=calls))
        finally:
            kv_cache.cached_generate = original
            if tap is not None:
                tap.close()


if __name__ == "__main__":
    main()
