"""Preserve the original global batches across different physical GPU counts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def batch_partition(width, rank, world_size, *, validation=False):
    """Keep training disjoint; uniformly repeat small validation batches.

    The fixed validation rows contain sixteen examples. Larger worlds assign
    each example to the same number of ranks, preserving its weight in the
    distributed mean without changing the frozen validation plan.
    """
    if width < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid batch width or GPU rank/topology")
    if validation and world_size > width and world_size % width == 0:
        return rank % width, 1
    if width % world_size:
        raise ValueError("GPU topology does not divide the fixed global batch")
    return rank, width // world_size


def batch_plan(total_chunks, virtual_world, reference_batch, steps, seed=42):
    """Replay the historical two-worker IterableDataset, including epoch wraps."""
    if virtual_world * reference_batch * 8 != 128:
        raise ValueError("Expected the published 128-sequence, eight-microstep recipe")
    val_start = int(total_chunks * 0.99)
    plan = np.empty((steps, 8, virtual_world, reference_batch), dtype=np.int64)
    for rank in range(virtual_world):
        shards = []
        for worker in range(2):
            gid = rank * 2 + worker
            ids = np.arange(gid, val_start, virtual_world * 2, dtype=np.int64)
            np.random.default_rng(seed + gid).shuffle(ids)
            shards.append(ids)
        counts = [len(ids) // reference_batch for ids in shards]
        if min(counts) < 1:
            raise ValueError("Each reference worker needs a complete batch")
        cycle = np.arange(steps * 8) % sum(counts)
        paired = cycle < 2 * min(counts)
        worker = np.where(paired, cycle % 2, int(np.argmax(counts)))
        number = np.where(paired, cycle // 2, min(counts) + cycle - 2 * min(counts))
        draws = np.empty((steps * 8, reference_batch), dtype=np.int64)
        for wid in range(2):
            selected = worker == wid
            indices = number[selected, None] * reference_batch + np.arange(reference_batch)
            draws[selected] = shards[wid][indices]
        plan[:, :, rank] = draws.reshape(steps, 8, reference_batch)
    val_count = min(320, (total_chunks - val_start) // 16 * 16)
    if not val_count:
        raise ValueError("Need at least sixteen validation chunks")
    return plan.reshape(steps, 128), np.arange(val_start, val_start + val_count).reshape(-1, 16)


def write_plan(folder, *, corpus_bytes, corpus_sha256, seq_len, virtual_world,
               reference_batch, steps, seed=42):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    train, val = batch_plan(corpus_bytes // (2 * (seq_len + 1)), virtual_world,
                            reference_batch, steps, seed)
    files = {}
    for name, array in (("train", train), ("validation", val)):
        path = folder / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        files[name] = dict(file=path.name, shape=list(array.shape),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = dict(schema="freesolo_matched_batches/1", corpus_bytes=corpus_bytes,
                    corpus_sha256=corpus_sha256, seq_len=seq_len, seed=seed, steps=steps,
                    reference_world=virtual_world, reference_batch=reference_batch,
                    reference_workers=2, reference_accumulation=8, files=files)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class MatchedBatchDataset(Dataset):
    def __init__(self, data_path, manifest_path, seq_len, rank, world_size, *,
                 validation=False, corpus_sha256=None):
        self.data_path = str(data_path)
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if (manifest["schema"] != "freesolo_matched_batches/1"
                or manifest["seq_len"] != seq_len
                or Path(data_path).stat().st_size != manifest["corpus_bytes"]
                or manifest["corpus_sha256"] != corpus_sha256):
            raise ValueError("Data plan differs from the declared corpus/protocol")
        entry = manifest["files"]["validation" if validation else "train"]
        path = manifest_path.parent / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Data-plan hash mismatch")
        self.plan = np.load(path, mmap_mode="r", allow_pickle=False)
        rank, per_rank = batch_partition(self.plan.shape[1], rank, world_size,
                                         validation=validation)
        self.seq_len, self.rank = seq_len, rank
        self.per_rank = per_rank
        self.tokens = None
        self.start_update = 0

    def __len__(self):
        return (self.plan.shape[0] - self.start_update) * self.per_rank

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.tokens is None:
            self.tokens = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        row, offset = divmod(index, self.per_rank)
        row += self.start_update
        cid = int(self.plan[row, self.rank * self.per_rank + offset])
        start = cid * (self.seq_len + 1)
        chunk = self.tokens[start:start + self.seq_len + 1].astype(np.int64)
        if len(chunk) != self.seq_len + 1:
            raise ValueError("Data plan references an incomplete chunk")
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--corpus-bytes", type=int, required=True)
    p.add_argument("--corpus-sha256", required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--virtual-world", type=int, required=True)
    p.add_argument("--reference-batch", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = vars(p.parse_args()); folder = a.pop("output")
    print(json.dumps(write_plan(folder, **a), indent=2))
