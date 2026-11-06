"""Verified, ordered batches from the unchanged corpus and global sample plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_plan(path):
    path = Path(path)
    meta = json.loads(path.read_text())
    if meta["schema"] != "freesolo_matched_batches/1":
        raise ValueError("Unsupported global-batch plan")
    arrays = {}
    for split, entry in meta["files"].items():
        file = path.parent / entry["file"]
        if sha256(file) != entry["sha256"]:
            raise ValueError("Global-batch plan hash mismatch")
        arrays[split] = np.load(file, mmap_mode="r", allow_pickle=False)
        if list(arrays[split].shape) != entry["shape"]:
            raise ValueError("Global-batch plan shape mismatch")
    return meta, arrays


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


def source_identity(path):
    path = Path(path).resolve()
    st = path.stat()
    return dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)


def packet_entries(meta):
    """Validation arrives first; each training packet contains complete updates."""
    yield "validation", 0, meta["validation_rows"], "validation.bin"
    for start in range(0, meta["steps"], meta["shard_updates"]):
        yield "train", start, min(meta["shard_updates"], meta["steps"] - start), f"train_{start:06d}.bin"


def write_packets(data_path, plan_path, source_receipt, output, shard_updates=128):
    plan, arrays = read_plan(plan_path)
    source = json.loads(Path(source_receipt).read_text())
    identity = source_identity(data_path)
    if (any(source.get(k) != v for k, v in identity.items())
            or source["sha256"] != plan["corpus_sha256"]
            or source["bytes"] != plan["corpus_bytes"] or shard_updates < 1):
        raise ValueError("Source corpus differs from its complete hash receipt or plan")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    meta = dict(schema="freesolo_ordered_packets/1", source=source,
                corpus_sha256=plan["corpus_sha256"], corpus_bytes=plan["corpus_bytes"],
                plan_sha256=sha256(plan_path), seq_len=plan["seq_len"], steps=plan["steps"],
                shard_updates=shard_updates, validation_rows=len(arrays["validation"]),
                source_host=os.uname().nodename)
    atomic_json(root / "manifest.json", meta)
    tokens = np.memmap(data_path, dtype=np.uint16, mode="r")
    stride = plan["seq_len"] + 1
    try:
        for split, first, rows, name in packet_entries(meta):
            path = root / name
            temporary = path.with_suffix(".bin.part")
            h = hashlib.sha256()
            count = 0
            with temporary.open("xb") as out:
                for row in arrays[split][first:first + rows]:
                    for cid in row:
                        start = int(cid) * stride
                        chunk = tokens[start:start + stride]
                        if len(chunk) != stride:
                            raise ValueError("Plan references an incomplete source chunk")
                        blob = chunk.tobytes()
                        out.write(blob); h.update(blob); count += len(blob)
                out.flush(); os.fsync(out.fileno())
            if source_identity(data_path) != identity:
                raise ValueError("Source corpus changed during packet extraction")
            temporary.replace(path)
            receipt = dict(file=name, split=split, first_update=first, updates=rows,
                           bytes=count, sha256=h.hexdigest(), plan_sha256=meta["plan_sha256"],
                           corpus_sha256=meta["corpus_sha256"])
            atomic_json(root / (name + ".json"), receipt)
            print(json.dumps(receipt), flush=True)
        atomic_json(root / "complete.json", dict(manifest_sha256=sha256(root / "manifest.json"),
                                                packets=sum(1 for _ in packet_entries(meta))))
    except BaseException as error:
        atomic_json(root / "producer_failed.json", dict(error=repr(error)))
        raise
    return meta


class PacketBatchDataset(Dataset):
    def __init__(self, root, plan_path, seq_len, rank, world_size, *, validation=False,
                 corpus_sha256=None, wait_seconds=7200):
        self.root = Path(root)
        self.meta = meta = json.loads((self.root / "manifest.json").read_text())
        plan, arrays = read_plan(plan_path)
        if (meta["schema"] != "freesolo_ordered_packets/1" or meta["seq_len"] != seq_len
                or meta["plan_sha256"] != sha256(plan_path)
                or meta["corpus_sha256"] != corpus_sha256
                or meta["source"]["sha256"] != corpus_sha256
                or meta["source"]["bytes"] != plan["corpus_bytes"]
                or meta["steps"] != plan["steps"] or meta["shard_updates"] < 1):
            raise ValueError("Packet corpus or sampling plan mismatch")
        self.validation = validation
        self.plan = arrays["validation" if validation else "train"]
        if self.plan.shape[1] % world_size or not 0 <= rank < world_size:
            raise ValueError("GPU topology does not divide the global batch")
        self.per_rank = self.plan.shape[1] // world_size
        self.rank, self.seq_len, self.wait_seconds = rank, seq_len, wait_seconds
        self._name, self._tokens = None, None
        self.start_update = 0

    def __len__(self):
        return (len(self.plan) - self.start_update) * self.per_rank

    def _load(self, name, first, rows):
        if self._name == name:
            return self._tokens
        receipt_path = self.root / (name + ".json")
        deadline = time.monotonic() + self.wait_seconds
        while not receipt_path.exists():
            for marker in ("producer_failed.json", "transfer_failed.json"):
                if (self.root / marker).exists():
                    raise RuntimeError("Packet producer/transfer failed: " + (self.root / marker).read_text())
            if time.monotonic() >= deadline:
                raise TimeoutError("Verified batch packet unavailable: " + name)
            time.sleep(1)
        rec = json.loads(receipt_path.read_text())
        path = self.root / name
        expected_bytes = rows * self.plan.shape[1] * (self.seq_len + 1) * 2
        if (rec["file"] != name or rec["first_update"] != first or rec["updates"] != rows
                or rec["plan_sha256"] != self.meta["plan_sha256"]
                or rec["corpus_sha256"] != self.meta["corpus_sha256"]
                or rec["bytes"] != expected_bytes or path.stat().st_size != expected_bytes
                or sha256(path) != rec["sha256"]):
            raise ValueError("Packet identity, order or checksum mismatch: " + name)
        self._tokens = np.memmap(path, mode="r", dtype=np.uint16).reshape(
            rows, self.plan.shape[1], self.seq_len + 1)
        self._name = name
        return self._tokens

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        row, offset = divmod(index, self.per_rank)
        row += self.start_update
        if self.validation:
            first, rows, name = 0, len(self.plan), "validation.bin"
        else:
            first = row // self.meta["shard_updates"] * self.meta["shard_updates"]
            rows = min(self.meta["shard_updates"], len(self.plan) - first)
            name = f"train_{first:06d}.bin"
        chunk = self._load(name, first, rows)[row - first, self.rank * self.per_rank + offset]
        chunk = chunk.astype(np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--source-receipt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-updates", type=int, default=128)
    a = p.parse_args()
    write_packets(a.data, a.plan, a.source_receipt, a.output, a.shard_updates)
