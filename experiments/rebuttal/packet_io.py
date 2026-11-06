"""Packet metadata helpers usable on an entry host without a CUDA runtime."""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def packet_entries(meta):
    yield "validation", 0, meta["validation_rows"], "validation.bin"
    for start in range(0, meta["steps"], meta["shard_updates"]):
        yield "train", start, min(meta["shard_updates"], meta["steps"] - start), f"train_{start:06d}.bin"
