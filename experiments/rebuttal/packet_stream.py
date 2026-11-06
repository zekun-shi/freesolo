"""A finite framed stream; publish each packet only after complete SHA256 verification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from experiments.rebuttal.packet_io import atomic_json, packet_entries, sha256


def read_json(stream):
    line = stream.readline(1 << 20)
    if not line.endswith(b"\n"):
        raise ValueError("Missing, incomplete or oversized packet header")
    return json.loads(line)


def emit_json(stream, value):
    stream.write(json.dumps(value, separators=(",", ":")).encode() + b"\n")
    stream.flush()


def send(root, stream, *, max_seconds=259200):
    root = Path(root)
    end = time.monotonic() + max_seconds
    def wait(path):
        while not path.exists():
            if (root / "producer_failed.json").exists():
                raise RuntimeError((root / "producer_failed.json").read_text())
            if time.monotonic() >= end:
                raise TimeoutError(str(path))
            time.sleep(1)
    wait(root / "manifest.json")
    meta = json.loads((root / "manifest.json").read_text())
    emit_json(stream, dict(kind="manifest", manifest=meta))
    count = 0
    for _, _, _, name in packet_entries(meta):
        wait(root / (name + ".json"))
        rec = json.loads((root / (name + ".json")).read_text())
        path = root / name
        before = path.stat()
        if before.st_size != rec["bytes"]:
            raise ValueError("Source packet size mismatch")
        emit_json(stream, dict(kind="packet", receipt=rec))
        h = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(8 << 20), b""):
                stream.write(block); h.update(block)
                if time.monotonic() >= end:
                    raise TimeoutError("Packet stream deadline exceeded")
        stream.flush()
        after = path.stat()
        if (h.hexdigest() != rec["sha256"] or
                (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise ValueError("Source packet changed during transfer")
        count += 1
        print(json.dumps(dict(sent=name, bytes=rec["bytes"])), file=sys.stderr, flush=True)
    wait(root / "complete.json")
    emit_json(stream, dict(kind="complete", packets=count))


def receive(root, stream, *, plan_sha256, corpus_sha256, corpus_bytes, seq_len, steps):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    try:
        header = read_json(stream)
        meta = header["manifest"]
        if (header["kind"] != "manifest" or meta["schema"] != "freesolo_ordered_packets/1"
                or meta["plan_sha256"] != plan_sha256 or meta["corpus_sha256"] != corpus_sha256
                or meta["corpus_bytes"] != corpus_bytes or meta["seq_len"] != seq_len
                or meta["steps"] != steps or meta["source"]["sha256"] != corpus_sha256
                or meta["source"]["bytes"] != corpus_bytes
                or not 1 <= meta["shard_updates"] <= 512 or meta["validation_rows"] != 20):
            raise ValueError("Stream does not match the registered source and sample plan")
        atomic_json(root / "manifest.json", meta)
        count = 0
        for split, first, rows, name in packet_entries(meta):
            header = read_json(stream)
            rec = header["receipt"]
            expected_bytes = rows * (16 if split == "validation" else 128) * (seq_len + 1) * 2
            if (header["kind"] != "packet" or rec["file"] != name or rec["split"] != split
                    or rec["first_update"] != first or rec["updates"] != rows
                    or rec["bytes"] != expected_bytes or rec["plan_sha256"] != plan_sha256
                    or rec["corpus_sha256"] != corpus_sha256):
                raise ValueError("Unexpected packet identity, size or order")
            partial = root / (name + ".part")
            h = hashlib.sha256()
            remaining = expected_bytes
            with partial.open("xb") as out:
                while remaining:
                    block = stream.read(min(8 << 20, remaining))
                    if not block:
                        raise ValueError("Truncated packet body")
                    out.write(block); h.update(block); remaining -= len(block)
                out.flush()
                import os
                os.fsync(out.fileno())
            if h.hexdigest() != rec["sha256"]:
                raise ValueError("Packet body checksum mismatch")
            partial.replace(root / name)
            atomic_json(root / (name + ".json"), rec)
            count += 1
            print(json.dumps(dict(verified=name, bytes=expected_bytes, completed_at=time.time())), flush=True)
        footer = read_json(stream)
        if footer != dict(kind="complete", packets=count):
            raise ValueError("Invalid packet stream footer")
        atomic_json(root / "complete.json", dict(packets=count, manifest_sha256=sha256(root / "manifest.json")))
    except BaseException as error:
        atomic_json(root / "transfer_failed.json", dict(error=repr(error)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("direction", choices=["send", "receive"])
    p.add_argument("--root", required=True)
    p.add_argument("--plan-sha256")
    p.add_argument("--corpus-sha256")
    p.add_argument("--corpus-bytes", type=int)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--steps", type=int)
    p.add_argument("--max-seconds", type=int, default=259200)
    a = p.parse_args()
    if a.direction == "send":
        send(a.root, sys.stdout.buffer, max_seconds=a.max_seconds)
    else:
        receive(a.root, sys.stdin.buffer, plan_sha256=a.plan_sha256, corpus_sha256=a.corpus_sha256,
                corpus_bytes=a.corpus_bytes, seq_len=a.seq_len, steps=a.steps)
