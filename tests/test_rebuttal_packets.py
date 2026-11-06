import hashlib
import io
import json

import numpy as np
import pytest

from experiments.rebuttal.matched_data import write_plan, MatchedBatchDataset
from experiments.rebuttal.packet_data import write_packets, PacketBatchDataset, source_identity, sha256
from experiments.rebuttal.packet_stream import send, receive


@pytest.fixture
def corpus(tmp_path):
    # Enough validation chunks for the real 20 x 16 validation protocol.
    path = tmp_path / "tokens.bin"
    rng = np.random.default_rng(191)
    rng.integers(0, 32000, size=33003 * 9, dtype=np.uint16).tofile(path)
    digest = sha256(path)
    receipt = tmp_path / "source.json"
    receipt.write_text(json.dumps(dict(source_identity(path), sha256=digest)))
    plan = tmp_path / "plan"
    write_plan(plan, corpus_bytes=path.stat().st_size, corpus_sha256=digest, seq_len=8,
               virtual_world=16, reference_batch=1, steps=5)
    packets = tmp_path / "packets"
    write_packets(path, plan / "manifest.json", receipt, packets, shard_updates=2)
    return path, plan / "manifest.json", packets, digest


@pytest.mark.parametrize("world", [1, 2, 4, 8, 16])
def test_packets_match_original_corpus_at_every_rank_boundary(corpus, world):
    path, plan, packets, digest = corpus
    for validation in (False, True):
        for rank in range(world):
            reference = MatchedBatchDataset(path, plan, 8, rank, world,
                                            validation=validation, corpus_sha256=digest)
            actual = PacketBatchDataset(packets, plan, 8, rank, world,
                                         validation=validation, corpus_sha256=digest)
            assert len(reference) == len(actual)
            for i in range(len(actual)):
                xr, yr = reference[i]; x, y = actual[i]
                assert (xr == x).all() and (yr == y).all()


def test_stream_roundtrip_and_reject_corrupt_or_truncated_body(corpus, tmp_path):
    path, plan, packets, digest = corpus
    stream = io.BytesIO()
    send(packets, stream)
    blob = stream.getvalue()
    kwargs = dict(plan_sha256=sha256(plan), corpus_sha256=digest,
                  corpus_bytes=path.stat().st_size, seq_len=8, steps=5)
    target = tmp_path / "received"
    receive(target, io.BytesIO(blob), **kwargs)
    for file in packets.glob("*.bin"):
        assert (target / file.name).read_bytes() == file.read_bytes()
    assert (target / "complete.json").exists()
    lines = blob.split(b"\n", 2)
    body_start = len(lines[0]) + len(lines[1]) + 2
    corrupted = bytearray(blob)
    corrupted[body_start + 2] ^= 1
    for label, bad in (("corrupt", corrupted), ("truncated", blob[:body_start + 10])):
        out = tmp_path / label
        with pytest.raises(ValueError):
            receive(out, io.BytesIO(bad), **kwargs)
        assert not (out / "validation.bin.json").exists()
        assert (out / "transfer_failed.json").exists()


def test_future_packets_cannot_be_read_early_and_failure_is_not_hidden(corpus):
    _, plan, packets, digest = corpus
    (packets / "train_000002.bin.json").unlink()
    ds = PacketBatchDataset(packets, plan, 8, 0, 4, corpus_sha256=digest, wait_seconds=0)
    ds[0]  # The initial verified packet is immediately usable.
    with pytest.raises(TimeoutError):
        ds[2 * ds.per_rank]
    (packets / "transfer_failed.json").write_text('{"error":"stream stopped"}')
    with pytest.raises(RuntimeError, match="stream stopped"):
        ds[2 * ds.per_rank]


def test_dataset_rejects_same_size_corruption(corpus):
    _, plan, packets, digest = corpus
    file = packets / "train_000000.bin"
    content = bytearray(file.read_bytes()); content[5] ^= 1; file.write_bytes(content)
    ds = PacketBatchDataset(packets, plan, 8, 0, 4, corpus_sha256=digest)
    with pytest.raises(ValueError, match="checksum mismatch"):
        ds[0]


def test_resume_begins_at_exact_update_without_replaying_earlier_packets(corpus):
    path, plan, packets, digest = corpus
    full = MatchedBatchDataset(path, plan, 8, 2, 4, corpus_sha256=digest)
    restored = PacketBatchDataset(packets, plan, 8, 2, 4, corpus_sha256=digest)
    restored.start_update = 3
    (packets / "train_000000.bin.json").unlink()
    assert len(restored) == len(full) - 3 * full.per_rank
    for i in range(len(restored)):
        xr, yr = full[i + 3 * full.per_rank]
        x, y = restored[i]
        assert (x == xr).all() and (y == yr).all()
