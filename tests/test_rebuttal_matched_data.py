"""Compare fixed update batches to the real historical worker/data-loader streams."""
import hashlib
import json

import numpy as np
import pytest
from torch.utils.data import DataLoader

from experiments.rebuttal.matched_data import batch_plan, write_plan, MatchedBatchDataset
from experiments.scaling.train_dist import DistPreTokenizedDataset


@pytest.mark.parametrize("world,batch", [(8, 2), (16, 1)])
def test_replays_actual_multiworker_loader_across_epoch_wraps(tmp_path, world, batch):
    chunks, seq, steps = 1703, 8, 40
    path = tmp_path / "tokens.bin"
    np.repeat(np.arange(chunks, dtype=np.uint16), seq + 1).tofile(path)
    plan, _ = batch_plan(chunks, world, batch, steps)
    actual = np.empty((steps, 8, world, batch), dtype=np.int64)
    for rank in range(world):
        loader = DataLoader(DistPreTokenizedDataset(path, seq, rank, world),
                            batch_size=batch, num_workers=2, drop_last=True)
        iterator = iter(loader)
        for micro in range(steps * 8):
            try:
                x, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                x, _ = next(iterator)
            actual[micro // 8, micro % 8, rank] = x[:, 0].numpy()
        del iterator, loader
    np.testing.assert_array_equal(plan, actual.reshape(steps, 128))


@pytest.mark.parametrize("world", [1, 2, 4, 8, 16])
def test_physical_topologies_receive_same_update_batches_and_validation(tmp_path, world):
    chunks, seq = 1703, 8
    path = tmp_path / "tokens.bin"
    np.repeat(np.arange(chunks, dtype=np.uint16), seq + 1).tofile(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    folder = tmp_path / "plan"
    write_plan(folder, corpus_bytes=path.stat().st_size, corpus_sha256=sha,
               seq_len=seq, virtual_world=16, reference_batch=1, steps=3)
    for validation in (False, True):
        datasets = [MatchedBatchDataset(path, folder / "manifest.json", seq, rank, world,
                      validation=validation, corpus_sha256=sha) for rank in range(world)]
        expected = np.load(folder / ("validation.npy" if validation else "train.npy"))
        for step in range(len(expected)):
            actual = []
            for ds in datasets:
                for offset in range(ds.per_rank):
                    x, y = ds[step * ds.per_rank + offset]
                    assert (x == y).all()
                    actual.append(int(x[0]))
            np.testing.assert_array_equal(actual, expected[step])


def test_rejects_wrong_corpus_or_changed_plan(tmp_path):
    path = tmp_path / "tokens.bin"
    np.zeros(1703 * 9, dtype=np.uint16).tofile(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    folder = tmp_path / "plan"
    write_plan(folder, corpus_bytes=path.stat().st_size, corpus_sha256=sha,
               seq_len=8, virtual_world=16, reference_batch=1, steps=3)
    with pytest.raises(ValueError, match="corpus/protocol"):
        MatchedBatchDataset(path, folder / "manifest.json", 8, 0, 4, corpus_sha256="bad")
    with (folder / "train.npy").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        MatchedBatchDataset(path, folder / "manifest.json", 8, 0, 4, corpus_sha256=sha)
