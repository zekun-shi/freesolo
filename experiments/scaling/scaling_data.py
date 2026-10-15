"""Single-pass, rank-sharded token data with a directly seekable resume cursor."""

import os

import numpy as np
import torch
from torch.utils.data import IterableDataset


class ScalingTokenDataset(IterableDataset):
    def __init__(self, path, seq_len, rank, world_size, batch_size, seed=42,
                 start_microbatch=0, validation_chunks=0):
        self.path = path
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.seed = seed
        self.start_microbatch = start_microbatch
        self.validation_chunks = validation_chunks
        total = os.path.getsize(path) // (2 * (seq_len + 1))
        self.train_chunks = total * 99 // 100
        if validation_chunks:
            if self.train_chunks + validation_chunks > total:
                raise ValueError("The held-out split is smaller than validation_chunks")
            if validation_chunks % (world_size * batch_size):
                raise ValueError("validation_chunks must divide into complete global batches")
        self.chunks = validation_chunks or self.train_chunks

    def __iter__(self):
        data = np.memmap(self.path, dtype="<u2", mode="r")
        if self.validation_chunks:
            order = np.arange(self.train_chunks, self.train_chunks + self.chunks)
        else:
            order = np.arange(self.train_chunks)
            np.random.default_rng(self.seed).shuffle(order)
        worker = torch.utils.data.get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        global_batch = self.world_size * self.batch_size
        for microbatch in range(self.start_microbatch + worker_id,
                                self.chunks // global_batch, workers):
            first = microbatch * global_batch + self.rank * self.batch_size
            for chunk_id in order[first:first + self.batch_size]:
                start = int(chunk_id) * (self.seq_len + 1)
                chunk = data[start:start + self.seq_len + 1].astype(np.int64)
                yield torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())
