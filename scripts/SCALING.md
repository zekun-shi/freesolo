# Matched scaling runs

`train_scaling_law.py` runs from-scratch FS and a dense RoPE Transformer (TF++)
in the same trainer. Both use tied embeddings, RMSNorm, SwiGLU, AdamW, bf16,
4,096-token contexts, and 524,288 training tokens per optimizer update. FS uses
a local 512-token window and reads four 32-token cells, without pruning.
TF++ uses causal PyTorch scaled-dot-product attention and its CUDA fused
backends, without requiring the optional FlashAttention Python package.

| `--size` | Layers | Width | Heads | FFN width | Budgets (billions of training tokens) |
| --- | --- | --- | --- | --- | --- |
| `150m` | 12 | 896 | 14 | 2,560 | 1.9, 3.75, 7.5, 15 |
| `340m` | 24 | 1,024 | 16 | 2,816 | 3.75, 7.5, 15, 30 |
| `800m` | 24 | 1,536 | 16 | 4,224 | 7.5, 15, 30, 60 |
| `1.3b_fs` | 24 | 2,048 | 16 | 5,632 | 12.5, 25, 50, 100 |

The smallest model uses width 896 to reach approximately 150M with the 32k
tokenizer. Width 768, 12 layers and FFN width 2048 would give about 110M.
Checkpoints report actual parameter counts, including FS's attention parameters.
The FS counts are 149,806,032 / 341,101,824 / 742,863,360 / 1,298,765,568.

## Environment and data

Install the same committed revision and Python/CUDA packages on every node:

```bash
pip install -e '.[dev]'
```

Use the Llama-32k tokenization of FineWeb-Edu `sample-100BT`, a flat
little-endian uint16 file containing 110,823,553,104 tokens. Required size:
221,647,106,208 bytes. Required SHA256:

```text
a22c79baec4f8b53217fde674d78f6e4f01cbaa32120c51f6936ace58ee0c5c0
```

The launcher checks the full hash before training and saves a verification
receipt. Every launch rechecks the bytes; a receipt is provenance rather than
a cache, since timestamps can miss same-size changes on some filesystems.
Use read-only corpus files. The 40BT corpus and other 100BT tokenizations are
not substitutes. Data transfer is separate. The tokenizer is
`fla-hub/transformer-1.3B-100B`, revision
`d6f66f4181fa669e5863327815b44533e3a395e7`.

The trainer packs 4,097-token chunks into 4,096 input/target pairs. It excludes
the last 1% of chunks from training and scores the first 2,560 held-out chunks
(10,485,760 target tokens) at evaluation points and every annealed endpoint.
A seeded global permutation fixes training order across architectures. Chunks
are used once per trajectory; branches reuse the corresponding trunk data.
Exhaustion is an error, never an automatic second epoch.

## Preview and launch a trunk

A preview imports no training libraries and starts no workers:

```bash
python scripts/train_scaling_law.py --arm fs --size 340m \
  --data /data/fineweb_edu_llama32k_100bt.bin \
  --output-root /runs/scaling --gpus-per-node 8 --dry-run
```

Remove `--dry-run` to train. Use `--arm tfpp` for the dense control.
Select free GPUs with `CUDA_VISIBLE_DEVICES` and match `--gpus-per-node` to
their count. Accumulation adjusts automatically: 8 GPUs with batch 1 use 16
microbatches per update; 16 GPUs use 8. Different jobs need different run
identities or output roots and master ports. Run one trunk per architecture,
size and LR multiplier, not one trunk per budget.

For two nodes with eight GPUs each, run on both nodes with their own rank:

```bash
NODE_RANK=0 NNODES=2 GPUS_PER_NODE=8 MASTER_ADDR=10.0.0.10 MASTER_PORT=29531 \
  python scripts/train_scaling_law.py --arm fs --size 800m \
  --data /data/fineweb_edu_llama32k_100bt.bin --output-root /runs/scaling
```

Use `NODE_RANK=1` on the second node. The master address must be reachable
from both nodes. Preserve the cluster's NCCL/network settings; the launcher
does not configure networking. Use a shared output filesystem, or gather all
rank RNG files and replicate complete checkpoints onto every node before
resuming. Dataset copies can live on local disks.

Keep the revision, package versions, GPU topology, batch size, worker count,
precision and kernel settings fixed between trunk and cooldowns. Resume rejects
incompatible settings. Launches require committed code, clear inherited
`FS_*`/`FSB_*` overrides, select base FS without pruning, and disable TF32.
`--compile`, `--fused-adam` and `--gradient-checkpointing` are explicit options;
match them across controls and branches and validate them on the target GPU.

## Cooldown branches and recovery

The WSD schedule uses 2,000 warmup updates, a constant peak, then linear
cooldown over the final 20% of each budget to 10% of peak. Peak LR is 3e-4
for 150M/341M and 4e-4 for 743M/1.3B. Budgets round up to whole updates;
previews and checkpoint metadata report exact steps and tokens.

The trunk saves full model, optimizer and per-rank RNG state at 80% of each
budget. It continues on the plateau for smaller budgets, then anneals to the
largest budget. Run three separate cooldown jobs from the smaller stable
checkpoints, for example:

```bash
python scripts/train_scaling_law.py --arm fs --size 340m \
  --phase cooldown --budget 7.5 \
  --data /data/fineweb_edu_llama32k_100bt.bin \
  --output-root /runs/scaling --gpus-per-node 8
```

This finds the 7.5B branch's full stable checkpoint and writes to a separate
directory. The largest budget is already annealed at the trunk's end. Branches
can run after their stable checkpoint completes, on separate allocated GPUs
with the same topology. Resume seeks to the saved data cursor and restores
optimizer and RNG state. Do not use weight-only `--init_from` for branches.

To recover an interrupted trunk or cooldown, repeat its command with
`--resume /runs/scaling/.../<run>_step<number>`. Keep all checkpoint files,
including `scaling_state.json` and RNG states. Back up before releasing nodes;
the launcher does not manage backups or delete old checkpoints.

Use `--lr-multiplier 0.5`, `1` or `2` for the planned small-model LR sweep.
The multiplier is in the run identity and must also match on cooldown commands.
Optional peer architectures and FS-B CPT are separate experiments.

## Outputs and checks

Launch manifests preserve the command, code revision, dataset hash, resolved
environment and topology. Each node writes a stdout log. WSD checkpoints record
parameters, updates, tokens, the data cursor and resume contract. Compare fixed
annealed endpoints, not intermediate `best` checkpoints. RULER and commonsense
evaluation are separate jobs. The release's task launchers currently target
FS variants; TF++ task-evaluation integration is separate from this training
launcher. Held-out loss evaluation supports both arms here.

Local checks include a tiny CPU full-state cooldown round trip:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_scaling_law.py -q
```

A short FS/TF++ GPU training and distributed resume smoke on the intended
hardware is still required before large allocations. Installation and
`--dry-run` never launch training.
