#!/bin/bash
# FreeSolo-L 1.3B continued pretraining on ONE 4-GPU GB200 node, inside the cuda-rt image.
# Recipe = scripts/cpt_freesolo_l_13b.sh (weights-only warm start from the dense 1.3B at
# final_step106811, bounded cell pool FS_PRUNE_*, cosine 4e-4 -> 4e-5 over 6000 steps).
# Only the layout differs from the 2-node script, and it keeps the campaign token batch:
# 4 GPUs x PER_GPU 4 x ACCUM 8 x 4096 = 524288 tok/step (guarded below).
# Single node has no IMEX channel, so NCCL uses tcp bootstrap on bond0; the data path
# stays on NVLink P2P and the patched socket helpers are not needed (NCCL_SOCKET_NTHREADS=0).
# Env: INIT_FROM (required), RUN, DATA, RUNS, MAXSTEPS, SAVE_EVERY, LR, MINLR, WARMUP,
#      PER_GPU, FS_COMM, COMPILE.
set -euo pipefail
RUN=${RUN:-freesolo_l_13b_cpt}
REPO=${REPO:-/workspace/freesolo-gb200}
INIT_FROM=${INIT_FROM:?set INIT_FROM to the dense-trained FreeSolo 1.3B checkpoint}
DATA=${DATA:-/data/fineweb_edu_llama32k_40bt.bin}
RUNS=${RUNS:-/runs}
MAXSTEPS=${MAXSTEPS:-6000}
SAVE_EVERY=${SAVE_EVERY:-200}   # 30 checkpoints x ~15.6 GB (model+optimizer) ~ 470 GB on /runs
LR=${LR:-4e-4}; MINLR=${MINLR:-4e-5}; WARMUP=${WARMUP:-300}
PER_GPU=${PER_GPU:-4}; WORLD=4
MASTER_IP=${MASTER_IP:-$(hostname -I | cut -d' ' -f1)}   # bond0 address, the binding NCCL uses
[[ $((128 % (WORLD * PER_GPU))) == 0 ]]
ACCUM=$((128 / (WORLD * PER_GPU)))
[[ $((WORLD * PER_GPU * ACCUM * 4096)) == 524288 ]]
COMPILE=${COMPILE:-0}; CF=""; [ "$COMPILE" = "1" ] && CF="--compile"
export FS_COMM=${FS_COMM:-tcp} FS_FUSED_ATTN=1 FS_STRICT_CAUSAL=1
export FS_PRUNE_KERNEL=1 FS_PRUNE_POOL=64 FS_PRUNE_WINDOW=8 FS_PRUNE_BLKCHUNK=16
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_NTHREADS=0
export TRITON_CACHE_DIR="$RUNS/triton_$RUN"; mkdir -p "$TRITON_CACHE_DIR" "$RUNS"
if [ "$FS_COMM" = tcp ]; then
  # Single node, no IMEX channel: tcp bootstrap on bond0, RoCE off, stock NCCL sockets.
  export NCCL_MNNVL_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none
  export NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0
else
  echo "without an IMEX channel device only FS_COMM=tcp is supported on this node" >&2; exit 2
fi
L=$RUNS/${RUN}.log
{
  echo "GB200_FSL13B_CPT $(date -u +%FT%TZ) run=$RUN world=$WORLD per_gpu=$PER_GPU accum=$ACCUM max_steps=$MAXSTEPS rev=$(cat "$REPO.rev" 2>/dev/null) host=$(hostname)"
  echo "NCCL mode=$FS_COMM mnnvl=$NCCL_MNNVL_ENABLE nvls=$NCCL_NVLS_ENABLE ib_disable=$NCCL_IB_DISABLE net_plugin=${NCCL_NET_PLUGIN:-} nthreads=${NCCL_SOCKET_NTHREADS:-}"
  echo "FS_PRUNE kernel=$FS_PRUNE_KERNEL pool=$FS_PRUNE_POOL window=$FS_PRUNE_WINDOW blkchunk=$FS_PRUNE_BLKCHUNK"
  echo "INIT_FROM $INIT_FROM DATA $DATA"
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
} >> "$L"
pushd "$REPO" >/dev/null
exec accelerate launch --num_processes "$WORLD" --num_machines 1 \
  --machine_rank 0 --main_process_ip "$MASTER_IP" \
  --main_process_port 29563 --same_network --mixed_precision bf16 \
  experiments/scaling/train_dist.py \
  --config 1.3b_fs --attn_type fsattn --require_freesolo \
  --long_range_variant cell_topk --cell_select score --cell_topk 4 --cell_stride 32 \
  --sel_strict_causal 1 --fs_nope_longrange --fs_uniform_sr --window_size 512 \
  --max_seq_len 4096 --vocab 32000 --per_gpu_batch_size "$PER_GPU" --grad_accum_steps "$ACCUM" \
  --max_steps "$MAXSTEPS" --lr "$LR" --min_lr "$MINLR" --warmup_steps "$WARMUP" \
  --weight_decay 0.1 --grad_clip 1.0 --seed 42 --save_every "$SAVE_EVERY" --eval_every 999999 \
  $CF --init_from "$INIT_FROM" --pretokenized "$DATA" \
  --run_name "$RUN" --output_dir "$RUNS/$RUN" >> "$L" 2>&1 < /dev/null
