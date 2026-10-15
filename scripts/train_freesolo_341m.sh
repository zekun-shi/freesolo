#!/bin/bash
# Train FreeSolo 341M from scratch on 14.68B tokens (28,000 steps x 524,288 tok/step, 2x8 GPUs).
# Env: DATA, RUNS, RUN, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, GPUS_PER_NODE, RESUME;
#      FS_FUSED_ATTN, FS_STRICT_CAUSAL select the kernel path. The bounded cell pool (FS_PRUNE_*) is NOT
#      enabled here: base FreeSolo trains without it; FS-L enables it during CPT (scripts/cpt_*.sh).
set -u
RUN=${RUN:-freesolo_341m}
PER_GPU=1; ACCUM=8; SEQLEN=4096; MAXSTEPS=28000; VOCAB=32000; TOPK=4
RESUME=${RESUME:-}   # set to a checkpoint dir to resume; empty = from scratch
NNODES=${NNODES:-2}; NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}; NPROC=$((NNODES * GPUS_PER_NODE))
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}; MASTER_PORT=${MASTER_PORT:-29531}
DATA=${DATA:-./data/fineweb_edu_llama32k.bin}
RUNS=${RUNS:-./runs}
export WANDB_MODE=${WANDB_MODE:-disabled}
export FS_FUSED_ATTN=1 FS_STRICT_CAUSAL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}   # set 1 to disable InfiniBand and force TCP
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
C="--config 340m --attn_type fsattn --long_range_variant cell_topk --cell_select score --cell_topk $TOPK --cell_stride 32 \
--sel_strict_causal 1 --fs_nope_longrange --fs_uniform_sr --window_size 512 \
--max_seq_len $SEQLEN --vocab $VOCAB \
--per_gpu_batch_size $PER_GPU --grad_accum_steps $ACCUM --max_steps $MAXSTEPS --lr 3e-4 --min_lr 3e-5 --warmup_steps 1000 \
--weight_decay 0.1 --grad_clip 1.0 --save_every 1000 --eval_every 1000 \
${RESUME:+--resume $RESUME} \
--pretokenized $DATA"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || { echo "cd failed"; exit 1; }
accelerate launch --num_machines "$NNODES" --num_processes "$NPROC" --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" --same_network --mixed_precision bf16 \
  experiments/scaling/train_dist.py $C --run_name "$RUN" --output_dir "$RUNS/$RUN"
