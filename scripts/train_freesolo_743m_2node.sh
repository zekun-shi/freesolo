#!/bin/bash
# Train FreeSolo 743M from scratch, single epoch (60,000 steps x 524,288 tok/step = 31.46B tokens, 2x8 GPUs).
# Run on every node with the matching NODE_RANK. Env: DATA, RUNS, RUN, NNODES, NODE_RANK, MASTER_ADDR,
#      MASTER_PORT, GPUS_PER_NODE, RESUME, COMPILE=1; FS_FUSED_ATTN, FS_STRICT_CAUSAL select the kernel path.
#      The bounded cell pool (FS_PRUNE_*) is NOT enabled here: base FreeSolo trains without it; FS-L enables
#      it during CPT (scripts/cpt_*.sh).
set -u
RUN=${RUN:-freesolo_743m}
RESUME=${RESUME:-}   # set to a checkpoint dir to resume; empty = from scratch
PER_GPU=1; ACCUM=8        # 1 * 16 GPUs * accum 8 * 4096 = 524288 tok/step
SEQLEN=4096; VOCAB=32000; TOPK=4
COMPILE=${COMPILE:-0}; CF=""; [ "$COMPILE" = "1" ] && CF="--compile"   # torch.compile (~1.5x on the fused-Triton path)
MAXSTEPS=${MAXSTEPS:-60000}   # 524288 * 60000 = 31.46B tokens (single epoch)
NNODES=${NNODES:-2}; NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}; NPROC=$((NNODES * GPUS_PER_NODE))
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}; MASTER_PORT=${MASTER_PORT:-29531}
DATA=${DATA:-./data/fineweb_edu_llama32k_40bt.bin}
RUNS=${RUNS:-./runs}
export WANDB_MODE=${WANDB_MODE:-disabled}
export FS_FUSED_ATTN=1 FS_STRICT_CAUSAL=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}   # set 1 to disable InfiniBand and force TCP
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
C="--config 800m --attn_type fsattn --long_range_variant cell_topk --cell_select score --cell_topk $TOPK --cell_stride 32 \
--sel_strict_causal 1 --fs_nope_longrange --fs_uniform_sr --window_size 512 --max_seq_len $SEQLEN --vocab $VOCAB \
--per_gpu_batch_size $PER_GPU --grad_accum_steps $ACCUM --max_steps $MAXSTEPS --lr 4e-4 --min_lr 4e-5 --warmup_steps 2000 \
--weight_decay 0.1 --grad_clip 1.0 --save_every 5000 --eval_every 2000 $CF \
${RESUME:+--resume $RESUME} \
--pretokenized $DATA"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || { echo "cd failed"; exit 1; }
accelerate launch --num_machines "$NNODES" --num_processes "$NPROC" --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" --same_network --mixed_precision bf16 \
  experiments/scaling/train_dist.py $C --run_name "$RUN" --output_dir "$RUNS/$RUN"
