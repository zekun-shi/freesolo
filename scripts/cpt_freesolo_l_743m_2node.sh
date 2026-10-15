#!/bin/bash
# Continued pretraining producing FreeSolo-L 743M: weights-only warm start from a FreeSolo 743M checkpoint,
# trained with the bounded cell pool (FS_PRUNE_*) enabled; LR re-warmed to 4e-4, cosine decay to 4e-5.
# Env: INIT_FROM (required), DATA, RUNS, RUN, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT.
set -u
RUN=${RUN:-freesolo_l_743m_cpt}
INIT_FROM=${INIT_FROM:?set INIT_FROM to the dense-trained FreeSolo 743M checkpoint (weights-only warm start)}
PER_GPU=1; ACCUM=8        # 1 * 16 GPUs * accum 8 * 4096 = 524288 tok/step
SEQLEN=4096; VOCAB=32000; TOPK=4
COMPILE=${COMPILE:-0}; CF=""; [ "$COMPILE" = "1" ] && CF="--compile"
MAXSTEPS=${MAXSTEPS:-3000}                    # cosine 4e-4 -> 4e-5 over 3000 steps (~1.57B tokens)
LR=${LR:-4e-4}; MINLR=${MINLR:-4e-5}; WARMUP=${WARMUP:-300}
SAVE_EVERY=${SAVE_EVERY:-50}                  # frequent checkpoints for the retrieval-vs-step curve
EVAL_EVERY=${EVAL_EVERY:-999999}              # in-loop val eval off; checkpoints are evaluated offline
NNODES=${NNODES:-2}; NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}; NPROC=$((NNODES * GPUS_PER_NODE))
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}; MASTER_PORT=${MASTER_PORT:-29534}
DATA=${DATA:-./data/fineweb_edu_llama32k_40bt.bin}   # same corpus the dense model was trained on
RUNS=${RUNS:-./runs}
export WANDB_MODE=${WANDB_MODE:-disabled}
export FS_FUSED_ATTN=1 FS_STRICT_CAUSAL=1
export FS_PRUNE_KERNEL=1 FS_PRUNE_POOL=64 FS_PRUNE_WINDOW=8 FS_PRUNE_BLKCHUNK=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}   # set 1 to disable InfiniBand and force TCP
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
C="--config 800m --attn_type fsattn --long_range_variant cell_topk --cell_select score --cell_topk $TOPK --cell_stride 32 \
--sel_strict_causal 1 --fs_nope_longrange --fs_uniform_sr --window_size 512 --max_seq_len $SEQLEN --vocab $VOCAB \
--per_gpu_batch_size $PER_GPU --grad_accum_steps $ACCUM --max_steps $MAXSTEPS --lr $LR --min_lr $MINLR --warmup_steps $WARMUP \
--weight_decay 0.1 --grad_clip 1.0 --save_every $SAVE_EVERY --eval_every $EVAL_EVERY $CF \
--init_from $INIT_FROM \
--pretokenized $DATA"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || { echo "cd failed"; exit 1; }
accelerate launch --num_machines "$NNODES" --num_processes "$NPROC" --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" --same_network --mixed_precision bf16 \
  experiments/scaling/train_dist.py $C --run_name "$RUN" --output_dir "$RUNS/$RUN"
