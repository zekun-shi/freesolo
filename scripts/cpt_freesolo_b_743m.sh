#!/bin/bash
# Continued pretraining producing FreeSolo-B 743M: two-stage far-recall weighted CPT of a FreeSolo 743M
# checkpoint. Each token's loss is weighted by w_t = 1 + beta * f_t + beta_sw * chi_t, where f_t (far
# continuation / far recall start) and chi_t (source switch) are exact n-gram statistics of the micro-batch's
# own tokens (k=4, m=4, dmin=512, gap=64, EOS id 2, single document), computed on the fly. Stage 1 runs
# beta / beta_sw = BETA / BETA_SW1 from INIT_FROM; stage 2 switches the extra source-switch weight to BETA_SW2
# and warm-starts (model weights only: fresh optimizer and warmup+cosine LR) from stage 1's step-MAXSTEPS
# checkpoint. Plain FS top-K selection (cell_topk $TOPK, no pruned pool). Single node, 8 GPUs,
# 1 * 8 * ACCUM * 4096 = 262144 tok/step.
# Env: INIT_FROM (base checkpoint dir, required), DATA (corpus .bin, required; its byte size is checked against
#      the 40bt fineweb-edu corpus when the file exists); RUN, RUNS, TOPK, LR, MINLR, WARMUP, MAXSTEPS, ACCUM,
#      SEED1, SEED2, BETA, BETA_SW1, BETA_SW2, SAVE_EVERY, PORT.
set -u
RUN=${RUN:-cpt_freesolo_b_743m}
INIT_FROM=${INIT_FROM:?set INIT_FROM to the base FreeSolo 743M checkpoint dir (stage 1 weights-only warm start)}
DATA=${DATA:?set DATA to the pretokenized corpus .bin}
TOPK=${TOPK:-16}; PER_GPU=1; ACCUM=${ACCUM:-8}
SEQLEN=4096; VOCAB=32000
LR=${LR:-4e-5}; MINLR=${MINLR:-4e-6}; WARMUP=${WARMUP:-20}
MAXSTEPS=${MAXSTEPS:-800}                              # per stage; warmup+cosine LR -> MINLR, fresh each stage
BETA=${BETA:-9}; BETA_SW1=${BETA_SW1:-9}; BETA_SW2=${BETA_SW2:-49}
SEED1=${SEED1:-42}; SEED2=${SEED2:-43}
SAVE_EVERY=${SAVE_EVERY:-200}                          # must divide MAXSTEPS: stage 2 loads ${RUN}_s1_step$MAXSTEPS
NPROC=8; PORT=${PORT:-29621}
RUNS=${RUNS:-./runs}
if [ -e "$DATA" ]; then
  [ "$(stat -c %s "$DATA")" = 77960195380 ] || { echo "refusing: $DATA is not the 40bt fineweb-edu corpus (size)"; exit 2; }
fi
[ -s "$INIT_FROM/model.safetensors" ] || { echo "refusing: no $INIT_FROM/model.safetensors"; exit 2; }
export WANDB_MODE=${WANDB_MODE:-disabled}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export FS_FUSED_ATTN=1 FS_STRICT_CAUSAL=1
export FS_PRUNE_KERNEL=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
C="--config 800m --attn_type fsattn --long_range_variant cell_topk --cell_select score --cell_topk $TOPK --cell_stride 32 \
--sel_strict_causal 1 --fs_nope_longrange --fs_uniform_sr --window_size 512 --max_seq_len $SEQLEN --vocab $VOCAB \
--per_gpu_batch_size $PER_GPU --grad_accum_steps $ACCUM --max_steps $MAXSTEPS --lr $LR --min_lr $MINLR --warmup_steps $WARMUP \
--weight_decay 0.1 --grad_clip 1.0 --save_every $SAVE_EVERY --eval_every 999999 \
--pretokenized $DATA"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || { echo "cd failed"; exit 1; }
set -e
accelerate launch --num_machines 1 --num_processes "$NPROC" --main_process_port "$PORT" --mixed_precision bf16 \
  experiments/scaling/train_dist.py $C --far_recall_beta "$BETA" --far_recall_switch_beta "$BETA_SW1" --seed "$SEED1" \
  --init_from "$INIT_FROM" --run_name "${RUN}_s1" --output_dir "$RUNS/${RUN}_s1"
S1="$RUNS/${RUN}_s1/${RUN}_s1_step${MAXSTEPS}"
[ -s "$S1/model.safetensors" ] || { echo "refusing: stage 1 left no $S1 (SAVE_EVERY must divide MAXSTEPS)"; exit 1; }
accelerate launch --num_machines 1 --num_processes "$NPROC" --main_process_port "$PORT" --mixed_precision bf16 \
  experiments/scaling/train_dist.py $C --far_recall_beta "$BETA" --far_recall_switch_beta "$BETA_SW2" --seed "$SEED2" \
  --init_from "$S1" --run_name "${RUN}_s2" --output_dir "$RUNS/${RUN}_s2"
echo "done: ${RUN}_s1 (beta_sw $BETA_SW1, seed $SEED1) -> ${RUN}_s2 (beta_sw $BETA_SW2, seed $SEED2)"
