#!/bin/bash
# FS-B-1.3B HELMET 64k re-run controller (v2, 2026-10-23).
# Verbatim copy of the campaign controller
# /home/aiops/shizk/fs_longctx_evals_20261016/harness/run_helmet.sh
# (sha256 938fc7c78b2f1373a0fc8aecaf7ad10baeeb1cb6156fefa05600e1f5ce492e61)
# with two recorded deltas:
#  1. FS_EVAL_CODE_ROOT is taken from the environment instead of the pinned
#     campaign snapshot (v2 runs this tracked freesolo tree, see README.md).
#  2. FSB_DECODE_WHOLE=0 is exported: the whole-step CUDA-graph wrapper fails
#     "capturing stream has unjoined work" in the pod venv (torch 2.5.1+cu121);
#     per-step graphs replay the same kernels and values.
# HELMET matrix controller (same claim-queue + GPU idle-gate as run_matrix.sh).
# Cell: venv|famL|label|modelspec
set -u
GPU=$1
CELLS=$2
W=/home/aiops/shizk/fs_longctx_evals_20261016
H=$W/helmet/HELMET

FS_EVAL_CODE_ROOT=${FS_EVAL_CODE_ROOT:-/home/aiops/shizk/fs13b_kernel_probe_20261023/freesolo-9bb60b1}
FS_SOURCE_GIT=${FS_SOURCE_GIT:-9bb60b1}
export FSB_DECODE_WHOLE=${FSB_DECODE_WHOLE:-0}

wait_idle() {
  local g=$1
  while :; do
    local used procs
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" | grep -c .)
    if [ "$used" -lt 100 ] && [ "$procs" -eq 0 ]; then return 0; fi
    sleep 30
  done
}

while IFS='|' read -r venv famL label spec; do
  [ -z "${venv:-}" ] && continue
  outdir="$W/results/helmet/${label}/${famL}"
  marker="$outdir/.DONE"
  [ -f "$marker" ] && continue
  claim="$outdir/.claim"
  mkdir -p "$outdir"
  if ! mkdir "$claim" 2>/dev/null; then continue; fi
  case "$venv" in
    pa) PY=/home/aiops/shizk/physics-attn-venv/bin/python; CITE=$W/wheels/cite_py311_pkg ;;
    nats) PY=/home/aiops/shizk/nats-venv/bin/python; CITE=$W/wheels/cite_py312_pkg ;;
    *) echo "BAD_VENV $venv $label $famL"; rmdir "$claim"; continue ;;
  esac
  wait_idle "$GPU"
  echo "RUN g$GPU $label $famL $(date +%H:%M:%S)"
  FS_EVAL_CODE_ROOT=$FS_EVAL_CODE_ROOT FS_SOURCE_GIT=$FS_SOURCE_GIT HELMET_ROOT=$H \
  HF_HOME=/home/aiops/shizk/hf_cache CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
  PYTHONPATH="$CITE" NLTK_DATA=$W/wheels/nltk_data \
    "$PY" "$W/harness/helmet_run.py" \
      --config "configs_fs/${famL}.yaml" --model_name_or_path "$spec" \
      --output_dir "$outdir" --tag "$famL" \
      >> "$W/logs/helmet_${label}_${famL}.log" 2>&1
  rc=$?
  # completion = one fresh output json per dataset in the yaml
  exp=$(grep '^datasets:' "$H/configs_fs/${famL}.yaml" | cut -d: -f2- | tr ',' '\n' | grep -c .)
  got=$(ls "$outdir"/${famL#*}_*in*.json "$outdir"/*.json 2>/dev/null | grep -v '\.score$' | sort -u | wc -l)
  if [ $rc -eq 0 ] && [ "$got" -ge "$exp" ]; then
    touch "$marker"; echo "DONE g$GPU $label $famL got=$got exp=$exp $(date +%H:%M:%S)"
  else
    echo "EXIT rc=$rc g$GPU $label $famL got=$got exp=$exp $(date +%H:%M:%S)"
    echo "INCOMPLETE g$GPU $label $famL rc=$rc got=$got exp=$exp" >> "$W/results/helmet/_failed.log"
  fi
  rmdir "$claim" 2>/dev/null || true
done < "$CELLS"
echo "CONTROLLER_DONE g$GPU" $(date +%F\ %T)
