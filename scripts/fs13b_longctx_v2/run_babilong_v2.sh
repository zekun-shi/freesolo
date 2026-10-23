#!/bin/bash
# FS-B-1.3B BABILong 64k re-run controller (v2, 2026-10-23).
# Verbatim copy of the campaign controller
# /home/aiops/shizk/fs_longctx_evals_20261016/harness/run_babilong.sh
# (sha256 d30a61ed96af4364ebcf89240146048d0ddf98e391f120e2cc5a51bde9f4aeec)
# with one recorded delta: FSB_DECODE_WHOLE=0 is exported (same protozk-venv
# capture-stream workaround as run_helmet_v2.sh). The per-cell --code-root in
# the v2 cells file carries the tracked snapshot path.
# BABILong matrix controller (same claim-queue + GPU idle-gate as run_matrix.sh).
# Cell: venv|task_len|label|args ; driver harness/babilong_eval.py ; offline (data is local).
set -u
GPU=$1
CELLS=$2
BENCH=babilong
W=/home/aiops/shizk/fs_longctx_evals_20261016
cd "$W"

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

while IFS='|' read -r venv task label args; do
  [ -z "${venv:-}" ] && continue
  out="results/${BENCH}/${label}/${task}.jsonl"
  claim="${out}.claim"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && grep -q '"kind": "summary"' "$out" 2>/dev/null; then
    continue
  fi
  if ! mkdir "$claim" 2>/dev/null; then
    continue  # claimed by another live controller
  fi
  if [ -f "$out" ] && ! grep -q '"kind": "summary"' "$out" 2>/dev/null; then
    rm -f "$out"  # partial file from a dead run; regenerate deterministically
  fi
  case "$venv" in
    pa) PY=/home/aiops/shizk/physics-attn-venv/bin/python ;;
    nats) PY=/home/aiops/shizk/nats-venv/bin/python ;;
    *) echo "BAD_VENV $venv $label $task"; rmdir "$claim"; continue ;;
  esac
  wait_idle "$GPU"
  echo "RUN g$GPU $label $task $(date +%H:%M:%S)"
  FS_SOURCE_GIT=$FS_SOURCE_GIT \
  HF_HOME=/home/aiops/shizk/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
    "$PY" harness/babilong_eval.py $args --output "$out" >> "logs/${BENCH}_${label}_${task}.log" 2>&1
  rc=$?
  echo "EXIT $rc g$GPU $label $task $(date +%H:%M:%S)"
  if [ $rc -ne 0 ]; then
    rm -f "$out"
    echo "FAILED g$GPU $label $task rc=$rc" >> "results/${BENCH}/_failed.log"
  fi
  rmdir "$claim" 2>/dev/null || true
done < "$CELLS"
echo "CONTROLLER_DONE g$GPU" $(date +%F\ %T)
