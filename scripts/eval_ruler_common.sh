#!/usr/bin/env bash
# Shared finite RULER grid. Each GPU runs one job at a time.
set -u
VARIANT=${1:?expected FS, FS-L or FS-B}
shift
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
REPO=$(pushd "$SCRIPT_DIR/.." >/dev/null && pwd && popd >/dev/null)
CKPT=${CKPT:?set CKPT to the checkpoint directory}
CONFIG=${CONFIG:-340m}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
LIMIT=${LIMIT:-100}
RULER_DATA=${RULER_DATA:-./data/ruler}
TAG=${TAG:-${VARIANT,,}}
OUT=${OUT:-./results/ruler_${TAG}}
TASKS=${TASKS:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue vt cwe fwe qa_1 qa_2"}
LENS=${LENS:-"16384 8192 4096 2048"}
[[ -d "$RULER_DATA" ]] || { echo "Missing RULER_DATA: $RULER_DATA" >&2; exit 2; }
mkdir -p "$OUT"
ARCH=(--config "$CONFIG" --use_rope --fs_uniform_sr --fs_nope_longrange --window_size 512)
export FS_STRICT_CAUSAL=1 TOKENIZERS_PARALLELISM=false FSL_LINEAR_CACHE=0
case "$VARIANT" in
  FS)
    export FS_PRUNE_KERNEL=0
    ;;
  FS-L)
    export FS_PRUNE_KERNEL=1 FS_PRUNE_POOL=64 FS_PRUNE_WINDOW=8 FS_PRUNE_BLKCHUNK=16
    ;;
  FS-B)
    export FS_PRUNE_KERNEL=0
    ARCH+=(--fsb --fsb_policy fixed --fsb_width 16 --fsb_delta 0
      --fsb_node_score bound --fsb_bound_radius_coef 8 --fsb_tau_override 0
      --fsb_tree forest --fsb_route_adapter none --fsb_route_alpha 0
      --fsb_read_top 4 --fsb_rerank_score lse_pos --fsb_rerank_query raw
      --fsb_union_width 16 --fsb_union_tau_quantile 0.5)
    if [[ -n ${FSB_TAU_FILE:-} ]]; then
      ARCH+=(--fsb_union_tau_file "$FSB_TAU_FILE")
    else
      : "${FSB_CALIB_DATA:?set FSB_TAU_FILE to the released thresholds, or FSB_CALIB_DATA to the calibration corpus}"
      ARCH+=(--fsb_tau_calib_data "$FSB_CALIB_DATA" --fsb_tau_cache_dir "${FSB_TAU_CACHE_DIR:-$OUT/tau_cache}")
    fi
    ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac
ARCH+=("$@")
read -r -a GARR <<< "$GPUS"
NG=${#GARR[@]}
[[ $NG -gt 0 ]] || { echo "GPUS is empty" >&2; exit 2; }
JOBS=()
for L in $LENS; do
  for T in $TASKS; do
    if [[ -s "$RULER_DATA/L$L/$T/validation.jsonl" ]]; then
      JOBS+=("$T:$L")
    else
      echo "SKIP missing data: $T L$L"
    fi
  done
done
[[ ${#JOBS[@]} -gt 0 ]] || { echo "No RULER input files found" >&2; exit 2; }

run_worker() {
  local wi=$1 failed=0 i T L O DD rc
  local CHECK
  for ((i=wi; i<${#JOBS[@]}; i+=NG)); do
    T=${JOBS[$i]%:*}; L=${JOBS[$i]#*:}
    O="$OUT/officialruler_${TAG}_${T}_L${L}.json"
    DD="$RULER_DATA/L$L"
    CHECK=(python "$REPO/experiments/scaling/eval_report.py" "$O" --task "$T"
      --length "$L" --checkpoint "$CKPT" --variant "$VARIANT"
      --data "$DD/$T/validation.jsonl" --limit "$LIMIT")
    if "${CHECK[@]}"; then
      echo "HAVE complete: $T L$L"
      continue
    fi
    CUDA_VISIBLE_DEVICES=${GARR[$wi]} python "$REPO/eval_ruler.py" --ckpt "$CKPT" \
      --data_dir "$DD" --tasks "$T" --length "$L" --output "$O" --limit "$LIMIT" \
      "${ARCH[@]}" > "$OUT/${TAG}_${T}_L${L}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]] || ! "${CHECK[@]}"; then
      echo "FAILED $T L$L rc=$rc; see $OUT/${TAG}_${T}_L${L}.log" >&2
      failed=1
    else
      echo "DONE $T L$L gpu=${GARR[$wi]}"
    fi
  done
  return "$failed"
}

echo "${TAG}_FULLRULER_START jobs=${#JOBS[@]} gpus=[$GPUS] limit=$LIMIT out=$OUT"
PIDS=()
for ((wi=0; wi<NG; wi++)); do
  run_worker "$wi" &
  PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ $failed -ne 0 ]]; then
  echo "${TAG}_FULLRULER_FAILED" >&2
  exit 1
fi
echo "${TAG}_FULLRULER_DONE"
