#!/usr/bin/env bash
# Zero-shot commonsense evaluation, one task per GPU. Use FS_FUSED_ATTN=0 for the reference path.
set -u
CKPT=${CKPT:?set CKPT to the model checkpoint to evaluate}
CONFIG=${CONFIG:-340m}
MODE=${MODE:-fs}
ARCH=(--long_range_variant cell_topk --cell_select score --cell_topk 4 --cell_stride 32 --sel_strict_causal 1)
case "$MODE" in
  fs) ARCH+=(--fs_nope_longrange --fs_uniform_sr --window_size 512) ;;
  nope) ARCH+=(--no_rope --window_size 128) ;;
  *) echo "MODE must be fs or nope" >&2; exit 2 ;;
esac
OUT=${OUT:-./results/cs_${MODE}}
mkdir -p "$OUT" || exit 1
export FS_FUSED_ATTN=${FS_FUSED_ATTN:-1} FS_STRICT_CAUSAL=1 TOKENIZERS_PARALLELISM=false
REPO=$(pushd "$(dirname "$0")/.." >/dev/null && pwd && popd >/dev/null)
TASKS=(hellaswag arc_easy arc_challenge piqa winogrande openbookqa sciq boolq)

complete_result() {
  python - "$1" "$2" "$CKPT" "$CONFIG" "$MODE" <<'PY'
import json, math, pathlib, sys
try:
    with open(sys.argv[1]) as stream:
        report = json.load(stream)
    config, result = report['config'], report['tasks'][sys.argv[2]]
    valid = (pathlib.Path(config['checkpoint']).resolve() == pathlib.Path(sys.argv[3]).resolve()
             and config['config'] == sys.argv[4] and config['limit'] == 0
             and config['num_fewshot'] == 0 and config['max_seq_len'] == 2048
             and config['no_rope'] == (sys.argv[5] == 'nope')
             and isinstance(result['n'], int) and result['n'] > 0
             and isinstance(result['score'], (float, int)) and math.isfinite(result['score'])
             and 0 <= result['score'] <= 1)
except (OSError, ValueError, TypeError, KeyError):
    valid = False
sys.exit(0 if valid else 1)
PY
}

echo "CS_${MODE}_START ckpt=$CKPT out=$OUT"
PIDS=()
STARTED=()
for gpu in "${!TASKS[@]}"; do
  task=${TASKS[$gpu]}
  output="$OUT/zs_${task}.json"
  if complete_result "$output" "$task"; then
    echo "HAVE complete: $task"
    continue
  fi
  CUDA_VISIBLE_DEVICES=$gpu python "$REPO/eval_downstream.py" --checkpoint "$CKPT" \
    --config "$CONFIG" --attn_type fsattn "${ARCH[@]}" --tasks "$task" \
    --num_fewshot 0 --max_seq_len 2048 --output "$output" > "$OUT/zs_${task}.log" 2>&1 &
  PIDS+=("$!")
  STARTED+=("$task")
done
failed=0
for i in "${!PIDS[@]}"; do
  task=${STARTED[$i]}
  if ! wait "${PIDS[$i]}" || ! complete_result "$OUT/zs_${task}.json" "$task"; then
    echo "FAILED $task; see $OUT/zs_${task}.log" >&2
    failed=1
  fi
done
[[ $failed -eq 0 ]] || { echo "CS_${MODE}_FAILED" >&2; exit 1; }
echo "CS_${MODE}_DONE"
