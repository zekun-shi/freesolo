#!/usr/bin/env bash
# Generate the paper's RULER inputs with the official NVIDIA generator.
set -u
RULER_REPO=${RULER_REPO:-./RULER}
OUT=${RULER_DATA:-./data/ruler}
TOK=${TOKENIZER:-fla-hub/transformer-1.3B-100B}
PREPDIR=$RULER_REPO/scripts/data
TASKS=${TASKS:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue vt cwe fwe qa_1 qa_2"}
LENGTHS=${LENGTHS:-"2048 4096 8192 16384"}
[[ -f "$PREPDIR/prepare.py" ]] || { echo "Missing RULER generator: $PREPDIR/prepare.py" >&2; exit 2; }
mkdir -p "$OUT" || exit 1
OUT=$(pushd "$OUT" >/dev/null && pwd && popd >/dev/null)

complete_data() {
  python - "$1" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    valid = len(rows) == 100 and all(isinstance(row, dict) and isinstance(row.get('input'), str)
                                    and isinstance(row.get('outputs'), list) for row in rows)
except (OSError, ValueError, TypeError):
    valid = False
sys.exit(0 if valid else 1)
PY
}

gen() {
  local task=$1 length=$2 dir="$OUT/L$2" rc
  mkdir -p "$dir" || return 1
  if complete_data "$dir/$task/validation.jsonl"; then
    echo "HAVE complete: $task L$length"
    return 0
  fi
  # prepare.py uses relative imports and launches generators through this environment's python.
  (
    pushd "$PREPDIR" >/dev/null || exit 1
    python prepare.py --save_dir "$dir" --benchmark synthetic --task "$task" --subset validation \
      --tokenizer_path "$TOK" --tokenizer_type hf --max_seq_length "$length" \
      --num_samples 100 --random_seed 42 --model_template_type base
  ) > "$OUT/genlog_${length}_${task}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]] || ! complete_data "$dir/$task/validation.jsonl"; then
    echo "FAILED $task L$length rc=$rc; see $OUT/genlog_${length}_${task}.log" >&2
    return 1
  fi
  echo "DONE $task L$length: 100 records"
}

failed=0
for length in $LENGTHS; do
  for task in $TASKS; do
    if ! gen "$task" "$length"; then failed=1; fi
  done
done
[[ $failed -eq 0 ]] || { echo GEN_FAILED >&2; exit 1; }
echo GEN_ALL_DONE
