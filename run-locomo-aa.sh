#!/usr/bin/env bash
set -euo pipefail

readonly REPO="/Volumes/LEXAR/hermes-work/wt-locomo-prep"
readonly TOOLS="/Volumes/LEXAR/hermes-work/wt-ci-fix/bench/tools"
readonly PINS="$REPO/data/pins-locomo.yaml"
readonly SOURCE_DATASET="$REPO/data/locomo10.json"
readonly LOG_ROOT="/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-r3-1/artifacts/laneLOCOMO-logs"

if [[ "${1:-}" != "--execute-paid-aa" || "${LOCOMO_PAID_RUN_AUTHORIZED:-0}" != "1" ]]; then
  echo "Paid A/A' recipe only. Requires --execute-paid-aa and LOCOMO_PAID_RUN_AUTHORIZED=1." >&2
  exit 64
fi

export HERMES_LCM_REPO="/Volumes/LEXAR/hermes-work/hermes-lcm"
export HERMES_MB_PROVIDER="fastembed"
export HERMES_MB_LLM_CLI="codex"
export HERMES_MB_CODEX_MODEL="gpt-5.6-sol"
export HERMES_MB_CODEX_ANSWER_EFFORT="medium"
export HERMES_MB_CODEX_JUDGE_EFFORT="low"
export HERMES_MB_ANSWER_PRESENTATION="evidence_cards_v1"
export HERMES_MB_ANSWER_READY_CONTENT_CHARS="2400"

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
readonly run_stamp
readonly RUN_ROOT="${LOCOMO_AA_RUN_ROOT:-$LOG_ROOT/paid-aa-$run_stamp}"
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing existing run root: $RUN_ROOT" >&2
  exit 73
fi
mkdir -p "$RUN_ROOT"

run_arm() {
  local arm="$1"
  local run_id="locomo-aa-${run_stamp}-${arm}"
  local arm_root="$RUN_ROOT/$arm"
  local dataset_dir="$arm_root/data/benchmarks/locomo"
  local store_dir="$arm_root/stores"
  local report_dir="$arm_root/data/runs/$run_id"

  mkdir -p "$dataset_dir" "$store_dir"
  cp "$SOURCE_DATASET" "$dataset_dir/locomo10.json"

  (
    cd "$arm_root"
    export HERMES_MB_WORKDIR="$store_dir"
    python3 "$TOOLS/pinverify.py" pre-run "$PINS" -o "$arm_root/PINS-PRERUN.txt" \
      2>&1 | tee "$arm_root/pinverify-pre.log"
    bun run "$REPO/src/index.ts" run -p hermes-lcm -b locomo -r "$run_id" \
      -j gpt-5.6-sol -m gpt-5.6-sol --force 2>&1 | tee "$arm_root/full-run.log"
    python3 "$TOOLS/storefreeze.py" freeze "$store_dir" \
      -o "$arm_root/store.manifest.json" 2>&1 | tee "$arm_root/storefreeze-store.log"
    python3 "$TOOLS/storefreeze.py" verify "$store_dir" "$arm_root/store.manifest.json" \
      2>&1 | tee "$arm_root/storeverify-store.log"
    python3 "$TOOLS/storefreeze.py" freeze "$dataset_dir" \
      -o "$arm_root/dataset.manifest.json" 2>&1 | tee "$arm_root/storefreeze-dataset.log"
    python3 "$TOOLS/storefreeze.py" verify "$dataset_dir" "$arm_root/dataset.manifest.json" \
      2>&1 | tee "$arm_root/storeverify-dataset.log"
    python3 "$TOOLS/failclose.py" "$report_dir" 2>&1 | tee "$arm_root/failclose.json"
    python3 "$TOOLS/pinverify.py" post-run "$PINS" -o "$arm_root/PINS-POSTRUN.txt" \
      2>&1 | tee "$arm_root/pinverify-post.log"
  )
}

run_arm "a"
run_arm "a-prime"

python3 "$TOOLS/failclose.py" \
  "$RUN_ROOT/a/data/runs/locomo-aa-${run_stamp}-a" \
  --compare "$RUN_ROOT/a-prime/data/runs/locomo-aa-${run_stamp}-a-prime" \
  2>&1 | tee "$RUN_ROOT/aa-paired-failclose.json"
