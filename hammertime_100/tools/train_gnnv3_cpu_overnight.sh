#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
cd "$REPO_ROOT"

case "${1:-}" in
  '') PREFLIGHT_ONLY=0 ;;
  --preflight-only) PREFLIGHT_ONLY=1 ;;
  *) printf 'Usage: %s [--preflight-only]\n' "$0" >&2; exit 2 ;;
esac

DEFAULT_OLD_RUN='/home/andromeda/Documents/code/projects/HammerTime/.local/research-history/2023-gnnv2-cpu'
DEFAULT_RAW_SESSIONS='/home/andromeda/Documents/code/projects/HammerTime/.local/research-history/2023-fastf1-season/race_cache/raw_sessions'
DEFAULT_PROD_BEST='/home/andromeda/Documents/code/projects/HammerTime/.local/research-history/2023-gnnv2-cpu/training/5408dd82d53ec69fbb65/best.pt'
HAMMERTIME_OLD_RUN="${HAMMERTIME_OLD_RUN:-$DEFAULT_OLD_RUN}"
read -r -e -i "$HAMMERTIME_OLD_RUN" -p 'Existing GNNv2 run root (normalized sessions and cohort): ' HAMMERTIME_OLD_RUN
HAMMERTIME_OLD_RUN="$(realpath -e -- "$HAMMERTIME_OLD_RUN")"
for expected in \
  "$HAMMERTIME_OLD_RUN/race_cache/sessions" \
  "$HAMMERTIME_OLD_RUN/cohorts"; do
  if [[ ! -d "$expected" ]]; then
    printf 'Required existing cache directory is missing: %s\n' "$expected" >&2
    exit 2
  fi
done

HAMMERTIME_RAW_SESSIONS="${HAMMERTIME_RAW_SESSIONS:-$DEFAULT_RAW_SESSIONS}"
read -r -e -i "$HAMMERTIME_RAW_SESSIONS" -p 'Authentic raw_sessions directory referenced by normalized stage manifests: ' HAMMERTIME_RAW_SESSIONS
HAMMERTIME_RAW_SESSIONS="$(realpath -e -- "$HAMMERTIME_RAW_SESSIONS")"
if [[ ! -d "$HAMMERTIME_RAW_SESSIONS" ]]; then
  printf 'Raw-session source is not a directory: %s\n' "$HAMMERTIME_RAW_SESSIONS" >&2
  exit 2
fi

printf '\nRead-only production checkpoint candidates (confirm best epoch 18 and R² near 0.09):\n'
uv run --frozen python hammertime_100/tools/inspect_production_checkpoints.py "$HAMMERTIME_OLD_RUN"
HAMMERTIME_PROD_BEST="${HAMMERTIME_PROD_BEST:-$DEFAULT_PROD_BEST}"
read -r -e -i "$HAMMERTIME_PROD_BEST" -p 'Production best.pt to preserve: ' HAMMERTIME_PROD_BEST
HAMMERTIME_PROD_BEST="$(realpath -e -- "$HAMMERTIME_PROD_BEST")"
if [[ ! -f "$HAMMERTIME_PROD_BEST" || "$(basename -- "$HAMMERTIME_PROD_BEST")" != 'best.pt' ]]; then
  printf 'The production checkpoint must be an existing best.pt file.\n' >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_SUFFIX=''
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then RUN_SUFFIX='-preflight'; fi
HAMMERTIME_NEW_RUN="$REPO_ROOT/.local/research-history/gnnv3-cpu-overnight$RUN_SUFFIX-$RUN_ID"
HAMMERTIME_RUN_LOG="$REPO_ROOT/.local/research-history/gnnv3-cpu-overnight$RUN_SUFFIX-$RUN_ID.log"
if [[ -e "$HAMMERTIME_NEW_RUN" || -e "$HAMMERTIME_RUN_LOG" ]]; then
  printf 'Refusing to overwrite an existing run output or log.\n' >&2
  exit 2
fi
mkdir -p "$REPO_ROOT/.local/research-history"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 PYTHONUTF8=1

PREFLIGHT_ARGS=()
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then PREFLIGHT_ARGS+=(--preflight-only); fi

uv run --frozen python hammertime_100/run_2023_fastf1_season.py \
  --config hammertime_100/configs/train_2023_fastf1_season_gnnv2.yaml \
  --output "$HAMMERTIME_NEW_RUN" \
  --reuse-prepared-from "$HAMMERTIME_OLD_RUN" \
  --preserve-production-checkpoint "$HAMMERTIME_PROD_BEST" \
  --device cpu \
  --epochs 30 \
  --window-stride 2 \
  --workers 1 \
  --processing-workers 1 \
  --torch-threads 12 \
  --local-data-only \
  --raw-sessions-dir "$HAMMERTIME_RAW_SESSIONS" \
  --require-existing-caches \
  --training-overrides hammertime_100/configs/gnnv3_cpu_overnight_overrides.yaml \
  --log-every-batches 10 "${PREFLIGHT_ARGS[@]}" 2>&1 | tee "$HAMMERTIME_RUN_LOG"

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  printf '\nCache/model preflight passed with no training or preprocessing.\n'
  printf 'Verified production checkpoint backup: %s/production_model_backup\n' "$HAMMERTIME_NEW_RUN"
  printf 'Preflight output: %s\nLog: %s\n' "$HAMMERTIME_NEW_RUN" "$HAMMERTIME_RUN_LOG"
  exit 0
fi

printf '\nRun completed successfully. The production original remains unchanged; the runner verified its SHA256 backup before fitting.\n'
printf 'Production checkpoint backup: %s/production_model_backup\n' "$HAMMERTIME_NEW_RUN"
printf 'Run output: %s\nLog: %s\n' "$HAMMERTIME_NEW_RUN" "$HAMMERTIME_RUN_LOG"
