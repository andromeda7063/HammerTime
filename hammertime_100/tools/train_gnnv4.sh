#!/usr/bin/env bash
# GNNv4 launcher (Linux). Local data only: reads an existing verified normalized
# source cache, builds/reuses a memory-mapped cohort_v4, trains and evaluates.
# Device: CUDA when available, otherwise CPU (also on CUDA probe failure/OOM).
#
# Usage:
#   hammertime_100/tools/train_gnnv4.sh [--preflight-only] [--device auto|cuda|cpu]
#                                       [--epochs N] [--resume DIR] [-- extra runner args]
# Environment overrides (skip the prompts):
#   HAMMERTIME_NORMALIZED_SOURCE  <old-run>/race_cache
#   HAMMERTIME_RAW_SESSIONS       raw_sessions dir (only for the yellow-flag review flag; optional)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
cd "$REPO_ROOT"

PREFLIGHT=0
DEVICE=auto
EPOCHS=''
RESUME_DIR=''
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight-only) PREFLIGHT=1; shift ;;
    --device) DEVICE="${2:?--device needs auto|cuda|cpu}"; shift 2 ;;
    --epochs) EPOCHS="${2:?--epochs needs a number}"; shift 2 ;;
    --resume) RESUME_DIR="${2:?--resume needs a run directory}"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$DEVICE" in auto|cuda|cpu) ;; *) printf 'Invalid --device %s\n' "$DEVICE" >&2; exit 2 ;; esac

command -v uv >/dev/null 2>&1 || { printf 'uv is required (https://docs.astral.sh/uv/).\n' >&2; exit 2; }

DEFAULT_SOURCE="$REPO_ROOT/../HammerTime/.local/research-history/2023-gnnv2-cpu/race_cache"
SOURCE="${HAMMERTIME_NORMALIZED_SOURCE:-}"
if [[ -z "$SOURCE" ]]; then
  read -r -e -i "$DEFAULT_SOURCE" -p 'Normalized source cache (<run>/race_cache): ' SOURCE
fi
SOURCE="$(realpath -e -- "$SOURCE")" || { printf 'Normalized source not found.\n' >&2; exit 2; }
for required in "$SOURCE/normalized_source_manifest.json" "$SOURCE/normalized_source_manifest.json.sha256" "$SOURCE/sessions"; do
  [[ -e "$required" ]] || { printf 'Required normalized-source input missing: %s\n' "$required" >&2; exit 2; }
done

RAW_ARGS=()
RAW="${HAMMERTIME_RAW_SESSIONS:-}"
if [[ -n "$RAW" ]]; then
  RAW="$(realpath -e -- "$RAW")"
  RAW_ARGS=(--raw-sessions-dir "$RAW")
fi

# Device probe (informational; the runner repeats it and falls back to CPU itself).
PROBE="$(uv run --frozen python -c 'import torch
ok = torch.cuda.is_available()
if ok:
    free, total = torch.cuda.mem_get_info()
    print("cuda", torch.cuda.get_device_name(0).replace(" ", "_"), round(free / 2**30, 1))
else:
    print("cpu", "-", 0)' 2>/dev/null || echo 'cpu - 0')"
read -r PROBE_DEVICE PROBE_NAME PROBE_FREE <<<"$PROBE"
if [[ "$DEVICE" == cuda && "$PROBE_DEVICE" != cuda ]]; then
  printf 'WARNING: CUDA requested but unavailable; falling back to CPU.\n' >&2
fi
EFFECTIVE="$DEVICE"
[[ "$DEVICE" == auto ]] && EFFECTIVE="$PROBE_DEVICE"
[[ "$PROBE_DEVICE" != cuda ]] && EFFECTIVE=cpu

CORES="$(nproc 2>/dev/null || echo 4)"
THREADS=$(( CORES > 3 ? CORES - 2 : 1 ))
if [[ "$EFFECTIVE" == cuda ]]; then
  BATCH=256; ACCUM=1; WORKERS=2
  printf 'Device: CUDA %s (%s GiB free) | batch 256 x accum 1 | workers 2\n' "$PROBE_NAME" "$PROBE_FREE"
else
  BATCH=128; ACCUM=2; WORKERS=0
  printf 'Device: CPU | batch 128 x accum 2 | torch threads %d\n' "$THREADS"
fi

if [[ -n "$RESUME_DIR" ]]; then
  OUT="$(realpath -e -- "$RESUME_DIR")"
  RUN_ARGS=(--resume)
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  SUFFIX=''; [[ "$PREFLIGHT" == 1 ]] && SUFFIX='-preflight'
  OUT="$REPO_ROOT/.local/research-history/gnnv4$SUFFIX-$RUN_ID"
  [[ -e "$OUT" ]] && { printf 'Refusing to overwrite %s\n' "$OUT" >&2; exit 2; }
  RUN_ARGS=()
fi
[[ "$PREFLIGHT" == 1 ]] && RUN_ARGS+=(--preflight-only)
[[ -n "$EPOCHS" ]] && RUN_ARGS+=(--epochs "$EPOCHS")
mkdir -p "$REPO_ROOT/.local/research-history"
LOG="$OUT.launcher.log"

export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" PYTHONUTF8=1

set +e
uv run --frozen python hammertime_100/train_gnnv4.py \
  --normalized-source "$SOURCE" "${RAW_ARGS[@]}" \
  --config hammertime_100/configs/train_2023_fastf1_season_gnnv4.yaml \
  --output "$OUT" \
  --device "$EFFECTIVE" \
  --batch-size "$BATCH" --grad-accum "$ACCUM" \
  --workers "$WORKERS" --threads "$THREADS" \
  --log-every-batches 50 \
  "${RUN_ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
case "$STATUS" in
  0) printf '\nDone. Outputs: %s (launcher log %s)\n' "$OUT" "$LOG" ;;
  3) printf '\nStopped by the memory preflight; see the message above.\n' >&2 ;;
  *) printf '\nRunner failed with exit code %s; see %s\n' "$STATUS" "$LOG" >&2 ;;
esac
exit "$STATUS"
