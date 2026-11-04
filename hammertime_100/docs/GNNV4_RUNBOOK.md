# GNNv4 runbook (Linux and Windows)

GNNv4 trains only on authentic local data. It reads an existing, hash-verified normalized source cache (`<old-run>/race_cache`). On first use it builds a memory-mapped `cohort_v4` under `.local/research-history/gnnv4-cohorts/<fingerprint>/`, then trains and evaluates the race-held-out split. It never requests FastF1/OpenF1 data, and it never writes into the old run, its caches or its checkpoints.

Device selection: `--device auto` (default) uses CUDA when `torch.cuda.is_available()` and otherwise CPU. If you request `cuda` and it is unavailable, the launcher falls back to CPU with a warning. A CUDA out-of-memory error during training continues on CPU from the last complete epoch.

| Profile | Batch × accumulation | Workers | Threads |
|---|---|---|---|
| CUDA | 256 × 1 | 2 (Linux) / 0 (Windows) | cores − 2 |
| CPU | 128 × 2 | 0 | cores − 2 |

## 1. Preflight (recommended, ~1 minute)

This verifies every normalized-source hash, the memory budget and model construction. It does not build anything or train.

Linux:

```bash
HAMMERTIME_NORMALIZED_SOURCE=../HammerTime/.local/research-history/2023-gnnv2-cpu/race_cache hammertime_100/tools/train_gnnv4.sh --preflight-only
```

Windows (PowerShell, from the repository root):

```powershell
powershell -ExecutionPolicy Bypass -File hammertime_100\tools\train_gnnv4.ps1 -NormalizedSource <old-run>\race_cache -PreflightOnly
```

The normalized source manifest records the absolute path of the config that produced it. Use a cache produced on the same machine (or one whose recorded config path exists there).

## 2. Full run

Linux:

```bash
HAMMERTIME_NORMALIZED_SOURCE=../HammerTime/.local/research-history/2023-gnnv2-cpu/race_cache hammertime_100/tools/train_gnnv4.sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File hammertime_100\tools\train_gnnv4.ps1 -NormalizedSource <old-run>\race_cache
```

Options: `--device cpu|cuda` / `-Device`, `--epochs N` / `-Epochs N`, `--resume <run-dir>` / `-Resume <run-dir>` (continues after an interruption from the last complete epoch). `HAMMERTIME_RAW_SESSIONS` / `-RawSessions` is only needed if you enable the yellow-flag review flag.

The first run builds the cohort (one race at a time). Later runs with the same config, source and code reuse it after re-hashing every shard.

## 3. Outputs (`.local/research-history/gnnv4-<UTC>-<pid>/`)

- `run.log`, `run_manifest.json`, `effective_config.yaml`
- `training/best.pt`, `training/best.last.pt`, `training/training_history.csv` (with validation R² and aero coefficients per epoch)
- `test_metrics.json`: unchanged gate metrics and physics tests; a `supplementary` block (trimmed R², residual deciles, train-only constants, per-sector DRS estimate, stride-1 dense test pass); aero coefficients
- `test_predictions.csv`

Exit code 3 means the memory preflight stopped the run before any work. Lower the batch size (`-- --batch-size 64` on Linux, `-ExtraArgs '--batch-size','64'` on Windows) or free memory.

## 4. Diagnostics

```bash
uv run --frozen python hammertime_100/tools/diagnose_r2.py --normalized-source <old-run>/race_cache --raw-sessions-dir <raw_sessions> --output .local/research-history/r2-diagnostics.json
```

See `docs/research/gnnv4_r2_root_cause.md`. Review-gated target changes live under `review_flags` in the config and are OFF.
