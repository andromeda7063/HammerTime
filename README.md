# HammerTime

HammerTime is a research prototype for modeling Formula 1 traffic delay from FastF1 or explicitly selected OpenF1 telemetry. It builds time-aligned race graphs, predicts mini-sector delay relative to a session-specific clean-air reference, and reports held-out model, baseline, physics, and presentation results.

The repository contains two implementations. `hammertime_50/` is the earlier prototype and its historical report. `hammertime_100/` contains the expanded M1–M9 pipeline, dashboard, visual generators, and broadcast-card generator. The 100-tier label describes scope; it does not mean that performance or physics gates have passed. See [the implementation report](hammertime_100/REPORT.md) for current status and limits.

## Setup

Use Python 3.12 or newer. From the repository root, install the locked dependencies with UV:

```bash
uv sync --frozen
```

PyTorch Geometric installation can depend on the selected PyTorch and platform build. Follow its installation guidance if the package resolver cannot select compatible wheels.

## Run the 100-tier pipeline

Edit [`hammertime_100/configs/default.yaml`](hammertime_100/configs/default.yaml) to select a FastF1 session and output locations. Then run from the repository root:

```bash
uv run --frozen python hammertime_100/run_pipeline.py
```

The runner uses FastF1 and its local cache. It raises an error when the requested session cannot be loaded; it does not create synthetic race data. One configured race uses a temporal split. To use a race-held-out split, set `data.split_method: race` and provide at least three entries in `sessions`, each with `year`, `gp`, and optionally `session_type` and `race_id`:

```yaml
sessions:
  - year: 2022
    gp: Monza
    session_type: R
    race_id: 2022-Monza-R
  - year: 2023
    gp: Monza
    session_type: R
    race_id: 2023-Monza-R
  - year: 2024
    gp: Monza
    session_type: R
    race_id: 2024-Monza-R
data:
  split_method: race
```

Race identifiers must be unique. Each session gets its own FastF1 cache subdirectory and processed-data directory. Clean-air target references are calculated within each session because sector times do not transfer across circuits. Feature scaling and model fitting use the training partition.

Sessions for a different circuit must supply their own positive `lap_length_m`; they cannot inherit Monza's geometry. Repeated source sessions under different IDs and IDs that map to the same directory are rejected.

### OpenF1 Monza 2023

OpenF1 historical coverage starts in 2023; Monza 2021 is not supported by this provider. Use the separate pinned configuration for the 2023 race (session key 9157):

```bash
uv run --frozen python hammertime_100/run_pipeline.py --config hammertime_100/configs/openf1_monza_2023.yaml
```

`session.source` defaults to `fastf1`; `openf1` requires an explicit `session_key` and validated HTTP settings. Source failures propagate without switching providers. Exact responses and request metadata are cached under the race's `openf1/` directory. Successful chunks can be resumed, and cached hashes/session identity are checked before reuse. Optional location/weather transport failures are recorded as unavailable; required timing, tyre metadata and car telemetry failures stop the run.

The adapter converts UTC timestamps to a shared clock and integrates observed speed separately within each lap. Unsupported speed gaps invalidate the lap's distance; distance is never scaled to force the circuit length. It retains the existing gap calculation instead of substituting the slower OpenF1 interval feed. Review `coverage_report.json` and `source_manifest.json` beside the per-session aligned table before interpreting model results. The separate output root is `data/processed/openf1_monza_2023/`.

## Reproducible hyperparameter selection

Use the committed lockfile with `uv sync --frozen`. Run a declared experiment with:

```bash
uv run --frozen python hammertime_100/run_pipeline.py --config hammertime_100/configs/default.yaml --tune
```

The optional `tuning` section declares three learning rates and three seeds; normal runs keep tuning disabled. Selection uses mean validation MAE across the declared seeds on one unchanged split and train-fitted scaler. The first declared seed supplies the selected candidate's checkpoint; the best seed is not cherry-picked. Every trial keeps its checkpoint and history under `data/processed/tuning/<study-id>/`, and `study.json` records all validation scores. Test evaluation follows selection once. Candidate overrides are restricted to model capacity/dropout and optimizer learning rate/weight decay; targets, split settings and gate thresholds cannot be searched.

Each ordinary run trains the configured model afresh. Existing checkpoints, histories and evaluation reports are copied to `data/processed/artifact_history/` before replacement. A checkpoint merely existing is not proof it belongs to the current experiment.

No tuned winner is claimed by the overnight audit: live FastF1 did not supply usable laps/telemetry. Session-wide clean-air references also make the present labels retrospective calibrations; strict prospective forecasting needs a separately agreed reference-fitting protocol. See [the morning report](MORNING_REPORT.md) and [open questions](OPEN_QUESTIONS.md).

## Dashboard

After a pipeline run, launch the artifact-backed dashboard from the repository root:

```bash
uv run --frozen streamlit run hammertime_100/dashboard/app.py
```

The dashboard reads the configured combined telemetry Parquet file, evaluation JSON report, generated visuals, and cards. Missing artifacts are reported as unavailable.

To view an OpenF1 run, set the same explicit configuration; relative config paths resolve from the current working directory:

```bash
HAMMERTIME_CONFIG=hammertime_100/configs/openf1_monza_2023.yaml uv run --frozen streamlit run hammertime_100/dashboard/app.py
```

The dashboard also honors `evaluation.report_path` when configured. Missing or invalid configuration is reported before results are displayed.

## Output locations

With the default configuration, outputs are written under `hammertime_100/`:

- `data/processed/<race-id>/raw/`: raw FastF1 laps and telemetry Parquet files.
- `data/processed/<race-id>/`: per-session aligned telemetry Parquet files.
- `data/processed/aligned_telemetry.parquet` and `target_labels.parquet`: combined session data.
- `data/processed/evaluation_report.json`: held-out metrics, baseline results, physics checks, split metadata, and prediction records.
- `best.pt`: model checkpoint, including the training-fitted feature scaler.
- `data/processed/presentation/<run-id>/visuals/v1_...png` through `v9_...png`: generated only when their required source data is available.
- `data/processed/presentation/<run-id>/cards/card_*.png`: cards generated from observed close-gap telemetry; predicted loss is shown only when a matching held-out prediction exists.

The returned runner summary reports artifact paths, quality-gate results, and presentation availability. A missing artifact or failed gate remains a failure; the runner does not fill gaps with demo output.

The evaluation report's `presentation` manifest identifies the current images. The dashboard does not treat older checked-in images as current results. The separate `frontend/` demonstration retains unverified historical/sample content and is not the audited dashboard.

Focused contract tests are being added under `hammertime_100/tests/`. To run them from the repository root:

```bash
uv run --frozen python -m unittest discover -s hammertime_100/tests
```

The tests use small in-memory fixtures and do not require FastF1 data or network access. They require the declared project dependencies. The overnight audit ran the expanded suite, including dashboard and checkpoint integration contracts; exact commands and results are in [MORNING_REPORT.md](MORNING_REPORT.md).

## Project records

- [100-tier implementation report](hammertime_100/REPORT.md)
- [Assumption ledger](ASSUMPTIONS.md)
- [Executive overview](EXECUTIVE_ONE_PAGER.md)
- [50-tier historical report](hammertime_50/REPORT.md)
- [50-tier top-level report](REPORT_50.md)
- [Architecture specification](claude_arc.md)
