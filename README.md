# HammerTime

HammerTime is a research prototype for modeling Formula 1 traffic delay from FastF1 telemetry. It builds time-aligned race graphs, predicts mini-sector delay relative to a session-specific clean-air reference, and reports held-out model, baseline, physics, and presentation results.

The repository contains two implementations. `hammertime_50/` is the earlier prototype and its historical report. `hammertime_100/` contains the expanded M1–M9 pipeline, dashboard, visual generators, and broadcast-card generator. The 100-tier label describes scope; it does not mean that performance or physics gates have passed. See [the implementation report](hammertime_100/REPORT.md) for current status and limits.

## Setup

Use Python 3.12 or newer. From the repository root, install the declared dependencies with either:

```bash
uv sync
```

or:

```bash
python -m pip install -r hammertime_100/requirements.txt
```

PyTorch Geometric installation can depend on the selected PyTorch and platform build. Follow its installation guidance if the package resolver cannot select compatible wheels.

## Run the 100-tier pipeline

Edit [`hammertime_100/configs/default.yaml`](hammertime_100/configs/default.yaml) to select a FastF1 session and output locations. Then run from the repository root:

```bash
uv run python hammertime_100/run_pipeline.py
```

The runner uses FastF1 and its local cache. It raises an error when the requested session cannot be loaded; it does not create synthetic race data. One configured race uses a temporal split. To use a race-held-out split, set `data.split_method: race` and provide at least three entries in `sessions`, each with `year`, `gp`, and optionally `session_type` and `race_id`:

```yaml
sessions:
  - year: 2023
    gp: Monza
    session_type: R
    race_id: 2023-Monza-R
  - year: 2023
    gp: Silverstone
    session_type: R
    race_id: 2023-Silverstone-R
  - year: 2024
    gp: Monaco
    session_type: R
    race_id: 2024-Monaco-R
data:
  split_method: race
```

Race identifiers must be unique. Each session gets its own FastF1 cache subdirectory and processed-data directory. Clean-air target references are calculated within each session because sector times do not transfer across circuits. Feature scaling and model fitting use the training partition.

## Dashboard

After a pipeline run, launch the artifact-backed dashboard from the repository root:

```bash
uv run streamlit run hammertime_100/dashboard/app.py
```

The dashboard reads the configured combined telemetry Parquet file, evaluation JSON report, generated visuals, and cards. Missing artifacts are reported as unavailable.

## Output locations

With the default configuration, outputs are written under `hammertime_100/`:

- `data/processed/<race-id>/raw/`: raw FastF1 laps and telemetry Parquet files.
- `data/processed/<race-id>/`: per-session aligned telemetry Parquet files.
- `data/processed/aligned_telemetry.parquet` and `target_labels.parquet`: combined session data.
- `data/processed/evaluation_report.json`: held-out metrics, baseline results, physics checks, split metadata, and prediction records.
- `best.pt`: model checkpoint, including the training-fitted feature scaler.
- `visuals/v1_...png` through `visuals/v9_...png`: generated only when their required source data is available.
- `broadcast/rendered/card_*.png`: cards generated from observed close-gap telemetry; predicted loss is shown only when a matching held-out prediction exists.

The returned runner summary reports artifact paths, quality-gate results, and presentation availability. A missing artifact or failed gate remains a failure; the runner does not fill gaps with demo output.

Focused contract tests are being added under `hammertime_100/tests/`. To run them from the repository root:

```bash
python -m unittest discover -s hammertime_100/tests
```

The tests use small in-memory fixtures and do not require FastF1 data or network access. They still require project dependencies such as PyTorch, PyTorch Geometric, pandas, NumPy, and scikit-learn. They were not executed as part of this source review.

## Project records

- [100-tier implementation report](hammertime_100/REPORT.md)
- [Assumption ledger](ASSUMPTIONS.md)
- [Executive overview](EXECUTIVE_ONE_PAGER.md)
- [50-tier historical report](hammertime_50/REPORT.md)
- [50-tier top-level report](REPORT_50.md)
- [Architecture specification](claude_arc.md)
