# HammerTime 100-tier Implementation Report

## Status

The `hammertime_100/` source now contains the configured M1–M9 pipeline, a multi-session runner, an artifact-backed Streamlit dashboard, nine visualization modules, and runtime broadcast-card generation. This report describes the source and its intended outputs. No pipeline, model, tests, build, linter, dependency installation, or dashboard was executed for this completion work. Therefore, runtime compatibility, data availability, model quality, physics-gate results, and generated artifact quality remain unverified.

“100-tier” identifies the intended implementation scope. It is not a claim that every quality gate passes or that the model reaches a target score.

## Data and model flow

| Module | File | Responsibility and output |
|---|---|---|
| M1 | `src/ingest.py` | Load a real FastF1 session, use FastF1 cache, and export raw laps and telemetry to Parquet. |
| M2 | `src/preprocess.py` | Align driver telemetry on a configured time grid, interpolate supported gaps, preserve validity and session identity, flag safety-car/pit laps, and compute leader/follower gaps. |
| M3 | `src/target.py` | Compute mini-sector durations and session-scoped clean-air references. Fallback order is driver, teammate, then session grid; rows without a supported reference are omitted. |
| M4 | `src/graph.py` | Build fixed-slot graph snapshots with nine node features, masks, directed leader-to-follower edges, and RBF edge weights. Each snapshot carries its RaceId, driver-name mapping, and safety-car state. |
| M5 | `src/dataset.py` | Match labels to graph timestamps, form contiguous W-step windows and K-step targets, split by time or race, and fit feature scaling on training data. |
| M6 | `src/model.py` | Define the spatio-temporal graph model. |
| M7 | `src/train.py` | Train with the configured optimizer, validation, early stopping, and checkpoint output. |
| M8 | `src/evaluate.py` | Score the held-out loader, run configured clean-air and DRS checks, and write an evaluation report with split metadata and prediction rows. |
| M9 | `src/baselines.py` | Run XGBoost and LSTM comparisons on the same split where supported; unavailable scores remain unavailable. |

The graph node inputs are speed, throttle, five tyre-compound indicators, tyre age, and mini-sector index. The default graph keeps 20 driver slots. Inactive, pit, or invalid nodes are masked. The default edge threshold is 2 seconds and `graph.rbf_sigma` sets the Gaussian weight scale.

Safety-car rows are excluded from target labels. M5 rejects any sample window containing a snapshot marked as safety-car. Pit and invalid rows are masked. These filters are implemented in source but have not been runtime-verified in this completion.

Target baselines are calculated within each RaceId. This preserves the target definition across different circuits: sector times from different tracks are not pooled. This per-session label calibration is separate from model feature scaling, which is fit on the training partition.

## Session configuration and split behavior

Run `uv run python hammertime_100/run_pipeline.py` from the repository root, or pass a configuration path when calling `run_100_percent_pipeline(config_path=...)` from Python. The default configuration in `configs/default.yaml` selects one session. An empty `sessions` list retains this single-session behavior.

For multiple sessions, configure entries containing `year`, `gp`, and optionally `session_type` and `race_id`. Race identifiers must be unique. Each session is ingested and processed separately, with a distinct cache subdirectory and processed directory. Its graph snapshots and labels carry the same RaceId before the runner combines them.

The default split is temporal. Set `data.split_method: race` only when at least three valid RaceIds are present. M5 enforces that minimum and raises a descriptive error when the request cannot be honored. Temporal windows are contiguous, and split boundaries are purged according to the window and prediction horizon.

Key timing/filter settings are declared in the YAML configuration: `timestamp_tolerance_sec`, `continuity_tolerance_sec`, `min_valid_targets_per_sample`, and `physics_tests.drs_max_gap_sec`. The default DRS proximity condition is strict: `0 < gap < drs_max_gap_sec`.

## Outputs

With the default paths, the runner writes under `hammertime_100/`:

| Artifact | Path | Contents |
|---|---|---|
| FastF1 cache | `data/cache/<race-id>/` | FastF1-managed session cache. |
| Raw tables | `data/processed/<race-id>/raw/raw_laps.parquet` and `raw_telemetry.parquet` | Source lap and per-driver telemetry tables when telemetry is available. |
| Per-session aligned telemetry | `data/processed/<race-id>/aligned_telemetry.parquet` | Aligned telemetry for that session. |
| Combined aligned telemetry | `data/processed/aligned_telemetry.parquet` | Aligned rows with RaceId preserved. |
| Combined labels | `data/processed/target_labels.parquet` | Target rows with RaceId, driver, time, lap, mini-sector, and target fields. |
| Model checkpoint | `best.pt` | Best validation checkpoint and training-fitted feature scaler. |
| Evaluation report | `data/processed/evaluation_report.json` | Held-out metrics, baselines, physics checks, split metadata, and row-level predictions. |
| Visuals | `visuals/v1_track_map.png` through `visuals/v9_clean_air_cal.png` | Outputs are generated only when the corresponding real inputs are available. |
| Broadcast cards | `broadcast/rendered/card_*.png` | Up to ten cards from observed close-gap telemetry; a prediction is attached only when a held-out record matches. |

The runner clears its known generated visual/card files before rendering, so an unavailable output does not leave an older image presented as current. Its summary returns absolute artifact paths, quality gates, and `presentation_status`, including missing visual names and the available versus required card count. The presentation gate requires all nine visuals and at least ten fresh cards; unavailable data leaves the gate false.

## Visuals and dashboard

The runner supplies real aligned telemetry, held-out predictions, baseline scores, physics results, tyre-age/gap observations, and configured session context to V1–V9. V4 sums predicted penalties only for complete laps containing every configured mini-sector; DRS intervals require all included sectors to satisfy the strict gap condition. Cards are based on observed close gaps; their projected lap loss is labeled extrapolated from the matched mini-sector prediction.

Start the dashboard from the repository root with `uv run streamlit run hammertime_100/dashboard/app.py`. It reads the combined telemetry Parquet, evaluation JSON, generated visuals, and card images. It reports missing or malformed artifacts. The dashboard is not a live connection to a race feed.

## Quality gates and evidence

The runner records gates for aligned data, targets/graphs, non-empty train/validation/test splits, checkpoint creation, baseline availability, physics-test results, and presentation output. The configured score and physics thresholds are acceptance criteria only. They are not evidence of passing performance. The report and visuals are produced from a run; source files alone do not establish a result.

No 100-tier MAE, R², race-loss, latency, aerodynamic-effect, or model-superiority number is asserted here. To establish those claims, run the pipeline on explicitly identified sessions and retain its evaluation report and generated artifacts. The historical 50-tier reports describe earlier work and are separate from this 100-tier implementation status.

Focused contract tests are in `tests/test_dataset_contracts.py` and `tests/test_baseline_isolation.py`. Run them from the repository root with `python -m unittest discover -s hammertime_100/tests`. They use in-memory fixtures and do not require FastF1 data or network access, but they need project dependencies including PyTorch, PyTorch Geometric, pandas, NumPy, and scikit-learn. They were not executed during this review.

## Known limits

- FastF1 session loading requires a usable local cache or access to FastF1's data source.
- Race-held-out evaluation requires at least three sessions with enough valid graph windows and targets in each split.
- Visuals and cards can be unavailable when required real session data is missing; the implementation does not invent data to complete the gate.
- The dashboard reads files from the default configured output locations. Editing configuration paths may require corresponding dashboard path changes.
- This report reflects static source review only. Dependency resolution, imports, data contracts under real sessions, training, model scoring, dashboard rendering, and artifact generation have not been executed here.
