# HammerTime — Assumption Ledger

**Document Standard:** ASD-STE100  
**Authority:** Law 10 (Master Directive)  

Every parameter tuned or assumed from `claude.md` and `claude_arc.md` is recorded in this ledger.

| ID | Parameter | Original Value | Final Value | File Location | Rationale |
|---|---|---|---|---|---|
| A01 | Gaussian RBF kernel scale ($\sigma$) | 1.0 s | 1.0 s | `configs/default.yaml` (`graph.rbf_sigma`) | Specified in `claude.md` §5.2. Smoothly decays edge weights between 0 s and 2.0 s. |
| A02 | Graph edge maximum gap | 2.0 s | 2.0 s | `hammertime_100/configs/default.yaml` (`graph.edge_threshold_sec`) | Configured graph cutoff from the architecture specification. It is not a measured aerodynamic boundary. |
| A03 | Clean-air reference threshold | > 3.0 s | 3.0 s | `hammertime_100/configs/default.yaml` (`target.clean_air_gap_sec`) | Defines the clean-air samples used to calculate the target reference. It is not a claim that wake effects are zero above this gap. |
| A04 | Number of graph nodes ($N$) | 20 | 20 | `configs/default.yaml` (`graph.num_nodes`) | Represents the standard 20-car F1 grid. Degraded/retired cars are retained and masked. |
| A05 | Rolling history window ($W$) | 10 snapshots | 10 snapshots | `configs/default.yaml` (`data.window_size`) | Captures dynamic approach and wake disturbance build-up over 2.5 seconds (at 4 Hz). |
| A06 | Prediction horizon ($K$) | 5 snapshots | 5 snapshots | `configs/default.yaml` (`data.horizon`) | Predicts delay 1.25 seconds ahead into the next mini-sector entry. |
| A07 | Telemetry resample frequency | [ASSUMED] Not specified | 4 Hz (0.25 s) | `hammertime_100/configs/default.yaml` (`preprocess.sample_hz`) | Configured temporal resolution. Resulting row and snapshot counts depend on each source session. |
| A08 | Circuit mini-sector count ($M$) | [ASSUMED] Not specified | 20 equal segments | `configs/default.yaml` (`track.mini_sectors`) | Divides lap distance into discrete sectors for localized baseline and feature extraction. |
| A09 | GAT hidden channels & heads | 64 / 4 heads | 64 / 4 heads | `hammertime_100/configs/default.yaml` (`model.hidden_dim`, `model.gat_heads`) | Configured model capacity. Training time and convergence have not been established for the current implementation. |
| A10 | Optimizer and learning rate | Adam / 0.001 | Adam / 0.001 | `hammertime_100/configs/default.yaml` (`train.lr`) | Configured Adam learning rate and weight decay. Convergence must be measured for each run. |
| A11 | Clean-air reference fallback hierarchy | Driver reference | Driver -> Teammate -> Session grid | `hammertime_100/src/target.py` | Use only observed clean sectors meeting `target.min_clean_samples`; otherwise omit the target. References are scoped to RaceId because sector times do not transfer between circuits. |
| A12 | Missing telemetry gap tolerance | [ASSUMED] Not specified | 1.0 s maximum | `hammertime_100/configs/default.yaml` (`preprocess.max_interpolate_sec`) | Interpolation is supported only across gaps up to this configured limit; unsupported rows are marked invalid. |

Values in this ledger describe configuration and label semantics. They do not establish predictive performance or physical validation results. See [`hammertime_100/REPORT.md`](hammertime_100/REPORT.md) for source status and verification limits.

## Overnight audit additions — 2026-10-30

The entries above are preserved as historical configuration records.

| ID | Parameter or contract | Previous behavior | Audited behavior | Location | Rationale |
|---|---|---|---|---|---|
| A13 | Mini-sector duration measurement | Prescribed sine/hash/wake formula | Interpolate observed distance boundary crossings, without extrapolating partial or unsupported sectors | `hammertime_100/src/target.py` | Targets must be measured from real telemetry. No change to clean-air thresholds, clipping, or duration plausibility limits. |
| A14 | Telemetry time origin | Driver-slice `Time` | Explicit `TimeSec`, otherwise shared `SessionTime`, otherwise `Time` | `hammertime_100/src/preprocess.py` | Lap timing and all drivers must use the same session clock. |
| A15 | Distance origin and missing lap/tyre metadata | Cumulative slice distance; default lap/compound/age | Anchor each lap at its observed start; mask unsupported metadata | `hammertime_100/src/preprocess.py` | Avoid accumulated distance offsets and invented clean laps or tyre features. Per-lap integration accuracy remains a reviewer question. |
| A16 | Session load success | Successful return from FastF1 `load()` | Loaded, nonempty laps and car telemetry are required | `hammertime_100/src/ingest.py` | FastF1 may internally log acquisition errors without raising them to the caller. |
| A17 | Missing dashboard results | Example MAEs and legacy images | Unavailable scores; only current report manifest images | `hammertime_100/dashboard/app.py` | Missing or malformed artifacts are not research evidence. |
| A18 | Hyperparameter selection | One configured fit; no explicit study | Optional declared model/optimizer candidates, scored by mean validation MAE across declared seeds; retain the first seed | `hammertime_100/src/tuning.py`, `configs/default.yaml` | Hold splits, scaling, targets, metrics and gates fixed. Test/physics results cannot select a candidate. The search remains unmeasured until authentic telemetry is available. |
| A19 | Presentation timestamp matching | NumPy default relative tolerance | Fixed absolute tolerance with `rtol=0` | `hammertime_100/run_pipeline.py`, `dashboard/app.py` | A larger session timestamp must not widen the time slice and mix observations from adjacent frames. Existing absolute tolerances are preserved. |
