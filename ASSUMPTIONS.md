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

## Implementation additions — 2026-10-30

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

## Follow-up additions — 2026-10-30

| ID | Parameter or contract | Previous behavior | Audited behavior | Location | Rationale |
|---|---|---|---|---|---|
| A20 | Temporal convolution context | Symmetric padding; last output omitted the oldest value of its configured kernel window | Left padding by `kernel_size - 1`; the last output uses the full latest causal window | `hammertime_100/src/model.py` | Match the documented history-only prediction contract. This changes model computation, so historical checkpoints are not evidence for the revised model. |
| A21 | Duplicate telemetry timestamps | Deduplicate rows before channel validity checks | Keep the first finite observation independently for each channel at a timestamp | `hammertime_100/src/preprocess.py` | Missing Speed or Distance in one duplicate row must not hide another observed value at the same time. |
| A22 | Held-out sample ordering | Assume evaluation loader iterates in dataset order | Require a sequential sampler before attaching `dataset.samples` metadata | `hammertime_100/src/evaluate.py` | Prevent wrong driver/lap/gap attribution when a custom or shuffled loader reorders predictions. |
| A23 | Run provenance | Report supplied split and scaler but no content hashes | Record source/effective config, raw exports, aligned data, targets, checkpoint hashes, Git state and software versions | `hammertime_100/run_pipeline.py` | Allow a reviewer to identify the exact files and environment behind each completed evaluation report. |

## OpenF1 adapter additions — 2026-10-30

The user's explicit OpenF1 adapter request extends the earlier FastF1-only scope recorded in `claude.md`; the earlier record is preserved.

| ID | Contract | Chosen behavior | Rationale |
|---|---|---|---|
| A24 | Source selection | Default FastF1; explicit OpenF1 source and session key 9157 for Monza 2023; no provider fallback | OpenF1 starts in 2023. The user selected 2023 after the 2021 coverage mismatch was established. Provider-independent race identity prevents holding out another copy of the same event. |
| A25 | OpenF1 clock | UTC session `date_start` supplies a common origin; ISO timestamps may use different fractional precision or UTC offsets | Driver slices must not have separate time origins. Valid mixed-precision timestamps must not be coerced to missing. |
| A26 | OpenF1 distance | Trapezoidal integration of observed km/h divided by 3.6, independently anchored at supported lap starts; any speed gap above the existing maximum invalidates that lap's distance | No extrapolation, gap filling beyond the configured limit, or stretching to circuit length. Integrated-distance deviations are evidence to report, not a parameter to tune around physics gates. |
| A27 | Metadata compatibility | Pit-event `lap_number` and explicit out-lap markers; lap-wide SC/VSC flags through observed clearing; tyre age equals stint start age + lap offset + 1 | Match the existing FastF1 lap-level feature/filter semantics. FastF1 3.8.3 increments stint StartLaps by one for its first completed lap. No fabricated pit timestamps are supplied. |
| A28 | Acquisition policy | 300-second windows, 30-second timeout, three attempts, 2.1-second request spacing, verified response cache | These are explicit client reliability settings, not model or physics thresholds. Optional channel transport failures remain visible; wrong-session rows and required-data failures remain fatal. |
| A29 | Study evidence | Persist declarations before fitting and update completed trials/failure context atomically | An interrupted search must retain its protocol and partial evidence without inventing a selected candidate. |
| A30 | Dashboard source configuration | Optional `HAMMERTIME_CONFIG` and configured evaluation-report path | Display the requested run's artifacts and validate reports even when their filename differs from the default. |
| A31 | GNNv2 auxiliary speed labels | Use aligned `Speed` from the next five eligible 4 Hz graph frames; valid only when the car is active at the final encoder frame and at the future frame, and the aligned speed is finite | `hammertime_100/src/graph.py`, `hammertime_100/src/dataset.py`, `hammertime_100/configs/train_2023_fastf1_season_gnnv2.yaml` | Labels remain separate from node covariates. They inherit the existing interpolation and coverage contract; they are not raw-observation validity masks. Auxiliary weight 0.05 and scale 100 km/h are initial experiment settings, not empirically calibrated values or new acceptance thresholds. |
| A32 | Typed physical-edge scaling | Fit the existing physical distance/relative-speed scaler on unique train-input edges, then apply it only to active attrs when `model.typed_edge_values` is enabled; derive from preserved raw physical-edge attrs | `hammertime_100/src/dataset.py`, `hammertime_100/src/physical_graph.py` | Legacy configurations retain their prior raw-edge path. The transformation is deterministic from raw attrs, so repeated loader creation is idempotent. Validation requires checkpoint scaler metadata to match the train-fitted loader scaler. |

## Windows resource-profile limits — 2026-11-03

The season worker cap retains a 2 GiB reserve and an estimated 700 MiB per normalization child as an inherited sizing heuristic. It has not been measured as a safe maximum for the 2023 race tables or on Windows. The context-v2 path releases normalized aligned tables race by race while constructing snapshots, but retains the complete graph-snapshot list for split/scaler fitting and training. These are implementation/resource assumptions only; they do not alter scientific feature or data assumptions. See [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) item 17 for the required authentic Windows profile before changing the heuristic or claiming a measured improvement.

## GNNv3 edge-temporal candidate — 2026-11-03

| ID | Parameter or contract | Chosen behavior | Rationale |
|---|---|---|---|
| A33 | Attention-free physical edge temporal encoder | Per input frame, apply one existing typed edge-conditioned message pass after node projection; preserve frame-specific physical-v2 edges, causal temporal stack, active masks, own-car branch and auxiliary head | Test whether edge-conditioned pair messages plus temporal modeling work better and cheaper than three GATv2 spatial passes. This is an implementation hypothesis only; no result or runtime gain is measured. |
| A34 | CPU overnight training profile | 64 hidden channels; batch 128; dropout 0.10; edge dropout 0; LR 5e-4; weight decay 1e-4; max 45 epochs; patience 12 | Conservative one-run settings for the reported 16 GiB/12-thread CPU. They are not selected hyperparameters. The prior model reportedly selected epoch 18, so more epochs are not assumed to cure its score. |
| A35 | GNNv3 source/output preservation | Verify exact existing cache read-only; train in a separate empty run directory; require explicit production best-checkpoint path and record a SHA256-verified copy before fitting | Avoid rewriting production checkpoints, prepared caches or reports. No production checkpoint path is inferred. |

All data, target, split, metric, coverage, acceptance, and physics contracts remain unchanged. See [`hammertime_100/docs/GNNV3_CPU_OVERNIGHT_RUNBOOK.md`](hammertime_100/docs/GNNV3_CPU_OVERNIGHT_RUNBOOK.md) and [`hammertime_100/docs/research/gnnv3_underperformance_and_cpu_candidate.md`](hammertime_100/docs/research/gnnv3_underperformance_and_cpu_candidate.md). These are proposed settings; no authentic GNNv3 training or score is claimed.
