# GNNv4: why R² is low, why no aero model appears, and what changed

Recorded 2026-11-04 on branch `dev/gnnv4`. No GNNv4 model was trained or evaluated; the numbers below come from the read-only diagnostics tool and from earlier logged runs.

## Evidence

- **Reported GNNv2 result** (OPEN_QUESTIONS #18): test MAE 0.0877 s, R² 0.0945. The gate is R² in [0.68, 0.82].
- **GNNv3 overnight log** (`.local/research-history/gnnv3-cpu-overnight-20261103T142919Z-502563.log`): validation MAE went 0.0882 → 0.0867 over epochs 1–4, and train-mode MAE (0.098) stayed above validation MAE. The model levels off almost at once.
- **`tools/diagnose_r2.py`** on the authentic 22-race normalized source (`.local/research-history/r2-diagnostics-20261104.json`, ignored by Git):

| Diagnostic | Value |
|---|---|
| Labels | 420,692; std 0.229 s; skew 6.1; p1/p50/p99 = −0.38 / 0.002 / 0.70 s; 0.76% with \|y\| > 1 s |
| Race-split test R², train-mean constant | −0.002 |
| Test R², gradient boosting on the sector's *own* gap + sector + race progress (explanatory ceiling, not a forecast) | 0.086 (squared loss) / 0.153 (absolute loss) |
| Clean-air lap-to-lap correlation of a driver's sector residual (median over races) | 0.19 (range 0.07–0.75) |
| Integrated lap Distance ÷ circuit length (median over races) | 0.988; per-race p05 0.960–0.988; >99% of laps off by more than 0.3% |
| Per-race linear lap trend R² (fuel/tyre proxy) | median 0.03, max 0.14 |
| Yellow-flag (TrackStatus 2/7) label share | median 2.1%, max 13.9%; mean delta 0.053 s versus 0.023 s otherwise |
| Complete DRS-train laps among all labels (gap < 0.8 s in every sector) | median 19.5 per race (2–98). At the stride-2 grid the expected surviving fraction is 0.5²⁰ |
| v2 context inputs present in the season cohort | 3 of 11 (`Brake`, `DRSActive`, `a_long`). `lap_sin/cos/phase`, `abs_curvature`, `lateral_accel_proxy`, `SpeedRatio/Delta`, `ReferenceAvailable` are constant |

## Diagnosis

1. **The target is mostly noise, which limits achievable R².** Even with the sector's own contemporaneous gap, which a forecaster cannot know, held-out R² is only 0.09–0.15. Clean-air residuals barely repeat from lap to lap. GNNv2's 0.095 already sits near this ceiling. The 0.68–0.82 gate is not reachable with this target definition unless the measurement noise is reduced. This is a scientific finding, not a gate change.
2. **Measurement noise from integrated distance (F6).** Laps integrate about 1.2% short, so mini-sector boundaries drift by up to tens of metres late in the lap. Undershoot also drops the last sector of many laps. Rescaling each lap's distance is the most promising way to reduce the noise (review flag `rescale_lap_distance`).
3. **Dead and skewed inputs (F1).** The v2 speed reference is keyed by training circuit, so every held-out circuit gets constants. Separately, the season cohort path never ran `prepare_context_features` for v2, so 8 of the 11 context inputs were constant everywhere. `circuit_context_v3` computes geometry and uses causal in-race references that exist for unseen circuits.
4. **Objective/metric mismatch (F4).** The primary SmoothL1 β was hard-coded at 0.02 s, which is effectively L1. On a right-skewed target, L1 estimates the median and gives up the tail variance that R² rewards. Proposed: Huber β 0.15 (≈1.345 × the ~0.11 s core σ), LR 1e-3 with 1-epoch warmup and cosine decay, dropout 0.05, weight decay 1e-5, gradient accumulation 2 (effective batch 256), training stride 1, auxiliary weight 0.2. Selection stays validation MAE.
5. **Outliers (F3) and race-order gap (F5).** These are smaller but real. Both are available only as review flags.

## Why the aero model did not appear

- The **DRS-train gate is almost never measurable**, because it needs all 20 mini-sectors of a held-out lap and the stride-2 grid keeps each one with probability ≈0.5. The gate is unchanged. GNNv4 adds a *supplementary* stride-1 test pass and a per-sector estimate, both labelled as supplementary.
- The model had **no explicit aero structure**. The only wake channel was a binary flag (≤35 m and ≤0.6 s) that is almost never on at racing speed. GNNv4 adds `physical_v3`, a continuous wake weight `exp(−gap/0.5 s)`, and an opt-in **aero wake head**: `mean_t[w^k (v/100)² (a·corner − b·(1−corner)·(1+d·DRS))]`. Its coefficients are non-negative, it is exactly zero without an incoming edge, and its coefficients are written to the history, checkpoint and test report.
- An asymmetric "no negative delay in a DRS train" penalty was **not** added, because slipstream gains are real negative delays.

## Memory

The pickled cohort (~527k PyG objects) needed ~12 GiB RSS with 0.38 GiB free. `cohort_v4` stores memory-mapped per-race shards and builds windows lazily. It is built one race at a time (preflight estimate 1.8 GiB peak; training ~1.1 GiB). It refuses to start when the memory preflight fails.

## What still needs review

All review flags are OFF in `configs/train_2023_fastf1_season_gnnv4.yaml`, so the target, coverage and gates match GNNv2/v3. Turning any flag on is a reviewed change; record it in `RESEARCH_CHANGELOG.md` and `ASSUMPTIONS.md`. Recommended order: `rescale_lap_distance`, then `exclude_lap1_and_yellow_targets`, then `physical_gap_for_targets`. Compare each against the frozen flags-off run and rerun `diagnose_r2.py` to see whether the noise ceiling moves.
