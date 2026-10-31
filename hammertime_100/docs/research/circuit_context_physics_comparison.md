# Circuit-context and physical-graph comparison

This document specifies the validation-only comparison implemented by
`configs/openf1_2023_context_comparison.yaml`, `src/context_features.py`,
`src/physical_graph.py`, and `src/ablation.py`. It uses one OpenF1 cohort
(Monza, Silverstone, Hungary), reusing the already-normalized aligned and
target tables in the frozen OpenF1 cohort. It does not merge FastF1 data or
fetch/normalize the same raw sessions again. The original cohort and run
records remain unchanged.

## Input representations

`legacy_v1` retains the original nine columns in this exact order:
`Speed`, `Throttle`, five configured compound indicators, `TyreLife`, and
`MiniSector`. `circuit_context_v1` appends ten fields: brake state and support,
within-lap sine/cosine phase, circuit lap length, absolute curvature, lateral
acceleration proxy, geometry support, speed minus a train-fitted circuit/sector
reference, and reference support. Circuit IDs are used only to look up
calibration and are not passed to the model.

The context builder is row-preserving and causal. Geometry uses current and two
previous 4 Hz position/speed rows for the same driver and race. At the earliest
frame of a model window, this can use two earlier frames outside that ten-frame
window; they are still historical observations available at inference time.
The curvature
and lateral-acceleration values are proxies derived from sampled X/Y positions
and speed; they are not vehicle-dynamics measurements. Gaps, invalid positions,
or unsupported brake channels produce zero values plus explicit support flags.
The speed reference is the median speed by circuit and mini-sector using only
unique active input frames present in train windows. The active-node rule is
the existing `IsValid & ~IsPitLap`; no global fallback is used for unseen
circuit-sector groups.

## Graph views

The unchanged `race_progress` graph uses the existing directed leader/follower
records and scalar RBF edge values. The opt-in `physical_circuit` graph sorts
position-supported cars by `Distance % lap_length_m` and considers each car's
next observed car around the circular circuit. It estimates temporal gap as
distance gap divided by `max(follower_speed_mps, 10)`, applies the existing
configured edge cutoff and sigma, and emits a directed leader-to-follower edge.
Because OpenF1 position telemetry can be sparse or noisy, “next observed car”
is an approximation to physical adjacency and remains a falsifiable hypothesis.
It selects the nearest position-supported car before checking active-node
availability; an inactive selected neighbor blocks that edge rather than
bridging to another car. Position support is separate from the existing active
and target masks. Coverage and blocked-neighbor counts are recorded.

`physical_v1` has four raw edge channels: RBF weight, distance gap in meters,
leader-minus-follower speed in m/s, and position coverage. A scaler for only the
distance and relative-speed channels is fitted once using unique snapshots
touched by train input windows. RBF and coverage pass through unchanged. The
same prepared cohort also carries both topology arrays so the suite can swap
views without rebuilding or changing sample membership.

## Fixed candidate views and hypotheses

All five candidates use the same gated-dilated temporal block, dilations
`[1, 2, 4]`, kernel 3, masked MAE objective, labels, windows, split, and
validation-MAE checkpoint selection:

1. `legacy_input_legacy_graph`: nine features, `race_progress`, scalar RBF.
2. `circuit_context_legacy_graph`: 19 features, `race_progress`, scalar RBF.
3. `circuit_context_physical_scalar`: 19 features, `physical_circuit`, RBF only.
4. `circuit_context_physical_v1`: 19 features, `physical_circuit`, four edge channels.
5. `circuit_context_self_loop_only`: 19 features, no stored inter-car edges.

The first contrast estimates the added value of context features; the second
and third contrasts test physical neighbor selection and richer edge features;
the fifth tests whether any graph interaction helps beyond the same temporal
model. Candidate copies must retain identical train/validation targets and
masks. The output records per-circuit/sector validation errors, target and
prediction variances, runtime, parameter count, scaler metadata, tensor hashes,
and unchanged physics-gate configuration. Test data is never passed to the
runner or scored.

The fixed selection procedure screens all five views with seed 42. Among the
four non-control views, it selects the highest validation R² only when that
view's validation MAE is no worse than the legacy control and its R² exceeds a
train-target-mean constant baseline's validation R². It then trains that one
view and the legacy control with seeds 43 and 44; it does not search a runner-up
if no view qualifies. The R² sign is reported as computed, including negative
values. These conditions select a follow-up candidate and do not change any
existing quality gate or acceptance threshold. XGBoost and LSTM are reported as
validation-only baselines; their scores do not affect candidate selection.

The comparison does not promise an improved R². The three-session cohort is
still a temporal concatenation protocol, and the existing clean-air target's
whole-session calibration limitation remains. No test-set claim follows from
the selection or confirmation runs.

## Commands

Run from `hammertime_100/` with the locked environment. The aligned-source path
is the existing frozen OpenF1 cohort in the repository-level local history.
Both outputs must be new directories.

```bash
uv run --frozen python src/prepare_cohort.py \
  --config configs/openf1_2023_context_comparison.yaml \
  --aligned-source ../.local/research-history/2026-10-31/openf1-2023-multi-session-temporal-cohort \
  --output ../.local/research-history/2026-10-31/openf1-2023-circuit-context-physical-v1

uv run --frozen python train_prepared.py \
  --prepared ../.local/research-history/2026-10-31/openf1-2023-circuit-context-physical-v1/prepared_cohort.pkl \
  --config configs/openf1_2023_context_comparison.yaml \
  --output ../.local/research-history/2026-10-31/circuit-physics-v1-validation \
  --suite circuit_physics_v1 --device cpu
```

The preparation command reuses only already-normalized aligned and target
tables and creates the graph and train-only calibrations. The suite command is
validation-only, records its registration before fitting, and writes no result
into the held-out test partition.
