# OpenF1 2023 multi-session cohort acquisition

**Status:** Raw source acquisition complete; preprocessing and model fitting are pending. This record does not certify sample coverage, register an experiment, or report race-prediction results. **Recorded:** 2026-10-31.

## Scope and separation

Two explicit OpenF1-only configurations define the expanded cohort:

- `configs/openf1_2023_race_holdout.yaml`: existing splitter's race partitioning; Monza 2023 is training data because its old test result was exposed, Silverstone is validation, and Hungary is the newly reserved race test.
- `configs/openf1_2023_multi_session_temporal.yaml`: the existing temporal splitter over the ordered concatenation Monza, Silverstone, Hungary. This is a distinct development protocol, not a second estimate of race-heldout performance.

Both configurations use OpenF1 session keys 9157 (Monza), 9126 (Silverstone), and 9133 (Hungary; OpenF1 location value `Budapest`). They set the existing 2023 circuit lengths individually: Monza 5,793 m, Silverstone 5,891 m, Hungary 4,381 m. These are sourced from the respective [Formula 1 Italy](https://www.formula1.com/en/racing/2023/italy), [Great Britain](https://www.formula1.com/en/racing/2023/great-britain), and [Hungary](https://www.formula1.com/en/racing/2023/hungary) event pages. The common `track.lap_length_m` is Monza's value because it remains required by the current config schema; session-specific values are the ones the multi-session loader must use for each event.

No FastF1 artifacts are included. A separate FastF1 Monza cache/artifact remains separate. Zandvoort is excluded because its race-control state sequence leaves an unresolved safety-car interpretation. No data coverage gates, split behavior, targets, metrics, physics thresholds, or acceptance criteria were changed. The configs opt into the separately implemented temporal hypothesis (`temporal_mode: gated_dilated`, dilations `[1, 2, 4]`, kernel 3); this does not make it a selected or validated model.

## Acquisition implementation and reproducibility

`acquire_openf1.py` acquires or loads only raw provider tables for configured OpenF1 sessions. It delegates query construction, exact query-hash cache lookup, validation, request pacing, retries, and response provenance to the existing `openf1_client`. It writes a per-session `openf1_acquisition_manifest.json` without overwriting a different existing manifest. The manifest records the pinned session metadata, effective query settings, config hash, table row counts, cached raw artifact paths, and per-response status/query/content SHA-256 metadata. It does not normalize, create graph samples, train, or score.

From `hammertime_100/`, acquire configured missing data using the locked environment:

```bash
uv run --frozen python acquire_openf1.py --config configs/openf1_2023_race_holdout.yaml
```

Restrict a network acquisition to selected configured sessions with repeatable `--race-id`, for example:

```bash
uv run --frozen python acquire_openf1.py \
  --config configs/openf1_2023_race_holdout.yaml \
  --race-id 2023-Monza-R
```

Verify exact query-cache reuse with the network blocked:

```bash
uv run --frozen python acquire_openf1.py \
  --config configs/openf1_2023_race_holdout.yaml --offline
```

Raw responses and manifests are local cache artifacts under `data/cache/openf1_2023_race_cohort/<race-id>/openf1/` and `<race-id>/openf1_acquisition_manifest.json`; they are intentionally not tracked. The client hashes the canonical endpoint URL plus sorted query parameters, so 300-second Monza cache entries do not stand in for the newly configured 7,200-second requests. Monza was requested under the new query shape; Silverstone and Hungary's raw response/meta cache pairs were copied from the preserved local 7,200-second acquisitions. The copy leaves the original local sources untouched.

## Acquisition evidence (not preprocessing evidence)

The command below completed successfully on 2026-10-31, with request interval 2.1 seconds and configured 7,200-second time chunks:

```bash
uv run --frozen python acquire_openf1.py --config configs/openf1_2023_race_holdout.yaml --offline
```

Offline mode blocks HTTP cache misses; all three sessions resolved using exact cached endpoint/query keys. The manifests recorded the following raw table row counts and request records:

| OpenF1 session | Request records | Laps | Race control | Car data | Location | Other table counts |
|---|---:|---:|---:|---:|---:|---|
| Monza 9157 | 45 | 968 | 45 | 435,423 | 453,568 | session 1; drivers 20; stints 45; pit 25; weather 0 |
| Silverstone 9126 | 47 | 974 | 46 | 412,000 | 426,260 | session 1; drivers 20; stints 44; pit 24; weather 0 |
| Hungary 9133 | 47 | 1,255 | 87 | 473,080 | 484,000 | session 1; drivers 20; stints 56; pit 36; weather 0 |

These are provider table counts only. They say nothing about valid laps, aligned snapshots, labels, graph samples, usable split size, or model quality. The content hashes and exact queries are preserved in the per-response `.meta.json` files and the per-session manifests in the ignored local cache. Before fitting, the preparation step must report and freeze the actual preprocessing outputs, source/config/code hashes, split membership, and test exposure status in a cohort manifest. Until that manifest exists, neither YAML constitutes a registered analysis.

## Pending decisions and interpretation

The multi-session preparation runner must demonstrate that each session's `lap_length_m`, cache root, processed output root, and provider identity are actually honored before any results are accepted. Existing temporal concatenation is a development protocol whose chronology and outcome-boundary interpretation require explicit reporting. The race-heldout cohort's new test event is Hungary; any previously exposed Monza result remains exposed and must not be represented as untouched test evidence. Training on additional races may help estimate model parameters, but no R² improvement is guaranteed; report the unchanged metrics and acceptance gates after the preprocessing manifest is frozen.
