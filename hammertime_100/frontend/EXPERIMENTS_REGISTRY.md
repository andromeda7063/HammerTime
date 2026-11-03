# Experiment registry

`experiments.html` is a standalone evidence index for historical and current model evaluations. It deliberately groups results by telemetry source, target and split instead of sorting every metric into a single leaderboard. The season dashboard remains the canonical view of the current FastF1 2023 held-out season run.

## Regenerate the static data

From the repository root, run:

```sh
uv run --frozen python hammertime_100/frontend/data/export_experiments.py
```

The exporter reads archived run records under `.local/research-history/`, the existing legacy `metrics.json`, and report text. It writes the self-contained static `data/experiments.json`. Archived `.local` inputs are ignored by Git, so regeneration requires the corresponding local archive; the generated JSON preserves source paths and SHA-256 fingerprints for provenance. No checkpoint or source dataset is copied, changed, or deleted.

## Evidence included

- FastF1 2023 season held-out test and its matched zero predictor: run `769a76aab164b514cc0a`. The target is retrospective and full-session calibrated. Counts distinguish 37,956 valid node-target predictions from 25,129 graph windows. MAE passes and R² fails; the clean-air check passes and DRS has no complete held-out laps, so it is unmeasurable.
- OpenF1 Hungary 2023 fixed validation seed screen: five ST-GNN variants plus mean, XGBoost, LSTM and zero baselines. The suite is marked inconclusive and none is a test result.
- Two incomplete FastF1 Monza validation pilots sharing the prepared validation target mask, plus a separately prepared incomplete OpenF1 multi-session validation pilot. Its preregistration shares the recorded validation-membership hash with the Hungary candidate screen; no target-mask digest was saved, so target-mask equivalence is unverified.
- Failed/interrupted archived runs with no completed score, shown without invented metric values.
- Legacy Monza frontend metrics and final-report saved results, shown as report-only and unverified. The legacy payload's count of 984 conflicts with the report's 983; the run predictions and split metadata are unavailable.
- HammerTime 50 historical report score, marked report-only and separated due to its different session, target and graph-window sample unit.

The registry contains no synthetic/demo scores and does not imply comparability across groups. Report-only entries are transcriptions, not independently reproducible evaluations.

## Browse and export

Family, evidence class, evaluation group, within-group sort and text search are encoded in the URL; browser back/forward restores prior states. Sort options are source order, MAE ascending and R² descending. They reorder records only inside each evidence group, keep missing scores last, and leave group order unchanged. Search ignores accents and treats underscores and hyphens as spaces. **Reset filters** clears all controls and their URL parameters. The filtered CSV preserves each record's evidence class, protocol, sample unit and source paths/hashes; its rows use the same within-group ordering as the page. It is a heterogeneous evidence export, not a leaderboard. On narrow screens each metrics table has a focusable, named scroll region with a caption and arrow-key horizontal scrolling.

Each model name links to a stable record fragment and its evidence-group filter. Direct links focus and highlight the row after the static registry loads; unknown fragments are ignored. The five Hungary circuit-screen model keys have concise display labels; their exact source keys remain in secondary text and searchable.

The page validates the payload before rendering: scored evidence requires finite MAE and R² (RMSE may be null when the source did not report it), while failed groups require null metric and count fields. Invalid payloads or unknown URL selections show explicit messages; no fallback values are displayed.

## Serve locally

From this directory, use any static HTTP server so `data/experiments.json` is fetched from the same origin. Opening the HTML as a `file://` URL is not supported. If the JSON request fails, the page shows an explicit unavailable state and no fallback metrics.
