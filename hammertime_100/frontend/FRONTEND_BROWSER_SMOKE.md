# Frontend browser smoke checks

**Updated:** 2026-11-03
**Scope:** Singapore in-sample inference dashboard, preserved held-out run split, experiment registry, and archived Marina Bay replay.
**Tooling:** Locked Python environment (`uv --frozen`), Python standard library, and headless Chromium.

## Current checks

Run from the repository root:

```sh
uv run --frozen python hammertime_100/frontend/tools/singapore_dashboard_smoke.py
uv run --frozen python hammertime_100/frontend/tools/experiments_smoke.py
uv run --frozen python hammertime_100/frontend/tools/marina_bay_smoke.py
uv run --frozen python hammertime_100/frontend/tools/marina_bay_visual_smoke.py
```

The Singapore dashboard smoke reconciles the displayed inference scope with the archived run split, checks checkpoint provenance and physics-gate status, loads the dashboard in Chromium, and checks for page-level horizontal overflow at 1440, 390, and 320 px. It saves the wide screenshot under ignored `.local/research-history/2026-11-03/`.

The registry smoke checks its evidence filters, URL/history state, sorting, CSV output, accessibility mechanics, and narrow layouts. The Marina Bay checks independently reconcile replay samples to cached FastF1 Parquet and exercise the replay controls, responsive layout, interpolation, and team colors.

## Local preview

```sh
uv run --frozen python -m http.server 8765 --bind 127.0.0.1 --directory hammertime_100/frontend
```

Open `http://127.0.0.1:8765/` for the Singapore in-sample inference dashboard or `http://127.0.0.1:8765/experiments.html` for the evidence registry. Use HTTP rather than `file://` because these pages fetch adjacent JSON files.

## Interpretation

Singapore is in the checkpoint run's training split. The page's Singapore MAE/RMSE/R² values describe in-sample inference, not held-out performance. The run's original held-out split metadata remains unchanged and is shown separately. Clean-air and DRS acceptance gates are marked not assessed for the in-sample result. The dashboard does not load a checkpoint or run inference in the browser.

The bounded SAI #55 lap 1 preview uses recorded cached features. Its first mini-sector remains unavailable because the recorded inputs do not contain the required 10-frame history; the exporter does not synthesize grid-state frames.

The older `browser_smoke.py` and `point_inspection_smoke.py` scripts contain assertions for the superseded four-race held-out page (including Mexico-specific filters). They are historical harnesses and are not current verification commands. Their prior detailed results remain in Git history and the dated report/changelog.

These checks do not establish the scientific validity of the checkpoint, the equivalence of the current aligned Singapore cache to the original prepared cohort features, full WCAG conformance, or deployment/CDN behavior.
