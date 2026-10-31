# Research and Change Record

This file records durable implementation changes and their verification limits. Scientific assumptions and unresolved decisions remain in [ASSUMPTIONS.md](ASSUMPTIONS.md) and [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). Local execution records are archived under the ignored `.local/research-history/` directory and are not included in Git commits.

## OpenF1 and pipeline correctness updates — 2026-10-30

The current `hammertime_100/` source includes validated explicit provider/session selection, cached and verified OpenF1 acquisition, shared-clock timestamp normalization, observed lap-boundary targets, conservative masking of ambiguous lap ownership, and stricter handling of missing or incomplete source data. Pipeline artifacts record source and configuration provenance; dashboard outputs are tied to explicit configurations and current artifact manifests. Optional tuning records its declarations and selects only by validation results on the declared split and scaler. Existing metrics, splits, gates, and physics thresholds were retained.

The latest adapter follow-up groups and sorts telemetry by driver and restricts per-lap processing to the observed interval plus potential interpolation brackets. Regression coverage compares bounded-window results with full-source calculations. This is not a runtime benchmark and does not establish faster end-to-end normalization.

The recorded offline test run reported 158 tests passing in 6.920 seconds with HTTP requests explicitly blocked. This verifies fixture and contract behavior without network acquisition; it does not establish authentic race-data quality, model performance, or physics acceptance. The OpenF1 preflight status and single-driver diagnostic observations, including partial location coverage and limited supported-distance coverage, are archived in the local research history. They are not multi-driver evaluation or race-performance evidence.

No complete authentic multi-driver preprocessing, training, and held-out evaluation result or physics-gate result is established by these records. Preserve insufficient-coverage outcomes as unmeasurable and do not use held-out test results or gates to select candidates or thresholds.

## Run-record cleanup — 2026-10-31

The detailed `MORNING_REPORT.md` and pre-cleanup versions of affected research documents, along with the branch/worktree inventory, commit list, branch diff summary, and pre-cleanup tracked-document diff, were copied to `.local/research-history/2026-10-31/` and hash recorded before removal. This local archive is ignored and contains no environment file or credentials. The local branch was renamed from `overnight-improvements` to `codex/research-improvements`; commits and remote refs were preserved. The prior working-tree edits in `OPEN_QUESTIONS.md` are retained, and this record documents the cleanup.

## Supplied FastF1 cache and offline baseline — 2026-10-31

Added an explicit cache importer and a separate `supplied_fastf1_monza_2023.yaml` configuration. The importer checks all destination hashes before copying, aborts on conflicting content, skips identical files, rejects symlinks and active SQLite WAL/SHM sidecars, and leaves the source cache untouched. The supplied archive was imported to `hammertime_100/data/cache/supplied-fastf1/2023-Monza-R/`; the existing default race cache has a different HTTP-cache database and was preserved. An offline load through the imported cache reproduced the Italian Grand Prix 2023 Race session with FastF1 3.8.3: 957 lap rows, 20 drivers, 719,620 car-telemetry rows, and 746,600 position rows. FastF1 exposes position telemetry as `pos_data`.

An offline preprocessing and target diagnostic on that authentic cache produced 348,384 aligned rows (329,577 valid), 19 represented drivers, 944 valid driver-laps, 18,336 graph snapshots, and 17,244 target labels. The unchanged default temporal split produced 7,855 train, 1,656 validation, and 1,672 test windows after purging. Source file sizes and SHA-256 hashes, software versions, and counts are recorded in `.local/research-history/2026-10-31/monza-2023-baseline.json`; the cache-copy manifest is in the same local archive. This establishes successful data loading and preprocessing only. No authentic model was trained or evaluated, and these counts do not establish performance or physics gates. The untracked project report contains prior test metrics, so the held-out set must not be treated as untouched.

## Temporal outcome-boundary diagnostics — 2026-10-31

Added sector-exit timestamps and diagnostic counts for observed outcomes crossing into the next chronological partition's first input time. Diagnostics are unavailable when exit metadata is missing, session clocks are incomparable, or race-held-out mode is used. Targets, partition construction, gates, and thresholds are unchanged. This is a leakage audit signal, not a split correction or performance result; outcome-boundary coverage requires authentic session verification.
