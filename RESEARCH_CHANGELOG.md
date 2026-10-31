# Research and Change Record

This file records durable implementation changes and their verification limits. Scientific assumptions and unresolved decisions remain in [ASSUMPTIONS.md](ASSUMPTIONS.md) and [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). Local execution records are archived under the ignored `.local/research-history/` directory and are not included in Git commits.

## OpenF1 and pipeline correctness updates — 2026-10-30

The current `hammertime_100/` source includes validated explicit provider/session selection, cached and verified OpenF1 acquisition, shared-clock timestamp normalization, observed lap-boundary targets, conservative masking of ambiguous lap ownership, and stricter handling of missing or incomplete source data. Pipeline artifacts record source and configuration provenance; dashboard outputs are tied to explicit configurations and current artifact manifests. Optional tuning records its declarations and selects only by validation results on the declared split and scaler. Existing metrics, splits, gates, and physics thresholds were retained.

The latest adapter follow-up groups and sorts telemetry by driver and restricts per-lap processing to the observed interval plus potential interpolation brackets. Regression coverage compares bounded-window results with full-source calculations. This is not a runtime benchmark and does not establish faster end-to-end normalization.

The recorded offline test run reported 158 tests passing in 6.920 seconds with HTTP requests explicitly blocked. This verifies fixture and contract behavior without network acquisition; it does not establish authentic race-data quality, model performance, or physics acceptance. The OpenF1 preflight status and single-driver diagnostic observations, including partial location coverage and limited supported-distance coverage, are archived in the local research history. They are not multi-driver evaluation or race-performance evidence.

No complete authentic multi-driver preprocessing, training, and held-out evaluation result or physics-gate result is established by these records. Preserve insufficient-coverage outcomes as unmeasurable and do not use held-out test results or gates to select candidates or thresholds.

## Run-record cleanup — 2026-10-31

The detailed `MORNING_REPORT.md` and pre-cleanup versions of affected research documents, along with the branch/worktree inventory, commit list, branch diff summary, and pre-cleanup tracked-document diff, were copied to `.local/research-history/2026-10-31/` and hash recorded before removal. This local archive is ignored and contains no environment file or credentials. The local branch was renamed from `overnight-improvements` to `codex/research-improvements`; commits and remote refs were preserved. The prior working-tree edits in `OPEN_QUESTIONS.md` are retained, and this record documents the cleanup.
