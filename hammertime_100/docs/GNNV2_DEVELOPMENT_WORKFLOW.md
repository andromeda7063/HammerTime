# GNNv2 development and training workflow

## Branch and workspace

Work on `dev/gnnv2` in the GNNv2 worktree. In the shared checkout this is `/home/andromeda/.codex/worktrees/gnnv2-attention/HammerTime`; on another machine, clone `dev/gnnv2` into a separate checkout. Keep website work in its own `dev/frontend` worktree. Do not switch either person's active worktree to the other branch.

Start or resume with:

```sh
git status --short --branch
git pull --ff-only
```

Commit and push model, graph, dataset, training, config, and ML-test changes only to `dev/gnnv2`. Keep new model variants opt-in until reviewed. Preserve the existing target definitions, coverage rules, split membership rules, metrics, physics thresholds, and acceptance gates unless a separately reviewed change explicitly authorizes a change.

## Training and artifacts

Use `uv` with `--frozen` for project commands. The GNNv2 season configuration is `hammertime_100/configs/train_2023_fastf1_season_gnnv2.yaml`. For GPU setup, local-data requirements, and the exact Windows season-run command, follow `hammertime_100/docs/WINDOWS_GPU_RUNBOOK.md`.

Give each run a distinct output directory under `.local/research-history/`, for example:

```sh
RUN_OUT=.local/research-history/2023-fastf1-gnnv2/local-run
uv run --frozen python hammertime_100/run_2023_fastf1_season.py \
  --config hammertime_100/configs/train_2023_fastf1_season_gnnv2.yaml \
  --output "$RUN_OUT" \
  --device cuda --epochs 30 --window-stride 2
```

For local-data-only execution, also pass `--local-data-only --raw-sessions-dir PATH` as documented in the runbook. Keep caches in this worktree's ignored data/cache path and run artifacts in ignored `.local/research-history/`. Never commit credentials, raw race tables, caches, checkpoints, or unreviewed generated results.

Before reporting a score, verify the source manifests and required race coverage, the saved split and test counts, the completed test metrics and prediction file, and the checkpoint/config/implementation hashes. Keep logs and unresolved scientific assumptions in the repository's prescribed locations. Contract tests use fixtures and do not establish race performance.

Run focused GNN tests and, when appropriate, the full suite:

```sh
uv run --frozen python -m unittest discover \
  -s hammertime_100/tests -p 'test_gnnv2_model.py'
uv run --frozen python -m unittest discover \
  -s hammertime_100/tests -p 'test_future_speed_auxiliary.py'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen python -m unittest discover -s hammertime_100/tests
```

## Handoff to frontend

When a run is ready for website review, give the frontend developer its run ID and output location, config and implementation fingerprints, source/coverage manifests, held-out split and race IDs, `test_metrics.json`, `test_predictions.csv`, and checkpoint SHA-256. Identify any in-sample outputs separately from held-out test results. Do not claim model improvement until the authentic held-out comparison and existing gates support it.

The website consumes static JSON and does not load checkpoints. The frontend developer updates the exporter to select and validate the new run, regenerates the tracked presentation payload, and runs the browser checks. Keep the checkpoint and full raw data local; only commit reviewed code and the minimum reviewed presentation data required by the website.

## Integration

The branches share a base and each has commits not yet in `main`. Integrate them together so the common history appears once in the PR. From any existing worktree, create a third worktree once both branches are ready:

```sh
git fetch origin
git worktree add -b dev/integration ../HammerTime-integration origin/dev/frontend
git -C ../HammerTime-integration merge --no-ff origin/dev/gnnv2
```

Resolve the shared edits to `ASSUMPTIONS.md`, `OPEN_QUESTIONS.md`, and `RESEARCH_CHANGELOG.md` there. Run the full ML suite and the frontend checks in `hammertime_100/frontend/DEVELOPMENT_WORKFLOW.md`, then push `dev/integration` and open one PR from it to `main`. Keep both contributor worktrees on their own branches while integration is reviewed.
