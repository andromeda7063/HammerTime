# Frontend development workflow

## Branch and workspace

Work on `dev/frontend` in the frontend worktree. In the shared checkout this is `/home/andromeda/Documents/code/projects/HammerTime`; on another machine, clone `dev/frontend` into a separate checkout. Keep GNNv2 work in its own `dev/gnnv2` worktree. Do not switch either person's active worktree to the other branch.

Start or resume with:

```sh
git status --short --branch
git pull --ff-only
```

Commit and push website changes only to `dev/frontend`. Keep commits focused on frontend code, static data exports, and frontend checks. Coordinate edits to `ASSUMPTIONS.md`, `OPEN_QUESTIONS.md`, and `RESEARCH_CHANGELOG.md` with the GNNv2 developer because both branches touch them.

## Website and model-result handoff

The website is a static client. It reads exported JSON and does not load a PyTorch checkpoint or perform inference in the browser. Keep browser-facing payloads and exporters under `hammertime_100/frontend/`.

When adopting a GNNv2 result, ask the ML developer for the run ID, config and source hashes, held-out split and coverage record, test metrics, and prediction export. Confirm whether the population is held-out or in-sample and label it accurately. Never publish fixture or synthetic values as race results, and do not copy checkpoints or raw race caches into tracked frontend data.

The current season exporter is pinned to the historical run ID in `data/export_season_model.py`. Before displaying a new GNNv2 run, update the exporter to select that run and validate its provenance and row counts, then regenerate the static payload. Keep the existing legacy and historical evidence groups distinct from the new run.

## Preview and verification

Use the locked environment (`uv` with `--frozen`) for project commands. Run the current browser checks from the repository root:

```sh
uv run --frozen python hammertime_100/frontend/tools/singapore_dashboard_smoke.py
uv run --frozen python hammertime_100/frontend/tools/experiments_smoke.py
uv run --frozen python hammertime_100/frontend/tools/marina_bay_smoke.py
uv run --frozen python hammertime_100/frontend/tools/marina_bay_visual_smoke.py
```

For manual preview, serve `hammertime_100/frontend/` over HTTP; the pages fetch adjacent JSON and are not supported through `file://`. Record screenshots and local logs under ignored `.local/research-history/` paths. Do not commit generated screenshots or user caches unless specifically reviewed.

## Integration

The branches share a base and each has commits not yet in `main`. Integrate them together so the common history appears once in the PR. From any existing worktree, create a third worktree once both branches are ready:

```sh
git fetch origin
git worktree add -b dev/integration ../HammerTime-integration origin/dev/frontend
git -C ../HammerTime-integration merge --no-ff origin/dev/gnnv2
```

Resolve shared research-document edits to `ASSUMPTIONS.md`, `OPEN_QUESTIONS.md`, and `RESEARCH_CHANGELOG.md` there. Run the checks above and the ML checks in `hammertime_100/docs/GNNV2_DEVELOPMENT_WORKFLOW.md`, then push `dev/integration` and open one PR from it to `main`. Keep both contributor worktrees on their own branches while integration is reviewed.
