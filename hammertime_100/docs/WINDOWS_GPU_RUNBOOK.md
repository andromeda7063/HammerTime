# Run GNNv2 on native Windows with an RTX 4060

This runbook uses an existing Windows clone that already contains the authentic 2023 raw race tables under `race_cache`. It opens the preprocessing implementation in a separate Git worktree, reuses those raw files in place, and runs the 2023 GNNv2 season workflow with an RTX 4060. It uses PowerShell and `uv`. Package installation may use internet access; the local-data-only run does not authenticate to FastF1, discover the calendar, or request race data.

The explicit GNNv2 YAML sets `spatial_layers: 3`, `own_car_branch: true`, `own_car_hidden_dim: 32`, and batch size 256. The season runner defaults to 25 epochs and window stride 1, so the command below passes `--epochs 30 --window-stride 2` to match the config.

## Install the Windows tools

Install [Git for Windows](https://git-scm.com/download/win), the current [NVIDIA Windows driver](https://www.nvidia.com/Download/index.aspx) for the RTX 4060, and [uv for Windows](https://docs.astral.sh/uv/getting-started/installation/#windows). CUDA 13.x requires an NVIDIA R580-family driver or newer; the current Game Ready or Studio driver is the simplest choice. Reboot if the driver installer asks. The uv installer command is:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new PowerShell window after installing Git for Windows, uv, and the NVIDIA driver so their PATH and driver changes are available. Check that Windows can see the GPU:

```powershell
nvidia-smi
```

The prebuilt CUDA wheel below supplies the CUDA runtime. A separate CUDA toolkit installation is not needed for this run. See NVIDIA's [CUDA minor-version compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## Open the preprocessing branch beside the existing clone

The implementation branch created for this change is `dev/gnnv2-preprocessing`; Git cannot represent `dev/gnnv2/preprocessing` while a `dev/gnnv2` branch already exists. It is published on `origin`. Open a separate worktree from your existing clone so the current checkout and its untracked files stay in place.

```powershell
$ExistingRepo = 'C:\F1\HammerTime' # Change to the path of your already-populated clone.
$Repo = 'C:\F1\HammerTime-gnnv2-preprocessing'
git -C $ExistingRepo status --short
git -C $ExistingRepo remote set-branches --add origin dev/gnnv2-preprocessing
git -C $ExistingRepo fetch origin refs/heads/dev/gnnv2-preprocessing:refs/remotes/origin/dev/gnnv2-preprocessing
git -C $ExistingRepo worktree add --track -b dev/gnnv2-preprocessing $Repo origin/dev/gnnv2-preprocessing
Set-Location $Repo
git branch --show-current
```

The last command should print `dev/gnnv2-preprocessing`. Adding the branch to the origin fetch settings supports clones created with `--single-branch` and lets later pulls fetch updates. These commands add a separate worktree and do not change or delete the original checkout or its `race_cache` files.

## Install the locked Python environment

The project requires Python 3.12 or newer; the pinned Windows CUDA wheel below is for Python 3.12. Install the base locked environment with:

```powershell
uv python install 3.12
uv sync --frozen --python 3.12
```

On native Windows, the PyTorch wheel selected by `uv.lock` is the standard Windows wheel, which does not enable CUDA in this project setup. For the GPU run, create a per-command overlay with the official PyTorch 2.14.1 CUDA 13.0 wheel. This leaves `uv.lock` unchanged:

```powershell
$TorchCuda = 'torch @ https://download-r2.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-win_amd64.whl'
$GpuRun = @(
    'run', '--frozen', '--python', '3.12',
    '--with', $TorchCuda
)
```

Keep this PowerShell session open through the run. If you open another session, recreate `$TorchCuda` and `$GpuRun` there.

The CUDA overlay uses the same PyTorch release as the lock but the official `cu130` Windows build. The wheel is about 2 GB, so allow time and disk space for uv to cache it. Do not use `uv pip install` to mutate the environment or `--torch-backend` here; the overlay is applied to each `uv run` command.

## Check that the overlay can use CUDA

Run this check before starting the season workflow. It prints the runtime versions, requires CUDA availability, and performs a small GPU matrix operation:

```powershell
@'
import fastf1
import torch
import torch_geometric

print(f"FastF1 {fastf1.__version__}; torch {torch.__version__}; PyG {torch_geometric.__version__}")
print(f"CUDA build {torch.version.cuda}; available {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "PyTorch cannot access the CUDA device"
print(f"GPU: {torch.cuda.get_device_name(0)}")
x = torch.ones((256, 256), device="cuda")
y = x @ x
torch.cuda.synchronize()
print(f"GPU check: {y[0, 0].item()}")
'@ | & uv @GpuRun python -
if ($LASTEXITCODE -ne 0) { throw 'GPU readiness check failed; do not start training.' }
```

Continue only if it reports CUDA available and names the RTX 4060. If it fails, stop before the long season run and resolve the driver or wheel issue first. This check imports the third-party libraries only; it does not import or run HammerTime.

## Point to the existing authentic raw race tables

The runner expects `$RawSessions` to point directly to a directory containing one subdirectory per registered race. Each race subdirectory must contain `raw_manifest.json`, `raw_laps.parquet`, and `raw_telemetry.parquet`. For the common layout where files are already under `<existing clone>\race_cache\raw_sessions`, use:

```powershell
$RawSessions = Join-Path $ExistingRepo 'race_cache\raw_sessions'
if (-not (Test-Path $RawSessions -PathType Container)) {
    throw "Expected raw_sessions directory is missing: $RawSessions. Inspect race_cache with: Get-ChildItem (Join-Path $ExistingRepo 'race_cache') -Force"
}
$RaceDirs = @(Get-ChildItem $RawSessions -Directory | Where-Object { -not $_.Name.StartsWith('.') })
"Raw race directories found: $($RaceDirs.Count)"
if ($RaceDirs.Count -ne 22) { throw "Expected 22 race directories; found $($RaceDirs.Count) under $RawSessions. Inspect this directory and verify the raw_sessions path." }
Get-ChildItem $RawSessions -Directory | Select-Object -First 3 -ExpandProperty FullName
Get-ChildItem (Join-Path $RaceDirs[0].FullName 'raw_manifest.json'), `
              (Join-Path $RaceDirs[0].FullName 'raw_laps.parquet'), `
              (Join-Path $RaceDirs[0].FullName 'raw_telemetry.parquet') | Select-Object FullName, Length
```

The count command should report 22 race directories. If the files are under a different `race_cache` subdirectory, set `$RawSessions` to that exact `raw_sessions` directory; do not point to its parent and do not rename or copy the files. The runner verifies all 22 registered sessions, event identity, consistent round numbering, byte counts, and SHA256 hashes. Missing, corrupt, duplicate/inconsistent, or unregistered sessions stop the run with the expected race and manifest problem; the program never falls back to a FastF1 request. SHA256 verifies transfer integrity, not data origin.

```powershell
$RunOut = Join-Path $Repo '.local\research-history\2023-fastf1-gnnv2-preprocessing-windows'
New-Item -ItemType Directory -Force -Path $RunOut | Out-Null
$RunTimestamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$RunLog = Join-Path $RunOut "runner-$RunTimestamp.log"
"Repo: $Repo`nRaw sessions: $RawSessions`nOutput: $RunOut" | Set-Content -Encoding utf8 (Join-Path $RunOut 'run_inputs.txt')
```

The new worktree writes normalized tables, a prepared cohort, checkpoints, logs, and reports under `$RunOut`. The raw tables remain in the old clone. Do not copy old prepared cohorts or checkpoints into this output: their source/configuration signatures may not match. Do not use the tracked historical `hammertime_100/best.pt` as the GNNv2 checkpoint. No FastF1 account credentials are needed in local-data-only mode.

## Run the 2023 FastF1 season workflow

From the cloned repository root, run the GPU overlay and explicit GNNv2 config:

```powershell
$env:PYTHONUTF8 = '1' # Optional: helps Windows terminals handle Unicode race/session names in logs.
& uv @GpuRun python hammertime_100/run_2023_fastf1_season.py `
    --config hammertime_100/configs/train_2023_fastf1_season_gnnv2.yaml `
    --output $RunOut `
    --device cuda `
    --epochs 30 `
    --window-stride 2 `
    --workers 1 `
    --processing-workers 1 `
    --torch-threads 1 `
    --local-data-only `
    --raw-sessions-dir $RawSessions `
    --log-every-batches 10 2>&1 | Tee-Object -FilePath $RunLog
if ($LASTEXITCODE -ne 0) { throw "Season run failed; inspect $RunLog and $(Join-Path $RunOut 'run_failure.json')" }
```

The config's batch size of 256 is used because `--batch-size` is omitted. `--processing-workers 1` controls independent CPU normalization processes; `--workers 1` controls source validation/export in online mode (FastF1 acquisition is serialized); `--torch-threads 1` controls PyTorch CPU intra-op math. These are separate controls. The GNNv2 runbook uses one normalization process as the conservative starting point for a 16 GB host. The new code logs requested/effective normalization workers and available RAM; on Windows it caps processes using current free physical memory and defaults to one process if the probe fails. Do not raise the worker count while RAM is exhausted. If training reports CUDA out-of-memory, retry with `--batch-size 128`, then 64 if needed. A changed batch size creates a different training fingerprint and a distinct output under `training`; record the value used.

The runner verifies all 22 local source manifests, preprocesses the raw tables, applies existing coverage checks, prepares the graph cohort, trains, evaluates on the existing held-out split, and exports artifacts. It makes no FastF1 source requests in local-data-only mode. The first full run can take substantial time, particularly during season-data preparation; no duration or score is guaranteed. Training progress is logged on the first batch and every 10 batches after that. Check the run log and then inspect:

- `$RunOut\usable_sessions.json` and `failed_sessions.json` for race coverage;
- `$RunOut\training\<fingerprint>\best.pt` and `training_history.csv` for the selected validation checkpoint and history;
- `$RunOut\training\<fingerprint>\test_metrics.json` and `test_predictions.csv` for held-out results;
- `$RunOut\run_failure.json` if the run stops with a recorded failure.

There is no `--resume` flag. An interrupted run resumes automatically from `best.last.pt` only when the config, implementation, data, scaler, device type, thread count, and checkpoint path all match. Re-run the same command with the same `$RunOut` to continue it. Changing CPU to CUDA, changing output path or training flags, or using an older checkpoint creates a different training identity; the runner will not resume an incompatible checkpoint.

Reuse is checked separately at each stage. The runner still hashes and verifies the raw inventory. A normalized race stage is reused only if its recorded source and normalization fingerprints validate. Graph/cohort preparation is skipped only when the prepared artifact and manifest match the effective season config and normalized-source hash; an invalid older artifact is retained and a new rebuild directory is used. Training resumes from its last checkpoint only for a matching training fingerprint, while a verified completed run skips fitting. A changed stride, schema, config, source, device, or other fingerprinted option may require rebuilding one or more later stages. Use this season runner with the same validated GNNv2 config and protocol; a standalone ablation or arbitrary-config training command is not an equivalent continuation path.

This runbook does not report a completed RTX 4060 training run or measured performance. The GNNv2 model and config are implementation candidates; interpret the authentic held-out results only after the run completes and its coverage and acceptance gates are reviewed.

## References

- [uv installation for Windows](https://docs.astral.sh/uv/getting-started/installation/#windows)
- [uv project run options and dependency overlays](https://docs.astral.sh/uv/concepts/projects/run/)
- [Official PyTorch CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/)
- [Direct PyTorch 2.14.1 CUDA 13.0 Windows/Python 3.12 wheel](https://download-r2.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-win_amd64.whl)
