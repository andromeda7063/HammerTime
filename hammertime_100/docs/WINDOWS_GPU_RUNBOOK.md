# Run GNNv2 on native Windows with an RTX 4060

This runbook transfers the pushed `dev/gnnv2` branch and authentic raw race tables to a native Windows laptop, then runs the 2023 GNNv2 season workflow with an RTX 4060. It uses PowerShell and `uv`. Package installation may use internet access; the local-data-only run does not authenticate to FastF1, discover the calendar, or request race data.

Clone the current `dev/gnnv2` branch. The GNNv2 YAML sets `spatial_layers: 3`, `own_car_branch: true`, `own_car_hidden_dim: 32`, and batch size 256. The season runner defaults to 25 epochs and window stride 1, so the command below passes `--epochs 30 --window-stride 2` to match the config.

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

## Clone the pushed branch

Use a short local path to keep generated file paths manageable. From PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path 'C:\F1' | Out-Null
$Repo = 'C:\F1\HammerTime'
git clone --branch dev/gnnv2 --single-branch https://github.com/andromeda7063/HammerTime.git $Repo
Set-Location $Repo
$RunOut = Join-Path $Repo '.local\research-history\2023-fastf1-gnnv2-windows'
git branch --show-current
```

The last command should print `dev/gnnv2`.

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

## Transfer the authentic raw race tables

Only the raw-session archive is needed from the source laptop. It contains about 1.1 GB of per-race Parquet tables and `raw_manifest.json` files. Do not copy `.env`, credentials, the FastF1 HTTP cache, old prepared cohorts, or old checkpoints. Copy this archive from a trusted source run: the runner checks manifest FastF1 provenance, event identity, round-number scheme, byte counts, and SHA256 hashes to detect transfer corruption, but the manifests are not cryptographic attestations of data origin. The raw tables must cover every race in the 22-race registry. It accepts the source archive's consistent completed-race rounds 1–22 or a consistent published-calendar numbering (1–5, then 7–23). Missing, corrupt, duplicate/inconsistent, or unregistered sessions stop the run; the program never falls back to a FastF1 request.

Set `$UsbRoot` to the pendrive folder containing the old `.local` directory, then copy raw sessions into this run's output tree:

```powershell
$UsbRoot = 'E:\HammerTime' # Change this to the pendrive path.
$RawSource = Join-Path $UsbRoot '.local\research-history\2023-fastf1-season\race_cache\raw_sessions'
$RawTarget = Join-Path $RunOut 'race_cache\raw_sessions'
if (-not (Test-Path $RawSource)) { throw "Raw source directory is missing: $RawSource" }
robocopy $RawSource $RawTarget /E /COPY:DAT /DCOPY:DAT /R:1 /W:1
if ($LASTEXITCODE -ge 8) { throw "Raw-session copy failed with robocopy exit code $LASTEXITCODE" }
```

`robocopy` exit codes 0–7 are success/informational. If the pendrive backup has a different root, change `$UsbRoot`. A complete raw archive is required; the run will not proceed with partial coverage.

The target creates new normalized tables, prepared cohort, checkpoints, and reports under `$RunOut`. Do not transfer old prepared cohorts or checkpoints; their absolute paths and earlier model/configuration signatures do not match this run. Do not use the tracked historical `hammertime_100/best.pt` as the GNNv2 checkpoint. No FastF1 account credentials are needed in local-data-only mode.

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
    --raw-sessions-dir $RawTarget `
    --log-every-batches 10
```

The config's batch size of 256 is used because `--batch-size` is omitted. If training reports CUDA out-of-memory, retry with `--batch-size 128`, then 64 if needed. A changed batch size creates a different training fingerprint and a distinct output under `training`; record the value used. The conservative worker counts keep CPU preprocessing/export parallelism low while the model trains on the GPU.

The runner verifies all 22 local source manifests, preprocesses the raw tables, applies existing coverage checks, prepares the graph cohort, trains, evaluates on the existing held-out split, and exports artifacts. It makes no FastF1 source requests in local-data-only mode. The first full run can take substantial time, particularly during season-data preparation; no duration or score is guaranteed. Training progress is logged on the first batch and every 10 batches after that. Check the run log and then inspect:

- `$RunOut\usable_sessions.json` and `failed_sessions.json` for race coverage;
- `$RunOut\training\<fingerprint>\best.pt` and `training_history.csv` for the selected validation checkpoint and history;
- `$RunOut\training\<fingerprint>\test_metrics.json` and `test_predictions.csv` for held-out results;
- `$RunOut\run_failure.json` if the run stops with a recorded failure.

There is no `--resume` flag. An interrupted run resumes automatically from `best.last.pt` only when the config, implementation, data, scaler, device type, thread count, and checkpoint path all match. Re-run the same command with the same `$RunOut` to continue it. Changing CPU to CUDA, changing output path or training flags, or using an older checkpoint creates a different training identity; the runner will not resume an incompatible checkpoint.

This runbook does not report a completed RTX 4060 training run or measured performance. The GNNv2 model and config are implementation candidates; interpret the authentic held-out results only after the run completes and its coverage and acceptance gates are reviewed.

## References

- [uv installation for Windows](https://docs.astral.sh/uv/getting-started/installation/#windows)
- [uv project run options and dependency overlays](https://docs.astral.sh/uv/concepts/projects/run/)
- [Official PyTorch CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/)
- [Direct PyTorch 2.14.1 CUDA 13.0 Windows/Python 3.12 wheel](https://download-r2.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-win_amd64.whl)
