# Run GNNv2 on native Windows with an RTX 4060

This runbook transfers the pushed `codex/gnnv2` branch to a Windows laptop and runs the 2023 FastF1 season workflow with the RTX 4060. It uses PowerShell and `uv`. It does not require copying a Python virtual environment, and it keeps old research artifacts separate from the new run.

The target revision is `b4da63a92851a3e405466c021036251de3e26b24` (`b4da63a`). The GNNv2 YAML sets `spatial_layers: 3`, `own_car_branch: true`, `own_car_hidden_dim: 32`, and batch size 256. The season runner's CLI defaults override two YAML values: it defaults to 25 epochs and window stride 1. The run command below passes `--epochs 30 --window-stride 2` to retain the config's settings.

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
git clone --branch codex/gnnv2 --single-branch https://github.com/andromeda7063/HammerTime.git $Repo
Set-Location $Repo
$RunOut = Join-Path $Repo '.local\research-history\2023-fastf1-gnnv2-windows'
git rev-parse --short HEAD
```

The last command should print `b4da63a` unless the branch has since advanced.

## Install the locked Python environment

The project requires Python 3.12 or newer; the pinned Windows CUDA wheel below is for Python 3.12. Install the base locked environment with:

```powershell
uv python install 3.12
uv sync --frozen --python 3.12
```

On native Windows, the PyTorch wheel selected by `uv.lock` is the standard Windows wheel, which does not enable CUDA in this project setup. For the GPU run, create a per-command overlay with the official PyTorch 2.14.1 CUDA 13.0 wheel and the FastF1 version required by the season runner. This leaves `uv.lock` unchanged:

```powershell
$TorchCuda = 'torch @ https://download-r2.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-win_amd64.whl'
$GpuRun = @(
    'run', '--frozen', '--python', '3.12',
    '--with', 'fastf1==3.9.0',
    '--with', $TorchCuda
)
```

Keep this PowerShell session open through the run. If you open another session, recreate `$TorchCuda` and `$GpuRun` there.

The extra FastF1 version is needed because the lock currently pins 3.8.3 while `run_2023_fastf1_season.py` requires FastF1 3.9.0 or newer for account authentication. The CUDA overlay uses the same PyTorch release as the lock but the official `cu130` Windows build. The wheel is about 2 GB, so allow time and disk space for uv to cache it. Do not use `uv pip install` to mutate the environment or `--torch-backend` here; the overlay is applied to each `uv run` command.

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

## Transfer reusable data from the pendrive

The workflow needs FastF1 credentials and network access even when caches are present, because it authenticates before event discovery. Enter the credentials for the FastF1 account in the current PowerShell session; the password is read without echoing and is not put in a command literal:

```powershell
$env:F1_USERNAME = Read-Host 'FastF1 username'
$SecurePassword = Read-Host 'FastF1 password' -AsSecureString
$env:F1_PASSWORD = [System.Net.NetworkCredential]::new('', $SecurePassword).Password
```

Do not copy `.env` or saved credentials from the old laptop. The runner reads `F1_USERNAME` and `F1_PASSWORD` from the process environment or a repository-root `.env`; it sets the FastF1 authentication method internally. Clear the temporary environment values when finished:

```powershell
Remove-Item Env:F1_USERNAME -ErrorAction SilentlyContinue
Remove-Item Env:F1_PASSWORD -ErrorAction SilentlyContinue
```

The copied FastF1 cache and the old season output are ignored by Git. Paths observed on the source machine:

- `hammertime_100/data/cache/fastf1-2023-authenticated` — about 2.5 GB; this is the runner's FastF1 cache location relative to the cloned repository.
- `.local/research-history/2023-fastf1-season/race_cache/raw_sessions` — about 1.1 GB; raw per-race FastF1 tables and hash manifests can be reused when their recorded FastF1 version and hashes match.
- The full old `.local/research-history/2023-fastf1-season` directory — about 11 GB, including old normalized tables, prepared cohorts, checkpoints, and reports.

To reuse the FastF1 cache, set `$UsbRoot` to the directory on the pendrive that contains the old repository's `hammertime_100` and `.local` folders, then copy the cache to the runner's expected repository-relative path:

```powershell
$UsbRoot = 'E:\HammerTime' # Change this to the pendrive path.
$FastF1CacheSource = Join-Path $UsbRoot 'hammertime_100\data\cache\fastf1-2023-authenticated'
$FastF1CacheTarget = Join-Path $Repo 'hammertime_100\data\cache\fastf1-2023-authenticated'
if (Test-Path $FastF1CacheSource) {
    robocopy $FastF1CacheSource $FastF1CacheTarget /E /COPY:DAT /DCOPY:DAT /R:1 /W:1
}
```

To reuse raw per-race tables, copy only the raw stage into the new output directory. This keeps the existing archive intact and gives the new run its own log, prepared cohort, and training outputs:

```powershell
$RawSource = Join-Path $UsbRoot '.local\research-history\2023-fastf1-season\race_cache\raw_sessions'
$RawTarget = Join-Path $RunOut 'race_cache\raw_sessions'
if (Test-Path $RawSource) {
    robocopy $RawSource $RawTarget /E /COPY:DAT /DCOPY:DAT /R:1 /W:1
}
```

`robocopy` exit codes 0–7 are success/informational; 8 or higher means a copy failure. If the pendrive backup has a different root, change `$UsbRoot`; if these source paths are absent, skip the copies and let FastF1 reacquire data. Keep the complete old output as an archive if desired, but do not copy its `.venv` or `.env`.

The raw stage is the useful transferable cache. Normalized stage manifests record absolute source paths, and prepared cohort manifests validate their absolute artifact path and configuration/source hashes. The old prepared cohorts (about 8.6 GB total) therefore will be rebuilt or rejected after transfer; the old training checkpoints also do not match the new three-layer/own-car configuration and CUDA run identity. Do not use the tracked historical `hammertime_100/best.pt` as the GNNv2 checkpoint.

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
    --torch-threads 1
```

The config's batch size of 256 is used because `--batch-size` is omitted. If training reports CUDA out-of-memory, retry with `--batch-size 128`, then 64 if needed. A changed batch size creates a different training fingerprint and a distinct output under `training`; record the value used. The conservative worker counts keep CPU preprocessing/export parallelism low while the model trains on the GPU.

The runner performs calendar discovery, authentic data acquisition or raw-cache reuse, preprocessing, coverage checks, graph preparation, training, held-out evaluation, and exports. The first full run can take substantial time, particularly for the season data preparation; no duration or score is guaranteed. Check the run log and then inspect:

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
