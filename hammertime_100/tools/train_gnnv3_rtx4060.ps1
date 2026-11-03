param(
    [Parameter(Mandatory = $true)]
    [string]$RawSessions,

    [Parameter(Mandatory = $true)]
    [string]$ReusePreparedFrom,

    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,

    [string]$ProductionCheckpoint,

    [int]$TorchThreads = 8,

    [int]$ProcessingWorkers = 1,

    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'

function Resolve-ExistingDirectory([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label directory does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Invoke-Uv([string[]]$Arguments, [string]$LogPath) {
    $PreviousEnvironment = $env:UV_PROJECT_ENVIRONMENT
    Push-Location $RepoRoot
    try {
        $env:UV_PROJECT_ENVIRONMENT = $UvEnvironment
        & uv @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE. See log: $LogPath"
        }
    } finally {
        if ($null -eq $PreviousEnvironment) {
            Remove-Item Env:\UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
        } else {
            $env:UV_PROJECT_ENVIRONMENT = $PreviousEnvironment
        }
        Pop-Location
    }
}

function Invoke-UvSetup([string[]]$Arguments) {
    $PreviousEnvironment = $env:UV_PROJECT_ENVIRONMENT
    Push-Location $RepoRoot
    try {
        $env:UV_PROJECT_ENVIRONMENT = $UvEnvironment
        & uv @Arguments
        if ($LASTEXITCODE -ne 0) { throw "uv setup command failed with exit code $LASTEXITCODE." }
    } finally {
        if ($null -eq $PreviousEnvironment) {
            Remove-Item Env:\UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
        } else {
            $env:UV_PROJECT_ENVIRONMENT = $PreviousEnvironment
        }
        Pop-Location
    }
}

$RepoRoot = Resolve-ExistingDirectory $RepoRoot 'Repository'
$RawSessions = Resolve-ExistingDirectory $RawSessions 'Raw sessions'
$ReusePreparedFrom = Resolve-ExistingDirectory $ReusePreparedFrom 'Prepared-cache run'

$RequiredPaths = @(
    (Join-Path $ReusePreparedFrom 'race_cache\normalized_source_manifest.json'),
    (Join-Path $ReusePreparedFrom 'race_cache\normalized_source_manifest.json.sha256'),
    (Join-Path $ReusePreparedFrom 'race_cache\sessions'),
    (Join-Path $ReusePreparedFrom 'cohorts'),
    (Join-Path $RepoRoot 'hammertime_100\configs\train_2023_fastf1_season_gnnv2.yaml'),
    (Join-Path $RepoRoot 'hammertime_100\configs\gnnv3_cpu_overnight_overrides.yaml')
)
foreach ($Path in $RequiredPaths) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required existing prepared-cache input is missing: $Path. This launcher will not preprocess a replacement cohort."
    }
}

$RaceDirs = @(Get-ChildItem -LiteralPath $RawSessions -Directory | Where-Object { -not $_.Name.StartsWith('.') })
if ($RaceDirs.Count -ne 22) {
    throw "Expected 22 authentic raw race directories under $RawSessions; found $($RaceDirs.Count)."
}
foreach ($RaceDir in $RaceDirs) {
    foreach ($FileName in @('raw_manifest.json', 'raw_laps.parquet', 'raw_telemetry.parquet')) {
        if (-not (Test-Path -LiteralPath (Join-Path $RaceDir.FullName $FileName) -PathType Leaf)) {
            throw "Required file $FileName is missing from $($RaceDir.FullName)."
        }
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is not on PATH. Install uv for Windows and open a new PowerShell window.'
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git is not on PATH. Install Git for Windows and open a new PowerShell window.'
}

$RepoBranch = (& git -C $RepoRoot branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $RepoBranch -ne 'dev/gnnv3') {
    throw "Expected branch dev/gnnv3 in $RepoRoot; found '$RepoBranch'."
}

$TorchCudaWheel = 'https://download-r2.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp312-cp312-win_amd64.whl'
$UvEnvironment = Join-Path $RepoRoot '.venv-gnnv3-win'
$PythonExe = Join-Path $UvEnvironment 'Scripts\python.exe'
Write-Host 'Syncing the locked Python 3.12 environment in this new worktree.'
Push-Location $RepoRoot
try {
    & uv python install 3.12
    if ($LASTEXITCODE -ne 0) { throw 'uv could not install Python 3.12.' }
} finally {
    Pop-Location
}
Invoke-UvSetup @('sync', '--frozen', '--python', '3.12')
Write-Host 'Applying the verified PyTorch CUDA wheel to this worktree only.'
& uv pip install --python $PythonExe --no-deps $TorchCudaWheel
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch wheel installation failed.' }
$GpuRun = @('run', '--frozen', '--no-sync', '--python', '3.12')
$GpuCheck = @'
import torch, torch_geometric
assert torch.cuda.is_available(), "CUDA is unavailable; refusing CPU fallback"
assert torch.version.cuda is not None, "Installed PyTorch wheel is CPU-only"
name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
print(f"torch={torch.__version__}; torch CUDA={torch.version.cuda}; PyG={torch_geometric.__version__}")
print(f"GPU={name}; VRAM={props.total_memory / 1024**3:.1f} GiB")
x = torch.ones((128, 128), device="cuda")
torch.cuda.synchronize()
print(f"CUDA smoke test={float((x @ x)[0, 0])}")
'@
Write-Host 'Checking the locked Windows dependencies and CUDA-enabled PyTorch overlay.'
& uv @GpuRun python -c $GpuCheck
if ($LASTEXITCODE -ne 0) {
    throw 'CUDA/GPU readiness check failed. No preprocessing or training has started.'
}

$RunStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$RunRoot = Join-Path $RepoRoot '.local\research-history'
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$RunOut = Join-Path $RunRoot "gnnv3-rtx4060-$RunStamp-$PID"
$RunLog = "$RunOut.log"
$PreflightOut = "$RunOut-preflight"
$PreflightLog = "$PreflightOut.log"
foreach ($Candidate in @($RunOut, $RunLog, $PreflightOut, $PreflightLog)) {
    if (Test-Path -LiteralPath $Candidate) { throw "Refusing to overwrite existing output: $Candidate" }
}

$CommonArgs = @(
    'hammertime_100/run_2023_fastf1_season.py',
    '--config', 'hammertime_100/configs/train_2023_fastf1_season_gnnv2.yaml',
    '--reuse-prepared-from', $ReusePreparedFrom,
    '--device', 'cuda',
    '--epochs', '30',
    '--window-stride', '2',
    '--workers', '1',
    '--processing-workers', "$ProcessingWorkers",
    '--torch-threads', "$TorchThreads",
    '--local-data-only',
    '--raw-sessions-dir', $RawSessions,
    '--require-existing-caches',
    '--training-overrides', 'hammertime_100/configs/gnnv3_cpu_overnight_overrides.yaml',
    '--log-every-batches', '10'
)

$ProductionArgs = @()
if ($ProductionCheckpoint) {
    $ProductionCheckpoint = (Resolve-Path -LiteralPath $ProductionCheckpoint).Path
    if ((Split-Path -Leaf $ProductionCheckpoint) -ne 'best.pt') {
        throw 'ProductionCheckpoint must point to the selected best.pt file.'
    }
    $ProductionArgs = @('--preserve-production-checkpoint', $ProductionCheckpoint)
} else {
    $ProductionArgs = @('--no-production-checkpoint')
    Write-Host 'No user production checkpoint was supplied; the isolated runner is explicitly told not to expect one.'
    $OldTrainingRoot = Join-Path $ReusePreparedFrom 'training'
    $OldCheckpoints = @()
    if (Test-Path -LiteralPath $OldTrainingRoot -PathType Container) {
        $OldCheckpoints = @(Get-ChildItem -LiteralPath $OldTrainingRoot -Filter 'best.pt' -File -Recurse -ErrorAction SilentlyContinue)
    }
    if ($OldCheckpoints.Count -gt 0) {
        Write-Host 'Existing candidate checkpoints in the source tree are left untouched:'
        $OldCheckpoints | ForEach-Object { Write-Host "  $($_.FullName)" }
    }
}

$PreflightArgs = @(
    $CommonArgs + @('--output', $PreflightOut) + $ProductionArgs + @('--preflight-only')
)
Write-Host 'Verifying the exact prepared cohort and constructing the GNNv3 candidate without training or preprocessing.'
Invoke-Uv ($GpuRun + @('python') + $PreflightArgs) $PreflightLog
if ($PreflightOnly) {
    Write-Host "Preflight passed. Log: $PreflightLog"
    exit 0
}

$TrainArgs = @($CommonArgs + @('--output', $RunOut) + $ProductionArgs)
Write-Host 'Preflight passed; beginning the isolated RTX 4060 training run.'
Invoke-Uv ($GpuRun + @('python') + $TrainArgs) $RunLog
Write-Host "Training completed. Output: $RunOut"
Write-Host "Training log: $RunLog"
Write-Host "Preflight output: $PreflightOut"
