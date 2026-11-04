<#
.SYNOPSIS
GNNv4 launcher (Windows). Local data only: reads an existing verified normalized
source cache, builds/reuses a memory-mapped cohort_v4, trains and evaluates.

.DESCRIPTION
Device: CUDA when available, otherwise CPU (also when the CUDA probe fails or
CUDA runs out of memory during training). Existing outputs are never
overwritten. DataLoader workers default to 0 on Windows (spawn would re-open
every memory map per worker).

.EXAMPLE
powershell -ExecutionPolicy Bypass -File hammertime_100\tools\train_gnnv4.ps1 `
  -NormalizedSource D:\HammerTime\.local\research-history\2023-gnnv2-cpu\race_cache -PreflightOnly
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$NormalizedSource,

    [string]$RawSessions,

    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'auto',

    [int]$Epochs = 0,

    [string]$Resume,

    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,

    [switch]$PreflightOnly,

    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'

function Resolve-ExistingPath([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label does not exist: $Path" }
    return (Resolve-Path -LiteralPath $Path).Path
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required (https://docs.astral.sh/uv/).'
}

$RepoRoot = Resolve-ExistingPath $RepoRoot 'Repository'
$NormalizedSource = Resolve-ExistingPath $NormalizedSource 'Normalized source'
foreach ($Required in @('normalized_source_manifest.json', 'normalized_source_manifest.json.sha256', 'sessions')) {
    $null = Resolve-ExistingPath (Join-Path $NormalizedSource $Required) "Normalized-source input $Required"
}

Push-Location $RepoRoot
try {
    # Device probe (informational; the runner repeats it and falls back to CPU itself).
    $ProbeCode = @'
import torch
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print("cuda|" + torch.cuda.get_device_name(0) + "|" + str(round(free / 2**30, 1)))
else:
    print("cpu|-|0")
'@
    $Probe = 'cpu|-|0'
    try {
        $ProbeOutput = & uv run --frozen python -c $ProbeCode 2>$null
        if ($LASTEXITCODE -eq 0 -and $ProbeOutput) { $Probe = ($ProbeOutput | Select-Object -Last 1) }
    } catch {
        Write-Warning "CUDA probe failed ($($_.Exception.Message)); using CPU."
    }
    $ProbeParts = $Probe.Split('|')
    $ProbeDevice = $ProbeParts[0]
    if ($Device -eq 'cuda' -and $ProbeDevice -ne 'cuda') {
        Write-Warning 'CUDA requested but unavailable; falling back to CPU.'
    }
    $Effective = if ($ProbeDevice -eq 'cuda' -and $Device -ne 'cpu') { 'cuda' } else { 'cpu' }

    $Cores = [Environment]::ProcessorCount
    $Threads = [Math]::Max(1, $Cores - 2)
    if ($Effective -eq 'cuda') {
        $Batch = 256; $Accum = 1
        Write-Host "Device: CUDA $($ProbeParts[1]) ($($ProbeParts[2]) GiB free) | batch 256 x accum 1 | workers 0"
    } else {
        $Batch = 128; $Accum = 2
        Write-Host "Device: CPU | batch 128 x accum 2 | torch threads $Threads"
    }

    $History = Join-Path $RepoRoot '.local\research-history'
    New-Item -ItemType Directory -Force -Path $History | Out-Null
    $RunArgs = @()
    if ($Resume) {
        $Output = Resolve-ExistingPath $Resume 'Resume run directory'
        $RunArgs += '--resume'
    } else {
        $Suffix = if ($PreflightOnly) { '-preflight' } else { '' }
        $Output = Join-Path $History ("gnnv4$Suffix-" + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + "-$PID")
        if (Test-Path -LiteralPath $Output) { throw "Refusing to overwrite $Output" }
    }
    if ($PreflightOnly) { $RunArgs += '--preflight-only' }
    if ($Epochs -gt 0) { $RunArgs += @('--epochs', "$Epochs") }
    if ($RawSessions) { $RunArgs += @('--raw-sessions-dir', (Resolve-ExistingPath $RawSessions 'Raw sessions')) }
    $Log = "$Output.launcher.log"

    $env:OMP_NUM_THREADS = "$Threads"
    $env:MKL_NUM_THREADS = "$Threads"
    $env:OPENBLAS_NUM_THREADS = "$Threads"
    $env:PYTHONUTF8 = '1'

    $Arguments = @('run', '--frozen', 'python', 'hammertime_100\train_gnnv4.py',
        '--normalized-source', $NormalizedSource,
        '--config', 'hammertime_100\configs\train_2023_fastf1_season_gnnv4.yaml',
        '--output', $Output,
        '--device', $Effective,
        '--batch-size', "$Batch", '--grad-accum', "$Accum",
        '--workers', '0', '--threads', "$Threads",
        '--log-every-batches', '50') + $RunArgs + $ExtraArgs

    & uv @Arguments 2>&1 | Tee-Object -FilePath $Log
    $Status = $LASTEXITCODE
    switch ($Status) {
        0 { Write-Host "`nDone. Outputs: $Output (launcher log $Log)" }
        3 { Write-Error "Stopped by the memory preflight; see the message above." }
        default { Write-Error "Runner failed with exit code $Status; see $Log" }
    }
    exit $Status
} finally {
    Pop-Location
}
