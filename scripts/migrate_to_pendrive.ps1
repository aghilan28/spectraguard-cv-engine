# =============================================================================
# SPECTRAGUARD - Pendrive Migration Script
# =============================================================================
# This script copies the VIRAT dataset and all data artifacts to a pendrive
# for migration to a new laptop.
#
# USAGE:
#   .\scripts\migrate_to_pendrive.ps1 -DriveLetter "F"
#
# The script preserves the full directory structure so you can simply
# copy the folder back to the same location on the new laptop.
# =============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$DriveLetter
)

$ErrorActionPreference = "Stop"

# Validate drive exists
$drivePath = "${DriveLetter}:"
if (-not (Test-Path $drivePath)) {
    Write-Host "ERROR: Drive $drivePath not found. Please check your pendrive is connected." -ForegroundColor Red
    exit 1
}

# Check available space on pendrive
$drive = Get-PSDrive -Name $DriveLetter -ErrorAction SilentlyContinue
if ($drive) {
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    Write-Host "Pendrive $drivePath has $freeGB GB free space" -ForegroundColor Cyan
    if ($freeGB -lt 55) {
        Write-Host "WARNING: You need at least 55 GB free. Current free: $freeGB GB" -ForegroundColor Yellow
        $confirm = Read-Host "Continue anyway? (y/n)"
        if ($confirm -ne "y") { exit 0 }
    }
}

$sourceRoot = "E:\SPECTRAGUARD\spectraguard-cv-engine"
$destRoot = "$drivePath\SPECTRAGUARD_MIGRATION\spectraguard-cv-engine"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "  SPECTRAGUARD Pendrive Migration" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Source: $sourceRoot" -ForegroundColor White
Write-Host "Destination: $destRoot" -ForegroundColor White
Write-Host ""

# ---- STEP 1: Copy the entire data/ directory (datasets, models, manifests, reports) ----
Write-Host "[1/2] Copying data/ directory (VIRAT dataset + models + reports)..." -ForegroundColor Yellow
Write-Host "       This is ~52.82 GB and will take some time..." -ForegroundColor DarkGray

$dataSource = Join-Path $sourceRoot "data"
$dataDest = Join-Path $destRoot "data"

if (Test-Path $dataSource) {
    # Use robocopy for reliable large file copy with progress
    # /E = copy subdirectories including empty ones
    # /R:3 = retry 3 times on failure
    # /W:5 = wait 5 seconds between retries
    # /MT:4 = multi-threaded (4 threads)
    # /ETA = show estimated time of arrival
    # /NP = no percentage (cleaner output for large transfers)
    robocopy $dataSource $dataDest /E /R:3 /W:5 /MT:4 /ETA /NJH
    
    if ($LASTEXITCODE -le 7) {
        Write-Host "       data/ directory copied successfully!" -ForegroundColor Green
    } else {
        Write-Host "       WARNING: Some files may have failed to copy (exit code: $LASTEXITCODE)" -ForegroundColor Yellow
    }
} else {
    Write-Host "       WARNING: data/ directory not found at $dataSource" -ForegroundColor Yellow
}

# ---- STEP 2: Copy the run_training logs (small files not in git) ----
Write-Host "[2/2] Copying training log files..." -ForegroundColor Yellow

$logFiles = @("run_training.log", "run_training_unique.log")
foreach ($logFile in $logFiles) {
    $src = Join-Path $sourceRoot $logFile
    if (Test-Path $src) {
        $dst = Join-Path $destRoot $logFile
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        Copy-Item $src $dst -Force
        Write-Host "       Copied: $logFile" -ForegroundColor DarkGray
    }
}

# ---- Summary ----
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "  MIGRATION COMPLETE!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Files on pendrive: $destRoot" -ForegroundColor Cyan
Write-Host ""
Write-Host "ON YOUR NEW LAPTOP, follow these steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Clone all 3 repos:" -ForegroundColor White
Write-Host "     git clone https://github.com/aghilan28/spectraguard-cv-engine.git" -ForegroundColor DarkGray
Write-Host "     git clone https://github.com/aghilan28/spectraguard-core-infra.git" -ForegroundColor DarkGray
Write-Host "     git clone https://github.com/lathika-mohan/spectroguard-frontend-refined.git" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  2. Copy data/ from pendrive into spectraguard-cv-engine/:" -ForegroundColor White
Write-Host "     robocopy <pendrive>\SPECTRAGUARD_MIGRATION\spectraguard-cv-engine\data spectraguard-cv-engine\data /E /MT:4" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  3. Set up Python environment:" -ForegroundColor White
Write-Host "     python -m venv .venv" -ForegroundColor DarkGray
Write-Host "     .\.venv\Scripts\Activate.ps1" -ForegroundColor DarkGray
Write-Host "     pip install -r requirements.txt" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  4. For frontend, install dependencies:" -ForegroundColor White
Write-Host "     cd spectroguard-frontend-refined && npm install" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  5. You are ready for M0.4!" -ForegroundColor Green
Write-Host ""
