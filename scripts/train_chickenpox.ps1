# Train CET-Epi on Hungary Chickenpox Dataset
# Optimized for MI300X

$ErrorActionPreference = "Stop"

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ConfigFile = "$ProjectRoot\configs\chickenpox.yaml"

# Environment setup
$env:PYTHONPATH = $ProjectRoot
$env:HIP_VISIBLE_DEVICES = "0"  # Use first MI300X
$env:PYTORCH_HIP_ALLOC_CONF = "max_split_size_mb:512"

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Cyan

if (-not (Test-Path $ConfigFile)) {
    Write-Error "Config file not found: $ConfigFile"
    exit 1
}

# Python check
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Error "Python not found in PATH"
    exit 1
}

# Check PyTorch and ROCm
Write-Host "Checking PyTorch/ROCm installation..." -ForegroundColor Cyan
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Create experiment directory
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ExpDir = "$ProjectRoot\experiments\chickenpox_cet_epi\$Timestamp"
New-Item -ItemType Directory -Force -Path $ExpDir | Out-Null

Write-Host "`nStarting CET-Epi training..." -ForegroundColor Green
Write-Host "Config: $ConfigFile" -ForegroundColor White
Write-Host "Experiment directory: $ExpDir" -ForegroundColor White

# Run training
python -m src.training.trainer --config $ConfigFile 2>&1 | Tee-Object "$ExpDir\training.log"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nTraining completed successfully!" -ForegroundColor Green
    Write-Host "Results saved to: $ExpDir" -ForegroundColor Green
} else {
    Write-Host "`nTraining failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}
