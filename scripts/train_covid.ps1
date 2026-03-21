# Train CET-Epi on COVID-19 Dataset
# For MI300X with large-scale data

param(
    [string]$Country = "italy",
    [int]$Epochs = 100,
    [int]$HiddenDim = 128
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $ProjectRoot

# Create config on-the-fly
$ConfigContent = @"
# Auto-generated COVID-19 config
inherit: base.yaml

data:
  name: "covid19_$Country"
  country: "$Country"
  micro_nodes: $(if ($Country -eq "italy") { 107 } else { 313 })
  macro_nodes: $(if ($Country -eq "italy") { 21 } else { 9 })
  features: 4
  train_ratio: 0.8
  lags: 14  # 2 weeks history

model:
  hidden_dim: $HiddenDim
  horizon: 1
  num_layers: 3
  ceo:
    n_macro_nodes: $(if ($Country -eq "italy") { 21 } else { 9 })
    ei_weight: 0.15
    sparsity_weight: 0.005
    balance_weight: 0.01

training:
  epochs: $Epochs
  learning_rate: 0.0005
  batch_size: 1
  gradient_clip: 1.0
  early_stopping: 30

logging:
  checkpoint_interval: 20
  val_interval: 1
"@

$ConfigPath = "$ProjectRoot\configs\covid19_$Country.yaml"
Set-Content -Path $ConfigPath -Value $ConfigContent

Write-Host "Training CET-Epi on COVID-19 $Country dataset" -ForegroundColor Green
Write-Host "Config: $ConfigPath" -ForegroundColor Cyan

# Run training
python -m src.training.trainer --config $ConfigPath

# Cleanup config
Remove-Item $ConfigPath -Force
