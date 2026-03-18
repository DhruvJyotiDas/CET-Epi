# Evaluate trained CET-Epi model
# Run emergence analysis and generate reports

param(
    [Parameter(Mandatory=$true)]
    [string]$CheckpointPath,
    
    [string]$ConfigPath = "..\configs\chickenpox.yaml",
    
    [switch]$AnalyzeEI,
    [switch]$SimulateIntervention,
    [switch]$GenerateFigures
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $ProjectRoot

Write-Host "CET-Epi Model Evaluation" -ForegroundColor Cyan
Write-Host "Checkpoint: $CheckpointPath" -ForegroundColor White

# Create evaluation script on-the-fly
$EvalScript = @"
import sys
sys.path.append('$ProjectRoot')

import torch
import pickle
from pathlib import Path

from src.models.cet_epi import CET_Epi
from src.data.chickenpox_loader import MultiScaleChickenpoxLoader
from src.evaluation.ei_analyzer import EIAnalyzer
from src.evaluation.intervention import InterventionSimulator
from src.evaluation.visualizer import CET_EpiVisualizer
from src.utils.config import load_config
from src.utils.gpu import setup_gpu

def main():
    # Setup
    device = setup_gpu()
    config = load_config('$ConfigPath', '$ProjectRoot/configs')
    
    # Load model
    checkpoint = torch.load('$CheckpointPath', map_location=device)
    model = CET_Epi(
        n_micro=config.data.micro_nodes,
        n_macro=config.data.macro_nodes,
        in_channels=config.data.features,
        hidden_dim=config.model.hidden_dim
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load data
    loader = MultiScaleChickenpoxLoader(lags=config.data.get('lags', 4))
    train_data, test_data = loader.get_split(config.data.train_ratio)
    
    results = {}
    
    # 1. EI Analysis
    if $AnalyzeEI:
        print("\n" + "="*60)
        print("EFFECTIVE INFORMATION ANALYSIS")
        print("="*60)
        
        analyzer = EIAnalyzer(model, device)
        ei_results = analyzer.analyze_over_time(test_data)
        results['ei_analysis'] = ei_results
        
        # Summary
        avg_emergence = sum(r['emergence_score'] for r in ei_results) / len(ei_results)
        positive_emergence = sum(1 for r in ei_results if r['emergence_score'] > 0)
        print(f"\nAverage Emergence Score: {avg_emergence:.4f}")
        print(f"Timesteps with positive emergence: {positive_emergence}/{len(ei_results)}")
        
        # Visualize
        save_path = Path('$ProjectRoot/logs/figures/emergence_analysis.png')
        analyzer.plot_emergence_analysis(ei_results, save_path=str(save_path))
        
        # Partition visualization
        partition_save = Path('$ProjectRoot/logs/figures/learned_partition.png')
        analyzer.visualize_partition(save_path=str(partition_save))
    
    # 2. Intervention Simulation
    if $SimulateIntervention:
        print("\n" + "="*60)
        print("INTERVENTION SIMULATION")
        print("="*60)
        
        simulator = InterventionSimulator(model, device)
        
        # Get a test sample
        test_snapshot = next(iter(test_data))
        x = test_snapshot.x.to(device)
        edge_index = test_snapshot.edge_index.to(device)
        edge_attr = test_snapshot.edge_attr.to(device) if test_snapshot.edge_attr is not None else None
        
        # Test different strategies
        strategies = {
            'high_lockdown_center': {'nodes': [0, 1, 2, 3, 4], 'effect': 0.8},
            'medium_lockdown_center': {'nodes': [0, 1, 2, 3, 4], 'effect': 0.5},
            'low_lockdown_center': {'nodes': [0, 1, 2, 3, 4], 'effect': 0.2},
            'high_lockdown_periphery': {'nodes': [15, 16, 17, 18, 19], 'effect': 0.8},
        }
        
        comparison = simulator.compare_intervention_strategies(
            x, edge_index, strategies, edge_attr
        )
        
        results['intervention_comparison'] = comparison
        
        print("\nIntervention Strategy Comparison:")
        print("-" * 60)
        for name, metrics in comparison.items():
            print(f"{name:25s} | Micro: {metrics['micro_reduction']*100:5.1f}% | "
                  f"Macro: {metrics['macro_reduction']*100:5.1f}% | "
                  f"Ratio: {metrics['propagation_ratio']:.3f}")
        
        # Detailed report for best strategy
        best_strategy = max(comparison.items(), key=lambda x: x[1]['macro_reduction'])
        print(f"\nDetailed analysis of best strategy: {best_strategy[0]}")
        detailed = simulator.simulate_intervention(
            x, edge_index, edge_attr,
            intervention_nodes=strategies[best_strategy[0]]['nodes'],
            intervention_effect=strategies[best_strategy[0]]['effect']
        )
        report = simulator.generate_intervention_report(detailed)
        print(report)
    
    # 3. Generate comprehensive figures
    if $GenerateFigures:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        visualizer = CET_EpiVisualizer()
        
        # Get predictions for all test data
        all_preds = []
        all_targets = []
        
        for snapshot in test_data:
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device) if snapshot.edge_attr is not None else None
            
            pred, _, int_dict = model(x, edge_index, edge_attr, return_all=True)
            all_preds.append(pred.cpu())
            all_targets.append(snapshot.y.unsqueeze(-1).unsqueeze(-1))
        
        predictions = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Prediction plots
        visualizer.plot_predictions(predictions, targets, n_samples=5,
                                   save_name="prediction_samples.png")
        
        # Assignment matrix
        # Use last snapshot's assignment
        last_S = int_dict['S']
        visualizer.plot_assignment_matrix(last_S, save_name="final_assignment.png")
        
        # Scale comparison
        visualizer.plot_scale_comparison(int_dict['h_micro'], int_dict['h_macro'],
                                       save_name="scale_features.png")
        
        print("All figures saved to logs/figures/")
    
    # Save results
    results_path = Path('$ProjectRoot/logs/evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
"@

# Save and run evaluation script
$TempScript = "$env:TEMP\cet_epi_eval.py"
Set-Content -Path $TempScript -Value $EvalScript

Write-Host "Running evaluation with options:" -ForegroundColor Yellow
if ($AnalyzeEI) { Write-Host "  - EI Analysis" }
if ($SimulateIntervention) { Write-Host "  - Intervention Simulation" }
if ($GenerateFigures) { Write-Host "  - Figure Generation" }

python $TempScript

Remove-Item $TempScript -Force