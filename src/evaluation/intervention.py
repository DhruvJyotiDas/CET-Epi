"""
Counterfactual intervention simulation for CET-Epi.
Test 'what-if' scenarios: lockdowns, vaccination campaigns, etc.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class InterventionSimulator:
    """
    Simulate interventions and propagate effects across scales.
    
    Key feature: Micro interventions (individual behavior) should
    correctly propagate to macro effects (R0 changes).
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    @torch.no_grad()
    def simulate_intervention(self,
                            baseline_x: torch.Tensor,
                            edge_index: torch.Tensor,
                            edge_weight: torch.Tensor = None,
                            intervention_nodes: List[int] = None,
                            intervention_effect: float = 0.5,
                            horizon: int = 4) -> Dict:
        """
        Simulate intervention at specific nodes.
        
        Args:
            baseline_x: [N, T, F] baseline features
            edge_index: Graph connectivity
            edge_weight: Edge weights
            intervention_nodes: List of node indices to intervene on
            intervention_effect: Effect strength (0=none, 1=full suppression)
            horizon: Prediction horizon
        
        Returns:
            Dictionary with baseline, intervention, and effect metrics
        """
        # Baseline prediction
        baseline_pred, _, baseline_int = self.model(
            baseline_x, edge_index, edge_weight, return_all=True
        )
        
        # Apply intervention: reduce features at intervention nodes
        intervention_x = baseline_x.clone()
        if intervention_nodes is not None:
            # Reduce transmission-related features (assuming feature 0 is cases)
            intervention_x[intervention_nodes, -1, 0] *= (1 - intervention_effect)
        
        # Intervention prediction
        intervention_pred, _, intervention_int = self.model(
            intervention_x, edge_index, edge_weight, return_all=True
        )
        
        # Compute effects at both scales
        results = {
            'baseline_micro': baseline_pred.cpu(),
            'intervention_micro': intervention_pred.cpu(),
            'effect_micro': (baseline_pred - intervention_pred).cpu(),
            
            # Macro-scale effects
            'baseline_macro': self._aggregate_to_macro(
                baseline_pred, baseline_int['S']
            ).cpu(),
            'intervention_macro': self._aggregate_to_macro(
                intervention_pred, intervention_int['S']
            ).cpu(),
            
            # Cross-scale propagation metrics
            'micro_reduction': self._compute_reduction(baseline_pred, intervention_pred),
            'macro_reduction': None,  # Computed below
            'propagation_ratio': None,  # Computed below
            
            'intervention_nodes': intervention_nodes,
            'affected_macro_regions': self._get_affected_macros(
                intervention_int['S'], intervention_nodes
            )
        }
        
        # Compute macro effects
        results['macro_reduction'] = self._compute_reduction(
            results['baseline_macro'], results['intervention_macro']
        )
        
        # Propagation ratio: how much micro effect transfers to macro
        if results['macro_reduction'] > 0:
            results['propagation_ratio'] = (
                results['macro_reduction'] / (results['micro_reduction'] + 1e-10)
            )
        
        return results
    
    def _aggregate_to_macro(self, micro_pred: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Aggregate micro predictions to macro scale."""
        # S: [N_micro, N_macro]
        # micro_pred: [N_micro, horizon, 1]
        N_macro = S.shape[1]
        
        # Weighted average by assignment
        weights = S / (S.sum(dim=0, keepdim=True).t() + 1e-10)  # [N_micro, N_macro]
        # micro_pred -> [N_micro, horizon]
        micro_flat = micro_pred.squeeze(-1)  # [N_micro, horizon]
        
        macro_pred = torch.mm(weights.t(), micro_flat)  # [N_macro, horizon]
        return macro_pred.unsqueeze(-1)  # [N_macro, horizon, 1]
    
    def _compute_reduction(self, baseline: torch.Tensor, intervention: torch.Tensor) -> float:
        """Compute percentage reduction."""
        baseline_total = baseline.sum().item()
        intervention_total = intervention.sum().item()
        if baseline_total > 0:
            return (baseline_total - intervention_total) / baseline_total
        return 0.0
    
    def _get_affected_macros(self, S: torch.Tensor, intervention_nodes: List[int]) -> List[int]:
        """Which macro regions contain intervention nodes?"""
        if intervention_nodes is None:
            return []
        affected = S[intervention_nodes].argmax(dim=1).unique().cpu().tolist()
        return affected
    
    def compare_intervention_strategies(self,
                                       baseline_x: torch.Tensor,
                                       edge_index: torch.Tensor,
                                       strategies: Dict[str, Dict],
                                       edge_weight: torch.Tensor = None) -> Dict:
        """
        Compare multiple intervention strategies.
        
        Strategies example:
        {
            'lockdown_high': {'nodes': [0,1,2], 'effect': 0.8},
            'lockdown_low': {'nodes': [0,1,2], 'effect': 0.3},
            'vaccination': {'nodes': [5,6,7], 'effect': 0.6}
        }
        """
        results = {}
        
        for name, params in strategies.items():
            print(f"Simulating: {name}")
            result = self.simulate_intervention(
                baseline_x, edge_index, edge_weight,
                intervention_nodes=params['nodes'],
                intervention_effect=params['effect']
            )
            results[name] = {
                'micro_reduction': result['micro_reduction'],
                'macro_reduction': result['macro_reduction'],
                'propagation_ratio': result['propagation_ratio'],
                'affected_regions': len(result['affected_macro_regions'])
            }
        
        return results
    
    def generate_intervention_report(self, results: Dict) -> str:
        """Generate human-readable intervention analysis."""
        report = []
        report.append("=" * 60)
        report.append("INTERVENTION SIMULATION REPORT")
        report.append("=" * 60)
        
        report.append(f"\nIntervention Nodes: {results['intervention_nodes']}")
        report.append(f"Affected Macro Regions: {results['affected_macro_regions']}")
        
        report.append(f"\nMICRO-SCALE EFFECTS:")
        report.append(f"  Baseline total cases: {results['baseline_micro'].sum():.2f}")
        report.append(f"  Intervention total: {results['intervention_micro'].sum():.2f}")
        report.append(f"  Reduction: {results['micro_reduction']*100:.1f}%")
        
        report.append(f"\nMACRO-SCALE EFFECTS:")
        report.append(f"  Baseline total: {results['baseline_macro'].sum():.2f}")
        report.append(f"  Intervention total: {results['intervention_macro'].sum():.2f}")
        report.append(f"  Reduction: {results['macro_reduction']*100:.1f}%")
        
        if results['propagation_ratio'] is not None:
            report.append(f"\nCROSS-SCALE PROPAGATION:")
            report.append(f"  Propagation Ratio: {results['propagation_ratio']:.3f}")
            report.append(f"  (1.0 = perfect propagation, >1 = amplification)")
        
        report.append("=" * 60)
        return "\n".join(report)