# Phase 1: Scale-aware prediction head
"""
Scale-aware prediction head for epidemic forecasting.
"""

import torch
import torch.nn as nn


class ScaleAwarePredictor(nn.Module):
    """
    Predicts future epidemic states using both micro and macro features.
    
    Strategy: Predict at macro scale, broadcast to micro, 
    then refine with micro-specific features.
    """
    
    def __init__(self, hidden_dim: int, out_dim: int = 1, horizon: int = 1):
        super().__init__()
        
        self.horizon = horizon
        
        # Macro prediction (coarse trend)
        self.macro_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * out_dim)
        )
        
        # Micro refinement (local correction)
        self.micro_refiner = nn.Sequential(
            nn.Linear(hidden_dim + horizon * out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * out_dim)
        )
        
    def forward(self, 
                h_micro: torch.Tensor,
                h_macro: torch.Tensor,
                assignment: torch.Tensor) -> torch.Tensor:
        """
        Predict future states.
        
        Args:
            h_micro: [N_micro, H] micro features
            h_macro: [N_macro, H] macro features
            assignment: [N_micro, N_macro] assignment
        
        Returns:
            predictions: [N_micro, horizon, out_dim]
        """
        # Predict at macro scale
        macro_pred = self.macro_predictor(h_macro)  # [N_macro, horizon*out]
        
        # Broadcast to micro scale
        micro_pred_coarse = torch.mm(assignment, macro_pred)  # [N_micro, horizon*out]
        
        # Refine with micro features
        combined = torch.cat([h_micro, micro_pred_coarse], dim=-1)
        micro_residual = self.micro_refiner(combined)  # [N_micro, horizon*out]
        
        # Final prediction: coarse + residual
        predictions = micro_pred_coarse + 0.1 * micro_residual
        
        # Reshape to [N_micro, horizon, out_dim]
        N = predictions.shape[0]
        predictions = predictions.view(N, self.horizon, -1)
        
        return predictions