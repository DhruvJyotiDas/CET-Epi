# Phase 1: Main CET-Epi model architecture
"""
CET-Epi: Causal Emergence Theory for Epidemics
Main model integrating all components.
"""

import torch
import torch.nn as nn
from ..core.ceo import CausalEmergenceOperator
from .encoders import MicroEncoder, MacroEncoder
from .cross_scale import CrossScaleAttention
from .predictor import ScaleAwarePredictor


class CET_Epi(nn.Module):
    """
    Full CET-Epi architecture.
    
    Flow:
    1. Micro encoding (county level)
    2. CEO: Learn macro scale with higher EI
    3. Macro encoding (region level)
    4. Cross-scale fusion
    5. Scale-aware prediction
    """
    
    def __init__(self,
                 n_micro: int,
                 n_macro: int,
                 in_channels: int,
                 hidden_dim: int,
                 out_channels: int = 1,
                 horizon: int = 1,
                 K: int = 2):
        super().__init__()
        
        self.n_micro = n_micro
        self.n_macro = n_macro
        
        # Micro-scale processing
        self.micro_encoder = MicroEncoder(in_channels, hidden_dim, K)
        
        # Causal Emergence Operator
        self.ceo = CausalEmergenceOperator(
            n_micro=n_micro,
            n_macro=n_macro,
            micro_features=hidden_dim,
            macro_features=hidden_dim,
            temperature=1.0
        )
        
        # Macro-scale processing
        self.macro_encoder = MacroEncoder(hidden_dim, hidden_dim, K)
        
        # Cross-scale information flow
        self.cross_scale = CrossScaleAttention(hidden_dim)
        
        # Prediction head
        self.predictor = ScaleAwarePredictor(hidden_dim, out_channels, horizon)
        
        # Learnable parameter for EI loss weight
        self.ei_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None,
                return_all: bool = False):
        """
        Forward pass.
        
        Args:
            x: [N_micro, T, F] or [N_micro, F] (temporal or static features)
            edge_index: [2, E] micro graph edges
            edge_weight: [E] edge weights
            return_all: Return intermediate outputs for analysis
        
        Returns:
            predictions: [N_micro, horizon, out_dim]
            ei_loss: Emergence regularization term
            (optional) intermediate dict
        """
        # 1. Micro encoding
        h_micro = self.micro_encoder(x, edge_index, edge_weight)
        
        # 2. CEO: Learn optimal macro scale
        h_macro, edge_macro, weight_macro, ei_score, S = self.ceo(
            h_micro, edge_index, edge_weight, return_ei=True
        )
        
        # 3. Macro encoding
        h_macro = self.macro_encoder(h_macro, edge_macro, weight_macro)
        
        # 4. Cross-scale fusion
        h_micro_fused, h_macro_fused = self.cross_scale(h_micro, h_macro, S)
        
        # 5. Prediction
        predictions = self.predictor(h_micro_fused, h_macro_fused, S)
        
        # Emergence loss (negative EI to maximize)
        if ei_score is None:
            ei_loss = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        else:
            ei_loss = -ei_score.to(device=predictions.device, dtype=predictions.dtype)
        
        if return_all:
            intermediates = {
                'h_micro': h_micro,
                'h_macro': h_macro,
                'S': S,
                'ei_score': ei_score,
                'edge_macro': edge_macro,
                'h_micro_fused': h_micro_fused,
                'h_macro_fused': h_macro_fused
            }
            return predictions, ei_loss, intermediates
        
        return predictions, ei_loss
    
    def get_macro_partition(self) -> torch.Tensor:
        """Get hard assignment for interpretability."""
        return self.ceo.get_assignment(hard=True).argmax(dim=1)
    
    def anneal_temperature(self, epoch: int, total_epochs: int):
        """Anneal CEO temperature for harder assignments."""
        # Linear annealing from 1.0 to 0.1
        new_temp = max(0.1, 1.0 - 0.9 * (epoch / total_epochs))
        self.ceo.set_temperature(new_temp)
