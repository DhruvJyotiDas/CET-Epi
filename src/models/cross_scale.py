# Phase 1: Cross-scale attention mechanism
"""
Cross-scale attention mechanism.
Allows information flow between micro and macro scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleAttention(nn.Module):
    """
    Bidirectional attention between micro and macro scales.
    
    Micro -> Macro: Aggregation based on assignment
    Macro -> Micro: Broadcast based on assignment
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Micro to macro attention
        self.micro_query = nn.Linear(hidden_dim, hidden_dim)
        self.macro_key = nn.Linear(hidden_dim, hidden_dim)
        self.macro_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Macro to micro attention  
        self.macro_query = nn.Linear(hidden_dim, hidden_dim)
        self.micro_key = nn.Linear(hidden_dim, hidden_dim)
        self.micro_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projections
        self.micro_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.macro_out = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                h_micro: torch.Tensor,
                h_macro: torch.Tensor,
                assignment: torch.Tensor) -> tuple:
        """
        Bidirectional cross-scale attention.
        
        Args:
            h_micro: [N_micro, H] micro features
            h_macro: [N_macro, H] macro features  
            assignment: [N_micro, N_macro] soft assignment
        
        Returns:
            h_micro_fused: [N_micro, H] updated micro features
            h_macro_fused: [N_macro, H] updated macro features
        """
        # Micro -> Macro: Aggregate micro info to macro
        # macro_j attends to all micro_i weighted by S[i,j]
        micro_to_macro = torch.mm(assignment.t(), h_micro)  # [N_macro, H]
        
        # Macro -> Micro: Broadcast macro info to micro
        # micro_i gets info from assigned macro_j
        macro_to_micro = torch.mm(assignment, h_macro)  # [N_micro, H]
        
        # Self-attention within scales
        micro_self = self._self_attention_micro(h_micro)
        macro_self = self._self_attention_macro(h_macro)
        
        # Fuse: original + cross-scale + self-attention
        h_micro_fused = self.micro_out(
            torch.cat([h_micro, macro_to_micro + micro_self], dim=-1)
        )
        h_macro_fused = self.macro_out(
            torch.cat([h_macro, micro_to_macro + macro_self], dim=-1)
        )
        
        # Residual + LayerNorm
        h_micro_fused = self.layer_norm(h_micro + h_micro_fused)
        h_macro_fused = self.layer_norm(h_macro + h_macro_fused)
        
        return h_micro_fused, h_macro_fused
    
    def _self_attention_micro(self, x: torch.Tensor) -> torch.Tensor:
        """Simple self-attention for micro scale."""
        # Simplified: just return x for now (can add full attention later)
        return x
    
    def _self_attention_macro(self, x: torch.Tensor) -> torch.Tensor:
        """Simple self-attention for macro scale."""
        return x