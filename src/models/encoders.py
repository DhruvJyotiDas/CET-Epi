# Phase 1: DCRNN/T-GCN encoders
"""
Temporal graph encoders for micro and macro scales.
Uses DCRNN from PyTorch Geometric Temporal.
"""

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import DCRNN


class MicroEncoder(nn.Module):
    """Encoder for micro-scale (county-level) dynamics."""
    
    def __init__(self, in_channels: int, out_channels: int, K: int = 2):
        super().__init__()
        
        self.dcrnn1 = DCRNN(in_channels, out_channels, K)
        self.dcrnn2 = DCRNN(out_channels, out_channels, K)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_weight=None):
        h = self.dcrnn1(x, edge_index, edge_weight)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.dcrnn2(h, edge_index, edge_weight)
        return h


class MacroEncoder(nn.Module):
    """Encoder for macro-scale (region-level) dynamics."""
    
    def __init__(self, in_channels: int, out_channels: int, K: int = 2):
        super().__init__()
        
        self.dcrnn1 = DCRNN(in_channels, out_channels, K)
        self.dcrnn2 = DCRNN(out_channels, out_channels, K)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_weight=None):
        h = self.dcrnn1(x, edge_index, edge_weight)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.dcrnn2(h, edge_index, edge_weight)
        return h