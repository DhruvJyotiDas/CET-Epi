# Phase 1: Multi-scale Chickenpox dataset wrapper
"""
Multi-scale wrapper for Hungary Chickenpox dataset.
"""

import torch
import numpy as np
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split


class MultiScaleChickenpoxLoader:
    """
    Wraps PyTorch Geometric Temporal Chickenpox dataset
    with multi-scale structure for CET-Epi.
    """
    
    def __init__(self, lags: int = 4):
        self.loader = ChickenpoxDatasetLoader()
        self.dataset = self.loader.get_dataset(lags=lags)
        self.lags = lags
        
        # Dataset properties
        self.n_nodes = 20  # 20 counties in Hungary
        self.n_features = 4  # From dataset
        self.n_timesteps = len(list(self.dataset))
        
    def get_split(self, train_ratio: float = 0.8):
        """Get train/test split."""
        return temporal_signal_split(self.dataset, train_ratio=train_ratio)
    
    def get_static_graph(self):
        """Get static county adjacency."""
        # First snapshot has edge_index
        first = next(iter(self.dataset))
        return first.edge_index, first.edge_attr
    
    def create_geographic_macro(self, n_regions: int = 5):
        """
        Create initial macro partition based on geography.
        For Hungary, can use NUTS regions or k-means on coordinates.
        """
        # Simplified: random initialization (replace with real coordinates)
        partition = torch.randint(0, n_regions, (self.n_nodes,))
        return partition
    
    def __iter__(self):
        """Iterate over temporal snapshots."""
        return iter(self.dataset)
    
    def __len__(self):
        return self.n_timesteps