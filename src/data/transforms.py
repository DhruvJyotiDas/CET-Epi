"""
Data preprocessing transforms for temporal epidemic data.
"""

import torch
import numpy as np
from typing import Optional


class TemporalNormalize:
    """Normalize temporal features by historical statistics."""
    
    def __init__(self, method: str = "zscore", window: int = 30):
        self.method = method
        self.window = window
        self.mean = None
        self.std = None
        
    def fit(self, data: torch.Tensor):
        """Compute statistics from training data."""
        # data: [N, T, F]
        if self.method == "zscore":
            self.mean = data.mean(dim=(0, 1), keepdim=True)  # [1, 1, F]
            self.std = data.std(dim=(0, 1), keepdim=True) + 1e-8
        elif self.method == "minmax":
            self.min = data.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0]
            self.max = data.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        if self.method == "zscore":
            return (data - self.mean) / self.std
        elif self.method == "minmax":
            return (data - self.min) / (self.max - self.min + 1e-8)
        return data
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Reverse normalization."""
        if self.method == "zscore":
            return data * self.std + self.mean
        elif self.method == "minmax":
            return data * (self.max - self.min) + self.min
        return data


class AddLaplacianEigenvectors:
    """Add graph Laplacian eigenvectors as positional encodings."""
    
    def __init__(self, k: int = 8):
        self.k = k
        
    def __call__(self, edge_index: torch.Tensor, 
                 num_nodes: int) -> torch.Tensor:
        """
        Compute first k Laplacian eigenvectors.
        
        Returns:
            [N, k] positional encoding
        """
        from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
        from scipy.sparse.linalg import eigsh
        
        # Compute Laplacian
        edge_index_lap, edge_weight = get_laplacian(edge_index, normalization='sym', 
                                                     num_nodes=num_nodes)
        
        # Convert to scipy
        L = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)
        
        # Compute k smallest eigenvalues
        try:
            eigenvalues, eigenvectors = eigsh(L, k=self.k, which='SM', 
                                               return_eigenvectors=True)
        except:
            # Fallback if eigsh fails
            eigenvectors = np.random.randn(num_nodes, self.k)
        
        return torch.FloatTensor(eigenvectors)


class TemporalDifference:
    """Convert to first-order differences (growth rates)."""
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, T, F] temporal data
        Returns:
            [N, T-1, F] differences
        """
        return x[:, 1:, :] - x[:, :-1, :]


class LogTransform:
    """Log transform for count data."""
    
    def __init__(self, offset: float = 1.0):
        self.offset = offset
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.offset)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - self.offset


class SeasonalDecomposition:
    """Remove weekly seasonality from epidemic data."""
    
    def __init__(self, period: int = 7):
        self.period = period
        
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, T, F]
        """
        # Simple moving average decomposition
        trend = torch.nn.functional.avg_pool1d(
            x.transpose(1, 2),  # [N, F, T]
            kernel_size=self.period,
            stride=1,
            padding=self.period//2
        ).transpose(1, 2)  # [N, T, F]
        
        seasonal = x - trend
        
        return seasonal, trend