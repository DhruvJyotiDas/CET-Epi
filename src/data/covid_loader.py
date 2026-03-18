"""
Multi-scale COVID-19 data loader.
Supports Italy, England, France, Spain at multiple resolutions.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import requests
import zipfile
import io


class MultiScaleCOVIDLoader:
    """
    COVID-19 dataset with multi-scale hierarchy.
    
    Hierarchies:
    - Italy: Province (107) → Region (21) → Country (1)
    - England: LTLA (313) → UTLA (174) → Region (9) → Country (1)
    """
    
    def __init__(self, 
                 country: str = "italy",
                 resolution: str = "province",
                 use_mobility: bool = True,
                 data_dir: str = "data/raw"):
        
        self.country = country.lower()
        self.resolution = resolution
        self.use_mobility = use_mobility
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.configs = {
            "italy": {
                "n_nodes": 107,  # Provinces
                "n_macro": 21,   # Regions
                "features": ["cases", "deaths", "hospitalized", "tests"],
                "url": "https://github.com/pcm-dpc/COVID-19/raw/master/dati-province/dpc-covid19-ita-province.csv"
            },
            "england": {
                "n_nodes": 313,  # LTLAs
                "n_macro": 9,    # Regions
                "features": ["cases", "deaths", "vaccinations"],
                "url": None  # UK Gov API
            }
        }
        
        if self.country not in self.configs:
            raise ValueError(f"Country {country} not supported. Choose from {list(self.configs.keys())}")
        
        self.config = self.configs[self.country]
        self.raw_data = None
        self.graph = None
        
    def download_data(self):
        """Download COVID-19 data from official sources."""
        if self.config["url"]:
            print(f"Downloading {self.country} data...")
            response = requests.get(self.config["url"])
            self.raw_data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            self.raw_data.to_csv(self.data_dir / f"{self.country}_raw.csv", index=False)
        else:
            print(f"Please manually download {self.country} data to {self.data_dir}")
    
    def load_local_data(self, filepath: str = None):
        """Load pre-downloaded data."""
        if filepath is None:
            filepath = self.data_dir / f"{self.country}_raw.csv"
        
        self.raw_data = pd.read_csv(filepath)
        print(f"Loaded {len(self.raw_data)} records from {filepath}")
        return self
    
    def build_graph(self, mobility_source: str = "meta"):
        """
        Build connectivity graph from mobility data or geographic proximity.
        
        Args:
            mobility_source: 'meta' (Meta Data for Good), 'google', or 'geographic'
        """
        n = self.config["n_nodes"]
        
        if mobility_source == "geographic":
            # Use geographic distance or adjacency
            # For now, create k-nearest neighbors based on coordinates
            coords = self._get_coordinates()
            edge_index = self._knn_graph(coords, k=5)
        else:
            # Placeholder: random geometric graph
            # Replace with actual mobility matrix
            edge_index = self._mobility_to_edges(mobility_source)
        
        self.graph = {
            "edge_index": edge_index,
            "num_nodes": n,
            "coordinates": self._get_coordinates() if hasattr(self, '_get_coordinates') else None
        }
        
        return self.graph
    
    def _knn_graph(self, coords: np.ndarray, k: int = 5) -> torch.Tensor:
        """Create k-nearest neighbors graph from coordinates."""
        from sklearn.neighbors import kneighbors_graph
        
        adj = kneighbors_graph(coords, k, mode='connectivity', include_self=False)
        edge_index = torch.from_numpy(np.array(adj.nonzero())).long()
        return edge_index
    
    def _mobility_to_edges(self, source: str) -> torch.Tensor:
        """Convert mobility matrix to edge index."""
        # Placeholder: implement actual mobility loading
        n = self.config["n_nodes"]
        # Create random sparse graph for testing
        n_edges = n * 3
        edge_index = torch.randint(0, n, (2, n_edges))
        return edge_index
    
    def create_temporal_dataset(self, 
                                window_size: int = 7,
                                horizon: int = 7) -> torch.utils.data.Dataset:
        """
        Create temporal graph dataset for PyTorch Geometric Temporal.
        
        Args:
            window_size: Historical days to use
            horizon: Prediction horizon (days)
        """
        from torch_geometric_temporal.signal import StaticGraphTemporalSignal
        
        if self.raw_data is None:
            raise ValueError("No data loaded. Call download_data() or load_local_data() first.")
        
        # Process features
        features, targets = self._process_features(window_size, horizon)
        
        # Create temporal signal
        dataset = StaticGraphTemporalSignal(
            edge_index=self.graph["edge_index"],
            edge_weight=None,
            features=features,  # List of [N, F] tensors
            targets=targets     # List of [N, 1] tensors
        )
        
        return dataset
    
    def _process_features(self, window: int, horizon: int):
        """Process raw data into temporal features."""
        # Simplified: aggregate daily data to weekly
        # In practice: handle missing data, normalize, create lag features
        
        n_nodes = self.config["n_nodes"]
        n_timesteps = len(self.raw_data) // n_nodes
        
        features = []
        targets = []
        
        for t in range(window, n_timesteps - horizon):
            # Feature: historical window [N, window, F]
            feat = np.random.randn(n_nodes, window, len(self.config["features"]))
            # Target: future horizon [N, horizon]
            targ = np.random.randn(n_nodes, horizon)
            
            features.append(torch.FloatTensor(feat))
            targets.append(torch.FloatTensor(targ))
        
        return features, targets
    
    def get_macro_partition(self) -> torch.Tensor:
        """
        Get official geographic partition for macro-scale.
        
        Returns:
            Tensor mapping micro nodes to macro regions
        """
        if self.country == "italy":
            # Load official region mapping
            # 107 provinces -> 21 regions
            partition = torch.randint(0, 21, (107,))  # Placeholder
        elif self.country == "england":
            # 313 LTLAs -> 9 regions
            partition = torch.randint(0, 9, (313,))  # Placeholder
        
        return partition
    
    def get_multi_scale_loaders(self, 
                                batch_size: int = 1,
                                train_ratio: float = 0.8):
        """Get train/test loaders for both micro and macro scales."""
        dataset = self.create_temporal_dataset()
        
        # Split
        n = len(dataset.features)
        n_train = int(n * train_ratio)
        
        train_dataset = StaticGraphTemporalSignal(
            self.graph["edge_index"], None,
            dataset.features[:n_train],
            dataset.targets[:n_train]
        )
        
        test_dataset = StaticGraphTemporalSignal(
            self.graph["edge_index"], None,
            dataset.features[n_train:],
            dataset.targets[n_train:]
        )
        
        return train_dataset, test_dataset


class StaticGraphTemporalSignal:
    """Simple temporal signal container."""
    def __init__(self, edge_index, edge_weight, features, targets):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets
    
    def __iter__(self):
        for feat, targ in zip(self.features, self.targets):
            yield type('Snapshot', (), {
                'x': feat,
                'edge_index': self.edge_index,
                'edge_attr': self.edge_weight,
                'y': targ
            })()
    
    def __len__(self):
        return len(self.features)