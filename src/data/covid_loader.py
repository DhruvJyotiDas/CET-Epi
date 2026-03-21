"""
Multi-scale COVID-19 data loader.
Supports deterministic synthetic fallbacks for smoke testing and generic
numeric CSV processing when local data is available.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from .signal import TemporalGraphDataset, TemporalGraphSnapshot, temporal_signal_split


class MultiScaleCOVIDLoader:
    """
    COVID-19 dataset with multi-scale hierarchy.

    When a local CSV is available, the loader attempts a generic numeric-column
    reshape into [time, node, feature]. If no local data is available, it falls
    back to a deterministic synthetic dataset so the training pipeline remains
    executable end to end.
    """

    def __init__(
        self,
        country: str = "italy",
        resolution: str = "province",
        use_mobility: bool = True,
        data_dir: str = "data/raw",
        n_nodes: Optional[int] = None,
        n_macro: Optional[int] = None,
        n_features: Optional[int] = None,
    ):
        self.country = country.lower()
        self.resolution = resolution
        self.use_mobility = use_mobility
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.configs: Dict[str, Dict[str, object]] = {
            "italy": {
                "n_nodes": 107,
                "n_macro": 21,
                "features": ["cases", "deaths", "hospitalized", "tests"],
                "url": "https://github.com/pcm-dpc/COVID-19/raw/master/dati-province/dpc-covid19-ita-province.csv",
            },
            "england": {
                "n_nodes": 313,
                "n_macro": 9,
                "features": ["cases", "deaths", "vaccinations"],
                "url": None,
            },
        }

        if self.country not in self.configs:
            raise ValueError(f"Country {country} not supported. Choose from {list(self.configs.keys())}")

        base_config = self.configs[self.country]
        self.n_nodes = int(n_nodes or base_config["n_nodes"])
        self.n_macro = int(n_macro or base_config["n_macro"])
        self.n_features = int(n_features or len(base_config["features"]))
        self.raw_data: Optional[pd.DataFrame] = None
        self.graph = None
        self.uses_synthetic_data = False

    def download_data(self):
        """Download COVID-19 data from official sources when possible."""
        url = self.configs[self.country]["url"]
        if not url:
            print(f"No automatic download configured for {self.country}.")
            return self

        import requests

        print(f"Downloading {self.country} data...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        output_path = self.data_dir / f"{self.country}_raw.csv"
        output_path.write_bytes(response.content)
        self.raw_data = pd.read_csv(output_path)
        return self

    def load_local_data(self, filepath: str | None = None):
        """Load pre-downloaded data."""
        if filepath is None:
            filepath = self.data_dir / f"{self.country}_raw.csv"

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(filepath)

        self.raw_data = pd.read_csv(filepath)
        print(f"Loaded {len(self.raw_data)} records from {filepath}")
        return self

    def build_graph(self, mobility_source: str = "geographic"):
        """Build a deterministic sparse graph."""
        edge_index = self._structured_graph(k=3 if mobility_source == "geographic" else 2)
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

        self.graph = {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "num_nodes": self.n_nodes,
        }
        return self.graph

    def _structured_graph(self, k: int = 3) -> torch.Tensor:
        edges = []
        for node in range(self.n_nodes):
            for offset in range(1, k + 1):
                neighbor = (node + offset) % self.n_nodes
                edges.append((node, neighbor))
                edges.append((neighbor, node))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def create_temporal_dataset(
        self,
        window_size: int = 7,
        horizon: int = 1,
    ) -> TemporalGraphDataset:
        """Create temporal graph dataset."""
        if self.graph is None:
            self.build_graph()

        if self.raw_data is None:
            default_path = self.data_dir / f"{self.country}_raw.csv"
            if default_path.exists():
                self.load_local_data(default_path)

        if self.raw_data is None:
            self.uses_synthetic_data = True
            features, targets = self._synthetic_features(window_size, horizon)
        else:
            features, targets = self._process_features(window_size, horizon)

        snapshots = [
            TemporalGraphSnapshot(
                x=feat,
                edge_index=self.graph["edge_index"],
                edge_attr=self.graph["edge_weight"],
                y=targ,
            )
            for feat, targ in zip(features, targets)
        ]
        return TemporalGraphDataset(snapshots)

    def _process_features(self, window: int, horizon: int):
        """Process numeric local CSV data into temporal features."""
        numeric = self.raw_data.select_dtypes(include=[np.number])
        if numeric.empty or len(numeric) < self.n_nodes * (window + horizon + 1):
            self.uses_synthetic_data = True
            return self._synthetic_features(window, horizon)

        values = numeric.to_numpy(dtype=np.float32)
        n_complete_rows = (values.shape[0] // self.n_nodes) * self.n_nodes
        values = values[:n_complete_rows]

        n_timesteps = values.shape[0] // self.n_nodes
        if n_timesteps <= window + horizon:
            self.uses_synthetic_data = True
            return self._synthetic_features(window, horizon)

        values = values.reshape(n_timesteps, self.n_nodes, -1)
        if values.shape[2] == 0:
            self.uses_synthetic_data = True
            return self._synthetic_features(window, horizon)

        feature_tensor = torch.from_numpy(values)
        feature_tensor = self._pad_or_trim_features(feature_tensor)

        features = []
        targets = []
        for t in range(window, n_timesteps - horizon + 1):
            feat = feature_tensor[t - window:t].permute(1, 0, 2).contiguous()
            targ = feature_tensor[t:t + horizon, :, 0].permute(1, 0).contiguous()
            if horizon == 1:
                targ = targ.squeeze(-1)
            features.append(feat)
            targets.append(targ)

        if not features:
            self.uses_synthetic_data = True
            return self._synthetic_features(window, horizon)

        return features, targets

    def _pad_or_trim_features(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        current_features = feature_tensor.shape[2]
        if current_features == self.n_features:
            return feature_tensor
        if current_features > self.n_features:
            return feature_tensor[:, :, :self.n_features]

        repeats = [feature_tensor[:, :, -1:].clone() for _ in range(self.n_features - current_features)]
        return torch.cat([feature_tensor] + repeats, dim=2)

    def _synthetic_features(self, window: int, horizon: int):
        total_snapshots = 96
        total_steps = total_snapshots + window + horizon

        phase = torch.linspace(0, 2 * math.pi, self.n_nodes + 1)[:-1]
        cases = torch.zeros(total_steps, self.n_nodes, dtype=torch.float32)
        cases[0] = 10.0 + 2.0 * torch.sin(phase) + 0.5 * torch.cos(2 * phase)

        for t in range(1, total_steps):
            prev = cases[t - 1]
            neighbor_mean = 0.5 * (torch.roll(prev, 1) + torch.roll(prev, -1))
            seasonal = 1.8 * torch.sin(0.09 * t + phase) + 0.7 * torch.cos(0.03 * t - 0.5 * phase)
            trend = 0.03 * t
            cases[t] = torch.relu(0.72 * prev + 0.18 * neighbor_mean + seasonal + trend + 3.0)

        features = []
        targets = []
        for t in range(window, total_steps - horizon + 1):
            window_values = cases[t - window:t].transpose(0, 1)
            feat = self._build_feature_tensor(window_values)
            targ = cases[t:t + horizon].transpose(0, 1)
            if horizon == 1:
                targ = targ.squeeze(-1)
            features.append(feat)
            targets.append(targ)

        return features, targets

    def _build_feature_tensor(self, window_values: torch.Tensor) -> torch.Tensor:
        feature_list = [window_values]

        diff = torch.zeros_like(window_values)
        diff[:, 1:] = window_values[:, 1:] - window_values[:, :-1]
        feature_list.append(diff)

        centered = window_values - window_values.mean(dim=1, keepdim=True)
        feature_list.append(centered)

        relative = window_values / window_values.mean(dim=1, keepdim=True).clamp_min(1e-6)
        feature_list.append(relative)

        while len(feature_list) < self.n_features:
            time_feature = torch.linspace(0.0, 1.0, window_values.shape[1], dtype=window_values.dtype)
            time_feature = time_feature.unsqueeze(0).repeat(window_values.shape[0], 1)
            feature_list.append(time_feature)

        return torch.stack(feature_list[:self.n_features], dim=-1)

    def get_macro_partition(self) -> torch.Tensor:
        """Get a deterministic geographic partition."""
        return torch.arange(self.n_nodes) % self.n_macro

    def get_multi_scale_loaders(
        self,
        batch_size: int = 1,
        train_ratio: float = 0.8,
    ):
        """Get train/test loaders for both micro and macro scales."""
        return self.get_split(train_ratio=train_ratio)

    def get_split(
        self,
        train_ratio: float = 0.8,
        window_size: int = 7,
        horizon: int = 1,
    ):
        dataset = self.create_temporal_dataset(window_size=window_size, horizon=horizon)
        return temporal_signal_split(dataset, train_ratio=train_ratio)
