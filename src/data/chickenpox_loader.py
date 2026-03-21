# Phase 1: Multi-scale Chickenpox dataset wrapper
"""
Multi-scale wrapper for Hungary Chickenpox dataset.
"""

from __future__ import annotations

import math

import torch

from .signal import TemporalGraphDataset, TemporalGraphSnapshot, temporal_signal_split

try:
    from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader as PGTChickenpoxDatasetLoader
except ImportError:
    PGTChickenpoxDatasetLoader = None


class MultiScaleChickenpoxLoader:
    """
    Wraps the Chickenpox dataset with a consistent [N, T, F] interface for CET-Epi.
    Falls back to a deterministic synthetic dataset when optional dependencies
    are unavailable.
    """

    def __init__(self, lags: int = 4, horizon: int = 1):
        self.lags = lags
        self.horizon = horizon
        self.n_nodes = 20
        self.n_features = 4
        self.uses_synthetic_data = False

        self.dataset = self._build_dataset()
        self.n_timesteps = len(self.dataset)

    def _build_dataset(self) -> TemporalGraphDataset:
        if PGTChickenpoxDatasetLoader is not None and self.horizon == 1:
            try:
                return self._load_external_dataset()
            except Exception:
                pass

        self.uses_synthetic_data = True
        return self._build_synthetic_dataset()

    def _load_external_dataset(self) -> TemporalGraphDataset:
        loader = PGTChickenpoxDatasetLoader()
        dataset = loader.get_dataset(lags=self.lags)

        snapshots = []
        for snapshot in dataset:
            lag_values = torch.as_tensor(snapshot.x, dtype=torch.float32)
            x = self._expand_lag_features(lag_values)

            y = torch.as_tensor(snapshot.y, dtype=torch.float32)
            if y.dim() != 1:
                y = y.reshape(y.shape[0], -1)[:, 0]

            edge_index = torch.as_tensor(snapshot.edge_index, dtype=torch.long)
            edge_attr = None
            if snapshot.edge_attr is not None:
                edge_attr = torch.as_tensor(snapshot.edge_attr, dtype=torch.float32).reshape(-1)

            snapshots.append(
                TemporalGraphSnapshot(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                )
            )

        return TemporalGraphDataset(snapshots)

    def _build_synthetic_dataset(self) -> TemporalGraphDataset:
        edge_index, edge_attr = self._build_static_graph()
        total_snapshots = 80
        total_steps = total_snapshots + self.lags + self.horizon

        phase = torch.linspace(0, 2 * math.pi, self.n_nodes + 1)[:-1]
        cases = torch.zeros(total_steps, self.n_nodes, dtype=torch.float32)
        cases[0] = 4.0 + 1.5 * torch.sin(phase) + 0.5 * torch.cos(2 * phase)

        for t in range(1, total_steps):
            prev = cases[t - 1]
            neighbor_mean = 0.5 * (torch.roll(prev, 1) + torch.roll(prev, -1))
            seasonal = 1.2 * torch.sin(0.18 * t + phase) + 0.6 * torch.cos(0.07 * t - phase)
            local_shock = 0.25 * torch.sin(0.11 * t + 0.5 * phase)
            cases[t] = torch.relu(0.65 * prev + 0.25 * neighbor_mean + seasonal + local_shock + 2.0)

        snapshots = []
        for t in range(self.lags, total_steps - self.horizon + 1):
            window = cases[t - self.lags:t].transpose(0, 1)
            x = self._expand_lag_features(window)

            target = cases[t:t + self.horizon].transpose(0, 1)
            if self.horizon == 1:
                target = target.squeeze(-1)

            snapshots.append(
                TemporalGraphSnapshot(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=target,
                )
            )

        return TemporalGraphDataset(snapshots)

    def _build_static_graph(self):
        edges = []
        for node in range(self.n_nodes):
            for offset in (1, 2):
                neighbor = (node + offset) % self.n_nodes
                edges.append((node, neighbor))
                edges.append((neighbor, node))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float32)
        return edge_index, edge_attr

    def _expand_lag_features(self, lag_values: torch.Tensor) -> torch.Tensor:
        raw = lag_values

        diff = torch.zeros_like(raw)
        diff[:, 1:] = raw[:, 1:] - raw[:, :-1]

        centered = raw - raw.mean(dim=1, keepdim=True)
        relative = raw / raw.mean(dim=1, keepdim=True).clamp_min(1e-6)

        return torch.stack([raw, diff, centered, relative], dim=-1)

    def get_split(self, train_ratio: float = 0.8):
        """Get train/test split."""
        return temporal_signal_split(self.dataset, train_ratio=train_ratio)

    def get_static_graph(self):
        """Get static county adjacency."""
        first = self.dataset[0]
        return first.edge_index, first.edge_attr

    def create_geographic_macro(self, n_regions: int = 5):
        """Create a deterministic geographic-style macro partition."""
        return torch.arange(self.n_nodes) % n_regions

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.n_timesteps
