# REAL DATA LOADER (UCI Hungarian Chickenpox) — FIXED VERSION

import torch
import pandas as pd
from pathlib import Path

from .temporal_signal import (
    TemporalGraphDataset,
    TemporalGraphSnapshot,
    temporal_signal_split,
)


class MultiScaleChickenpoxLoader:
    def __init__(self, lags=4, horizon=1):
        self.lags = lags
        self.horizon = horizon

        self.data_path = Path("data/raw/hungary_chickenpox.csv")
        self.edge_path = Path("data/raw/hungary_county_edges.csv")

        self.n_features = 1

        # 🔥 store stats for denormalization
        self.mean = None
        self.std = None

        self.dataset = self._build_dataset()

    def _load_data(self):
        print("📂 Loading REAL dataset...")

        # Load CSV
        df = pd.read_csv(self.data_path)

        # Drop date column
        if "Date" in df.columns:
            df = df.drop(columns=["Date"])

        # Convert to float
        df = df.astype(float)

        # Convert to tensor
        data = torch.tensor(df.values, dtype=torch.float32)  # [T, N]

        # 🔥 STORE stats (VERY IMPORTANT)
        self.mean = data.mean()
        self.std = data.std()

        # 🔥 Normalize
        # 🔥 LOG TRANSFORM (BEST FOR COUNT DATA)
        data = torch.log1p(data)

        # Load edges
        edges_df = pd.read_csv(self.edge_path)

        edge_index = torch.tensor(
            edges_df[["id_1", "id_2"]].values.T,
            dtype=torch.long
        )

        edge_attr = torch.ones(edge_index.shape[1])

        return data, edge_index, edge_attr

    def _build_dataset(self):
        data, edge_index, edge_attr = self._load_data()

        T, N = data.shape
        snapshots = []

        for t in range(self.lags, T - self.horizon + 1):

            # Input window
            x = data[t - self.lags:t]      # [lags, N]
            x = x.transpose(0, 1)          # [N, lags]
            x = x.unsqueeze(-1)            # [N, lags, 1]

            # Target
            y = data[t:t + self.horizon]   # [horizon, N]
            y = y.transpose(0, 1)          # [N, horizon]

            if self.horizon == 1:
                y = y.squeeze(-1)

            snapshots.append(
                TemporalGraphSnapshot(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y
                )
            )

        print(f"✅ Real dataset loaded: {len(snapshots)} samples")

        return TemporalGraphDataset(snapshots)

    def get_split(self, train_ratio=0.8):
        return temporal_signal_split(self.dataset, train_ratio)
