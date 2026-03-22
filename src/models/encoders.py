# Phase 2: Improved Graph-temporal encoders (STRONGER VERSION)

from __future__ import annotations

import torch
import torch.nn as nn


def _graph_average(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    num_nodes = x.shape[0]
    device = x.device
    dtype = x.dtype

    if edge_index.numel() == 0:
        return x

    src = edge_index[0].long()
    dst = edge_index[1].long()

    if edge_weight is None:
        weight = torch.ones(src.shape[0], device=device, dtype=dtype)
    else:
        weight = edge_weight.to(device=device, dtype=dtype).reshape(-1)

    # add self loops
    self_index = torch.arange(num_nodes, device=device)
    src = torch.cat([src, self_index], dim=0)
    dst = torch.cat([dst, self_index], dim=0)
    weight = torch.cat([weight, torch.ones(num_nodes, device=device, dtype=dtype)], dim=0)

    aggregated = torch.zeros_like(x)
    aggregated.index_add_(0, dst, x[src] * weight.unsqueeze(-1))

    degree = torch.zeros(num_nodes, device=device, dtype=dtype)
    degree.index_add_(0, dst, weight)

    return aggregated / degree.clamp_min(1e-6).unsqueeze(-1)


class TemporalGraphEncoder(nn.Module):
    """
    🔥 Improved encoder:
    - Stronger temporal modeling
    - Residual connections
    - Better feature mixing
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()

        self.hidden_dim = out_channels

        self.input_proj = nn.Linear(in_channels, out_channels)

        self.message_proj = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.recurrent = nn.GRUCell(out_channels, out_channels)

        self.output_proj = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected [N, T, F], got {tuple(x.shape)}")

        n_nodes = x.shape[0]

        hidden = torch.zeros(
            n_nodes,
            self.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )

        for t in range(x.shape[1]):
            # 🔹 input transform
            x_t = self.input_proj(x[:, t, :])

            # 🔹 spatial aggregation
            x_spatial = _graph_average(x_t, edge_index, edge_weight)
            h_spatial = _graph_average(hidden, edge_index, edge_weight)

            # 🔹 message passing
            message = self.message_proj(torch.cat([x_spatial, h_spatial], dim=-1))

            # 🔹 recurrent update
            new_hidden = self.recurrent(self.dropout(message), hidden)

            # 🔥 RESIDUAL CONNECTION (VERY IMPORTANT)
            hidden = hidden + new_hidden

        return self.output_proj(hidden)


class MicroEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int = 2):
        super().__init__()
        self.encoder = TemporalGraphEncoder(in_channels, out_channels, dropout=0.2)

    def forward(self, x, edge_index, edge_weight=None):
        return self.encoder(x, edge_index, edge_weight)


class MacroEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int = 2):
        super().__init__()
        self.encoder = TemporalGraphEncoder(in_channels, out_channels, dropout=0.2)

    def forward(self, x, edge_index, edge_weight=None):
        return self.encoder(x, edge_index, edge_weight)
