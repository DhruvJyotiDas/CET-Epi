"""
Lightweight temporal graph dataset utilities used across CET-Epi loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import torch


@dataclass
class TemporalGraphSnapshot:
    """Single temporal graph snapshot."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor]
    y: torch.Tensor


class TemporalGraphDataset(Sequence[TemporalGraphSnapshot]):
    """Simple in-memory dataset for temporal graph snapshots."""

    def __init__(self, snapshots: Iterable[TemporalGraphSnapshot]):
        self.snapshots: List[TemporalGraphSnapshot] = list(snapshots)

    def __getitem__(self, index: int) -> TemporalGraphSnapshot:
        return self.snapshots[index]

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self) -> Iterator[TemporalGraphSnapshot]:
        return iter(self.snapshots)


def temporal_signal_split(
    dataset: Sequence[TemporalGraphSnapshot],
    train_ratio: float = 0.8,
) -> Tuple[TemporalGraphDataset, TemporalGraphDataset]:
    """Split a temporal dataset while preserving order."""

    if len(dataset) < 2:
        raise ValueError("Temporal dataset must contain at least two snapshots.")

    n_train = max(1, int(len(dataset) * train_ratio))
    n_train = min(n_train, len(dataset) - 1)

    train_data = TemporalGraphDataset(dataset[:n_train])
    test_data = TemporalGraphDataset(dataset[n_train:])
    return train_data, test_data
