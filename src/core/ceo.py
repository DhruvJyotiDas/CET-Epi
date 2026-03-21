# Phase 1: Causal Emergence Operator core implementation
"""
Causal Emergence Operator (CEO)
Learns optimal coarse-graining that maximizes Effective Information.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .effective_information import torch_ei_approximation


class CausalEmergenceOperator(nn.Module):
    """
    Learned coarse-graining for causal emergence.

    Input: Micro-scale graph (N_micro nodes)
    Output: Macro-scale graph (N_macro nodes) with higher EI
    """

    def __init__(
        self,
        n_micro: int,
        n_macro: int,
        micro_features: int,
        macro_features: int,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.n_micro = n_micro
        self.n_macro = n_macro
        self.temperature = temperature

        # S[i, j] = probability micro i belongs to macro j
        self.assignment_logits = nn.Parameter(torch.empty(n_micro, n_macro))

        self.feature_transform = nn.Sequential(
            nn.Linear(micro_features, macro_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(macro_features, macro_features),
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(macro_features * 2, macro_features),
            nn.ReLU(),
            nn.Linear(macro_features, 1),
            nn.Sigmoid(),
        )

        self._init_assignment()

    def _init_assignment(self):
        """Initialize with a balanced but still soft partition."""
        with torch.no_grad():
            preferred = torch.arange(self.n_micro) % self.n_macro
            self.assignment_logits.fill_(-0.25)
            self.assignment_logits.scatter_(1, preferred.unsqueeze(1), 0.25)
            self.assignment_logits.add_(0.02 * torch.randn_like(self.assignment_logits))

    def get_assignment(self, hard: bool = False) -> torch.Tensor:
        """Get the current assignment matrix."""
        if hard:
            return F.one_hot(
                self.assignment_logits.argmax(dim=1),
                num_classes=self.n_macro,
            ).float()
        return F.softmax(self.assignment_logits / self.temperature, dim=1)

    def forward(
        self,
        x_micro: torch.Tensor,
        edge_index_micro: torch.Tensor,
        edge_weight_micro: Optional[torch.Tensor] = None,
        return_ei: bool = True,
    ) -> tuple:
        """
        Coarse-grain micro graph to macro graph.

        Args:
            x_micro: [N_micro, F_micro] micro features
            edge_index_micro: [2, E] micro edges
            edge_weight_micro: [E] edge weights (optional)
            return_ei: Whether to compute EI approximation
        """

        S = self.get_assignment(hard=False)
        cluster_mass = S.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1e-6)

        x_macro = torch.mm(S.t(), x_micro) / cluster_mass
        x_macro = self.feature_transform(x_macro)

        edge_index_macro, edge_weight_macro = self._pool_edges(
            edge_index_micro,
            edge_weight_micro,
            S,
            x_macro,
            cluster_mass,
        )

        ei_score = None
        if return_ei:
            ei_score = torch_ei_approximation(x_micro, edge_index_micro, S)

        return x_macro, edge_index_macro, edge_weight_macro, ei_score, S

    def _pool_edges(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        assignment: torch.Tensor,
        x_macro: torch.Tensor,
        cluster_mass: torch.Tensor,
    ) -> tuple:
        """Pool micro edges to macro edges using soft assignments."""

        adj_macro = self._soft_pool_adjacency(edge_index, edge_weight, assignment)
        cluster_outer = torch.mm(cluster_mass, cluster_mass.t()).clamp_min(1e-6)
        adj_macro = adj_macro / cluster_outer
        adj_macro.fill_diagonal_(0.0)

        macro_edges = (adj_macro > 1e-8).nonzero(as_tuple=False)
        if macro_edges.numel() == 0:
            empty_index = torch.empty((2, 0), dtype=torch.long, device=x_macro.device)
            empty_weight = torch.empty((0,), dtype=x_macro.dtype, device=x_macro.device)
            return empty_index, empty_weight

        edge_index_macro = macro_edges.t().contiguous()
        structural_weight = adj_macro[macro_edges[:, 0], macro_edges[:, 1]]

        src_features = x_macro[edge_index_macro[0]]
        tgt_features = x_macro[edge_index_macro[1]]
        edge_feats = torch.cat([src_features, tgt_features], dim=1)
        learned_gate = self.edge_predictor(edge_feats).view(-1)
        edge_weight_macro = structural_weight * learned_gate

        return edge_index_macro, edge_weight_macro

    def _soft_pool_adjacency(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        assignment: torch.Tensor,
    ) -> torch.Tensor:
        """Build a differentiable macro adjacency matrix."""

        num_micro = assignment.shape[0]
        device = assignment.device
        dtype = assignment.dtype

        adjacency = torch.zeros((num_micro, num_micro), device=device, dtype=dtype)
        if edge_index.numel() > 0:
            if edge_weight is None:
                weight = torch.ones(edge_index.shape[1], device=device, dtype=dtype)
            else:
                weight = edge_weight.to(device=device, dtype=dtype).reshape(-1)

            adjacency.index_put_(
                (edge_index[0].long(), edge_index[1].long()),
                weight,
                accumulate=True,
            )
            adjacency = 0.5 * (adjacency + adjacency.t())

        return torch.mm(assignment.t(), torch.mm(adjacency, assignment))

    def set_temperature(self, temp: float):
        """Anneal temperature for harder assignments."""
        self.temperature = temp
