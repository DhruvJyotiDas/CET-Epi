# Phase 1: Causal Emergence Operator core implementation
"""
Causal Emergence Operator (CEO)
Learns optimal coarse-graining that maximizes Effective Information.
"""

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
    
    def __init__(self, 
                 n_micro: int,
                 n_macro: int,
                 micro_features: int,
                 macro_features: int,
                 temperature: float = 1.0):
        super().__init__()
        
        self.n_micro = n_micro
        self.n_macro = n_macro
        self.temperature = temperature
        
        # Soft assignment matrix (differentiable)
        # S[i,j] = probability micro i belongs to macro j
        self.assignment_logits = nn.Parameter(torch.randn(n_micro, n_macro) * 0.1)
        
        # Feature transformation for macro nodes
        self.feature_transform = nn.Sequential(
            nn.Linear(micro_features, macro_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(macro_features, macro_features)
        )
        
        # Edge weight learner for macro graph
        self.edge_predictor = nn.Sequential(
            nn.Linear(macro_features * 2, macro_features),
            nn.ReLU(),
            nn.Linear(macro_features, 1),
            nn.Sigmoid()
        )
        
        self._init_assignment()
        
    def _init_assignment(self):
        """Initialize with soft clustering."""
        # Start with uniform assignment
        with torch.no_grad():
            self.assignment_logits.fill_(0.0)
        
    def get_assignment(self, hard: bool = False) -> torch.Tensor:
        """
        Get assignment matrix.
        
        Args:
            hard: If True, return one-hot (for evaluation)
        """
        if hard:
            # Gumbel-softmax or straight-through for hard assignment
            return F.one_hot(self.assignment_logits.argmax(dim=1), 
                           num_classes=self.n_macro).float()
        else:
            # Soft assignment with temperature annealing
            return F.softmax(self.assignment_logits / self.temperature, dim=1)
    
    def forward(self, 
                x_micro: torch.Tensor,
                edge_index_micro: torch.Tensor,
                edge_weight_micro: Optional[torch.Tensor] = None,
                return_ei: bool = True) -> tuple:
        """
        Coarse-grain micro graph to macro graph.
        
        Args:
            x_micro: [N_micro, F_micro] micro features
            edge_index_micro: [2, E] micro edges
            edge_weight_micro: [E] edge weights (optional)
            return_ei: Whether to compute EI approximation
        
        Returns:
            x_macro: [N_macro, F_macro] macro features
            edge_index_macro: [2, E_macro] macro edges
            edge_weight_macro: [E_macro] macro edge weights
            ei_score: Approximate EI (if return_ei=True)
            assignment: [N_micro, N_macro] assignment matrix
        """
        # Get soft assignment
        S = self.get_assignment(hard=False)  # [N_micro, N_macro]
        
        # Coarse-grain features: weighted aggregation
        # macro_j = sum_i S[i,j] * micro_i
        x_macro = torch.mm(S.t(), x_micro)  # [N_macro, F_micro]
        x_macro = self.feature_transform(x_macro)  # [N_macro, F_macro]
        
        # Coarse-grain graph structure
        edge_index_macro, edge_weight_macro = self._pool_edges(
            edge_index_micro, edge_weight_micro, S, x_macro
        )
        
        # Compute EI approximation for regularization
        ei_score = None
        if return_ei:
            ei_score = torch_ei_approximation(x_micro, edge_index_micro, S)
        
        return x_macro, edge_index_macro, edge_weight_macro, ei_score, S
    
    def _pool_edges(self,
                    edge_index: torch.Tensor,
                    edge_weight: Optional[torch.Tensor],
                    S: torch.Tensor,
                    x_macro: torch.Tensor) -> tuple:
        """
        Pool micro edges to macro edges using assignment.
        """
        N_macro = S.shape[1]
        
        # Map micro edges to macro edges
        source_macro = S[edge_index[0]].argmax(dim=1)  # [E]
        target_macro = S[edge_index[1]].argmax(dim=1)  # [E]
        
        # Create macro edge index
        edge_index_macro = torch.stack([source_macro, target_macro], dim=0)
        
        # Remove self-loops and duplicates
        edge_index_macro, edge_weight_macro = self._deduplicate_edges(
            edge_index_macro, edge_weight, N_macro
        )
        
        # Predict edge weights based on macro node features
        if edge_index_macro.shape[1] > 0:
            src_features = x_macro[edge_index_macro[0]]
            tgt_features = x_macro[edge_index_macro[1]]
            edge_feats = torch.cat([src_features, tgt_features], dim=1)
            edge_weight_macro = self.edge_predictor(edge_feats).squeeze()
        else:
            edge_weight_macro = torch.ones(edge_index_macro.shape[1], 
                                          device=edge_index.device)
        
        return edge_index_macro, edge_weight_macro
    
    def _deduplicate_edges(self,
                          edge_index: torch.Tensor,
                          edge_weight: Optional[torch.Tensor],
                          num_nodes: int) -> tuple:
        """Remove duplicate edges and self-loops."""
        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        if edge_weight is not None:
            edge_weight = edge_weight[mask]
        
        # Deduplicate using hashing
        if edge_index.shape[1] == 0:
            return edge_index, edge_weight if edge_weight is not None else torch.ones(0)
        
        # Sort and unique
        edges = edge_index.t().contiguous()
        # Use torch.unique (requires recent PyTorch)
        unique_edges, inverse_indices = torch.unique(edges, dim=0, return_inverse=True)
        
        edge_index = unique_edges.t()
        
        # Aggregate weights for duplicate edges
        if edge_weight is not None:
            edge_weight_agg = torch.zeros(unique_edges.shape[0], device=edge_weight.device)
            edge_weight_agg.index_add_(0, inverse_indices, edge_weight)
            edge_weight = edge_weight_agg / torch.bincount(inverse_indices).float()
        else:
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
        
        return edge_index, edge_weight
    
    def set_temperature(self, temp: float):
        """Anneal temperature for harder assignments."""
        self.temperature = temp