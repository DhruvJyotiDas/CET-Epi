# Phase 1: EI calculation wrapper around einet library
"""
Effective Information (EI) calculation for CET-Epi.
Wraps einet library with temporal graph support.
"""

import torch
import numpy as np
import networkx as nx
from typing import Optional, Tuple


class EffectiveInformation:
    """
    Calculate Effective Information for temporal graphs.
    
    EI = Determinism - Degeneracy
    Higher EI = more causal power = better macro-scale.
    """
    
    def __init__(self, graph: nx.Graph, transition_matrix: Optional[np.ndarray] = None):
        self.G = graph
        self.n_nodes = graph.number_of_nodes()
        
        # If no transition matrix provided, create from graph structure
        if transition_matrix is None:
            self.T = self._create_transition_from_graph()
        else:
            self.T = transition_matrix
            
    def _create_transition_from_graph(self) -> np.ndarray:
        """Create random walk transition matrix from graph."""
        A = nx.to_numpy_array(self.G)
        # Row normalize
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        T = A / row_sums
        return T
    
    def compute(self, noise: float = 0.0) -> float:
        """
        Compute EI using einet-style calculation.
        
        Args:
            noise: Intervention noise level (0 = perfect interventions)
        
        Returns:
            Effective Information value (higher is better)
        """
        # Determinism: expected predictability of transitions
        determinism = self._compute_determinism()
        
        # Degeneracy: how many past states lead to same future
        degeneracy = self._compute_degeneracy()
        
        ei = determinism - degeneracy
        return max(0.0, ei)  # EI >= 0
    
    def _compute_determinism(self) -> float:
        """Average determinism of transitions."""
        # For each state, how deterministic is its transition?
        # Higher entropy in transition = lower determinism
        entropies = []
        for i in range(self.n_nodes):
            probs = self.T[i]
            probs = probs[probs > 0]  # Remove zeros
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log2(probs))
                entropies.append(entropy)
        
        # Determinism = 1 - normalized entropy
        max_entropy = np.log2(self.n_nodes)
        avg_entropy = np.mean(entropies) if entropies else max_entropy
        determinism = 1 - (avg_entropy / max_entropy) if max_entropy > 0 else 0
        return determinism
    
    def _compute_degeneracy(self) -> float:
        """Measure of convergent causation."""
        # How many-to-one are the transitions?
        # Look at column similarities in T
        degeneracy = 0.0
        for j in range(self.n_nodes):
            col = self.T[:, j]
            # Count how many sources contribute significantly
            contributors = np.sum(col > 0.01)
            degeneracy += max(0, contributors - 1)  # Penalize many-to-one
        
        # Normalize
        degeneracy = degeneracy / (self.n_nodes * (self.n_nodes - 1)) if self.n_nodes > 1 else 0
        return degeneracy
    
    def compute_for_macro(self, partition: np.ndarray) -> float:
        """
        Compute EI for coarse-grained (macro) representation.
        
        Args:
            partition: Array of shape [n_micro] with macro assignments
        
        Returns:
            EI at macro scale
        """
        # Create macro transition matrix
        n_macro = int(partition.max()) + 1
        T_macro = np.zeros((n_macro, n_macro))
        
        # Aggregate transitions
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                macro_i = int(partition[i])
                macro_j = int(partition[j])
                T_macro[macro_i, macro_j] += self.T[i, j]
        
        # Normalize
        row_sums = T_macro.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T_macro = T_macro / row_sums
        
        # Create macro graph
        G_macro = nx.DiGraph()
        for i in range(n_macro):
            G_macro.add_node(i)
            for j in range(n_macro):
                if T_macro[i, j] > 0.01:
                    G_macro.add_edge(i, j, weight=T_macro[i, j])
        
        # Compute EI
        ei_macro = EffectiveInformation(G_macro, T_macro).compute()
        return ei_macro


def torch_ei_approximation(features: torch.Tensor, 
                          edge_index: torch.Tensor,
                          assignment: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation of EI for training.
    Uses feature determinism as proxy.
    
    Args:
        features: [N, F] node features
        edge_index: [2, E] edge indices
        assignment: [N, M] soft assignment to macro nodes
    
    Returns:
        Approximate EI score (differentiable)
    """
    # Coarse-grain features
    macro_features = torch.mm(assignment.t(), features)  # [M, F]
    
    # Determinism: variance of macro features (lower = more deterministic)
    feature_variance = torch.var(macro_features, dim=0).mean()
    determinism = torch.exp(-feature_variance)  # [0, 1]
    
    # Degeneracy: entropy of assignment (lower = less degenerate)
    assignment_entropy = -torch.sum(assignment * torch.log(assignment + 1e-10), dim=1).mean()
    max_entropy = np.log(assignment.shape[1])
    degeneracy = assignment_entropy / max_entropy if max_entropy > 0 else 0
    
    # EI approximation
    ei = determinism - degeneracy
    return ei