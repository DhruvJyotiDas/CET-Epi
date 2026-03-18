"""
Effective Information analysis tools.
Validates that macro-scale has higher EI than micro-scale.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

from ..core.effective_information import EffectiveInformation


class EIAnalyzer:
    """
    Analyze Causal Emergence in trained CET-Epi model.
    
    Key validation: EI_macro > EI_micro (emergence condition)
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    @torch.no_grad()
    def compute_scale_ei(self, 
                        x: torch.Tensor,
                        edge_index: torch.Tensor,
                        edge_weight: torch.Tensor = None) -> Dict[str, float]:
        """
        Compute EI at both micro and macro scales.
        
        Returns:
            Dictionary with EI_micro, EI_macro, emergence_score
        """
        # Forward pass to get intermediate representations
        predictions, ei_loss, intermediates = self.model(
            x, edge_index, edge_weight, return_all=True
        )
        
        h_micro = intermediates['h_micro']  # [N_micro, H]
        h_macro = intermediates['h_macro']  # [N_macro, H]
        S = intermediates['S']  # [N_micro, N_macro]
        edge_macro = intermediates['edge_macro']
        
        # Build micro graph
        G_micro = self._build_graph(edge_index, h_micro.shape[0])
        
        # Build macro graph
        G_macro = self._build_graph(edge_macro, h_macro.shape[0])
        
        # Compute EI for micro (using feature-based transition)
        ei_micro = self._compute_feature_ei(h_micro, G_micro)
        
        # Compute EI for macro
        ei_macro = self._compute_feature_ei(h_macro, G_macro)
        
        # Emergence score
        emergence = ei_macro - ei_micro
        
        return {
            'ei_micro': ei_micro,
            'ei_macro': ei_macro,
            'emergence_score': emergence,
            'emergence_ratio': ei_macro / max(ei_micro, 1e-10),
            'assignment_entropy': self._assignment_entropy(S)
        }
    
    def _build_graph(self, edge_index: torch.Tensor, n_nodes: int) -> nx.Graph:
        """Build NetworkX graph from edge index."""
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        
        edges = edge_index.cpu().t().numpy()
        for i in range(edges.shape[0]):
            u, v = edges[i]
            if u != v:  # No self-loops
                G.add_edge(int(u), int(v))
        
        return G
    
    def _compute_feature_ei(self, features: torch.Tensor, graph: nx.Graph) -> float:
        """
        Compute EI using feature similarity as transition proxy.
        """
        # Create transition matrix based on feature similarity
        features_np = features.cpu().numpy()
        n = features_np.shape[0]
        
        # Cosine similarity between node features
        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        normalized = features_np / (norms + 1e-10)
        similarity = np.dot(normalized, normalized.T)
        
        # Row normalize to get transition probabilities
        row_sums = similarity.sum(axis=1, keepdims=True)
        transition = similarity / (row_sums + 1e-10)
        
        # Compute EI
        ei_calc = EffectiveInformation(graph, transition)
        return ei_calc.compute()
    
    def _assignment_entropy(self, S: torch.Tensor) -> float:
        """Compute entropy of assignment matrix (lower = more confident)."""
        entropy = -torch.sum(S * torch.log(S + 1e-10), dim=1).mean()
        return entropy.item()
    
    def analyze_over_time(self, data_loader) -> List[Dict]:
        """Track EI over temporal snapshots."""
        results = []
        
        for i, snapshot in enumerate(data_loader):
            x = snapshot.x.to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            edge_attr = snapshot.edge_attr.to(self.device) if snapshot.edge_attr is not None else None
            
            ei_stats = self.compute_scale_ei(x, edge_index, edge_attr)
            ei_stats['timestep'] = i
            results.append(ei_stats)
            
            if i % 10 == 0:
                print(f"Timestep {i}: EI_micro={ei_stats['ei_micro']:.4f}, "
                      f"EI_macro={ei_stats['ei_macro']:.4f}, "
                      f"Emergence={ei_stats['emergence_score']:.4f}")
        
        return results
    
    def plot_emergence_analysis(self, results: List[Dict], save_path: str = None):
        """Plot EI comparison over time."""
        timesteps = [r['timestep'] for r in results]
        ei_micro = [r['ei_micro'] for r in results]
        ei_macro = [r['ei_macro'] for r in results]
        emergence = [r['emergence_score'] for r in results]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # EI comparison
        axes[0].plot(timesteps, ei_micro, label='Micro EI', alpha=0.7)
        axes[0].plot(timesteps, ei_macro, label='Macro EI', linewidth=2)
        axes[0].fill_between(timesteps, ei_micro, ei_macro, 
                            where=[m > micro for m, micro in zip(ei_macro, ei_micro)],
                            alpha=0.3, label='Emergence Zone')
        axes[0].set_ylabel('Effective Information')
        axes[0].legend()
        axes[0].set_title('Causal Emergence: EI Comparison')
        axes[0].grid(True, alpha=0.3)
        
        # Emergence score
        axes[1].plot(timesteps, emergence, color='green', linewidth=2)
        axes[1].axhline(y=0, color='red', linestyle='--', label='No Emergence')
        axes[1].fill_between(timesteps, 0, emergence, 
                            where=[e > 0 for e in emergence],
                            alpha=0.3, color='green', label='Positive Emergence')
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Emergence Score (EI_macro - EI_micro)')
        axes[1].legend()
        axes[1].set_title('Causal Emergence Score Over Time')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved emergence analysis to {save_path}")
        
        plt.show()
        
        return fig
    
    def get_macro_partition(self) -> np.ndarray:
        """Get hard clustering assignment."""
        return self.model.get_macro_partition().cpu().numpy()
    
    def visualize_partition(self, coordinates: np.ndarray = None, save_path: str = None):
        """
        Visualize learned macro partition.
        
        Args:
            coordinates: [N_micro, 2] geographic coordinates (optional)
        """
        partition = self.get_macro_partition()
        n_macro = int(partition.max()) + 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if coordinates is not None:
            # Geographic plot
            scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                               c=partition, cmap='tab10', s=200, alpha=0.7,
                               edgecolors='black', linewidth=2)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            # Network plot (circular layout)
            n_micro = len(partition)
            angles = np.linspace(0, 2*np.pi, n_micro, endpoint=False)
            x = np.cos(angles)
            y = np.sin(angles)
            scatter = ax.scatter(x, y, c=partition, cmap='tab10', s=300, alpha=0.7,
                               edgecolors='black', linewidth=2)
            
            # Add labels
            for i, (xi, yi) in enumerate(zip(x, y)):
                ax.annotate(str(i), (xi, yi), ha='center', va='center', fontsize=8)
        
        ax.set_title(f'Learned Macro Partition (K={n_macro} regions)')
        plt.colorbar(scatter, ax=ax, label='Macro Region')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig