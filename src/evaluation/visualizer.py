"""
Visualization utilities for CET-Epi results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional


class CET_EpiVisualizer:
    """Comprehensive visualization for CET-Epi analysis."""
    
    def __init__(self, save_dir: str = "logs/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_training_curves(self, 
                              train_losses: List[float],
                              val_losses: List[float],
                              val_rmses: List[float],
                              save_name: str = "training_curves.png"):
        """Plot training progression."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(len(train_losses))
        
        # Loss curves
        axes[0].plot(epochs, train_losses, label='Train Loss', alpha=0.8)
        axes[0].plot(epochs, val_losses, label='Val Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RMSE
        axes[1].plot(epochs, val_rmses, label='Val RMSE', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Validation RMSE')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
        plt.show()
        return fig
    
    def plot_predictions(self,
                        predictions: torch.Tensor,
                        targets: torch.Tensor,
                        node_names: List[str] = None,
                        n_samples: int = 5,
                        save_name: str = "predictions.png"):
        """Plot prediction vs actual for sample nodes."""
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        n_nodes = min(n_samples, pred_np.shape[0])
        fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 3*n_nodes))
        
        if n_nodes == 1:
            axes = [axes]
        
        for i in range(n_nodes):
            ax = axes[i]
            
            # Time series plot
            if pred_np.ndim > 1 and pred_np.shape[1] > 1:
                # Multiple timesteps
                ax.plot(target_np[i], 'o-', label='Actual', alpha=0.7)
                ax.plot(pred_np[i], 's--', label='Predicted', alpha=0.7)
            else:
                # Single value - scatter
                ax.scatter(target_np[i], pred_np[i], alpha=0.5)
                ax.plot([target_np[i].min(), target_np[i].max()],
                       [target_np[i].min(), target_np[i].max()],
                       'r--', label='Perfect')
            
            name = node_names[i] if node_names else f"Node {i}"
            ax.set_title(f"{name}")
            ax.set_ylabel("Cases")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_assignment_matrix(self, 
                                S: torch.Tensor,
                                save_name: str = "assignment_matrix.png"):
        """Visualize soft assignment matrix."""
        S_np = S.cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Heatmap
        sns.heatmap(S_np, annot=True, fmt='.2f', cmap='Blues', 
                   ax=axes[0], cbar_kws={'label': 'Assignment Probability'})
        axes[0].set_xlabel('Macro Region')
        axes[0].set_ylabel('Micro Node (County)')
        axes[0].set_title('CEO Assignment Matrix')
        
        # Stacked bar chart
        S_cumsum = np.cumsum(S_np, axis=1)
        x = np.arange(S_np.shape[0])
        colors = plt.cm.tab10(np.linspace(0, 1, S_np.shape[1]))
        
        for j in range(S_np.shape[1]):
            bottoms = S_cumsum[:, j-1] if j > 0 else np.zeros_like(x)
            axes[1].bar(x, S_np[:, j], bottom=bottoms, 
                       label=f'Macro {j}', color=colors[j], alpha=0.8)
        
        axes[1].set_xlabel('Micro Node')
        axes[1].set_ylabel('Assignment Probability')
        axes[1].set_title('Assignment Distribution')
        axes[1].legend()
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_scale_comparison(self,
                             micro_features: torch.Tensor,
                             macro_features: torch.Tensor,
                             save_name: str = "scale_comparison.png"):
        """Compare feature distributions at micro vs macro scales."""
        micro_np = micro_features.cpu().numpy()
        macro_np = macro_features.cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Dimensionality reduction visualization (first 2 PCA components or just first 2 dims)
        if micro_np.shape[1] >= 2:
            axes[0, 0].scatter(micro_np[:, 0], micro_np[:, 1], 
                              alpha=0.6, s=100, edgecolors='black')
            axes[0, 0].set_title('Micro-Scale Feature Space')
            axes[0, 0].set_xlabel('Feature 1')
            axes[0, 0].set_ylabel('Feature 2')
            
            axes[0, 1].scatter(macro_np[:, 0], macro_np[:, 1], 
                              alpha=0.6, s=200, c='red', edgecolors='black')
            axes[0, 1].set_title('Macro-Scale Feature Space')
            axes[0, 1].set_xlabel('Feature 1')
            axes[0, 1].set_ylabel('Feature 2')
        
        # Feature statistics
        axes[1, 0].boxplot([micro_np.flatten(), macro_np.flatten()],
                          labels=['Micro', 'Macro'])
        axes[1, 0].set_title('Feature Distribution')
        axes[1, 0].set_ylabel('Value')
        
        # Variance comparison
        micro_vars = np.var(micro_np, axis=0)
        macro_vars = np.var(macro_np, axis=0)
        x = np.arange(len(micro_vars))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, micro_vars, width, label='Micro', alpha=0.7)
        axes[1, 1].bar(x + width/2, macro_vars, width, label='Macro', alpha=0.7)
        axes[1, 1].set_title('Feature Variance Comparison')
        axes[1, 1].set_xlabel('Feature Dimension')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].legend()
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig