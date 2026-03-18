# Phase 1: CET-Epi loss functions
"""
Loss functions for CET-Epi.
"""

import torch
import torch.nn.functional as F


def cet_epi_loss(predictions, targets, ei_loss, 
                 assignment, 
                 ei_weight: float = 0.1,
                 sparsity_weight: float = 0.01):
    """
    Combined loss: prediction + emergence regularization + sparsity.
    
    Args:
        predictions: [N, horizon, 1]
        targets: [N, horizon, 1]
        ei_loss: Negative EI score (to minimize = maximize EI)
        assignment: [N, M] soft assignment matrix
        ei_weight: Weight for emergence regularization
        sparsity_weight: Weight for assignment sparsity
    """
    # Prediction loss (MSE)
    pred_loss = F.mse_loss(predictions, targets)
    
    # Emergence regularization (already negative in model)
    emergence_loss = ei_loss
    
    # Sparsity: encourage concentrated assignments (L1 norm)
    # Lower entropy in assignment = more confident clustering
    entropy = -torch.sum(assignment * torch.log(assignment + 1e-10), dim=1).mean()
    sparsity_loss = entropy  # Lower entropy = sparser
    
    # Total loss
    total_loss = pred_loss + ei_weight * emergence_loss + sparsity_weight * sparsity_loss
    
    return {
        'total': total_loss,
        'prediction': pred_loss,
        'emergence': emergence_loss,
        'sparsity': sparsity_loss
    }