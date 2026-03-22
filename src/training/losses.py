# Phase 1: CET-Epi loss functions
"""
Loss functions for CET-Epi.
"""

import torch
import torch.nn.functional as F


def _align_targets(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Coerce targets into [N, horizon, 1] to match predictions."""

    if targets.dim() == 1:
        targets = targets.unsqueeze(-1).unsqueeze(-1)
    elif targets.dim() == 2:
        targets = targets.unsqueeze(-1)

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {tuple(predictions.shape)} vs {tuple(targets.shape)}"
        )

    return targets.to(device=predictions.device, dtype=predictions.dtype)


def cet_epi_loss(
    predictions,
    targets,
    ei_loss,
    assignment,
    ei_weight: float = 0.1,
    sparsity_weight: float = 0.01,
    balance_weight: float = 0.01,
    **kwargs  # 🔥 IMPORTANT FIX (absorbs unexpected args)
):
    """
    Combined loss:
    - Prediction loss (MSE)
    - Emergence (maximize EI)
    - Sparsity (low entropy assignments)
    - Balance (equal cluster usage)
    """

    targets = _align_targets(predictions, targets)

    # 1. Prediction loss
    mae = F.l1_loss(predictions, targets)
    mse = F.mse_loss(predictions, targets)
    pred_loss = 0.7 * mae + 0.3 * mse
    # 2. Emergence loss (already negative EI)
    emergence_loss = ei_loss

    # 3. Sparsity (encourage confident clustering)
    entropy = -torch.sum(assignment * torch.log(assignment + 1e-10), dim=1).mean()
    sparsity_loss = entropy

    # 4. Balance (avoid all nodes in one cluster)
    cluster_usage = assignment.mean(dim=0)
    uniform_usage = torch.full_like(cluster_usage, 1.0 / assignment.shape[1])
    balance_loss = F.mse_loss(cluster_usage, uniform_usage)

    # Total loss
    total_loss = (
        pred_loss
        + ei_weight * emergence_loss
        + sparsity_weight * sparsity_loss
        + balance_weight * balance_loss
    )

    return {
        'total': total_loss,
        'prediction': pred_loss,
        'emergence': emergence_loss,
        'sparsity': sparsity_loss,
        'balance': balance_loss,
    }
