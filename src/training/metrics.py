"""
Evaluation metrics for CET-Epi.
"""

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error


class EpidemicMetrics:
    """Comprehensive metrics for epidemic forecasting."""
    
    @staticmethod
    def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Root Mean Squared Error."""
        return torch.sqrt(torch.nn.functional.mse_loss(predictions, targets)).item()
    
    @staticmethod
    def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Mean Absolute Error."""
        return torch.nn.functional.l1_loss(predictions, targets).item()
    
    @staticmethod
    def mape(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = targets != 0
        if mask.sum() == 0:
            return 0.0
        return (torch.abs((targets[mask] - predictions[mask]) / targets[mask]).mean() * 100).item()
    
    @staticmethod
    def r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """R-squared coefficient."""
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        return r2_score(target_np, pred_np)
    
    @staticmethod
    def peak_timing_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Error in predicting peak timing (critical for epidemics).
        Returns average days/weeks off from actual peak.
        """
        # Assuming temporal dimension is present
        if predictions.dim() < 2:
            return 0.0
        
        pred_peaks = torch.argmax(predictions, dim=0)
        target_peaks = torch.argmax(targets, dim=0)
        errors = torch.abs(pred_peaks.float() - target_peaks.float())
        return errors.mean().item()
    
    @staticmethod
    def peak_magnitude_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Error in predicting peak magnitude."""
        pred_max = predictions.max()
        target_max = targets.max()
        return torch.abs(pred_max - target_max).item()
    
    @classmethod
    def compute_all(cls, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute all metrics."""
        return {
            'rmse': cls.rmse(predictions, targets),
            'mae': cls.mae(predictions, targets),
            'mape': cls.mape(predictions, targets),
            'r2': cls.r2(predictions, targets),
            'peak_timing_error': cls.peak_timing_error(predictions, targets),
            'peak_magnitude_error': cls.peak_magnitude_error(predictions, targets)
        }