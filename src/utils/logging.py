"""
Experiment logging utilities.
"""

import csv
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class ExperimentLogger:
    """
    Comprehensive logging for CET-Epi experiments.
    Handles metrics, hyperparameters, and artifacts.
    """
    
    def __init__(self, exp_dir: str, exp_name: str = None):
        self.exp_dir = Path(exp_dir)
        if exp_name:
            self.exp_dir = self.exp_dir / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.exp_dir / "metrics.csv"
        self.params_file = self.exp_dir / "params.json"
        self.artifacts_dir = self.exp_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Initialize metrics CSV
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'split', 'metric', 'value'])
        
        self.start_time = datetime.now()
        
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=2, default=str)
            
    def log_metric(self, epoch: int, split: str, metric: str, value: float):
        """Log single metric."""
        timestamp = datetime.now().isoformat()
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, epoch, split, metric, value])
            
    def log_metrics(self, epoch: int, split: str, metrics: Dict[str, float]):
        """Log multiple metrics."""
        for metric, value in metrics.items():
            self.log_metric(epoch, split, metric, value)
            
    def log_artifact(self, name: str, data: Any, format: str = "pkl"):
        """Save artifact (predictions, embeddings, etc.)."""
        path = self.artifacts_dir / f"{name}.{format}"
        
        if format == "pkl":
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif format == "json":
            with open(path, 'w') as f:
                json.dump(data, f, default=str)
        elif format == "pt":
            torch.save(data, path)
            
        return path
    
    def get_metrics_history(self) -> Dict[str, list]:
        """Load all metrics as history."""
        history = {}
        with open(self.metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['split']}_{row['metric']}"
                if key not in history:
                    history[key] = []
                history[key].append(float(row['value']))
        return history
    
    def summary(self) -> str:
        """Generate experiment summary."""
        duration = datetime.now() - self.start_time
        
        summary = [
            "=" * 60,
            "EXPERIMENT SUMMARY",
            "=" * 60,
            f"Directory: {self.exp_dir}",
            f"Duration: {duration}",
            f"Metrics logged: {len(list(open(self.metrics_file))) - 1}",
            "=" * 60
        ]
        
        return "\n".join(summary)
    
    def save_checkpoint(self, 
                        model_state: Dict,
                        optimizer_state: Dict,
                        epoch: int,
                        metrics: Dict[str, float],
                        is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save latest
        path = self.exp_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            best_path = self.exp_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            
        # Save epoch-specific (every 10 epochs)
        if epoch % 10 == 0:
            epoch_path = self.exp_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, epoch_path)
            
        return path
