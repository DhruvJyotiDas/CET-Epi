# Phase 1: Main training loop
"""
CET-Epi Trainer: Full training loop with logging and checkpointing.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[override]
        """No-op TensorBoard writer for minimal environments."""

        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from ..data.chickenpox_loader import MultiScaleChickenpoxLoader
from ..data.covid_loader import MultiScaleCOVIDLoader
from ..models.cet_epi import CET_Epi
from ..utils.config import Config, load_config
from ..utils.gpu import get_memory_stats, optimize_model, setup_gpu
from .losses import cet_epi_loss


class CET_EpiTrainer:
    """
    Main trainer for CET-Epi model.
    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = setup_gpu()
        self.horizon = int(config.model.get('horizon', config.data.get('horizon', 1)))
        self.use_mixed_precision = (
            self.device.type == "cuda"
            and bool(config.hardware.get('mixed_precision', False))
        )
        self.autocast_dtype = torch.bfloat16

        self.exp_name = f"{config.data.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path("checkpoints") / config.data.name / self.exp_name
        self.log_dir = Path("logs/runs") / self.exp_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

        with open(self.checkpoint_dir / "config.json", 'w', encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

        self.core_model = self._build_model().to(self.device)
        self.model = self.core_model
        if config.hardware.get('compile_mode', False):
            self.model = optimize_model(self.core_model, mode="reduce-overhead")

        self.optimizer = torch.optim.Adam(
            self.core_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
        )

        self.train_data, self.test_data = self._load_data()

        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _build_model(self) -> CET_Epi:
        """Build CET-Epi model from config."""
        return CET_Epi(
            n_micro=self.config.data.micro_nodes,
            n_macro=self.config.data.macro_nodes,
            in_channels=self.config.data.features,
            hidden_dim=self.config.model.hidden_dim,
            out_channels=1,
            horizon=self.horizon,
            K=2,
        )

    def _load_data(self):
        """Load and split dataset."""
        if self.config.data.name == "chickenpox":
            loader = MultiScaleChickenpoxLoader(
                lags=self.config.data.get('lags', 4),
                horizon=self.horizon,
            )
            return loader.get_split(self.config.data.train_ratio)

        if self.config.data.name.startswith("covid19"):
            loader = MultiScaleCOVIDLoader(
                country=self.config.data.get('country', 'italy'),
                resolution=self.config.data.get('resolution', 'province'),
                use_mobility=self.config.data.get('use_mobility', False),
                data_dir=self.config.data.get('data_dir', 'data/raw'),
                n_nodes=self.config.data.micro_nodes,
                n_macro=self.config.data.macro_nodes,
                n_features=self.config.data.features,
            )
            return loader.get_split(
                train_ratio=self.config.data.train_ratio,
                window_size=self.config.data.get('lags', 14),
                horizon=self.horizon,
            )

        raise ValueError(f"Unknown dataset: {self.config.data.name}")

    def _autocast_context(self):
        if not self.use_mixed_precision:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _format_targets_like(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1).unsqueeze(-1)
        elif targets.dim() == 2:
            targets = targets.unsqueeze(-1)

        if targets.shape != predictions.shape:
            raise ValueError(
                f"Target shape {tuple(targets.shape)} does not match prediction shape {tuple(predictions.shape)}"
            )

        return targets.to(device=predictions.device, dtype=predictions.dtype)

    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_ei_loss = 0.0
        n_batches = 0

        for snapshot in tqdm(self.train_data, desc=f"Epoch {self.epoch}"):
            self.optimizer.zero_grad()

            x = snapshot.x.to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            edge_attr = snapshot.edge_attr.to(self.device) if snapshot.edge_attr is not None else None
            y = snapshot.y.to(self.device)

            with self._autocast_context():
                predictions, ei_loss, intermediates = self.model(
                    x,
                    edge_index,
                    edge_attr,
                    return_all=True,
                )
                y = self._format_targets_like(predictions, y)
                losses = cet_epi_loss(
                    predictions,
                    y,
                    ei_loss,
                    intermediates['S'],
                    ei_weight=self.config.model.ceo.ei_weight,
                    sparsity_weight=self.config.model.ceo.sparsity_weight,
                    balance_weight=self.config.model.ceo.get('balance_weight', 0.01),
                )

            losses['total'].backward()

            torch.nn.utils.clip_grad_norm_(
                self.core_model.parameters(),
                self.config.training.gradient_clip,
            )

            self.optimizer.step()

            total_loss += losses['total'].item()
            total_pred_loss += losses['prediction'].item()
            total_ei_loss += losses['emergence'].item()
            n_batches += 1

        self.core_model.anneal_temperature(self.epoch, self.config.training.epochs)

        return {
            'loss': total_loss / max(n_batches, 1),
            'pred_loss': total_pred_loss / max(n_batches, 1),
            'ei_loss': total_ei_loss / max(n_batches, 1),
            'lr': self.optimizer.param_groups[0]['lr'],
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on test set."""
        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0
        total_mae = 0.0
        n_batches = 0

        all_predictions = []
        all_targets = []

        for snapshot in self.test_data:
            x = snapshot.x.to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            edge_attr = snapshot.edge_attr.to(self.device) if snapshot.edge_attr is not None else None
            y = snapshot.y.to(self.device)

            predictions, _, _ = self.model(
                x,
                edge_index,
                edge_attr,
                return_all=True,
            )
            y = self._format_targets_like(predictions, y)

            pred_loss = torch.nn.functional.mse_loss(predictions, y)
            rmse = torch.sqrt(pred_loss)
            mae = torch.nn.functional.l1_loss(predictions, y)

            total_loss += pred_loss.item()
            total_rmse += rmse.item()
            total_mae += mae.item()

            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())
            n_batches += 1

        all_preds = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        ei_score = self._compute_validation_ei(all_preds, all_targets)

        return {
            'loss': total_loss / max(n_batches, 1),
            'rmse': total_rmse / max(n_batches, 1),
            'mae': total_mae / max(n_batches, 1),
            'ei_score': ei_score,
        }

    def _compute_validation_ei(self, preds, targets):
        """Compute approximate EI on validation set."""
        pred_variance = torch.var(preds).item()
        return -pred_variance

    def save_checkpoint(self, filename: str | None = None, is_best: bool = False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pt"

        path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.core_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.core_model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from epoch {self.epoch}")

    def log_metrics(self, train_metrics: dict, val_metrics: dict):
        """Log to TensorBoard."""
        self.writer.add_scalar('Train/Loss', train_metrics['loss'], self.epoch)
        self.writer.add_scalar('Train/Pred_Loss', train_metrics['pred_loss'], self.epoch)
        self.writer.add_scalar('Train/EI_Loss', train_metrics['ei_loss'], self.epoch)
        self.writer.add_scalar('Train/LR', train_metrics['lr'], self.epoch)

        self.writer.add_scalar('Val/Loss', val_metrics['loss'], self.epoch)
        self.writer.add_scalar('Val/RMSE', val_metrics['rmse'], self.epoch)
        self.writer.add_scalar('Val/MAE', val_metrics['mae'], self.epoch)
        self.writer.add_scalar('Val/EI_Score', val_metrics['ei_score'], self.epoch)

        mem_stats = get_memory_stats()
        if mem_stats:
            self.writer.add_scalar('System/GPU_Memory_GB', mem_stats['allocated'], self.epoch)

    def train(self):
        """Main training loop."""
        print(f"Starting training: {self.exp_name}")
        print(f"Model parameters: {sum(p.numel() for p in self.core_model.parameters()):,}")

        for epoch in range(self.config.training.epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()

            if epoch % self.config.logging.get('val_interval', 1) == 0:
                val_metrics = self.validate()
                self.scheduler.step(val_metrics['loss'])
                self.log_metrics(train_metrics, val_metrics)

                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(is_best=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.training.early_stopping:
                    print(f"Early stopping at epoch {epoch}")
                    break

                if epoch % self.config.logging.checkpoint_interval == 0:
                    self.save_checkpoint()

                print(
                    f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                    f"Val RMSE={val_metrics['rmse']:.4f}, "
                    f"Val MAE={val_metrics['mae']:.4f}, "
                    f"EI={val_metrics['ei_score']:.4f}"
                )
            else:
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}")

        self.writer.close()
        print("Training completed!")
        return self.checkpoint_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = CET_EpiTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    checkpoint_dir = trainer.train()
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
