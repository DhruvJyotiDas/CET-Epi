# Phase 1: Main training loop
"""
CET-Epi Trainer: Full training loop with logging and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from ..models.cet_epi import CET_Epi
from ..data.chickenpox_loader import MultiScaleChickenpoxLoader
from .losses import cet_epi_loss
from ..utils.config import Config
from ..utils.gpu import setup_gpu, optimize_model, empty_cache, get_memory_stats


class CET_EpiTrainer:
    """
    Main trainer for CET-Epi model.
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = setup_gpu()
        
        # Setup directories
        self.exp_name = f"{config.data.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path("checkpoints") / config.data.name / self.exp_name
        self.log_dir = Path("logs/runs") / self.exp_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # Save config
        with open(self.checkpoint_dir / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Initialize model
        self.model = self._build_model().to(self.device)
        
        # Optimize for MI300X
        if config.hardware.get('compile_mode', False):
            self.model = optimize_model(self.model, mode="reduce-overhead")
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Data
        self.train_data, self.test_data = self._load_data()
        
        # Training state
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
            horizon=1,
            K=2
        )
    
    def _load_data(self):
        """Load and split dataset."""
        if self.config.data.name == "chickenpox":
            loader = MultiScaleChickenpoxLoader(lags=self.config.data.get('lags', 4))
            return loader.get_split(self.config.data.train_ratio)
        else:
            raise ValueError(f"Unknown dataset: {self.config.data.name}")
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_ei_loss = 0.0
        n_batches = 0
        
        # Iterate through temporal snapshots
        for snapshot in tqdm(self.train_data, desc=f"Epoch {self.epoch}"):
            self.optimizer.zero_grad()
            
            # Move to device
            x = snapshot.x.to(self.device)  # [N, T, F]
            edge_index = snapshot.edge_index.to(self.device)
            edge_attr = snapshot.edge_attr.to(self.device) if snapshot.edge_attr is not None else None
            y = snapshot.y.to(self.device)  # [N, 1]
            
            # Forward pass
            predictions, ei_loss, intermediates = self.model(
                x, edge_index, edge_attr, return_all=True
            )
            
            # Reshape targets for prediction
            y = y.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1] to match predictions [N, horizon, 1]
            
            # Compute loss
            losses = cet_epi_loss(
                predictions, y, ei_loss,
                intermediates['S'],
                ei_weight=self.config.model.ceo.ei_weight,
                sparsity_weight=self.config.model.ceo.sparsity_weight
            )
            
            # Backward
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip
            )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += losses['total'].item()
            total_pred_loss += losses['prediction'].item()
            total_ei_loss += losses['emergence'].item()
            n_batches += 1
        
        # Anneal CEO temperature
        self.model.anneal_temperature(self.epoch, self.config.training.epochs)
        
        return {
            'loss': total_loss / n_batches,
            'pred_loss': total_pred_loss / n_batches,
            'ei_loss': total_ei_loss / n_batches,
            'lr': self.optimizer.param_groups[0]['lr']
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
            
            predictions, ei_loss, intermediates = self.model(
                x, edge_index, edge_attr, return_all=True
            )
            
            y = y.unsqueeze(-1).unsqueeze(-1)
            
            # Metrics
            pred_loss = torch.nn.functional.mse_loss(predictions, y)
            rmse = torch.sqrt(pred_loss)
            mae = torch.nn.functional.l1_loss(predictions, y)
            
            total_loss += pred_loss.item()
            total_rmse += rmse.item()
            total_mae += mae.item()
            
            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())
            n_batches += 1
        
        # Concatenate all predictions
        all_preds = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute EI if possible
        ei_score = self._compute_validation_ei(all_preds, all_targets)
        
        return {
            'loss': total_loss / n_batches,
            'rmse': total_rmse / n_batches,
            'mae': total_mae / n_batches,
            'ei_score': ei_score
        }
    
    def _compute_validation_ei(self, preds, targets):
        """Compute approximate EI on validation set."""
        # Simplified: use prediction determinism as proxy
        pred_variance = torch.var(preds).item()
        return -pred_variance  # Lower variance = higher determinism
    
    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pt"
        
        path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }, path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'best_val_loss': self.best_val_loss
            }, best_path)
            
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
        
        # GPU memory
        mem_stats = get_memory_stats()
        if mem_stats:
            self.writer.add_scalar('System/GPU_Memory_GB', mem_stats['allocated'], self.epoch)
    
    def train(self):
        """Main training loop."""
        print(f"Starting training: {self.exp_name}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate every N epochs
            if epoch % self.config.logging.get('val_interval', 1) == 0:
                val_metrics = self.validate()
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Logging
                self.log_metrics(train_metrics, val_metrics)
                
                # Checkpointing
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(is_best=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.training.early_stopping:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                # Regular checkpoint
                if epoch % self.config.logging.checkpoint_interval == 0:
                    self.save_checkpoint()
                
                print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                      f"Val RMSE={val_metrics['rmse']:.4f}, "
                      f"Val MAE={val_metrics['mae']:.4f}, "
                      f"EI={val_metrics['ei_score']:.4f}")
            
            else:
                # Just log training metrics
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}")
        
        self.writer.close()
        print("Training completed!")
        
        return self.checkpoint_dir


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create trainer
    trainer = CET_EpiTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    checkpoint_dir = trainer.train()
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()