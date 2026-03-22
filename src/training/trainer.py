# Phase 2: Fixed + Stable Trainer

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
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from ..data.chickenpox_loader import MultiScaleChickenpoxLoader
from ..data.covid_loader import MultiScaleCOVIDLoader
from ..models.cet_epi import CET_Epi
from ..utils.config import load_config
from ..utils.gpu import get_memory_stats, optimize_model, setup_gpu
from .losses import cet_epi_loss


class CET_EpiTrainer:

    def __init__(self, config):
        self.config = config
        self.device = setup_gpu()

        self.horizon = int(getattr(config.model, 'horizon', 1))

        self.exp_name = f"{config.data.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path("checkpoints") / config.data.name / self.exp_name
        self.log_dir = Path("logs/runs") / self.exp_name

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

        with open(self.checkpoint_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        self.core_model = self._build_model().to(self.device)
        self.model = self.core_model

        self.optimizer = torch.optim.Adam(
            self.core_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=getattr(config.training, 'weight_decay', 0.0),
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.train_data, self.test_data = self._load_data()

        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _build_model(self):
        return CET_Epi(
            n_micro=self.config.data.micro_nodes,
            n_macro=self.config.data.macro_nodes,
            in_channels=self.config.data.features,
            hidden_dim=self.config.model.hidden_dim,
            horizon=self.horizon,
        )

    def _load_data(self):
        if self.config.data.name == "chickenpox":
            loader = MultiScaleChickenpoxLoader(lags=4, horizon=self.horizon)
            return loader.get_split(self.config.data.train_ratio)

        raise ValueError("Unknown dataset")

    def train_epoch(self):
        self.model.train()

        # 🔥 Anneal temperature at START
        self.core_model.anneal_temperature(self.epoch, self.config.training.epochs)

        total_loss = 0
        n_batches = 0

        for snapshot in tqdm(self.train_data, desc=f"Epoch {self.epoch}"):

            self.optimizer.zero_grad()

            x = snapshot.x.to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            y = snapshot.y.to(self.device)

            pred, ei_loss, extra = self.model(x, edge_index, return_all=True)

            # safe weights
            ceo_cfg = getattr(self.config.model, "ceo", {})
            ei_w = getattr(ceo_cfg, "ei_weight", 0.1)
            sp_w = getattr(ceo_cfg, "sparsity_weight", 0.01)
            bal_w = getattr(ceo_cfg, "balance_weight", 0.01)

            losses = cet_epi_loss(
                pred,
                y,
                ei_loss,
                extra["S"],
                ei_weight=ei_w,
                sparsity_weight=sp_w,
                balance_weight=bal_w,
            )

            loss = losses["total"]

            if torch.isnan(loss):
                print("⚠️ NaN loss detected, skipping batch")
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.core_model.parameters(),
                getattr(self.config.training, "gradient_clip", 1.0),
            )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_rmse = 0
        total_mae = 0
        n_batches = 0

        for snapshot in self.test_data:
            x = snapshot.x.to(self.device)
            edge_index = snapshot.edge_index.to(self.device)
            y = snapshot.y.to(self.device)

            pred, _, _ = self.model(x, edge_index, return_all=True)

            rmse = torch.sqrt(torch.nn.functional.mse_loss(pred.squeeze(), y))
            mae = torch.nn.functional.l1_loss(pred.squeeze(), y)

            total_rmse += rmse.item()
            total_mae += mae.item()
            n_batches += 1

        return {
            "rmse": total_rmse / n_batches,
            "mae": total_mae / n_batches,
        }

    def train(self):
        print(f"Starting training: {self.exp_name}")

        for epoch in range(self.config.training.epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.scheduler.step(val_metrics["rmse"])

            print(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"RMSE={val_metrics['rmse']:.4f}, "
                f"MAE={val_metrics['mae']:.4f}"
            )

            if val_metrics["rmse"] < self.best_val_loss:
                self.best_val_loss = val_metrics["rmse"]
                self.save_checkpoint(is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= getattr(self.config.training, "early_stopping", 10):
                print("Early stopping triggered")
                break

        print("Training complete")
        return self.checkpoint_dir

    def save_checkpoint(self, is_best=False):
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {"model_state_dict": self.core_model.state_dict()},
            path,
        )
        print(f"Saved model → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = CET_EpiTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
