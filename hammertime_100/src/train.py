"""
Module M7: Training
Responsibility: Train HammerTime ST-GNN with Adam optimizer (weight decay 1e-4),
L1 loss over active masked nodes, early stopping (patience 10), and NaN guard.
Saves checkpoint to best.pt.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

from model import HammerTimeSTGNN, create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.train")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def masked_l1_loss(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Computes L1 loss strictly over active, unmasked nodes:
    L = (1 / N_active) * sum |y_hat - y|
    """
    diff = torch.abs(preds - targets)
    masked_diff = diff * masks.float()
    total_active = masks.float().sum()
    if total_active > 0:
        return masked_diff.sum() / total_active
    return torch.tensor(0.0, device=preds.device, requires_grad=True)


@torch.no_grad()
def evaluate_val_mae(model: nn.Module, val_loader, device: torch.device) -> float:
    """
    Computes validation MAE across all active driver predictions.
    """
    model.eval()
    total_abs_err = 0.0
    total_samples = 0

    for batch_snaps, targets, masks in val_loader:
        targets = targets.to(device)
        masks = masks.to(device)
        preds = model(batch_snaps)

        diff = torch.abs(preds - targets) * masks.float()
        total_abs_err += diff.sum().item()
        total_samples += masks.float().sum().item()

    return total_abs_err / max(total_samples, 1)


def train_model(model: nn.Module, train_loader, val_loader, config: dict, device: torch.device):
    """
    Executes training loop with early stopping, L1 loss, Adam weight decay, and NaN detection.
    """
    t_cfg = config["train"]
    lr = float(t_cfg.get("lr", 0.001))
    wd = float(t_cfg.get("weight_decay", 0.0001))
    max_epochs = int(t_cfg.get("max_epochs", 100))
    patience = int(t_cfg.get("early_stopping_patience", 10))
    ckpt_path = Path(t_cfg.get("checkpoint_path", "best.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    logger.info(f"Starting ST-GNN training on {device} (Max epochs: {max_epochs}, Patience: {patience})...")

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_snaps, targets, masks in train_loader:
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(batch_snaps)

            loss = masked_l1_loss(preds, targets, masks)

            # Immutable Law 9 / Failure Case: NaN Guard
            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError(
                    f"NaN or Inf loss encountered at epoch {epoch}! Check feature scaling in M5."
                )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        val_mae = evaluate_val_mae(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_mae": val_mae
        })

        logger.info(f"Epoch {epoch:03d} | Train L1: {avg_train_loss:.4f} s | Val MAE: {val_mae:.4f} s")

        # Early stopping and checkpointing
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": val_mae,
                "config": config
            }, ckpt_path)
            logger.info(f"[*] Best model updated. Saved checkpoint to {ckpt_path} (Val MAE: {val_mae:.4f} s)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (Best Val MAE: {best_val_mae:.4f} s).")
                break

    return best_val_mae, history


if __name__ == "__main__":
    from dataset import create_dataloaders
    cfg = load_config()
    proc_dir = Path(cfg["session"]["processed_dir"])
    logger.info("M7 train module ready.")
