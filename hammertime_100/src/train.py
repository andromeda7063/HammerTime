"""
Module M7: Training
Responsibility: Train HammerTime ST-GNN with Adam optimizer (weight decay 1e-4),
L1 loss over active masked nodes, early stopping (patience 10), and NaN guard.
Saves checkpoint to best.pt.
Standard: ASD-STE100 Plain English
"""

import csv
import logging
import os
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
    if preds.shape != targets.shape or masks.shape != targets.shape:
        raise ValueError("Predictions, targets, and masks must have the same shape.")
    valid = masks.bool()
    if not valid.any():
        raise ValueError("Cannot train a batch with no valid target nodes.")
    return torch.abs(preds[valid] - targets[valid]).mean()


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

        valid = masks.bool()
        if valid.any():
            diff = torch.abs(preds[valid] - targets[valid])
            total_abs_err += diff.sum().item()
            total_samples += int(valid.sum().item())

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
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if len(train_loader) == 0:
        raise ValueError("Training loader is empty.")
    if len(val_loader) == 0:
        raise ValueError("Validation loader is empty; early stopping needs held-out validation data.")
    scaler = getattr(train_loader.dataset, "feature_scaler", None)

    best_val_mae = float("inf")
    patience_counter = 0
    history = []

    if device.type == "cuda":
        logger.info(f"[*] CUDA IS AVAILABLE & ACTIVE! Utilizing GPU: {torch.cuda.get_device_name(0)} (Max epochs: {max_epochs}, Patience: {patience})...")
    else:
        logger.info(f"[*] Running on CPU (CUDA unavailable). (Max epochs: {max_epochs}, Patience: {patience})...")

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
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"NaN or Inf loss encountered at epoch {epoch}! Check feature scaling in M5."
                )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        val_mae = evaluate_val_mae(model, val_loader, device)
        if not np.isfinite(val_mae):
            raise FloatingPointError(f"Non-finite validation MAE at epoch {epoch}.")

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
            checkpoint = {
                "format_version": 1,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "config": config,
                "feature_scaler": ({"mean_": scaler.mean_.tolist(),
                                    "scale_": scaler.scale_.tolist(),
                                    "var_": scaler.var_.tolist(),
                                    "n_features_in_": int(scaler.n_features_in_)}
                                   if scaler is not None and hasattr(scaler, "mean_") else None),
            }
            temp_path = ckpt_path.with_name(ckpt_path.name + ".tmp")
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, ckpt_path)
            logger.info(f"[*] Best model updated. Saved checkpoint to {ckpt_path} (Val MAE: {val_mae:.4f} s)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (Best Val MAE: {best_val_mae:.4f} s).")
                break

    history_path = Path(t_cfg.get("history_path", ckpt_path.parent / "training_history.csv"))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_mae"])
            writer.writeheader()
            writer.writerows(history)
    if not np.isfinite(best_val_mae):
        raise RuntimeError("Training did not produce a finite validation score; no usable checkpoint was saved.")
    logger.info("Saved training history to %s", history_path)
    return best_val_mae, history


if __name__ == "__main__":
    from dataset import create_dataloaders
    cfg = load_config()
    proc_dir = Path(cfg["session"]["processed_dir"])
    logger.info("M7 train module ready.")
