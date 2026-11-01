"""Standalone turn-key GPU training & evaluation runner for HammerTime.

Enables anyone with a CUDA GPU to train HammerTime v2 in ~3-5 minutes directly
from the frozen prepared_cohort.pkl artifact, without needing FastF1 accounts
or CPU telemetry normalization.

Usage:
    python hammertime_100/train_gpu.py \
        --cohort path/to/prepared_cohort.pkl \
        --config hammertime_100/configs/train_2023_fastf1_season_physical_v2.yaml \
        --output-dir results_gpu \
        --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset import create_dataloaders
from evaluate import (compute_metrics, predict_test_set, validate_checkpoint_edge_scaler,
                      validate_clean_air_physics)
from model import create_model
from prepare_cohort import load_prepared_cohort
from train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("hammertime.gpu_runner")


def run_gpu_training(cohort_path: Path, config_path: Path, output_dir: Path,
                     device_str: str = "auto", epochs: int | None = None,
                     batch_size: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device selection
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda":
        LOG.info("[*] CUDA GPU ACTIVE: %s", torch.cuda.get_device_name(0))
        LOG.info("    CUDA Memory: %.2f GB", torch.cuda.get_device_properties(0).total_memory / (1024**3))
    else:
        LOG.warning("[!] Running on CPU (no CUDA GPU detected).")

    # 2. Load prepared cohort
    LOG.info("Loading prepared cohort from %s...", cohort_path)
    payload, manifest = load_prepared_cohort(cohort_path)
    LOG.info("Loaded cohort: %d snapshots, %d sessions, %d driver slots",
             len(payload["snapshots"]), len(manifest.get("sessions", [])), len(payload["driver_to_idx"]))

    # 3. Load configuration
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if epochs is not None:
        cfg["train"]["max_epochs"] = int(epochs)
    if batch_size is not None:
        cfg["train"]["batch_size"] = int(batch_size)

    cfg["train"]["checkpoint_path"] = str(output_dir / "best.pt")
    cfg["train"]["last_checkpoint_path"] = str(output_dir / "best.last.pt")
    cfg["train"]["history_path"] = str(output_dir / "training_history.csv")

    # 4. Create Dataloaders
    LOG.info("Creating PyTorch Geometric dataloaders (batch_size=%d)...", cfg["train"]["batch_size"])
    snapshots = payload.pop("snapshots")
    labels = payload.pop("labels")
    driver_to_idx = payload["driver_to_idx"]

    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        snapshots, labels, driver_to_idx, cfg)
    del snapshots, labels

    # 5. Initialize Model
    LOG.info("Initializing HammerTime ST-GNN (in_dim=%d, edge_dim=%d, hidden=%d)...",
             cfg["model"]["in_dim"], cfg["model"]["edge_dim"], cfg["model"]["hidden_dim"])
    model = create_model(cfg).to(device)

    # 6. Train Model
    LOG.info("Starting model training for up to %d epochs on %s...",
             cfg["train"]["max_epochs"], device)
    best_val_mae, history = train_model(model, train_loader, val_loader, cfg, device)

    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "training_history.csv", index=False)
    LOG.info("Saved training history to %s", output_dir / "training_history.csv")

    # 7. Evaluate on Held-Out Test Set
    LOG.info("Evaluating best model on held-out test partition (%d samples)...", len(test_loader.dataset))
    best_ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    validate_checkpoint_edge_scaler(best_ckpt, model, test_loader)

    test_preds, test_actuals, test_weights = predict_test_set(model, test_loader, device)
    test_metrics = compute_metrics(test_actuals, test_preds, test_weights)
    LOG.info("[*] HELD-OUT TEST RESULTS: MAE = %.4f s | RMSE = %.4f s | R2 = %.4f",
             test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"])

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Save predictions
    preds_df = pd.DataFrame({
        "actual_delta_sec": test_actuals,
        "predicted_delta_sec": test_preds,
        "weight": test_weights,
        "residual": test_preds - test_actuals,
    })
    preds_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # 8. Generate Visualizations
    _generate_plots(hist_df, preds_df, test_metrics, output_dir)
    LOG.info("[✔] GPU Training and Evaluation complete! Results saved to %s", output_dir)


def _generate_plots(history: pd.DataFrame, preds: pd.DataFrame, metrics: dict, out_dir: Path):
    plt.style.use("dark_background")

    # Plot 1: Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = history["epoch"]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="Train Loss (Smooth L1)", color="#00D2BE", lw=2)
    ax.plot(epochs, history["val_mae"], label="Val MAE (s)", color="#E10600", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MAE (s)")
    ax.set_title("HammerTime v2 Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=150)
    plt.close(fig)

    # Plot 2: Predicted vs Actual
    fig, ax = plt.subplots(figsize=(7, 7))
    actuals = preds["actual_delta_sec"]
    predicted = preds["predicted_delta_sec"]
    ax.scatter(actuals, predicted, alpha=0.2, s=8, color="#00D2BE", label="Held-Out Samples")
    min_v = min(actuals.min(), predicted.min(), -1.0)
    max_v = max(actuals.max(), predicted.max(), 3.0)
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.5, label="Ideal y=x")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Actual Delta T (s)")
    ax.set_ylabel("Predicted Delta T (s)")
    ax.set_title(f"Predicted vs Actual (Test MAE: {metrics['mae']:.4f} s, R²: {metrics['r2']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    # Plot 3: Residuals
    fig, ax = plt.subplots(figsize=(8, 4.5))
    residuals = preds["residual"]
    ax.hist(residuals, bins=60, color="#E10600", alpha=0.75, edgecolor="black")
    ax.axvline(0, color="white", lw=1.5, ls="--")
    ax.set_xlabel("Residual (Predicted - Actual) (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution (Mean Bias: {residuals.mean():+.4f} s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="HammerTime GPU Training Runner")
    parser.add_argument("--cohort", type=Path, required=True,
                        help="Path to prepared_cohort.pkl")
    parser.add_argument("--config", type=Path,
                        default=HERE / "configs/train_2023_fastf1_season_physical_v2.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--output-dir", type=Path, default=Path("results_gpu"),
                        help="Directory to save checkpoint, history, and test evaluation")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    args = parser.parse_args()
    run_gpu_training(
        cohort_path=args.cohort,
        config_path=args.config,
        output_dir=args.output_dir,
        device_str=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
