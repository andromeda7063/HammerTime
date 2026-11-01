"""
Module M7: Training
Responsibility: Train HammerTime ST-GNN with Adam optimizer (weight decay 1e-4),
masked MAE or MSE over active nodes, MAE-based early stopping, and NaN guard.
Saves checkpoint to best.pt.
Standard: ASD-STE100 Plain English
"""

import csv
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
import torch.nn as nn
import yaml

from model import HammerTimeSTGNN, create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.train")


def _tensor_digest_update(digest, tensor) -> None:
    if tensor is None:
        digest.update(b"<none>")
        return
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())


def _training_data_signature(train_loader, val_loader) -> str:
    """Fingerprint train/validation samples and unique graph inputs, never test data."""
    digest = hashlib.sha256()
    unique_snapshots = {}
    for split_name, loader in (("train", train_loader), ("validation", val_loader)):
        dataset = getattr(loader, "dataset", None)
        samples = getattr(dataset, "samples", None)
        if samples is None:
            # Keep historical training usable with arbitrary iterable loaders.
            # Such a loader cannot provide the strong data-compatibility check
            # required to resume, so its last checkpoint is marked non-resumable.
            return None
        digest.update(split_name.encode("ascii"))
        digest.update(str(len(samples)).encode("ascii"))
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError("Resumable training dataset samples must be mappings.")
            identity = {
                "race_id": sample.get("race_id"),
                "input_start_time": sample.get("input_start_time"),
                "target_time": sample.get("target_time"),
                "time_sec": sample.get("time_sec"),
            }
            digest.update(json.dumps(identity, sort_keys=True, default=str,
                                     separators=(",", ":")).encode("utf-8"))
            digest.update(str(index).encode("ascii"))
            _tensor_digest_update(digest, sample.get("target"))
            _tensor_digest_update(digest, sample.get("mask"))
            for snapshot in sample.get("snapshots", ()):
                race_id = str(getattr(snapshot, "race_id", ""))
                time_sec = float(getattr(snapshot, "time_sec", float("nan")))
                key = (race_id, time_sec)
                if key not in unique_snapshots:
                    unique_snapshots[key] = snapshot

    for key in sorted(unique_snapshots):
        snapshot = unique_snapshots[key]
        digest.update(json.dumps(key, separators=(",", ":")).encode("utf-8"))
        for field in ("x", "mask", "position_valid", "edge_index", "edge_attr",
                      "edge_index_race_progress", "edge_attr_race_progress",
                      "edge_index_physical_circuit", "edge_attr_physical_circuit"):
            _tensor_digest_update(digest, getattr(snapshot, field, None))
        for field in ("driver_names", "position_coverage", "blocked_neighbor_count"):
            digest.update(repr(getattr(snapshot, field, None)).encode("utf-8"))
    return digest.hexdigest()


def _scaler_signature(train_loader) -> dict:
    dataset = getattr(train_loader, "dataset", None)
    feature_scaler = getattr(dataset, "feature_scaler", None)
    edge_scaler = getattr(dataset, "edge_feature_scaler", None)

    def standard_state(scaler):
        if scaler is None:
            return None
        fields = {}
        for name in ("mean_", "scale_", "var_"):
            value = getattr(scaler, name, None)
            if value is not None:
                fields[name] = np.asarray(value).tolist()
        if hasattr(scaler, "n_features_in_"):
            fields["n_features_in_"] = int(scaler.n_features_in_)
        return fields

    edge_state = edge_scaler.metadata() if edge_scaler is not None else None
    return {"feature": standard_state(feature_scaler), "edge": edge_state}


def _resume_signature(model, train_loader, val_loader, config: dict) -> dict:
    try:
        config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False, default=str).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Training config cannot be fingerprinted for resume.") from exc
    model_shapes = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in model.state_dict().items()}
    scaler = _scaler_signature(train_loader)
    scaler_bytes = json.dumps(scaler, sort_keys=True, separators=(",", ":"),
                              allow_nan=False, default=str).encode("utf-8")
    return {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "data_sha256": _training_data_signature(train_loader, val_loader),
        "scaler_sha256": hashlib.sha256(scaler_bytes).hexdigest(),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_state_shapes": model_shapes,
    }


def _loader_generator(loader):
    loader_generator = getattr(loader, "generator", None)
    if loader_generator is None:
        loader_generator = getattr(getattr(loader, "sampler", None), "generator", None)
    return loader_generator


def _capture_rng_state(train_loader, val_loader) -> dict:
    train_generator = _loader_generator(train_loader)
    val_generator = _loader_generator(val_loader)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader_generator": train_generator.get_state() if train_generator is not None else None,
        "val_loader_generator": val_generator.get_state() if val_generator is not None else None,
    }


def _restore_rng_state(state: dict, train_loader, val_loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    train_generator = _loader_generator(train_loader)
    val_generator = _loader_generator(val_loader)
    if train_generator is not None and state.get("train_loader_generator") is not None:
        train_generator.set_state(state["train_loader_generator"].cpu())
    if val_generator is not None and state.get("val_loader_generator") is not None:
        val_generator.set_state(state["val_loader_generator"].cpu())


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def masked_l1_loss(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Compute mean absolute error strictly over active, unmasked nodes.
    """
    if preds.shape != targets.shape or masks.shape != targets.shape:
        raise ValueError("Predictions, targets, and masks must have the same shape.")
    valid = masks.bool()
    if not valid.any():
        raise ValueError("Cannot train a batch with no valid target nodes.")
    return torch.abs(preds[valid] - targets[valid]).mean()


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error over active target nodes only."""
    if preds.shape != targets.shape or masks.shape != targets.shape:
        raise ValueError("Predictions, targets, and masks must have the same shape.")
    valid = masks.bool()
    if not valid.any():
        raise ValueError("Cannot train a batch with no valid target nodes.")
    return torch.square(preds[valid] - targets[valid]).mean()


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

    if total_samples == 0:
        raise ValueError("Validation loader has no valid target nodes.")
    return total_abs_err / total_samples


def train_model(model: nn.Module, train_loader, val_loader, config: dict, device: torch.device,
                *, resume_from: str | Path | None = None):
    """
    Train with the configured masked MAE or MSE objective; select checkpoints by validation MAE.
    """
    t_cfg = config["train"]
    loss_name = str(t_cfg.get("loss", "mae")).casefold()
    loss_functions = {"mae": masked_l1_loss, "mse": masked_mse_loss}
    if loss_name not in loss_functions:
        raise ValueError("train.loss must be 'mae' or 'mse'.")
    loss_function = loss_functions[loss_name]
    lr = float(t_cfg.get("lr", 0.001))
    wd = float(t_cfg.get("weight_decay", 0.0001))
    max_epochs = int(t_cfg.get("max_epochs", 100))
    patience = int(t_cfg.get("early_stopping_patience", 10))
    ckpt_path = Path(t_cfg.get("checkpoint_path", "best.pt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = Path(t_cfg.get(
        "last_checkpoint_path",
        ckpt_path.with_name(f"{ckpt_path.stem}.last{ckpt_path.suffix}"),
    ))
    if resume_from is not None and Path(resume_from).resolve() == ckpt_path.resolve():
        raise ValueError("resume_from must point to a last-epoch checkpoint, not best.pt.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if len(train_loader) == 0:
        raise ValueError("Training loader is empty.")
    if len(val_loader) == 0:
        raise ValueError("Validation loader is empty; early stopping needs held-out validation data.")
    scaler = getattr(train_loader.dataset, "feature_scaler", None)
    resume_signature = _resume_signature(model, train_loader, val_loader, config)
    if resume_from is not None and resume_signature["data_sha256"] is None:
        raise ValueError("Cannot resume: train/validation datasets do not expose fingerprintable sample records.")

    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    first_epoch = 1
    run_id = uuid.uuid4().hex
    best_checkpoint_state = None

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("checkpoint_kind") != "last_epoch" or checkpoint.get("format_version") != 2:
            raise ValueError("Resume file is not a supported complete-epoch training checkpoint.")
        if checkpoint.get("resume_signature") != resume_signature:
            raise ValueError(
                "Resume checkpoint config, train/validation data, scaler, or model shape does not match."
            )
        completed_epoch = int(checkpoint.get("completed_epoch", 0))
        if completed_epoch < 1 or completed_epoch > max_epochs:
            raise ValueError("Resume checkpoint epoch is outside the configured training budget.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_mae = float(checkpoint["best_val_mae"])
        best_epoch = int(checkpoint["best_epoch"])
        patience_counter = int(checkpoint["patience_counter"])
        history = list(checkpoint["history"])
        run_id = checkpoint.get("run_id")
        best_checkpoint_state = checkpoint.get("best_checkpoint_state")
        if not run_id or not isinstance(best_checkpoint_state, dict):
            raise ValueError("Resume checkpoint is missing its run identity or recoverable best state.")
        if len(history) != completed_epoch:
            raise ValueError("Resume checkpoint history does not match its completed epoch.")
        if checkpoint.get("best_checkpoint_path") != str(ckpt_path.resolve()):
            raise ValueError("Resume checkpoint points to a different best-checkpoint path.")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Best checkpoint from the interrupted run is missing: {ckpt_path}")
        best_checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        disk_best_epoch = int(best_checkpoint.get("epoch", -1))
        if (best_checkpoint.get("resume_signature") != resume_signature
                or best_checkpoint.get("run_id") != run_id):
            raise ValueError("Best checkpoint does not match the last-epoch resume state.")
        if disk_best_epoch == completed_epoch + 1:
            # Crash after atomically replacing best.pt but before committing the
            # same epoch's last checkpoint. Roll best.pt back to the best state
            # known to the last fully committed epoch, then replay this epoch.
            if (int(best_checkpoint_state.get("epoch", -1)) != best_epoch
                    or best_checkpoint_state.get("resume_signature") != resume_signature
                    or best_checkpoint_state.get("run_id") != run_id):
                raise ValueError("Last checkpoint cannot recover the ahead-of-resume best checkpoint.")
            _atomic_torch_save(best_checkpoint_state, ckpt_path)
        elif disk_best_epoch != best_epoch:
            raise ValueError("Best checkpoint epoch is inconsistent with the last-epoch resume state.")
        _restore_rng_state(checkpoint["rng_state"], train_loader, val_loader)
        first_epoch = completed_epoch + 1
        if patience_counter >= patience:
            first_epoch = max_epochs + 1
        logger.info("Resuming after complete epoch %d from %s (best Val MAE %.6f s)",
                    completed_epoch, resume_path, best_val_mae)

    if device.type == "cuda":
        logger.info(f"[*] CUDA IS AVAILABLE & ACTIVE! Utilizing GPU: {torch.cuda.get_device_name(0)} (Max epochs: {max_epochs}, Patience: {patience})...")
    else:
        logger.info(f"[*] Running on CPU (CUDA unavailable). (Max epochs: {max_epochs}, Patience: {patience})...")

    for epoch in range(first_epoch, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_snaps, targets, masks in train_loader:
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(batch_snaps)

            loss = loss_function(preds, targets, masks)

            # Immutable Law 9 / Failure Case: NaN Guard
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"NaN or Inf loss encountered at epoch {epoch}! Check feature scaling in M5."
                )

            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Non-finite gradient for {name} at epoch {epoch}.")
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

        train_unit = "s" if loss_name == "mae" else "s²"
        train_label = "L1" if loss_name == "mae" else "MSE"
        logger.info(f"Epoch {epoch:03d} | Train {train_label}: {avg_train_loss:.4f} {train_unit} | Val MAE: {val_mae:.4f} s")

        # Early stopping and checkpointing
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "format_version": 1,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "config": config,
                "resume_signature": resume_signature,
                "run_id": run_id,
                "feature_scaler": ({"mean_": scaler.mean_.tolist(),
                                    "scale_": scaler.scale_.tolist(),
                                    "var_": scaler.var_.tolist(),
                                    "n_features_in_": int(scaler.n_features_in_)}
                                   if scaler is not None and hasattr(scaler, "mean_") else None),
            }
            best_checkpoint_state = copy.deepcopy(checkpoint)
            _atomic_torch_save(checkpoint, ckpt_path)
            logger.info(f"[*] Best model updated. Saved checkpoint to {ckpt_path} (Val MAE: {val_mae:.4f} s)")
        else:
            patience_counter += 1

        # Persist only at complete-epoch boundaries. This file is separate from
        # best.pt so a restart can continue Adam and early stopping exactly.
        _atomic_torch_save({
            "format_version": 2,
            "checkpoint_kind": "last_epoch",
            "completed_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "history": history,
            "best_checkpoint_path": str(ckpt_path.resolve()),
            "resume_signature": resume_signature,
            "run_id": run_id,
            "best_checkpoint_state": best_checkpoint_state,
            "rng_state": _capture_rng_state(train_loader, val_loader),
        }, last_ckpt_path)
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
        # A resume from an already completed budget can return its prior score.
        if resume_from is None or best_epoch == 0:
            raise RuntimeError("Training did not produce a finite validation score; no usable checkpoint was saved.")
    logger.info("Saved training history to %s", history_path)
    return best_val_mae, history


if __name__ == "__main__":
    from dataset import create_dataloaders
    cfg = load_config()
    proc_dir = Path(cfg["session"]["processed_dir"])
    logger.info("M7 train module ready.")
