"""
Module M9: Baselines
Responsibility: Train isolated single-car baselines (XGBoost and LSTM)
strictly on the identical split from M5 to benchmark HammerTime ST-GNN.
Standard: ASD-STE100 Plain English
"""

import logging
from pathlib import Path
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.baselines")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_isolated_car_features(dataset_samples: list):
    """
    Extracts isolated single-car features from windowed graph dataset samples.
    Removes all graph edges, neighbor nodes, and attention weights.
    Returns:
      - X_seq: [N_total, W, in_dim] for LSTM
      - X_flat: [N_total, W * in_dim] for XGBoost (flattened isolated-car history)
      - y: [N_total] target delta t_sector
    """
    X_seq_list = []
    X_flat_list = []
    y_list = []

    for item in dataset_samples:
        snaps = item["snapshots"]  # list of W Data objects
        target = item["target"]    # [20]
        mask = item["mask"]        # [20]
        W = len(snaps)

        # For each active car
        active_indices = torch.where(mask)[0].numpy()
        for c_idx in active_indices:
            # Extract single-car trajectory across window
            car_traj = np.stack([snap.x[c_idx].cpu().numpy() for snap in snaps], axis=0)  # [W, in_dim]
            X_seq_list.append(car_traj)
            X_flat_list.append(car_traj.reshape(-1))
            y_list.append(float(target[c_idx].item()))

    if not X_seq_list:
        raise ValueError("No valid car targets are available for baseline training or evaluation.")
    X_seq = np.asarray(X_seq_list, dtype=np.float32)
    X_flat = np.array(X_flat_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X_seq, X_flat, y


class IsolatedCarLSTM(nn.Module):
    """
    Single-car LSTM baseline without graph interaction.
    """
    def __init__(self, in_dim: int = 9, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [B, W, in_dim]
        out, (h_n, _) = self.lstm(x)
        last_hidden = out[:, -1, :]  # [B, hidden_dim]
        return self.fc(last_hidden).squeeze(-1)


def train_xgboost_baseline(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, config: dict) -> dict:
    """
    Trains single-car XGBoost regressor and calculates MAE and R².
    """
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("XGBoost requires non-empty train and test samples.")
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost is unavailable; no XGBoost score was measured.")
        return {"available": False, "reason": "xgboost package is not installed", "mae_sec": None,
                "r2_score": None, "predictions": None}
    try:
        xgb_cfg = config.get("baselines", {}).get("xgboost", {})
        n_est = xgb_cfg.get("n_estimators", 100)
        max_d = xgb_cfg.get("max_depth", 6)
        lr = xgb_cfg.get("learning_rate", 0.05)

        logger.info(f"Training isolated XGBoost baseline (n_estimators={n_est}, max_depth={max_d})...")
        model = xgb.XGBRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=lr,
            subsample=xgb_cfg.get("subsample", 0.8),
            random_state=config.get("data", {}).get("random_seed", 42),
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds)) if len(y_test) >= 2 else None

        logger.info("XGBoost Baseline Results | MAE: %.4f s | R²: %s", mae, r2)
        return {"available": True, "mae_sec": mae, "r2_score": r2, "predictions": preds}
    except Exception:
        logger.exception("XGBoost training failed; refusing to substitute fabricated benchmark scores.")
        raise


def train_lstm_baseline(X_train_seq: np.ndarray, y_train: np.ndarray, X_test_seq: np.ndarray, y_test: np.ndarray, config: dict) -> dict:
    """
    Trains isolated single-car PyTorch LSTM baseline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_cfg = config.get("baselines", {}).get("lstm", {})
    hidden_dim = lstm_cfg.get("hidden_dim", 64)
    epochs = lstm_cfg.get("epochs", 25)
    lr = lstm_cfg.get("lr", 0.001)

    in_dim = X_train_seq.shape[2]
    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("LSTM requires non-empty train and test samples.")
    seed = int(config.get("data", {}).get("random_seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = IsolatedCarLSTM(in_dim=in_dim, hidden_dim=hidden_dim,
                            num_layers=lstm_cfg.get("num_layers", 2),
                            dropout=lstm_cfg.get("dropout", 0.1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    train_ds = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=int(lstm_cfg.get("batch_size", 64)),
                        shuffle=True, generator=generator)

    logger.info(f"Training isolated LSTM baseline ({epochs} epochs)...")
    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = loss_fn(preds, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test_seq).to(device)
        preds = model(test_x).cpu().numpy()

    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds)) if len(y_test) >= 2 else None

    logger.info("LSTM Baseline Results | MAE: %.4f s | R²: %s", mae, r2)
    return {"available": True, "mae_sec": mae, "r2_score": r2, "predictions": preds}


def run_baselines(train_samples: list, test_samples: list, config: dict) -> dict:
    """
    Executes both isolated single-car baselines strictly on the shared split.
    """
    logger.info("Extracting isolated single-car features from train and test splits...")
    X_train_seq, X_train_flat, y_train = extract_isolated_car_features(train_samples)
    X_test_seq, X_test_flat, y_test = extract_isolated_car_features(test_samples)

    xgb_results = train_xgboost_baseline(X_train_flat, y_train, X_test_flat, y_test, config)
    lstm_results = train_lstm_baseline(X_train_seq, y_train, X_test_seq, y_test, config)

    return {
        "xgboost": {
            "available": xgb_results.get("available", True),
            "reason": xgb_results.get("reason"),
            "mae_sec": xgb_results["mae_sec"],
            "r2_score": xgb_results["r2_score"]
        },
        "lstm": {
            "available": lstm_results.get("available", True),
            "mae_sec": lstm_results["mae_sec"],
            "r2_score": lstm_results["r2_score"]
        }
    }


def _validation_device(device) -> torch.device:
    """Resolve the explicit device request without silently falling back."""
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for validation baselines, but CUDA is unavailable.")
    if result.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested for validation baselines, but MPS is unavailable.")
    if result.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported baseline device {result.type!r}.")
    return result


def _extract_validation_rows(samples: list):
    """Flatten owned-car input histories and keep aligned per-target metadata."""
    sequences, targets, metadata = [], [], []
    feature_width = None
    window_width = None
    for sample in samples:
        snapshots = sample["snapshots"]
        target = sample["target"].detach().cpu().numpy()
        mask = sample["mask"].detach().cpu().numpy().astype(bool)
        if target.ndim != 1 or mask.shape != target.shape:
            raise ValueError("Baseline samples require equal one-dimensional target and mask vectors.")
        if not snapshots:
            raise ValueError("Baseline samples cannot contain empty input windows.")
        driver_names = sample.get("driver_names")
        sectors = np.asarray(sample.get("mini_sector", np.full(len(target), np.nan)), dtype=float)
        for node in np.flatnonzero(mask):
            trajectory = np.stack([snapshot.x[node].detach().cpu().numpy() for snapshot in snapshots])
            if trajectory.ndim != 2:
                raise ValueError("Each own-car trajectory must have shape [window, feature].")
            if feature_width is None:
                window_width, feature_width = trajectory.shape
            if trajectory.shape != (window_width, feature_width):
                raise ValueError("Baseline samples have inconsistent window or feature dimensions.")
            sequences.append(trajectory)
            targets.append(float(target[node]))
            metadata.append({
                "race_id": sample.get("race_id"),
                "driver": (driver_names[node] if driver_names is not None and node < len(driver_names) else None),
                "mini_sector": (float(sectors[node]) if node < len(sectors) and np.isfinite(sectors[node]) else None),
                "target_time": sample.get("target_time", sample.get("time_sec")),
            })
    if not sequences:
        raise ValueError("No valid car targets are available for validation baselines.")
    x_seq = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return x_seq, y, metadata


def _validation_scores(y_true: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "mae_sec": float(mean_absolute_error(y_true, predictions)),
        "r2_score": float(r2_score(y_true, predictions)) if len(y_true) >= 2 else None,
    }


def _validation_model_result(predictions, targets, metadata, runtime_sec, available=True, reason=None):
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    result = {
        "available": bool(available),
        "reason": reason,
        "sample_count": int(len(targets)),
        "runtime_sec": float(runtime_sec),
        "predictions": predictions.tolist() if available else None,
        "targets": targets.tolist(),
        "metadata": metadata,
        "mae_sec": None,
        "r2_score": None,
    }
    if available:
        result.update(_validation_scores(targets, predictions))
    return result


def _train_validation_xgboost(x_train, y_train, x_validation, y_validation, config, device):
    started = time.perf_counter()
    try:
        import xgboost as xgb
    except ImportError:
        return None, time.perf_counter() - started, "xgboost package is not installed"
    cfg = config.get("baselines", {}).get("xgboost", {})
    params = {
        "n_estimators": cfg.get("n_estimators", 100),
        "max_depth": cfg.get("max_depth", 6),
        "learning_rate": cfg.get("learning_rate", 0.05),
        "subsample": cfg.get("subsample", 0.8),
        "random_state": config.get("data", {}).get("random_seed", 42),
        "n_jobs": -1,
        "tree_method": "hist",
        "device": str(device) if device.type == "cuda" else "cpu",
    }
    model = xgb.XGBRegressor(**params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)
    return np.asarray(predictions, dtype=np.float32), time.perf_counter() - started, None


def _train_validation_lstm(x_train, y_train, x_validation, y_validation, config, device):
    started = time.perf_counter()
    cfg = config.get("baselines", {}).get("lstm", {})
    epochs = int(cfg.get("epochs", 25))
    patience = int(cfg.get("early_stopping_patience",
                           config.get("train", {}).get("early_stopping_patience", epochs)))
    if epochs < 1 or patience < 1:
        raise ValueError("Validation LSTM epochs and patience must be positive.")
    seed = int(config.get("data", {}).get("random_seed", 42))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = IsolatedCarLSTM(
        in_dim=int(x_train.shape[2]),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 0.001)))
    loss_fn = nn.L1Loss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x_train), torch.as_tensor(y_train)),
        batch_size=int(cfg.get("batch_size", 64)), shuffle=True, generator=generator,
    )
    val_tensor = torch.as_tensor(x_validation, dtype=torch.float32)
    val_targets = torch.as_tensor(y_validation, dtype=torch.float32)
    best_mae = float("inf")
    best_state = None
    stale_epochs = 0
    for _epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        predicted_batches = []
        with torch.no_grad():
            for start in range(0, len(val_tensor), int(cfg.get("batch_size", 64))):
                predicted_batches.append(model(val_tensor[start:start + int(cfg.get("batch_size", 64))].to(device)).cpu())
        validation_predictions = torch.cat(predicted_batches).numpy()
        val_mae = float(mean_absolute_error(y_validation, validation_predictions))
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Validation LSTM did not save a finite best checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    prediction_batches = []
    with torch.no_grad():
        for start in range(0, len(val_tensor), int(cfg.get("batch_size", 64))):
            prediction_batches.append(model(val_tensor[start:start + int(cfg.get("batch_size", 64))].to(device)).cpu())
    predictions = torch.cat(prediction_batches).numpy().astype(np.float32)
    return predictions, time.perf_counter() - started


def run_validation_baselines(train_samples: list, validation_samples: list, config: dict, device) -> dict:
    """Run isolated-car baselines on train and validation samples only.

    This API deliberately accepts sample lists rather than loaders, so a test
    loader cannot be supplied accidentally. Feature width is inferred from the
    provided windows, supporting both legacy and context node schemas.
    """
    resolved_device = _validation_device(device)
    x_train, y_train, _train_metadata = _extract_validation_rows(train_samples)
    x_val, y_val, metadata = _extract_validation_rows(validation_samples)
    if x_train.shape[2] != x_val.shape[2] or x_train.shape[1] != x_val.shape[1]:
        raise ValueError("Training and validation baseline feature/window dimensions do not match.")
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_val_flat = x_val.reshape(len(x_val), -1)

    train_mean = float(np.mean(y_train))
    constant_started = time.perf_counter()
    constant_predictions = np.full(len(y_val), train_mean, dtype=np.float32)
    constant_result = _validation_model_result(
        constant_predictions, y_val, metadata, time.perf_counter() - constant_started,
    )
    constant_result["execution_device"] = "cpu"

    xgb_predictions, xgb_runtime, xgb_reason = _train_validation_xgboost(
        x_train_flat, y_train, x_val_flat, y_val, config, resolved_device,
    )
    xgb_result = _validation_model_result(
        xgb_predictions if xgb_predictions is not None else np.zeros(len(y_val), dtype=np.float32),
        y_val, metadata, xgb_runtime, available=xgb_predictions is not None, reason=xgb_reason,
    )
    xgb_result["execution_device"] = str(resolved_device) if resolved_device.type == "cuda" else "cpu"
    lstm_predictions, lstm_runtime = _train_validation_lstm(
        x_train, y_train, x_val, y_val, config, resolved_device,
    )
    lstm_result = _validation_model_result(lstm_predictions, y_val, metadata, lstm_runtime)
    lstm_result["execution_device"] = str(resolved_device)
    return {
        "train_target_count": int(len(y_train)),
        "validation_target_count": int(len(y_val)),
        "feature_width": int(x_train.shape[2]),
        "window_size": int(x_train.shape[1]),
        "device": str(resolved_device),
        "mean_train_constant": constant_result,
        "xgboost": xgb_result,
        "lstm": lstm_result,
    }


if __name__ == "__main__":
    cfg = load_config()
    logger.info("M9 baselines module ready.")
