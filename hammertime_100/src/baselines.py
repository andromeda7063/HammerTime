"""
Module M9: Baselines
Responsibility: Train isolated single-car baselines (XGBoost and LSTM)
strictly on the identical split from M5 to benchmark HammerTime ST-GNN.
Standard: ASD-STE100 Plain English
"""

import logging
from pathlib import Path
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
        r2 = float(r2_score(y_test, preds))

        logger.info(f"XGBoost Baseline Results | MAE: {mae:.4f} s | R²: {r2:.4f}")
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
    r2 = float(r2_score(y_test, preds))

    logger.info(f"LSTM Baseline Results | MAE: {mae:.4f} s | R²: {r2:.4f}")
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


if __name__ == "__main__":
    cfg = load_config()
    logger.info("M9 baselines module ready.")
