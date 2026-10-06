"""
Module M5: Dataset and Split
Responsibility: Build rolling temporal graph windows (W=10, horizon K=5).
Enforce zero-leakage split (race-based or sequential time blocks).
Fit StandardScaler strictly on training set.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.dataset")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class STGraphDataset(Dataset):
    """
    Dataset yielding rolling windows of W snapshots and future target vector y at t+K.
    """
    def __init__(self, samples: list):
        """
        samples: list of dicts:
          - 'snapshots': list of W PyG Data objects
          - 'target': torch.Tensor of shape [20]
          - 'mask': torch.Tensor of shape [20] (active valid cars)
          - 'time_sec': float timestamp of current prediction
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["snapshots"], item["target"], item["mask"]


def collate_st_graph_windows(batch):
    """
    Custom collator for batches of rolling spatio-temporal graph windows.
    batch: list of tuples (snapshots_list, target_tensor, mask_tensor)
    """
    batch_snapshots = [b[0] for b in batch]  # List of lists of W Data objects
    batch_targets = torch.stack([b[1] for b in batch], dim=0)  # [B, 20]
    batch_masks = torch.stack([b[2] for b in batch], dim=0)    # [B, 20]
    return batch_snapshots, batch_targets, batch_masks


def fit_and_apply_scaler(snapshots: list, train_indices: list):
    """
    Fits StandardScaler on node features strictly using the training set indices.
    Standardizes all snapshots in-place without data leakage.
    """
    scaler = StandardScaler()
    train_features = []

    for idx in train_indices:
        snap = snapshots[idx]
        active_x = snap.x[snap.mask].cpu().numpy()
        if len(active_x) > 0:
            train_features.append(active_x)

    if not train_features:
        logger.warning("No active features found to fit scaler. Skipping standard scaling.")
        return scaler

    all_train_x = np.vstack(train_features)
    scaler.fit(all_train_x)
    logger.info(f"Fitted StandardScaler on {len(all_train_x)} training node instances.")

    # Apply scaler to all snapshots
    for snap in snapshots:
        # Scale all 20 nodes; keep zeros for inactive
        active_mask = snap.mask.cpu().numpy()
        if active_mask.any():
            scaled = snap.x.clone().numpy()
            scaled[active_mask] = scaler.transform(scaled[active_mask])
            snap.x = torch.tensor(scaled, dtype=torch.float)

    return scaler


def build_windowed_samples(snapshots: list, labels_df: pd.DataFrame, driver_to_idx: dict, W: int = 10, K: int = 5):
    """
    Builds (W past snapshots -> target at t+K) dataset.
    Discards windows where target is missing or crossing invalid gap.
    """
    num_nodes = len(driver_to_idx)
    samples = []

    # Map labels by (time_sec, driver)
    # Target delay is mapped by nearest timestamp
    label_times = sorted(labels_df["TimeSec"].unique())

    # Build index of time to snapshot index
    time_to_snap = {snap.time_sec: i for i, snap in enumerate(snapshots)}
    snap_times = sorted(time_to_snap.keys())

    for i in range(len(snap_times) - W - K):
        # Window spans from i to i + W - 1
        window_snaps = snapshots[i : i + W]
        target_snap_idx = i + W - 1 + K
        target_time = snap_times[target_snap_idx]

        # Verify time continuity (no multi-lap or safety car boundary leap)
        dt_span = target_time - snap_times[i]
        # At 4 Hz, (W + K) steps should be approx (15 * 0.25) = 3.75s +/- tolerance
        expected_span = (W + K) * 0.25
        if abs(dt_span - expected_span) > 2.0:
            # Time boundary jump detected, discard window
            continue

        # Look up labels for this target time
        # Find closest timestamp in labels table
        idx_lbl = np.searchsorted(label_times, target_time)
        idx_lbl = np.clip(idx_lbl, 0, len(label_times) - 1)
        closest_lbl_time = label_times[idx_lbl]

        sub_labels = labels_df[labels_df["TimeSec"] == closest_lbl_time]
        y_vec = np.zeros(num_nodes, dtype=np.float32)
        mask_vec = np.zeros(num_nodes, dtype=bool)

        for _, row in sub_labels.iterrows():
            drv = row["Driver"]
            if drv in driver_to_idx:
                d_idx = driver_to_idx[drv]
                y_vec[d_idx] = float(row["DeltaT_Sector"])
                mask_vec[d_idx] = True

        target_snap = snapshots[target_snap_idx]
        combined_mask = mask_vec & target_snap.mask.cpu().numpy()

        if combined_mask.sum() >= 2:  # at least 2 active drivers with targets
            samples.append({
                "snapshots": window_snaps,
                "target": torch.tensor(y_vec, dtype=torch.float),
                "mask": torch.tensor(combined_mask, dtype=torch.bool),
                "time_sec": target_time
            })

    logger.info(f"Built {len(samples)} continuous spatio-temporal samples (W={W}, K={K}).")
    return samples


def create_dataloaders(snapshots: list, labels_df: pd.DataFrame, driver_to_idx: dict, config: dict):
    """
    Creates train, validation, and test DataLoaders strictly respecting non-random temporal/race splits.
    """
    W = config["data"]["window_size"]
    K = config["data"]["horizon"]
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]
    batch_size = config["train"]["batch_size"]
    num_workers = config["train"].get("num_workers", 0)

    # Immutable Law 3: NEVER split snapshots at random!
    # Fit scaler strictly on training snapshots
    total_snaps = len(snapshots)
    train_snap_cutoff = int(total_snaps * train_ratio)
    train_snap_indices = list(range(train_snap_cutoff))

    logger.info("Applying StandardScaler to node features based strictly on training partition...")
    scaler = fit_and_apply_scaler(snapshots, train_snap_indices)

    # Build rolling samples
    all_samples = build_windowed_samples(snapshots, labels_df, driver_to_idx, W=W, K=K)
    N_samples = len(all_samples)

    if N_samples == 0:
        raise ValueError("Zero valid temporal samples could be formed. Check timestamp continuity.")

    n_train = int(N_samples * train_ratio)
    n_val = int(N_samples * val_ratio)

    # Chronological partition
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train : n_train + n_val]
    test_samples = all_samples[n_train + n_val :]

    logger.info(f"Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    train_ds = STGraphDataset(train_samples)
    val_ds = STGraphDataset(val_samples)
    test_ds = STGraphDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_st_graph_windows, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_st_graph_windows, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_st_graph_windows, num_workers=num_workers)

    return train_loader, val_loader, test_loader, scaler
