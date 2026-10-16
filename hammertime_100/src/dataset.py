"""M5: build chronological graph windows and fit preprocessing on train data only."""

import logging

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("hammertime.dataset")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class STGraphDataset(Dataset):
    """A list of input windows, target vectors, and target validity masks."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["snapshots"], item["target"], item["mask"]


def collate_st_graph_windows(batch):
    windows, targets, masks = zip(*batch)
    return list(windows), torch.stack(targets), torch.stack(masks)


def _as_bool_mask(snapshot):
    mask = getattr(snapshot, "mask", None)
    if mask is None:
        return np.ones(snapshot.x.size(0), dtype=bool)
    return mask.detach().cpu().numpy().astype(bool)


def fit_and_apply_scaler(snapshots: list, train_indices: list):
    """Fit on active training nodes; transform features while preserving inactive zeros."""
    scaler = StandardScaler()
    blocks = []
    for idx in train_indices:
        snap = snapshots[idx]
        mask = _as_bool_mask(snap)
        if mask.any():
            blocks.append(snap.x.detach().cpu().numpy()[mask])
    if not blocks:
        raise ValueError("No active training nodes are available to fit the feature scaler.")
    scaler.fit(np.concatenate(blocks, axis=0))
    for snap in snapshots:
        mask = _as_bool_mask(snap)
        if mask.any():
            raw = snap.x.detach().cpu().numpy().copy()
            raw[mask] = scaler.transform(raw[mask])
            snap.x = torch.as_tensor(raw, dtype=torch.float32)
    return scaler


def _build_windowed_samples_one_race(snapshots: list, labels_df: pd.DataFrame, driver_to_idx: dict,
                                     W: int = 10, K: int = 5, sample_period_sec: float = 0.25,
                                     timestamp_tolerance_sec: float | None = None,
                                     continuity_tolerance_sec: float | None = None,
                                     min_valid_targets: int = 1):
    """Build chronological W-frame inputs with labels exactly K frames after the input."""
    required = {"Driver", "TimeSec", "DeltaT_Sector"}
    missing = required.difference(labels_df.columns)
    if missing:
        raise ValueError(f"Target table is missing required columns: {sorted(missing)}")
    if W < 1 or K < 0:
        raise ValueError("window_size must be positive and horizon cannot be negative.")
    ordered = sorted(snapshots, key=lambda snap: float(snap.time_sec))
    times = np.asarray([float(snap.time_sec) for snap in ordered], dtype=np.float64)
    if len(times) != len(np.unique(times)):
        raise ValueError("Graph snapshots must have unique timestamps.")
    if timestamp_tolerance_sec is None:
        timestamp_tolerance_sec = sample_period_sec * 0.51
    if continuity_tolerance_sec is None:
        continuity_tolerance_sec = sample_period_sec * 0.2

    # Match each label to its nearest grid point, but never silently attach a label
    # to a distant timestamp (which can leak a later sector's target backwards).
    labels = labels_df.copy()
    labels["TimeSec"] = pd.to_numeric(labels["TimeSec"], errors="coerce")
    labels = labels.dropna(subset=["TimeSec", "DeltaT_Sector"])
    labels["_snap_idx"] = np.searchsorted(times, labels["TimeSec"].to_numpy())
    right = labels["_snap_idx"].to_numpy().clip(0, max(len(times) - 1, 0))
    left = (labels["_snap_idx"].to_numpy() - 1).clip(0, max(len(times) - 1, 0))
    if len(times) == 0:
        return []
    choose_left = np.abs(times[left] - labels["TimeSec"].to_numpy()) < np.abs(times[right] - labels["TimeSec"].to_numpy())
    labels["_snap_idx"] = np.where(choose_left, left, right)
    labels["_snap_time"] = times[labels["_snap_idx"].to_numpy()]
    labels = labels[np.abs(labels["TimeSec"] - labels["_snap_time"]) <= timestamp_tolerance_sec]
    # A collision for the same driver/grid point is resolved deterministically by
    # retaining the target whose source time is closest to the graph timestamp.
    labels["_distance"] = np.abs(labels["TimeSec"] - labels["_snap_time"])
    labels = labels.sort_values(["_distance", "TimeSec"]).drop_duplicates(["_snap_idx", "Driver"])
    labels_by_idx = {int(i): grp for i, grp in labels.groupby("_snap_idx", sort=False)}

    samples = []
    span = W + K
    for start in range(0, len(ordered) - span + 1):
        stop = start + span
        local_times = times[start:stop]
        race_ids = [getattr(snap, "race_id", None) for snap in ordered[start:stop]]
        if any(race_id != race_ids[0] for race_id in race_ids[1:]):
            continue
        if len(local_times) > 1 and not np.all(np.diff(local_times) > 0):
            continue
        # Every adjacent frame must be on the configured grid. This rejects gaps,
        # session boundaries, and pauses rather than making a window across them.
        if len(local_times) > 1 and np.any(np.abs(np.diff(local_times) - sample_period_sec) > continuity_tolerance_sec):
            continue
        target_idx = stop - 1
        target_rows = labels_by_idx.get(target_idx)
        if target_rows is None:
            continue
        n_nodes = ordered[target_idx].x.size(0)
        y = np.zeros(n_nodes, dtype=np.float32)
        valid = np.zeros(n_nodes, dtype=bool)
        gaps = np.full(n_nodes, np.nan, dtype=np.float32)
        laps = np.full(n_nodes, np.nan, dtype=np.float32)
        sectors = np.full(n_nodes, np.nan, dtype=np.float32)
        driver_names = getattr(ordered[target_idx], "driver_names", None)
        for row in target_rows.itertuples(index=False):
            driver = getattr(row, "Driver")
            if driver_names is not None:
                try:
                    node = list(driver_names).index(driver)
                except ValueError:
                    continue
            else:
                if driver not in driver_to_idx:
                    continue
                node = driver_to_idx[driver]
            value = float(getattr(row, "DeltaT_Sector"))
            if np.isfinite(value):
                y[node] = value
                valid[node] = True
                if hasattr(row, "AvgGap") and pd.notna(getattr(row, "AvgGap")):
                    gaps[node] = float(getattr(row, "AvgGap"))
                if hasattr(row, "LapNumber") and pd.notna(getattr(row, "LapNumber")):
                    laps[node] = float(getattr(row, "LapNumber"))
                if hasattr(row, "MiniSector") and pd.notna(getattr(row, "MiniSector")):
                    sectors[node] = float(getattr(row, "MiniSector"))
        valid &= _as_bool_mask(ordered[target_idx])
        if valid.sum() < min_valid_targets:
            continue
        samples.append({"snapshots": ordered[start:start + W],
                        "target": torch.tensor(y, dtype=torch.float32),
                        "mask": torch.tensor(valid, dtype=torch.bool),
                        "gap_sec": gaps,
                        "lap_number": laps,
                        "mini_sector": sectors,
                        "time_sec": float(times[target_idx]),
                        "input_start_time": float(times[start]),
                        "target_time": float(times[target_idx]),
                        "race_id": getattr(ordered[target_idx], "race_id", None),
                        "driver_names": list(driver_names) if driver_names is not None else None})
    return samples


def build_windowed_samples(snapshots: list, labels_df: pd.DataFrame, driver_to_idx: dict,
                           W: int = 10, K: int = 5, sample_period_sec: float = 0.25,
                           timestamp_tolerance_sec: float | None = None,
                           continuity_tolerance_sec: float | None = None,
                           min_valid_targets: int = 1):
    """Build windows independently per race when race IDs are available."""
    race_ids = [getattr(snap, "race_id", None) for snap in snapshots]
    if race_ids and all(race_id is not None for race_id in race_ids) and "RaceId" in labels_df.columns:
        samples = []
        for race_id in dict.fromkeys(race_ids):
            race_snaps = [snap for snap in snapshots if getattr(snap, "race_id", None) == race_id]
            race_labels = labels_df[labels_df["RaceId"] == race_id]
            samples.extend(_build_windowed_samples_one_race(
                race_snaps, race_labels, driver_to_idx, W, K, sample_period_sec,
                timestamp_tolerance_sec, continuity_tolerance_sec, min_valid_targets
            ))
        return samples
    return _build_windowed_samples_one_race(snapshots, labels_df, driver_to_idx, W, K,
                                            sample_period_sec, timestamp_tolerance_sec,
                                            continuity_tolerance_sec, min_valid_targets)


def _partition_samples(samples, train_ratio, val_ratio, purge_steps):
    """Chronological partitions with a purge gap so windows cannot share frames."""
    n = len(samples)
    if n < 3:
        raise ValueError(f"Need at least three valid windows for train/validation/test; found {n}.")
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    # Drop windows around each boundary. Each sample spans W+K graph frames.
    train = samples[:max(0, train_end - purge_steps)]
    val = samples[min(n, train_end + purge_steps):max(train_end + purge_steps, val_end - purge_steps)]
    test = samples[min(n, val_end + purge_steps):]
    if not train or not val or not test:
        raise ValueError("Chronological split is too small after purging overlapping windows; add more race data or reduce W/K.")
    return train, val, test


def _partition_races(samples, train_ratio, val_ratio):
    race_ids = []
    for sample in samples:
        race_id = sample.get("race_id")
        if race_id is not None and race_id not in race_ids:
            race_ids.append(race_id)
    if len(race_ids) < 3:
        return None
    n = len(race_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    train_ids = set(race_ids[:n_train])
    val_ids = set(race_ids[n_train:n_train + n_val])
    test_ids = set(race_ids[n_train + n_val:])
    groups = ([s for s in samples if s["race_id"] in train_ids],
              [s for s in samples if s["race_id"] in val_ids],
              [s for s in samples if s["race_id"] in test_ids])
    return groups if all(groups) else None


def create_dataloaders(snapshots: list, labels_df: pd.DataFrame, driver_to_idx: dict, config: dict):
    """Create chronological loaders and fit the scaler on training input frames only."""
    data_cfg = config["data"]
    W, K = int(data_cfg["window_size"]), int(data_cfg["horizon"])
    method = data_cfg.get("split_method", "temporal")
    if method not in ("temporal", "race"):
        raise ValueError(f"Unsupported split_method {method!r}; use 'temporal' or 'race'.")
    if method == "race" and ("RaceId" not in labels_df.columns or
                              not snapshots or any(getattr(s, "race_id", None) is None for s in snapshots)):
        raise ValueError("split_method='race' requires RaceId on labels and race_id on every graph snapshot.")

    prep = config.get("preprocess", {})
    hz = float(prep.get("sample_hz", 4.0))
    if hz <= 0:
        raise ValueError("preprocess.sample_hz must be positive.")
    period = 1.0 / hz
    samples = build_windowed_samples(
        snapshots, labels_df, driver_to_idx, W, K, period,
        float(data_cfg.get("timestamp_tolerance_sec", period * 0.51)),
        float(data_cfg.get("continuity_tolerance_sec", period * 0.2)),
        int(data_cfg.get("min_valid_targets_per_sample", 1)),
    )
    if not samples:
        raise ValueError("No valid graph windows matched target labels on the configured time grid.")
    train_ratio = float(data_cfg["train_ratio"])
    val_ratio = float(data_cfg["val_ratio"])
    test_ratio = float(data_cfg.get("test_ratio", 1 - train_ratio - val_ratio))
    if min(train_ratio, val_ratio, test_ratio) <= 0 or not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must be positive and sum to 1.")
    if method == "race":
        split = _partition_races(samples, train_ratio, val_ratio)
        if split is None:
            raise ValueError("Race split requires at least three races with valid windows and non-empty train/validation/test partitions. Use temporal split for one or two races.")
    else:
        split = _partition_samples(samples, train_ratio, val_ratio, W + K)
    train_samples, val_samples, test_samples = split

    train_keys = {(sample.get("race_id"), float(s.time_sec))
                  for sample in train_samples for s in sample["snapshots"]}
    train_indices = [i for i, snap in enumerate(snapshots)
                     if (getattr(snap, "race_id", None), float(snap.time_sec)) in train_keys]
    scaler = fit_and_apply_scaler(snapshots, train_indices)
    # Windows hold references to snapshots, so the transformation is reflected in all splits.
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 0))
    seed = int(data_cfg.get("random_seed", 42))
    split_metadata = {
        "method": method,
        "purge_windows_per_boundary": 0 if method == "race" else W + K,
        "train": {"sample_count": len(train_samples), "race_ids": sorted({str(s["race_id"]) for s in train_samples if s.get("race_id") is not None}),
                  "first_target_time": train_samples[0].get("target_time"), "last_target_time": train_samples[-1].get("target_time")},
        "validation": {"sample_count": len(val_samples), "race_ids": sorted({str(s["race_id"]) for s in val_samples if s.get("race_id") is not None}),
                       "first_target_time": val_samples[0].get("target_time"), "last_target_time": val_samples[-1].get("target_time")},
        "test": {"sample_count": len(test_samples), "race_ids": sorted({str(s["race_id"]) for s in test_samples if s.get("race_id") is not None}),
                 "first_target_time": test_samples[0].get("target_time"), "last_target_time": test_samples[-1].get("target_time")},
    }
    def loader(rows, shuffle):
        dataset = STGraphDataset(rows)
        dataset.driver_to_idx = dict(driver_to_idx)
        dataset.feature_scaler = scaler
        dataset.split_metadata = split_metadata
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_st_graph_windows, num_workers=num_workers,
                          generator=generator)
    logger.info("Chronological split: train=%d, validation=%d, test=%d (purged %d windows per boundary)",
                len(train_samples), len(val_samples), len(test_samples), W + K)
    return loader(train_samples, True), loader(val_samples, False), loader(test_samples, False), scaler
