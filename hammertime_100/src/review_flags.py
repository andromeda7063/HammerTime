"""Review-gated target/coverage changes for GNNv4 (all OFF by default).

AGENTS.md protects target definitions, coverage rules and gates. Each function
here is a *candidate* reviewed change that only runs when its flag in the
config's ``review_flags`` mapping is ``true``. With every flag off,
``apply_review_flags`` returns the cached normalized tables unchanged, so the
target is byte-identical to GNNv2/GNNv3.

Flags:
- ``rescale_lap_distance`` (F6): FastF1 Distance is integrated speed. Each lap
  is rescaled so its integrated length equals ``lap_length_m``; mini-sector
  indices and targets are rebuilt from the corrected distance.
- ``physical_gap_for_targets`` (F5): the target's AvgGap/clean-air rule uses
  the gap to the physically next car around the circuit instead of the next
  car in race order; targets are rebuilt.
- ``detrend_clean_air_reference`` (F2): the driver-level clean-air reference
  follows a per-(driver, compound, sector) linear lap trend (see target.py).
- ``exclude_lap1_and_yellow_targets`` (F3): drop labels on lap 1 and on laps
  whose TrackStatus contains yellow ("2") or VSC-ending ("7"). Inputs keep them.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_component(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "session"


def rescale_lap_distance(aligned: pd.DataFrame, lap_length_m: float, mini_sectors: int) -> tuple[pd.DataFrame, dict]:
    """Scale each complete lap's integrated Distance so it ends at ``lap_length_m``.

    The integrated length of lap n is estimated at the timing line using the
    next lap's first sample: ``last_distance + v_last * dt - next_first_distance``
    (all within the same driver). Laps without a following sample within 0.5 s,
    or with an implausible scale (outside 0.9-1.1), are left unchanged.
    """
    result = aligned.copy()
    result["_row"] = np.arange(len(result))
    ordered = result.sort_values(["Driver", "TimeSec"], kind="stable")
    scales = {}
    for driver, rows in ordered.groupby("Driver", sort=False):
        laps = rows["LapNumber"].to_numpy(dtype=float)
        dist = pd.to_numeric(rows["Distance"], errors="coerce").to_numpy(dtype=float)
        times = rows["TimeSec"].to_numpy(dtype=float)
        speed = pd.to_numeric(rows["Speed"], errors="coerce").to_numpy(dtype=float) / 3.6
        change = np.flatnonzero(np.isfinite(laps[1:]) & np.isfinite(laps[:-1]) & (laps[1:] == laps[:-1] + 1))
        for index in change:
            dt = times[index + 1] - times[index]
            if not (0 < dt <= 0.5) or not np.isfinite(dist[index]) or not np.isfinite(dist[index + 1]):
                continue
            integrated = dist[index] + speed[index] * dt - dist[index + 1]
            if not np.isfinite(integrated) or integrated <= 0:
                continue
            scale = float(lap_length_m) / integrated
            if 0.9 <= scale <= 1.1:
                scales[(driver, laps[index])] = scale
    if scales:
        keys = list(zip(result["Driver"], result["LapNumber"].astype(float)))
        factor = np.array([scales.get(key, 1.0) for key in keys], dtype=float)
        result["Distance"] = pd.to_numeric(result["Distance"], errors="coerce") * factor
        sector_length = float(lap_length_m) / int(mini_sectors)
        within = result["Distance"].fillna(0.0).clip(lower=0.0, upper=float(lap_length_m) - 1e-6)
        result["MiniSector"] = np.clip(np.floor(within / sector_length).astype(int), 0, int(mini_sectors) - 1)
    values = np.array(list(scales.values()), dtype=float)
    report = {"laps_rescaled": int(values.size),
              "median_scale": float(np.median(values)) if values.size else None}
    return result.drop(columns="_row"), report


def compute_physical_time_gaps(aligned: pd.DataFrame, lap_length_m: float) -> pd.DataFrame:
    """Gap (s) to the physically next car around the circuit at each timestamp.

    Mirrors ``preprocess.compute_time_gaps`` but orders cars by
    ``Distance % lap_length_m`` (circular track position) so a lapped car in
    front counts as the car ahead. The race-order gap is kept in
    ``DeltaT_RaceOrder``.
    """
    result = aligned.copy()
    result["DeltaT_RaceOrder"] = result.get("DeltaT", np.nan)
    gaps = np.full(len(result), 99.0, dtype=float)
    valid = result.get("IsValid", pd.Series(True, index=result.index)).fillna(False).astype(bool)
    valid &= ~result["IsPitLap"].fillna(False).astype(bool)
    position = pd.to_numeric(result["Distance"], errors="coerce") % float(lap_length_m)
    speed = pd.to_numeric(result["Speed"], errors="coerce") / 3.6
    frame = pd.DataFrame({"t": result["TimeSec"].to_numpy(), "pos": position.to_numpy(),
                          "v": speed.to_numpy(), "row": np.arange(len(result))})[valid.to_numpy()]
    frame = frame[np.isfinite(frame["pos"]) & np.isfinite(frame["v"])]
    for _, group in frame.groupby("t", sort=False):
        if len(group) < 2:
            continue
        order = group.sort_values("pos")
        pos = order["pos"].to_numpy()
        ahead = np.roll(pos, -1)
        distance = (ahead - pos) % float(lap_length_m)
        follower_speed = np.maximum(order["v"].to_numpy(), 10.0)
        gap = np.where(distance > 0, distance / follower_speed, 99.0)
        gaps[order["row"].to_numpy()] = gap
    result["DeltaT"] = gaps
    return result


def _yellow_laps(raw_sessions_dir: Path, race_id: str) -> pd.DataFrame:
    root = Path(raw_sessions_dir).expanduser()
    path = root / race_id / "raw_laps.parquet"
    if not path.is_file():
        path = root / _safe_component(race_id) / "raw_laps.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"raw_laps.parquet for {race_id} is required for the yellow-flag review flag.")
    laps = pd.read_parquet(path, columns=["DriverNumber", "LapNumber", "TrackStatus"])
    laps["Driver"] = laps["DriverNumber"].astype(str)
    laps["LapNumber"] = pd.to_numeric(laps["LapNumber"], errors="coerce")
    laps["Yellow"] = laps["TrackStatus"].astype(str).str.contains("2|7")
    return laps[["Driver", "LapNumber", "Yellow"]]


def apply_review_flags(aligned: pd.DataFrame, labels: pd.DataFrame, race_config: dict, flags: dict,
                       raw_sessions_dir: str | Path | None = None):
    """Return (aligned, labels, report). With every flag off, inputs are returned unchanged."""
    report = {key: bool(value) for key, value in flags.items()}
    if not any(flags.values()):
        return aligned, labels, report
    lap_length = float(race_config["track"]["lap_length_m"])
    mini_sectors = int(race_config["track"]["mini_sectors"])
    rebuild = False
    if flags.get("rescale_lap_distance"):
        aligned, info = rescale_lap_distance(aligned, lap_length, mini_sectors)
        report["rescale_lap_distance_info"] = info
        rebuild = True
    if flags.get("physical_gap_for_targets"):
        aligned = compute_physical_time_gaps(aligned, lap_length)
        rebuild = True
    if flags.get("detrend_clean_air_reference"):
        rebuild = True
    if rebuild:
        from target import build_targets
        config = copy.deepcopy(race_config)
        config.setdefault("review_flags", {})
        config["review_flags"]["detrend_clean_air_reference"] = bool(flags.get("detrend_clean_air_reference"))
        before = len(labels)
        labels = build_targets(aligned, config)
        report["labels_before_rebuild"] = int(before)
        report["labels_after_rebuild"] = int(len(labels))
    if flags.get("exclude_lap1_and_yellow_targets"):
        if raw_sessions_dir is None:
            raise ValueError("exclude_lap1_and_yellow_targets requires raw_sessions_dir.")
        race_id = str(race_config["session"]["race_id"])
        yellow = _yellow_laps(Path(raw_sessions_dir), race_id)
        merged = labels.assign(Driver=labels["Driver"].astype(str)).merge(
            yellow, on=["Driver", "LapNumber"], how="left")
        drop = (merged["LapNumber"] == 1) | merged["Yellow"].astype("boolean").fillna(False).astype(bool)
        report["labels_dropped_lap1_or_yellow"] = int(drop.sum())
        labels = labels.loc[~drop.to_numpy()].reset_index(drop=True)
    return aligned, labels, report
