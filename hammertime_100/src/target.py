"""
Module M3: Target Builder
Responsibility: Calculate the traffic delay target (Δt_sector = t_actual - t_clean_air_baseline)
for each car, mini-sector, and tyre compound.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.target")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_mini_sector_durations(
    df: pd.DataFrame,
    max_gap_sec: float | None = None,
    lap_length_m: float = 5793.0,
    mini_sectors: int = 20,
) -> pd.DataFrame:
    """Interpolate observed distance crossings; omit unsupported or partial sectors."""
    required = {"Driver", "LapNumber", "Compound", "TimeSec", "Distance", "IsPitLap", "IsSafetyCar"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Telemetry is missing required duration columns: {sorted(missing)}")
    if lap_length_m <= 0 or mini_sectors <= 0:
        raise ValueError("lap_length_m and mini_sectors must be positive.")
    sector_length = lap_length_m / mini_sectors
    group_keys = ["RaceId", "Driver"] if "RaceId" in df.columns else ["Driver"]
    records = []
    for _, driver_rows in df.groupby(group_keys, sort=False):
        rows = driver_rows.sort_values("TimeSec").copy()
        times = pd.to_numeric(rows["TimeSec"], errors="coerce").to_numpy(dtype=float)
        distances = pd.to_numeric(rows["Distance"], errors="coerce").to_numpy(dtype=float)
        laps = pd.to_numeric(rows["LapNumber"], errors="coerce").to_numpy(dtype=float)
        # M2 supplies distance relative to each observed lap start. Keep an
        # observed endpoint at lap_length_m instead of wrapping it to zero.
        progress = (laps - 1) * lap_length_m + distances
        valid = rows.get("IsValid", pd.Series(True, index=rows.index)).fillna(False).to_numpy(dtype=bool)
        valid &= ~rows["IsPitLap"].fillna(False).to_numpy(dtype=bool)
        valid &= ~rows["IsSafetyCar"].fillna(False).to_numpy(dtype=bool)
        valid &= np.isfinite(times) & np.isfinite(progress) & (distances >= 0)
        breaks = np.ones(len(rows), dtype=bool)
        if len(rows) > 1:
            time_step, distance_step = np.diff(times), np.diff(progress)
            connected = valid[:-1] & valid[1:] & (time_step > 0) & (distance_step > 0)
            if max_gap_sec is not None:
                connected &= time_step <= max_gap_sec + 1e-9
            connected &= np.diff(laps) <= 1
            breaks[1:] = ~connected
        segments = np.cumsum(breaks)
        for segment_id in np.unique(segments[valid]):
            indices = np.flatnonzero((segments == segment_id) & valid)
            if len(indices) < 2:
                continue
            segment_distance, segment_time = progress[indices], times[indices]
            first = int(np.ceil(segment_distance[0] / sector_length - 1e-9))
            last = int(np.floor(segment_distance[-1] / sector_length + 1e-9))
            if last <= first:
                continue
            boundaries = np.arange(first, last + 1) * sector_length
            crossing_times = np.interp(boundaries, segment_distance, segment_time)
            for index in range(len(boundaries) - 1):
                lap = (first + index) // mini_sectors + 1
                sector = (first + index) % mini_sectors
                entry, exit_time = crossing_times[index:index + 2]
                observed = indices[(segment_time >= entry) & (segment_time < exit_time)]
                if not len(observed):
                    continue
                sector_rows = rows.iloc[observed]
                compounds = sector_rows["Compound"].dropna().unique()
                if len(compounds) != 1 or not (sector_rows["LapNumber"] == lap).all():
                    continue
                duration = float(exit_time - entry)
                if not 0.5 < duration < 20.0:
                    continue
                record = {
                    "Driver": sector_rows["Driver"].iloc[0], "LapNumber": lap,
                    "MiniSector": sector, "Compound": compounds[0],
                    "Duration": duration, "AvgGap": float(sector_rows["DeltaT"].mean()) if "DeltaT" in rows else np.nan,
                    "TimeSec": float(entry),
                }
                if "Team" in rows:
                    teams = sector_rows["Team"].dropna().astype(str)
                    record["Team"] = teams.iloc[0] if not teams.empty else None
                if "RaceId" in rows:
                    record["RaceId"] = sector_rows["RaceId"].iloc[0]
                records.append(record)
    return pd.DataFrame(records)


def compute_clean_air_baselines(sector_durations: pd.DataFrame, clean_air_gap: float = 3.0, min_samples: int = 3) -> dict:
    """
    Computes median clean-air sector times per driver, mini-sector, and tyre compound.
    Clean air condition: AvgGap > clean_air_gap (3.0 s).
    Provides hierarchical fallback: (Driver, Compound, Sector) -> (Compound, Sector).
    """
    clean_laps = sector_durations[sector_durations["AvgGap"] > clean_air_gap]
    logger.info(f"Found {len(clean_laps)} clean-air sector segments across grid.")

    if min_samples < 1:
        raise ValueError("target.min_clean_samples must be at least 1.")

    # Use a baseline only when the configured number of clean laps supports it.
    # Keep references within each session/circuit; sector times do not transfer
    # between tracks even for the same driver and tyre compound.
    primary_keys = ["Driver", "Compound", "MiniSector"]
    team_keys = ["Team", "Compound", "MiniSector"]
    grid_keys = ["Compound", "MiniSector"]
    if "RaceId" in clean_laps.columns:
        primary_keys.insert(0, "RaceId")
        team_keys.insert(0, "RaceId")
        grid_keys.insert(0, "RaceId")
    primary_groups = clean_laps.groupby(primary_keys)["Duration"]
    primary_baseline = primary_groups.median()[primary_groups.size() >= min_samples].to_dict()
    team_baseline = {}
    if "Team" in clean_laps.columns:
        team_laps = clean_laps.dropna(subset=["Team"])
        team_groups = team_laps.groupby(team_keys)["Duration"]
        team_baseline = team_groups.median()[team_groups.size() >= min_samples].to_dict()
    grid_groups = clean_laps.groupby(grid_keys)["Duration"]
    grid_baseline = grid_groups.median()[grid_groups.size() >= min_samples].to_dict()

    return {
        "primary": primary_baseline,
        "team": team_baseline,
        "grid": grid_baseline,
    }


def build_targets(aligned_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Builds the target table containing: Driver, TimeSec, LapNumber, MiniSector, Delta_t_sector.
    Formula: Delta_t_sector = t_actual - t_clean_air_baseline
    """
    clean_gap = config["target"]["clean_air_gap_sec"]
    min_samples = config["target"]["min_clean_samples"]
    clip_range = config["target"]["clip_delta_sec"]

    aligned_df = aligned_df.copy()
    # Provenance and coverage are descriptive metadata, not target inputs.
    # Pandas propagates and deep-copies attrs for every slice; sector extraction
    # creates many slices, so keep that metadata on the caller's original frame
    # and strip it from this working copy.
    aligned_df.attrs = {}
    if "RaceId" not in aligned_df.columns:
        session_cfg = config.get("session", {})
        aligned_df["RaceId"] = str(session_cfg.get("race_id") or (
            f"{session_cfg.get('year', 'unknown')}-{session_cfg.get('gp', 'unknown')}-"
            f"{session_cfg.get('session_type', 'R')}"
        ))

    logger.info("Computing sector durations from aligned telemetry...")
    sector_df = calculate_mini_sector_durations(
        aligned_df,
        max_gap_sec=config.get("preprocess", {}).get("max_interpolate_sec"),
        lap_length_m=float(config["track"]["lap_length_m"]),
        mini_sectors=int(config["track"]["mini_sectors"]),
    )

    if sector_df.empty:
        raise ValueError("Could not extract mini-sector durations from telemetry.")

    baselines = compute_clean_air_baselines(sector_df, clean_air_gap=clean_gap, min_samples=min_samples)

    labels = []
    for _, row in sector_df.iterrows():
        race_id = row["RaceId"]
        drv = row["Driver"]
        team = row.get("Team")
        cmp = row["Compound"]
        sec = row["MiniSector"]
        actual_dur = row["Duration"]

        # Resolve driver, teammate, then grid baselines. Do not invent a nominal
        # value when clean samples are absent.
        base_dur = baselines["primary"].get((race_id, drv, cmp, sec))
        if base_dur is None and team is not None and pd.notna(team):
            base_dur = baselines["team"].get((race_id, str(team), cmp, sec))
        if base_dur is None:
            base_dur = baselines["grid"].get((race_id, cmp, sec))
        if base_dur is None:
            continue

        # Delta t_sector = t_actual - t_baseline
        delta_t_sector = actual_dur - base_dur

        # Clip extreme noise (e.g. spins or crashes)
        delta_t_sector = float(np.clip(delta_t_sector, clip_range[0], clip_range[1]))

        labels.append({
            "Driver": drv,
            "TimeSec": row["TimeSec"],
            "LapNumber": row["LapNumber"],
            "MiniSector": sec,
            "Duration": actual_dur,
            "Baseline": base_dur,
            "DeltaT_Sector": delta_t_sector,
            "AvgGap": row["AvgGap"],
            "Team": team,
            "RaceId": race_id,
        })

    label_table = pd.DataFrame(labels)
    if label_table.empty:
        raise ValueError("No target labels have a valid clean-air baseline. Need more clean laps per compound and mini-sector.")
    logger.info(f"Generated {len(label_table)} target labels. Mean DeltaT_Sector = {label_table['DeltaT_Sector'].mean():.4f} s.")

    # Save to processed directory
    out_dir = Path(config["session"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "target_labels.parquet"
    label_table.to_parquet(out_file)
    logger.info(f"Saved target labels to: {out_file}")

    return label_table


if __name__ == "__main__":
    cfg = load_config()
    proc_file = Path(cfg["session"]["processed_dir"]) / "aligned_telemetry.parquet"
    if not proc_file.exists():
        logger.error(f"Missing {proc_file}. Run preprocess.py first.")
        sys.exit(1)
    aligned_data = pd.read_parquet(proc_file)
    build_targets(aligned_data, cfg)
