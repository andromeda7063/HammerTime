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
) -> pd.DataFrame:
    """
    Computes the actual time elapsed in each mini-sector per lap and driver.
    """
    # Group by Driver, LapNumber, MiniSector
    # Filter out safety car and pit laps
    valid = df.get("IsValid", pd.Series(True, index=df.index)).fillna(False).astype(bool)
    clean_candidates = df[
        valid
        & ~df["IsPitLap"].fillna(False).astype(bool)
        & ~df["IsSafetyCar"].fillna(False).astype(bool)
    ].copy()
    if clean_candidates.empty:
        return pd.DataFrame()

    sample_group = ["Driver"]
    if "RaceId" in clean_candidates.columns:
        sample_group.insert(0, "RaceId")
    clean_candidates = clean_candidates.sort_values(sample_group + ["TimeSec"])
    sample_steps = clean_candidates.groupby(sample_group)["TimeSec"].diff()
    sample_steps = sample_steps[sample_steps > 0]
    sample_period = float(sample_steps.median()) if not sample_steps.empty else 0.0

    records = []
    group_keys = ["Driver", "LapNumber", "MiniSector", "Compound"]
    if "RaceId" in clean_candidates.columns:
        group_keys.insert(0, "RaceId")
    grouped = clean_candidates.groupby(group_keys)

    for key, grp in grouped:
        if "RaceId" in clean_candidates.columns:
            race_id, driver, lap, sector, compound = key
        else:
            driver, lap, sector, compound = key
        grp = grp.sort_values("TimeSec")
        if max_gap_sec is not None and len(grp) > 1:
            if grp["TimeSec"].diff().dropna().max() > max_gap_sec:
                continue
        t_min = grp["TimeSec"].min()
        t_max = grp["TimeSec"].max()
        # Samples represent intervals. Include one grid interval so duration does
        # not systematically lose one 4 Hz step at each sector boundary.
        duration = t_max - t_min + sample_period

        # Realistic mini-sector duration check (e.g. 1.5s to 12s for a 20-sector lap)
        if 0.5 < duration < 20.0:
            avg_gap = grp["DeltaT"].mean()
            record = {
                "Driver": driver,
                "LapNumber": lap,
                "MiniSector": sector,
                "Compound": compound,
                "Duration": duration,
                "AvgGap": avg_gap,
                "TimeSec": t_min  # reference timestamp
            }
            if "RaceId" in clean_candidates.columns:
                record["RaceId"] = race_id
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
    primary_groups = clean_laps.groupby(["Driver", "Compound", "MiniSector"])["Duration"]
    primary_baseline = primary_groups.median()[primary_groups.size() >= min_samples].to_dict()
    grid_groups = clean_laps.groupby(["Compound", "MiniSector"])["Duration"]
    grid_baseline = grid_groups.median()[grid_groups.size() >= min_samples].to_dict()

    return {
        "primary": primary_baseline,
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

    logger.info("Computing sector durations from aligned telemetry...")
    sector_df = calculate_mini_sector_durations(
        aligned_df,
        max_gap_sec=config.get("preprocess", {}).get("max_interpolate_sec"),
    )

    if sector_df.empty:
        raise ValueError("Could not extract mini-sector durations from telemetry.")

    baselines = compute_clean_air_baselines(sector_df, clean_air_gap=clean_gap, min_samples=min_samples)

    labels = []
    for _, row in sector_df.iterrows():
        drv = row["Driver"]
        cmp = row["Compound"]
        sec = row["MiniSector"]
        actual_dur = row["Duration"]

        # Resolve the driver/compound baseline, then use the compound-grid
        # baseline. Do not invent a nominal baseline when clean samples are absent.
        base_dur = baselines["primary"].get((drv, cmp, sec))
        if base_dur is None:
            base_dur = baselines["grid"].get((cmp, sec))
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
            "AvgGap": row["AvgGap"]
        })

    label_table = pd.DataFrame(labels)
    if "RaceId" not in label_table.columns and config.get("session"):
        session_cfg = config["session"]
        label_table["RaceId"] = str(session_cfg.get("race_id") or (
            f"{session_cfg.get('year', 'unknown')}-{session_cfg.get('gp', 'unknown')}-"
            f"{session_cfg.get('session_type', 'R')}"
        ))
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
