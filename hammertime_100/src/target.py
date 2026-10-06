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


def calculate_mini_sector_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the actual time elapsed in each mini-sector per lap and driver.
    """
    # Group by Driver, LapNumber, MiniSector
    # Filter out safety car and pit laps
    clean_candidates = df[~df["IsPitLap"] & ~df["IsSafetyCar"]].copy()

    records = []
    grouped = clean_candidates.groupby(["Driver", "LapNumber", "MiniSector", "Compound"])

    for (driver, lap, sector, compound), grp in grouped:
        t_min = grp["TimeSec"].min()
        t_max = grp["TimeSec"].max()
        duration = t_max - t_min

        # Realistic mini-sector duration check (e.g. 1.5s to 12s for a 20-sector lap)
        if 0.5 < duration < 20.0:
            avg_gap = grp["DeltaT"].mean()
            records.append({
                "Driver": driver,
                "LapNumber": lap,
                "MiniSector": sector,
                "Compound": compound,
                "Duration": duration,
                "AvgGap": avg_gap,
                "TimeSec": t_min  # reference timestamp
            })

    return pd.DataFrame(records)


def compute_clean_air_baselines(sector_durations: pd.DataFrame, clean_air_gap: float = 3.0, min_samples: int = 3) -> dict:
    """
    Computes median clean-air sector times per driver, mini-sector, and tyre compound.
    Clean air condition: AvgGap > clean_air_gap (3.0 s).
    Provides hierarchical fallback: (Driver, Compound, Sector) -> (Compound, Sector).
    """
    clean_laps = sector_durations[sector_durations["AvgGap"] > clean_air_gap]
    logger.info(f"Found {len(clean_laps)} clean-air sector segments across grid.")

    # Primary baseline: (Driver, Compound, MiniSector) -> median duration
    primary_baseline = clean_laps.groupby(["Driver", "Compound", "MiniSector"])["Duration"].median().to_dict()

    # Fallback baseline: (Compound, MiniSector) -> grid median duration
    grid_baseline = clean_laps.groupby(["Compound", "MiniSector"])["Duration"].median().to_dict()

    # Global sector baseline fallback
    global_baseline = clean_laps.groupby("MiniSector")["Duration"].median().to_dict()

    return {
        "primary": primary_baseline,
        "grid": grid_baseline,
        "global": global_baseline
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
    sector_df = calculate_mini_sector_durations(aligned_df)

    if sector_df.empty:
        raise ValueError("Could not extract mini-sector durations from telemetry.")

    baselines = compute_clean_air_baselines(sector_df, clean_air_gap=clean_gap, min_samples=min_samples)

    labels = []
    for _, row in sector_df.iterrows():
        drv = row["Driver"]
        cmp = row["Compound"]
        sec = row["MiniSector"]
        actual_dur = row["Duration"]

        # Hierarchical baseline resolution
        base_dur = baselines["primary"].get((drv, cmp, sec))
        if base_dur is None:
            base_dur = baselines["grid"].get((cmp, sec))
        if base_dur is None:
            base_dur = baselines["global"].get(sec, 4.5)  # nominal fallback

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
