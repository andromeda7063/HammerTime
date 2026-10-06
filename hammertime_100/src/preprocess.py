"""
Module M2: Preprocessing
Responsibility: Align all car telemetry on a uniform 4 Hz time grid,
interpolate short gaps, flag safety cars/pit laps, and compute the gap Δt to the car ahead.
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
logger = logging.getLogger("hammertime.preprocess")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def align_driver_telemetry(
    driver: str,
    tel_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    sample_hz: float = 4.0,
    max_interpolate_sec: float = 1.0,
    mini_sectors: int = 20,
    lap_length_m: float = 5793.0
) -> pd.DataFrame:
    """
    Resamples a single driver's telemetry to a uniform time grid.
    Computes mini-sector index and merges tyre compound and tyre age.
    """
    if tel_df.empty:
        logger.warning(f"Driver {driver} has no telemetry data.")
        return pd.DataFrame()

    # Ensure Time column is in seconds
    if pd.api.types.is_timedelta64_dtype(tel_df["Time"]):
        tel_df["TimeSec"] = tel_df["Time"].dt.total_seconds()
    elif "TimeSec" not in tel_df.columns:
        tel_df["TimeSec"] = tel_df["Time"].astype(float)

    tel_sorted = tel_df.sort_values("TimeSec").drop_duplicates(subset=["TimeSec"])
    t_start = tel_sorted["TimeSec"].min()
    t_end = tel_sorted["TimeSec"].max()

    dt = 1.0 / sample_hz
    uniform_times = np.arange(t_start, t_end, dt)

    # Base grid
    resampled = pd.DataFrame({"TimeSec": uniform_times})
    resampled["Driver"] = driver

    # Interpolate numerical columns
    interp_cols = ["Speed", "Throttle", "Brake", "Distance", "X", "Y", "Z"]
    for col in interp_cols:
        if col in tel_sorted.columns:
            valid = tel_sorted[["TimeSec", col]].dropna()
            if len(valid) > 1:
                resampled[col] = np.interp(
                    uniform_times, valid["TimeSec"].values, valid[col].values
                )
            else:
                resampled[col] = 0.0
        else:
            resampled[col] = 0.0

    # Attach Lap information
    if not laps_df.empty:
        # Match each timestamp to its corresponding lap
        laps_sorted = laps_df.sort_values("LapNumber").copy()
        if "LapStartTime" in laps_sorted.columns and pd.api.types.is_timedelta64_dtype(laps_sorted["LapStartTime"]):
            laps_sorted["LapStartSec"] = laps_sorted["LapStartTime"].dt.total_seconds()
        else:
            laps_sorted["LapStartSec"] = 0.0

        # Assign lap number, compound, tyre age, pit status, track status
        resampled["LapNumber"] = 1
        resampled["Compound"] = "MEDIUM"
        resampled["TyreLife"] = 1
        resampled["IsPitLap"] = False
        resampled["IsSafetyCar"] = False

        for _, lap in laps_sorted.iterrows():
            lap_num = int(lap.get("LapNumber", 1))
            lap_start = lap.get("LapStartSec", 0.0)
            lap_time = lap.get("LapTime")
            lap_dur = lap_time.total_seconds() if pd.notnull(lap_time) and hasattr(lap_time, "total_seconds") else 90.0
            lap_end = lap_start + lap_dur

            mask = (resampled["TimeSec"] >= lap_start) & (resampled["TimeSec"] < lap_end)
            if mask.any():
                resampled.loc[mask, "LapNumber"] = lap_num
                compound = str(lap.get("Compound", "MEDIUM")).upper()
                resampled.loc[mask, "Compound"] = compound if compound in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"] else "MEDIUM"
                resampled.loc[mask, "TyreLife"] = float(lap.get("TyreLife", 1.0))
                
                # Check pit
                if pd.notnull(lap.get("PitInTime")) or pd.notnull(lap.get("PitOutTime")):
                    resampled.loc[mask, "IsPitLap"] = True
                
                # Check track status for safety car (status 4 = SC, 6 = VSC)
                status = str(lap.get("TrackStatus", "1"))
                if "4" in status or "5" in status or "6" in status:
                    resampled.loc[mask, "IsSafetyCar"] = True

    # Calculate mini-sector index [0, M-1] based on distance within lap
    dist_in_lap = resampled["Distance"] % lap_length_m
    sector_len = lap_length_m / mini_sectors
    resampled["MiniSector"] = np.clip(np.floor(dist_in_lap / sector_len).astype(int), 0, mini_sectors - 1)

    return resampled


def compute_time_gaps(aligned_grid: pd.DataFrame, sample_hz: float = 4.0) -> pd.DataFrame:
    """
    Computes the time gap Delta t_ij to the immediate car ahead at every timestamp.
    Delta t_ij = Distance_ahead / Speed_follower
    """
    results = []
    # Group by uniform timestamp
    for t, group in aligned_grid.groupby("TimeSec"):
        active = group[~group["IsPitLap"]].copy()
        if len(active) <= 1:
            group["DriverAhead"] = None
            group["DeltaT"] = np.nan
            results.append(group)
            continue

        # Sort cars by track progression (LapNumber then Distance)
        active = active.sort_values(by=["LapNumber", "Distance"], ascending=False)
        active_drivers = active["Driver"].tolist()
        
        # Leader of group has no car ahead
        driver_ahead_map = {active_drivers[0]: None}
        gap_map = {active_drivers[0]: 99.0}

        for idx in range(1, len(active_drivers)):
            follower = active_drivers[idx]
            leader = active_drivers[idx - 1]
            
            fol_row = active[active["Driver"] == follower].iloc[0]
            lead_row = active[active["Driver"] == leader].iloc[0]
            
            dist_diff = lead_row["Distance"] - fol_row["Distance"]
            fol_speed_ms = max(fol_row["Speed"] / 3.6, 10.0)  # avoid div by zero; km/h to m/s
            
            # If distance difference is positive and plausible
            if 0 < dist_diff < 1000.0:
                dt_gap = dist_diff / fol_speed_ms
            else:
                dt_gap = 99.0
                
            driver_ahead_map[follower] = leader
            gap_map[follower] = dt_gap

        group["DriverAhead"] = group["Driver"].map(driver_ahead_map)
        group["DeltaT"] = group["Driver"].map(gap_map)
        results.append(group)

    return pd.concat(results, ignore_index=True)


def process_race(session, config: dict) -> pd.DataFrame:
    """
    Executes end-to-end preprocessing for all drivers in a session.
    """
    sample_hz = config["preprocess"]["sample_hz"]
    max_interp = config["preprocess"]["max_interpolate_sec"]
    mini_sectors = config["track"]["mini_sectors"]
    lap_length_m = config["track"]["lap_length_m"]

    all_driver_dfs = []
    drivers = session.drivers
    logger.info(f"Preprocessing telemetry for {len(drivers)} drivers at {sample_hz} Hz...")

    for d in drivers:
        try:
            drv_laps = session.laps.pick_driver(d)
            if drv_laps.empty:
                continue
            drv_tel = drv_laps.get_telemetry()
            aligned = align_driver_telemetry(
                d, drv_tel, drv_laps,
                sample_hz=sample_hz,
                max_interpolate_sec=max_interp,
                mini_sectors=mini_sectors,
                lap_length_m=lap_length_m
            )
            all_driver_dfs.append(aligned)
        except Exception as err:
            logger.warning(f"Failed to align driver {d}: {err}")

    if not all_driver_dfs:
        raise ValueError("No driver telemetry could be aligned.")

    combined = pd.concat(all_driver_dfs, ignore_index=True)
    logger.info(f"Total aligned rows before gap calculation: {len(combined)}")

    # Compute time gaps to driver ahead
    logger.info("Computing dynamic leader-follower gaps (Delta t)...")
    processed = compute_time_gaps(combined, sample_hz=sample_hz)

    # Save to processed directory
    out_dir = Path(config["session"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "aligned_telemetry.parquet"
    processed.to_parquet(out_file)
    logger.info(f"Saved aligned telemetry table to: {out_file}")

    return processed


if __name__ == "__main__":
    cfg = load_config()
    from ingest import load_race
    sess = load_race(cfg["session"]["year"], cfg["session"]["gp"], cfg["session"]["session_type"])
    process_race(sess, cfg)
