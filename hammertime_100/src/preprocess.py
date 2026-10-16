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


def _time_seconds(values: pd.Series) -> pd.Series:
    """Convert FastF1 time values to seconds without changing their origin."""
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds()
    return pd.to_numeric(values, errors="coerce")


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
    lap_length_m: float = 5793.0,
    time_grid: np.ndarray | None = None,
    flag_safety_car: bool = True,
    flag_pit_laps: bool = True,
) -> pd.DataFrame:
    """
    Resamples a single driver's telemetry to a uniform time grid.
    Computes mini-sector index and merges tyre compound and tyre age.
    """
    if tel_df.empty:
        logger.warning(f"Driver {driver} has no telemetry data.")
        return pd.DataFrame()
    if sample_hz <= 0 or max_interpolate_sec < 0:
        raise ValueError("sample_hz must be positive and max_interpolate_sec must not be negative.")
    if mini_sectors <= 0 or lap_length_m <= 0:
        raise ValueError("mini_sectors and lap_length_m must be positive.")

    tel_df = tel_df.copy()
    # Ensure Time column is in seconds.
    if "TimeSec" in tel_df.columns:
        tel_df["TimeSec"] = _time_seconds(tel_df["TimeSec"])
    elif "Time" in tel_df.columns:
        tel_df["TimeSec"] = _time_seconds(tel_df["Time"])
    else:
        raise ValueError("Telemetry must contain a Time or TimeSec column.")

    tel_sorted = tel_df.dropna(subset=["TimeSec"]).sort_values("TimeSec").drop_duplicates(subset=["TimeSec"])
    if tel_sorted.empty:
        logger.warning(f"Driver {driver} has no valid timestamps.")
        return pd.DataFrame()
    t_start = tel_sorted["TimeSec"].min()
    t_end = tel_sorted["TimeSec"].max()

    dt = 1.0 / sample_hz
    if time_grid is None:
        uniform_times = np.arange(t_start, t_end + dt * 1e-6, dt)
    else:
        uniform_times = np.asarray(time_grid, dtype=float)

    # Base grid
    resampled = pd.DataFrame({"TimeSec": uniform_times})
    resampled["Driver"] = driver

    # Interpolate numerical columns only across short, observed gaps. Rows outside
    # a driver's telemetry coverage remain invalid on the shared race time grid.
    row_valid = (uniform_times >= t_start) & (uniform_times <= t_end)
    interp_cols = ["Speed", "Throttle", "Brake", "Distance", "X", "Y", "Z"]
    for col in interp_cols:
        if col in tel_sorted.columns:
            valid = tel_sorted[["TimeSec", col]].copy()
            valid[col] = pd.to_numeric(valid[col], errors="coerce")
            valid = valid.dropna()
            values = np.full(len(uniform_times), np.nan, dtype=float)
            if len(valid) > 1:
                source_times = valid["TimeSec"].to_numpy(dtype=float)
                source_values = valid[col].to_numpy(dtype=float)
                values[:] = np.interp(uniform_times, source_times, source_values)
                right = np.searchsorted(source_times, uniform_times, side="left")
                exact = (right < len(source_times)) & np.isclose(
                    source_times[np.minimum(right, len(source_times) - 1)], uniform_times, atol=1e-9
                )
                left_idx = np.maximum(right - 1, 0)
                right_idx = np.minimum(right, len(source_times) - 1)
                bracket_gap = source_times[right_idx] - source_times[left_idx]
                supported = (uniform_times >= source_times[0]) & (uniform_times <= source_times[-1])
                supported &= exact | (bracket_gap <= max_interpolate_sec + 1e-9)
                values[~supported] = np.nan
            elif len(valid) == 1:
                source_time = float(valid["TimeSec"].iloc[0])
                exact = np.isclose(uniform_times, source_time, atol=1e-9)
                values[exact] = float(valid[col].iloc[0])
            resampled[col] = values
            if col in ("Speed", "Throttle", "Distance"):
                row_valid &= np.isfinite(values)
        else:
            resampled[col] = np.nan
            if col in ("Speed", "Throttle", "Distance"):
                row_valid[:] = False
    resampled["IsValid"] = row_valid

    # Attach Lap information
    resampled["LapNumber"] = 1
    resampled["Compound"] = "MEDIUM"
    resampled["TyreLife"] = 1.0
    resampled["IsPitLap"] = False
    resampled["IsSafetyCar"] = False
    if not laps_df.empty:
        # Match each timestamp to its corresponding lap
        laps_sorted = laps_df.sort_values("LapNumber").copy()
        if "LapStartTime" in laps_sorted.columns:
            laps_sorted["LapStartSec"] = _time_seconds(laps_sorted["LapStartTime"])
        else:
            laps_sorted["LapStartSec"] = np.nan

        for _, lap in laps_sorted.iterrows():
            lap_num = int(lap.get("LapNumber", 1))
            lap_start = lap.get("LapStartSec", np.nan)
            lap_time = lap.get("LapTime")
            lap_dur = lap_time.total_seconds() if pd.notnull(lap_time) and hasattr(lap_time, "total_seconds") else np.nan
            if pd.isna(lap_start) or pd.isna(lap_dur) or lap_dur <= 0:
                continue
            lap_end = lap_start + lap_dur

            mask = (resampled["TimeSec"] >= lap_start) & (resampled["TimeSec"] < lap_end)
            if mask.any():
                resampled.loc[mask, "LapNumber"] = lap_num
                compound = str(lap.get("Compound", "MEDIUM")).upper()
                resampled.loc[mask, "Compound"] = compound if compound in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"] else "MEDIUM"
                resampled.loc[mask, "TyreLife"] = float(lap.get("TyreLife", 1.0))
                
                # Check pit
                if flag_pit_laps and (pd.notnull(lap.get("PitInTime")) or pd.notnull(lap.get("PitOutTime"))):
                    resampled.loc[mask, "IsPitLap"] = True
                
                # Check track status for safety car (status 4 = SC, 6 = VSC)
                status = str(lap.get("TrackStatus", "1"))
                if flag_safety_car and ("4" in status or "5" in status or "6" in status):
                    resampled.loc[mask, "IsSafetyCar"] = True

    # Calculate mini-sector index [0, M-1] based on distance within lap
    dist_in_lap = resampled["Distance"].fillna(0.0) % lap_length_m
    sector_len = lap_length_m / mini_sectors
    resampled["MiniSector"] = np.clip(np.floor(dist_in_lap / sector_len).astype(int), 0, mini_sectors - 1)

    return resampled


def compute_time_gaps(
    aligned_grid: pd.DataFrame,
    sample_hz: float = 4.0,
    lap_length_m: float = 5793.0,
) -> pd.DataFrame:
    """
    Computes the time gap Delta t_ij to the immediate car ahead at every timestamp.
    Delta t_ij = Distance_ahead / Speed_follower
    """
    results = []
    # Group by uniform timestamp
    for _, group in aligned_grid.groupby("TimeSec", sort=True):
        group = group.copy()
        valid_rows = group.get("IsValid", pd.Series(True, index=group.index)).fillna(False).astype(bool)
        active = group[~group["IsPitLap"].fillna(False).astype(bool) & valid_rows].copy()
        if len(active) <= 1:
            group["DriverAhead"] = None
            group["DeltaT"] = np.nan
            results.append(group)
            continue

        # Sort cars by track progression (LapNumber then Distance)
        # Distance may be cumulative for some sources and lap-relative for others.
        # Reconstruct race distance to handle both conventions and lap crossings.
        active["RaceDistance"] = (
            (active["LapNumber"].astype(float) - 1.0) * lap_length_m
            + active["Distance"].astype(float).mod(lap_length_m)
        )
        active = active.sort_values(by="RaceDistance", ascending=False)
        active_drivers = active["Driver"].tolist()
        
        # Leader of group has no car ahead
        driver_ahead_map = {active_drivers[0]: None}
        gap_map = {active_drivers[0]: 99.0}

        for idx in range(1, len(active_drivers)):
            follower = active_drivers[idx]
            leader = active_drivers[idx - 1]
            
            fol_row = active[active["Driver"] == follower].iloc[0]
            lead_row = active[active["Driver"] == leader].iloc[0]
            
            dist_diff = lead_row["RaceDistance"] - fol_row["RaceDistance"]
            fol_speed_ms = max(fol_row["Speed"] / 3.6, 10.0)  # avoid div by zero; km/h to m/s
            
            # If distance difference is positive and plausible
            if dist_diff > 0:
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
    flag_safety_car = config["preprocess"].get("flag_safety_car", True)
    flag_pit_laps = config["preprocess"].get("flag_pit_laps", True)
    mini_sectors = config["track"]["mini_sectors"]
    lap_length_m = config["track"]["lap_length_m"]

    if sample_hz <= 0:
        raise ValueError("preprocess.sample_hz must be positive.")
    drivers = session.drivers
    logger.info(f"Preprocessing telemetry for {len(drivers)} drivers at {sample_hz} Hz...")

    raw_drivers = []
    for d in drivers:
        try:
            drv_laps = session.laps.pick_driver(d)
            if drv_laps.empty:
                continue
            drv_tel = drv_laps.get_telemetry()
            if drv_tel.empty:
                continue
            time_column = "TimeSec" if "TimeSec" in drv_tel.columns else "Time"
            if time_column not in drv_tel.columns:
                raise ValueError(f"Driver {d} telemetry has no Time or TimeSec column.")
            drv_tel = drv_tel.copy()
            drv_tel["TimeSec"] = _time_seconds(drv_tel[time_column])
            drv_tel = drv_tel.dropna(subset=["TimeSec"])
            if drv_tel.empty:
                continue
            raw_drivers.append((d, drv_tel, drv_laps))
        except Exception as err:
            logger.warning(f"Failed to load telemetry for driver {d}: {err}")

    if not raw_drivers:
        raise ValueError("No driver telemetry could be loaded.")

    dt = 1.0 / sample_hz
    session_start = min(tel["TimeSec"].min() for _, tel, _ in raw_drivers)
    session_end = max(tel["TimeSec"].max() for _, tel, _ in raw_drivers)
    time_grid = np.arange(session_start, session_end + dt * 1e-6, dt)
    all_driver_dfs = []
    for d, drv_tel, drv_laps in raw_drivers:
        try:
            aligned = align_driver_telemetry(
                d, drv_tel, drv_laps,
                sample_hz=sample_hz,
                max_interpolate_sec=max_interp,
                mini_sectors=mini_sectors,
                lap_length_m=lap_length_m,
                time_grid=time_grid,
                flag_safety_car=flag_safety_car,
                flag_pit_laps=flag_pit_laps,
            )
            if not aligned.empty:
                all_driver_dfs.append(aligned)
        except Exception as err:
            logger.warning(f"Failed to align driver {d}: {err}")

    if not all_driver_dfs:
        raise ValueError("No driver telemetry could be aligned.")

    combined = pd.concat(all_driver_dfs, ignore_index=True)
    session_cfg = config.get("session", {})
    race_id = session_cfg.get("race_id") or (
        f"{session_cfg.get('year', 'unknown')}-{session_cfg.get('gp', 'unknown')}-"
        f"{session_cfg.get('session_type', 'R')}"
    )
    combined["RaceId"] = str(race_id)
    logger.info(f"Total aligned rows before gap calculation: {len(combined)}")

    # Compute time gaps to driver ahead
    logger.info("Computing dynamic leader-follower gaps (Delta t)...")
    processed = compute_time_gaps(
        combined,
        sample_hz=sample_hz,
        lap_length_m=lap_length_m,
    )

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
