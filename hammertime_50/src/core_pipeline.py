"""
HammerTime 50%: Core Pipeline
Unified lightweight implementation of telemetry alignment, gap computation,
and clean-air baseline target calculation.
Supports actual FastF1 race data (e.g. 2021 Abu Dhabi Grand Prix) with local caching.
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
logger = logging.getLogger("hammertime.core_pipeline")


def load_core_config(config_path: str = "configs/core.yaml") -> dict:
    """Load configuration from YAML."""
    p = Path(config_path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / "configs" / "core.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fastf1_race_telemetry(
    year: int = 2021,
    gp: str = "Abu Dhabi",
    session_type: str = "R",
    cache_dir: str = "data/cache",
    sample_hz: float = 4.0,
    lap_length_m: float = 5281.0,
    mini_sectors: int = 20,
    max_drivers: int = 20
) -> pd.DataFrame:
    """
    Downloads and aligns actual race telemetry from FastF1 for the specified race.
    Caches processed aligned table to Parquet for instant offline re-use.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    parquet_cache = cache_path / f"aligned_{year}_{gp.lower().replace(' ', '_')}_{session_type.lower()}.parquet"

    if parquet_cache.exists():
        logger.info(f"Loading cached real FastF1 telemetry from: {parquet_cache}")
        return pd.read_parquet(parquet_cache)

    logger.info(f"Downloading actual FastF1 data for {year} {gp} [{session_type}]...")
    import fastf1
    fastf1.Cache.enable_cache(str(cache_path))

    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)

    drivers = session.drivers[:max_drivers]
    logger.info(f"Loaded session: {session.event['EventName']}. Aligning {len(drivers)} drivers at {sample_hz} Hz...")

    aligned_dfs = []
    dt = 1.0 / sample_hz

    # Determine common time window across top drivers
    t_min_common = float("inf")
    t_max_common = float("-inf")

    driver_telemetries = {}
    driver_laps_dict = {}

    for d in drivers:
        try:
            drv_laps = session.laps.pick_driver(d)
            if drv_laps.empty:
                continue
            drv_tel = drv_laps.get_telemetry()
            if drv_tel.empty:
                continue
            # Ensure time in seconds
            if "Time" in drv_tel.columns and hasattr(drv_tel["Time"], "dt"):
                drv_tel["TimeSec"] = drv_tel["Time"].dt.total_seconds()
            else:
                drv_tel["TimeSec"] = drv_tel["Time"].astype(float)

            t_min_common = min(t_min_common, drv_tel["TimeSec"].min())
            t_max_common = max(t_max_common, drv_tel["TimeSec"].max())
            driver_telemetries[d] = drv_tel.sort_values("TimeSec").drop_duplicates("TimeSec")
            driver_laps_dict[d] = drv_laps
        except Exception as e:
            logger.warning(f"Could not load telemetry for driver {d}: {e}")

    if not driver_telemetries:
        raise RuntimeError("No driver telemetry could be extracted from FastF1.")

    uniform_times = np.arange(t_min_common, t_max_common, dt)

    # Resample each driver onto uniform grid
    for d, drv_tel in driver_telemetries.items():
        drv_laps = driver_laps_dict[d]
        res = pd.DataFrame({"TimeSec": uniform_times})
        res["Driver"] = d

        for col in ["Speed", "Throttle", "Distance"]:
            if col in drv_tel.columns:
                valid = drv_tel[["TimeSec", col]].dropna()
                res[col] = np.interp(uniform_times, valid["TimeSec"].values, valid[col].values)
            else:
                res[col] = 0.0

        # Assign lap number and mini-sector
        dist_in_lap = res["Distance"] % lap_length_m
        sec_len = lap_length_m / mini_sectors
        res["MiniSector"] = np.clip(np.floor(dist_in_lap / sec_len).astype(int), 0, mini_sectors - 1)
        res["LapNumber"] = (res["Distance"] // lap_length_m).astype(int) + 1
        res["Compound"] = "MEDIUM"
        res["TyreLife"] = res["LapNumber"].astype(float)
        res["IsPitLap"] = False

        aligned_dfs.append(res)

    combined = pd.concat(aligned_dfs, ignore_index=True)

    # Compute leader-follower gap Delta t_ij at each time step
    logger.info("Computing actual gap Delta t_ij to driver ahead...")
    results = []
    for t, grp in combined.groupby("TimeSec"):
        grp_sorted = grp.sort_values(by=["Distance"], ascending=False)
        active_drvs = grp_sorted["Driver"].tolist()

        ahead_map = {active_drvs[0]: None}
        gap_map = {active_drvs[0]: 99.0}

        for idx in range(1, len(active_drvs)):
            fol = active_drvs[idx]
            lead = active_drvs[idx - 1]
            fol_row = grp_sorted[grp_sorted["Driver"] == fol].iloc[0]
            lead_row = grp_sorted[grp_sorted["Driver"] == lead].iloc[0]

            dist_diff = lead_row["Distance"] - fol_row["Distance"]
            fol_speed_ms = max(fol_row["Speed"] / 3.6, 10.0)

            if 0 < dist_diff < 1500.0:
                dt_gap = dist_diff / fol_speed_ms
            else:
                dt_gap = 99.0

            ahead_map[fol] = lead
            gap_map[fol] = dt_gap

        grp["DriverAhead"] = grp["Driver"].map(ahead_map)
        grp["DeltaT"] = grp["Driver"].map(gap_map)
        results.append(grp)

    final_df = pd.concat(results, ignore_index=True)
    logger.info(f"Saving aligned FastF1 telemetry to: {parquet_cache}")
    final_df.to_parquet(parquet_cache)
    return final_df


def synthesize_core_race_data(num_drivers: int = 20, num_snapshots: int = 2000, sample_hz: float = 4.0):
    """
    Creates high-fidelity Grand Prix telemetry for core demonstration fallback.
    """
    logger.info(f"Generating core telemetry data ({num_drivers} drivers, {num_snapshots} snapshots)...")
    drivers = ["VER", "PER", "HAM", "RUS", "LEC", "SAI", "NOR", "PIA", "ALO", "STR",
               "GAS", "OCO", "ALB", "SAR", "TSU", "RIC", "BOT", "ZHO", "MAG", "HUL"][:num_drivers]

    dt = 1.0 / sample_hz
    timestamps = np.arange(0, num_snapshots * dt, dt)
    lap_length_m = 5793.0
    rows = []

    for t in timestamps:
        for d_idx, drv in enumerate(drivers):
            dist = (t * (270.0 / 3.6) + (num_drivers - d_idx) * 95.0) % (lap_length_m * 20)
            dist_in_lap = dist % lap_length_m
            lap_num = int(dist // lap_length_m) + 1
            mini_sec = int(np.clip(np.floor(dist_in_lap / (lap_length_m / 20)), 0, 19))
            speed = 260.0 + 40.0 * np.sin(dist_in_lap / 450.0)

            if d_idx == 0:
                gap = 99.0
                ahead = None
            elif 4 <= d_idx <= 8:
                gap = 0.60 + np.random.uniform(0.02, 0.15)
                ahead = drivers[d_idx - 1]
            else:
                gap = 1.6 + (d_idx % 4) * 0.8
                ahead = drivers[d_idx - 1]

            rows.append({
                "TimeSec": t,
                "Driver": drv,
                "Speed": max(speed, 60.0),
                "Throttle": 95.0 if speed > 250 else 60.0,
                "Distance": dist,
                "LapNumber": lap_num,
                "MiniSector": mini_sec,
                "Compound": "MEDIUM",
                "TyreLife": float(lap_num),
                "IsPitLap": False,
                "DriverAhead": ahead,
                "DeltaT": gap
            })

    df = pd.DataFrame(rows)
    logger.info(f"Synthesized {len(df)} aligned telemetry instances.")
    return df


def calculate_core_clean_air_targets(df: pd.DataFrame, clean_air_gap: float = 3.0) -> pd.DataFrame:
    """
    Computes baseline duration per mini-sector and delta t_sector target.
    Formula: Delta_t_sector = t_actual - t_clean_air_baseline
    """
    durations = df.groupby(["Driver", "LapNumber", "MiniSector", "Compound"]).agg(
        Duration=("TimeSec", lambda x: max(x) - min(x)),
        AvgGap=("DeltaT", "mean"),
        TimeSec=("TimeSec", "min")
    ).reset_index()

    clean_laps = durations[durations["AvgGap"] > clean_air_gap]
    baseline_lookup = clean_laps.groupby("MiniSector")["Duration"].median().to_dict()
    global_median = durations["Duration"].median()

    durations["Baseline"] = durations["MiniSector"].map(lambda s: baseline_lookup.get(s, global_median))
    durations["DeltaT_Sector"] = durations["Duration"] - durations["Baseline"]
    durations["DeltaT_Sector"] = np.clip(durations["DeltaT_Sector"], -1.0, 5.0)

    logger.info(f"Calculated targets for {len(durations)} mini-sector segments.")
    return durations


if __name__ == "__main__":
    df = load_fastf1_race_telemetry(2021, "Abu Dhabi")
    print(f"Loaded {len(df)} rows of actual 2021 Abu Dhabi FastF1 telemetry.")
