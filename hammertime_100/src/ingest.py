"""
Module M1: Data Ingestion
Responsibility: Download and cache raw race telemetry and lap data from FastF1.
Standard: ASD-STE100 Plain English
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.ingest")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_cache(cache_dir: str):
    """Enable FastF1 cache in the specified directory."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    try:
        import fastf1
        fastf1.Cache.enable_cache(cache_dir)
        logger.info(f"FastF1 cache enabled at: {cache_dir}")
    except Exception as e:
        logger.warning(f"Could not enable FastF1 cache: {e}")


def load_race(year: int, gp: str, session_type: str = "R", cache_dir: str = "data/cache"):
    """
    Download and return a FastF1 session object.
    Loads laps, telemetry, and weather data.
    """
    init_cache(cache_dir)
    try:
        import fastf1
        logger.info(f"Loading FastF1 session: {year} {gp} [{session_type}]...")
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=True, laps=True, weather=True)
        logger.info(f"Successfully loaded session: {session.event['EventName']}")
        return session
    except Exception as e:
        logger.error(f"Failed to load session from FastF1: {e}")
        raise


def export_raw_tables(session, output_dir: str = "data/cache"):
    """
    Save raw laps and telemetry to Parquet files for offline reproducibility.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save laps
    laps_file = out_path / "raw_laps.parquet"
    session.laps.to_parquet(laps_file)
    logger.info(f"Saved raw laps to {laps_file}")
    
    # Save telemetry per driver
    drivers = session.drivers
    telemetry_dict = {}
    for d in drivers:
        try:
            drv_laps = session.laps.pick_driver(d)
            if not drv_laps.empty:
                tel = drv_laps.get_telemetry()
                tel["Driver"] = d
                telemetry_dict[d] = tel
        except Exception as err:
            logger.warning(f"Could not get telemetry for driver {d}: {err}")
            
    if telemetry_dict:
        all_tel = pd.concat(telemetry_dict.values(), ignore_index=True)
        tel_file = out_path / "raw_telemetry.parquet"
        all_tel.to_parquet(tel_file)
        logger.info(f"Saved raw telemetry to {tel_file} ({len(all_tel)} rows)")
        
    return laps_file, tel_file if telemetry_dict else None


if __name__ == "__main__":
    cfg = load_config()
    year = cfg["session"]["year"]
    gp = cfg["session"]["gp"]
    stype = cfg["session"]["session_type"]
    cdir = cfg["session"]["cache_dir"]
    
    sess = load_race(year, gp, stype, cdir)
    export_raw_tables(sess, cdir)
