"""
Demo data export: Parquet -> JSON for the HammerTime 100 frontend.
Run from repository root: python hammertime_100/frontend/data/demo_data.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]  # -> m:\HammerTIme
H100 = BASE / "hammertime_100"
OUT = H100 / "frontend" / "data"
OUT.mkdir(parents=True, exist_ok=True)

TEL_PATH = H100 / "data" / "processed" / "aligned_telemetry.parquet"
LBL_PATH = H100 / "data" / "processed" / "target_labels.parquet"
REP_PATH = H100 / "data" / "processed" / "evaluation_report.json"


def clean(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def export_telemetry(tel: pd.DataFrame):
    """Sample max 300 rows per driver, keep key columns only."""
    cols = [c for c in ["Driver", "TimeSec", "LapNumber", "Speed", "DeltaT",
                         "DriverAhead", "MiniSector", "Compound", "TyreLife",
                         "X", "Y", "RaceId", "IsValid"] if c in tel.columns]
    sampled = (
        tel[cols]
        .groupby("Driver", group_keys=False)
        .apply(lambda g: g.iloc[::max(1, len(g)//300)])
        .reset_index(drop=True)
    )
    records = []
    for r in sampled.to_dict("records"):
        clean_r = {}
        for k, v in r.items():
            if isinstance(v, (float, np.floating)):
                clean_r[k] = None if (math.isnan(v) or math.isinf(v)) else float(v)
            elif isinstance(v, (int, np.integer)):
                clean_r[k] = int(v)
            else:
                clean_r[k] = v
        records.append(clean_r)
    with open(OUT / "telemetry.json", "w", encoding="utf-8") as f:
        json.dump(records, f, default=str)
    print(f"Exported telemetry.json ({len(records)} rows)")


def export_drivers(tel: pd.DataFrame):
    """Per-driver summary with tyre info, gap stats, and lap counts."""
    drivers = []
    for driver, grp in tel.groupby("Driver"):
        grp_sorted = grp.sort_values("TimeSec")
        compound = grp["Compound"].mode()[0] if "Compound" in grp and not grp["Compound"].empty else "UNKNOWN"
        tyre_age = float(grp["TyreLife"].median()) if "TyreLife" in grp else None
        avg_gap = float(grp["DeltaT"].median()) if "DeltaT" in grp else None
        max_lap = int(grp["LapNumber"].max()) if "LapNumber" in grp else 0
        min_gap = float(grp["DeltaT"].min()) if "DeltaT" in grp else None
        drivers.append({
            "driver": str(driver),
            "compound": str(compound),
            "tyre_age": clean(tyre_age),
            "avg_gap_sec": clean(avg_gap),
            "min_gap_sec": clean(min_gap),
            "max_lap": max_lap,
        })
    drivers.sort(key=lambda d: d["driver"])
    with open(OUT / "drivers.json", "w", encoding="utf-8") as f:
        json.dump(drivers, f)
    print(f"Exported drivers.json ({len(drivers)} drivers)")


def export_race_timeline(tel: pd.DataFrame):
    """Lap-by-lap mean gap to car ahead for each driver."""
    if not {"Driver", "LapNumber", "DeltaT"}.issubset(tel.columns):
        with open(OUT / "race_timeline.json", "w", encoding="utf-8") as f:
            json.dump({}, f)
        return
    timeline = {}
    for driver, grp in tel.groupby("Driver"):
        lap_gap = (
            grp.groupby("LapNumber")["DeltaT"]
            .mean()
            .dropna()
        )
        timeline[str(driver)] = {
            "laps": [int(l) for l in lap_gap.index.tolist()],
            "gaps": [clean(float(g)) for g in lap_gap.values.tolist()],
        }
    with open(OUT / "race_timeline.json", "w", encoding="utf-8") as f:
        json.dump(timeline, f)
    print(f"Exported race_timeline.json ({len(timeline)} drivers)")


def export_sectors(labels: pd.DataFrame):
    """Mini-sector delay data per driver."""
    if not {"Driver", "MiniSector", "DeltaT_Sector"}.issubset(labels.columns):
        with open(OUT / "sectors.json", "w", encoding="utf-8") as f:
            json.dump({}, f)
        return
    sectors = {}
    for driver, grp in labels.groupby("Driver"):
        sector_avg = grp.groupby("MiniSector")["DeltaT_Sector"].mean().dropna()
        sectors[str(driver)] = {
            "sectors": [int(s) for s in sector_avg.index.tolist()],
            "delays": [clean(float(d)) for d in sector_avg.values.tolist()],
        }
    with open(OUT / "sectors.json", "w", encoding="utf-8") as f:
        json.dump(sectors, f)
    print(f"Exported sectors.json ({len(sectors)} drivers)")


def export_metrics():
    """Export model metrics from evaluation_report.json or use synthetic fallback."""
    SYNTHETIC = {
        "mae_sec": 0.087,
        "r2_score": 0.743,
        "sample_count": 2847,
        "target_mae_pass": True,
        "target_r2_pass": True,
        "physics_clean_air": {"passed": True, "observed_mae": 0.031, "sample_count": 412},
        "physics_drs_train": {"passed": True, "observed_lap_delay": 0.612, "complete_laps": 38},
        "baselines": {
            "xgboost": {"mae_sec": 0.134, "r2_score": 0.581},
            "lstm": {"mae_sec": 0.119, "r2_score": 0.624}
        },
        "source": "synthetic"
    }
    if REP_PATH.is_file():
        try:
            with open(REP_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)
            metrics = report.get("metrics", {})
            physics = report.get("physics_tests", [])
            baselines = report.get("baselines", {})
            result = {
                "mae_sec": metrics.get("mae_sec"),
                "r2_score": metrics.get("r2_score"),
                "sample_count": metrics.get("sample_count"),
                "target_mae_pass": metrics.get("target_mae_pass"),
                "target_r2_pass": metrics.get("target_r2_pass"),
                "physics_clean_air": next((p for p in physics if "Clean Air" in p.get("test", "")), {}),
                "physics_drs_train": next((p for p in physics if "DRS Train" in p.get("test", "")), {}),
                "baselines": {
                    k: {"mae_sec": v.get("mae_sec"), "r2_score": v.get("r2_score")}
                    for k, v in baselines.items()
                },
                "source": "pipeline"
            }
            with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print("Exported metrics.json (from evaluation_report.json)")
            return
        except Exception as e:
            print(f"Could not read evaluation report ({e}), using synthetic metrics.")
    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(SYNTHETIC, f, indent=2)
    print("Exported metrics.json (SYNTHETIC fallback — run pipeline to get real metrics)")


if __name__ == "__main__":
    print(f"Loading Parquet data from {H100 / 'data' / 'processed'}")
    if not TEL_PATH.exists():
        print(f"ERROR: {TEL_PATH} not found. Run the pipeline first.")
        sys.exit(1)
    tel = pd.read_parquet(TEL_PATH)
    labels = pd.read_parquet(LBL_PATH) if LBL_PATH.exists() else pd.DataFrame()
    print(f"Telemetry: {tel.shape}, Labels: {labels.shape}")
    export_telemetry(tel)
    export_drivers(tel)
    export_race_timeline(tel)
    export_sectors(labels)
    export_metrics()
    print(f"\nAll demo data exported to {OUT}")
    print("Now serve the frontend: python -m http.server 8000 --directory hammertime_100/frontend")
