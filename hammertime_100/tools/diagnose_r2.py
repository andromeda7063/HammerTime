"""Read-only R² root-cause diagnostics for HammerTime season caches.

The script reads authentic per-race normalized tables (``aligned_telemetry.parquet``
and ``target_labels.parquet``) one race at a time. It never unpickles a prepared
cohort, trains a model, or writes into the source directories.

Usage::

    uv run --frozen python hammertime_100/tools/diagnose_r2.py \
        --normalized-source <run>/race_cache \
        [--raw-sessions-dir <run>/race_cache/raw_sessions] \
        [--output .local/research-history/r2-diagnostics.json]

Outputs are descriptive diagnostics. They are not model results and must not be
reported as held-out performance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V2_CONTEXT_COLUMNS = (
    "SpeedRatioToTrainReference", "SpeedDeltaToTrainReference", "Brake", "DRSActive",
    "a_long", "lap_sin", "lap_cos", "lap_phase", "abs_curvature",
    "lateral_accel_proxy", "ReferenceAvailable",
)


def safe_component(value: str) -> str:
    """Directory name used by the pipeline for a race identifier (run_pipeline._safe_component)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "session"


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return None
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    if total <= 0:
        return None
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / total


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _session_order(source: Path) -> list[dict]:
    manifest = json.loads((source / "normalized_source_manifest.json").read_text(encoding="utf-8"))
    return [{"race_id": str(item["race_id"]), "lap_length_m": float(item["lap_length_m"])}
            for item in manifest.get("sessions", [])]


def label_statistics(labels: pd.DataFrame) -> dict:
    y = pd.to_numeric(labels["DeltaT_Sector"], errors="coerce").dropna()
    quantiles = y.quantile([0.01, 0.5, 0.9, 0.99]).to_dict() if len(y) else {}
    return {
        "count": int(len(y)),
        "mean": _finite(y.mean()) if len(y) else None,
        "std": _finite(y.std()) if len(y) > 1 else None,
        "skew": _finite(y.skew()) if len(y) > 2 else None,
        "quantiles": {str(k): _finite(v) for k, v in quantiles.items()},
        "share_abs_gt_1s": _finite((y.abs() > 1.0).mean()) if len(y) else None,
        "share_lap1": _finite((labels["LapNumber"] == 1).mean()) if "LapNumber" in labels else None,
        "share_at_clip_max": _finite((y >= y.max() - 1e-9).mean()) if len(y) else None,
    }


def context_column_audit(aligned: pd.DataFrame) -> dict:
    """Report whether each v2 context input is present and non-constant."""
    audit = {}
    for column in V2_CONTEXT_COLUMNS:
        if column not in aligned.columns:
            audit[column] = {"present": False, "constant": True}
            continue
        values = pd.to_numeric(aligned[column], errors="coerce")
        audit[column] = {"present": True,
                         "constant": bool(values.nunique(dropna=True) <= 1),
                         "mean_abs": _finite(values.abs().mean())}
    return audit


def lap_trend(labels: pd.DataFrame) -> dict:
    """Variance share explained by a per-race linear trend in lap number."""
    frame = labels[["LapNumber", "DeltaT_Sector"]].dropna()
    if len(frame) < 3 or frame["LapNumber"].nunique() < 2:
        return {"corr_lap": None, "r2_linear_lap": None}
    slope, intercept = np.polyfit(frame["LapNumber"], frame["DeltaT_Sector"], 1)
    fitted = slope * frame["LapNumber"] + intercept
    return {"corr_lap": _finite(frame["LapNumber"].corr(frame["DeltaT_Sector"])),
            "slope_s_per_lap": _finite(slope),
            "r2_linear_lap": _r2(frame["DeltaT_Sector"], fitted)}


def distance_scale_factors(aligned: pd.DataFrame, lap_length_m: float) -> dict:
    """Integrated lap-end distance divided by the configured circuit length."""
    needed = {"Driver", "LapNumber", "Distance"}
    if not needed.issubset(aligned.columns):
        return {"available": False}
    frame = aligned[list(needed)].dropna()
    frame = frame[frame["Distance"] >= 0]
    per_lap = frame.groupby(["Driver", "LapNumber"])["Distance"].max() / float(lap_length_m)
    # The first and last laps are often partial in telemetry coverage.
    per_lap = per_lap[(per_lap > 0.9) & (per_lap < 1.1)]
    if per_lap.empty:
        return {"available": False}
    return {"available": True, "laps": int(len(per_lap)),
            "median": _finite(per_lap.median()),
            "p05": _finite(per_lap.quantile(0.05)), "p95": _finite(per_lap.quantile(0.95)),
            "share_outside_0_3pct": _finite(((per_lap - 1.0).abs() > 0.003).mean()),
            "share_undershoot": _finite((per_lap < 1.0).mean())}


def lap_to_lap_repeatability(labels: pd.DataFrame, clean_gap: float = 3.0) -> dict:
    """Correlation of a driver's clean-air sector residual with the previous lap.

    A low value means most clean-air residual variance is measurement or
    driving noise that no input can forecast; it bounds achievable R².
    """
    frame = labels[["Driver", "LapNumber", "MiniSector", "DeltaT_Sector", "AvgGap"]].dropna()
    frame = frame[frame["AvgGap"] > clean_gap]
    previous = frame.copy()
    previous["LapNumber"] = previous["LapNumber"] + 1
    joined = frame.merge(previous, on=["Driver", "LapNumber", "MiniSector"], suffixes=("", "_prev"))
    if len(joined) < 3:
        return {"pairs": int(len(joined)), "corr_prev_lap": None}
    return {"pairs": int(len(joined)),
            "corr_prev_lap": _finite(joined["DeltaT_Sector"].corr(joined["DeltaT_Sector_prev"]))}


def drs_lap_measurability(labels: pd.DataFrame, mini_sectors: int, drs_gap: float = 0.8) -> dict:
    """Complete driver-laps whose every sector has 0 < gap < drs_gap."""
    frame = labels[["Driver", "LapNumber", "MiniSector", "AvgGap"]].dropna()
    close = frame[(frame["AvgGap"] > 0) & (frame["AvgGap"] < drs_gap)]
    counts = close.groupby(["Driver", "LapNumber"])["MiniSector"].nunique()
    return {"close_sector_labels": int(len(close)),
            "complete_drs_laps_all_labels": int((counts >= mini_sectors).sum()),
            "expected_complete_laps_at_stride2_fraction": 0.5 ** mini_sectors}


def yellow_share(labels: pd.DataFrame, raw_laps: pd.DataFrame | None) -> dict:
    if raw_laps is None or "TrackStatus" not in raw_laps:
        return {"available": False}
    status = raw_laps[["Driver", "LapNumber", "TrackStatus"]].copy()
    if "DriverNumber" in raw_laps:
        # Normalized labels identify cars by racing number, raw laps by abbreviation.
        status["Driver"] = raw_laps["DriverNumber"].astype(str)
    status["Driver"] = status["Driver"].astype(str)
    status["LapNumber"] = pd.to_numeric(status["LapNumber"], errors="coerce")
    labels = labels.assign(Driver=labels["Driver"].astype(str))
    status["yellow"] = status["TrackStatus"].astype(str).str.contains("2|7")
    merged = labels.merge(status[["Driver", "LapNumber", "yellow"]], on=["Driver", "LapNumber"], how="left")
    yellow = merged["yellow"].astype("boolean").fillna(False).astype(bool)
    y = merged["DeltaT_Sector"]
    return {"available": True, "share_yellow_labels": _finite(yellow.mean()),
            "mean_delta_yellow": _finite(y[yellow].mean()) if yellow.any() else None,
            "mean_delta_other": _finite(y[~yellow].mean()) if (~yellow).any() else None}


def split_baselines(per_race_labels: list[pd.DataFrame], race_ids: list[str],
                    train_ratio: float = 0.70, val_ratio: float = 0.15) -> dict:
    """Train-only constants and a contemporaneous-gap ceiling on the race split.

    The gradient-boosting model uses the sector's *own* mean gap (AvgGap), which
    is observed during the outcome window. It is an explanatory ceiling for how
    much variance traffic gap can explain, not a forecast.
    """
    n = len(race_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    train = pd.concat(per_race_labels[:n_train], ignore_index=True)
    test = pd.concat(per_race_labels[n_train + n_val:], ignore_index=True)
    y_train = train["DeltaT_Sector"].to_numpy(float)
    y_test = test["DeltaT_Sector"].to_numpy(float)
    result = {
        "train_races": race_ids[:n_train], "test_races": race_ids[n_train + n_val:],
        "test_r2_train_mean": _r2(y_test, np.full_like(y_test, y_train.mean())),
        "test_r2_train_median": _r2(y_test, np.full_like(y_test, np.median(y_train))),
        "test_mae_train_median": _finite(np.mean(np.abs(y_test - np.median(y_train)))),
    }
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return result

    def design(frame: pd.DataFrame) -> np.ndarray:
        gap = pd.to_numeric(frame["AvgGap"], errors="coerce").clip(upper=5.0).fillna(5.0)
        laps = pd.to_numeric(frame["LapNumber"], errors="coerce").fillna(0.0)
        max_lap = frame.groupby("RaceId")["LapNumber"].transform("max").replace(0, np.nan)
        return np.column_stack([gap, frame["MiniSector"].to_numpy(float),
                                (laps / max_lap).fillna(0.0).to_numpy(float)])

    for loss in ("squared_error", "absolute_error"):
        model = HistGradientBoostingRegressor(loss=loss, max_iter=200, random_state=42)
        model.fit(design(train), y_train)
        pred = model.predict(design(test))
        result[f"test_r2_gap_ceiling_{loss}"] = _r2(y_test, pred)
        result[f"test_mae_gap_ceiling_{loss}"] = _finite(np.mean(np.abs(y_test - pred)))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normalized-source", type=Path, required=True,
                        help="Directory with normalized_source_manifest.json and sessions/.")
    parser.add_argument("--raw-sessions-dir", type=Path, default=None,
                        help="Optional raw_sessions directory (raw_laps.parquet supplies TrackStatus).")
    parser.add_argument("--mini-sectors", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    source = args.normalized_source.expanduser().resolve(strict=True)
    sessions = _session_order(source)
    report = {"artifact_type": "hammertime-r2-diagnostics", "source": str(source),
              "note": "Descriptive diagnostics of authentic normalized tables; not model results.",
              "races": {}}
    per_race_labels, race_ids = [], []
    for session in sessions:
        race_id = session["race_id"]
        race_dir = source / "sessions" / safe_component(race_id)
        aligned = pd.read_parquet(race_dir / "aligned_telemetry.parquet",
                                  columns=None)
        labels = pd.read_parquet(race_dir / "target_labels.parquet")
        raw_laps = None
        if args.raw_sessions_dir is not None:
            raw_root = args.raw_sessions_dir.expanduser()
            raw_path = raw_root / race_id / "raw_laps.parquet"
            if not raw_path.is_file():
                raw_path = raw_root / safe_component(race_id) / "raw_laps.parquet"
            if raw_path.is_file():
                raw_laps = pd.read_parquet(raw_path, columns=["Driver", "DriverNumber", "LapNumber", "TrackStatus"])
        report["races"][race_id] = {
            "labels": label_statistics(labels),
            "context_columns": context_column_audit(aligned),
            "lap_trend": lap_trend(labels),
            "distance_scale": distance_scale_factors(aligned, session["lap_length_m"]),
            "clean_air_repeatability": lap_to_lap_repeatability(labels),
            "drs_measurability": drs_lap_measurability(labels, args.mini_sectors),
            "yellow_flags": yellow_share(labels, raw_laps),
        }
        per_race_labels.append(labels)
        race_ids.append(race_id)
        print(f"[diagnose_r2] {race_id}: {report['races'][race_id]['labels']['count']} labels", file=sys.stderr)
        del aligned
    if len(race_ids) >= 3:
        report["race_split_baselines"] = split_baselines(per_race_labels, race_ids)
    all_labels = pd.concat(per_race_labels, ignore_index=True)
    report["all_labels"] = label_statistics(all_labels)
    text = json.dumps(report, indent=2, allow_nan=False, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
