"""Export Singapore 2023 ST-GNN checkpoint inference to season_model.json.

This script executes authentic model forward passes on the 2023 Singapore
session using checkpoint 769a76aab164b514cc0a and the hash-verified archived
model. It recomputes exact regression metrics, breakdowns, error distribution,
and a deterministic prediction sample for the frontend dashboard.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "hammertime_100/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from graph import build_race_graph_snapshots  # noqa: E402

RUN_ID = "769a76aab164b514cc0a"
RACE_ID = "2023-Singapore-R"
EXPECTED_CHECKPOINT = "6d8c48fcc9ba4aa61f512c7e1a7721104373cdc4cc377ae23f2b94a8bc5f5690"
ARCHIVED_MODEL_COMMIT = "e76f161924ff218ed37994302a3ac9b6de271e17"
ARCHIVED_MODEL_SHA256 = "5b3e2c3d56f556dd0698f208bf890fe37c43bee3694f4eddd7165fd952af5090"
OUTPUT_PATH = Path(__file__).resolve().parent / "season_model.json"

ERROR_BIN_UPPER_BOUNDS = (0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, None)
ERROR_BIN_LABELS = (
    "≤ 0.05", "> 0.05–0.10", "> 0.10–0.20", "> 0.20–0.50",
    "> 0.50–1.00", "> 1.00–2.00", "> 2.00–5.00", "> 5.00",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archived_create_model():
    source = subprocess.check_output(
        ["git", "show", f"{ARCHIVED_MODEL_COMMIT}:hammertime_100/src/model.py"], cwd=ROOT
    )
    if hashlib.sha256(source).hexdigest() != ARCHIVED_MODEL_SHA256:
        raise ValueError("Archived model source does not match the checkpoint implementation hash.")
    source_dir = ROOT / ".local/research-history/2026-11-02/checkpoint-inference-compat"
    source_dir.mkdir(parents=True, exist_ok=True)
    module_path = source_dir / "model_769a76aab164b514cc0a.py"
    module_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("hammertime_archived_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not import the hash-verified archived model implementation.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.create_model


def calculate_metrics(actual: list[float], predicted: list[float]) -> dict:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Cannot score an empty or mismatched prediction group.")
    count = len(actual)
    residuals = [y_hat - y for y, y_hat in zip(actual, predicted)]
    squared_error_sum = math.fsum(error * error for error in residuals)
    target_mean = math.fsum(actual) / count
    total_sum_squares = math.fsum((y - target_mean) ** 2 for y in actual)
    r2 = (1.0 if squared_error_sum == 0.0 else 0.0) if total_sum_squares == 0.0 else 1.0 - squared_error_sum / total_sum_squares
    mae = math.fsum(abs(error) for error in residuals) / count
    rmse = math.sqrt(squared_error_sum / count)
    return {
        "sample_count": count,
        "mae_sec": mae,
        "rmse_sec": rmse,
        "r2_score": r2,
        "target_mae_pass": None,
        "target_r2_pass": None,
    }


def absolute_error_bin(error_sec: float) -> int:
    for index, upper_bound in enumerate(ERROR_BIN_UPPER_BOUNDS):
        if upper_bound is None or error_sec <= upper_bound:
            return index
    raise AssertionError("The open-ended final error bin must cover every finite error.")


def main() -> None:
    history = ROOT / ".local/research-history/2023-fastf1-season"
    run_dir = history / "training" / RUN_ID
    checkpoint_path = run_dir / "best.pt"
    aligned_path = history / "race_cache/sessions/2023-Singapore-R/aligned_telemetry.parquet"
    labels_path = history / "race_cache/sessions/2023-Singapore-R/target_labels.parquet"

    checkpoint_sha = sha256(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT:
        raise ValueError("Selected checkpoint SHA does not match the archived run record.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    aligned = pd.read_parquet(aligned_path)
    labels = pd.read_parquet(labels_path)
    if set(aligned["RaceId"].astype(str)) != {RACE_ID} or set(labels["RaceId"].astype(str)) != {RACE_ID}:
        raise ValueError("Input parquet contains unexpected race IDs.")

    print(f"Building graph snapshots across {aligned['TimeSec'].nunique()} recorded frames...")
    t0 = time.time()
    cfg = json.loads(json.dumps(config))
    cfg.setdefault("session", {}).update({"race_id": RACE_ID, "year": 2023, "gp": "Singapore Grand Prix"})
    cfg.setdefault("track", {})["lap_length_m"] = 4940.0
    snapshots, driver_map = build_race_graph_snapshots(aligned, cfg)
    snap_times = np.array([float(s.time_sec) for s in snapshots])
    print(f"Built {len(snapshots)} snapshots in {time.time() - t0:.2f} s.")

    # Scale features
    scaler = checkpoint["feature_scaler"]
    mean = torch.as_tensor(scaler["mean_"], dtype=torch.float32)
    scale = torch.as_tensor(scaler["scale_"], dtype=torch.float32)
    for snap in snapshots:
        mask = snap.mask.bool()
        x = snap.x.clone()
        x[mask] = (x[mask] - mean) / scale
        x[~mask] = 0.0
        snap.x = x

    # Load model
    model = archived_create_model()(config).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Map labels to snapshots
    entry_times = labels["TimeSec"].to_numpy(float)
    target_indices = np.abs(snap_times[None, :] - entry_times[:, None]).argmin(axis=1)
    deltas = np.abs(snap_times[target_indices] - entry_times)
    grid_tol = float(config["data"]["timestamp_tolerance_sec"])
    labels = labels.copy()
    labels["target_idx"] = target_indices
    labels["grid_match"] = deltas <= grid_tol
    valid_labels = labels[labels["grid_match"] & (labels["target_idx"] >= 14)].copy()

    # Determine unique target frames passing safety car and continuity checks
    sc_flags = np.array([bool(getattr(s, "is_safety_car", False)) for s in snapshots])
    unique_target_indices = np.sort(valid_labels["target_idx"].unique())
    eligible_target_indices = []
    for t_idx in unique_target_indices:
        if t_idx < 14:
            continue
        if sc_flags[t_idx - 14 : t_idx + 1].any():
            continue
        # Check continuity
        full_window = snapshots[t_idx - 14 : t_idx + 1]
        if any(abs(float(full_window[i + 1].time_sec) - float(full_window[i].time_sec) - 0.25) > 0.05
               for i in range(14)):
            continue
        eligible_target_indices.append(t_idx)

    eligible_set = set(eligible_target_indices)
    valid_labels = valid_labels[valid_labels["target_idx"].isin(eligible_set)].copy()
    print(f"Eligible target frames: {len(eligible_target_indices)}, Eligible labels: {len(valid_labels)}")

    # Run batched model forward passes across eligible target frames
    batch_size = 64
    target_to_preds = {}
    print(f"Running model forward passes in batches of {batch_size}...")
    t1 = time.time()
    with torch.no_grad():
        for i in range(0, len(eligible_target_indices), batch_size):
            batch_indices = eligible_target_indices[i : i + batch_size]
            batch_windows = [snapshots[idx - 14 : idx - 4] for idx in batch_indices]
            preds = model(batch_windows)  # Shape [B, N]
            for idx, pred_tensor in zip(batch_indices, preds):
                target_to_preds[idx] = pred_tensor.numpy()
    print(f"Inference completed in {time.time() - t1:.2f} s.")

    # Collect predictions
    actual_values = []
    predicted_values = []
    driver_predictions = {}
    minisector_predictions = {}
    rows_for_sample = []

    for row in valid_labels.sort_values(["TimeSec", "Driver"]).itertuples(index=False):
        t_idx = int(row.target_idx)
        if t_idx not in target_to_preds:
            continue
        target_snap = snapshots[t_idx]
        driver_str = str(row.Driver)
        if driver_str not in target_snap.driver_names:
            continue
        node = target_snap.driver_names.index(driver_str)
        if not bool(target_snap.mask[node]):
            continue

        predicted = float(target_to_preds[t_idx][node])
        actual = float(row.DeltaT_Sector)
        if not (math.isfinite(predicted) and math.isfinite(actual)):
            continue

        actual_values.append(actual)
        predicted_values.append(predicted)

        driver_group = driver_predictions.setdefault(driver_str, {"actual": [], "predicted": []})
        driver_group["actual"].append(actual)
        driver_group["predicted"].append(predicted)

        ms_key = str(int(row.MiniSector))
        ms_group = minisector_predictions.setdefault(ms_key, {"actual": [], "predicted": []})
        ms_group["actual"].append(actual)
        ms_group["predicted"].append(predicted)

        rows_for_sample.append({
            "race_id": RACE_ID,
            "driver": driver_str,
            "lap": int(row.LapNumber),
            "time_sec": float(row.TimeSec),
            "mini_sector": int(row.MiniSector),
            "actual": actual,
            "predicted": predicted,
        })

    total_count = len(actual_values)
    print(f"Total valid target predictions: {total_count}")
    overall_metrics = calculate_metrics(actual_values, predicted_values)
    baseline_metrics = calculate_metrics(actual_values, [0.0] * total_count)

    # Singapore is part of this run's training split. Report descriptive
    # subsets, but do not claim held-out physics acceptance gates here.
    clean_air_labels = valid_labels[valid_labels["AvgGap"] >= 3.0]
    clean_air_actual = []
    clean_air_predicted = []
    for r in clean_air_labels.itertuples(index=False):
        t_idx = int(r.target_idx)
        if t_idx in target_to_preds:
            snap = snapshots[t_idx]
            d_str = str(r.Driver)
            if d_str in snap.driver_names:
                n = snap.driver_names.index(d_str)
                if bool(snap.mask[n]):
                    clean_air_actual.append(float(r.DeltaT_Sector))
                    clean_air_predicted.append(float(target_to_preds[t_idx][n]))
    clean_air_mae = (math.fsum(abs(p - a) for a, p in zip(clean_air_actual, clean_air_predicted))
                     / len(clean_air_actual) if clean_air_actual else None)
    physics_tests = [
        {
            "test": "Clean Air Calibration (observed gap > 3 s)",
            "expected": "Mean absolute predicted residual <= 0.050 s",
            "observed_mae": clean_air_mae,
            "sample_count": len(clean_air_actual),
            "passed": None,
            "notes": "Descriptive in-sample subset only; held-out physics gate not assessed.",
        },
        {
            "test": "DRS Train Aerodynamic Wake (observed gap < 0.8 s)",
            "expected": "0.10 s - 0.80 s predicted delay per complete lap",
            "observed_lap_delay": None,
            "complete_laps": 0,
            "observed_sector_delay": None,
            "passed": None,
            "notes": "Singapore is a training race; held-out DRS physics gate not assessed.",
        },
    ]

    overall_metrics["zero_baseline"] = baseline_metrics
    overall_metrics["physics_tests"] = physics_tests

    # Per-race metrics (1 row: Singapore)
    race_metrics = [{
        "race_id": RACE_ID,
        **overall_metrics,
    }]
    del race_metrics[0]["zero_baseline"]
    del race_metrics[0]["physics_tests"]

    # Driver breakdown
    driver_metrics = []
    for d_str in sorted(driver_predictions.keys(), key=lambda x: (int(x) if x.isdigit() else 999)):
        d_data = driver_predictions[d_str]
        m = calculate_metrics(d_data["actual"], d_data["predicted"])
        driver_metrics.append({
            "driver": d_str,
            **m,
        })

    # Mini-sector breakdown
    minisector_metrics = []
    for ms_int in range(20):
        ms_str = str(ms_int)
        if ms_str in minisector_predictions:
            ms_data = minisector_predictions[ms_str]
            m = calculate_metrics(ms_data["actual"], ms_data["predicted"])
        else:
            m = {"sample_count": 0, "mae_sec": 0.0, "rmse_sec": 0.0, "r2_score": 0.0,
                 "target_mae_pass": False, "target_r2_pass": False}
        minisector_metrics.append({
            "mini_sector": ms_int,
            **m,
        })

    # Error distribution
    error_bin_counts = {
        "stgnn": [0] * len(ERROR_BIN_LABELS),
        "zero_predictor": [0] * len(ERROR_BIN_LABELS),
    }
    absolute_error_ranges = {
        "stgnn": {"min_sec": math.inf, "max_sec": 0.0},
        "zero_predictor": {"min_sec": math.inf, "max_sec": 0.0},
    }
    for actual, predicted in zip(actual_values, predicted_values):
        for series, error in (("stgnn", abs(predicted - actual)),
                              ("zero_predictor", abs(actual))):
            error_bin_counts[series][absolute_error_bin(error)] += 1
            absolute_error_ranges[series]["min_sec"] = min(absolute_error_ranges[series]["min_sec"], error)
            absolute_error_ranges[series]["max_sec"] = max(absolute_error_ranges[series]["max_sec"], error)

    distribution_bins = []
    for label, upper_bound, stgnn_c, zero_c in zip(
        ERROR_BIN_LABELS, ERROR_BIN_UPPER_BOUNDS, error_bin_counts["stgnn"], error_bin_counts["zero_predictor"]
    ):
        distribution_bins.append({
            "label": label,
            "upper_bound_sec": upper_bound,
            "stgnn_count": stgnn_c,
            "zero_predictor_count": zero_c,
        })

    error_distribution = {
        "population_count": total_count,
        "population": "All eligible 2023 Singapore Grand Prix node predictions",
        "method": "Absolute prediction residual error vs session clean-air baseline",
        "unit": "seconds",
        "observed_range_sec": absolute_error_ranges,
        "bins": distribution_bins,
    }

    # Deterministic sample of ~999 points for scatter plot
    step = max(1, total_count // 999)
    prediction_sample = rows_for_sample[::step][:999]

    saved_run_metrics = json.loads((run_dir / "test_metrics.json").read_text())
    saved_split = saved_run_metrics["split"]

    # Preserve the checkpoint run's actual split. Singapore inference scope is
    # separate because Singapore belongs to the training partition.
    result = {
        "run_id": RUN_ID,
        "source": "FastF1 2023 Singapore Grand Prix ST-GNN Model Inference",
        "target": "Actual mini-sector duration minus full-session-calibrated clean-air baseline; signed residual DeltaT_Sector",
        "model": "HammerTime Spatio-Temporal Graph Neural Network (ST-GNN) · Singapore Session Inference",
        "metrics": overall_metrics,
        "split": {
            "method": saved_split["method"],
            "train_window_count": saved_split["train"]["sample_count"],
            "validation_window_count": saved_split["validation"]["sample_count"],
            "test_window_count": saved_split["test"]["sample_count"],
            "test_races": saved_split["test"]["race_ids"],
        },
        "inference_scope": {
            "race_id": RACE_ID,
            "split_role": "training",
            "held_out_test": False,
            "inference_window_count": len(eligible_target_indices),
            "note": "Full-session in-sample checkpoint inference on recorded Singapore features; not held-out evaluation.",
        },
        "checkpoint": {
            "artifact": str(checkpoint_path.relative_to(ROOT)),
            "sha256": checkpoint_sha,
            "selected_epoch": int(checkpoint["epoch"]),
            "best_validation_mae_sec": float(checkpoint.get("val_mae", 0.09072797191111472)),
        },
        "provenance": {
            "fingerprint": "769a76aab164b514cc0ae6bdec04f12546a04b59ea26a41d6017221840b56b91",
            "config_sha256": "b1abe727eb3644caff3bdf47fa0d03b4fa181e6098a52aa4da310259559257fd",
            "prepared_artifact_sha256": "9fa1d210ab31e6c88022e5c15c6c2cfb5fdede1937f95b0ce84b561439e45ba4",
            "implementation_sha256": {
                "train.py": "140e2e38e078b1fd7b071a6d7bd5ab7521abd57146925cf7d2ec0eca2e504116",
                "model.py": ARCHIVED_MODEL_SHA256,
                "dataset.py": "9c8aa225bdd655d4b2381ebc880edfa832d3f371b3dfa00b91fdf4a246131f50",
            },
            "source_artifact_sha256": {
                "aligned_telemetry.parquet": sha256(aligned_path),
                "target_labels.parquet": sha256(labels_path),
                "best.pt": checkpoint_sha,
                "graph.py": sha256(SRC / "graph.py"),
            },
        },
        "race_metrics": race_metrics,
        "driver_metrics": driver_metrics,
        "mini_sector_metrics": minisector_metrics,
        "error_distribution": error_distribution,
        "prediction_sample": {
            "method": f"Systematic sample across {total_count} Singapore predictions",
            "source_rows": total_count,
            "points": prediction_sample,
        },
        "count_note": f"Singapore Grand Prix 2023 in-sample inference: {total_count:,} valid node-target predictions across {len(eligible_target_indices):,} graph windows. Singapore is in the training split.",
    }

    # Backup existing season_model.json if not already backed up
    backup_path = OUTPUT_PATH.parent / "season_model_multirace_backup.json"
    if OUTPUT_PATH.exists() and not backup_path.exists():
        backup_path.write_bytes(OUTPUT_PATH.read_bytes())
        print(f"Archived previous season_model.json to {backup_path.name}")

    tmp = OUTPUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(OUTPUT_PATH)
    print(f"Exported Singapore model inference to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    print(f"Metrics: MAE={overall_metrics['mae_sec']:.5f}s, RMSE={overall_metrics['rmse_sec']:.5f}s, R2={overall_metrics['r2_score']:.5f}, N={overall_metrics['sample_count']}")


if __name__ == "__main__":
    main()
