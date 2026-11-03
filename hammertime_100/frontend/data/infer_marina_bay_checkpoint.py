"""Run a bounded, in-sample Singapore inference sample with the saved ST-GNN checkpoint.

This intentionally does not load the multi-gigabyte season cohort pickle. It rebuilds
one recorded Singapore lap's graph windows from the currently cached aligned parquet,
using the scaler/config embedded in best.pt. The aligned cache fingerprint is recorded
because it differs from the training cohort's archived aligned-source fingerprint.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "hammertime_100/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from graph import build_race_graph_snapshots  # noqa: E402

HISTORY = ROOT / ".local/research-history/2023-fastf1-season"
RUN_ID = "769a76aab164b514cc0a"
RACE_ID = "2023-Singapore-R"
DRIVER = "55"
LAP = 1
EXPECTED_CHECKPOINT = "6d8c48fcc9ba4aa61f512c7e1a7721104373cdc4cc377ae23f2b94a8bc5f5690"
ARCHIVED_MODEL_COMMIT = "e76f161924ff218ed37994302a3ac9b6de271e17"
ARCHIVED_MODEL_SHA256 = "5b3e2c3d56f556dd0698f208bf890fe37c43bee3694f4eddd7165fd952af5090"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def main() -> None:
    run = HISTORY / "training" / RUN_ID
    checkpoint_path = run / "best.pt"
    aligned_path = HISTORY / "race_cache/sessions/2023-Singapore-R/aligned_telemetry.parquet"
    labels_path = HISTORY / "race_cache/sessions/2023-Singapore-R/target_labels.parquet"
    checkpoint_sha = sha(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT:
        raise ValueError("Selected checkpoint SHA does not match the archived run record.")
    completion = json.loads((run / "training_complete.json").read_text())
    if completion.get("best_sha256") != checkpoint_sha:
        raise ValueError("Checkpoint differs from training_complete.json.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["model"].get("in_dim") != 9 or config["model"].get("edge_dim") != 1:
        raise ValueError("Checkpoint input architecture differs from the verified legacy 9/1 schema.")
    if (config["data"].get("window_size"), config["data"].get("horizon"),
            config["data"].get("window_stride")) != (10, 5, 2):
        raise ValueError("Unexpected checkpoint window configuration.")

    aligned = pd.read_parquet(aligned_path)
    labels = pd.read_parquet(labels_path)
    if set(aligned["RaceId"].astype(str)) != {RACE_ID} or set(labels["RaceId"].astype(str)) != {RACE_ID}:
        raise ValueError("Input parquet contains unexpected race IDs.")
    frame_times = np.sort(pd.to_numeric(aligned["TimeSec"], errors="coerce").dropna().unique())
    if len(frame_times) < 15 or not np.allclose(np.diff(frame_times), 0.25, atol=1e-6):
        raise ValueError("Current cached Singapore rows do not form the expected 4 Hz grid.")

    lap_labels = labels[(labels["Driver"].astype(str) == DRIVER)
                        & (pd.to_numeric(labels["LapNumber"], errors="coerce") == LAP)].copy()
    if lap_labels.empty:
        raise ValueError(f"No cached target labels for #{DRIVER} lap {LAP}.")
    entry_times = pd.to_numeric(lap_labels["TimeSec"], errors="coerce").to_numpy(float)
    target_indices = np.abs(frame_times[None, :] - entry_times[:, None]).argmin(axis=1)
    target_deltas = np.abs(frame_times[target_indices] - entry_times)
    eligible = (target_deltas <= float(config["data"]["timestamp_tolerance_sec"])) & (target_indices >= 14)
    if not eligible.any():
        raise ValueError("No mini-sector entry has a valid W10/K5 graph-history window.")
    selected = lap_labels.copy()
    selected["target_idx"] = target_indices
    selected["grid_eligible"] = eligible
    first_frame = int(target_indices[eligible].min()) - 14
    last_frame = int(target_indices[eligible].max())
    selected_times = frame_times[first_frame:last_frame + 1]
    aligned = aligned[aligned["TimeSec"].isin(selected_times)].copy()
    cfg = json.loads(json.dumps(config))
    cfg.setdefault("session", {}).update({"race_id": RACE_ID, "year": 2023, "gp": "Singapore Grand Prix"})
    cfg.setdefault("track", {})["lap_length_m"] = 4940.0
    snapshots, local_map = build_race_graph_snapshots(aligned, cfg)
    if len(snapshots) != len(selected_times):
        raise ValueError(f"Graph builder returned {len(snapshots)} frames for {len(selected_times)} requested times.")

    scaler = checkpoint["feature_scaler"]
    mean = torch.as_tensor(scaler["mean_"], dtype=torch.float32)
    scale = torch.as_tensor(scaler["scale_"], dtype=torch.float32)
    for snap in snapshots:
        mask = snap.mask.bool()
        x = snap.x.clone()
        x[mask] = (x[mask] - mean) / scale
        x[~mask] = 0.0
        snap.x = x

    model = archived_create_model()(config).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    records = []
    with torch.no_grad():
        for label in selected.sort_values(["MiniSector", "TimeSec"]).itertuples(index=False):
            target_idx = int(label.target_idx)
            grid_eligible = bool(label.grid_eligible)
            local_target = target_idx - first_frame
            target_snap = snapshots[local_target] if grid_eligible else None
            names = target_snap.driver_names if target_snap is not None else []
            node = names.index(DRIVER) if DRIVER in names else None
            predicted = None
            input_end = None
            status = ("unavailable_no_grid_match" if abs(frame_times[target_idx] - float(label.TimeSec))
                      > float(config["data"]["timestamp_tolerance_sec"]) else "unavailable_no_input_window")
            if node is not None and bool(target_snap.mask[node]) and local_target >= 14:
                full_window = snapshots[local_target - 14:local_target + 1]
                window = full_window[:10]
                if len(full_window) != 15:
                    status = "unavailable_no_window"
                elif any(bool(getattr(snap, "is_safety_car", False)) for snap in full_window):
                    status = "unavailable_safety_car_window"
                elif any(abs(float(full_window[i + 1].time_sec) - float(full_window[i].time_sec) - 0.25) > 0.05
                         for i in range(14)):
                    status = "unavailable_discontinuous_window"
                elif len(window) == 10:
                    predicted = float(model(window)[node].item())
                    input_end = float(window[-1].time_sec)
                    status = "in_sample_inference"
            actual = float(label.DeltaT_Sector)
            records.append({
                "race_id": RACE_ID, "driver": DRIVER, "acronym": "SAI",
                "lap_number": int(label.LapNumber), "mini_sector_index_zero_based": int(label.MiniSector),
                "mini_sector_display": int(label.MiniSector) + 1,
                "target_entry_time_sec": float(label.TimeSec),
                "matched_graph_time_sec": float(frame_times[target_idx]) if grid_eligible else None,
                "input_end_time_sec": input_end,
                "actual_residual_sec": actual,
                "predicted_residual_sec": predicted,
                "absolute_error_sec": abs(predicted - actual) if predicted is not None else None,
                "prediction_status": status,
            })

    valid = [r for r in records if r["predicted_residual_sec"] is not None]
    output = {
        "schema_version": 1,
        "title": "Singapore 2023 bounded checkpoint inference · SAI #55, lap 1",
        "run_id": RUN_ID,
        "checkpoint": {"path": str(checkpoint_path.relative_to(ROOT)), "sha256": checkpoint_sha,
                       "epoch": int(checkpoint["epoch"])},
        "scope": {"race_id": RACE_ID, "split_role": "training race", "held_out_test": False,
                  "in_sample": True, "driver": DRIVER, "lap_number": LAP,
                  "input_window_size": 10, "forecast_horizon_frames": 5,
                  "forecast_lead_sec": 1.25, "sample_period_sec": 0.25,
                  "training_window_stride": int(config["data"].get("window_stride", 1)),
                  "inference_window_stride": 1,
                  "inference_coverage": "all label-aligned windows passing history, target-mask, safety-car, and continuity checks; stride restriction not applied to this preview",
                  "safety_car_windows_excluded": True,
                  "note": "Checkpoint forward pass on recorded cached Singapore graph features for every label-aligned window passing the saved history, target-mask, safety-car, and continuity rules. No grid-state padding is added; targets without recorded history remain unavailable. The saved training stride is reported but not applied to this bounded preview. Not held-out evaluation, live inference, or causal traffic loss."},
        "source": {"provider": "FastF1 archived project cache", "aligned_path": str(aligned_path.relative_to(ROOT)),
                   "aligned_sha256": sha(aligned_path), "target_labels_path": str(labels_path.relative_to(ROOT)),
                   "target_labels_sha256": sha(labels_path), "graph_builder_sha256": sha(SRC / "graph.py"),
                   "model_builder_sha256": ARCHIVED_MODEL_SHA256,
                   "model_builder_git_commit": ARCHIVED_MODEL_COMMIT,
                   "cache_note": "Aligned cache SHA differs from the source recorded by the original prepared cohort; this is a new bounded inference pass on the current cached features."},
        "target_definition": "Signed residual DeltaT_Sector in seconds: observed sector duration minus the full-session clean-air reference; absolute_error_sec is per-row absolute prediction error.",
        "count": {"target_labels_in_lap": int(len(lap_labels)), "windows_run": len(valid), "unavailable": len(records) - len(valid),
                  "unavailable_by_reason": {key: sum(r["prediction_status"] == key for r in records)
                                             for key in sorted({r["prediction_status"] for r in records if r["predicted_residual_sec"] is None})}},
        "records": records,
        "aggregate": {"mae_sec": float(np.mean([r["absolute_error_sec"] for r in valid])) if valid else None,
                      "rmse_sec": float(np.sqrt(np.mean([r["absolute_error_sec"] ** 2 for r in valid]))) if valid else None,
                      "interpretation": "descriptive errors on this bounded Singapore training-race preview only; not benchmark or held-out performance",
                      "sample_count": len(valid)},
    }
    out = Path(__file__).with_name("marina_bay_inference.json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    tmp.replace(out)
    print(json.dumps({"output": str(out), "counts": output["count"], "aggregate": output["aggregate"],
                      "aligned_sha256": output["source"]["aligned_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
