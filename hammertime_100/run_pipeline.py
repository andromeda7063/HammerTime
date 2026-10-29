"""Run the configured FastF1-to-evaluation HammerTime pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
VISUALS_DIR = PROJECT_DIR / "visuals"
BROADCAST_DIR = PROJECT_DIR / "broadcast"
for path in (SRC_DIR, VISUALS_DIR, BROADCAST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hammertime.pipeline")


def _session_specs(config: dict) -> list[dict]:
    """Return configured sessions, retaining the original single-session form."""
    sessions = config.get("sessions")
    if sessions is None or sessions == []:
        sessions = [config["session"]]
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("sessions must be a non-empty list when provided.")
    normalized = []
    seen = set()
    seen_sources, seen_directories = set(), set()
    for item in sessions:
        if not isinstance(item, dict) or not {"year", "gp"}.issubset(item):
            raise ValueError("Each sessions entry must contain year and gp; session_type defaults to 'R'.")
        spec = {**config["session"], **item}
        if (isinstance(spec["year"], bool) or not isinstance(spec["year"], int)
                or spec["year"] <= 0 or not isinstance(spec["gp"], str) or not spec["gp"].strip()):
            raise ValueError("Each sessions entry needs a positive integer year and non-empty gp name.")
        spec["session_type"] = str(spec.get("session_type", "R"))
        source = (spec["year"], spec["gp"].strip().casefold(), spec["session_type"].strip().upper())
        if source in seen_sources:
            raise ValueError(f"Duplicate configured session: {source}")
        seen_sources.add(source)
        if spec["gp"].strip().casefold() != str(config["session"]["gp"]).strip().casefold() and "lap_length_m" not in item:
            raise ValueError(f"Session {spec['gp']} requires its own lap_length_m; the default is for {config['session']['gp']}.")
        if "lap_length_m" in item and (
                isinstance(item["lap_length_m"], bool) or not isinstance(item["lap_length_m"], (int, float))
                or not np.isfinite(item["lap_length_m"]) or item["lap_length_m"] <= 0):
            raise ValueError("sessions.lap_length_m must be a finite positive distance.")
        race_id = str(spec.get("race_id") or f"{spec['year']}-{spec['gp']}-{spec['session_type']}")
        if race_id in seen:
            raise ValueError(f"Duplicate configured race_id: {race_id}")
        seen.add(race_id)
        directory = _safe_component(race_id)
        if directory in seen_directories:
            raise ValueError(f"Configured race IDs share the same output directory: {directory}")
        seen_directories.add(directory)
        spec["race_id"] = race_id
        normalized.append(spec)
    return normalized


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "session"


def _preserve_artifact(path: Path, processed_dir: Path) -> None:
    if path.is_file():
        archive_dir = processed_dir / "artifact_history" / uuid4().hex
        archive_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(path, archive_dir / path.name)


def _contiguous_lap_ranges(laps: list[int]) -> list[tuple[int, int]]:
    if not laps:
        return []
    ordered = sorted(set(int(value) for value in laps))
    ranges = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return ranges


def _render_real_outputs(aligned: pd.DataFrame, snapshots: list, labels: pd.DataFrame,
                         report: dict, config: dict) -> list[str]:
    """Render visuals and cards only from aligned telemetry and held-out results."""
    from v1_track_map import generate_track_map
    from v2_pred_vs_actual import generate_pred_vs_actual_plot
    from v3_delay_heatmap import generate_delay_heatmap
    from v4_drs_timeline import generate_drs_timeline
    from v5_baseline_duel import generate_baseline_duel_chart
    from v6_physics_cards import render_physics_validation_cards
    from v7_blockers import generate_blocker_drilldown
    from v8_residuals import generate_residual_sanity_plots
    from v9_clean_air_cal import generate_clean_air_calibration_plot
    from cards import build_broadcast_cards, generate_all_broadcast_cards

    generated = []
    presentation_dir = Path(config["session"]["processed_dir"]) / "presentation" / uuid4().hex
    vis_dir = presentation_dir / "visuals"
    cards_dir = presentation_dir / "cards"
    vis_dir.mkdir(parents=True, exist_ok=False)
    cards_dir.mkdir(parents=True, exist_ok=True)
    report["presentation"] = {"visuals_dir": str(vis_dir.resolve()),
                              "cards_dir": str(cards_dir.resolve()), "artifacts": generated}
    predictions = report.get("predictions", [])
    prediction_df = pd.DataFrame(predictions)
    metrics = report.get("metrics", {})
    physics = report.get("physics_tests", [])
    physics_by_name = {item.get("test", ""): item for item in physics}
    clean_result = next((v for k, v in physics_by_name.items() if "Clean Air" in k), {})
    drs_result = next((v for k, v in physics_by_name.items() if "DRS Train" in k), {})
    race_ids = [str(value) for value in prediction_df.get("RaceId", pd.Series(dtype=str)).dropna().unique()]
    race_id = race_ids[-1] if race_ids else (str(aligned["RaceId"].iloc[0]) if "RaceId" in aligned and not aligned.empty else None)
    race_rows = aligned[aligned["RaceId"].astype(str) == race_id] if race_id and "RaceId" in aligned else aligned
    race_predictions = prediction_df[prediction_df.get("RaceId", pd.Series(index=prediction_df.index, dtype=object)).astype(str) == race_id] if race_id else prediction_df

    # V1 uses positions and directed edges from one observed graph snapshot.
    for snapshot in reversed(snapshots):
        if race_id and str(getattr(snapshot, "race_id", "")) != race_id:
            continue
        at_time = race_rows[np.isclose(race_rows["TimeSec"].to_numpy(dtype=float), float(snapshot.time_sec), rtol=0)]
        if at_time.empty:
            continue
        position_rows = at_time.dropna(subset=["X", "Y"]) if {"X", "Y"}.issubset(at_time.columns) else pd.DataFrame()
        position_rows = position_rows[np.isfinite(position_rows[["X", "Y"]].to_numpy(dtype=float)).all(axis=1)] if not position_rows.empty else position_rows
        if position_rows.empty:
            continue
        positions = {str(row.Driver): (float(row.X), float(row.Y)) for row in position_rows.itertuples()}
        names = list(getattr(snapshot, "driver_names", ()))
        edge_index = snapshot.edge_index.detach().cpu().numpy()
        edge_weights = snapshot.edge_attr.detach().cpu().numpy().reshape(-1)
        gap_by_driver = at_time.set_index("Driver")["DeltaT"].to_dict() if "DeltaT" in at_time else {}
        edges = []
        for edge_idx, (source, destination) in enumerate(edge_index.T):
            if source < len(names) and destination < len(names) and names[source] and names[destination]:
                follower = names[destination]
                gap = gap_by_driver.get(follower)
                if pd.notna(gap):
                    edges.append((names[source], follower, float(edge_weights[edge_idx]), float(gap)))
        route = race_rows[race_rows["Driver"].astype(str) == next(iter(positions))]
        if "LapNumber" in route and not route.empty:
            route = route[route["LapNumber"] == route["LapNumber"].iloc[-1]]
        route = route.sort_values("TimeSec").dropna(subset=["X", "Y"])
        track_xy = route[["X", "Y"]].to_numpy(dtype=float)
        track_xy = track_xy[np.isfinite(track_xy).all(axis=1)]
        output = vis_dir / "v1_track_map.png"
        generate_track_map(positions, edges, circuit_name=str(race_id or "Configured session"),
                           lap_num=int(at_time["LapNumber"].max()), output_path=str(output), track_xy=track_xy)
        generated.append(str(output))
        break
    if not any(path.endswith("v1_track_map.png") for path in generated):
        logger.info("V1 track map unavailable: no observed positions for the configured session.")

    if not prediction_df.empty and {"y_true", "y_pred"}.issubset(prediction_df.columns):
        output = vis_dir / "v2_pred_vs_actual.png"
        generate_pred_vs_actual_plot(
            prediction_df["y_true"].to_numpy(), prediction_df["y_pred"].to_numpy(),
            mae=float(metrics["mae_sec"]),
            r2=(float(metrics["r2_score"]) if metrics.get("r2_score") is not None else None),
            circuit_name="Configured sessions (held-out aggregate)", output_path=str(output),
        )
        generated.append(str(output))

        sectors = int(config["track"]["mini_sectors"])
        drivers = sorted(race_predictions["Driver"].dropna().astype(str).unique()) if "Driver" in race_predictions else []
        matrix = np.full((len(drivers), sectors), np.nan, dtype=float)
        if drivers and {"MiniSector", "y_pred"}.issubset(race_predictions.columns):
            for row_index, driver in enumerate(drivers):
                rows = race_predictions[race_predictions["Driver"].astype(str) == driver]
                for sector, group in rows.groupby("MiniSector"):
                    sector_index = int(sector)
                    if 0 <= sector_index < sectors:
                        matrix[row_index, sector_index] = float(group["y_pred"].mean())
        if matrix.size and np.isfinite(matrix).any():
            drs_limit = float(config["physics_tests"]["drs_max_gap_sec"])
            drs_rows = []
            if "gap_sec" in race_predictions:
                for row_index, driver in enumerate(drivers):
                    gaps = pd.to_numeric(
                        race_predictions.loc[race_predictions["Driver"].astype(str) == driver, "gap_sec"],
                        errors="coerce",
                    )
                    if (gaps.gt(0) & gaps.lt(drs_limit)).any():
                        drs_rows.append(row_index)
            output = vis_dir / "v3_delay_heatmap.png"
            generate_delay_heatmap(matrix, drivers, mini_sectors=sectors, drs_cars=drs_rows,
                                   output_path=str(output))
            generated.append(str(output))

        # V4 plots predicted loss only for laps containing every configured sector.
        complete_laps = {}
        drs_laps = set()
        if {"Driver", "LapNumber", "MiniSector", "y_true", "y_pred", "gap_sec"}.issubset(race_predictions.columns):
            lap_keys = ["Driver", "LapNumber"]
            for (driver, lap), group in race_predictions.groupby(lap_keys):
                sector_values = group.drop_duplicates("MiniSector")
                expected_sectors = set(range(sectors))
                observed_sectors = set(pd.to_numeric(sector_values["MiniSector"], errors="coerce").dropna().astype(int))
                if observed_sectors != expected_sectors:
                    continue
                complete_laps.setdefault(str(driver), {})[int(lap)] = float(sector_values["y_pred"].sum())
                threshold = float(config["physics_tests"]["drs_max_gap_sec"])
                if sector_values["gap_sec"].notna().all() and sector_values["gap_sec"].gt(0).all() and sector_values["gap_sec"].lt(threshold).all():
                    drs_laps.add(int(lap))
        all_laps = sorted({lap for values in complete_laps.values() for lap in values})
        if all_laps:
            penalties = {driver: np.asarray([values.get(lap, np.nan) for lap in all_laps], dtype=float)
                         for driver, values in complete_laps.items()}
            output = vis_dir / "v4_drs_timeline.png"
            generate_drs_timeline(np.asarray(all_laps), penalties, _contiguous_lap_ranges(list(drs_laps)),
                                  target_band=(float(config["physics_tests"]["drs_train_min_lap_loss"]),
                                               float(config["physics_tests"]["drs_train_max_lap_loss"])),
                                  output_path=str(output))
            generated.append(str(output))

        # V5 uses held-out scores; unavailable baselines are rendered as N/A.
        baseline_scores = report.get("baselines", {})
        duel = {"HammerTime": {"mae": metrics.get("mae_sec"), "r2": metrics.get("r2_score")}}
        for key, name in (("xgboost", "XGBoost"), ("lstm", "LSTM")):
            score = baseline_scores.get(key, {})
            duel[name] = {"mae": score.get("mae_sec") if score.get("available", True) else None,
                          "r2": score.get("r2_score") if score.get("available", True) else None}
        output = vis_dir / "v5_baseline_duel.png"
        generate_baseline_duel_chart(duel, output_path=str(output))
        generated.append(str(output))

    if clean_result and drs_result:
        output = vis_dir / "v6_physics_cards.png"
        render_physics_validation_cards(clean_result, drs_result, output_path=str(output))
        generated.append(str(output))

    if not race_predictions.empty and {"Driver", "LapNumber", "gap_sec", "y_pred"}.issubset(race_predictions.columns):
        positive_predictions = race_predictions.assign(_positive_delay=race_predictions["y_pred"].clip(lower=0))
        driver_totals = positive_predictions.groupby("Driver")["_positive_delay"].sum().sort_values(ascending=False)
        focus_driver = str(driver_totals.index[0])
        focus_rows = race_predictions[race_predictions["Driver"].astype(str) == focus_driver]
        telemetry_keys = ["RaceId", "Driver", "LapNumber", "MiniSector", "TimeSec"]
        telemetry_keys = [key for key in telemetry_keys if key in aligned.columns and key in focus_rows.columns]
        joined = focus_rows.merge(aligned, on=telemetry_keys, how="left", suffixes=("", "_observed")) if telemetry_keys else pd.DataFrame()
        blockers = []
        if not joined.empty and "DriverAhead" in joined.columns:
            for leader, group in joined.dropna(subset=["DriverAhead"]).groupby("DriverAhead"):
                blockers.append((str(leader), float(group["y_pred"].clip(lower=0).sum()), float(group["gap_sec"].mean())))
            blockers.sort(key=lambda item: item[1], reverse=True)
        lap_history = race_rows[race_rows["Driver"].astype(str) == focus_driver].groupby("LapNumber")["DeltaT"].mean()
        output = vis_dir / "v7_blockers.png"
        generate_blocker_drilldown(
            focus_driver=focus_driver,
            total_seconds_lost=float(focus_rows["y_pred"].clip(lower=0).sum()),
            top_blockers=blockers[:3],
            lap_history=lap_history.index.to_numpy(dtype=float),
            gap_history=lap_history.to_numpy(dtype=float),
            output_path=str(output),
        )
        generated.append(str(output))

        residuals = race_predictions["y_pred"].to_numpy(dtype=float) - race_predictions["y_true"].to_numpy(dtype=float)
        prediction_telemetry_keys = [key for key in telemetry_keys
                                     if key in aligned.columns and key in race_predictions.columns]
        joined = race_predictions.merge(aligned, on=prediction_telemetry_keys, how="left",
                                       suffixes=("", "_observed")) if prediction_telemetry_keys else pd.DataFrame()
        if not joined.empty and {"TyreLife", "gap_sec"}.issubset(joined.columns):
            output = vis_dir / "v8_residuals.png"
            generate_residual_sanity_plots(residuals, joined["TyreLife"].to_numpy(dtype=float),
                                           joined["gap_sec"].to_numpy(dtype=float), output_path=str(output))
            generated.append(str(output))

        clean_gap = float(config["target"]["clean_air_gap_sec"])
        clean_predictions = race_predictions.loc[
            pd.to_numeric(race_predictions["gap_sec"], errors="coerce") > clean_gap, "y_pred"
        ].to_numpy(dtype=float)
        if np.isfinite(clean_predictions).any():
            output = vis_dir / "v9_clean_air_cal.png"
            generate_clean_air_calibration_plot(clean_predictions,
                                               threshold=float(config["physics_tests"]["clean_air_max_mae"]),
                                               output_path=str(output))
            generated.append(str(output))

    card_data = aligned.copy()
    if "IsValid" in card_data:
        card_data = card_data[card_data["IsValid"].fillna(False)]
    if race_id and "RaceId" in card_data:
        card_data = card_data[card_data["RaceId"].astype(str) == race_id]
    cards = build_broadcast_cards(card_data, predictions=race_predictions.to_dict("records"),
                                  count=10, mini_sectors_per_lap=int(config["track"]["mini_sectors"]))
    if cards:
        _, paths = generate_all_broadcast_cards(output_dir=str(cards_dir), cards=cards)
        generated.extend(paths)
    else:
        logger.info("Broadcast cards unavailable: no observed close-gap telemetry was found.")
    return generated


def run_100_percent_pipeline(config_path: str | Path | None = None, tune: bool = False) -> dict:
    """Run all configured sessions, train/evaluate, and render observed outputs."""
    start_time = time.time()
    logger.info("==================================================================")
    logger.info("  HAMMERTIME PIPELINE EXECUTION ENGINE")
    logger.info("  Compute Device: CPU (High-Performance Multi-Threading)")
    logger.info("  >>> RUNNING TRAINING ON CPU <<<")
    logger.info("==================================================================")
    cfg_path = Path(config_path) if config_path else PROJECT_DIR / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration is not a mapping: {cfg_path}")
    from config_validation import validate_config
    validate_config(config)
    if tune:
        config.setdefault("tuning", {})["enabled"] = True

    base_processed_dir = Path(config["session"]["processed_dir"])
    if not base_processed_dir.is_absolute():
        base_processed_dir = PROJECT_DIR / base_processed_dir
    base_cache_dir = Path(config["session"].get("cache_dir", "data/cache"))
    if not base_cache_dir.is_absolute():
        base_cache_dir = PROJECT_DIR / base_cache_dir
    config["session"]["processed_dir"] = str(base_processed_dir)
    config["session"]["cache_dir"] = str(base_cache_dir)
    checkpoint_path = Path(config["train"].get("checkpoint_path", "best.pt"))
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    config["train"]["checkpoint_path"] = str(checkpoint_path)

    from ingest import load_race
    from ingest import export_raw_tables
    from preprocess import process_race
    from target import build_targets
    from graph import build_race_graph_snapshots

    session_specs = _session_specs(config)
    aligned_tables, label_tables, snapshots = [], [], []
    raw_artifacts = []
    driver_maps = {}
    for spec in session_specs:
        race_id = spec["race_id"]
        race_config = copy.deepcopy(config)
        race_config["session"].update(spec)
        race_config["session"]["race_id"] = race_id
        session_cache_dir = Path(spec.get("cache_dir", base_cache_dir))
        if not session_cache_dir.is_absolute():
            session_cache_dir = PROJECT_DIR / session_cache_dir
        race_config["session"]["cache_dir"] = str(session_cache_dir / _safe_component(race_id))
        race_dir = base_processed_dir / _safe_component(race_id)
        race_config["session"]["processed_dir"] = str(race_dir)
        if "lap_length_m" in spec:
            race_config["track"]["lap_length_m"] = float(spec["lap_length_m"])

        logger.info("Loading configured FastF1 session %s", race_id)
        session = load_race(spec["year"], spec["gp"], spec["session_type"], race_config["session"]["cache_dir"])
        raw_laps_path, raw_telemetry_path = export_raw_tables(session, str(race_dir / "raw"))
        raw_artifacts.append(str(Path(raw_laps_path).resolve()))
        if raw_telemetry_path is not None:
            raw_artifacts.append(str(Path(raw_telemetry_path).resolve()))
        aligned = process_race(session, race_config)
        if "RaceId" not in aligned:
            aligned["RaceId"] = race_id
        labels = build_targets(aligned, race_config)
        race_snapshots, race_driver_map = build_race_graph_snapshots(aligned, race_config)
        aligned_tables.append(aligned)
        label_tables.append(labels)
        snapshots.extend(race_snapshots)
        driver_maps[race_id] = race_driver_map

    if not aligned_tables or not snapshots or not label_tables:
        raise ValueError("Configured sessions produced no aligned telemetry, graph snapshots, or target labels.")
    aligned_all = pd.concat(aligned_tables, ignore_index=True)
    labels_all = pd.concat(label_tables, ignore_index=True)
    base_processed_dir.mkdir(parents=True, exist_ok=True)
    aligned_all.to_parquet(base_processed_dir / "aligned_telemetry.parquet", index=False)
    labels_all.to_parquet(base_processed_dir / "target_labels.parquet", index=False)

    # The model agent uses per-snapshot driver_names for multi-race node identity.
    legacy_driver_map = next(iter(driver_maps.values())) if len(driver_maps) == 1 else {}
    from dataset import create_dataloaders
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        snapshots, labels_all, legacy_driver_map, config
    )

    from model import create_model
    from train import train_model
    device = torch.device("cpu")
    logger.info("[+] Training ST-GNN model on CPU (Max Epochs: %s, Patience: %s)...", config["train"].get("max_epochs", 100), config["train"].get("early_stopping_patience", 15))
    checkpoint_path = Path(config["train"]["checkpoint_path"])
    tuning_result = None
    if config.get("tuning", {}).get("enabled", False):
        from tuning import tune_hyperparameters
        config, tuning_result = tune_hyperparameters(train_loader, val_loader, config, device)
        checkpoint_path = Path(config["train"]["checkpoint_path"])
        best_val_mae = tuning_result["selected_validation_mae_sec"]
    else:
        model = create_model(config).to(device)
        _preserve_artifact(checkpoint_path, base_processed_dir)
        _preserve_artifact(Path(config["train"].get("history_path", checkpoint_path.parent / "training_history.csv")),
                           base_processed_dir)
        best_val_mae, history = train_model(model, train_loader, val_loader, config, device)
    gates = {
        "aligned_data": bool(len(aligned_all) and len(aligned_all["TimeSec"].unique())),
        "targets_and_graphs": bool(len(labels_all) and len(snapshots)),
        "data_split": bool(len(train_loader) and len(val_loader) and len(test_loader)),
        "training": bool(checkpoint_path.is_file() and np.isfinite(best_val_mae)),
    }

    from baselines import run_baselines
    baseline_scores = run_baselines(train_loader.dataset.samples, test_loader.dataset.samples, config)
    gates["baselines"] = all(
        score.get("available", True) and score.get("mae_sec") is not None and score.get("r2_score") is not None
        for score in baseline_scores.values()
    )

    from evaluate import run_full_evaluation
    report_path = Path(config.get("evaluation", {}).get("report_path", base_processed_dir / "evaluation_report.json"))
    _preserve_artifact(report_path, base_processed_dir)
    report = run_full_evaluation(str(checkpoint_path), test_loader, config, baseline_scores)
    if tuning_result is not None:
        report.setdefault("metadata", {})["tuning"] = tuning_result
    physics = report.get("physics_tests", [])
    gates["physics"] = bool(len(physics) == 2 and all(result.get("passed") is True for result in physics))
    artifacts = _render_real_outputs(aligned_all, snapshots, labels_all, report, config)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
    expected_visuals = {f"v{index}_" for index in range(1, 10)}
    present_visuals = {Path(path).name.split("_", 1)[0] + "_" for path in artifacts if "v" in Path(path).name}
    card_count = sum(1 for path in artifacts if Path(path).name.startswith("card_"))
    missing_visuals = sorted(name.rstrip("_").upper() for name in expected_visuals - present_visuals)
    presentation_status = {
        "passed": not missing_visuals and card_count >= 10,
        "missing_visuals": missing_visuals,
        "card_count": card_count,
        "required_card_count": 10,
    }
    gates["presentation"] = presentation_status["passed"]
    gates["all_passed"] = all(gates.values())

    elapsed = time.time() - start_time
    summary = {
        "sessions": [spec["race_id"] for spec in session_specs],
        "snapshot_count": len(snapshots),
        "target_count": len(labels_all),
        "best_validation_mae_sec": float(best_val_mae),
        "metrics": report.get("metrics", {}),
        "gates": gates,
        "presentation_status": presentation_status,
        "artifacts": artifacts,
        "raw_artifacts": raw_artifacts,
        "evaluation_report": str(report_path),
        "checkpoint": str(checkpoint_path),
        "tuning_study": tuning_result["study_path"] if tuning_result else None,
        "elapsed_seconds": elapsed,
    }
    logger.info("Pipeline finished in %.1f seconds. Quality gates passed: %s", elapsed, gates["all_passed"])
    logger.info("Evaluation report: %s", summary["evaluation_report"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Experiment YAML path; defaults to configs/default.yaml.")
    parser.add_argument("--tune", action="store_true", help="Select configured candidates using validation MAE only.")
    parser.add_argument("--retrain", "--force-train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run_100_percent_pipeline(args.config, tune=args.tune)
