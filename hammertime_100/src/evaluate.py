"""M8: score a saved model and audit physical behavior on held-out observations."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import SequentialSampler

from model import create_model
from physical_graph import TYPED_EDGE_SCHEMAS

logger = logging.getLogger("hammertime.evaluate")


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_test_set(model: nn.Module, test_loader, device: torch.device):
    """Return flattened valid targets/predictions and row-level held-out records."""
    # Records are enriched from ``dataset.samples`` below. A shuffled or custom
    # sampler would silently associate predictions with the wrong metadata.
    if not isinstance(getattr(test_loader, "sampler", None), SequentialSampler):
        raise ValueError("Held-out evaluation requires a sequential, non-shuffled data loader.")
    model.eval()
    all_targets, all_preds, records = [], [], []
    samples = getattr(test_loader.dataset, "samples", [])
    inverse_drivers = {idx: driver for driver, idx in getattr(test_loader.dataset, "driver_to_idx", {}).items()}
    sample_index = 0
    for batch in test_loader:
        # Auxiliary runs append speed targets/masks, but held-out scoring
        # continues to use only the primary sector-residual target.
        batch_windows, targets, masks = batch[:3]
        targets = targets.to(device)
        masks = masks.to(device).bool()
        preds = model(batch_windows)
        target_np = targets.detach().cpu().numpy()
        pred_np = preds.detach().cpu().numpy()
        mask_np = masks.detach().cpu().numpy()
        for b in range(target_np.shape[0]):
            sample = samples[sample_index] if sample_index < len(samples) else {}
            sample_index += 1
            gap = np.asarray(sample.get("gap_sec", []))
            lap = np.asarray(sample.get("lap_number", []))
            sector = np.asarray(sample.get("mini_sector", []))
            driver_names = sample.get("driver_names")
            for node in np.flatnonzero(mask_np[b]):
                truth, pred = float(target_np[b, node]), float(pred_np[b, node])
                all_targets.append(truth)
                all_preds.append(pred)
                records.append({
                    "RaceId": sample.get("race_id"),
                    "Driver": (driver_names[node] if driver_names and node < len(driver_names)
                               else inverse_drivers.get(int(node))),
                    "NodeIndex": int(node),
                    "TimeSec": float(sample.get("target_time", sample.get("time_sec", np.nan))),
                    "LapNumber": float(lap[node]) if len(lap) > node and np.isfinite(lap[node]) else None,
                    "MiniSector": float(sector[node]) if len(sector) > node and np.isfinite(sector[node]) else None,
                    "gap_sec": float(gap[node]) if len(gap) > node and np.isfinite(gap[node]) else None,
                    "y_true": truth,
                    "y_pred": pred,
                })
    return np.asarray(all_targets, dtype=float), np.asarray(all_preds, dtype=float), records


def validate_checkpoint_edge_scaler(checkpoint: dict, model: nn.Module, loader) -> None:
    """Require matching train-fitted physical-edge scaling for typed messages."""
    if not bool(getattr(model, "typed_edge_values", False)):
        return
    saved = checkpoint.get("edge_feature_scaler") if isinstance(checkpoint, dict) else None
    active_scaler = getattr(getattr(loader, "dataset", None), "edge_feature_scaler", None)
    if active_scaler is None or not hasattr(active_scaler, "metadata"):
        raise ValueError("Typed-edge evaluation requires the train-fitted physical edge scaler.")
    active = active_scaler.metadata()
    expected_schema = getattr(model, "edge_feature_schema", "physical_v2")
    if expected_schema not in TYPED_EDGE_SCHEMAS:
        expected_schema = "physical_v2"
    if (not isinstance(saved, dict) or saved.get("schema") != expected_schema
            or saved != active or active.get("schema") != expected_schema):
        raise ValueError(
            f"Typed-edge checkpoint is missing compatible {expected_schema} edge-scaler metadata; "
            "rebuild the prepared cohort and checkpoint with matching train-only scaling."
        )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Evaluation requires equal, non-empty target and prediction arrays.")
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    return {
        "mae_sec": mae,
        "r2_score": r2,
        "sample_count": int(len(y_true)),
        "target_mae_pass": bool(mae < 0.15),
        "target_r2_pass": bool(r2 is not None and 0.68 <= r2 <= 0.82),
    }


def validate_clean_air_physics(records: list, max_mae: float = 0.05, clean_gap_sec: float = 3.0) -> dict:
    """Measure absolute predicted delay for held-out labels in clean air (gap > 3 s)."""
    rows = [r for r in records if r["gap_sec"] is not None and r["gap_sec"] > clean_gap_sec]
    score = float(np.mean([abs(r["y_pred"]) for r in rows])) if rows else None
    return {"test": f"Clean Air Calibration (observed gap > {clean_gap_sec:g} s)",
            "expected": f"Mean absolute predicted penalty <= {max_mae:.3f} s",
            "observed_mae": score, "sample_count": len(rows),
            "passed": bool(score is not None and score <= max_mae),
            "notes": "Measured on held-out clean-air observations." if rows else "No held-out clean-air labels; test is not measurable."}


def validate_drs_train_physics(records: list, mini_sectors_per_lap: int = 20,
                               min_lap_loss: float = 0.40, max_lap_loss: float = 0.80,
                               drs_max_gap_sec: float = 0.8) -> dict:
    """Measure full-lap predicted delay for held-out driver laps in a DRS train."""
    rows = [r for r in records if r["gap_sec"] is not None and 0.0 < r["gap_sec"] < drs_max_gap_sec
            and r["Driver"] is not None and r["LapNumber"] is not None
            and r["MiniSector"] is not None]
    by_lap = {}
    for row in rows:
        key = (row["RaceId"], row["Driver"], row["LapNumber"])
        by_lap.setdefault(key, {})[row["MiniSector"]] = row["y_pred"]
    full_lap = [float(sum(sectors.values())) for sectors in by_lap.values()
                if len(sectors) >= mini_sectors_per_lap]
    score = float(np.median(full_lap)) if full_lap else None
    return {"test": f"DRS Train Aerodynamic Wake (observed gap < {drs_max_gap_sec:g} s)",
            "expected": f"{min_lap_loss:.2f} s - {max_lap_loss:.2f} s predicted delay per complete lap",
            "observed_lap_delay": score, "complete_laps": len(full_lap),
            "observed_sector_delay": (score / mini_sectors_per_lap) if score is not None else None,
            "passed": bool(score is not None and min_lap_loss <= score <= max_lap_loss),
            "notes": "Measured on complete held-out driver laps." if full_lap else
                     "No complete held-out DRS-train laps; test is not measurable."}


def supplementary_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float | None = None,
                          train_median: float | None = None) -> dict:
    """Diagnostics reported next to (never instead of) the unchanged gate metrics.

    ``trimmed_r2_p01_p99`` scores rows whose observed target lies inside the
    1st-99th percentile of the scored partition, to show how much of the R² gap
    comes from a few multi-second labels. Residual deciles group rows by
    observed target. Train-only constants are fitted on training labels.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = {"rmse_sec": float(np.sqrt(np.mean(np.square(y_true - y_pred))))}
    low, high = np.quantile(y_true, [0.01, 0.99])
    inside = (y_true >= low) & (y_true <= high)
    result["trimmed_r2_p01_p99"] = (float(r2_score(y_true[inside], y_pred[inside]))
                                    if inside.sum() >= 2 else None)
    result["trimmed_bounds_sec"] = [float(low), float(high)]
    edges = np.quantile(y_true, np.linspace(0, 1, 11))
    deciles = []
    for index in range(10):
        upper = y_true <= edges[index + 1] if index == 9 else y_true < edges[index + 1]
        rows = (y_true >= edges[index]) & upper
        if rows.any():
            deciles.append({"decile": index + 1, "count": int(rows.sum()),
                            "mean_true": float(y_true[rows].mean()), "mean_pred": float(y_pred[rows].mean()),
                            "mae": float(np.mean(np.abs(y_true[rows] - y_pred[rows])))})
    result["residual_by_true_decile"] = deciles
    for name, constant in (("train_mean_constant", train_mean), ("train_median_constant", train_median)):
        if constant is not None and np.isfinite(constant):
            baseline = np.full_like(y_true, float(constant))
            result[name] = {"value": float(constant),
                            "mae_sec": float(np.mean(np.abs(y_true - baseline))),
                            "r2_score": float(r2_score(y_true, baseline)) if len(y_true) >= 2 else None}
    return result


def drs_train_per_sector_estimate(records: list, mini_sectors_per_lap: int = 20,
                                  drs_max_gap_sec: float = 0.8) -> dict:
    """Supplementary DRS-wake estimate: mean predicted delay per close sector x sectors per lap.

    The unchanged gate needs complete laps, which the stride-2 evaluation grid
    almost never supplies. This estimate uses every held-out sector label with
    0 < gap < drs_max_gap_sec. It is reported separately and does not change
    the gate's pass/fail result.
    """
    rows = [r for r in records if r.get("gap_sec") is not None and 0.0 < r["gap_sec"] < drs_max_gap_sec]
    if not rows:
        return {"sector_labels": 0, "estimated_lap_delay": None, "observed_lap_delay": None}
    predicted = float(np.mean([r["y_pred"] for r in rows]))
    observed = float(np.mean([r["y_true"] for r in rows]))
    return {"sector_labels": len(rows), "estimated_lap_delay": predicted * mini_sectors_per_lap,
            "observed_lap_delay": observed * mini_sectors_per_lap,
            "note": "Supplementary; the configured DRS gate is evaluated unchanged elsewhere."}


def run_full_evaluation(model_weights_path: str, test_loader, config: dict, baseline_scores: dict = None) -> dict:
    """Load the best checkpoint, evaluate held-out data, and write a JSON report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(model_weights_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint does not exist: {checkpoint_path}")
    model = create_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    validate_checkpoint_edge_scaler(checkpoint, model, test_loader)

    saved_scaler = checkpoint.get("feature_scaler") if isinstance(checkpoint, dict) else None
    active_scaler = getattr(test_loader.dataset, "feature_scaler", None)
    if not saved_scaler or active_scaler is None:
        raise ValueError("Checkpoint and evaluation loader must both provide the train-fitted feature scaler.")
    if (not np.allclose(active_scaler.mean_, saved_scaler["mean_"]) or
            not np.allclose(active_scaler.scale_, saved_scaler["scale_"])):
        raise ValueError("Evaluation features were not transformed with the checkpoint's train-fitted scaler.")

    y_true, y_pred, records = predict_test_set(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    physics_cfg = config.get("physics_tests", {})
    physics = [
        validate_clean_air_physics(records, float(physics_cfg.get("clean_air_max_mae", 0.05)),
                                   float(config.get("target", {}).get("clean_air_gap_sec", 3.0))),
        validate_drs_train_physics(
            records,
            int(config.get("track", {}).get("mini_sectors", 20)),
            float(physics_cfg.get("drs_train_min_lap_loss", 0.40)),
            float(physics_cfg.get("drs_train_max_lap_loss", 0.80)),
            float(physics_cfg.get("drs_max_gap_sec", 0.8)),
        ),
    ]
    report = {"metrics": metrics, "physics_tests": physics, "baselines": baseline_scores or {},
              "predictions": records,
              "metadata": {"checkpoint": str(checkpoint_path),
                           "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
                           "race_ids": sorted({str(r["RaceId"]) for r in records if r["RaceId"] is not None}),
                           "split": getattr(test_loader.dataset, "split_metadata", {}),
                           "feature_scaler": saved_scaler}}
    out_dir = Path(config.get("session", {}).get("processed_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(config.get("evaluation", {}).get("report_path", out_dir / "evaluation_report.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    logger.info("Saved held-out evaluation report to %s", out_path)
    return report
