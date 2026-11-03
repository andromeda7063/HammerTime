"""Export the saved FastF1 2023 evaluation to the static research dashboard.

Run from the repository root with:
    uv run --frozen python hammertime_100/frontend/data/export_season_model.py
Reads only the archived run. It does not train or run inference.
"""
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "769a76aab164b514cc0a"
RUN_DIR = ROOT / ".local/research-history/2023-fastf1-season/training" / RUN_ID
OUTPUT_PATH = Path(__file__).resolve().parent / "season_model.json"
METRIC_ABS_TOLERANCE = 1e-12
ERROR_BIN_UPPER_BOUNDS = (0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, None)
ERROR_BIN_LABELS = (
    "≤ 0.05", "> 0.05–0.10", "> 0.10–0.20", "> 0.20–0.50",
    "> 0.50–1.00", "> 1.00–2.00", "> 2.00–5.00", "> 5.00",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_breakdown(filename: str, key_name: str, key_field: str) -> list[dict]:
    path = RUN_DIR / filename
    rows = []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            parsed = {key_name: row[key_field]}
            for key in ("mae_sec", "rmse_sec", "r2_score", "sample_count", "target_mae_pass", "target_r2_pass"):
                if key not in row:
                    continue
                value = row[key]
                if key.startswith("target_"):
                    parsed[key] = value.casefold() == "true"
                elif value:
                    parsed[key] = int(value) if key == "sample_count" else float(value)
            rows.append(parsed)
    return rows


def calculate_metrics(actual: list[float], predicted: list[float]) -> dict:
    """Recompute standard regression metrics from the saved prediction rows."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Cannot score an empty or mismatched prediction group.")
    count = len(actual)
    residuals = [y_hat - y for y, y_hat in zip(actual, predicted)]
    squared_error_sum = math.fsum(error * error for error in residuals)
    target_mean = math.fsum(actual) / count
    total_sum_squares = math.fsum((y - target_mean) ** 2 for y in actual)
    r2 = (1.0 if squared_error_sum == 0.0 else 0.0) if total_sum_squares == 0.0 else 1.0 - squared_error_sum / total_sum_squares
    return {
        "sample_count": count,
        "mae_sec": math.fsum(abs(error) for error in residuals) / count,
        "rmse_sec": math.sqrt(squared_error_sum / count),
        "r2_score": r2,
    }


def require_close(label: str, observed: float, expected: float) -> None:
    if not math.isfinite(observed) or not math.isfinite(expected):
        raise ValueError(f"{label} is not finite: observed={observed}, expected={expected}.")
    if abs(observed - expected) > METRIC_ABS_TOLERANCE:
        raise ValueError(
            f"{label} differs from archived predictions: observed={observed:.17g}, "
            f"expected={expected:.17g}, tolerance={METRIC_ABS_TOLERANCE:g}."
        )


def verify_metric_record(label: str, observed: dict, expected: dict) -> None:
    actual = calculate_metrics(observed["actual"], observed["predicted"])
    if actual["sample_count"] != expected["sample_count"]:
        raise ValueError(
            f"{label} target count differs from archived predictions: "
            f"{actual['sample_count']} != {expected['sample_count']}."
        )
    for metric in ("mae_sec", "rmse_sec", "r2_score"):
        if metric in expected:
            require_close(f"{label} {metric}", actual[metric], expected[metric])


def verify_breakdown_metrics(label: str, rows: list[dict], grouped: dict[str, dict]) -> None:
    expected_keys = {row[label]: row for row in rows}
    if set(expected_keys) != set(grouped):
        raise ValueError(f"{label} breakdown keys do not match the prediction CSV groups.")
    for key, values in grouped.items():
        verify_metric_record(f"{label} {key}", values, expected_keys[key])


def absolute_error_bin(error_sec: float) -> int:
    for index, upper_bound in enumerate(ERROR_BIN_UPPER_BOUNDS):
        if upper_bound is None or error_sec <= upper_bound:
            return index
    raise AssertionError("The open-ended final error bin must cover every finite error.")


def main() -> None:
    metrics_path = RUN_DIR / "test_metrics.json"
    predictions_path = RUN_DIR / "test_predictions.csv"
    checkpoint_path = RUN_DIR / "best.pt"
    required = [metrics_path, predictions_path, checkpoint_path,
                RUN_DIR / "per_raceid_metrics.csv", RUN_DIR / "per_driver_metrics.csv",
                RUN_DIR / "per_minisector_metrics.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing archived model-run artifacts: " + ", ".join(missing))

    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = raw["metrics"]
    prediction_sample = []
    actual_values = []
    predicted_values = []
    grouped_predictions = {
        "race_id": {}, "driver": {}, "mini_sector": {},
    }
    error_bin_counts = {
        "stgnn": [0] * len(ERROR_BIN_LABELS),
        "zero_predictor": [0] * len(ERROR_BIN_LABELS),
    }
    absolute_error_ranges = {
        "stgnn": {"min_sec": math.inf, "max_sec": 0.0},
        "zero_predictor": {"min_sec": math.inf, "max_sec": 0.0},
    }
    prediction_count = 0
    with predictions_path.open(newline="", encoding="utf-8") as source:
        for prediction_count, row in enumerate(csv.DictReader(source), start=1):
            actual = float(row["y_true"])
            predicted = float(row["y_pred"])
            if not math.isfinite(actual) or not math.isfinite(predicted):
                raise ValueError(f"Non-finite target/prediction at CSV row {prediction_count + 1}.")
            actual_values.append(actual)
            predicted_values.append(predicted)
            for series, error in (("stgnn", abs(predicted - actual)),
                                  ("zero_predictor", abs(actual))):
                error_bin_counts[series][absolute_error_bin(error)] += 1
                absolute_error_ranges[series]["min_sec"] = min(absolute_error_ranges[series]["min_sec"], error)
                absolute_error_ranges[series]["max_sec"] = max(absolute_error_ranges[series]["max_sec"], error)
            for key, value in (("race_id", row["RaceId"]),
                               ("driver", row["Driver"]),
                               ("mini_sector", row["MiniSector"])):
                group = grouped_predictions[key].setdefault(value, {"actual": [], "predicted": []})
                group["actual"].append(actual)
                group["predicted"].append(predicted)
            # Deterministic systematic sample, retaining fields for filtering.
            if (prediction_count - 1) % 38 == 0:
                prediction_sample.append({
                    "race_id": row["RaceId"],
                    "driver": row["Driver"],
                    "lap": int(float(row["LapNumber"])),
                    "time_sec": float(row["TimeSec"]),
                    "mini_sector": int(float(row["MiniSector"])),
                    "actual": actual,
                    "predicted": predicted,
                })
    if prediction_count != metrics["sample_count"]:
        raise ValueError(
            f"Prediction CSV has {prediction_count} rows but saved metrics report "
            f"{metrics['sample_count']} target predictions."
        )
    if any(sum(counts) != prediction_count for counts in error_bin_counts.values()):
        raise ValueError("Absolute-error histogram bins do not cover every held-out prediction.")
    verify_metric_record(
        "full test", {"actual": actual_values, "predicted": predicted_values}, metrics,
    )
    baseline = metrics["zero_baseline"]
    verify_metric_record(
        "zero baseline", {"actual": actual_values, "predicted": [0.0] * prediction_count}, baseline,
    )

    split = raw["split"]
    race_metrics = read_breakdown("per_raceid_metrics.csv", "race_id", "RaceId")
    driver_metrics = read_breakdown("per_driver_metrics.csv", "driver", "Driver")
    mini_sector_metrics = read_breakdown("per_minisector_metrics.csv", "mini_sector", "MiniSector")
    for label, rows in (("race_id", race_metrics), ("driver", driver_metrics),
                        ("mini_sector", mini_sector_metrics)):
        verify_breakdown_metrics(label, rows, grouped_predictions[label])
    for name, rows in (("race", race_metrics), ("driver", driver_metrics), ("mini-sector", mini_sector_metrics)):
        covered = sum(row["sample_count"] for row in rows)
        if covered != metrics["sample_count"]:
            raise ValueError(f"{name} breakdown covers {covered} targets, expected {metrics['sample_count']}.")
    race_ids = [row["race_id"] for row in race_metrics]
    if sorted(race_ids) != sorted(split["test"]["race_ids"]):
        raise ValueError("Per-race metric rows do not match the frozen held-out race IDs.")
    resume_manifest = json.loads((RUN_DIR / "resume_manifest.json").read_text(encoding="utf-8"))
    result = {
        "run_id": RUN_ID,
        "source": "FastF1 2023 full-season race-held-out evaluation",
        "target": "Actual mini-sector duration minus full-session-calibrated baseline; clipped to [-2, 10] seconds",
        "model": "HammerTime ST-GNN · saved model evaluation",
        "metrics": {
            "mae_sec": metrics["mae_sec"], "rmse_sec": metrics["rmse_sec"],
            "r2_score": metrics["r2_score"], "sample_count": metrics["sample_count"],
            "target_mae_pass": metrics["target_mae_pass"],
            "target_r2_pass": metrics["target_r2_pass"],
            "zero_baseline": metrics["zero_baseline"],
            "physics_tests": metrics["physics_tests"],
        },
        "split": {
            "method": split["method"],
            "train_window_count": split["train"]["sample_count"],
            "validation_window_count": split["validation"]["sample_count"],
            "test_window_count": split["test"]["sample_count"],
            "test_races": split["test"]["race_ids"],
        },
        "provenance": {
            "fingerprint": resume_manifest["fingerprint"],
            "config_sha256": resume_manifest["config_sha256"],
            "prepared_artifact_sha256": resume_manifest["prepared_artifact_sha256"],
            "implementation_sha256": resume_manifest["implementation_sha256"],
            "source_artifact_sha256": {
                path.name: sha256(path) for path in required
            },
        },
        "checkpoint": {
            "artifact": f".local/research-history/2023-fastf1-season/training/{RUN_ID}/best.pt",
            "sha256": sha256(checkpoint_path),
            "selected_epoch": raw["checkpoint_epoch"],
            "best_validation_mae_sec": raw["best_validation_mae_sec"],
        },
        "race_metrics": race_metrics,
        "driver_metrics": driver_metrics,
        "mini_sector_metrics": mini_sector_metrics,
        "prediction_sample": {
            "method": "Every 38th row of archived test_predictions.csv in file order",
            "source_rows": prediction_count,
            "points": prediction_sample,
        },
        "error_distribution": {
            "population_count": prediction_count,
            "population": "Every valid held-out node-target prediction in test_predictions.csv",
            "unit": "absolute residual error in seconds",
            "method": "Counts use inclusive upper bounds; the first bin includes zero and the last bin is open-ended.",
            "observed_range_sec": absolute_error_ranges,
            "bins": [
                {"label": label, "stgnn_count": error_bin_counts["stgnn"][index],
                 "zero_predictor_count": error_bin_counts["zero_predictor"][index]}
                for index, label in enumerate(ERROR_BIN_LABELS)
            ],
        },
        "count_note": (
            "Target count is valid node-target predictions. Split counts are graph windows: "
            "37,956 predictions and 25,129 test windows are different units."
        ),
    }
    temporary_output = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=OUTPUT_PATH.parent,
            prefix=f".{OUTPUT_PATH.name}.", suffix=".tmp", delete=False,
        ) as target:
            temporary_output = Path(target.name)
            json.dump(result, target, indent=2, ensure_ascii=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_output, OUTPUT_PATH)
    except Exception:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        raise
    print(f"Exported {OUTPUT_PATH} from run {RUN_ID}: {prediction_count} predictions, {len(prediction_sample)} chart points")


if __name__ == "__main__":
    main()
