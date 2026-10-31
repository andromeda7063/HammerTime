"""Acquire and train one OpenF1-only, race-held-out 2023 season cohort.

Run from the repository root with:
    uv run --frozen python hammertime_100/run_2023_season.py

Circuit lengths are the official 2023 Formula 1 circuit distances published on
each event's formula1.com race page. OpenF1 session location/session_key and
source timestamps remain the event identity checks; the mapping only supplies
the physics geometry required by HammerTime.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

from acquire_openf1 import acquire  # noqa: E402
from dataset import create_dataloaders  # noqa: E402
from evaluate import compute_metrics, predict_test_set  # noqa: E402
from model import create_model  # noqa: E402
from openf1_client import _OpenF1Client, load_openf1_tables  # noqa: E402
from openf1_adapter import normalize_openf1_tables  # noqa: E402
from prepare_cohort import prepare_cohort, load_prepared_cohort  # noqa: E402
from target import build_targets  # noqa: E402
from train import train_model  # noqa: E402
from evaluate import validate_clean_air_physics, validate_drs_train_physics  # noqa: E402

LOG = logging.getLogger("hammertime.season2023")
DISCOVERY_SKIPS: list[dict] = []
ACTIVE_OUTPUT: Path | None = None

# Round locations follow OpenF1's historical sessions endpoint. Lengths cite
# the matching official Formula 1 2023 event page (miles converted to metres).
# https://www.formula1.com/en/racing/2023
CIRCUITS = {
    "Sakhir": ("Bahrain", 5412.0, "bahrain"),
    "Jeddah": ("Saudi Arabia", 6174.0, "saudi-arabia"),
    "Melbourne": ("Australia", 5278.0, "australia"),
    "Baku": ("Azerbaijan", 6003.0, "azerbaijan"),
    "Miami": ("Miami", 5412.0, "miami"),
    "Monaco": ("Monaco", 3337.0, "monaco"),
    "Barcelona": ("Spain", 4657.0, "spain"),
    "Montreal": ("Canada", 4361.0, "canada"),
    "Spielberg": ("Austria", 4318.0, "austria"),
    "Silverstone": ("Great Britain", 5891.0, "great-britain"),
    "Budapest": ("Hungary", 4381.0, "hungary"),
    "Spa-Francorchamps": ("Belgium", 7004.0, "belgium"),
    "Zandvoort": ("Netherlands", 4259.0, "netherlands"),
    "Monza": ("Italy", 5793.0, "italy"),
    "Singapore": ("Singapore", 4940.0, "singapore"),  # https://www.formula1.com/en/racing/2023/singapore
    "Suzuka": ("Japan", 5807.0, "japan"),
    "Lusail": ("Qatar", 5419.0, "qatar"),
    "Austin": ("United States", 5513.0, "usa"),
    "Mexico City": ("Mexico", 4304.0, "mexico"),
    "São Paulo": ("São Paulo", 4309.0, "brazil"),
    "Las Vegas": ("Las Vegas", 6201.0, "las-vegas"),
    "Yas Marina Circuit": ("Abu Dhabi", 5281.0, "abu-dhabi"),
}
CIRCUIT_ALIASES = {
    "Marina Bay": "Singapore", "MarinaBay": "Singapore",
    "Losail": "Lusail", "Budapest": "Budapest",
    "Sao Paulo": "São Paulo", "SaoPaulo": "São Paulo",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(config: dict, cache: Path) -> list[dict]:
    """Fetch the provider's 2023 calendar, retaining official Race sessions only."""
    client = _OpenF1Client(config["openf1"], cache / "calendar", get=__import__("requests").get,
                           sleeper=__import__("time").sleep, clock=__import__("time").monotonic)
    rows = client.fetch("sessions", {"year": 2023, "session_name": "Race"})
    seen_sessions: set[str] = set()
    seen_meetings: set[str] = set()
    result = []
    races_seen = 0
    for row in sorted(rows, key=lambda item: (item.get("date_start", ""), str(item.get("session_key", "")))):
        key = str(row.get("session_key", ""))
        if (str(row.get("year")) != "2023" or row.get("session_name") != "Race"
                or row.get("session_type") != "Race"):
            continue
        if row.get("is_cancelled") in (True, 1, "1", "true", "True"):
            DISCOVERY_SKIPS.append({"session_key": key, "meeting_key": row.get("meeting_key"),
                                    "reason": "OpenF1 marks race session cancelled"})
            continue
        meeting = str(row.get("meeting_key", ""))
        if not key or key in seen_sessions or (meeting and meeting in seen_meetings):
            continue
        seen_sessions.add(key)
        if meeting:
            seen_meetings.add(meeting)
        races_seen += 1
        raw_location = str(row.get("location", ""))
        location = CIRCUIT_ALIASES.get(raw_location, raw_location)
        if location not in CIRCUITS:
            LOG.warning("Skipping unreviewed circuit location %r, session_key=%s", location, key)
            DISCOVERY_SKIPS.append({"session_key": key, "location": location,
                                    "reason": "No reviewed 2023 circuit-length mapping"})
            continue
        gp, length, slug = CIRCUITS[location]
        result.append({"year": 2023, "gp": raw_location, "session_type": "R", "session_key": int(key),
                       "race_id": f"2023-{gp.replace(' ', '')}-R", "source": "openf1",
                       "circuit_id": location, "lap_length_m": length,
                       "official_circuit_url": f"https://www.formula1.com/en/racing/2023/{slug}"})
    if races_seen != 22:
        LOG.warning("OpenF1 calendar returned %d non-cancelled 2023 Race meetings; expected 22. Continuing with %d mapped races.",
                    races_seen, len(result))
    return result


def session_config(base: dict, spec: dict, cache_root: Path, processed: Path) -> dict:
    cfg = copy.deepcopy(base)
    cfg["session"].update(spec)
    cfg["session"]["race_id"] = spec["race_id"]
    cfg["session"]["cache_dir"] = str(cache_root / spec["race_id"])
    cfg["session"]["processed_dir"] = str(processed / spec["race_id"])
    cfg["track"]["lap_length_m"] = spec["lap_length_m"]
    return cfg


def validate_event(base: dict, spec: dict, cache_root: Path, out: Path) -> tuple[bool, str | None, int]:
    cfg = session_config(base, spec, cache_root, out)
    bundle = load_openf1_tables(cfg)
    aligned = normalize_openf1_tables(bundle, cfg)
    labels = build_targets(aligned, cfg)
    if labels.empty:
        raise ValueError("existing target builder produced no labels")
    if int(aligned.get("IsValid", pd.Series(dtype=bool)).fillna(False).sum()) == 0:
        raise ValueError("existing aligned telemetry validity gate rejected all rows")
    return True, None, int(len(labels))


def prediction_metrics(records: list[dict]) -> tuple[dict, list[dict]]:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("held-out evaluation produced no predictions")
    y, p = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
    metrics = compute_metrics(y, p)
    metrics["rmse_sec"] = float(np.sqrt(np.mean(np.square(y - p))))
    metrics["zero_baseline"] = compute_metrics(y, np.zeros_like(y)) | {
        "rmse_sec": float(np.sqrt(np.mean(np.square(y))))}
    return metrics, frame.to_dict(orient="records")


def aggregate(frame: pd.DataFrame, group: str) -> list[dict]:
    rows = []
    for key, part in frame.groupby(group, dropna=False):
        y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
        rows.append({group: None if pd.isna(key) else key, **compute_metrics(y, p),
                     "rmse_sec": float(np.sqrt(np.mean(np.square(y - p))))})
    return rows


def run(args: argparse.Namespace) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output.expanduser().resolve() if args.output else ROOT / ".local/research-history" / f"2023-season-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT = out
    print(f"Run artifacts: {out}", flush=True)
    for name in ("events", "cohort", "training"):
        (out / name).mkdir()
    log_path = out / "run.log"
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                        format="%(asctime)s %(levelname)s %(message)s")
    with (HERE / "configs/openf1_2023_race_holdout.yaml").open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    base["openf1"]["chunk_seconds"] = 7200
    base["train"]["max_epochs"] = args.epochs
    base["data"]["random_seed"] = args.seed
    base["graph"]["node_feature_schema"] = "legacy_v1"
    base["model"]["temporal_mode"] = "gated_dilated"
    cache_root = (HERE / base["session"]["cache_dir"]).resolve()
    DISCOVERY_SKIPS.clear()
    try:
        events = discover(base, cache_root)
    except Exception as exc:
        (out / "run_failure.json").write_text(json.dumps({"stage": "session_discovery",
            "exception": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n", encoding="utf-8")
        raise
    (out / "discovered_sessions.json").write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    failures: list[dict] = list(DISCOVERY_SKIPS)
    good = []
    for event in events:
        try:
            # Isolated event normalization checks let later sessions continue after a bad event.
            ok, _, nlabels = validate_event(base, event, cache_root, out / "events")
            acquire_config = session_config(base, event, cache_root, out / "events")
            cfg_file = out / "events" / f"{event['session_key']}.yaml"
            cfg_file.write_text(yaml.safe_dump(acquire_config, sort_keys=False), encoding="utf-8")
            acquire(cfg_file)
            event["validated_target_rows"] = nlabels
            good.append(event)
        except Exception as exc:  # per-event failure is retained and the calendar continues
            failures.append({"session_key": event["session_key"], "race_id": event["race_id"],
                             "exception": f"{type(exc).__name__}: {exc}"})
            LOG.exception("Event %s failed; continuing", event["race_id"])
    (out / "failed_sessions.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    if len(good) < 3:
        (out / "run_failure.json").write_text(json.dumps({"stage": "usable_session_threshold",
            "exception": f"Only {len(good)} usable races; at least three are required.",
            "failures": failures}, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Only {len(good)} usable 2023 races; at least three are required for race-heldout training.")
    drivers = set()
    for event in good:
        cfg = session_config(base, event, cache_root, out / "events")
        bundle = load_openf1_tables(cfg)
        drivers.update(bundle["tables"]["drivers"]["driver_number"].dropna().astype(str))
    base["graph"]["num_nodes"] = len(drivers)
    if not drivers:
        raise RuntimeError("No source-authenticated driver union was found.")
    base["sessions"] = good
    base["session"].update(good[0])
    base["session"]["cache_dir"] = str(cache_root)
    base["session"]["processed_dir"] = str(out / "events")
    base["track"]["lap_length_m"] = good[0]["lap_length_m"]
    base["train"]["checkpoint_path"] = str(out / "training/best.pt")
    base["train"]["history_path"] = str(out / "training/training_history.csv")
    base["tuning"]["enabled"] = False
    config_path = out / "season_config.yaml"
    config_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    (out / "source_hashes.json").write_text(json.dumps({"script_sha256": sha256(Path(__file__)),
        "base_config_sha256": sha256(HERE / "configs/openf1_2023_race_holdout.yaml")}, indent=2) + "\n", encoding="utf-8")
    try:
        prepare_cohort(config_path, out / "cohort/prepared")
        payload, manifest = load_prepared_cohort(out / "cohort/prepared/prepared_cohort.pkl")
        if manifest["provider"] != "openf1" or len(manifest["sessions"]) < 3:
            raise ValueError("Prepared cohort provider/session-count invariant failed")
        train_loader, val_loader, test_loader, scaler = create_dataloaders(
            copy.deepcopy(payload["snapshots"]), payload["labels"], payload["driver_to_idx"], payload["config"])
        cfg = copy.deepcopy(payload["config"])
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = create_model(cfg).to(device)
        best_val, history = train_model(model, train_loader, val_loader, cfg, device)
        hist = pd.DataFrame(history)
        hist.to_csv(out / "training/training_history.csv", index=False)
        checkpoint = torch.load(out / "training/best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        saved = checkpoint.get("feature_scaler") or {}
        if not np.allclose(scaler.mean_, saved.get("mean_", [])) or not np.allclose(scaler.scale_, saved.get("scale_", [])):
            raise ValueError("Best checkpoint scaler does not match train-only prepared-cohort scaler")
        # The held-out race is opened only once, after validation checkpoint selection is complete.
        y, pred, records = predict_test_set(model, test_loader, device)
        metrics, records = prediction_metrics(records)
        frame = pd.DataFrame(records)
        train_targets = np.concatenate([
            np.asarray(sample["target"])[np.asarray(sample["mask"], dtype=bool)]
            for sample in train_loader.dataset.samples
            if np.asarray(sample["mask"], dtype=bool).any()
        ])
        mean_prediction = float(np.mean(train_targets))
        mean_baseline = compute_metrics(y, np.full_like(y, mean_prediction))
        mean_baseline["rmse_sec"] = float(np.sqrt(np.mean(np.square(y - mean_prediction))))
        metrics["train_mean_baseline"] = mean_baseline
        metrics["physics_tests"] = [
            validate_clean_air_physics(records, float(cfg["physics_tests"].get("clean_air_max_mae", .05)),
                                       float(cfg["target"].get("clean_air_gap_sec", 3.0))),
            validate_drs_train_physics(records, int(cfg["track"].get("mini_sectors", 20)),
                                       float(cfg["physics_tests"].get("drs_train_min_lap_loss", .4)),
                                       float(cfg["physics_tests"].get("drs_train_max_lap_loss", .8)),
                                       float(cfg["physics_tests"].get("drs_max_gap_sec", .8))),
        ]
        (out / "training/test_metrics.json").write_text(json.dumps({"metrics": metrics, "best_validation_mae_sec": best_val,
            "checkpoint_epoch": checkpoint.get("epoch"), "split": manifest["split"], "provider": "openf1",
            "session_identity": manifest["session_identity"], "target_definition": "existing target builder; full-session calibrated"},
            indent=2, allow_nan=False) + "\n", encoding="utf-8")
        frame.to_csv(out / "training/test_predictions.csv", index=False)
        flat_metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float, bool)) or value is None}
        pd.DataFrame([flat_metrics]).to_csv(out / "training/test_metrics.csv", index=False)
        pd.DataFrame(aggregate(frame, "RaceId")).to_csv(out / "training/per_race_metrics.csv", index=False)
        pd.DataFrame(aggregate(frame, "Driver")).to_csv(out / "training/per_driver_metrics.csv", index=False)
        if "MiniSector" in frame:
            pd.DataFrame(aggregate(frame, "MiniSector")).to_csv(out / "training/per_sector_metrics.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(); ax.plot(hist.epoch, hist.train_loss, label="train loss"); ax.plot(hist.epoch, hist.val_mae, label="validation MAE"); ax.legend(); ax.set_xlabel("Epoch"); fig.savefig(out / "training/loss.png", dpi=140); plt.close(fig)
        fig, ax = plt.subplots(); ax.scatter(y, pred, s=4, alpha=.35); lim=[min(y.min(),pred.min()), max(y.max(),pred.max())]; ax.plot(lim,lim,"k--"); ax.set_xlabel("Observed delay (s)"); ax.set_ylabel("Predicted delay (s)"); fig.savefig(out / "training/predicted_vs_actual.png", dpi=140); plt.close(fig)
        fig, ax = plt.subplots(); ax.hist(pred-y, bins=50); ax.set_xlabel("Prediction residual (s)"); fig.savefig(out / "training/residuals.png", dpi=140); plt.close(fig)
        return out
    except Exception as exc:
        (out / "run_failure.json").write_text(json.dumps({"exception": f"{type(exc).__name__}: {exc}",
            "stage": "cohort_or_training", "events_kept": [e["race_id"] for e in good], "failures": failures},
            indent=2) + "\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New run directory; default .local/research-history/2023-season-UTC.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error("--epochs must be positive and --seed non-negative")
    try:
        print(run(args))
    except Exception as exc:
        logging.exception("2023 season run failed: %s", exc)
        if ACTIVE_OUTPUT is not None and not (ACTIVE_OUTPUT / "run_failure.json").exists():
            (ACTIVE_OUTPUT / "run_failure.json").write_text(json.dumps({
                "stage": "run", "exception": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n",
                encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
