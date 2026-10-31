"""Authenticate to FastF1, acquire 2023 Grand Prix races, and train HammerTime.

From the repository root (FastF1 3.9.0 is supplied as a runtime overlay):
    uv run --frozen --with fastf1==3.9.0 python hammertime_100/run_2023_fastf1_season.py

The script reads F1_USERNAME and F1_PASSWORD from the process environment or
the repository-root .env file. It never writes credentials to run artifacts.
Only FastF1 is used for event discovery and race data. Failed/unsupported races
are recorded and skipped; the existing coverage, physics, split, and model
gates are applied unchanged by the HammerTime preparation and training code.
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
import re
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

from dataset import create_dataloaders  # noqa: E402
from evaluate import (  # noqa: E402
    compute_metrics, predict_test_set, validate_clean_air_physics,
    validate_drs_train_physics,
)
from model import create_model  # noqa: E402
from prepare_cohort import prepare_cohort, load_prepared_cohort  # noqa: E402
from target import build_targets  # noqa: E402
from train import train_model  # noqa: E402

LOG = logging.getLogger("hammertime.fastf1_season2023")
ACTIVE_OUTPUT: Path | None = None

# Circuit lengths (metres) for the official 2023 F1 calendar. Values and
# event slugs follow the official F1 event pages; unknown locations are skipped.
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
    "Singapore": ("Singapore", 4940.0, "singapore"),
    "Suzuka": ("Japan", 5807.0, "japan"),
    "Lusail": ("Qatar", 5419.0, "qatar"),
    "Austin": ("United States", 5513.0, "usa"),
    "Mexico City": ("Mexico", 4304.0, "mexico"),
    "São Paulo": ("São Paulo", 4309.0, "brazil"),
    "Las Vegas": ("Las Vegas", 6201.0, "las-vegas"),
    "Yas Marina Circuit": ("Abu Dhabi", 5281.0, "abu-dhabi"),
}


def _dotenv_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    # Support ordinary unquoted KEY=value lines and whitespace comments.
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def load_credentials() -> tuple[str, str]:
    """Load only the two expected variables, without displaying or persisting them."""
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            key, sep, raw = stripped.partition("=")
            if sep and key.strip() in {"F1_USERNAME", "F1_PASSWORD"}:
                os.environ.setdefault(key.strip(), _dotenv_value(raw))
    username, password = os.environ.get("F1_USERNAME", ""), os.environ.get("F1_PASSWORD", "")
    if not username or not password:
        raise RuntimeError("Set F1_USERNAME and F1_PASSWORD in the process environment or repository-root .env.")
    return username, password


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(fastf1, output: Path) -> tuple[list[dict], list[dict]]:
    """Use FastF1's own calendar and retain mapped 2023 Grand Prix events."""
    schedule = fastf1.get_event_schedule(2023, include_testing=False)
    events, skips = [], []
    for _, row in schedule.iterrows():
        name = str(row.get("EventName", "")).strip()
        location = str(row.get("Location", "")).strip()
        if not name or name.casefold().startswith("pre-season"):
            continue
        circuit = CIRCUITS.get(location)
        if not circuit:
            skips.append({"event": name, "location": location, "reason": "No reviewed circuit-length mapping"})
            continue
        gp, length, slug = circuit
        round_number = int(row.get("RoundNumber", 0))
        events.append({
            "year": 2023, "gp": name, "session_type": "R", "source": "fastf1",
            "race_id": f"2023-{gp.replace(' ', '')}-R", "round_number": round_number,
            "circuit_id": location, "lap_length_m": length,
            "official_circuit_url": f"https://www.formula1.com/en/racing/2023/{slug}",
        })
    events.sort(key=lambda item: item["round_number"])
    (output / "discovered_sessions.json").write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    return events, skips


def validate_event(event: dict, cache_root: Path) -> int:
    """Exercise the existing FastF1 ingest, alignment, validity, and target gates."""
    from ingest import load_race
    from preprocess import process_race

    cache = cache_root / event["race_id"]
    session = load_race(2023, event["gp"], "R", str(cache))
    cfg_path = HERE / "configs" / "supplied_fastf1_monza_2023_gated.yaml"
    with cfg_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg["session"].update(event)
    cfg["session"]["cache_dir"] = str(cache)
    cfg["track"]["lap_length_m"] = event["lap_length_m"]
    aligned = process_race(session, cfg)
    labels = build_targets(aligned, cfg)
    if labels.empty:
        raise ValueError("existing target builder produced no labels")
    if int(aligned.get("IsValid", pd.Series(dtype=bool)).fillna(False).sum()) == 0:
        raise ValueError("existing aligned telemetry validity gate rejected all rows")
    return int(len(labels))


def aggregate(frame: pd.DataFrame, group: str) -> list[dict]:
    rows = []
    for key, part in frame.groupby(group, dropna=False):
        y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
        rows.append({group: None if pd.isna(key) else key, **compute_metrics(y, p),
                     "rmse_sec": float(np.sqrt(np.mean(np.square(y - p))))})
    return rows


def run(args: argparse.Namespace) -> Path:
    secrets = load_credentials()
    # These are the documented FastF1 3.9+ password-flow environment names.
    os.environ["F1_USERNAME"], os.environ["F1_PASSWORD"] = secrets
    os.environ["FASTF1_AUTH_METHOD"] = "password"
    import fastf1
    from packaging.version import Version
    if Version(fastf1.__version__) < Version("3.9.0"):
        raise RuntimeError("FastF1 account authentication requires 3.9.0+. Run with: uv run --frozen --with fastf1==3.9.0 python hammertime_100/run_2023_fastf1_season.py")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output.expanduser().resolve() if args.output else ROOT / ".local/research-history" / f"2023-fastf1-season-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT = out
    for name in ("cohort", "training"):
        (out / name).mkdir()
    log_path = out / "run.log"
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("fastf1").setLevel(logging.INFO)
    cache_root = HERE / "data/cache/fastf1-2023-authenticated"
    cache_root.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_root))
    failures: list[dict] = []
    try:
        events, skips = discover(fastf1, out)
        failures.extend(skips)
        good = []
        for event in events:
            try:
                event["validated_target_rows"] = validate_event(event, cache_root)
                good.append(event)
                LOG.info("Accepted authentic FastF1 session %s", event["race_id"])
            except Exception as exc:
                # Store only exception class: upstream messages can contain sensitive request details.
                failures.append({"race_id": event["race_id"], "exception_type": type(exc).__name__})
                LOG.warning("Session %s failed ingestion or existing data gates (%s)",
                            event["race_id"], type(exc).__name__)
        (out / "failed_sessions.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        (out / "usable_sessions.json").write_text(json.dumps(good, indent=2) + "\n", encoding="utf-8")
        if len(good) < 3:
            raise RuntimeError(f"Only {len(good)} authentic, gate-passing races; existing race-held-out training requires at least three.")

        with (HERE / "configs/supplied_fastf1_monza_2023_gated.yaml").open(encoding="utf-8") as stream:
            base = yaml.safe_load(stream)
        base["sessions"] = good
        base["session"].update(good[0])
        base["session"]["race_id"] = good[0]["race_id"]
        base["session"]["cache_dir"] = str(cache_root)
        base["session"]["processed_dir"] = str(out / "cohort/events")
        base["track"]["lap_length_m"] = good[0]["lap_length_m"]
        base["data"]["split_method"] = "race"
        base["data"]["random_seed"] = args.seed
        base["train"]["max_epochs"] = args.epochs
        base["train"]["checkpoint_path"] = str(out / "training/best.pt")
        base["train"]["history_path"] = str(out / "training/training_history.csv")
        base["tuning"]["enabled"] = False
        config_path = out / "season_config.yaml"
        config_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        (out / "source_hashes.json").write_text(json.dumps({
            "script_sha256": sha256(Path(__file__)),
            "base_config_sha256": sha256(HERE / "configs/supplied_fastf1_monza_2023_gated.yaml"),
            "fastf1_version": fastf1.__version__,
        }, indent=2) + "\n", encoding="utf-8")

        prepare_cohort(config_path, out / "cohort/prepared")
        payload, manifest = load_prepared_cohort(out / "cohort/prepared/prepared_cohort.pkl")
        if manifest["provider"] != "fastf1" or len(manifest["sessions"]) < 3:
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
        y, pred, records = predict_test_set(model, test_loader, device)
        frame = pd.DataFrame(records)
        metrics = compute_metrics(y, pred)
        metrics["rmse_sec"] = float(np.sqrt(np.mean(np.square(y - pred))))
        metrics["zero_baseline"] = compute_metrics(y, np.zeros_like(y))
        metrics["physics_tests"] = [
            validate_clean_air_physics(records, float(cfg["physics_tests"].get("clean_air_max_mae", .05)),
                                       float(cfg["target"].get("clean_air_gap_sec", 3.0))),
            validate_drs_train_physics(records, int(cfg["track"].get("mini_sectors", 20)),
                                       float(cfg["physics_tests"].get("drs_train_min_lap_loss", .4)),
                                       float(cfg["physics_tests"].get("drs_train_max_lap_loss", .8)),
                                       float(cfg["physics_tests"].get("drs_max_gap_sec", .8))),
        ]
        (out / "training/test_metrics.json").write_text(json.dumps({
            "metrics": metrics, "best_validation_mae_sec": best_val,
            "checkpoint_epoch": checkpoint.get("epoch"), "split": manifest["split"],
            "provider": "fastf1", "sessions": manifest["sessions"],
            "target_definition": "existing target builder; full-session calibrated",
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        frame.to_csv(out / "training/test_predictions.csv", index=False)
        pd.DataFrame([{key: val for key, val in metrics.items()
                       if isinstance(val, (int, float, bool)) or val is None}]).to_csv(
                           out / "training/test_metrics.csv", index=False)
        for group in ("RaceId", "Driver", "MiniSector"):
            if group in frame:
                pd.DataFrame(aggregate(frame, group)).to_csv(out / f"training/per_{group.lower()}_metrics.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(); ax.plot(hist.epoch, hist.train_loss, label="train loss"); ax.plot(hist.epoch, hist.val_mae, label="validation MAE"); ax.legend(); ax.set_xlabel("Epoch"); fig.savefig(out / "training/loss.png", dpi=140); plt.close(fig)
        fig, ax = plt.subplots(); ax.scatter(y, pred, s=4, alpha=.35); lim=[min(y.min(),pred.min()), max(y.max(),pred.max())]; ax.plot(lim,lim,"k--"); ax.set_xlabel("Observed delay (s)"); ax.set_ylabel("Predicted delay (s)"); fig.savefig(out / "training/predicted_vs_actual.png", dpi=140); plt.close(fig)
        fig, ax = plt.subplots(); ax.hist(pred-y, bins=50); ax.set_xlabel("Prediction residual (s)"); fig.savefig(out / "training/residuals.png", dpi=140); plt.close(fig)
        return out
    except Exception as exc:
        # Keep secrets out of durable diagnostics; exception type is sufficient here.
        (out / "run_failure.json").write_text(json.dumps({"exception_type": type(exc).__name__,
            "stage": "acquisition_or_training"}, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"2023 FastF1 season run failed ({type(exc).__name__}); see {out} for sanitized diagnostics.") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New run directory; default .local/research-history/2023-fastf1-season-UTC.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error("--epochs must be positive and --seed non-negative")
    try:
        print(run(args))
    except Exception as exc:
        logging.error("2023 FastF1 season run failed (%s)", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
