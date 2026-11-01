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
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import logging
import os
import pickle
import shutil
from pathlib import Path
import re
import sys
import threading
import uuid

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
    "Abu Dhabi": ("Abu Dhabi", 5281.0, "abu-dhabi"),
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


def canonical_hash(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False, default=str) + "\n").encode())


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "session"


def discover(fastf1, output: Path) -> tuple[list[dict], list[dict]]:
    """Reuse the calendar cache when compatible; otherwise query FastF1 once."""
    cache_path = output / "discovered_sessions.json"
    fingerprint = canonical_hash({"year": 2023, "circuits": CIRCUITS, "fastf1": fastf1.__version__})
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint and cached.get("events"):
                return cached["events"], cached.get("skips", [])
        except (OSError, ValueError, TypeError):
            pass
    schedule = fastf1.get_event_schedule(2023, include_testing=False)
    events, skips = [], []
    for _, row in schedule.iterrows():
        name, location = str(row.get("EventName", "")).strip(), str(row.get("Location", "")).strip()
        if not name or name.casefold().startswith("pre-season"):
            continue
        circuit = CIRCUITS.get(location)
        if not circuit:
            skips.append({"event": name, "location": location, "reason": "No reviewed circuit-length mapping"})
            continue
        gp, length, slug = circuit
        events.append({"year": 2023, "gp": name, "session_type": "R", "source": "fastf1",
                       "race_id": f"2023-{gp.replace(' ', '')}-R",
                       "round_number": int(row.get("RoundNumber", 0)), "circuit_id": slug,
                       "lap_length_m": length,
                       "official_circuit_url": f"https://www.formula1.com/en/racing/2023/{slug}"})
    events.sort(key=lambda item: item["round_number"])
    atomic_json(cache_path, {"fingerprint": fingerprint, "events": events, "skips": skips})
    return events, skips


def _race_fingerprint(event: dict, config: dict, fastf1_version: str) -> str:
    # Hash only inputs to ingest/normalization/labels. Training and export knobs
    # do not invalidate authentic per-race work.
    modules = [HERE / "src/ingest.py", HERE / "src/preprocess.py", HERE / "src/target.py"]
    return canonical_hash({
        "event": {key: event.get(key) for key in ("year", "gp", "session_type", "race_id", "circuit_id", "lap_length_m")},
        "preprocess": config["preprocess"], "target": config["target"],
        "track": config["track"], "fastf1_version": fastf1_version,
        "implementation": {str(path.name): sha256(path) for path in modules},
    })


def _valid_race_stage(stage: Path, expected_fingerprint: str) -> dict | None:
    marker = stage / "stage_manifest.json"
    if not marker.is_file():
        return None
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("fingerprint") != expected_fingerprint:
            return None
        for name, expected in record["files"].items():
            path = stage / name
            if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def ensure_race_stage(event: dict, cache_root: Path, stage_root: Path,
                     config: dict, fastf1_version: str) -> tuple[dict | None, str | None]:
    """Load/preprocess one race once and commit all reusable files atomically."""
    from ingest import export_raw_tables, load_race
    from preprocess import process_race

    stage_root.mkdir(parents=True, exist_ok=True)
    stage = stage_root / _safe_id(event["race_id"])
    fingerprint = _race_fingerprint(event, config, fastf1_version)
    prior = _valid_race_stage(stage, fingerprint)
    if prior is not None:
        return {**event, **prior["event"]}, None

    # Incomplete staging directories are ignored; this invocation builds its
    # own uniquely named staging directory and never overwrites another run.
    staging = stage_root / f".{stage.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        session = load_race(2023, event["gp"], "R", str(cache_root / event["race_id"]))
        cfg = copy.deepcopy(config)
        cfg["session"].update(event)
        cfg["track"]["lap_length_m"] = float(event["lap_length_m"])
        raw_laps, raw_telemetry = export_raw_tables(session, str(staging / "raw"))
        aligned = process_race(session, cfg)
        labels = build_targets(aligned, cfg)
        if labels.empty:
            raise ValueError("existing target builder produced no labels")
        if int(aligned.get("IsValid", pd.Series(dtype=bool)).fillna(False).sum()) == 0:
            raise ValueError("existing aligned telemetry validity gate rejected all rows")
        aligned.to_parquet(staging / "aligned_telemetry.parquet", index=False)
        labels.to_parquet(staging / "target_labels.parquet", index=False)
        circuit_key = str(session.event.get("CircuitKey", event.get("circuit_id", "")))
        final_event = {**event, "circuit_id": circuit_key, "validated_target_rows": int(len(labels))}
        raw_files = [Path(raw_laps)] + ([Path(raw_telemetry)] if raw_telemetry is not None else [])
        artifact_paths = [staging / "aligned_telemetry.parquet", staging / "target_labels.parquet", *raw_files]
        files = {path.relative_to(staging).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in artifact_paths}
        source_metadata = {"provider": "fastf1", "session": {"circuit_key": circuit_key}}
        record = {"fingerprint": fingerprint, "event": final_event, "files": files,
                  "source_metadata": source_metadata,
                  "coverage": {"aligned_rows": int(len(aligned)), "target_rows": int(len(labels))}}
        atomic_json(staging / "stage_manifest.json", record)
        if stage.exists():
            backup = stage_root / f".{stage.name}.stale-{uuid.uuid4().hex}"
            os.replace(stage, backup)
        os.replace(staging, stage)
        return final_event, None
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return None, type(exc).__name__


def write_normalized_source(stage_root: Path, source_root: Path,
                            config_path: Path, events: list[dict]) -> str:
    """Write an authenticated normalized-source cache manifest, not a cohort manifest."""
    from prepare_cohort import _canonical_json, _sha256_bytes

    records, hashes = [], {}
    for event in events:
        race_id = str(event["race_id"])
        stage = stage_root / _safe_id(race_id)
        stage_meta = json.loads((stage / "stage_manifest.json").read_text(encoding="utf-8"))
        source_files = []
        for name, info in stage_meta["files"].items():
            path = stage / name
            try:
                rel = str(path.resolve().relative_to(source_root.resolve()))
            except ValueError:
                continue
            hashes[rel] = {"bytes": int(info["bytes"]), "sha256": info["sha256"]}
            source_files.append(rel)
        records.append({"race_id": race_id, "provider": "fastf1", "year": int(event["year"]),
                        "gp": str(event["gp"]), "session_type": str(event["session_type"]),
                        "session_key": event.get("session_key"), "cache_root": str((HERE / "data/cache/fastf1-2023-authenticated" / race_id).resolve()),
                        "lap_length_m": float(event["lap_length_m"]), "source_files": source_files,
                        "coverage": stage_meta["coverage"], "source_metadata": stage_meta["source_metadata"]})
    manifest = {"schema_version": 1, "artifact_type": "hammertime-normalized-source-cache",
                "provider": "fastf1", "providers": ["fastf1"],
                "config_path": str(config_path.resolve()), "config_sha256": sha256(config_path),
                "sessions": records, "source_hashes": hashes, "normalized_source_manifest_sha256": None}
    manifest["normalized_source_manifest_sha256"] = _sha256_bytes(_canonical_json(
        {key: value for key, value in manifest.items() if key != "normalized_source_manifest_sha256"}))
    raw = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    atomic_bytes(source_root / "normalized_source_manifest.json", raw)
    atomic_bytes(source_root / "normalized_source_manifest.json.sha256",
                 f"{hashlib.sha256(raw).hexdigest()}  normalized_source_manifest.json\n".encode())
    return hashlib.sha256(raw).hexdigest()


def _cohort_cached(cohort_dir: Path, config: dict, source_hash: str):
    from prepare_cohort import load_prepared_cohort, _canonical_json, _sha256_bytes
    artifact = cohort_dir / "prepared_cohort.pkl"
    if not artifact.is_file():
        return None
    try:
        payload, manifest = load_prepared_cohort(artifact)
        expected_config = _sha256_bytes(_canonical_json(config))
        if (manifest.get("provider") != "fastf1"
                or manifest.get("effective_config_sha256") != expected_config
                or manifest.get("aligned_source_manifest_sha256") != source_hash):
            return None
        return payload, manifest
    except (OSError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
        return None


def _cache_predictions(path: Path, fingerprint: str) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_pickle(path)
        if isinstance(frame, pd.DataFrame) and frame.attrs.get("fingerprint") == fingerprint:
            return frame
    except Exception:
        return None
    return None


def load_or_compute_predictions(path: Path, fingerprint: str, compute) -> pd.DataFrame:
    """Persist inference atomically before any retryable CSV/plot exports."""
    frame = _cache_predictions(path, fingerprint)
    if frame is not None:
        return frame
    frame = compute()
    frame.attrs["fingerprint"] = fingerprint
    _atomic_pickle(path, frame)
    return frame


def _atomic_pickle(path: Path, value) -> None:
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        value.to_pickle(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", prefix=f".{path.name}.",
                                     suffix=".tmp", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_OUTPUT_MANIFEST_LOCK = threading.Lock()


def _output_manifest(directory: Path) -> dict:
    try:
        return json.loads((directory / "output_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _output_is_current(path: Path, fingerprint: str) -> bool:
    entry = _output_manifest(path.parent).get("artifacts", {}).get(path.name, {})
    try:
        return (entry.get("fingerprint") == fingerprint and path.is_file()
                and entry.get("sha256") == sha256(path))
    except OSError:
        return False


def _record_output(path: Path, fingerprint: str) -> None:
    with _OUTPUT_MANIFEST_LOCK:
        manifest_path = path.parent / "output_manifest.json"
        manifest = _output_manifest(path.parent)
        artifacts = manifest.setdefault("artifacts", {})
        artifacts[path.name] = {"fingerprint": fingerprint, "sha256": sha256(path)}
        atomic_json(manifest_path, manifest)


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
    out = args.output.expanduser().resolve() if args.output else ROOT / ".local/research-history/2023-fastf1-season"
    out.mkdir(parents=True, exist_ok=True)
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT = out
    for name in ("race_cache/sessions", "cohorts", "training"):
        (out / name).mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("fastf1").setLevel(logging.INFO)
    cache_root = HERE / "data/cache/fastf1-2023-authenticated"
    cache_root.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_root))
    try:
        template_path = HERE / "configs/supplied_fastf1_monza_2023_gated.yaml"
        with template_path.open(encoding="utf-8") as stream:
            base = yaml.safe_load(stream)
        base["data"]["split_method"] = "race"
        base["data"]["random_seed"] = args.seed
        base["train"]["max_epochs"] = args.epochs
        base["tuning"]["enabled"] = False
        events, skips = discover(fastf1, out)
        stages_root = out / "race_cache/sessions"
        fingerprints = {event["race_id"]: _race_fingerprint(event, base, fastf1.__version__) for event in events}
        # Only validation/hashing is parallel. FastF1 uses global auth/cache state,
        # so incomplete races are always downloaded and normalized serially below.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            stage_results = list(pool.map(
                lambda event: _valid_race_stage(stages_root / _safe_id(event["race_id"]),
                                                fingerprints[event["race_id"]]), events))
        failures: list[dict] = list(skips)
        good = []
        for event, prior in zip(events, stage_results):
            if prior is not None:
                good.append({**event, **prior["event"]})
                continue
            result, failure_type = ensure_race_stage(event, cache_root, stages_root, base, fastf1.__version__)
            if result is None:
                failures.append({"race_id": event["race_id"], "exception_type": failure_type})
                LOG.warning("Session %s failed ingestion or existing data gates (%s)",
                            event["race_id"], failure_type)
            else:
                good.append(result)
                LOG.info("Accepted authentic FastF1 session %s", event["race_id"])
        atomic_json(out / "failed_sessions.json", failures)
        atomic_json(out / "usable_sessions.json", good)
        if len(good) < 3:
            raise RuntimeError(f"Only {len(good)} authentic, gate-passing races; existing race-held-out training requires at least three.")

        base["sessions"] = good
        base["session"].update(good[0])
        base["session"]["race_id"] = good[0]["race_id"]
        base["session"]["cache_dir"] = str(cache_root)
        base["session"]["processed_dir"] = str(out / "race_cache/processed")
        base["track"]["lap_length_m"] = good[0]["lap_length_m"]
        base["train"]["checkpoint_path"] = str(out / "training" / "pending" / "best.pt")
        base["train"]["history_path"] = str(out / "training" / "pending" / "training_history.csv")
        config_path = out / "season_config.yaml"
        atomic_bytes(config_path, yaml.safe_dump(base, sort_keys=False).encode())
        source_root = out / "race_cache"
        source_hash = write_normalized_source(stages_root, source_root, config_path, good)
        cohort_fingerprint = canonical_hash({"source_manifest_sha256": source_hash,
                                             "config": base,
                                             "prepare_code": {name: sha256(HERE / "src" / name) for name in (
                                                 "prepare_cohort.py", "run_pipeline.py", "graph.py",
                                                 "context_features.py", "dataset.py", "preprocess.py", "target.py")}})
        cohort_prefix = out / "cohorts" / cohort_fingerprint[:20]
        candidates = [cohort_prefix, *sorted(out / "cohorts".glob(f"{cohort_fingerprint[:20]}-rebuild*"))]
        cached = None
        cohort_dir = cohort_prefix
        for candidate in candidates:
            cached = _cohort_cached(candidate, base, source_hash)
            if cached is not None:
                cohort_dir = candidate
                break
        if cached is None:
            if cohort_prefix.exists():
                # Keep a corrupt/incompatible artifact for investigation and
                # build a new immutable output directory.
                suffix = 1
                while (out / "cohorts" / f"{cohort_fingerprint[:20]}-rebuild{suffix}").exists():
                    suffix += 1
                cohort_dir = out / "cohorts" / f"{cohort_fingerprint[:20]}-rebuild{suffix}"
            prepare_cohort(config_path, cohort_dir, normalized_source_dir=source_root)
            cached = _cohort_cached(cohort_dir, base, source_hash)
            if cached is None:
                raise RuntimeError("Prepared cohort failed its saved manifest/config/source checks.")
        payload, manifest = cached
        if manifest["provider"] != "fastf1" or len(manifest["sessions"]) < 3:
            raise ValueError("Prepared cohort provider/session-count invariant failed")
        cfg = copy.deepcopy(payload["config"])
        cfg["train"]["checkpoint_path"] = str(out / "training" / "_path_placeholder" / "best.pt")
        cfg["train"]["history_path"] = str(out / "training" / "_path_placeholder" / "training_history.csv")
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(args.threads)
        code_inputs = {name: sha256(HERE / "src" / name) for name in ("train.py", "model.py", "dataset.py")}
        train_fingerprint = canonical_hash({"cohort_artifact": manifest["prepared_artifact_sha256"],
                                            "config": cfg, "code": code_inputs,
                                            "fastf1": fastf1.__version__, "device": device.type,
                                            "threads": args.threads})
        training_dir = out / "training" / train_fingerprint[:20]
        training_dir.mkdir(parents=True, exist_ok=True)
        cfg["train"]["checkpoint_path"] = str(training_dir / "best.pt")
        cfg["train"]["last_checkpoint_path"] = str(training_dir / "best.last.pt")
        cfg["train"]["history_path"] = str(training_dir / "training_history.csv")
        expected_resume_manifest = {
            "fingerprint": train_fingerprint,
            "prepared_artifact_sha256": manifest["prepared_artifact_sha256"],
            "prepared_manifest_sha256": manifest["prepared_manifest_sha256"],
            "config_sha256": canonical_hash(cfg), "implementation_sha256": code_inputs,
        }
        resume_manifest_path = training_dir / "resume_manifest.json"
        if resume_manifest_path.is_file():
            resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
            if resume_manifest != expected_resume_manifest:
                raise ValueError("Training resume manifest does not match this run configuration/cohort.")
        else:
            atomic_json(resume_manifest_path, expected_resume_manifest)
            resume_manifest = expected_resume_manifest
        if resume_manifest != expected_resume_manifest:
            raise ValueError("Training resume manifest does not match this run configuration/cohort.")
        best_path = Path(cfg["train"]["checkpoint_path"])
        last_path = Path(cfg["train"]["last_checkpoint_path"])
        train_marker = training_dir / "training_complete.json"
        complete_training = False
        if train_marker.is_file() and best_path.is_file() and last_path.is_file():
            try:
                saved_marker = json.loads(train_marker.read_text(encoding="utf-8"))
                complete_training = (saved_marker.get("fingerprint") == train_fingerprint
                                     and saved_marker.get("best_sha256") == sha256(best_path)
                                     and saved_marker.get("last_sha256") == sha256(last_path)
                                     and saved_marker.get("history_sha256") == sha256(Path(cfg["train"]["history_path"])))
            except (OSError, ValueError):
                complete_training = False
        loaders = None
        if not complete_training:
            train_loader, val_loader, test_loader, scaler = create_dataloaders(
                copy.deepcopy(payload["snapshots"]), payload["labels"], payload["driver_to_idx"], cfg)
            model = create_model(cfg).to(device)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            resume_from = last_path if last_path.is_file() else None
            best_val, history = train_model(model, train_loader, val_loader, cfg, device, resume_from=resume_from)
            hist = pd.DataFrame(history)
            atomic_csv(hist, Path(cfg["train"]["history_path"]))
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            saved = checkpoint.get("feature_scaler") or {}
            if not np.allclose(scaler.mean_, saved.get("mean_", [])) or not np.allclose(scaler.scale_, saved.get("scale_", [])):
                raise ValueError("Best checkpoint scaler does not match train-only prepared-cohort scaler")
            atomic_json(train_marker, {"fingerprint": train_fingerprint, "best_sha256": sha256(best_path),
                                       "last_sha256": sha256(last_path),
                                       "history_sha256": sha256(Path(cfg["train"]["history_path"]))})
            loaders = (train_loader, val_loader, test_loader, scaler)
        else:
            hist = pd.read_csv(cfg["train"]["history_path"])
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_val = float(checkpoint.get("val_mae"))

        pred_fingerprint = canonical_hash({"best_sha256": sha256(best_path),
                                           "test_membership_sha256": manifest["split_membership_sha256"]["test"],
                                           "prediction_code": sha256(HERE / "src/evaluate.py")})
        predictions_path = training_dir / "test_predictions.pkl"
        def predict_frame() -> pd.DataFrame:
            if loaders is None:
                _, _, test_loader, scaler = create_dataloaders(
                    copy.deepcopy(payload["snapshots"]), payload["labels"], payload["driver_to_idx"], cfg)
            else:
                _, _, test_loader, scaler = loaders
            model = create_model(cfg).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            saved = checkpoint.get("feature_scaler") or {}
            if not np.allclose(scaler.mean_, saved.get("mean_", [])) or not np.allclose(scaler.scale_, saved.get("scale_", [])):
                raise ValueError("Best checkpoint scaler does not match train-only prepared-cohort scaler")
            _, pred, records = predict_test_set(model, test_loader, device)
            return pd.DataFrame(records)
        frame = load_or_compute_predictions(predictions_path, pred_fingerprint, predict_frame)
        y = frame["y_true"].to_numpy(float)
        pred = frame["y_pred"].to_numpy(float)
        records = frame.to_dict(orient="records")
        metric_fingerprint = canonical_hash({"prediction_fingerprint": pred_fingerprint,
                                             "physics_tests": cfg["physics_tests"],
                                             "target": cfg["target"], "track": cfg["track"],
                                             "evaluation_code": sha256(HERE / "src/evaluate.py")})
        metrics_path = training_dir / "test_metrics.json"
        if _output_is_current(metrics_path, metric_fingerprint):
            metric_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = metric_payload["metrics"]
        else:
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
            atomic_json(metrics_path, {
                "metrics": metrics, "best_validation_mae_sec": best_val,
                "checkpoint_epoch": checkpoint.get("epoch"), "split": manifest["split"],
                "provider": "fastf1", "sessions": manifest["sessions"],
                "target_definition": "existing target builder; full-session calibrated",
            })
            _record_output(metrics_path, metric_fingerprint)
        exports = [(frame, training_dir / "test_predictions.csv"),
                   (pd.DataFrame([{key: val for key, val in metrics.items()
                                   if isinstance(val, (int, float, bool)) or val is None}]),
                    training_dir / "test_metrics.csv")]
        for group in ("RaceId", "Driver", "MiniSector"):
            if group in frame:
                exports.append((pd.DataFrame(aggregate(frame, group)), training_dir / f"per_{group.lower()}_metrics.csv"))
        def export_csv(item) -> None:
            frame_to_write, path = item
            if not _output_is_current(path, metric_fingerprint):
                _atomic_csv(frame_to_write, path)
                _record_output(path, metric_fingerprint)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(export_csv, exports))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_fingerprint = canonical_hash({"prediction": pred_fingerprint,
                                           "history": sha256(Path(cfg["train"]["history_path"]))})
        loss_path = training_dir / "loss.png"
        if not _output_is_current(loss_path, plot_fingerprint):
            fig, ax = plt.subplots(); ax.plot(hist.epoch, hist.train_loss, label="train loss"); ax.plot(hist.epoch, hist.val_mae, label="validation MAE"); ax.legend(); ax.set_xlabel("Epoch"); fig.savefig(loss_path, dpi=140); plt.close(fig)
            _record_output(loss_path, plot_fingerprint)
        predicted_actual_path = training_dir / "predicted_vs_actual.png"
        if not _output_is_current(predicted_actual_path, plot_fingerprint):
            fig, ax = plt.subplots(); ax.scatter(y, pred, s=4, alpha=.35); lim=[min(y.min(),pred.min()), max(y.max(),pred.max())]; ax.plot(lim,lim,"k--"); ax.set_xlabel("Observed delay (s)"); ax.set_ylabel("Predicted delay (s)"); fig.savefig(predicted_actual_path, dpi=140); plt.close(fig)
            _record_output(predicted_actual_path, plot_fingerprint)
        residual_path = training_dir / "residuals.png"
        if not _output_is_current(residual_path, plot_fingerprint):
            fig, ax = plt.subplots(); ax.hist(pred-y, bins=50); ax.set_xlabel("Prediction residual (s)"); fig.savefig(residual_path, dpi=140); plt.close(fig)
            _record_output(residual_path, plot_fingerprint)
        return out
    except Exception as exc:
        # Keep secrets out of durable diagnostics; exception type is sufficient here.
        atomic_json(out / "run_failure.json", {"exception_type": type(exc).__name__,
                                               "stage": "acquisition_or_training"})
        raise RuntimeError(f"2023 FastF1 season run failed ({type(exc).__name__}); see {out} for sanitized diagnostics.") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Reusable run workspace; default .local/research-history/2023-fastf1-season.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=2, help="Bounded I/O validation/export workers (1-8); downloads stay serialized.")
    parser.add_argument("--torch-threads", "--threads", dest="threads", type=int, default=1,
                        help="PyTorch intra-op CPU threads.")
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0 or not 1 <= args.workers <= 8 or args.threads < 1:
        parser.error("--epochs and --threads must be positive, --seed non-negative, and --workers between 1 and 8")
    try:
        print(run(args))
    except Exception as exc:
        logging.error("2023 FastF1 season run failed (%s)", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
