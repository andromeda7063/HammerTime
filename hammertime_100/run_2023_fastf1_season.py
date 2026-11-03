"""Prepare and train a race-complete 2023 FastF1 season cohort.

Online acquisition mode (FastF1 3.9.0 is supplied as a runtime overlay):
    uv run --frozen --with fastf1==3.9.0 python hammertime_100/run_2023_fastf1_season.py --processing-workers 8 --workers 2 --torch-threads 8

Use --local-data-only --raw-sessions-dir PATH to avoid all FastF1 source/auth
access. That mode requires hash-verified local raw tables for all 22 registered
2023 Grand Prix races and fails closed if any session is missing or invalid.
Online mode reads credentials from the process environment or repository .env.
Credentials are never written to run artifacts. Existing coverage, physics,
split, and model gates are applied unchanged by preparation and training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy
import ctypes
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
import shutil
from pathlib import Path
import re
import sys
import threading
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

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
    "Montréal": ("Canada", 4361.0, "canada"),
    "Spielberg": ("Austria", 4318.0, "austria"),
    "Silverstone": ("Great Britain", 5891.0, "great-britain"),
    "Budapest": ("Hungary", 4381.0, "hungary"),
    "Spa-Francorchamps": ("Belgium", 7004.0, "belgium"),
    "Zandvoort": ("Netherlands", 4259.0, "netherlands"),
    "Monza": ("Italy", 5793.0, "italy"),
    "Singapore": ("Singapore", 4940.0, "singapore"),
    "Marina Bay": ("Singapore", 4940.0, "singapore"),
    "Suzuka": ("Japan", 5807.0, "japan"),
    "Lusail": ("Qatar", 5419.0, "qatar"),
    "Austin": ("United States", 5513.0, "usa"),
    "Mexico City": ("Mexico", 4304.0, "mexico"),
    "São Paulo": ("São Paulo", 4309.0, "brazil"),
    "Las Vegas": ("Las Vegas", 6201.0, "las-vegas"),
    "Yas Marina Circuit": ("Abu Dhabi", 5281.0, "abu-dhabi"),
    "Yas Island": ("Abu Dhabi", 5281.0, "abu-dhabi"),
    "Abu Dhabi": ("Abu Dhabi", 5281.0, "abu-dhabi"),
}

# The complete 2023 Grand Prix race inventory. Local-data-only runs must match
# this registry exactly; they never infer season completeness from whatever
# directories happen to be present on disk.
EXPECTED_2023_RACE_IDS = (
    "2023-Bahrain-R", "2023-SaudiArabia-R", "2023-Australia-R", "2023-Azerbaijan-R",
    "2023-Miami-R", "2023-Monaco-R", "2023-Spain-R", "2023-Canada-R",
    "2023-Austria-R", "2023-GreatBritain-R", "2023-Hungary-R", "2023-Belgium-R",
    "2023-Netherlands-R", "2023-Italy-R", "2023-Singapore-R", "2023-Japan-R",
    "2023-Qatar-R", "2023-UnitedStates-R", "2023-Mexico-R", "2023-SãoPaulo-R",
    "2023-LasVegas-R", "2023-AbuDhabi-R",
)
EXPECTED_2023_ROUND_SCHEMES = {
    "completed_races_1_to_22": {
        race_id: round_number
        for round_number, race_id in enumerate(EXPECTED_2023_RACE_IDS, start=1)
    },
    "published_calendar_1_to_5_then_7_to_23": {
        race_id: round_number
        for race_id, round_number in zip(EXPECTED_2023_RACE_IDS, (*range(1, 6), *range(7, 24)))
    },
}


def _dotenv_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    # Support ordinary unquoted KEY=value lines and whitespace comments.
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def load_credentials() -> tuple[str, str]:
    """Load only the two expected variables, without displaying or persisting them."""
    dotenv_values = {}
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
                dotenv_values[key.strip()] = _dotenv_value(raw)
    username = os.environ.get("F1_USERNAME") or dotenv_values.get("F1_USERNAME", "")
    password = os.environ.get("F1_PASSWORD") or dotenv_values.get("F1_PASSWORD", "")
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


def _sanitized_error(exc: Exception, secrets: tuple[str, str]) -> dict[str, str]:
    """Capture actionable failure details while scrubbing account credentials."""
    message = str(exc)
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
            formatted = formatted.replace(secret, "[REDACTED]")
    sensitive_assignment = re.compile(
        r"(?i)\b(password|token|authorization|api[_-]?key)(\s*[:=]\s*)[^\s,;]+"
    )
    bearer = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
    message = bearer.sub("Bearer [REDACTED]", sensitive_assignment.sub(r"\1\2[REDACTED]", message))
    formatted = bearer.sub("Bearer [REDACTED]", sensitive_assignment.sub(r"\1\2[REDACTED]", formatted))
    return {"exception_type": type(exc).__name__, "message": message, "traceback": formatted}


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


def _raw_fingerprint(event: dict, fastf1_version: str) -> str:
    return canonical_hash({"year": event["year"], "gp": event["gp"], "session_type": event["session_type"],
                           "fastf1_version": fastf1_version, "ingest_code": sha256(HERE / "src/ingest.py")})


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
        for name, expected in record.get("source_files", {}).items():
            path = Path(name)
            if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _valid_raw_stage(stage: Path, expected_fingerprint: str) -> dict | None:
    marker = stage / "raw_manifest.json"
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


def _validate_local_raw_manifest(stage: Path, expected_race_id: str) -> dict:
    """Validate a transferred authentic raw stage without FastF1 or network access."""
    marker = stage / "raw_manifest.json"
    if stage.is_symlink() or not marker.is_file() or marker.is_symlink():
        raise ValueError(f"missing manifest: {marker}")
    record = json.loads(marker.read_text(encoding="utf-8"))
    event = record.get("event")
    if not isinstance(event, dict):
        raise ValueError("manifest has no event object")
    round_number = event.get("round_number")
    if (event.get("year") != 2023 or event.get("session_type") != "R"
            or event.get("race_id") != expected_race_id
            or isinstance(round_number, bool) or not isinstance(round_number, int)
            or round_number < 1 or round_number > 23):
        raise ValueError("manifest event identity/year/session/round is invalid for the registered 2023 race")
    gp = event.get("gp")
    if not isinstance(gp, str) or not gp.strip():
        raise ValueError("manifest GP event name is missing")
    length = event.get("lap_length_m")
    if isinstance(length, bool) or not isinstance(length, (int, float)) or not np.isfinite(length) or length <= 0:
        raise ValueError("manifest circuit length is missing or invalid")
    source_metadata = record.get("source_metadata")
    if (not isinstance(source_metadata, dict) or source_metadata.get("provider") != "fastf1"
            or not isinstance(source_metadata.get("session"), dict)
            or not source_metadata["session"].get("circuit_key")):
        raise ValueError("manifest does not identify FastF1 as its source provider")
    manifest_fingerprint = record.get("fingerprint")
    if not isinstance(manifest_fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", manifest_fingerprint) is None:
        raise ValueError("manifest has no valid raw-stage fingerprint")
    files = record.get("files")
    if not isinstance(files, dict) or not {"raw_laps.parquet", "raw_telemetry.parquet"}.issubset(files):
        raise ValueError("manifest is missing the required raw lap or telemetry table")
    root = stage.resolve(strict=True)
    for relative_name, expected in files.items():
        if not isinstance(relative_name, str) or Path(relative_name).name != relative_name:
            raise ValueError("manifest contains a non-canonical raw table path")
        path = stage / relative_name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"raw table is missing or is a symlink: {relative_name}")
        if path.resolve(strict=True).parent != root:
            raise ValueError(f"raw table escapes its race directory: {relative_name}")
        if not isinstance(expected, dict) or set(("bytes", "sha256")) - expected.keys():
            raise ValueError(f"raw table has incomplete integrity metadata: {relative_name}")
        if not isinstance(expected["bytes"], int) or isinstance(expected["bytes"], bool) or expected["bytes"] <= 0:
            raise ValueError(f"raw table has invalid byte count: {relative_name}")
        if not isinstance(expected["sha256"], str) or len(expected["sha256"]) != 64:
            raise ValueError(f"raw table has invalid SHA256 metadata: {relative_name}")
        if (path.stat().st_size != expected["bytes"]
                or sha256(path) != expected["sha256"]):
            raise ValueError(f"raw table size or SHA256 mismatch: {relative_name}")
    drivers = record.get("drivers")
    if not isinstance(drivers, list) or not drivers:
        raise ValueError("manifest has no driver inventory")
    return record


def load_local_raw_inventory(raw_root: Path) -> tuple[list[dict], list[dict], str]:
    """Require and verify every registered 2023 race in a transferred raw cache."""
    try:
        root = raw_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Local raw-session directory is unavailable: {raw_root} ({type(exc).__name__}).") from None
    if not root.is_dir():
        raise RuntimeError(f"Local raw-session path is not a directory: {root}")
    events, manifests, problems = [], [], []
    expected_directories = {_safe_id(race_id) for race_id in EXPECTED_2023_RACE_IDS}
    for child in root.iterdir():
        if child.name.startswith("."):
            continue  # Preserve and ignore prior atomic staging/stale backups.
        if child.is_dir() and child.name not in expected_directories:
            problems.append(f"unregistered raw-session directory: {child.name}")
    for race_id in EXPECTED_2023_RACE_IDS:
        stage = root / _safe_id(race_id)
        try:
            manifest = _validate_local_raw_manifest(stage, race_id)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems.append(f"{race_id}: {exc}")
            continue
        events.append(manifest["event"])
        manifests.append(manifest)
    if problems or len(events) != len(EXPECTED_2023_RACE_IDS):
        details = "\n  - ".join(problems) if problems else "registered race inventory count mismatch"
        raise RuntimeError(
            f"Local-data-only mode requires all {len(EXPECTED_2023_RACE_IDS)} verified 2023 Grand Prix race stages. "
            f"No races were downloaded or skipped. Problems:\n  - {details}\n"
            f"Expected directory: {root} (22 race subdirectories, each with raw_manifest.json, "
            "raw_laps.parquet, and raw_telemetry.parquet). Inspect it in PowerShell with "
            f"Get-ChildItem -Force '{root}'. Correct the directory or file layout, then rerun with "
            f"--raw-sessions-dir '{root}'."
        )
    gp_names = [str(event["gp"]).strip().casefold() for event in events]
    circuit_keys = [str(manifest["source_metadata"]["session"]["circuit_key"]).strip().casefold()
                    for manifest in manifests]
    if len(set(gp_names)) != len(EXPECTED_2023_RACE_IDS) or len(set(circuit_keys)) != len(EXPECTED_2023_RACE_IDS):
        raise RuntimeError("Local-data-only mode found duplicate Grand Prix names or circuit identities in the 22-race inventory.")
    round_assignments = {event["race_id"]: event["round_number"] for event in events}
    matching_schemes = [name for name, expected in EXPECTED_2023_ROUND_SCHEMES.items()
                        if round_assignments == expected]
    if len(matching_schemes) != 1:
        raise RuntimeError(
            "Local-data-only mode requires a complete, internally consistent 2023 round-numbering scheme "
            "(completed-race rounds 1–22 or published-calendar rounds 1–5 then 7–23); "
            "mixed, duplicate, or unknown round assignments are rejected."
        )
    return events, manifests, matching_schemes[0]


def ensure_raw_stage(event: dict, cache_root: Path, raw_root: Path,
                     fastf1_version: str, credentials: tuple[str, str]) -> tuple[dict | None, str | None]:
    """Serialize FastF1 access in the parent and atomically persist raw session tables."""
    from ingest import export_raw_tables, load_race

    raw_root.mkdir(parents=True, exist_ok=True)
    stage = raw_root / _safe_id(event["race_id"])
    fingerprint = _raw_fingerprint(event, fastf1_version)
    prior = _valid_raw_stage(stage, fingerprint)
    if prior is not None:
        return {**event, **prior["event"]}, None
    staging = raw_root / f".{stage.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        os.environ["F1_USERNAME"], os.environ["F1_PASSWORD"] = credentials
        os.environ["FASTF1_AUTH_METHOD"] = "password"
        try:
            session = load_race(2023, event["gp"], "R", str(cache_root / event["race_id"]))
        finally:
            os.environ.pop("F1_USERNAME", None)
            os.environ.pop("F1_PASSWORD", None)
            os.environ.pop("FASTF1_AUTH_METHOD", None)
        raw_laps, raw_telemetry = export_raw_tables(session, str(staging))
        circuit_key = str(session.event.get("CircuitKey", event.get("circuit_id", "")))
        final_event = {**event, "circuit_id": circuit_key}
        raw_files = [Path(raw_laps)] + ([Path(raw_telemetry)] if raw_telemetry is not None else [])
        files = {path.relative_to(staging).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in raw_files}
        source_metadata = {"provider": "fastf1", "session": {"circuit_key": circuit_key}}
        record = {"fingerprint": fingerprint, "event": final_event, "files": files,
                  "source_metadata": source_metadata,
                  "fastf1_version": fastf1_version,
                  "drivers": [str(driver) for driver in session.drivers],
                  "coverage": {"raw_lap_rows": int(len(session.laps))}}
        atomic_json(staging / "raw_manifest.json", record)
        if stage.exists():
            backup = raw_root / f".{stage.name}.stale-{uuid.uuid4().hex}"
            os.replace(stage, backup)
        os.replace(staging, stage)
        return final_event, None
    except Exception as exc:
        os.environ.pop("F1_USERNAME", None)
        os.environ.pop("F1_PASSWORD", None)
        os.environ.pop("FASTF1_AUTH_METHOD", None)
        shutil.rmtree(staging, ignore_errors=True)
        return None, type(exc).__name__


def _driver_key(value) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def normalize_race_worker(task: dict) -> tuple[dict | None, str | None]:
    """CPU-only normalization worker; consumes persisted tables, never FastF1/auth."""
    from preprocess import _time_seconds, align_driver_telemetry, compute_time_gaps
    from target import build_targets

    event = task["event"]
    config = task["config"]
    stage = Path(task["stage"])
    raw_laps_path, raw_telemetry_path = Path(task["raw_laps"]), Path(task["raw_telemetry"])
    staging = stage.parent / f".{stage.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        laps = pd.read_parquet(raw_laps_path)
        telemetry = pd.read_parquet(raw_telemetry_path)
        if "Driver" not in telemetry or "DriverNumber" not in laps:
            raise ValueError("raw FastF1 tables do not contain driver identity columns")
        drivers = task["drivers"]
        raw_drivers = []
        for driver in drivers:
            key = _driver_key(driver)
            drv_laps = laps.loc[laps["DriverNumber"].map(_driver_key) == key].copy()
            if drv_laps.empty:
                continue
            drv_tel = telemetry.loc[telemetry["Driver"].map(_driver_key) == key].copy()
            if drv_tel.empty:
                continue
            time_column = next((column for column in ("TimeSec", "SessionTime", "Time")
                                if column in drv_tel.columns), None)
            if time_column is None:
                continue
            drv_tel["TimeSec"] = _time_seconds(drv_tel[time_column])
            drv_tel = drv_tel.dropna(subset=["TimeSec"])
            if not drv_tel.empty:
                raw_drivers.append((key, drv_tel, drv_laps))
        # Per-driver frames above are copies. Drop the full race tables before
        # resampling to avoid keeping both representations live together.
        del laps, telemetry
        if not raw_drivers:
            raise ValueError("no usable per-driver telemetry in the persisted FastF1 tables")
        sample_hz = float(config["preprocess"]["sample_hz"])
        interval = 1.0 / sample_hz
        t_start = min(frame["TimeSec"].min() for _, frame, _ in raw_drivers)
        t_end = max(frame["TimeSec"].max() for _, frame, _ in raw_drivers)
        time_grid = np.arange(t_start, t_end + interval * 1e-6, interval)
        all_driver_dfs = []
        for driver_index, (driver, drv_tel, drv_laps) in enumerate(raw_drivers):
            try:
                aligned = align_driver_telemetry(
                    driver, drv_tel, drv_laps, sample_hz=sample_hz,
                    max_interpolate_sec=float(config["preprocess"]["max_interpolate_sec"]),
                    mini_sectors=int(config["track"]["mini_sectors"]),
                    lap_length_m=float(event["lap_length_m"]), time_grid=time_grid,
                    flag_safety_car=bool(config["preprocess"].get("flag_safety_car", True)),
                    flag_pit_laps=bool(config["preprocess"].get("flag_pit_laps", True)))
                if not aligned.empty:
                    all_driver_dfs.append(aligned)
            except Exception:
                continue
            finally:
                raw_drivers[driver_index] = None
                del drv_tel, drv_laps
        del raw_drivers
        if not all_driver_dfs:
            raise ValueError("no driver telemetry could be aligned")
        combined = pd.concat(all_driver_dfs, ignore_index=True)
        del all_driver_dfs
        combined["RaceId"] = str(event["race_id"])
        aligned = compute_time_gaps(combined, sample_hz=sample_hz,
                                    lap_length_m=float(event["lap_length_m"]))
        target_config = copy.deepcopy(config)
        target_config["session"].update(event)
        target_config["track"]["lap_length_m"] = float(event["lap_length_m"])
        labels = build_targets(aligned, target_config)
        if labels.empty:
            raise ValueError("existing target builder produced no labels")
        if int(aligned.get("IsValid", pd.Series(dtype=bool)).fillna(False).sum()) == 0:
            raise ValueError("existing aligned telemetry validity gate rejected all rows")
        aligned.to_parquet(staging / "aligned_telemetry.parquet", index=False)
        labels.to_parquet(staging / "target_labels.parquet", index=False)
        outputs = [staging / "aligned_telemetry.parquet", staging / "target_labels.parquet"]
        files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in outputs}
        source_files = {str(Path(name).resolve()): expected for name, expected in task["raw_files"].items()}
        record = {"fingerprint": task["fingerprint"], "event": {**event, "validated_target_rows": int(len(labels))},
                  "files": files, "source_files": source_files,
                  "source_metadata": task["source_metadata"],
                  "coverage": {"aligned_rows": int(len(aligned)), "target_rows": int(len(labels))}}
        atomic_json(staging / "stage_manifest.json", record)
        if stage.exists():
            backup = stage.parent / f".{stage.name}.stale-{uuid.uuid4().hex}"
            os.replace(stage, backup)
        os.replace(staging, stage)
        return record["event"], None
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        return None, type(exc).__name__


def write_normalized_source(stage_root: Path, source_root: Path,
                            config_path: Path, events: list[dict],
                            cache_root: Path | None = None,
                            persist: bool = True) -> str:
    """Write an authenticated normalized-source cache manifest, not a cohort manifest."""
    from prepare_cohort import _canonical_json, _sha256_bytes

    records, hashes = [], {}
    for event in events:
        race_id = str(event["race_id"])
        stage = stage_root / _safe_id(race_id)
        stage_meta = json.loads((stage / "stage_manifest.json").read_text(encoding="utf-8"))
        source_files = []
        artifacts = {str((stage / name).resolve()): info for name, info in stage_meta["files"].items()}
        artifacts.update(stage_meta.get("source_files", {}))
        for filename, info in artifacts.items():
            path = Path(filename)
            try:
                rel = str(path.resolve().relative_to(source_root.resolve()))
            except ValueError:
                continue
            hashes[rel] = {"bytes": int(info["bytes"]), "sha256": info["sha256"]}
            source_files.append(rel)
        records.append({"race_id": race_id, "provider": "fastf1", "year": int(event["year"]),
                        "gp": str(event["gp"]), "session_type": str(event["session_type"]),
                        "session_key": event.get("session_key"),
                        "cache_root": str((cache_root / race_id).resolve()) if cache_root is not None else None,
                        "lap_length_m": float(event["lap_length_m"]), "source_files": source_files,
                        "coverage": stage_meta["coverage"], "source_metadata": stage_meta["source_metadata"]})
    manifest = {"schema_version": 1, "artifact_type": "hammertime-normalized-source-cache",
                "provider": "fastf1", "providers": ["fastf1"],
                "config_path": str(config_path.resolve()), "config_sha256": sha256(config_path),
                "sessions": records, "source_hashes": hashes, "normalized_source_manifest_sha256": None}
    manifest["normalized_source_manifest_sha256"] = _sha256_bytes(_canonical_json(
        {key: value for key, value in manifest.items() if key != "normalized_source_manifest_sha256"}))
    raw = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    if persist:
        atomic_bytes(source_root / "normalized_source_manifest.json", raw)
        atomic_bytes(source_root / "normalized_source_manifest.json.sha256",
                     f"{hashlib.sha256(raw).hexdigest()}  normalized_source_manifest.json\n".encode())
    return hashlib.sha256(raw).hexdigest()


def _available_memory_bytes() -> int | None:
    """Return currently available physical memory on Windows and Unix hosts."""
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        try:
            memory_status = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
            memory_status.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
            memory_status.restype = ctypes.c_bool
            if memory_status(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, TypeError):
            return None
        return None

    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and available_pages >= 0:
            return page_size * available_pages
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _worker_memory_limit(requested: int, available_bytes: int | None = None) -> int:
    """Cap dataframe processes by available RAM; fail closed to one if unknown."""
    available = _available_memory_bytes() if available_bytes is None else int(available_bytes)
    if available is None:
        return min(requested, 1)
    reserve = 2 * 1024**3
    # Inherited conservative sizing heuristic; child RSS has not been measured
    # on the target Windows data and this is not a safety-bound guarantee.
    estimate_per_worker = 700 * 1024**2
    safe_count = max(1, (available - reserve) // estimate_per_worker)
    return min(requested, int(safe_count))


def _process_memory_bytes() -> int | None:
    """Best-effort resident memory for this process, with no optional package."""
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            get_memory_info = psapi.GetProcessMemoryInfo
            get_memory_info.argtypes = [ctypes.c_void_p,
                                        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), ctypes.c_ulong]
            get_memory_info.restype = ctypes.c_int
            ok = get_memory_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return int(counters.WorkingSetSize) if ok else None
        except (AttributeError, OSError, TypeError):
            return None
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _resource_summary() -> str:
    rss = _process_memory_bytes()
    available = _available_memory_bytes()
    fields = []
    if rss is not None:
        fields.append(f"process RSS={rss / 1024**3:.2f} GiB")
    if available is not None:
        fields.append(f"system available={available / 1024**3:.2f} GiB")
    return ", ".join(fields) if fields else "memory probe unavailable"


def _worker_warmup() -> int:
    """Start process workers while auth variables are absent from the environment."""
    for name in ("F1_USERNAME", "F1_PASSWORD", "FASTF1_AUTH_METHOD"):
        os.environ.pop(name, None)
    import time
    time.sleep(0.1)
    return os.getpid()


def _normalization_fingerprint(event: dict, raw_manifest: dict, config: dict,
                              fastf1_version: str,
                              raw_manifest_sha256: str | None = None) -> str:
    return canonical_hash({"processing": _race_fingerprint(event, config, fastf1_version),
                           "raw_source_sha256": raw_manifest["files"],
                           "raw_manifest_sha256": raw_manifest_sha256,
                           "raw_manifest_inputs": {
                               "event": raw_manifest.get("event"),
                               "drivers": raw_manifest.get("drivers"),
                               "source_metadata": raw_manifest.get("source_metadata"),
                               "fastf1_version": raw_manifest.get("fastf1_version"),
                               "fingerprint": raw_manifest.get("fingerprint"),
                           }})


def _normalization_task(event: dict, raw_stage: Path, raw_manifest: dict,
                        config: dict, fastf1_version: str, normalized_root: Path,
                        source_metadata: dict | None = None) -> dict:
    fingerprint = _normalization_fingerprint(
        event, raw_manifest, config, fastf1_version, sha256(raw_stage / "raw_manifest.json"))
    raw_files = {str((raw_stage / name).resolve()): info for name, info in raw_manifest["files"].items()}
    source_metadata = dict(source_metadata or raw_manifest["source_metadata"])
    source_metadata["raw_manifest_sha256"] = sha256(raw_stage / "raw_manifest.json")
    source_metadata["raw_manifest_fingerprint"] = raw_manifest.get("fingerprint")
    source_metadata["raw_source_files"] = raw_manifest["files"]
    return {"event": event, "config": config,
            "stage": str(normalized_root / _safe_id(event["race_id"])),
            "raw_laps": str(raw_stage / "raw_laps.parquet"),
            "raw_telemetry": str(raw_stage / "raw_telemetry.parquet"),
            "drivers": raw_manifest["drivers"], "raw_files": raw_files,
            "fingerprint": fingerprint,
            "source_metadata": source_metadata}


def _normalization_source_version(raw_manifest: dict, local_data_only: bool,
                                 installed_version: str | None = None) -> str:
    version = raw_manifest.get("fastf1_version")
    if isinstance(version, str) and version.strip():
        return version
    if local_data_only:
        # Legacy manifests predate explicit version metadata. Keep their opaque
        # source fingerprint in the cache identity without guessing a version.
        return f"unrecorded-local-raw:{raw_manifest.get('fingerprint', 'missing-fingerprint')}"
    return installed_version or "unrecorded-online-raw"


def _normalization_source_metadata(raw_manifest: dict, local_data_only: bool,
                                   installed_version: str | None = None,
                                   round_scheme: str | None = None) -> dict:
    metadata = dict(raw_manifest["source_metadata"])
    version = raw_manifest.get("fastf1_version")
    if isinstance(version, str) and version.strip():
        metadata["fastf1_version"] = version
        metadata["fastf1_version_provenance"] = "raw_manifest"
    elif local_data_only:
        metadata["fastf1_version"] = "unrecorded"
        metadata["fastf1_version_provenance"] = "legacy_raw_manifest_without_version_field"
    else:
        metadata["fastf1_version"] = installed_version or "unrecorded"
        metadata["fastf1_version_provenance"] = (
            "runtime_environment_fallback" if installed_version else "online_raw_manifest_without_version_field")
    if local_data_only and round_scheme:
        metadata["round_numbering_scheme"] = round_scheme
    return metadata


def _start_worker_pool(workers: int) -> tuple[ProcessPoolExecutor, set[int]]:
    """Spawn the complete bounded pool while parent F1 credential env is clear."""
    for name in ("F1_USERNAME", "F1_PASSWORD", "FASTF1_AUTH_METHOD"):
        os.environ.pop(name, None)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        pids: set[int] = set()
        for _ in range(4):
            futures = [pool.submit(_worker_warmup) for _ in range(workers)]
            pids.update(future.result() for future in futures)
            if len(pids) == workers:
                break
        if len(pids) < workers:
            raise RuntimeError("Could not start the requested number of normalization worker processes.")
        return pool, pids
    except Exception:
        pool.shutdown(wait=True, cancel_futures=True)
        raise


def _collect_normalization_futures(future_to_id: dict) -> tuple[dict[str, dict], dict[str, str]]:
    """Collect in completion order while keying results/errors by session ID."""
    results, failures = {}, {}
    for future in as_completed(future_to_id):
        race_id = future_to_id[future]
        try:
            result, failure_type = future.result()
        except Exception as exc:
            result, failure_type = None, type(exc).__name__
        if result is None:
            failures[race_id] = failure_type or "NormalizationFailed"
        else:
            results[race_id] = result
    return results, failures


def _cohort_fingerprint(config: dict, source_hash: str) -> str:
    prepare_files = (
        "prepare_cohort.py", "run_pipeline.py", "graph.py",
        "context_features.py", "dataset.py", "preprocess.py", "target.py",
    )
    prepare_code = {
        name: sha256(HERE / name if (HERE / name).is_file() else HERE / "src" / name)
        for name in prepare_files
    }
    return canonical_hash({
        "source_manifest_sha256": source_hash,
        "config": config,
        "prepare_code": prepare_code,
    })


def _training_identity_config(config: dict) -> dict:
    """Exclude progress logging, which cannot change the trained model."""
    identity = copy.deepcopy(config)
    identity.get("train", {}).pop("log_every_batches", None)
    return identity


def _align_cohort_num_nodes(config: dict, events: list[dict], stages_root: Path) -> int:
    """Ensure graph.num_nodes accommodates distinct drivers (e.g. substitutions) across sessions."""
    cohort_drivers: set[str] = set()
    for event in events:
        stage = stages_root / _safe_id(event["race_id"])
        aligned_path = stage / "aligned_telemetry.parquet"
        if aligned_path.is_file():
            cohort_drivers.update(pd.read_parquet(aligned_path, columns=["Driver"])["Driver"].astype(str).unique())
    required = max(int(config.get("graph", {}).get("num_nodes", 20)), len(cohort_drivers))
    config.setdefault("graph", {})["num_nodes"] = required
    return required


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


def _cohort_manifest_cached(cohort_dir: Path, config: dict, source_hash: str):
    """Verify a prepared cohort's manifest and artifact bytes without unpickling it."""
    from prepare_cohort import _canonical_json, _sha256_bytes

    artifact = cohort_dir / "prepared_cohort.pkl"
    manifest_path = cohort_dir / "prepared_manifest.json"
    sidecar = Path(str(manifest_path) + ".sha256")
    try:
        if not artifact.is_file() or not manifest_path.is_file() or not sidecar.is_file():
            return None
        manifest_bytes = manifest_path.read_bytes()
        exact_hash = hashlib.sha256(manifest_bytes).hexdigest()
        if sidecar.read_text(encoding="utf-8").strip().split()[0] != exact_hash:
            return None
        manifest = json.loads(manifest_bytes)
        canonical_hash = _sha256_bytes(_canonical_json(
            {key: value for key, value in manifest.items() if key != "prepared_manifest_sha256"}))
        expected_config = _sha256_bytes(_canonical_json(config))
        if (manifest.get("prepared_manifest_sha256") != canonical_hash
                or manifest.get("prepared_manifest_path") != str(manifest_path.resolve())
                or manifest.get("prepared_artifact_path") != str(artifact.resolve())
                or manifest.get("prepared_artifact_bytes") != artifact.stat().st_size
                or manifest.get("prepared_artifact_sha256") != sha256(artifact)
                or manifest.get("provider") != "fastf1"
                or manifest.get("effective_config_sha256") != expected_config
                or manifest.get("aligned_source_manifest_sha256") != source_hash):
            return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def _load_training_overrides(config: dict, path: Path) -> dict:
    """Apply a small training/model-only overlay after immutable cohort loading."""
    with path.open(encoding="utf-8") as stream:
        overrides = yaml.safe_load(stream)
    if not isinstance(overrides, dict) or set(overrides) - {"model", "train"}:
        raise ValueError("Training override YAML may contain only model and train mappings.")
    allowed = {
        "model": {"spatial_encoder", "spatial_layers", "dropout", "drop_edge_rate"},
        "train": {"batch_size", "lr", "weight_decay", "max_epochs",
                  "early_stopping_patience"},
    }
    effective = copy.deepcopy(config)
    for section, values in overrides.items():
        if not isinstance(values, dict) or set(values) - allowed[section]:
            raise ValueError(f"Unsupported keys in {section} training override.")
        effective.setdefault(section, {}).update(values)
    from config_validation import validate_config
    validate_config(effective)
    return effective


def _archive_production_checkpoints(paths: list[Path], output: Path) -> None:
    """Copy explicit production checkpoints to the new run before any work starts."""
    if not paths:
        raise ValueError("GNNv3 training requires an explicit production best checkpoint path.")
    output_resolved = output.resolve()
    archive = output_resolved / "production_model_backup"
    records = []
    for index, requested in enumerate(paths, start=1):
        source = requested.expanduser().resolve(strict=True)
        if not source.is_file() or output_resolved == source or output_resolved in source.parents:
            raise ValueError(f"Invalid production checkpoint path: {requested}")
        digest = sha256(source)
        destination = archive / f"{index:02d}-{source.name}"
        if destination.exists():
            if sha256(destination) != digest:
                raise RuntimeError(f"Existing production backup differs from source: {destination}")
        else:
            archive.mkdir(parents=True, exist_ok=True)
            atomic_bytes(destination, source.read_bytes())
            if sha256(destination) != digest:
                raise RuntimeError(f"Production checkpoint copy failed SHA256 verification: {destination}")
        related = []
        metadata_dir = archive / f"{index:02d}-metadata"
        related_candidates = [source.parent / filename for filename in (
            "effective_config.yaml", "resume_manifest.json", "training_complete.json",
            "training_history.csv", "test_metrics.json", "edge_scaler.json")]
        related_candidates.append(source.parent.parent.parent / "season_config.yaml")
        for related_source in related_candidates:
            if not related_source.is_file():
                continue
            related_destination = metadata_dir / related_source.name
            related_hash = sha256(related_source)
            if related_destination.exists():
                if sha256(related_destination) != related_hash:
                    raise RuntimeError(f"Existing production metadata backup differs: {related_destination}")
            else:
                metadata_dir.mkdir(parents=True, exist_ok=True)
                atomic_bytes(related_destination, related_source.read_bytes())
            related.append({"source": str(related_source), "backup": str(related_destination),
                            "sha256": related_hash, "bytes": related_source.stat().st_size})
        records.append({"source": str(source), "backup": str(destination),
                        "sha256": digest, "bytes": source.stat().st_size,
                        "related_metadata": related})
    atomic_json(archive / "production_checkpoint_manifest.json",
                {"schema_version": 1, "checkpoints": records})


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
        # Windows requires a writable descriptor for fsync/FlushFileBuffers.
        with temporary.open("rb+") as stream:
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
    from evaluate import compute_metrics

    rows = []
    for key, part in frame.groupby(group, dropna=False):
        y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
        rows.append({group: None if pd.isna(key) else key, **compute_metrics(y, p),
                     "rmse_sec": float(np.sqrt(np.mean(np.square(y - p))))})
    return rows


def run(args: argparse.Namespace) -> Path:
    out = args.output.expanduser().resolve() if args.output else ROOT / ".local/research-history/2023-fastf1-season"
    no_production_checkpoint = False
    reuse_arg = getattr(args, "reuse_prepared_from", None)
    if reuse_arg is not None:
        source_root_arg = reuse_arg.expanduser().resolve(strict=True)
        if out == source_root_arg or out in source_root_arg.parents or source_root_arg in out.parents:
            raise ValueError("GNNv3 output and reused source run must be separate, disjoint directories.")
        if out.exists() and any(out.iterdir()):
            raise ValueError("GNNv3 output must be new or empty to protect existing run artifacts.")
    if getattr(args, "training_overrides", None) is not None:
        if reuse_arg is None:
            raise ValueError("GNNv3 training requires --reuse-prepared-from and a separate new --output.")
        best_checkpoint = getattr(args, "preserve_production_checkpoint", None)
        no_production_checkpoint = bool(getattr(args, "no_production_checkpoint", False))
        if best_checkpoint is None and not no_production_checkpoint:
            raise ValueError("Select --preserve-production-checkpoint or explicitly confirm --no-production-checkpoint.")
        production_paths = [getattr(args, "preserve_production_checkpoint", None),
                            getattr(args, "preserve_production_last_checkpoint", None)]
        best_path = production_paths[0]
        if best_path is not None and production_paths[1] is None:
            sibling_last = best_path.expanduser().with_name("best.last.pt")
            if sibling_last.is_file():
                production_paths[1] = sibling_last
        paths_to_archive = [path for path in production_paths if path is not None]
        if paths_to_archive:
            _archive_production_checkpoints(paths_to_archive, out)
    local_data_only = bool(getattr(args, "local_data_only", False))
    secrets = ("", "") if local_data_only else load_credentials()
    # Keep credentials out of worker environments in either source mode. The
    # offline mode neither reads nor uses them.
    for name in ("F1_USERNAME", "F1_PASSWORD", "FASTF1_AUTH_METHOD"):
        os.environ.pop(name, None)
    fastf1 = None
    if not local_data_only:
        # ensure_raw_stage exposes credentials briefly around serialized loads.
        import fastf1
        from packaging.version import Version
        if Version(fastf1.__version__) < Version("3.9.0"):
            raise RuntimeError("FastF1 account authentication requires 3.9.0+. Run with: uv run --frozen --with fastf1==3.9.0 python hammertime_100/run_2023_fastf1_season.py")
        source_version = fastf1.__version__
    else:
        source_version = "local-data-only"
    reuse_root = (args.reuse_prepared_from.expanduser().resolve(strict=True)
                  if getattr(args, "reuse_prepared_from", None) is not None else out)
    if reuse_root == out or out in reuse_root.parents:
        raise ValueError("Training output must be separate from and outside the reused cache root.")
    out.mkdir(parents=True, exist_ok=True)
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT = out
    (out / "training").mkdir(parents=True, exist_ok=True)
    if reuse_root == out:
        (out / "race_cache/sessions").mkdir(parents=True, exist_ok=True)
        (out / "cohorts").mkdir(parents=True, exist_ok=True)
        if not local_data_only:
            (out / "race_cache/raw_sessions").mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
                        format="%(asctime)s %(levelname)s %(message)s")
    if no_production_checkpoint:
        LOG.info("No production checkpoint was supplied for preservation; run explicitly marked --no-production-checkpoint.")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    if not local_data_only:
        logging.getLogger("fastf1").setLevel(logging.INFO)
    cache_root = HERE / "data/cache/fastf1-2023-authenticated"
    if not local_data_only:
        cache_root.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_root))
    failure_stage = "load template and discover or validate source data"
    run_started = time.perf_counter()
    try:
        template_path = Path(args.config) if getattr(args, "config", None) is not None else HERE / "configs/supplied_fastf1_monza_2023_gated.yaml"
        with template_path.open(encoding="utf-8") as stream:
            base = yaml.safe_load(stream)
        log_every_batches = getattr(args, "log_every_batches", None)
        if log_every_batches is None:
            log_every_batches = base.get("train", {}).get("log_every_batches", 10)
        if (isinstance(log_every_batches, bool) or not isinstance(log_every_batches, int)
                or log_every_batches < 1):
            raise ValueError("train.log_every_batches must be a positive integer.")
        # Batch-report cadence does not alter preparation or training results.
        base["train"].pop("log_every_batches", None)
        base["data"]["split_method"] = "race"
        base["data"]["random_seed"] = args.seed
        base["train"]["max_epochs"] = args.epochs
        if getattr(args, "batch_size", None) is not None:
            base["train"]["batch_size"] = int(args.batch_size)
        if getattr(args, "window_stride", None) is not None:
            base["data"]["window_stride"] = int(args.window_stride)
        if getattr(args, "patience", None) is not None:
            base["train"]["early_stopping_patience"] = int(args.patience)
        if getattr(args, "gat_type", None) is not None:
            base["model"]["gat_type"] = str(args.gat_type)
        if getattr(args, "lr_scheduler", None) is not None:
            base["train"]["lr_scheduler"] = str(args.lr_scheduler)
        if getattr(args, "node_schema", None) is not None:
            base["graph"]["node_feature_schema"] = str(args.node_schema)
            base["model"]["in_dim"] = 20 if args.node_schema == "circuit_context_v2" else (19 if args.node_schema == "circuit_context_v1" else 9)
        if getattr(args, "edge_schema", None) is not None:
            base["graph"]["edge_feature_schema"] = str(args.edge_schema)
            base["model"]["edge_dim"] = 5 if args.edge_schema == "physical_v2" else (4 if args.edge_schema == "physical_v1" else 1)
        if getattr(args, "neighbor_mode", None) is not None:
            base["graph"]["neighbor_mode"] = str(args.neighbor_mode)
        base["tuning"]["enabled"] = False
        if local_data_only:
            raw_root_arg = getattr(args, "raw_sessions_dir", None)
            if raw_root_arg is None:
                raise ValueError("--local-data-only requires --raw-sessions-dir pointing at the transferred raw_sessions directory.")
            raw_root = Path(raw_root_arg).expanduser().resolve(strict=True)
            inventory_started = time.perf_counter()
            events, raw_cached, round_scheme = load_local_raw_inventory(raw_root)
            LOG.info("Verified raw inventory and SHA256 manifests in %.1f seconds.",
                     time.perf_counter() - inventory_started)
            skips = []
            LOG.info("Verified complete local inventory: %d of %d authentic 2023 race sessions; round scheme %s.",
                     len(events), len(EXPECTED_2023_RACE_IDS), round_scheme)
        else:
            os.environ["F1_USERNAME"], os.environ["F1_PASSWORD"] = secrets
            os.environ["FASTF1_AUTH_METHOD"] = "password"
            try:
                events, skips = discover(fastf1, out)
            finally:
                os.environ.pop("F1_USERNAME", None)
                os.environ.pop("F1_PASSWORD", None)
                os.environ.pop("FASTF1_AUTH_METHOD", None)
            raw_root = reuse_root / "race_cache/raw_sessions"
        failure_stage = "acquire and normalize race sessions"
        stages_root = reuse_root / "race_cache/sessions"
        if not local_data_only:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                raw_cached = list(pool.map(
                    lambda event: _valid_raw_stage(raw_root / _safe_id(event["race_id"]),
                                                   _raw_fingerprint(event, fastf1.__version__)), events))

        failures: list[dict] = list(skips)
        normalized_by_id: dict[str, dict] = {}
        future_to_id = {}
        process_pool = None
        pool_start_failure = None
        available_memory = _available_memory_bytes()
        process_workers = _worker_memory_limit(args.processing_workers, available_memory)
        if available_memory is None:
            LOG.warning("Available-memory probe failed; limiting normalization to one process.")
        else:
            LOG.info("Available physical memory=%.1f GiB.", available_memory / 1024**3)
        LOG.info("Normalization workers requested=%d effective=%d.",
                 args.processing_workers, process_workers)
        require_cache_hit = (getattr(args, "require_existing_caches", False)
                             or reuse_root != out)
        if require_cache_hit:
            missing = []
            for event, raw_manifest in zip(events, raw_cached):
                race_id = str(event["race_id"])
                if raw_manifest is None:
                    missing.append(race_id + " (raw stage)")
                    continue
                merged_event = {**event, **raw_manifest["event"]}
                source_version_for_cache = _normalization_source_version(
                    raw_manifest, local_data_only,
                    source_version if not local_data_only else None)
                expected = _normalization_fingerprint(
                    merged_event, raw_manifest, base, source_version_for_cache,
                    sha256(raw_root / _safe_id(race_id) / "raw_manifest.json"))
                if _valid_race_stage(stages_root / _safe_id(race_id), expected) is None:
                    missing.append(race_id + " (normalized stage)")
            if missing:
                raise RuntimeError("--require-existing-caches: exact prepared inputs are missing: "
                                   + ", ".join(missing))
        normalization_started = time.perf_counter()
        try:
            for event, raw_manifest in zip(events, raw_cached):
                race_id = str(event["race_id"])
                if raw_manifest is None and local_data_only:
                    raise RuntimeError(f"Verified local raw session disappeared for {race_id}; refusing network fallback.")
                if raw_manifest is None:
                    raw_event, failure_type = ensure_raw_stage(event, cache_root, raw_root,
                                                               fastf1.__version__, secrets)
                    if raw_event is None:
                        failures.append({"race_id": race_id, "exception_type": failure_type})
                        LOG.warning("FastF1 source acquisition failed for %s (%s)", race_id, failure_type)
                        continue
                    event = raw_event
                    raw_stage = raw_root / _safe_id(race_id)
                    raw_manifest = json.loads((raw_stage / "raw_manifest.json").read_text(encoding="utf-8"))
                else:
                    event = {**event, **raw_manifest["event"]}
                    raw_stage = raw_root / _safe_id(race_id)

                if local_data_only:
                    # Recheck immediately before normalization to guard against
                    # changes between inventory validation and worker launch.
                    raw_check_started = time.perf_counter()
                    raw_manifest = _validate_local_raw_manifest(
                        raw_stage, race_id)
                    LOG.info("Raw SHA256 verification for %s took %.1f seconds.",
                             race_id, time.perf_counter() - raw_check_started)

                event_source_version = _normalization_source_version(
                    raw_manifest, local_data_only, source_version if not local_data_only else None)
                fingerprint = _normalization_fingerprint(
                    event, raw_manifest, base, event_source_version,
                    sha256(raw_stage / "raw_manifest.json"))
                normalized_stage = stages_root / _safe_id(race_id)
                cached = _valid_race_stage(normalized_stage, fingerprint)
                if cached is not None:
                    normalized_by_id[race_id] = {**event, **cached["event"]}
                    continue

                # Start all requested, memory-safe workers before more parent
                # FastF1 calls. Workers inherit no F1 account variables.
                if process_pool is None:
                    if pool_start_failure is not None:
                        failures.append({"race_id": race_id, "exception_type": pool_start_failure})
                        continue
                    LOG.info("Starting %d CPU normalization workers (requested %d; memory-capped)",
                             process_workers, args.processing_workers)
                    try:
                        process_pool, worker_pids = _start_worker_pool(process_workers)
                    except Exception as exc:
                        pool_start_failure = type(exc).__name__
                        failures.append({"race_id": race_id, "exception_type": pool_start_failure})
                        LOG.warning("Could not start normalization workers (%s)", pool_start_failure)
                        continue
                    LOG.info("CPU normalization worker processes started: %d", len(worker_pids))
                task = _normalization_task(event, raw_stage, raw_manifest, base,
                                           event_source_version, stages_root,
                                           _normalization_source_metadata(
                                               raw_manifest, local_data_only,
                                               source_version if not local_data_only else None,
                                               round_scheme if local_data_only else None))
                try:
                    future_to_id[process_pool.submit(normalize_race_worker, task)] = race_id
                except Exception as exc:
                    failures.append({"race_id": race_id, "exception_type": type(exc).__name__})

            if process_pool is not None:
                completed, normalization_failures = _collect_normalization_futures(future_to_id)
                normalized_by_id.update(completed)
                failures.extend({"race_id": race_id, "exception_type": reason}
                                 for race_id, reason in normalization_failures.items())
                for race_id, reason in normalization_failures.items():
                    LOG.warning("CPU normalization failed for %s (%s)", race_id, reason)
                for race_id in completed:
                    LOG.info("Completed normalized session %s", race_id)
        finally:
            if process_pool is not None:
                process_pool.shutdown(wait=True, cancel_futures=False)
        LOG.info("Normalization and initial cache verification completed in %.1f seconds.",
                 time.perf_counter() - normalization_started)
        LOG.info("Resources after normalization: %s", _resource_summary())

        # A worker can commit its stage and die before returning its result.
        # Revalidate all expected fingerprints so those completed calculations
        # survive a broken pool and failures remain isolated per session.
        revalidation_started = time.perf_counter()
        for event in events:
            race_id = str(event["race_id"])
            try:
                raw_manifest = json.loads((raw_root / _safe_id(race_id) / "raw_manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError, KeyError):
                continue
            event = {**event, **raw_manifest["event"]}
            event_source_version = _normalization_source_version(
                raw_manifest, local_data_only, source_version if not local_data_only else None)
            raw_stage = raw_root / _safe_id(race_id)
            fingerprint = _normalization_fingerprint(
                event, raw_manifest, base, event_source_version,
                sha256(raw_stage / "raw_manifest.json"))
            stage_result = _valid_race_stage(stages_root / _safe_id(race_id), fingerprint)
            if stage_result is not None:
                normalized_by_id[race_id] = {**event, **stage_result["event"]}
        LOG.info("Post-normalization source and stage verification completed in %.1f seconds.",
                 time.perf_counter() - revalidation_started)
        LOG.info("Resources after cache verification: %s", _resource_summary())
        good = [normalized_by_id[event["race_id"]] for event in events
                if event["race_id"] in normalized_by_id]
        good_ids = set(normalized_by_id)
        failures = [failure for failure in failures if failure.get("race_id") not in good_ids]
        atomic_json(out / "failed_sessions.json", failures)
        atomic_json(out / "usable_sessions.json", good)
        if local_data_only and len(good) != len(EXPECTED_2023_RACE_IDS):
            missing = [event["race_id"] for event in events if event["race_id"] not in good_ids]
            raise RuntimeError(
                f"Local-data-only preparation requires all {len(EXPECTED_2023_RACE_IDS)} races; "
                f"{len(good)} passed existing normalization gates. Failed races: {', '.join(missing)}. "
                "No partial-season training was started; see failed_sessions.json for sanitized reasons."
            )
        if len(good) < 3:
            raise RuntimeError(f"Only {len(good)} authentic, gate-passing races; existing race-held-out training requires at least three.")

        base["sessions"] = good
        base["session"].update(good[0])
        base["session"]["race_id"] = good[0]["race_id"]
        if not local_data_only:
            base["session"]["cache_dir"] = str(cache_root)
        base["session"]["processed_dir"] = str(reuse_root / "race_cache/processed")
        base["track"]["lap_length_m"] = good[0]["lap_length_m"]
        base["train"]["checkpoint_path"] = str(reuse_root / "training" / "pending" / "best.pt")
        base["train"]["history_path"] = str(reuse_root / "training" / "pending" / "training_history.csv")
        _align_cohort_num_nodes(base, good, stages_root)
        config_path = out / "season_config.yaml"
        atomic_bytes(config_path, yaml.safe_dump(base, sort_keys=False).encode())
        cohort_started = time.perf_counter()
        source_root = reuse_root / "race_cache"
        source_config_path = config_path
        if reuse_root != out:
            old_source_manifest = json.loads((source_root / "normalized_source_manifest.json").read_text(encoding="utf-8"))
            source_config_path = Path(old_source_manifest["config_path"]).expanduser().resolve(strict=True)
            if sha256(source_config_path) != old_source_manifest.get("config_sha256"):
                raise RuntimeError("Reused source config no longer matches its normalized-source manifest.")
        source_hash = write_normalized_source(
            stages_root, source_root, source_config_path, good,
            cache_root if not local_data_only else None, persist=(reuse_root == out))
        if reuse_root != out and sha256(source_root / "normalized_source_manifest.json") != source_hash:
            raise RuntimeError("Read-only normalized-source verification failed; source cache changed or is incompatible.")
        failure_stage = "validate or prepare graph cohort"
        cohort_fingerprint = _cohort_fingerprint(base, source_hash)
        cohort_prefix = reuse_root / "cohorts" / cohort_fingerprint[:20]
        candidates = [cohort_prefix, *sorted((reuse_root / "cohorts").glob(f"{cohort_fingerprint[:20]}-rebuild*"))]
        cached = None
        cohort_dir = cohort_prefix
        for candidate in candidates:
            if getattr(args, "preflight_only", False):
                cached = _cohort_manifest_cached(candidate, base, source_hash)
            else:
                cached = _cohort_cached(candidate, base, source_hash)
            if cached is not None:
                cohort_dir = candidate
                break
        cohort_cache_hit = cached is not None
        if cached is None and (getattr(args, "require_existing_caches", False)
                               or reuse_root != out):
            raise RuntimeError("--require-existing-caches: exact prepared cohort is absent; refusing preparation.")
        if cached is None:
            failure_stage = "prepare graph cohort from normalized race tables"
            if cohort_prefix.exists():
                # Keep a corrupt/incompatible artifact for investigation and
                # build a new immutable output directory.
                suffix = 1
                while (reuse_root / "cohorts" / f"{cohort_fingerprint[:20]}-rebuild{suffix}").exists():
                    suffix += 1
                cohort_dir = reuse_root / "cohorts" / f"{cohort_fingerprint[:20]}-rebuild{suffix}"
            from prepare_cohort import prepare_cohort
            prepare_cohort(config_path, cohort_dir, normalized_source_dir=source_root)
            cached = _cohort_cached(cohort_dir, base, source_hash)
            if cached is None:
                raise RuntimeError("Prepared cohort failed its saved manifest/config/source checks.")
        LOG.info("Cohort lookup/preparation completed in %.1f seconds (cache %s).",
                 time.perf_counter() - cohort_started,
                 "hit" if cohort_cache_hit else "miss")
        LOG.info("Resources after cohort preparation: %s", _resource_summary())
        if getattr(args, "preflight_only", False):
            manifest = cached
            if manifest["provider"] != "fastf1" or len(manifest.get("sessions", [])) != len(EXPECTED_2023_RACE_IDS):
                raise ValueError("Prepared cohort preflight requires the complete verified 22-race FastF1 cohort.")
            import torch
            from model import create_model

            cfg = copy.deepcopy(base)
            if getattr(args, "training_overrides", None) is not None:
                cfg = _load_training_overrides(cfg, args.training_overrides.expanduser().resolve(strict=True))
            torch.set_num_threads(args.threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            model = create_model(cfg)
            LOG.info("PRE-FLIGHT PASSED (no cohort unpickle, no training, no preprocessing): "
                     "source manifests and normalized tables verified; prepared artifact SHA256 verified; "
                     "22-race cohort config/split/scalers match; candidate model constructed (%d parameters).",
                     sum(parameter.numel() for parameter in model.parameters()))
            return out
        payload, manifest = cached
        if manifest["provider"] != "fastf1" or len(manifest["sessions"]) < 3:
            raise ValueError("Prepared cohort provider/session-count invariant failed")
        import torch
        from dataset import create_dataloaders
        from evaluate import (compute_metrics, predict_test_set, validate_checkpoint_edge_scaler,
                              validate_clean_air_physics, validate_drs_train_physics)
        from model import create_model
        from prepare_cohort import clone_snapshots
        from train import train_model

        cfg = copy.deepcopy(payload["config"])
        if getattr(args, "training_overrides", None) is not None:
            cfg = _load_training_overrides(cfg, args.training_overrides.expanduser().resolve(strict=True))
            LOG.info("Applied training-only overlay after verified prepared-cohort load: %s",
                     args.training_overrides)
        cfg["train"]["log_every_batches"] = log_every_batches
        cfg["train"]["checkpoint_path"] = str(out / "training" / "_path_placeholder" / "best.pt")
        cfg["train"]["history_path"] = str(out / "training" / "_path_placeholder" / "training_history.csv")
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            LOG.warning("PyTorch inter-op threads were already initialized; retaining their current setting.")
        code_inputs = {name: sha256(HERE / "src" / name) for name in ("train.py", "model.py", "dataset.py")}
        train_fingerprint = canonical_hash({"cohort_artifact": manifest["prepared_artifact_sha256"],
                                            "config": _training_identity_config(cfg), "code": code_inputs,
                                            "fastf1": source_version, "source_mode": "local-data-only" if local_data_only else "online",
                                            "device": device.type,
                                            "threads": args.threads})
        training_dir = out / "training" / train_fingerprint[:20]
        training_dir.mkdir(parents=True, exist_ok=True)
        cfg["train"]["checkpoint_path"] = str(training_dir / "best.pt")
        cfg["train"]["last_checkpoint_path"] = str(training_dir / "best.last.pt")
        cfg["train"]["history_path"] = str(training_dir / "training_history.csv")
        atomic_bytes(training_dir / "effective_config.yaml",
                     yaml.safe_dump(cfg, sort_keys=False).encode())
        expected_resume_manifest = {
            "fingerprint": train_fingerprint,
            "prepared_artifact_sha256": manifest["prepared_artifact_sha256"],
            "prepared_manifest_sha256": manifest["prepared_manifest_sha256"],
            "config_sha256": canonical_hash(_training_identity_config(cfg)),
            "implementation_sha256": code_inputs,
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
            failure_stage = "train or resume model"
            snapshots_to_load = payload.pop("snapshots")
            labels_to_load = payload.pop("labels")
            train_loader, val_loader, test_loader, scaler = create_dataloaders(
                snapshots_to_load, labels_to_load, payload["driver_to_idx"], cfg)
            LOG.info("Training inputs: %d snapshots; windows train=%d validation=%d test=%d; %s",
                     len(snapshots_to_load), len(train_loader.dataset), len(val_loader.dataset),
                     len(test_loader.dataset), _resource_summary())
            del snapshots_to_load, labels_to_load
            import gc
            gc.collect()
            model = create_model(cfg).to(device)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            resume_from = last_path if last_path.is_file() else None
            training_started = time.perf_counter()
            best_val, history = train_model(model, train_loader, val_loader, cfg, device, resume_from=resume_from)
            LOG.info("Model fit/resume completed in %.1f seconds.",
                     time.perf_counter() - training_started)
            LOG.info("Resources after model fit: %s", _resource_summary())
            hist = pd.DataFrame(history)
            _atomic_csv(hist, Path(cfg["train"]["history_path"]))
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
        failure_stage = "held-out prediction and export"
        predictions_path = training_dir / "test_predictions.pkl"
        def predict_frame() -> pd.DataFrame:
            nonlocal loaders
            if loaders is None:
                snapshots_to_eval = payload.pop("snapshots") if "snapshots" in payload else payload["snapshots"]
                labels_to_eval = payload.pop("labels") if "labels" in payload else payload["labels"]
                _, _, test_loader, scaler = create_dataloaders(
                    snapshots_to_eval, labels_to_eval, payload["driver_to_idx"], cfg)
                del snapshots_to_eval, labels_to_eval
                import gc
                gc.collect()
            else:
                _, _, test_loader, scaler = loaders
            model = create_model(cfg).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            validate_checkpoint_edge_scaler(checkpoint, model, test_loader)
            saved = checkpoint.get("feature_scaler") or {}
            if not np.allclose(scaler.mean_, saved.get("mean_", [])) or not np.allclose(scaler.scale_, saved.get("scale_", [])):
                raise ValueError("Best checkpoint scaler does not match train-only prepared-cohort scaler")
            _, pred, records = predict_test_set(model, test_loader, device)
            return pd.DataFrame(records)
        frame = load_or_compute_predictions(predictions_path, pred_fingerprint, predict_frame)
        del loaders
        import gc
        gc.collect()
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
        LOG.info("Season runner completed in %.1f seconds.", time.perf_counter() - run_started)
        return out
    except Exception as exc:
        details = _sanitized_error(exc, secrets)
        details["stage"] = failure_stage
        atomic_json(out / "run_failure.json", details)
        raise RuntimeError(f"2023 FastF1 season run failed ({type(exc).__name__}); see {out} for sanitized diagnostics.") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Reusable run workspace; default .local/research-history/2023-fastf1-season.")
    parser.add_argument("--local-data-only", action="store_true",
                        help="Use only a complete, hash-verified local inventory of all 22 authentic 2023 raw race stages; never authenticate, discover events, or download source data.")
    parser.add_argument("--raw-sessions-dir", type=Path, default=None,
                        help="Path to the transferred race_cache/raw_sessions directory (required with --local-data-only).")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=2,
                        help="Bounded raw-session validation workers in online mode (1-8); FastF1 downloads stay serialized.")
    parser.add_argument("--processing-workers", type=int, default=4,
                        help="Independent CPU-only race normalization processes (1-8), memory-capped.")
    parser.add_argument("--torch-threads", "--threads", dest="threads", type=int, default=1,
                        help="PyTorch intra-op CPU threads.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size (default: config batch size, e.g. 128 or 256).")
    parser.add_argument("--log-every-batches", type=int, default=None,
                        help="Override training progress interval in batches (default: config value, otherwise 10).")
    parser.add_argument("--window-stride", type=int, default=1,
                        help="Temporal window subsampling stride (default: 1; e.g. 2 for 0.5s stride).")
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional YAML config path override (default: configs/supplied_fastf1_monza_2023_gated.yaml).")
    parser.add_argument("--training-overrides", type=Path, default=None,
                        help="Model/train-only YAML applied after loading the exact base-config cohort cache.")
    parser.add_argument("--reuse-prepared-from", type=Path, default=None,
                        help="Read-only root of a prior run's exact raw/normalized/prepared cache tree.")
    production_group = parser.add_mutually_exclusive_group()
    production_group.add_argument("--preserve-production-checkpoint", type=Path, default=None,
                                  help="Explicit production best.pt to copy and SHA256-verify before GNNv3 work.")
    production_group.add_argument("--no-production-checkpoint", action="store_true",
                                  help="Explicitly state that this isolated candidate run has no production checkpoint to preserve.")
    parser.add_argument("--preserve-production-last-checkpoint", type=Path, default=None,
                        help="Optional sibling best.last.pt checkpoint to preserve as well.")
    parser.add_argument("--require-existing-caches", action="store_true",
                        help="Fail before preparation/training if exact normalized race caches or cohort are missing.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Verify exact source/cohort artifacts and construct the selected model, then exit without unpickling the cohort or training.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience in epochs.")
    parser.add_argument("--gat-type", choices=("gat", "gatv2"), default=None,
                        help="GAT convolution layer type ('gat' or 'gatv2').")
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default=None,
                        help="Learning rate scheduler ('none' or 'plateau').")
    parser.add_argument("--node-schema", choices=("legacy_v1", "circuit_context_v1", "circuit_context_v2"), default=None,
                        help="Graph node feature schema ('legacy_v1' [9-dim], 'circuit_context_v1' [19-dim], or 'circuit_context_v2' [20-dim]).")
    parser.add_argument("--edge-schema", choices=("scalar_rbf", "physical_v1", "physical_v2"), default=None,
                        help="Graph edge feature schema ('scalar_rbf' [1-dim], 'physical_v1' [4-dim], or 'physical_v2' [5-dim]).")
    parser.add_argument("--neighbor-mode", choices=("race_progress", "physical_circuit"), default=None,
                        help="Graph neighbor connection mode ('race_progress' or 'physical_circuit').")
    args = parser.parse_args()
    if (args.epochs < 1 or args.seed < 0 or not 1 <= args.workers <= 8
            or not 1 <= args.processing_workers <= 8 or args.threads < 1
            or (args.batch_size is not None and args.batch_size < 1)
            or (args.log_every_batches is not None and args.log_every_batches < 1)
            or args.window_stride < 1
            or (args.patience is not None and args.patience < 1)):
        parser.error("--epochs, --threads, and any supplied logging interval must be positive, --seed non-negative, worker counts between 1 and 8, and positive batch-size/window-stride/patience")
    if args.local_data_only and args.raw_sessions_dir is None:
        parser.error("--local-data-only requires --raw-sessions-dir.")
    if args.raw_sessions_dir is not None and not args.local_data_only:
        parser.error("--raw-sessions-dir is used only with --local-data-only.")
    try:
        print(run(args))
    except Exception as exc:
        logging.error("2023 FastF1 season run failed (%s)", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
