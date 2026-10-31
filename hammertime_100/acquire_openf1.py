"""Acquire raw OpenF1 tables for configured sessions without preprocessing or training."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from unittest.mock import patch

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config_validation import validate_config  # noqa: E402
from openf1_client import load_openf1_tables  # noqa: E402
from run_pipeline import PROJECT_DIR as PIPELINE_PROJECT_DIR  # noqa: E402
from run_pipeline import _safe_component, _session_specs  # noqa: E402


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _config_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def _cache_root(spec: dict[str, Any], config: dict[str, Any]) -> Path:
    cache_dir = Path(spec.get("cache_dir", config["session"]["cache_dir"])).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = PIPELINE_PROJECT_DIR / cache_dir
    return cache_dir.resolve() / _safe_component(spec["race_id"])


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_for(config_path: Path, spec: dict[str, Any], config: dict[str, Any],
                  bundle: dict[str, Any]) -> dict[str, Any]:
    session_config = copy.deepcopy(config)
    session_config["session"].update(spec)
    session_config["session"]["race_id"] = spec["race_id"]
    provenance = bundle["provenance"]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "race_id": spec["race_id"],
        "provider": "openf1",
        "provider_base_url": config["openf1"]["base_url"],
        "session": bundle["session"],
        "session_config_sha256": _canonical_hash(session_config["session"]),
        "source_config_path": str(config_path),
        "source_config_sha256": _sha256_bytes(config_path.read_bytes()),
        "request_settings": {
            "chunk_seconds": config["openf1"]["chunk_seconds"],
            "request_interval_sec": config["openf1"]["request_interval_sec"],
            "fetch_location": config["openf1"]["fetch_location"],
            "max_attempts": config["openf1"]["max_attempts"],
        },
        "table_row_counts": {name: int(len(frame))
                             for name, frame in bundle["tables"].items()},
        "raw_artifacts": bundle["raw_artifacts"],
        "request_provenance": provenance,
    }


def _write_manifest_once(path: Path, manifest: dict[str, Any]) -> Path:
    """Write a manifest without replacing an existing user artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    existing = path
    suffix = 1
    while existing.exists():
        try:
            old = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if (old is not None
                and old.get("race_id") == manifest["race_id"]
                and old.get("request_provenance") == manifest["request_provenance"]
                and old.get("table_row_counts") == manifest["table_row_counts"]):
            return existing
        existing = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
        suffix += 1

    fd, temp_name = tempfile.mkstemp(prefix=f".{existing.name}.", dir=existing.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.link(temp_name, existing)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return existing


def _configured_sessions(config: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    specs = _session_specs(config)
    mixed = [spec["race_id"] for spec in specs if spec.get("source", "fastf1") != "openf1"]
    if mixed:
        raise ValueError(
            "The OpenF1 acquisition command accepts OpenF1-only cohorts; "
            f"non-OpenF1 sessions were configured: {', '.join(mixed)}."
        )
    if requested:
        by_id = {spec["race_id"]: spec for spec in specs}
        unknown = [race_id for race_id in requested if race_id not in by_id]
        if unknown:
            raise ValueError(f"Unknown configured race_id(s): {', '.join(unknown)}.")
        return [by_id[race_id] for race_id in requested]
    return specs


def _offline_cache_miss(url: str, *args: Any, **kwargs: Any) -> None:
    raise RuntimeError(f"offline cache miss; network request blocked for {url}")


def acquire(config_path: Path, requested_race_ids: list[str] | None = None,
            offline: bool = False) -> list[Path]:
    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration is not a mapping: {config_path}")
    validate_config(config)
    specs = _configured_sessions(config, requested_race_ids)
    written: list[Path] = []

    for spec in specs:
        session_config = copy.deepcopy(config)
        session_config["session"].update(spec)
        session_config["session"]["race_id"] = spec["race_id"]
        cache_dir = _cache_root(spec, config)
        session_config["session"]["cache_dir"] = str(cache_dir)

        if offline:
            with patch("openf1_client.requests.get", side_effect=_offline_cache_miss):
                bundle = load_openf1_tables(session_config)
        else:
            bundle = load_openf1_tables(session_config)
        manifest = _manifest_for(config_path, spec, session_config, bundle)
        manifest_path = _write_manifest_once(cache_dir / "openf1_acquisition_manifest.json", manifest)
        written.append(manifest_path)
        print(
            f"{spec['race_id']}: OpenF1 tables acquired/cached; "
            f"rows={manifest['table_row_counts']}; requests={len(manifest['request_provenance'])}; "
            f"manifest={manifest_path}"
        )
        del bundle

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/openf1_2023_race_holdout.yaml",
        help="OpenF1-only HammerTime YAML config (default: configs/openf1_2023_race_holdout.yaml).",
    )
    parser.add_argument(
        "--race-id", action="append", dest="race_ids",
        help="Acquire only this configured race_id; may be repeated. Default: all configured sessions.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Require exact cached OpenF1 queries and block network requests.",
    )
    args = parser.parse_args()
    try:
        acquire(_config_path(args.config), args.race_ids, offline=args.offline)
    except Exception as exc:  # Keep the CLI boundary readable while preserving the failure.
        parser.exit(2, f"OpenF1 acquisition failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
