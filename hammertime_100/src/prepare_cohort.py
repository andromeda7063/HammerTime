"""Prepare an isolated, provenance-frozen multi-session graph dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import pickle
import platform
import subprocess
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dataset import create_dataloaders
from graph import build_race_graph_snapshots, remap_cohort_driver_nodes
from context_features import (
    apply_context_calibration, feature_schema_metadata, fit_context_calibration,
    node_feature_schema, prepare_context_features,
)
from run_pipeline import PROJECT_DIR, _load_configured_race, _safe_component, _session_specs
from target import build_targets

logger = logging.getLogger("hammertime.prepare_cohort")


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment() -> dict:
    packages = {}
    for package in ("fastf1", "numpy", "pandas", "torch", "torch-geometric", "scikit-learn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def _git_revision() -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(PROJECT_DIR.parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _relative_or_absolute(path: Path, staging_dir: Path, output_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return str(output_dir / resolved.relative_to(staging_dir.resolve()))
    except ValueError:
        return str(resolved)


def _membership(loader) -> list[dict]:
    return [
        {
            "race_id": sample.get("race_id"),
            "input_start_time": float(sample["input_start_time"]),
            "target_time": float(sample["target_time"]),
        }
        for sample in loader.dataset.samples
    ]


def _training_input_frame_keys(loader) -> list[dict]:
    """Return unique (race, time) keys from input windows in the train partition."""
    keys = set()
    for sample in loader.dataset.samples:
        for snapshot in sample["snapshots"]:
            keys.add((str(snapshot.race_id), round(float(snapshot.time_sec), 9)))
    return [{"race_id": race_id, "time_sec": time_sec} for race_id, time_sec in sorted(keys)]


def _target_signature(loader) -> str:
    rows = []
    for sample in loader.dataset.samples:
        rows.append({
            "race_id": sample.get("race_id"),
            "input_start_time": float(sample["input_start_time"]),
            "target_time": float(sample["target_time"]),
            "driver_names": sample.get("driver_names"),
            "target": sample["target"].detach().cpu().numpy().astype(float).tolist(),
            "mask": sample["mask"].detach().cpu().numpy().astype(bool).tolist(),
        })
    return _sha256_bytes(_canonical_json(rows))


def _build_snapshots_by_race(aligned: pd.DataFrame, config: dict, specs: list[dict]):
    snapshots, driver_maps = [], {}
    for spec in specs:
        race_id = str(spec["race_id"])
        race_frame = aligned.loc[aligned["RaceId"].astype(str) == race_id].copy()
        if race_frame.empty:
            raise ValueError(f"No aligned rows remain for configured race {race_id!r}.")
        race_config = copy.deepcopy(config)
        race_config["session"].update(spec)
        race_config["session"]["race_id"] = race_id
        if "lap_length_m" in spec:
            race_config["track"]["lap_length_m"] = float(spec["lap_length_m"])
        race_snaps, driver_map = build_race_graph_snapshots(race_frame, race_config)
        snapshots.extend(race_snaps)
        driver_maps[race_id] = driver_map
    if len(driver_maps) > 1:
        remap_cohort_driver_nodes(snapshots, driver_maps, int(config["graph"]["num_nodes"]))
        all_drivers = sorted({str(driver) for mapping in driver_maps.values() for driver in mapping})
        driver_to_idx = {driver: index for index, driver in enumerate(all_drivers)}
    else:
        driver_to_idx = next(iter(driver_maps.values()))
    return snapshots, driver_to_idx


def _coverage(aligned: pd.DataFrame, labels: pd.DataFrame, snapshots: list) -> dict:
    valid = aligned.get("IsValid", pd.Series(False, index=aligned.index)).fillna(False).astype(bool)
    valid_rows = aligned.loc[valid]
    laps = (valid_rows.groupby(["RaceId", "Driver", "LapNumber"], dropna=True).ngroups
            if {"RaceId", "Driver", "LapNumber"}.issubset(valid_rows.columns) else 0)
    edge_total = sum(int(snapshot.edge_index.size(1)) for snapshot in snapshots)
    return {
        "aligned_rows": int(len(aligned)),
        "valid_aligned_rows": int(valid.sum()),
        "drivers": int(aligned["Driver"].nunique()) if "Driver" in aligned else 0,
        "valid_laps": int(laps),
        "target_rows": int(len(labels)),
        "target_drivers": int(labels["Driver"].nunique()) if "Driver" in labels else 0,
        "graph_snapshots": int(len(snapshots)),
        "mean_active_edges": (float(edge_total / len(snapshots)) if snapshots else 0.0),
    }


def _artifact_relative_hashes(paths: list[str], staging_dir: Path, output_dir: Path) -> dict:
    records = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Prepared source artifact is missing: {path}")
        records[_relative_or_absolute(path, staging_dir, output_dir)] = {
            "bytes": int(path.stat().st_size), "sha256": _sha256_file(path)
        }
    return records


def _provider_for_specs(specs: list[dict]) -> str:
    providers = {str(spec.get("source", "fastf1")).casefold() for spec in specs}
    if len(providers) != 1:
        raise ValueError("A prepared cohort must use one data provider; create separate FastF1 and OpenF1 cohorts.")
    provider = next(iter(providers))
    if provider not in ("fastf1", "openf1"):
        raise ValueError(f"Unsupported cohort provider {provider!r}.")
    return provider


def _read_aligned_source(source_dir: Path, specs: list[dict], provider: str):
    """Load already-normalized per-race tables from a checksum-verified cohort."""
    manifest_path = source_dir / "prepared_manifest.json"
    sidecar = Path(str(manifest_path) + ".sha256")
    if not manifest_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("--aligned-source must contain prepared_manifest.json and its SHA-256 sidecar.")
    manifest_bytes_hash = _sha256_file(manifest_path)
    if sidecar.read_text(encoding="utf-8").strip().split()[0] != manifest_bytes_hash:
        raise ValueError("Aligned-source manifest bytes do not match its SHA-256 sidecar.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical = _sha256_bytes(_canonical_json(
        {key: value for key, value in manifest.items() if key != "prepared_manifest_sha256"}
    ))
    if manifest.get("prepared_manifest_sha256") != canonical:
        raise ValueError("Aligned-source cohort manifest canonical hash is invalid.")
    if manifest.get("provider") != provider or manifest.get("providers") != [provider]:
        raise ValueError("Aligned-source provider must match the configured single-provider cohort.")
    upstream_config_path = Path(manifest.get("config_path", ""))
    if (not upstream_config_path.is_file()
            or _sha256_file(upstream_config_path) != manifest.get("config_sha256")):
        raise ValueError("Aligned-source config file is missing or disagrees with its recorded hash.")
    with upstream_config_path.open("r", encoding="utf-8") as source:
        upstream_config = yaml.safe_load(source)
    upstream_specs = _session_specs(upstream_config)
    expected_ids = [str(item["race_id"]) for item in specs]
    source_sessions = manifest.get("sessions", [])
    if [str(item.get("race_id")) for item in source_sessions] != expected_ids:
        raise ValueError("Aligned-source session identities/order do not exactly match the requested config.")
    if [str(item["race_id"]) for item in upstream_specs] != expected_ids:
        raise ValueError("Aligned-source sessions do not match the source config's ordered sessions.")
    hash_records = manifest.get("source_hashes", {})
    records, aligned_tables, labels_tables, paths = [], [], [], [str(manifest_path)]
    for spec, upstream_spec, record in zip(specs, upstream_specs, source_sessions):
        for key in ("provider", "year", "gp", "session_type", "session_key"):
            expected = (str(spec.get("source", "fastf1")) if key == "provider"
                        else spec.get(key, "R" if key == "session_type" else None))
            if record.get(key) != expected:
                raise ValueError(f"Aligned-source session {spec['race_id']} disagrees on {key}.")
            upstream_value = upstream_spec.get("source") if key == "provider" else upstream_spec.get(key)
            if upstream_value != expected:
                raise ValueError(f"Requested config disagrees with upstream source config on {key}.")
        if float(spec.get("lap_length_m", 0.0)) != float(upstream_spec.get("lap_length_m", 0.0)):
            raise ValueError(f"Requested lap_length_m for {spec['race_id']} differs from source config.")
        race_dir = source_dir / "sessions" / _safe_component(str(spec["race_id"]))
        aligned_path = race_dir / "aligned_telemetry.parquet"
        labels_path = race_dir / "target_labels.parquet"
        for path in (aligned_path, labels_path):
            if not path.is_file():
                raise FileNotFoundError(f"Aligned-source session table is missing: {path}")
            key_candidates = (str(path), str(path.resolve()), str(path.relative_to(source_dir)))
            expected_record = next((hash_records[key] for key in key_candidates if key in hash_records), None)
            if expected_record is None or expected_record.get("sha256") != _sha256_file(path):
                raise ValueError(f"Aligned-source table hash is missing or invalid: {path}")
            paths.append(str(path))
        aligned = pd.read_parquet(aligned_path)
        labels = pd.read_parquet(labels_path)
        if "RaceId" not in aligned or set(aligned["RaceId"].astype(str).unique()) != {str(spec["race_id"])}:
            raise ValueError(f"Aligned-source rows do not all belong to configured race {spec['race_id']!r}.")
        if "RaceId" not in labels or set(labels["RaceId"].astype(str).unique()) != {str(spec["race_id"])}:
            raise ValueError(f"Aligned-source target rows do not all belong to configured race {spec['race_id']!r}.")
        source_meta = record.get("source_metadata", {})
        actual_circuit = str(source_meta.get("session", {}).get("circuit_key", ""))
        configured_circuit = str(spec.get("circuit_id", ""))
        if configured_circuit and configured_circuit != actual_circuit:
            raise ValueError(f"Configured circuit_id for {spec['race_id']} does not match source circuit_key.")
        if not configured_circuit:
            spec["circuit_id"] = actual_circuit
        aligned_tables.append(aligned)
        labels_tables.append(labels)
        records.append({**record, "source_metadata": source_meta,
                        "coverage": record.get("coverage", {})})
    return aligned_tables, labels_tables, records, paths, manifest


def prepare_cohort(config_path: str | Path, output_dir: str | Path,
                   aligned_source_dir: str | Path | None = None) -> dict:
    """Run the configured acquisition/preprocessing path and freeze its graph data.

    The destination must not exist. All intermediate race artifacts are written
    under a sibling staging directory and become visible together when the
    completed cohort directory is atomically renamed into place.
    """
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Cohort output already exists; choose a new directory: {output_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Cohort config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    from config_validation import validate_config
    validate_config(config)
    specs = _session_specs(config)
    provider = _provider_for_specs(specs)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent) as temp:
        staging_dir = Path(temp)
        try:
            aligned_tables, label_tables, snapshots = [], [], []
            session_records, source_files = [], []
            source_manifest = None
            base_cache = Path(config["session"].get("cache_dir", "data/cache"))
            if not base_cache.is_absolute():
                base_cache = PROJECT_DIR / base_cache

            if aligned_source_dir is not None:
                source_root = Path(aligned_source_dir).resolve()
                (aligned_tables, label_tables, session_records, source_files,
                 source_manifest) = _read_aligned_source(source_root, specs, provider)
            else:
                for spec in specs:
                    race_id = str(spec["race_id"])
                    race_config = copy.deepcopy(config)
                    race_config["session"].update(spec)
                    race_config["session"]["race_id"] = race_id
                    session_cache = Path(spec.get("cache_dir", base_cache))
                    if not session_cache.is_absolute():
                        session_cache = PROJECT_DIR / session_cache
                    race_config["session"]["cache_dir"] = str(session_cache / _safe_component(race_id))
                    race_dir = staging_dir / "sessions" / _safe_component(race_id)
                    race_config["session"]["processed_dir"] = str(race_dir)
                    if "lap_length_m" in spec:
                        race_config["track"]["lap_length_m"] = float(spec["lap_length_m"])
                    logger.info("Preparing %s session %s", provider, race_id)
                    aligned, race_artifacts, source_metadata = _load_configured_race(spec, race_config)
                    if "RaceId" not in aligned:
                        aligned["RaceId"] = race_id
                    labels = build_targets(aligned, race_config)
                    if labels.empty:
                        raise ValueError(f"Session {race_id} produced no target labels.")
                    race_dir.mkdir(parents=True, exist_ok=True)
                    aligned_path = race_dir / "aligned_telemetry.parquet"
                    labels_path = race_dir / "target_labels.parquet"
                    aligned.to_parquet(aligned_path, index=False)
                    labels.to_parquet(labels_path, index=False)
                    source_files.extend([*race_artifacts, str(aligned_path), str(labels_path)])
                    aligned_tables.append(aligned)
                    label_tables.append(labels)
                    session_records.append({
                        "race_id": race_id, "provider": provider, "year": int(spec["year"]),
                        "gp": str(spec["gp"]), "session_type": str(spec.get("session_type", "R")),
                        "session_key": spec.get("session_key"),
                        "cache_root": str(session_cache / _safe_component(race_id)),
                        "coverage": {"aligned_rows": int(len(aligned)), "target_rows": int(len(labels))},
                        "source_metadata": source_metadata,
                    })

            aligned_all = pd.concat(aligned_tables, ignore_index=True)
            labels_all = pd.concat(label_tables, ignore_index=True)

            # Derive the frozen split from the legacy input representation. The
            # context reference is then fitted only on unique frames actually
            # consumed by training windows and the final split is asserted equal.
            base_config = copy.deepcopy(config)
            base_config["graph"]["node_feature_schema"] = "legacy_v1"
            base_config["model"]["in_dim"] = 4 + len(base_config["graph"]["compound_categories"])
            base_config["model"]["edge_dim"] = 1 if config["graph"].get("edge_feature_schema", "scalar_rbf") == "scalar_rbf" else 4
            base_snapshots, base_driver_to_idx = _build_snapshots_by_race(aligned_all, base_config, specs)
            initial_train, initial_val, initial_test, _ = create_dataloaders(
                copy.deepcopy(base_snapshots), labels_all, base_driver_to_idx, base_config
            )
            initial_membership = {"train": _membership(initial_train),
                                  "validation": _membership(initial_val),
                                  "test": _membership(initial_test)}
            initial_target_signatures = {
                "train": _target_signature(initial_train),
                "validation": _target_signature(initial_val),
            }
            train_frame_keys = _training_input_frame_keys(initial_train)
            schema = node_feature_schema(config)
            calibration = {"schema_id": schema, "references": [],
                           "training_frame_count": len(train_frame_keys)}
            if schema == "circuit_context_v1":
                sessions_by_race = {}
                for spec in specs:
                    if not spec.get("circuit_id") or not spec.get("lap_length_m"):
                        raise ValueError(f"Context features require circuit_id and lap_length_m for {spec['race_id']}.")
                    sessions_by_race[str(spec["race_id"])] = {
                        "circuit_id": str(spec["circuit_id"]),
                        "lap_length_m": float(spec["lap_length_m"]),
                    }
                contextual = prepare_context_features(
                    aligned_all, sessions_by_race,
                    sample_hz=float(config["preprocess"]["sample_hz"]),
                    mini_sectors=int(config["track"]["mini_sectors"]), schema=schema,
                )
                calibration = fit_context_calibration(contextual, train_frame_keys, schema=schema)
                contextual = apply_context_calibration(contextual, calibration)
                snapshots, driver_to_idx = _build_snapshots_by_race(contextual, config, specs)
            else:
                snapshots, driver_to_idx = _build_snapshots_by_race(aligned_all, config, specs)

            train_loader, val_loader, test_loader, _scaler = create_dataloaders(
                copy.deepcopy(snapshots), labels_all, driver_to_idx, config
            )
            membership = {"train": _membership(train_loader),
                          "validation": _membership(val_loader),
                          "test": _membership(test_loader)}
            if membership != initial_membership:
                raise ValueError("Feature or graph views changed frozen split membership; this is not permitted.")
            final_target_signatures = {
                "train": _target_signature(train_loader),
                "validation": _target_signature(val_loader),
            }
            if final_target_signatures != initial_target_signatures:
                raise ValueError("Feature or graph views changed train/validation targets or masks.")
            membership_hashes = {name: _sha256_bytes(_canonical_json(rows))
                                 for name, rows in membership.items()}
            split_metadata = copy.deepcopy(train_loader.dataset.split_metadata)
            coverage = _coverage(aligned_all, labels_all, snapshots)
            source_hashes = _artifact_relative_hashes(source_files, staging_dir, output_dir)
            if not source_hashes:
                raise ValueError("Cohort preparation produced no hashable source artifacts.")

            artifact_final = output_dir / "prepared_cohort.pkl"
            manifest_final = output_dir / "prepared_manifest.json"
            config_hash = _sha256_bytes(_canonical_json(config))
            cohort_id = "+".join(spec["race_id"] for spec in specs)
            calibration_hash = _sha256_bytes(_canonical_json(calibration))
            train_key_hash = _sha256_bytes(_canonical_json(train_frame_keys))
            source_manifest_hash = (_sha256_file(Path(aligned_source_dir).resolve() / "prepared_manifest.json")
                                   if aligned_source_dir is not None else None)
            schema_metadata = feature_schema_metadata(schema, config["graph"]["compound_categories"])
            provenance = {
                "provider": provider, "providers": [provider], "session_identity": cohort_id,
                "sessions": [{key: record[key] for key in
                              ("race_id", "provider", "year", "gp", "session_type", "session_key")}
                             for record in session_records],
                "prepared_artifact_path": str(artifact_final),
                "prepared_manifest_path": str(manifest_final),
                "effective_config_sha256": config_hash, "source_hashes": source_hashes,
                "coverage": coverage, "split": split_metadata,
                "split_membership": membership, "split_membership_sha256": membership_hashes,
                "node_feature_schema": schema_metadata,
                "context_calibration_sha256": calibration_hash,
                "training_input_frame_keys_sha256": train_key_hash,
                "aligned_source_manifest_sha256": source_manifest_hash,
                "train_validation_target_signatures": final_target_signatures,
            }
            payload = {
                "schema_version": 2, "snapshots": snapshots, "labels": labels_all,
                "driver_to_idx": driver_to_idx, "config": config,
                "memberships": membership, "membership_sha256": membership_hashes,
                "provenance": provenance, "context_calibration": calibration,
                "context_calibration_sha256": calibration_hash,
                "training_input_frame_keys": train_frame_keys,
                "node_feature_schema_metadata": schema_metadata,
            }
            artifact_staged = staging_dir / artifact_final.name
            with artifact_staged.open("wb") as output:
                pickle.dump(payload, output, protocol=pickle.HIGHEST_PROTOCOL)

            manifest = {
                "schema_version": 2, "provider": provider, "providers": [provider],
                "session_identity": cohort_id, "sessions": session_records,
                "config_path": str(config_path), "config_sha256": _sha256_file(config_path),
                "effective_config_sha256": config_hash,
                "prepared_artifact_path": str(artifact_final),
                "prepared_artifact_sha256": _sha256_file(artifact_staged),
                "prepared_artifact_bytes": int(artifact_staged.stat().st_size),
                "prepared_manifest_path": str(manifest_final), "prepared_manifest_sha256": None,
                "source_hashes": source_hashes, "coverage": coverage, "split": split_metadata,
                "split_membership": membership, "split_membership_sha256": membership_hashes,
                "node_feature_schema": schema_metadata,
                "context_calibration": calibration, "context_calibration_sha256": calibration_hash,
                "training_input_frame_keys_sha256": train_key_hash,
                "aligned_source_manifest_sha256": source_manifest_hash,
                "edge_feature_scaler": (train_loader.dataset.edge_feature_scaler.metadata()
                                         if hasattr(train_loader.dataset.edge_feature_scaler, "metadata") else None),
                "environment": _environment(), "git_head": _git_revision(),
                "test_metrics_evaluated": False, "training_started": False,
            }
            manifest["prepared_manifest_sha256"] = _sha256_bytes(
                _canonical_json({key: value for key, value in manifest.items()
                                 if key != "prepared_manifest_sha256"})
            )
            manifest_staged = staging_dir / manifest_final.name
            manifest_staged.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                       encoding="utf-8")
            sidecar = staging_dir / (manifest_final.name + ".sha256")
            sidecar.write_text(f"{_sha256_file(manifest_staged)}  {manifest_final.name}\n",
                               encoding="utf-8")
            os.replace(staging_dir, output_dir)
            return manifest
        except BaseException:
            # The staging directory is created by this call and is cleaned by TemporaryDirectory.
            raise


def load_prepared_cohort(artifact_path: str | Path, manifest_path: str | Path | None = None) -> tuple[dict, dict]:
    """Load a trusted local cohort only after verifying its detached manifest."""
    artifact_path = Path(artifact_path).resolve()
    manifest_path = Path(manifest_path).resolve() if manifest_path else artifact_path.parent / "prepared_manifest.json"
    sidecar = Path(str(manifest_path) + ".sha256")
    if not artifact_path.is_file() or not manifest_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("Prepared cohort requires its pickle, JSON manifest, and SHA-256 sidecar.")
    exact_manifest_hash = _sha256_file(manifest_path)
    if sidecar.read_text(encoding="utf-8").strip().split()[0] != exact_manifest_hash:
        raise ValueError("Prepared cohort manifest does not match its exact-byte checksum.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_hash = _sha256_bytes(_canonical_json(
        {key: value for key, value in manifest.items() if key != "prepared_manifest_sha256"}
    ))
    if manifest.get("prepared_manifest_sha256") != canonical_hash:
        raise ValueError("Prepared cohort manifest canonical hash is invalid.")
    if manifest.get("prepared_artifact_path") != str(artifact_path):
        raise ValueError("Prepared cohort path disagrees with its manifest.")
    if manifest.get("prepared_artifact_sha256") != _sha256_file(artifact_path):
        raise ValueError("Prepared cohort artifact hash does not match its manifest.")
    with artifact_path.open("rb") as source:
        payload = pickle.load(source)
    if not isinstance(payload, dict) or payload.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported prepared cohort artifact schema.")
    provenance = payload.get("provenance", {})
    for key in ("provider", "session_identity", "effective_config_sha256", "split_membership_sha256"):
        if provenance.get(key) != manifest.get(key):
            raise ValueError(f"Prepared cohort artifact provenance disagrees on {key}.")
    if _sha256_bytes(_canonical_json(payload["config"])) != manifest.get("effective_config_sha256"):
        raise ValueError("Prepared cohort configuration hash is invalid.")
    if payload.get("membership_sha256") != manifest.get("split_membership_sha256"):
        raise ValueError("Prepared cohort sample membership hash is invalid.")
    if payload.get("memberships") != manifest.get("split_membership"):
        raise ValueError("Prepared cohort split membership differs from its manifest.")
    if payload.get("schema_version") == 2:
        for key in ("context_calibration_sha256", "training_input_frame_keys_sha256"):
            if provenance.get(key) != manifest.get(key):
                raise ValueError(f"Prepared cohort context provenance disagrees on {key}.")
        if _sha256_bytes(_canonical_json(payload.get("context_calibration"))) != provenance.get("context_calibration_sha256"):
            raise ValueError("Prepared cohort training-only context calibration hash is invalid.")
        if _sha256_bytes(_canonical_json(payload.get("training_input_frame_keys"))) != provenance.get("training_input_frame_keys_sha256"):
            raise ValueError("Prepared cohort training-input frame key hash is invalid.")
    membership_hashes = {
        name: _sha256_bytes(_canonical_json(rows))
        for name, rows in payload.get("memberships", {}).items()
    }
    if membership_hashes != manifest.get("split_membership_sha256"):
        raise ValueError("Prepared cohort split-membership hashes do not match their samples.")
    provenance.update({
        "prepared_artifact_path": str(artifact_path),
        "prepared_artifact_sha256": manifest["prepared_artifact_sha256"],
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": manifest["prepared_manifest_sha256"],
        "source_hashes": manifest["source_hashes"],
        "coverage": manifest["coverage"],
        "split": manifest["split"],
    })
    return payload, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Validated multi-session YAML config")
    parser.add_argument("--output", required=True, type=Path,
                        help="New output directory; existing directories are never replaced")
    parser.add_argument("--aligned-source", type=Path,
                        help="Reuse normalized aligned/target tables from a frozen prepared cohort; no source normalization is run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    manifest = prepare_cohort(args.config, args.output, args.aligned_source)
    print(json.dumps({
        "output_dir": str(Path(args.output).resolve()),
        "provider": manifest["provider"],
        "sessions": [item["race_id"] for item in manifest["sessions"]],
        "coverage": manifest["coverage"],
        "split": {name: metadata.get("sample_count") for name, metadata in manifest["split"].items()
                  if name in ("train", "validation", "test")},
        "prepared_artifact_sha256": manifest["prepared_artifact_sha256"],
        "prepared_manifest_sha256": manifest["prepared_manifest_sha256"],
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
