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


def prepare_cohort(config_path: str | Path, output_dir: str | Path) -> dict:
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
            driver_maps = {}
            base_cache = Path(config["session"].get("cache_dir", "data/cache"))
            if not base_cache.is_absolute():
                base_cache = PROJECT_DIR / base_cache

            for spec in specs:
                race_id = spec["race_id"]
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
                race_snapshots, race_driver_map = build_race_graph_snapshots(aligned, race_config)
                if not race_snapshots or labels.empty:
                    raise ValueError(f"Session {race_id} produced no graph snapshots or target labels.")

                coverage = _coverage(aligned, labels, race_snapshots)
                artifact_paths = list(race_artifacts)
                aligned_path = race_dir / "aligned_telemetry.parquet"
                if aligned_path.is_file():
                    artifact_paths.append(str(aligned_path))
                label_path = race_dir / "target_labels.parquet"
                if label_path.is_file():
                    artifact_paths.append(str(label_path))
                source_files.extend(artifact_paths)
                session_records.append({
                    "race_id": race_id,
                    "provider": provider,
                    "year": int(spec["year"]),
                    "gp": str(spec["gp"]),
                    "session_type": str(spec.get("session_type", "R")),
                    "session_key": spec.get("session_key"),
                    "cache_root": str(session_cache / _safe_component(race_id)),
                    "coverage": coverage,
                    "source_metadata": source_metadata,
                })
                aligned_tables.append(aligned)
                label_tables.append(labels)
                snapshots.extend(race_snapshots)
                driver_maps[race_id] = race_driver_map

            aligned_all = pd.concat(aligned_tables, ignore_index=True)
            labels_all = pd.concat(label_tables, ignore_index=True)
            driver_to_idx = remap_cohort_driver_nodes(
                snapshots, driver_maps, int(config["graph"]["num_nodes"])
            )

            # Split/scaler construction uses a deep copy because it transforms graph features in place.
            split_snapshots = copy.deepcopy(snapshots)
            train_loader, val_loader, _test_loader, _scaler = create_dataloaders(
                split_snapshots, labels_all, driver_to_idx, config
            )
            membership = {"train": _membership(train_loader),
                          "validation": _membership(val_loader),
                          "test": _membership(_test_loader)}
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
            provenance = {
                "provider": provider,
                "providers": [provider],
                "session_identity": cohort_id,
                "sessions": [{key: record[key] for key in
                              ("race_id", "provider", "year", "gp", "session_type", "session_key")}
                             for record in session_records],
                "prepared_artifact_path": str(artifact_final),
                "prepared_manifest_path": str(manifest_final),
                "effective_config_sha256": config_hash,
                "source_hashes": source_hashes,
                "coverage": coverage,
                "split": split_metadata,
                "split_membership": membership,
                "split_membership_sha256": membership_hashes,
            }
            payload = {
                "schema_version": 1,
                "snapshots": snapshots,
                "labels": labels_all,
                "driver_to_idx": driver_to_idx,
                "config": config,
                "memberships": membership,
                "membership_sha256": membership_hashes,
                "provenance": provenance,
            }
            artifact_staged = staging_dir / artifact_final.name
            with artifact_staged.open("wb") as output:
                pickle.dump(payload, output, protocol=pickle.HIGHEST_PROTOCOL)

            manifest = {
                "schema_version": 1,
                "provider": provider,
                "providers": [provider],
                "session_identity": cohort_id,
                "sessions": session_records,
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "effective_config_sha256": config_hash,
                "prepared_artifact_path": str(artifact_final),
                "prepared_artifact_sha256": _sha256_file(artifact_staged),
                "prepared_artifact_bytes": int(artifact_staged.stat().st_size),
                "prepared_manifest_path": str(manifest_final),
                "prepared_manifest_sha256": None,
                "source_hashes": source_hashes,
                "coverage": coverage,
                "split": split_metadata,
                "split_membership": membership,
                "split_membership_sha256": membership_hashes,
                "environment": _environment(),
                "git_head": _git_revision(),
                "test_metrics_evaluated": False,
                "training_started": False,
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
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    manifest = prepare_cohort(args.config, args.output)
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
