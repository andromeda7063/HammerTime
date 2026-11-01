"""Pre-registered validation-only runs for the fixed ST-GNN ablation set."""

import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
import yaml

from dataset import collate_st_graph_windows
from evaluate import compute_metrics, validate_checkpoint_edge_scaler
from model import create_model
from train import train_model


ABLATION_CANDIDATES = [
    {"name": "current_k3", "temporal_kernel": 3, "temporal_mode": "conv",
     "graph": "current", "loss": "mae"},
    {"name": "self_loop_only_k3", "temporal_kernel": 3, "temporal_mode": "conv",
     "graph": "self_loop_only", "loss": "mae"},
    {"name": "temporal_kernel_1", "temporal_kernel": 1, "temporal_mode": "conv",
     "graph": "current", "loss": "mae"},
    {"name": "temporal_kernel_10", "temporal_kernel": 10, "temporal_mode": "conv",
     "graph": "current", "loss": "mae"},
    {"name": "constant_edge_attribute_k3", "temporal_kernel": 3,
     "temporal_mode": "conv", "graph": "constant_edge_attribute", "constant": 1.0, "loss": "mae"},
    {"name": "gated_dilated_full_history", "temporal_kernel": 3,
     "temporal_mode": "gated_dilated", "temporal_dilations": [1, 2, 4], "graph": "current", "loss": "mae"},
    {"name": "gated_dilated_mse", "temporal_kernel": 3,
     "temporal_mode": "gated_dilated", "temporal_dilations": [1, 2, 4],
     "graph": "current", "loss": "mse"},
]

# Five declared, matched views for the circuit-context/edge comparison suite.
# Every candidate shares the same temporal model and MAE objective so only the
# input representation and graph view differ.
CONTEXT_GRAPH_CANDIDATES = [
    {"name": "legacy_input_legacy_graph", "input_schema": "legacy_v1",
     "edge_view": "race_progress", "temporal_kernel": 3,
     "temporal_mode": "gated_dilated", "temporal_dilations": [1, 2, 4],
     "loss": "mae"},
    {"name": "circuit_context_legacy_graph", "input_schema": "circuit_context_v1",
     "edge_view": "race_progress", "temporal_kernel": 3,
     "temporal_mode": "gated_dilated", "temporal_dilations": [1, 2, 4],
     "loss": "mae"},
    {"name": "circuit_context_physical_scalar", "input_schema": "circuit_context_v1",
     "edge_view": "physical_scalar", "temporal_kernel": 3,
     "temporal_mode": "gated_dilated", "temporal_dilations": [1, 2, 4],
     "loss": "mae"},
    {"name": "circuit_context_physical_v1", "input_schema": "circuit_context_v1",
     "edge_view": "physical_v1", "temporal_kernel": 3,
     "temporal_mode": "gated_dilated", "temporal_dilations": [1, 2, 4],
     "loss": "mae"},
    {"name": "circuit_context_self_loop_only", "input_schema": "circuit_context_v1",
     "edge_view": "self_loop_only", "temporal_kernel": 3,
     "temporal_mode": "gated_dilated", "temporal_dilations": [1, 2, 4],
     "loss": "mae"},
]


def _stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value):
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_mapping(value, prefix=""):
    flattened = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(_flatten_mapping(item, path))
        else:
            flattened[path] = item
    return flattened


def _validate_source_provenance(source_provenance, config):
    required = {
        "provider", "session_identity", "prepared_artifact_path", "prepared_artifact_sha256",
        "prepared_manifest_path", "prepared_manifest_sha256", "effective_config_sha256",
    }
    missing = required.difference(source_provenance)
    if missing:
        raise ValueError(f"Source provenance is missing required fields: {sorted(missing)}")
    provider = str(source_provenance["provider"]).casefold()
    if provider not in ("fastf1", "openf1"):
        raise ValueError("Ablation training requires an explicitly selected FastF1 or OpenF1 provider.")
    providers = source_provenance.get("providers", [provider])
    if not isinstance(providers, list) or {str(item).casefold() for item in providers} != {provider}:
        raise ValueError("An ablation cohort cannot mix data providers.")
    if _sha256_file(source_provenance["prepared_artifact_path"]) != source_provenance["prepared_artifact_sha256"]:
        raise ValueError("Prepared authentic data artifact hash does not match its manifest.")
    manifest_path = Path(source_provenance["prepared_manifest_path"])
    manifest_bytes_hash = _sha256_file(manifest_path)
    sidecar = Path(str(manifest_path) + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip().split()[0] != manifest_bytes_hash:
        raise ValueError("Prepared data manifest bytes do not match the SHA-256 sidecar.")
    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    manifest_without_self_hash = dict(manifest)
    canonical_manifest_hash = manifest_without_self_hash.pop("prepared_manifest_sha256", None)
    if canonical_manifest_hash != _sha256_json(manifest_without_self_hash):
        raise ValueError("Prepared data manifest canonical hash is invalid.")
    if canonical_manifest_hash != source_provenance["prepared_manifest_sha256"]:
        raise ValueError("Prepared data manifest hash disagrees with source provenance.")
    if manifest.get("provider", "").casefold() != provider:
        raise ValueError("Prepared artifact provider disagrees with its manifest.")
    manifest_providers = {str(item).casefold() for item in manifest.get("providers", [provider])}
    if manifest_providers != {provider}:
        raise ValueError("Prepared artifact manifest contains a mixed-provider cohort.")
    if manifest.get("session_identity") != source_provenance["session_identity"]:
        raise ValueError("Prepared artifact session identity disagrees with its manifest.")
    artifact_hash = manifest.get("artifact", {}).get("sha256", manifest.get("prepared_artifact_sha256"))
    if artifact_hash != source_provenance["prepared_artifact_sha256"]:
        raise ValueError("Prepared artifact hash disagrees with the manifest payload.")
    prepared_config_path = manifest.get("config_path")
    if not prepared_config_path or not Path(prepared_config_path).is_file():
        raise ValueError("Prepared artifact manifest does not reference its effective config file.")
    with Path(prepared_config_path).open("r", encoding="utf-8") as source:
        prepared_config = yaml.safe_load(source)
    prepared_config_hash = _sha256_json(prepared_config)
    if manifest.get("effective_config_sha256") != prepared_config_hash:
        raise ValueError("Prepared artifact effective config hash is invalid.")
    source_hashes = manifest.get("source_hashes") or manifest.get("imported_cache_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Authentic artifact manifest has no source cache hashes.")
    membership = manifest.get("split_membership")
    if not isinstance(membership, dict) or not membership:
        raise ValueError("Authentic artifact manifest has no split membership.")
    membership_hashes = manifest.get("split_membership_sha256")
    if isinstance(membership_hashes, dict):
        valid_membership = all(
            split_name in membership_hashes and membership_hashes[split_name] == _sha256_json(rows)
            for split_name, rows in membership.items()
        )
    else:
        valid_membership = membership_hashes == _sha256_json(membership)
    if not valid_membership:
        raise ValueError("Split membership hash in prepared artifact manifest is invalid.")
    if source_provenance["effective_config_sha256"] != prepared_config_hash:
        raise ValueError("Provided effective config hash does not match the prepared config.")
    approved_overrides = {
        "model.temporal_mode", "model.temporal_dilations", "session.cache_dir",
        "session.processed_dir", "train.checkpoint_path", "train.history_path",
    }
    prepared_flat = _flatten_mapping(prepared_config)
    current_flat = _flatten_mapping(config)
    differences = {
        key: {"prepared": prepared_flat.get(key, "<absent>"),
              "current": current_flat.get(key, "<absent>")}
        for key in set(prepared_flat) | set(current_flat)
        if _stable_json(prepared_flat.get(key, "<absent>")) !=
           _stable_json(current_flat.get(key, "<absent>"))
    }
    forbidden = set(differences).difference(approved_overrides)
    if forbidden:
        raise ValueError(f"Runner config changes data/training protocol keys outside the allowlist: {sorted(forbidden)}")
    return {
        "manifest": manifest,
        "prepared_config_sha256": prepared_config_hash,
        "data_protocol_config_sha256": _sha256_json({
            key: value for key, value in prepared_flat.items() if key not in approved_overrides
        }),
        "approved_config_overrides": differences,
    }


def _git_metadata():
    repo = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                    check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"revision": revision, "dirty": dirty}


def _environment_metadata(device):
    repo = Path(__file__).resolve().parents[2]
    module_names = (
        "ablation.py", "dataset.py", "model.py", "train.py", "evaluate.py",
        "graph.py", "physical_graph.py", "context_features.py", "config_validation.py",
        "prepare_cohort.py", "baselines.py",
    )
    modules = {}
    for name in module_names:
        path = repo / "hammertime_100" / "src" / name
        modules[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry_path = repo / "hammertime_100" / "train_prepared.py"
    modules["train_prepared.py"] = hashlib.sha256(entry_path.read_bytes()).hexdigest()
    lock_path = repo / "uv.lock"
    package_names = ("numpy", "pandas", "torch", "torch-geometric", "scikit-learn")
    versions = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cpu_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "packages": versions,
        "uv_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "module_source_sha256": modules,
    }


def _data_hash(train_loader, val_loader):
    digest = hashlib.sha256()
    hashed_snapshots = set()
    for split_name, loader in (("train", train_loader), ("validation", val_loader)):
        for sample in loader.dataset.samples:
            digest.update(split_name.encode("utf-8"))
            digest.update(_stable_json({
                "race_id": sample.get("race_id"),
                "target_time": sample.get("target_time"),
                "input_start_time": sample.get("input_start_time"),
                "driver_names": sample.get("driver_names"),
            }).encode("utf-8"))
            for key in ("target", "mask"):
                tensor = sample[key].detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
            for snapshot in sample["snapshots"]:
                snapshot_key = (getattr(snapshot, "race_id", None), float(snapshot.time_sec))
                if snapshot_key in hashed_snapshots:
                    continue
                hashed_snapshots.add(snapshot_key)
                digest.update(_stable_json({"snapshot_race_id": snapshot_key[0],
                                            "snapshot_time_sec": snapshot_key[1]}).encode("utf-8"))
                digest.update(_stable_json({
                    "position_coverage": getattr(snapshot, "position_coverage", None),
                    "blocked_neighbor_count": getattr(snapshot, "blocked_neighbor_count", None),
                }).encode("utf-8"))
                for tensor_name in (
                    "x", "edge_index", "edge_attr", "mask", "position_valid",
                    "edge_index_race_progress", "edge_attr_race_progress",
                    "edge_index_physical_circuit", "edge_attr_physical_circuit",
                ):
                    tensor = getattr(snapshot, tensor_name, None)
                    if tensor is None:
                        digest.update(b"<none>")
                    else:
                        tensor = tensor.detach().cpu().contiguous()
                        digest.update(str(tensor.dtype).encode("ascii"))
                        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _membership_hash(train_loader, val_loader):
    membership = {}
    for split_name, loader in (("train", train_loader), ("validation", val_loader)):
        membership[split_name] = [
            {
                "race_id": sample.get("race_id"),
                "input_start_time": sample.get("input_start_time"),
                "target_time": sample.get("target_time"),
                "driver_names": sample.get("driver_names"),
                "target_mask": sample["mask"].detach().cpu().numpy().astype(bool).tolist(),
            }
            for sample in loader.dataset.samples
        ]
    return _sha256_json(membership)


def _target_mask_hash(samples):
    digest = hashlib.sha256()
    for sample in samples:
        for key in ("target", "mask"):
            tensor = sample[key].detach().cpu().contiguous()
            digest.update(key.encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _baseline_report(baselines, config):
    """Keep validation baseline summaries while grouping the wrapper's rows by circuit/sector."""
    if not baselines:
        return {}
    circuit_by_race = {
        str(item.get("race_id")): str(item.get("circuit_id", "unknown"))
        for item in config.get("sessions", [])
    }
    report = {}
    for name, values in baselines.items():
        if not isinstance(values, dict):
            continue
        summary = {key: value for key, value in values.items()
                   if key not in ("predictions", "targets", "metadata")}
        predictions = np.asarray(values.get("predictions", []), dtype=float)
        targets = np.asarray(values.get("targets", []), dtype=float)
        if len(predictions) == len(targets) and len(targets):
            summary["prediction_variance"] = float(np.var(predictions))
            summary["target_variance"] = float(np.var(targets))
            groups = {}
            metadata = values.get("metadata", [])
            for index, (truth, prediction) in enumerate(zip(targets, predictions)):
                row = metadata[index] if index < len(metadata) else {}
                race_id = str(row.get("race_id", "unknown"))
                sector = row.get("mini_sector")
                try:
                    sector = int(sector) if sector is not None and np.isfinite(float(sector)) else None
                except (TypeError, ValueError):
                    sector = None
                key = (circuit_by_race.get(race_id, "unknown"), race_id, sector)
                groups.setdefault(key, [[], []])
                groups[key][0].append(float(truth))
                groups[key][1].append(float(prediction))
            summary["per_circuit_sector"] = []
            for (circuit, race, sector), (truths, preds) in sorted(groups.items(), key=lambda item: str(item[0])):
                error = np.asarray(preds) - np.asarray(truths)
                summary["per_circuit_sector"].append({
                    "circuit_id": circuit, "race_id": race, "mini_sector": sector,
                    "sample_count": len(truths), "mae_sec": float(np.mean(np.abs(error))),
                    "rmse_sec": float(np.sqrt(np.mean(np.square(error)))),
                    "bias_sec": float(np.mean(error)),
                    "target_variance": float(np.var(truths)),
                    "prediction_variance": float(np.var(preds)),
                })
        report[name] = summary
    return report


def _zero_prediction_validation_baseline(val_loader):
    targets = []
    for sample in val_loader.dataset.samples:
        valid = sample["mask"].detach().cpu().numpy().astype(bool)
        target = sample["target"].detach().cpu().numpy()
        targets.extend(target[valid].tolist())
    y_true = np.asarray(targets, dtype=float)
    return compute_metrics(y_true, np.zeros_like(y_true))


def _feature_scaler_metadata(train_loader, val_loader):
    train_scaler = getattr(train_loader.dataset, "feature_scaler", None)
    val_scaler = getattr(val_loader.dataset, "feature_scaler", None)
    if train_scaler is None or val_scaler is None:
        raise ValueError("Training and validation loaders must share the training-fitted scaler.")
    fields = ("mean_", "scale_", "var_")
    if any(not hasattr(train_scaler, field) or not hasattr(val_scaler, field) for field in fields):
        raise ValueError("Ablation requires a fitted feature scaler with mean, scale, and variance.")
    values = {field: np.asarray(getattr(train_scaler, field)).tolist() for field in fields}
    if any(not np.array_equal(np.asarray(getattr(train_scaler, field)),
                              np.asarray(getattr(val_scaler, field))) for field in fields):
        raise ValueError("Training and validation features do not share identical scaler parameters.")
    values["n_samples_seen_"] = np.asarray(getattr(train_scaler, "n_samples_seen_", [])).tolist()
    values["n_features_in_"] = int(train_scaler.n_features_in_)
    return values


def _clone_transformed_samples(samples, graph_mode, edge_scaler=None):
    candidate = graph_mode if isinstance(graph_mode, dict) else None
    if candidate is not None:
        graph_mode = candidate["edge_view"]
    transformed = []
    for sample in samples:
        item = dict(sample)
        snapshots = []
        for original in sample["snapshots"]:
            snap = original.clone() if hasattr(original, "clone") else copy.deepcopy(original)
            if candidate is not None:
                if candidate["input_schema"] == "legacy_v1":
                    snap.x = snap.x[:, :9].contiguous()
                elif candidate["input_schema"] != "circuit_context_v1":
                    raise ValueError(f"Unknown node feature schema {candidate['input_schema']!r}.")
                if graph_mode == "race_progress":
                    snap.edge_index = snap.edge_index_race_progress.clone()
                    snap.edge_attr = snap.edge_attr_race_progress.clone()
                elif graph_mode in ("physical_scalar", "physical_v1"):
                    snap.edge_index = snap.edge_index_physical_circuit.clone()
                    physical_attr = snap.edge_attr_physical_circuit
                    if graph_mode == "physical_scalar":
                        snap.edge_attr = physical_attr[:, :1].clone()
                    else:
                        if edge_scaler is None:
                            raise ValueError("physical_v1 comparison requires a train-fitted edge scaler.")
                        snap.edge_attr = edge_scaler.transform(physical_attr)[:, :4].contiguous()
                elif graph_mode == "self_loop_only":
                    snap.edge_index = torch.empty((2, 0), dtype=torch.long, device=snap.x.device)
                    snap.edge_attr = torch.empty((0, 1), dtype=torch.float32, device=snap.x.device)
                else:
                    raise ValueError(f"Unknown comparison edge view {graph_mode!r}.")
            elif graph_mode == "self_loop_only":
                edge_dim = int(snap.edge_attr.size(1)) if getattr(snap, "edge_attr", None) is not None else 1
                snap.edge_index = torch.empty((2, 0), dtype=torch.long,
                                              device=snap.x.device)
                snap.edge_attr = torch.empty((0, edge_dim), dtype=torch.float32,
                                             device=snap.x.device)
            elif graph_mode == "constant_edge_attribute":
                edge_index = getattr(snap, "edge_index", None)
                edge_count = int(edge_index.size(1)) if edge_index is not None else 0
                edge_dim = int(snap.edge_attr.size(1)) if getattr(snap, "edge_attr", None) is not None else 1
                if edge_dim == 1:
                    snap.edge_attr = torch.ones((edge_count, 1), dtype=torch.float32,
                                                device=snap.x.device)
                else:
                    attrs = torch.zeros((edge_count, edge_dim), dtype=torch.float32, device=snap.x.device)
                    if edge_count:
                        attrs[:, 0] = 1.0
                        if edge_dim == 4:
                            attrs[:, 3] = float(getattr(snap, "position_coverage", 0.0))
                    snap.edge_attr = attrs
            elif graph_mode == "current":
                if (edge_scaler is not None and getattr(snap, "edge_attr", None) is not None
                        and snap.edge_attr.ndim == 2 and snap.edge_attr.size(1) == 4):
                    snap.edge_attr = edge_scaler.transform(snap.edge_attr)
            else:
                raise ValueError(f"Unknown graph ablation {graph_mode!r}.")
            snapshots.append(snap)
        item["snapshots"] = snapshots
        transformed.append(item)
    return transformed


def _make_loader(source_loader, samples, shuffle, seed=None):
    dataset = copy.copy(source_loader.dataset)
    dataset.samples = samples
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=source_loader.batch_size, shuffle=shuffle,
                      collate_fn=collate_st_graph_windows, num_workers=source_loader.num_workers,
                      pin_memory=source_loader.pin_memory, drop_last=source_loader.drop_last,
                      generator=generator)


@torch.no_grad()
def _validation_metrics(model, loader, device, config=None):
    if not isinstance(getattr(loader, "sampler", None), SequentialSampler):
        raise ValueError("Ablation scoring requires sequential validation batches.")
    model.eval()
    targets, predictions, detail_rows = [], [], []
    sample_offset = 0
    circuit_by_race = {
        str(item.get("race_id")): str(item.get("circuit_id", "unknown"))
        for item in (config or {}).get("sessions", [])
    }
    if not circuit_by_race and (config or {}).get("session", {}).get("race_id"):
        circuit_by_race[str(config["session"]["race_id"])] = str(
            (config or {}).get("session", {}).get("circuit_id", "unknown")
        )
    for batch in loader:
        # Ablation scores remain on the primary residual task when optional
        # auxiliary labels are present in the same loader.
        windows, target, mask = batch[:3]
        prediction = model(windows)
        valid = mask.to(device=device, dtype=torch.bool)
        target_values = target.to(device).detach().cpu().numpy()
        prediction_values = prediction.detach().cpu().numpy()
        valid_values = valid.detach().cpu().numpy()
        for offset in range(len(windows)):
            sample = loader.dataset.samples[sample_offset + offset]
            race_id = str(sample.get("race_id", "unknown"))
            names = sample.get("driver_names") or [None] * len(valid_values[offset])
            sectors = np.asarray(sample.get("mini_sector", np.full(len(names), np.nan)), dtype=float)
            for node in np.flatnonzero(valid_values[offset]):
                y_value = float(target_values[offset, node])
                p_value = float(prediction_values[offset, node])
                targets.append(y_value)
                predictions.append(p_value)
                sector = float(sectors[node]) if node < len(sectors) and np.isfinite(sectors[node]) else None
                detail_rows.append({
                    "race_id": race_id,
                    "circuit_id": circuit_by_race.get(race_id, "unknown"),
                    "mini_sector": sector,
                    "driver": names[node] if node < len(names) else None,
                    "target": y_value,
                    "prediction": p_value,
                })
        sample_offset += len(windows)
    y_true = np.asarray(targets, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    metrics = compute_metrics(y_true, y_pred)
    metrics["target_variance"] = float(np.var(y_true)) if len(y_true) else None
    metrics["prediction_variance"] = float(np.var(y_pred)) if len(y_pred) else None
    metrics["prediction_mean"] = float(np.mean(y_pred)) if len(y_pred) else None
    groups = {}
    for row in detail_rows:
        key = (row["circuit_id"], row["race_id"], row["mini_sector"])
        groups.setdefault(key, {"target": [], "prediction": []})
        groups[key]["target"].append(row["target"])
        groups[key]["prediction"].append(row["prediction"])
    metrics["per_circuit_sector"] = []
    for (circuit_id, race_id, sector), values in sorted(groups.items(), key=lambda item: str(item[0])):
        group_y = np.asarray(values["target"], dtype=float)
        group_p = np.asarray(values["prediction"], dtype=float)
        error = group_p - group_y
        metrics["per_circuit_sector"].append({
            "circuit_id": circuit_id, "race_id": race_id, "mini_sector": sector,
            "sample_count": int(len(group_y)),
            "mae_sec": float(np.mean(np.abs(error))),
            "rmse_sec": float(np.sqrt(np.mean(np.square(error)))),
            "bias_sec": float(np.mean(error)),
            "target_variance": float(np.var(group_y)),
            "prediction_variance": float(np.var(group_p)),
        })
    return metrics


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
    os.replace(temporary, path)


def run_ablation_study(train_loader, val_loader, config, device, study_dir,
                       source_provenance, test_exposure, seeds=(42, 43, 44),
                       mode="complete", max_runs=None, epoch_cap=None,
                       candidate_indices=None, candidate_registry=None,
                       comparison_baselines=None):
    """Run a pre-registered candidate prefix; this API accepts no test loader."""
    device = torch.device(device)
    if mode not in ("pilot", "complete", "compare"):
        raise ValueError("mode must be 'pilot', 'complete', or 'compare'.")
    if not isinstance(source_provenance, dict) or not source_provenance:
        raise ValueError("Authentic source provenance must be supplied before registration.")
    if not isinstance(test_exposure, str) or not test_exposure.strip():
        raise ValueError("Record prior held-out test exposure before registration.")
    if not isinstance(seeds, (list, tuple)) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty sequence of unique integers.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers.")
    if (int(config["data"]["window_size"]) != 10 or
            int(config["data"]["horizon"]) != 5 or
            int(config["model"]["conv1d_kernel"]) != 3):
        raise ValueError("Ablations require the configured W=10, K=5, kernel=3 reference protocol.")
    prepared_validation = _validate_source_provenance(source_provenance, config)
    prepared_manifest = prepared_validation["manifest"]
    if not isinstance(getattr(val_loader, "sampler", None), SequentialSampler):
        raise ValueError("Ablation requires the existing sequential validation loader.")
    candidates = copy.deepcopy(candidate_registry if candidate_registry is not None else ABLATION_CANDIDATES)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate_registry must be a non-empty list.")
    if candidate_indices is None:
        registered_candidate_indices = list(range(len(candidates)))
    else:
        if (not isinstance(candidate_indices, (list, tuple)) or not candidate_indices
                or any(isinstance(index, bool) or not isinstance(index, int)
                       or not 0 <= index < len(candidates) for index in candidate_indices)
                or len(set(candidate_indices)) != len(candidate_indices)):
            raise ValueError("candidate_indices must be a non-empty sequence of unique valid candidate indexes.")
        registered_candidate_indices = list(candidate_indices)
    if mode == "complete" and registered_candidate_indices != list(range(len(candidates))):
        raise ValueError("A complete study must include every declared candidate.")
    if mode == "compare":
        if candidate_registry is None or candidates != CONTEXT_GRAPH_CANDIDATES:
            raise ValueError("compare mode requires the fixed CONTEXT_GRAPH_CANDIDATES registry.")
        if tuple(seeds) != (42, 43, 44):
            raise ValueError("The context comparison is preregistered for screen seed 42 and confirmation seeds 43, 44.")
        if registered_candidate_indices != list(range(len(candidates))):
            raise ValueError("The context comparison must screen all five declared views.")
        if not isinstance(comparison_baselines, dict) or "mean_train_constant" not in comparison_baselines:
            raise ValueError("compare mode requires validation-only training-mean baseline metrics.")
        total_runs = len(candidates) + 2 * len(seeds[1:])
    else:
        total_runs = len(registered_candidate_indices) * len(seeds)
    run_limit = total_runs if max_runs is None else int(max_runs)
    if not 1 <= run_limit <= total_runs:
        raise ValueError(f"max_runs must be in [1, {total_runs}].")
    if mode in ("complete", "compare") and run_limit != total_runs:
        raise ValueError("A complete study must run every candidate across every matched seed.")
    if mode == "complete" and len(seeds) < 3:
        raise ValueError("A complete ablation study requires at least three matched seeds.")
    if mode == "complete" and epoch_cap is not None:
        raise ValueError("A capped-epoch run is a pilot and cannot be registered as complete.")
    if epoch_cap is not None and (isinstance(epoch_cap, bool) or int(epoch_cap) < 1):
        raise ValueError("epoch_cap must be a positive integer when supplied.")

    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=False)
    effective_config = copy.deepcopy(config)
    if epoch_cap is not None:
        effective_config["train"]["max_epochs"] = int(epoch_cap)
        effective_config["train"]["early_stopping_patience"] = min(
            int(effective_config["train"].get("early_stopping_patience", epoch_cap)), int(epoch_cap)
        )
    split_metadata = copy.deepcopy(getattr(train_loader.dataset, "split_metadata", {}))
    feature_scaler = _feature_scaler_metadata(train_loader, val_loader)
    edge_feature_scaler = getattr(train_loader.dataset, "edge_feature_scaler", None)
    edge_scaler_metadata = (edge_feature_scaler.metadata()
                            if hasattr(edge_feature_scaler, "metadata") else None)
    feature_scalers_by_candidate = {}
    for candidate in candidates:
        schema = candidate.get("input_schema")
        if schema:
            width = (4 + len(config["graph"]["compound_categories"])) + (10 if schema == "circuit_context_v1" else 0)
            projected = copy.deepcopy(feature_scaler)
            projected["mean_"] = projected["mean_"][:width]
            projected["scale_"] = projected["scale_"][:width]
            projected["var_"] = projected["var_"][:width]
            projected["n_features_in_"] = width
            feature_scalers_by_candidate[candidate["name"]] = projected
    preregistration = {
        "schema_version": 1,
        "status": "preregistered",
        "mode": mode,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(),
        "environment": _environment_metadata(device),
        "configuration": effective_config,
        "configuration_sha256": _sha256_json(effective_config),
        "prepared_data_config_sha256": prepared_validation["prepared_config_sha256"],
        "data_protocol_config_sha256": prepared_validation["data_protocol_config_sha256"],
        "approved_config_overrides": prepared_validation["approved_config_overrides"],
        "source_provenance": source_provenance,
        "source_provenance_sha256": _sha256_json(source_provenance),
        "prepared_manifest_coverage": prepared_manifest.get("coverage", {}),
        "prepared_manifest_split_membership_sha256": prepared_manifest["split_membership_sha256"],
        "train_validation_data_sha256": _data_hash(train_loader, val_loader),
        "train_validation_membership_sha256": _membership_hash(train_loader, val_loader),
        "split_metadata": split_metadata,
        "split_metadata_sha256": _sha256_json(split_metadata),
        "feature_scaler": feature_scaler,
        "feature_scaler_sha256": _sha256_json(feature_scaler),
        "feature_scalers_by_candidate": feature_scalers_by_candidate,
        "physical_edge_scaler": edge_scaler_metadata,
        "physical_edge_scaler_sha256": _sha256_json(edge_scaler_metadata) if edge_scaler_metadata is not None else None,
        "validation_zero_prediction_baseline": _zero_prediction_validation_baseline(val_loader),
        "validation_zero_prediction_baseline_role": "context only; not a candidate and not used for selection",
        "candidates": candidates,
        "comparison_baselines": _baseline_report(comparison_baselines, config),
        "registered_candidate_indices": registered_candidate_indices,
        "seeds": list(seeds),
        "run_order": ("seed 42 screen in declaration order; select one eligible non-control candidate by validation R2; confirm it and legacy control on seeds 43, 44" if mode == "compare" else "candidate declaration order, then seed order"),
        "max_runs": run_limit,
        "full_study_run_count": total_runs,
        "max_epochs": int(effective_config["train"].get("max_epochs", 100)),
        "selection_metric": ("seed_42_validation_r2_score" if mode == "compare" else "mean_validation_mae_sec"),
        "selection_rule": (
            "Screen seed 42 only. Eligible non-control views must have validation R2 greater than the train-mean constant and validation MAE no greater than legacy_input_legacy_graph. Select highest R2; ties follow candidate declaration order. Then train only that view and the legacy control on seeds 43 and 44."
            if mode == "compare" else "lowest mean validation MAE; ties resolved by declared candidate order"
        ),
        "reported_validation_metrics": ["mae_sec", "r2_score", "sample_count"],
        "parameter_count": "trainable model parameters",
        "runtime_definition": "training plus best-checkpoint validation scoring; excludes preprocessing and loader construction",
        "constant_edge_attribute": 1.0,
        "self_loop_edge_attribute": 1.0,
        "edge_operator_comparison_note": (
            "Typed edge-conditioned values require the physical_v2 five-channel schema. "
            "When a candidate narrows the edge schema, the runner disables typed_edge_values; "
            "that result changes both schema and operator and is not an isolated edge-feature ablation."
        ),
        "gat_edge_loop_behavior": (
            "HammerTime inserts one self-loop per node in _ensure_self_loops; "
            "both GATConv layers have add_self_loops=False. Self-loop-only removes "
            "all stored inter-car edges; constant-edge replaces each stored edge "
            "attribute with 1.0 and leaves topology unchanged."
        ),
        "test_handling": {
            "test_loader_passed_to_runner": False,
            "test_iterated_or_scored": False,
            "prior_exposure": test_exposure,
        },
    }
    prereg_path = study_dir / "preregistration.json"
    with prereg_path.open("x", encoding="utf-8") as output:
        json.dump(preregistration, output, indent=2, allow_nan=False)
    results = {"status": "running", "preregistration": str(prereg_path),
               "runs": [], "started_at_utc": datetime.now(timezone.utc).isoformat()}
    results_path = study_dir / "results.json"
    _write_json(results_path, results)

    completed = 0
    compare_schedule = ([(index, 42) for index in registered_candidate_indices]
                        if mode == "compare"
                        else [(index, seed) for index in registered_candidate_indices for seed in seeds])
    compare_selected = None
    try:
        schedule_index = 0
        while schedule_index < len(compare_schedule):
            candidate_index, seed = compare_schedule[schedule_index]
            candidate = candidates[candidate_index]
            trial_config = copy.deepcopy(effective_config)
            trial_config["model"]["conv1d_kernel"] = int(candidate["temporal_kernel"])
            trial_config["model"]["temporal_mode"] = candidate["temporal_mode"]
            trial_config["model"]["temporal_dilations"] = candidate.get("temporal_dilations", [1, 2, 4])
            trial_config["data"]["random_seed"] = int(seed)
            trial_config["train"]["loss"] = candidate["loss"]
            if "input_schema" in candidate:
                trial_config["graph"]["node_feature_schema"] = candidate["input_schema"]
                compound_count = len(trial_config["graph"]["compound_categories"])
                trial_config["model"]["in_dim"] = (
                    4 + compound_count + (10 if candidate["input_schema"] == "circuit_context_v1" else 0)
                )
                view = candidate["edge_view"]
                if view == "physical_v1":
                    trial_config["graph"]["neighbor_mode"] = "physical_circuit"
                    trial_config["graph"]["edge_feature_schema"] = "physical_v1"
                    trial_config["model"]["edge_dim"] = 4
                elif view == "physical_scalar":
                    trial_config["graph"]["neighbor_mode"] = "physical_circuit"
                    trial_config["graph"]["edge_feature_schema"] = "scalar_rbf"
                    trial_config["model"]["edge_dim"] = 1
                else:
                    trial_config["graph"]["neighbor_mode"] = "race_progress"
                    trial_config["graph"]["edge_feature_schema"] = "scalar_rbf"
                    trial_config["model"]["edge_dim"] = 1
            # The typed value operator is defined over the five physical-v2
            # channels. Candidate views with narrower schemas keep their
            # declared edge representation and disable this operator.
            if trial_config["graph"].get("edge_feature_schema") != "physical_v2":
                trial_config["model"]["typed_edge_values"] = False
            trial_dir = study_dir / f"candidate_{candidate_index:02d}" / f"seed_{seed}"
            trial_dir.mkdir(parents=True, exist_ok=False)
            trial_config["train"]["checkpoint_path"] = str(trial_dir / "best.pt")
            trial_config["train"]["history_path"] = str(trial_dir / "training_history.csv")

            edge_scaler = getattr(train_loader.dataset, "edge_feature_scaler", None)
            if "input_schema" in candidate:
                train_samples = _clone_transformed_samples(
                    train_loader.dataset.samples, candidate, edge_scaler=edge_scaler
                )
                val_samples = _clone_transformed_samples(
                    val_loader.dataset.samples, candidate, edge_scaler=edge_scaler
                )
            else:
                graph_mode = candidate.get("graph", "current")
                train_samples = _clone_transformed_samples(
                    train_loader.dataset.samples, graph_mode, edge_scaler=edge_scaler
                )
                val_samples = _clone_transformed_samples(
                    val_loader.dataset.samples, graph_mode, edge_scaler=edge_scaler
                )
            train_target_mask_hash = _target_mask_hash(train_loader.dataset.samples)
            val_target_mask_hash = _target_mask_hash(val_loader.dataset.samples)
            if (_target_mask_hash(train_samples) != train_target_mask_hash
                    or _target_mask_hash(val_samples) != val_target_mask_hash):
                raise ValueError("A candidate view changed frozen train/validation targets or masks.")
            trial_train = _make_loader(train_loader, train_samples, shuffle=True, seed=int(seed))
            trial_val = _make_loader(val_loader, val_samples, shuffle=False)
            model = create_model(trial_config).to(device)
            parameter_count = sum(int(parameter.numel()) for parameter in model.parameters()
                                  if parameter.requires_grad)
            started = time.perf_counter()
            best_val_mae, history = train_model(model, trial_train, trial_val, trial_config, device)
            checkpoint_path = Path(trial_config["train"]["checkpoint_path"])
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            validate_checkpoint_edge_scaler(checkpoint, model, trial_val)
            metrics = _validation_metrics(model, trial_val, device, trial_config)
            runtime = time.perf_counter() - started
            if not math.isfinite(float(best_val_mae)) or not math.isclose(
                    float(best_val_mae), metrics["mae_sec"], rel_tol=1e-5, abs_tol=1e-7):
                raise FloatingPointError("Reloaded best checkpoint does not reproduce its validation MAE.")
            results["runs"].append({
                "candidate": candidate["name"], "candidate_index": candidate_index,
                "training_loss": candidate["loss"], "seed": int(seed),
                "effective_model_options": {
                    "edge_feature_schema": trial_config["graph"].get("edge_feature_schema", "scalar_rbf"),
                    "edge_dim": int(trial_config["model"].get("edge_dim", 1)),
                    "typed_edge_values": bool(trial_config["model"].get("typed_edge_values", False)),
                    "mask_inactive_nodes": bool(trial_config["model"].get("mask_inactive_nodes", False)),
                },
                "validation_metrics": metrics,
                "best_checkpoint_validation_mae_sec": float(best_val_mae),
                "epochs": len(history), "parameter_count": parameter_count,
                "train_target_mask_sha256": train_target_mask_hash,
                "validation_target_mask_sha256": val_target_mask_hash,
                "runtime_seconds": float(runtime), "checkpoint": str(checkpoint_path),
            })
            completed += 1
            results["completed_runs"] = completed
            _write_json(results_path, results)
            del trial_train, trial_val, train_samples, val_samples, model, checkpoint
            gc.collect()
            schedule_index += 1

            if mode == "compare" and completed == len(candidates):
                screen_runs = {run["candidate_index"]: run for run in results["runs"] if run["seed"] == 42}
                control_mae = screen_runs[0]["validation_metrics"]["mae_sec"]
                mean_r2 = comparison_baselines["mean_train_constant"].get("r2_score")
                eligible = []
                if mean_r2 is not None and math.isfinite(float(mean_r2)):
                    for index in range(1, len(candidates)):
                        r2_value = screen_runs[index]["validation_metrics"].get("r2_score")
                        if (r2_value is not None and math.isfinite(float(r2_value))
                                and float(r2_value) > float(mean_r2)
                                and screen_runs[index]["validation_metrics"]["mae_sec"] <= control_mae):
                            eligible.append(index)
                if eligible:
                    compare_selected = max(
                        eligible,
                        key=lambda index: (screen_runs[index]["validation_metrics"]["r2_score"], -index),
                    )
                    compare_schedule.extend((compare_selected, seed) for seed in (43, 44))
                    compare_schedule.extend((0, seed) for seed in (43, 44))
                    results["screen_selection"] = {
                        "eligible_candidate_indices": eligible,
                        "selected_candidate": candidates[compare_selected]["name"],
                        "selection_r2_seed_42": screen_runs[compare_selected]["validation_metrics"]["r2_score"],
                        "legacy_control_mae_seed_42": control_mae,
                        "train_mean_baseline_r2": mean_r2,
                        "confirmation_seeds": [43, 44],
                    }
                else:
                    results["screen_selection"] = {
                        "eligible_candidate_indices": [], "selected_candidate": None,
                        "reason": "No non-control view beat train-mean validation R2 while matching or improving legacy MAE.",
                        "legacy_control_mae_seed_42": control_mae,
                        "train_mean_baseline_r2": mean_r2,
                        "confirmation_seeds": [],
                    }
                    break
    except BaseException as error:
        results.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        _write_json(results_path, results)
        raise

    if mode == "compare":
        by_candidate = {}
        for index, candidate in enumerate(candidates):
            runs = [run for run in results["runs"] if run["candidate_index"] == index]
            valid_r2 = [float(run["validation_metrics"]["r2_score"]) for run in runs
                        if run["validation_metrics"].get("r2_score") is not None]
            by_candidate[candidate["name"]] = {
                "runs": [{"seed": run["seed"], **run["validation_metrics"]} for run in runs],
                "mean_validation_mae_sec": (float(np.mean([run["validation_metrics"]["mae_sec"] for run in runs]))
                                             if runs else None),
                "mean_validation_r2_score": float(np.mean(valid_r2)) if valid_r2 else None,
                "parameter_count": runs[0]["parameter_count"] if runs else None,
                "mean_runtime_seconds": (float(np.mean([run["runtime_seconds"] for run in runs]))
                                         if runs else None),
            }
        results["candidate_summaries"] = by_candidate
        results["selected_candidate"] = (candidates[compare_selected]["name"]
                                         if compare_selected is not None else None)
        results["status"] = ("complete" if compare_selected is not None and completed == total_runs
                             else "inconclusive")
        if compare_selected is None:
            results["note"] = "No candidate met the preregistered validation-only selection rule; no confirmation runs were started."
        elif completed < total_runs:
            results["note"] = "Screening selected a view, but confirmation did not complete."
    elif mode == "complete" and completed == total_runs:
        candidate_scores = []
        candidate_summaries = []
        for index, candidate in enumerate(candidates):
            candidate_runs = [run for run in results["runs"] if run["candidate_index"] == index]
            values = [run["validation_metrics"]["mae_sec"] for run in candidate_runs]
            candidate_scores.append(float(np.mean(values)))
            candidate_summaries.append({
                "candidate": candidate["name"],
                "training_loss": candidate["loss"],
                "seed_mae_sec": values,
                "mean_validation_mae_sec": float(np.mean(values)),
                "std_validation_mae_sec": float(np.std(values, ddof=0)),
                "min_validation_mae_sec": float(np.min(values)),
                "max_validation_mae_sec": float(np.max(values)),
                "mean_runtime_seconds": float(np.mean([run["runtime_seconds"] for run in candidate_runs])),
                "parameter_count": candidate_runs[0]["parameter_count"],
            })
        selected = min(range(len(candidates)), key=lambda index: candidate_scores[index])
        results.update(status="complete", candidate_summaries=candidate_summaries,
                       selected_candidate=candidates[selected]["name"])
    else:
        results.update(status="incomplete", candidate_selection=None,
                       note="A capped pilot is throughput evidence only; it does not select a candidate.")
    results["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(results_path, results)
    return preregistration, results
