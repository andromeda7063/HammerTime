"""Pre-registered validation-only runs for the fixed ST-GNN ablation set."""

import copy
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
from evaluate import compute_metrics
from model import create_model
from train import train_model


ABLATION_CANDIDATES = [
    {"name": "current_k3", "temporal_kernel": 3, "temporal_mode": "conv", "graph": "current"},
    {"name": "self_loop_only_k3", "temporal_kernel": 3, "temporal_mode": "conv", "graph": "self_loop_only"},
    {"name": "temporal_kernel_1", "temporal_kernel": 1, "temporal_mode": "conv", "graph": "current"},
    {"name": "temporal_kernel_10", "temporal_kernel": 10, "temporal_mode": "conv", "graph": "current"},
    {"name": "constant_edge_attribute_k3", "temporal_kernel": 3,
     "temporal_mode": "conv", "graph": "constant_edge_attribute", "constant": 1.0},
    {"name": "gated_dilated_full_history", "temporal_kernel": 3,
     "temporal_mode": "gated_dilated", "temporal_dilations": [1, 2, 4], "graph": "current"},
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
    module_names = ("ablation.py", "dataset.py", "model.py", "train.py", "evaluate.py")
    modules = {}
    for name in module_names:
        path = repo / "hammertime_100" / "src" / name
        modules[name] = hashlib.sha256(path.read_bytes()).hexdigest()
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
                for tensor_name in ("x", "edge_index", "edge_attr", "mask"):
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


def _clone_transformed_samples(samples, graph_mode):
    transformed = []
    for sample in samples:
        item = dict(sample)
        snapshots = []
        for original in sample["snapshots"]:
            snap = original.clone() if hasattr(original, "clone") else copy.deepcopy(original)
            if graph_mode == "self_loop_only":
                snap.edge_index = torch.empty((2, 0), dtype=torch.long,
                                              device=snap.x.device)
                snap.edge_attr = torch.empty((0, 1), dtype=torch.float32,
                                             device=snap.x.device)
            elif graph_mode == "constant_edge_attribute":
                edge_index = getattr(snap, "edge_index", None)
                edge_count = int(edge_index.size(1)) if edge_index is not None else 0
                snap.edge_attr = torch.ones((edge_count, 1), dtype=torch.float32,
                                            device=snap.x.device)
            elif graph_mode != "current":
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
def _validation_metrics(model, loader, device):
    if not isinstance(getattr(loader, "sampler", None), SequentialSampler):
        raise ValueError("Ablation scoring requires sequential validation batches.")
    model.eval()
    targets, predictions = [], []
    for windows, target, mask in loader:
        prediction = model(windows)
        valid = mask.to(device=device, dtype=torch.bool)
        targets.extend(target.to(device).masked_select(valid).detach().cpu().tolist())
        predictions.extend(prediction.masked_select(valid).detach().cpu().tolist())
    return compute_metrics(np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float))


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
    os.replace(temporary, path)


def run_ablation_study(train_loader, val_loader, config, device, study_dir,
                       source_provenance, test_exposure, seeds=(42, 43, 44),
                       mode="complete", max_runs=None, epoch_cap=None,
                       candidate_indices=None):
    """Run a pre-registered candidate prefix; this API accepts no test loader."""
    device = torch.device(device)
    if mode not in ("pilot", "complete"):
        raise ValueError("mode must be 'pilot' or 'complete'.")
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
    if candidate_indices is None:
        registered_candidate_indices = list(range(len(ABLATION_CANDIDATES)))
    else:
        if (not isinstance(candidate_indices, (list, tuple)) or not candidate_indices
                or any(isinstance(index, bool) or not isinstance(index, int)
                       or not 0 <= index < len(ABLATION_CANDIDATES) for index in candidate_indices)
                or len(set(candidate_indices)) != len(candidate_indices)):
            raise ValueError("candidate_indices must be a non-empty sequence of unique valid candidate indexes.")
        registered_candidate_indices = list(candidate_indices)
    if mode == "complete" and registered_candidate_indices != list(range(len(ABLATION_CANDIDATES))):
        raise ValueError("A complete study must include every declared candidate.")
    total_runs = len(registered_candidate_indices) * len(seeds)
    run_limit = total_runs if max_runs is None else int(max_runs)
    if not 1 <= run_limit <= total_runs:
        raise ValueError(f"max_runs must be in [1, {total_runs}].")
    if mode == "complete" and run_limit != total_runs:
        raise ValueError("A complete study must run every candidate across every matched seed.")
    if mode == "complete" and len(seeds) < 3:
        raise ValueError("A complete ablation study requires at least three matched seeds.")
    if mode == "complete" and epoch_cap is not None:
        raise ValueError("A capped-epoch run is a pilot and cannot be registered as complete.")
    if epoch_cap is not None and (isinstance(epoch_cap, bool) or int(epoch_cap) < 1):
        raise ValueError("epoch_cap must be a positive integer when supplied.")

    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=False)
    candidates = copy.deepcopy(ABLATION_CANDIDATES)
    effective_config = copy.deepcopy(config)
    if epoch_cap is not None:
        effective_config["train"]["max_epochs"] = int(epoch_cap)
        effective_config["train"]["early_stopping_patience"] = min(
            int(effective_config["train"].get("early_stopping_patience", epoch_cap)), int(epoch_cap)
        )
    split_metadata = copy.deepcopy(getattr(train_loader.dataset, "split_metadata", {}))
    feature_scaler = _feature_scaler_metadata(train_loader, val_loader)
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
        "validation_zero_prediction_baseline": _zero_prediction_validation_baseline(val_loader),
        "validation_zero_prediction_baseline_role": "context only; not a candidate and not used for selection",
        "candidates": candidates,
        "registered_candidate_indices": registered_candidate_indices,
        "seeds": list(seeds),
        "run_order": "candidate declaration order, then seed order",
        "max_runs": run_limit,
        "full_study_run_count": total_runs,
        "max_epochs": int(effective_config["train"].get("max_epochs", 100)),
        "selection_metric": "mean_validation_mae_sec",
        "selection_rule": "lowest mean validation MAE; ties resolved by declared candidate order",
        "reported_validation_metrics": ["mae_sec", "r2_score", "sample_count"],
        "parameter_count": "trainable model parameters",
        "runtime_definition": "training plus best-checkpoint validation scoring; excludes preprocessing and loader construction",
        "constant_edge_attribute": 1.0,
        "self_loop_edge_attribute": 1.0,
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
    try:
        for candidate_index in registered_candidate_indices:
            candidate = candidates[candidate_index]
            for seed in seeds:
                if completed >= run_limit:
                    break
                trial_config = copy.deepcopy(effective_config)
                trial_config["model"]["conv1d_kernel"] = int(candidate["temporal_kernel"])
                trial_config["model"]["temporal_mode"] = candidate["temporal_mode"]
                trial_config["model"]["temporal_dilations"] = candidate.get(
                    "temporal_dilations", [1, 2, 4]
                )
                trial_config["data"]["random_seed"] = int(seed)
                trial_dir = study_dir / f"candidate_{candidate_index:02d}" / f"seed_{seed}"
                trial_dir.mkdir(parents=True, exist_ok=False)
                trial_config["train"]["checkpoint_path"] = str(trial_dir / "best.pt")
                trial_config["train"]["history_path"] = str(trial_dir / "training_history.csv")

                train_samples = _clone_transformed_samples(train_loader.dataset.samples, candidate["graph"])
                val_samples = _clone_transformed_samples(val_loader.dataset.samples, candidate["graph"])
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
                metrics = _validation_metrics(model, trial_val, device)
                runtime = time.perf_counter() - started
                if not math.isfinite(float(best_val_mae)) or not math.isclose(
                        float(best_val_mae), metrics["mae_sec"], rel_tol=1e-5, abs_tol=1e-7):
                    raise FloatingPointError("Reloaded best checkpoint does not reproduce its validation MAE.")
                results["runs"].append({
                    "candidate": candidate["name"], "candidate_index": candidate_index,
                    "seed": int(seed), "validation_metrics": metrics,
                    "best_checkpoint_validation_mae_sec": float(best_val_mae),
                    "epochs": len(history), "parameter_count": parameter_count,
                    "runtime_seconds": float(runtime), "checkpoint": str(checkpoint_path),
                })
                completed += 1
                results["completed_runs"] = completed
                _write_json(results_path, results)
            if completed >= run_limit:
                break
    except BaseException as error:
        results.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        _write_json(results_path, results)
        raise

    if mode == "complete" and completed == total_runs:
        candidate_scores = []
        candidate_summaries = []
        for index, candidate in enumerate(candidates):
            candidate_runs = [run for run in results["runs"] if run["candidate_index"] == index]
            values = [run["validation_metrics"]["mae_sec"] for run in candidate_runs]
            candidate_scores.append(float(np.mean(values)))
            candidate_summaries.append({
                "candidate": candidate["name"],
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
