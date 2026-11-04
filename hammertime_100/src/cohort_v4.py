"""Memory-safe compact cohort (``cohort_v4``) with lazily materialized windows.

The legacy prepared cohort pickles every PyG ``Data`` snapshot of the season
(about 0.5 M objects, ~12 GiB resident when loaded). This module stores each
race as plain NumPy arrays that are memory-mapped at load time:

    races/<race>/frames_x.npy        float32 [T, N, F]  raw (unscaled) node features
    races/<race>/frames_mask.npy     bool    [T, N]     active-node mask
    races/<race>/frames_time.npy     float64 [T]
    races/<race>/frames_coverage.npy float32 [T]        physical position coverage
    races/<race>/edge_offsets.npy    int64   [T + 1]    per-frame slice into edge arrays
    races/<race>/edge_index.npy      int16   [2, E]     leader -> follower, per-frame node ids
    races/<race>/edge_attr.npy       float32 [E, D]     raw physical edge attributes
    races/<race>/win_*.npy                              window table (start frame, targets, masks,
                                                        metadata, future-speed supervision)

Windows are built once per race with the existing
``dataset._build_windowed_samples_one_race`` at stride 1, so label matching,
continuity, safety-car and coverage rules are exactly those of the legacy path.
Evaluation partitions select ``start % window_stride == 0``, which reproduces
the legacy stride loop exactly. Scalers are fitted in streaming form on the
unique frames touched by training inputs and applied when an item is
materialized; stored arrays are never modified.

Build peak memory is roughly one race (aligned table + its snapshots). Training
memory is the model, one batch, and the OS page cache for mapped shards.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from context_features import (CONTEXT_V3_SCHEMA, feature_names, feature_schema_metadata,
                              node_feature_schema, prepare_context_features)
from dataset import _build_windowed_samples_one_race, _partition_races, collate_st_graph_windows
from graph import aero_raw_columns, build_race_graph_snapshots
from physical_graph import (EDGE_FEATURES_BY_SCHEMA, PHYSICAL_EDGE_CONTINUOUS_COLUMNS,
                            PhysicalEdgeFeatureScaler)

logger = logging.getLogger("hammertime.cohort_v4")

COHORT_FORMAT = "hammertime-compact-cohort-v4"
COHORT_SCHEMA_VERSION = 1
SRC_DIR = Path(__file__).resolve().parent
CODE_FILES = ("cohort_v4.py", "context_features.py", "graph.py", "physical_graph.py",
              "dataset.py", "target.py", "preprocess.py", "review_flags.py")
FRAME_ARRAYS = ("frames_x", "frames_mask", "frames_time", "frames_coverage", "frames_sc",
                "edge_offsets", "edge_index", "edge_attr")
WINDOW_ARRAYS = ("win_start", "win_target", "win_mask", "win_gap", "win_lap", "win_sector",
                 "win_exit", "win_target_time", "win_aux", "win_aux_mask")
GIB = 1024 ** 3


# ───────────────────────────── utilities ──────────────────────────────

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=str).encode("utf-8")


def _sha256_json(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def safe_component(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "session"


def available_memory_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:  # pragma: no cover - psutil is a locked dependency
        return None


def process_rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover
        return None


def resource_summary() -> str:
    rss, available = process_rss_bytes(), available_memory_bytes()
    parts = []
    if rss is not None:
        parts.append(f"process RSS={rss / GIB:.2f} GiB")
    if available is not None:
        parts.append(f"system available={available / GIB:.2f} GiB")
    return ", ".join(parts) or "memory probe unavailable"


def code_fingerprint() -> dict:
    return {name: _sha256_file(SRC_DIR / name) for name in CODE_FILES if (SRC_DIR / name).is_file()}


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=SRC_DIR, check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


# ─────────────────────────── memory preflight ─────────────────────────

class MemoryPreflightError(MemoryError):
    """Raised before work starts when the host cannot fit a stage safely."""


def memory_preflight(stage: str, required_bytes: int, headroom_fraction: float = 0.25,
                     available_bytes: int | None = None) -> dict:
    """Fail early (with an actionable message) instead of swapping or OOM-killing.

    ``required_bytes`` is the estimated resident increase of the stage. The
    stage is allowed when ``available * (1 - headroom) >= required``.
    """
    available = available_memory_bytes() if available_bytes is None else int(available_bytes)
    result = {"stage": stage, "required_bytes": int(required_bytes),
              "available_bytes": available, "headroom_fraction": float(headroom_fraction)}
    if available is None:
        result["status"] = "unknown"
        logger.warning("Memory preflight (%s): available memory unknown; continuing.", stage)
        return result
    budget = available * (1.0 - float(headroom_fraction))
    result["budget_bytes"] = int(budget)
    if required_bytes > budget:
        result["status"] = "insufficient"
        raise MemoryPreflightError(
            f"Memory preflight failed for {stage}: needs ~{required_bytes / GIB:.2f} GiB, "
            f"only {available / GIB:.2f} GiB available ({headroom_fraction:.0%} kept free). "
            "Close other programs, lower --batch-size, use --workers 0, or run on a larger host."
        )
    result["status"] = "ok"
    logger.info("Memory preflight (%s): need ~%.2f GiB of %.2f GiB available — ok.",
                stage, required_bytes / GIB, available / GIB)
    return result


def estimate_build_bytes(aligned_paths: list[Path]) -> int:
    """Peak resident estimate for building one race at a time."""
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover
        return 4 * GIB
    peak = 0
    for path in aligned_paths:
        meta = pq.ParquetFile(path).metadata
        rows, columns = int(meta.num_rows), int(meta.num_columns)
        # aligned + context copy (~2.5x column growth) + snapshots (~6 KB / frame
        # for ~rows / 20 frames) + window samples.
        frame_bytes = rows * (columns + 30) * 8 * 3
        snapshot_bytes = (rows // 20 + 1) * 6 * 1024 * 2
        peak = max(peak, frame_bytes + snapshot_bytes)
    return int(peak + 0.5 * GIB)


def estimate_training_bytes(config: dict, num_nodes: int, in_dim: int, workers: int = 0) -> int:
    """Resident estimate for training: model, one batch with autograd, workers."""
    batch = int(config["train"]["batch_size"])
    window = int(config["data"]["window_size"])
    hidden = int(config["model"].get("hidden_dim", 64))
    activations = batch * window * num_nodes * (hidden * 48 + in_dim * 4) * 4
    worker_bytes = max(0, int(workers)) * 400 * 1024 ** 2
    return int(0.8 * GIB + activations + worker_bytes)


# ───────────────────────────── review flags ───────────────────────────

def review_flags(config: dict) -> dict:
    flags = dict(config.get("review_flags") or {})
    known = {"exclude_lap1_and_yellow_targets", "rescale_lap_distance",
             "physical_gap_for_targets", "detrend_clean_air_reference"}
    unknown = set(flags) - known
    if unknown:
        raise ValueError(f"Unknown review_flags: {sorted(unknown)}")
    for key in known:
        flags[key] = bool(flags.get(key, False))
    return flags


# ───────────────────────────── cohort build ───────────────────────────

def _read_source(normalized_source: Path):
    """Verify the normalized source cache and return (specs, records, aligned paths, labels, hash)."""
    from prepare_cohort import _read_normalized_source
    from run_pipeline import _session_specs
    import yaml

    manifest_path = normalized_source / "normalized_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_config_path = Path(manifest["config_path"]).expanduser()
    if not source_config_path.is_file():
        raise FileNotFoundError(
            f"Normalized source config is missing: {source_config_path}. The normalized cache must "
            "stay next to the run that produced it (its manifest records an absolute path).")
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    specs = _session_specs(source_config)
    provider = str(specs[0].get("source", "fastf1"))
    aligned_paths, label_tables, records, _paths, _manifest = _read_normalized_source(
        normalized_source, specs, provider, load_aligned=False)
    return specs, records, [Path(item) for item in aligned_paths], label_tables, _sha256_file(manifest_path)


def _race_frames_and_windows(snapshots: list, samples: list, edge_dim: int, num_nodes: int,
                             horizon: int, future_speed: bool) -> dict:
    ordered = sorted(snapshots, key=lambda snap: float(snap.time_sec))
    times = np.asarray([float(snap.time_sec) for snap in ordered], dtype=np.float64)
    T = len(ordered)
    F = int(ordered[0].x.size(1)) if T else 0
    x = np.zeros((T, num_nodes, F), dtype=np.float32)
    mask = np.zeros((T, num_nodes), dtype=bool)
    coverage = np.zeros(T, dtype=np.float32)
    safety = np.zeros(T, dtype=bool)
    offsets = np.zeros(T + 1, dtype=np.int64)
    edge_blocks, attr_blocks = [], []
    for index, snap in enumerate(ordered):
        rows = int(snap.x.size(0))
        x[index, :rows] = snap.x.numpy()
        mask[index, :rows] = snap.mask.numpy().astype(bool)
        coverage[index] = float(getattr(snap, "position_coverage", 0.0))
        safety[index] = bool(getattr(snap, "is_safety_car", False))
        edge_index = snap.edge_index
        edge_attr = snap.edge_attr
        if edge_attr is None or edge_attr.numel() == 0:
            edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)
        if edge_attr.size(1) != edge_dim:
            raise ValueError(f"Snapshot edge width {edge_attr.size(1)} does not match model edge_dim {edge_dim}.")
        edge_blocks.append(edge_index.numpy().astype(np.int16))
        attr_blocks.append(edge_attr.numpy().astype(np.float32))
        offsets[index + 1] = offsets[index] + int(edge_index.size(1))
    frame_of_time = {float(value): idx for idx, value in enumerate(times)}
    S = len(samples)
    win = {
        "win_start": np.zeros(S, dtype=np.int32),
        "win_target": np.zeros((S, num_nodes), dtype=np.float32),
        "win_mask": np.zeros((S, num_nodes), dtype=bool),
        "win_gap": np.full((S, num_nodes), np.nan, dtype=np.float32),
        "win_lap": np.full((S, num_nodes), np.nan, dtype=np.float32),
        "win_sector": np.full((S, num_nodes), np.nan, dtype=np.float32),
        "win_exit": np.full((S, num_nodes), np.nan, dtype=np.float64),
        "win_target_time": np.zeros(S, dtype=np.float64),
        "win_aux": np.zeros((S, num_nodes, max(horizon, 1)), dtype=np.float32),
        "win_aux_mask": np.zeros((S, num_nodes, max(horizon, 1)), dtype=bool),
    }
    for row, sample in enumerate(samples):
        win["win_start"][row] = frame_of_time[float(sample["input_start_time"])]
        n = int(sample["target"].numel())
        win["win_target"][row, :n] = sample["target"].numpy()
        win["win_mask"][row, :n] = sample["mask"].numpy()
        win["win_gap"][row, :n] = sample["gap_sec"]
        win["win_lap"][row, :n] = sample["lap_number"]
        win["win_sector"][row, :n] = sample["mini_sector"]
        win["win_exit"][row, :n] = sample["outcome_exit_time_sec"]
        win["win_target_time"][row] = float(sample["target_time"])
        if future_speed:
            win["win_aux"][row, :n] = sample["aux_speed_target"].numpy()
            win["win_aux_mask"][row, :n] = sample["aux_speed_mask"].numpy()
    edges = (np.concatenate(edge_blocks, axis=1) if edge_blocks else np.zeros((2, 0), dtype=np.int16))
    attrs = (np.concatenate(attr_blocks, axis=0) if attr_blocks else np.zeros((0, edge_dim), dtype=np.float32))
    return {"frames_x": x, "frames_mask": mask, "frames_time": times, "frames_coverage": coverage,
            "frames_sc": safety, "edge_offsets": offsets, "edge_index": edges, "edge_attr": attrs, **win}


def _max_drivers(aligned_paths: list[Path]) -> int:
    counts = [pd.read_parquet(path, columns=["Driver"])["Driver"].nunique() for path in aligned_paths]
    return int(max(counts)) if counts else 0


def build_compact_cohort(normalized_source: str | Path, config: dict, output_dir: str | Path,
                         raw_sessions_dir: str | Path | None = None,
                         headroom_fraction: float | None = None) -> dict:
    """Build a cohort_v4 directory from a verified normalized source cache.

    The destination must not exist; the build happens in a sibling staging
    directory that is renamed into place only when complete.
    """
    normalized_source = Path(normalized_source).expanduser().resolve(strict=True)
    specs, records, aligned_paths, label_tables, source_hash = _read_source(normalized_source)
    return build_compact_cohort_from_tables(
        specs, records, aligned_paths, label_tables, source_hash, config, output_dir,
        raw_sessions_dir=raw_sessions_dir, headroom_fraction=headroom_fraction,
        source_description=str(normalized_source))


def build_compact_cohort_from_tables(specs: list[dict], records: list[dict], aligned_paths: list[Path],
                                     label_tables: list[pd.DataFrame], source_hash: str, config: dict,
                                     output_dir: str | Path, raw_sessions_dir: str | Path | None = None,
                                     headroom_fraction: float | None = None,
                                     source_description: str = "") -> dict:
    """Build cohort_v4 from already verified per-race aligned parquet paths and label tables."""
    from config_validation import validate_config
    from review_flags import apply_review_flags

    started = time.perf_counter()
    config = copy.deepcopy(config)
    validate_config(config)
    schema = node_feature_schema(config)
    if schema != CONTEXT_V3_SCHEMA:
        raise ValueError("cohort_v4 is built for graph.node_feature_schema='circuit_context_v3'.")
    edge_schema = config["graph"]["edge_feature_schema"]
    edge_dim = len(EDGE_FEATURES_BY_SCHEMA[edge_schema])
    flags = review_flags(config)
    if flags["exclude_lap1_and_yellow_targets"] and raw_sessions_dir is None:
        raise ValueError("review_flags.exclude_lap1_and_yellow_targets needs --raw-sessions-dir (TrackStatus).")
    headroom = float(config.get("cohort", {}).get("memory_headroom_fraction", 0.25)
                     if headroom_fraction is None else headroom_fraction)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Compact cohort output already exists: {output_dir}")
    aligned_paths = [Path(path) for path in aligned_paths]
    preflight = memory_preflight("cohort build (one race at a time)",
                                 estimate_build_bytes(aligned_paths), headroom)
    num_nodes = max(int(config["graph"]["num_nodes"]), _max_drivers(aligned_paths))
    config["graph"]["num_nodes"] = num_nodes
    names = feature_names(schema, config["graph"]["compound_categories"])
    data_cfg = config["data"]
    W, K = int(data_cfg["window_size"]), int(data_cfg["horizon"])
    period = 1.0 / float(config["preprocess"]["sample_hz"])
    future_speed = bool(config.get("auxiliary", {}).get("future_speed_enabled", False))
    sessions_by_race = {}
    for spec in specs:
        if not spec.get("circuit_id") or not spec.get("lap_length_m"):
            raise ValueError(f"Context features need circuit_id and lap_length_m for {spec['race_id']}.")
        sessions_by_race[str(spec["race_id"])] = {"circuit_id": str(spec["circuit_id"]),
                                                  "lap_length_m": float(spec["lap_length_m"])}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        races_meta, file_hashes = [], {}
        label_counts = {}
        for spec, aligned_path, labels in zip(specs, aligned_paths, label_tables):
            race_id = str(spec["race_id"])
            race_started = time.perf_counter()
            aligned = pd.read_parquet(aligned_path)
            race_config = copy.deepcopy(config)
            race_config["session"].update(spec)
            race_config["session"]["race_id"] = race_id
            race_config["track"]["lap_length_m"] = float(spec["lap_length_m"])
            race_dir = staging / "races" / safe_component(race_id)
            race_dir.mkdir(parents=True, exist_ok=True)
            race_config["session"]["processed_dir"] = str(race_dir / "rebuilt_targets")
            aligned, labels, flag_report = apply_review_flags(
                aligned, labels, race_config, flags, raw_sessions_dir=raw_sessions_dir)
            contextual = prepare_context_features(aligned, {race_id: sessions_by_race[race_id]},
                                                  sample_hz=float(config["preprocess"]["sample_hz"]),
                                                  mini_sectors=int(config["track"]["mini_sectors"]),
                                                  schema=schema)
            del aligned
            snapshots, driver_to_idx = build_race_graph_snapshots(contextual, race_config)
            del contextual
            samples = _build_windowed_samples_one_race(
                snapshots, labels, driver_to_idx, W, K, period,
                float(data_cfg["timestamp_tolerance_sec"]), float(data_cfg["continuity_tolerance_sec"]),
                int(data_cfg["min_valid_targets_per_sample"]), stride=1,
                future_speed_enabled=future_speed)
            driver_names = [None] * num_nodes
            for driver, index in driver_to_idx.items():
                driver_names[index] = str(driver)
            arrays = _race_frames_and_windows(snapshots, samples, edge_dim, num_nodes, K, future_speed)
            del snapshots, samples
            for name, array in arrays.items():
                path = race_dir / f"{name}.npy"
                np.save(path, array, allow_pickle=False)
            meta = {"race_id": race_id, "directory": f"races/{safe_component(race_id)}",
                    "frames": int(arrays["frames_x"].shape[0]),
                    "edges": int(arrays["edge_index"].shape[1]),
                    "windows_stride1": int(arrays["win_start"].shape[0]),
                    "driver_names": driver_names, "circuit_id": sessions_by_race[race_id]["circuit_id"],
                    "lap_length_m": float(spec["lap_length_m"]), "review_flags": flag_report}
            (race_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            shutil.rmtree(race_dir / "rebuilt_targets", ignore_errors=True)
            label_counts[race_id] = int(len(labels))
            races_meta.append(meta)
            del arrays, labels
            gc.collect()
            logger.info("cohort_v4 %s: %d frames, %d windows (stride 1) in %.1fs; %s",
                        race_id, meta["frames"], meta["windows_stride1"],
                        time.perf_counter() - race_started, resource_summary())

        split = _split_races(races_meta, config)
        scaler_state = _fit_streaming_scalers(staging, races_meta, split["train"], config, edge_schema)
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                rel = path.relative_to(staging).as_posix()
                file_hashes[rel] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        manifest = {
            "artifact_type": COHORT_FORMAT, "schema_version": COHORT_SCHEMA_VERSION,
            "created_unix": time.time(), "git_head": _git_head(),
            "platform": platform.platform(), "python": platform.python_version(),
            "config": config, "config_sha256": _sha256_json(config),
            "normalized_source": source_description, "normalized_source_manifest_sha256": source_hash,
            "sessions": [{key: record.get(key) for key in ("race_id", "provider", "year", "gp",
                                                            "session_type", "session_key")}
                         for record in records],
            "code_sha256": code_fingerprint(), "review_flags": flags,
            "node_feature_schema": feature_schema_metadata(schema, config["graph"]["compound_categories"]),
            "edge_feature_schema": edge_schema, "num_nodes": num_nodes, "in_dim": len(names),
            "edge_dim": edge_dim, "window_size": W, "horizon": K,
            "aero_raw_columns": list(aero_raw_columns(names)),
            "races": races_meta, "label_counts": label_counts, "split": split,
            "scalers": scaler_state, "files": file_hashes,
            "build_memory_preflight": preflight,
            "test_metrics_evaluated": False, "training_started": False,
        }
        manifest["manifest_sha256"] = _sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        (staging / "cohort_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                                      encoding="utf-8")
        os.replace(staging, output_dir)
        logger.info("cohort_v4 complete: %s in %.1fs", output_dir, time.perf_counter() - started)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _split_races(races_meta: list[dict], config: dict) -> dict:
    """Race-held-out split identical to dataset._partition_races (race order = session order)."""
    data_cfg = config["data"]
    if data_cfg.get("split_method", "race") != "race":
        raise ValueError("cohort_v4 supports data.split_method='race' only.")
    stubs = [{"race_id": meta["race_id"]} for meta in races_meta if meta["windows_stride1"] > 0]
    groups = _partition_races(stubs, float(data_cfg["train_ratio"]), float(data_cfg["val_ratio"]))
    if groups is None:
        raise ValueError("Race split requires at least three races with valid windows.")
    return {"method": "race", "train": [item["race_id"] for item in groups[0]],
            "validation": [item["race_id"] for item in groups[1]],
            "test": [item["race_id"] for item in groups[2]],
            "train_window_stride": int(data_cfg.get("train_window_stride", data_cfg.get("window_stride", 1))),
            "eval_window_stride": int(data_cfg.get("window_stride", 1))}


def _select_windows(starts: np.ndarray, stride: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(starts) % max(1, int(stride)) == 0)


def _fit_streaming_scalers(root: Path, races_meta: list[dict], train_races: list[str],
                           config: dict, edge_schema: str) -> dict:
    """Fit node/edge scalers on unique frames touched by training inputs only."""
    W = int(config["data"]["window_size"])
    stride = int(config["data"].get("train_window_stride", config["data"].get("window_stride", 1)))
    node_scaler = StandardScaler()
    edge_count, edge_sum, edge_sq = 0, np.zeros(2), np.zeros(2)
    frames_used = 0
    for meta in races_meta:
        if meta["race_id"] not in train_races:
            continue
        directory = root / meta["directory"]
        starts = np.load(directory / "win_start.npy")
        starts = starts[_select_windows(starts, stride)]
        frames = np.zeros(meta["frames"], dtype=bool)
        for start in starts:
            frames[start:start + W] = True
        selected = np.flatnonzero(frames)
        frames_used += int(selected.size)
        x = np.load(directory / "frames_x.npy", mmap_mode="r")
        mask = np.load(directory / "frames_mask.npy", mmap_mode="r")
        for chunk in np.array_split(selected, max(1, selected.size // 4096 + 1)):
            if chunk.size == 0:
                continue
            rows = np.asarray(x[chunk])[np.asarray(mask[chunk])]
            if rows.size:
                node_scaler.partial_fit(rows.astype(np.float64))
        offsets = np.load(directory / "edge_offsets.npy")
        attrs = np.load(directory / "edge_attr.npy", mmap_mode="r")
        if selected.size and attrs.shape[0]:
            take = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in selected]) \
                if selected.size else np.zeros(0, dtype=np.int64)
            if take.size:
                values = np.asarray(attrs[take][:, list(PHYSICAL_EDGE_CONTINUOUS_COLUMNS)], dtype=np.float64)
                values = values[np.isfinite(values).all(axis=1)]
                edge_count += int(values.shape[0])
                edge_sum += values.sum(axis=0)
                edge_sq += np.square(values).sum(axis=0)
    if not hasattr(node_scaler, "mean_"):
        raise ValueError("No active training nodes are available to fit the feature scaler.")
    if edge_count:
        edge_mean = edge_sum / edge_count
        edge_var = np.maximum(edge_sq / edge_count - np.square(edge_mean), 0.0)
        edge_scale = np.sqrt(edge_var)
        edge_scale[edge_scale == 0.0] = 1.0
    else:
        edge_mean, edge_var, edge_scale = np.zeros(2), np.zeros(2), np.ones(2)
    return {
        "node": {"mean_": node_scaler.mean_.tolist(), "var_": node_scaler.var_.tolist(),
                 "scale_": node_scaler.scale_.tolist(), "n_samples_seen_": int(node_scaler.n_samples_seen_),
                 "n_features_in_": int(node_scaler.n_features_in_)},
        "edge": {"schema": edge_schema,
                 "continuous_features": [EDGE_FEATURES_BY_SCHEMA["physical_v1"][i]
                                         for i in PHYSICAL_EDGE_CONTINUOUS_COLUMNS],
                 "mean_": edge_mean.tolist(), "var_": edge_var.tolist(), "scale_": edge_scale.tolist(),
                 "n_samples_seen_": edge_count,
                 "fit_scope": "unique snapshots touched by training inputs only"},
        "train_frames": frames_used, "train_window_stride": stride,
    }


# ───────────────────────────── cohort load ────────────────────────────

class CompactCohort:
    """Verified, memory-mapped cohort_v4 directory."""

    def __init__(self, directory: str | Path, verify_hashes: bool = True):
        self.directory = Path(directory).expanduser().resolve(strict=True)
        manifest_path = self.directory / "cohort_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Not a cohort_v4 directory (no cohort_manifest.json): {self.directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_type") != COHORT_FORMAT or manifest.get("schema_version") != COHORT_SCHEMA_VERSION:
            raise ValueError("Unsupported compact cohort type or schema version.")
        expected = _sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        if manifest.get("manifest_sha256") != expected:
            raise ValueError("Compact cohort manifest canonical hash is invalid.")
        if verify_hashes:
            for rel, info in manifest["files"].items():
                path = self.directory / rel
                if not path.is_file() or path.stat().st_size != info["bytes"] or _sha256_file(path) != info["sha256"]:
                    raise ValueError(f"Compact cohort file is missing or corrupt: {path}")
        self.manifest = manifest
        self.races = manifest["races"]
        self.race_index = {meta["race_id"]: i for i, meta in enumerate(self.races)}
        node = manifest["scalers"]["node"]
        self.feature_scaler = StandardScaler()
        self.feature_scaler.mean_ = np.asarray(node["mean_"], dtype=np.float64)
        self.feature_scaler.var_ = np.asarray(node["var_"], dtype=np.float64)
        self.feature_scaler.scale_ = np.asarray(node["scale_"], dtype=np.float64)
        self.feature_scaler.n_features_in_ = int(node["n_features_in_"])
        self.feature_scaler.n_samples_seen_ = int(node["n_samples_seen_"])
        self.edge_feature_scaler = PhysicalEdgeFeatureScaler.from_metadata(manifest["scalers"]["edge"])
        self._cache: dict[int, dict] = {}

    @property
    def manifest_sha256(self) -> str:
        return self.manifest["manifest_sha256"]

    def __getstate__(self):
        # Never pickle memory maps into DataLoader workers (Windows spawn would copy them).
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def arrays(self, race_index: int) -> dict:
        cached = self._cache.get(race_index)
        if cached is None:
            directory = self.directory / self.races[race_index]["directory"]
            cached = {name: np.load(directory / f"{name}.npy", mmap_mode="r")
                      for name in (*FRAME_ARRAYS, *WINDOW_ARRAYS)}
            self._cache[race_index] = cached
        return cached

    def driver_to_idx(self) -> dict:
        names = sorted({name for meta in self.races for name in meta["driver_names"] if name})
        return {name: index for index, name in enumerate(names)}

    def total_bytes(self) -> int:
        return int(sum(info["bytes"] for info in self.manifest["files"].values()))


class _SampleView:
    """Lightweight per-window metadata used by evaluate.predict_test_set."""

    def __init__(self, dataset: "CompactWindowDataset"):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset.sample_metadata(index)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class CompactWindowDataset(Dataset):
    """Windows of a cohort_v4 split, materialized as PyG snapshots on access."""

    def __init__(self, cohort: CompactCohort, race_ids: list[str], stride: int, split_name: str,
                 future_speed: bool):
        self.cohort = cohort
        self.split_name = split_name
        self.stride = int(stride)
        self.future_speed = bool(future_speed)
        self.window_size = int(cohort.manifest["window_size"])
        self.aero_columns = list(cohort.manifest["aero_raw_columns"])
        entries = []
        for race_id in race_ids:
            race_index = cohort.race_index[race_id]
            starts = np.load(cohort.directory / cohort.races[race_index]["directory"] / "win_start.npy")
            selected = _select_windows(starts, self.stride)
            entries.append(np.column_stack([np.full(selected.size, race_index, dtype=np.int64),
                                            selected.astype(np.int64)]))
        self.entries = (np.concatenate(entries) if entries else np.zeros((0, 2), dtype=np.int64))
        self.feature_scaler = cohort.feature_scaler
        self.edge_feature_scaler = cohort.edge_feature_scaler
        self.driver_to_idx = cohort.driver_to_idx()
        self.samples = _SampleView(self)
        self._mean = torch.as_tensor(cohort.feature_scaler.mean_, dtype=torch.float32)
        self._scale = torch.as_tensor(cohort.feature_scaler.scale_, dtype=torch.float32)
        digest = hashlib.sha256()
        digest.update(cohort.manifest_sha256.encode("ascii"))
        digest.update(split_name.encode("ascii"))
        digest.update(str(self.stride).encode("ascii"))
        digest.update(np.ascontiguousarray(self.entries).tobytes())
        self.data_fingerprint = digest.hexdigest()

    def __len__(self):
        return int(self.entries.shape[0])

    def _snapshot(self, arrays: dict, frame: int, race_id: str, driver_names: tuple) -> Data:
        raw = torch.from_numpy(np.array(arrays["frames_x"][frame], dtype=np.float32))
        mask = torch.from_numpy(np.array(arrays["frames_mask"][frame], dtype=bool))
        x = torch.zeros_like(raw)
        if bool(mask.any()):
            x[mask] = (raw[mask] - self._mean) / self._scale
        start, stop = int(arrays["edge_offsets"][frame]), int(arrays["edge_offsets"][frame + 1])
        edge_index = torch.from_numpy(np.array(arrays["edge_index"][:, start:stop], dtype=np.int64))
        edge_attr = torch.from_numpy(np.array(arrays["edge_attr"][start:stop], dtype=np.float32))
        edge_attr = self.edge_feature_scaler.transform(edge_attr)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mask=mask,
                    aero_raw=raw[:, self.aero_columns].clone(),
                    position_coverage=float(arrays["frames_coverage"][frame]),
                    time_sec=float(arrays["frames_time"][frame]), race_id=race_id,
                    driver_names=driver_names, is_safety_car=bool(arrays["frames_sc"][frame]))

    def __getitem__(self, index):
        race_index, window = (int(v) for v in self.entries[index])
        arrays = self.cohort.arrays(race_index)
        meta = self.cohort.races[race_index]
        start = int(arrays["win_start"][window])
        names = tuple(meta["driver_names"])
        snapshots = [self._snapshot(arrays, frame, meta["race_id"], names)
                     for frame in range(start, start + self.window_size)]
        target = torch.from_numpy(np.array(arrays["win_target"][window], dtype=np.float32))
        mask = torch.from_numpy(np.array(arrays["win_mask"][window], dtype=bool))
        if self.future_speed:
            aux = torch.from_numpy(np.array(arrays["win_aux"][window], dtype=np.float32))
            aux_mask = torch.from_numpy(np.array(arrays["win_aux_mask"][window], dtype=bool))
            return snapshots, target, mask, aux, aux_mask
        return snapshots, target, mask

    def sample_metadata(self, index) -> dict:
        race_index, window = (int(v) for v in self.entries[index])
        arrays = self.cohort.arrays(race_index)
        meta = self.cohort.races[race_index]
        start = int(arrays["win_start"][window])
        times = arrays["frames_time"]
        return {"race_id": meta["race_id"], "driver_names": list(meta["driver_names"]),
                "gap_sec": np.array(arrays["win_gap"][window]),
                "lap_number": np.array(arrays["win_lap"][window]),
                "mini_sector": np.array(arrays["win_sector"][window]),
                "outcome_exit_time_sec": np.array(arrays["win_exit"][window]),
                "input_start_time": float(times[start]),
                "target_time": float(arrays["win_target_time"][window]),
                "time_sec": float(arrays["win_target_time"][window])}

    def valid_target_values(self) -> np.ndarray:
        """All valid primary targets of this split (for train-only constant baselines)."""
        values = []
        for race_index in np.unique(self.entries[:, 0]) if len(self) else []:
            arrays = self.cohort.arrays(int(race_index))
            rows = self.entries[self.entries[:, 0] == race_index, 1]
            target = np.asarray(arrays["win_target"][rows])
            mask = np.asarray(arrays["win_mask"][rows])
            values.append(target[mask])
        return np.concatenate(values) if values else np.zeros(0, dtype=np.float32)


def create_compact_dataloaders(cohort: CompactCohort, config: dict, num_workers: int = 0,
                               eval_stride: int | None = None):
    """Train/validation/test loaders over a verified cohort_v4 split."""
    split = cohort.manifest["split"]
    future_speed = bool(config.get("auxiliary", {}).get("future_speed_enabled", False))
    if future_speed != bool(cohort.manifest["config"].get("auxiliary", {}).get("future_speed_enabled", False)):
        raise ValueError("auxiliary.future_speed_enabled must match the cohort that stored aux targets.")
    train_stride = int(split["train_window_stride"])
    eval_stride = int(split["eval_window_stride"] if eval_stride is None else eval_stride)
    batch_size = int(config["train"]["batch_size"])
    seed = int(config["data"].get("random_seed", 42))
    split_metadata = {**{k: split[k] for k in ("method", "train", "validation", "test")},
                      "train_window_stride": train_stride, "eval_window_stride": eval_stride,
                      "purge_windows_per_boundary": 0}
    workers = max(0, int(num_workers))

    def loader(race_ids, stride, name, shuffle):
        dataset = CompactWindowDataset(cohort, race_ids, stride, name, future_speed)
        dataset.split_metadata = split_metadata
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_st_graph_windows, num_workers=workers,
                          generator=generator, persistent_workers=workers > 0)

    train = loader(split["train"], train_stride, "train", True)
    validation = loader(split["validation"], eval_stride, "validation", False)
    test = loader(split["test"], eval_stride, "test", False)
    for name, item in (("train", train), ("validation", validation), ("test", test)):
        if len(item.dataset) == 0:
            raise ValueError(f"cohort_v4 {name} split has no windows.")
    logger.info("Race split (cohort_v4): train=%d windows (stride %d), validation=%d, test=%d (stride %d)",
                len(train.dataset), train_stride, len(validation.dataset), len(test.dataset), eval_stride)
    return train, validation, test, cohort.feature_scaler
