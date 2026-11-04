"""GNNv4 runner: build/reuse a memory-safe cohort_v4, train, and evaluate held-out races.

Local data only: it reads an existing verified normalized source cache
(``<run>/race_cache`` containing ``normalized_source_manifest.json``) and never
requests FastF1/OpenF1 data. Existing run directories, caches and checkpoints
are only read; all outputs go to a new ``--output`` directory and a
fingerprinted cohort directory under ``--cohort-root``.

Example::

    uv run --frozen python hammertime_100/train_gnnv4.py \
        --normalized-source <old-run>/race_cache \
        --output .local/research-history/gnnv4-<timestamp> \
        --device auto

Gates, metrics and physics thresholds are reported unchanged; supplementary
diagnostics are labelled as such. A finished run is a candidate result that
still needs review before any research claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

LOG = logging.getLogger("hammertime.gnnv4")
DEFAULT_CONFIG = HERE / "configs" / "train_2023_fastf1_season_gnnv4.yaml"
REPO_ROOT = HERE.parent
COHORT_SECTIONS = ("graph", "target", "track", "preprocess", "data", "review_flags")


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
                          .encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    return value


def resolve_device(requested: str):
    """Return (torch.device, note). CUDA falls back to CPU with a warning."""
    import torch
    if requested == "cpu":
        return torch.device("cpu"), "cpu requested"
    available = False
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - broken driver installs
        LOG.warning("CUDA probe failed (%s); using CPU.", type(exc).__name__)
    if available:
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"cuda ({name})"
    if requested == "cuda":
        LOG.warning("CUDA was requested but is unavailable (torch %s, CUDA build %s); falling back to CPU.",
                    torch.__version__, torch.version.cuda)
    return torch.device("cpu"), "cpu (CUDA unavailable)"


def apply_cli_overrides(config: dict, args) -> dict:
    config = copy.deepcopy(config)
    train = config["train"]
    if args.epochs is not None:
        train["max_epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train["batch_size"] = int(args.batch_size)
    if args.grad_accum is not None:
        train["grad_accum_steps"] = int(args.grad_accum)
    if args.patience is not None:
        train["early_stopping_patience"] = int(args.patience)
    train["num_workers"] = int(args.workers)
    train["log_every_batches"] = int(args.log_every_batches)
    config["data"]["random_seed"] = int(args.seed)
    return config


def cohort_fingerprint(config: dict, source_hash: str) -> str:
    from cohort_v4 import COHORT_SCHEMA_VERSION, code_fingerprint
    identity = {section: config.get(section) for section in COHORT_SECTIONS}
    identity["future_speed"] = bool(config.get("auxiliary", {}).get("future_speed_enabled", False))
    identity["in_dim"] = config["model"]["in_dim"]
    identity["edge_dim"] = config["model"]["edge_dim"]
    # Training-only knobs never change stored arrays.
    identity["data"] = {k: v for k, v in (identity["data"] or {}).items() if k != "random_seed"}
    return _canonical_hash({"identity": identity, "source": source_hash, "code": code_fingerprint(),
                            "schema": COHORT_SCHEMA_VERSION})


def _source_manifest_hash(normalized_source: Path) -> str:
    from cohort_v4 import _sha256_file
    path = normalized_source / "normalized_source_manifest.json"
    sidecar = Path(str(path) + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"{normalized_source} is not a normalized source cache "
                                "(normalized_source_manifest.json and its .sha256 are required).")
    digest = _sha256_file(path)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("Normalized source manifest does not match its SHA-256 sidecar.")
    return digest


def evaluate_run(model, loaders, config, device, checkpoint, out_dir: Path, cohort, args) -> dict:
    import torch
    from cohort_v4 import create_compact_dataloaders
    from evaluate import (compute_metrics, drs_train_per_sector_estimate, predict_test_set,
                          supplementary_metrics, validate_checkpoint_edge_scaler,
                          validate_clean_air_physics, validate_drs_train_physics)
    import pandas as pd

    train_loader, _, test_loader = loaders
    model.load_state_dict(checkpoint["model_state_dict"])
    validate_checkpoint_edge_scaler(checkpoint, model, test_loader)
    saved = checkpoint.get("feature_scaler") or {}
    if (not np.allclose(cohort.feature_scaler.mean_, saved.get("mean_", []))
            or not np.allclose(cohort.feature_scaler.scale_, saved.get("scale_", []))):
        raise ValueError("Best checkpoint scaler does not match the cohort's train-only scaler.")
    y_true, y_pred, records = predict_test_set(model, test_loader, device)
    physics_cfg = config["physics_tests"]
    mini_sectors = int(config["track"]["mini_sectors"])

    def gates(rows):
        return [
            validate_clean_air_physics(rows, float(physics_cfg.get("clean_air_max_mae", 0.05)),
                                       float(config["target"].get("clean_air_gap_sec", 3.0))),
            validate_drs_train_physics(rows, mini_sectors,
                                       float(physics_cfg.get("drs_train_min_lap_loss", 0.40)),
                                       float(physics_cfg.get("drs_train_max_lap_loss", 0.80)),
                                       float(physics_cfg.get("drs_max_gap_sec", 0.8))),
        ]

    metrics = compute_metrics(y_true, y_pred)
    metrics["rmse_sec"] = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    metrics["zero_baseline"] = compute_metrics(y_true, np.zeros_like(y_true))
    train_targets = train_loader.dataset.valid_target_values()
    supplementary = supplementary_metrics(y_true, y_pred, float(np.mean(train_targets)),
                                          float(np.median(train_targets)))
    supplementary["drs_per_sector_estimate"] = drs_train_per_sector_estimate(
        records, mini_sectors, float(physics_cfg.get("drs_max_gap_sec", 0.8)))
    if not args.skip_dense_test:
        # Supplementary only: the same test races on every labelled frame (stride 1),
        # which is what a complete-lap DRS check needs. Gate results above are unchanged.
        _, _, dense_test, _ = create_compact_dataloaders(cohort, config, num_workers=0, eval_stride=1)
        dense_true, dense_pred, dense_records = predict_test_set(model, dense_test, device)
        supplementary["dense_test_stride1"] = {
            "note": "Supplementary coverage; not the configured gate.",
            "metrics": compute_metrics(dense_true, dense_pred),
            "physics_tests": gates(dense_records),
        }
    report = {
        "metrics": metrics, "physics_tests": gates(records), "supplementary": supplementary,
        "aero_coefficients": model.aero_coefficients() if hasattr(model, "aero_coefficients") else None,
        "best_epoch": checkpoint.get("epoch"), "best_validation_mae_sec": checkpoint.get("val_mae"),
        "best_validation_r2_diagnostic": checkpoint.get("val_r2"),
        "split": cohort.manifest["split"], "sessions": cohort.manifest["sessions"],
        "cohort_manifest_sha256": cohort.manifest_sha256,
        "target_definition": ("existing target builder; full-session calibrated"
                              if not any(cohort.manifest["review_flags"].values())
                              else f"review flags ON: {cohort.manifest['review_flags']}"),
        "device": str(device),
    }
    _atomic_json(out_dir / "test_metrics.json", _json_safe(report))
    pd.DataFrame(records).to_csv(out_dir / "test_predictions.csv", index=False)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normalized-source", type=Path, required=True,
                        help="Existing verified normalized source cache (<run>/race_cache).")
    parser.add_argument("--raw-sessions-dir", type=Path, default=None,
                        help="Optional raw_sessions directory (needed only for the yellow-flag review flag).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True,
                        help="Run directory. A new directory is created; an existing one is resumed only with --resume.")
    parser.add_argument("--cohort-root", type=Path, default=REPO_ROOT / ".local" / "research-history" / "gnnv4-cohorts")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run in --output.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Verify sources, cohort status, memory and model construction; no build or training.")
    parser.add_argument("--skip-dense-test", action="store_true",
                        help="Skip the supplementary stride-1 test pass.")
    parser.add_argument("--no-verify-cohort-hashes", action="store_true",
                        help="Skip re-hashing cohort shards on load (faster; manifest is still checked).")
    args = parser.parse_args(argv)

    out_dir = args.output.expanduser().resolve()
    if out_dir.exists() and not args.resume and not args.preflight_only:
        print(f"Refusing to reuse existing output {out_dir}; pass --resume or choose a new directory.", file=sys.stderr)
        return 2
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if not args.preflight_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(out_dir / "run.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=log_handlers, force=True)

    import torch
    from cohort_v4 import (CompactCohort, MemoryPreflightError, build_compact_cohort, create_compact_dataloaders,
                           estimate_training_bytes, memory_preflight, resource_summary)
    from config_validation import validate_config
    from model import create_model
    from train import train_model

    started = time.perf_counter()
    config = yaml.safe_load(args.config.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    config = apply_cli_overrides(config, args)
    validate_config(config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device, device_note = resolve_device(args.device)
    LOG.info("GNNv4 runner | python %s | torch %s | device %s | threads %d | %s",
             platform.python_version(), torch.__version__, device_note, args.threads, resource_summary())
    if device.type == "cpu" and args.workers > 0 and os.name == "nt":
        LOG.warning("Windows DataLoader workers use spawn; memory maps are reopened per worker.")

    normalized_source = args.normalized_source.expanduser().resolve(strict=True)
    source_hash = _source_manifest_hash(normalized_source)
    fingerprint = cohort_fingerprint(config, source_hash)
    cohort_dir = args.cohort_root.expanduser().resolve() / fingerprint[:20]
    headroom = float(config.get("cohort", {}).get("memory_headroom_fraction", 0.25))
    cohort = None
    if (cohort_dir / "cohort_manifest.json").is_file():
        LOG.info("Reusing cohort_v4 %s (verifying%s).", cohort_dir,
                 " manifest only" if args.no_verify_cohort_hashes else " all shard hashes")
        cohort = CompactCohort(cohort_dir, verify_hashes=not args.no_verify_cohort_hashes)
    elif args.preflight_only:
        from cohort_v4 import _read_source, estimate_build_bytes
        _specs, _records, aligned_paths, _labels, _ = _read_source(normalized_source)
        memory_preflight("cohort build (one race at a time)", estimate_build_bytes(aligned_paths), headroom)
        LOG.info("Preflight: normalized source verified (%d races); cohort %s would be built on the first run.",
                 len(aligned_paths), cohort_dir)
    else:
        LOG.info("Building cohort_v4 at %s from %s", cohort_dir, normalized_source)
        build_compact_cohort(normalized_source, config, cohort_dir, raw_sessions_dir=args.raw_sessions_dir,
                             headroom_fraction=headroom)
        cohort = CompactCohort(cohort_dir, verify_hashes=False)

    num_nodes = int(cohort.manifest["num_nodes"]) if cohort else int(config["graph"]["num_nodes"])
    try:
        memory_preflight("training", estimate_training_bytes(config, num_nodes, int(config["model"]["in_dim"]),
                                                             args.workers), headroom)
    except MemoryPreflightError as exc:
        LOG.error("%s", exc)
        return 3
    model = create_model(config).to(device)
    LOG.info("Model constructed: %d parameters; aero head %s.", sum(p.numel() for p in model.parameters()),
             "on" if getattr(model, "aero_head", None) is not None else "off")
    if args.preflight_only:
        LOG.info("PRE-FLIGHT PASSED: source verified, memory ok, model constructed on %s; no build, no training.",
                 device)
        return 0

    config["graph"]["num_nodes"] = num_nodes
    loaders = create_compact_dataloaders(cohort, config, num_workers=args.workers)
    train_loader, val_loader, test_loader, _ = loaders
    training_dir = out_dir / "training"
    config["train"]["checkpoint_path"] = str(training_dir / "best.pt")
    config["train"]["last_checkpoint_path"] = str(training_dir / "best.last.pt")
    config["train"]["history_path"] = str(training_dir / "training_history.csv")
    _atomic_json(out_dir / "run_manifest.json", _json_safe({
        "cohort_dir": str(cohort_dir), "cohort_manifest_sha256": cohort.manifest_sha256,
        "normalized_source": str(normalized_source), "normalized_source_manifest_sha256": source_hash,
        "config_path": str(args.config), "effective_config": config, "device": device_note,
        "argv": sys.argv[1:], "windows": {"train": len(train_loader.dataset), "validation": len(val_loader.dataset),
                                          "test": len(test_loader.dataset)}}))
    (out_dir / "effective_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    last_path = Path(config["train"]["last_checkpoint_path"])
    LOG.info("Training: %d/%d/%d windows; %s", len(train_loader.dataset), len(val_loader.dataset),
             len(test_loader.dataset), resource_summary())
    try:
        train_model(model, train_loader, val_loader, config, device,
                    resume_from=last_path if last_path.is_file() else None)
    except torch.cuda.OutOfMemoryError:
        LOG.warning("CUDA out of memory; continuing on CPU from the last complete epoch (if any).")
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        model = create_model(config).to(device)
        train_model(model, train_loader, val_loader, config, device,
                    resume_from=last_path if last_path.is_file() else None)
    checkpoint = torch.load(config["train"]["checkpoint_path"], map_location=device, weights_only=False)
    report = evaluate_run(model, (train_loader, val_loader, test_loader), config, device, checkpoint,
                          out_dir, cohort, args)
    m = report["metrics"]
    LOG.info("Held-out test: MAE=%.4f s, R2=%s (gate unchanged: MAE<0.15 %s, R2 in [0.68,0.82] %s); "
             "physics: %s | %.1f min total",
             m["mae_sec"], m["r2_score"], m["target_mae_pass"], m["target_r2_pass"],
             ", ".join(f"{t['test']}={'pass' if t['passed'] else 'fail'}" for t in report["physics_tests"]),
             (time.perf_counter() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
