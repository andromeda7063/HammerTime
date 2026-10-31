"""Run one declared validation-only ablation candidate on a frozen cohort."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ablation import ABLATION_CANDIDATES, run_ablation_study  # noqa: E402
from dataset import create_dataloaders  # noqa: E402
from prepare_cohort import load_prepared_cohort  # noqa: E402


def _membership(loader) -> list[dict]:
    return [
        {
            "race_id": sample.get("race_id"),
            "input_start_time": float(sample["input_start_time"]),
            "target_time": float(sample["target_time"]),
        }
        for sample in loader.dataset.samples
    ]


def _load_matching_config(path: Path, prepared: dict) -> dict:
    with path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration is not a mapping: {path}")
    if json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False) != json.dumps(
            prepared["config"], sort_keys=True, separators=(",", ":"), allow_nan=False):
        raise ValueError("--config must exactly match the configuration frozen into the prepared cohort.")
    return config


def train_prepared(artifact_path: Path, config_path: Path, output_dir: Path,
                   candidate_name: str, epochs: int | None, seed: int | None,
                   device: str) -> dict:
    artifact_path = artifact_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output_dir}")

    payload, manifest = load_prepared_cohort(artifact_path)
    config = _load_matching_config(config_path, payload)
    frozen_seed = int(config["data"]["random_seed"])
    selected_seed = frozen_seed if seed is None else seed
    if isinstance(selected_seed, bool) or not isinstance(selected_seed, int) or selected_seed < 0:
        raise ValueError("--seed must be a non-negative integer.")
    if selected_seed != frozen_seed:
        raise ValueError(
            f"--seed {selected_seed} differs from the frozen data.random_seed {frozen_seed}; "
            "prepare a separately frozen cohort to change the split seed."
        )
    if epochs is not None and (isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1):
        raise ValueError("--epochs must be a positive integer.")

    candidates = {candidate["name"]: index for index, candidate in enumerate(ABLATION_CANDIDATES)}
    if candidate_name not in candidates:
        raise ValueError(f"Unknown candidate {candidate_name!r}; choose one of {', '.join(candidates)}.")
    candidate_index = candidates[candidate_name]

    # Loader construction fits the scaler on train windows only. Work on a copy:
    # the artifact remains immutable and can be reused for another declared run.
    snapshots = copy.deepcopy(payload["snapshots"])
    train_loader, val_loader, _test_loader, _scaler = create_dataloaders(
        snapshots, payload["labels"], payload["driver_to_idx"], config
    )
    rebuilt_membership = {
        "train": _membership(train_loader),
        "validation": _membership(val_loader),
    }
    for split_name, rows in rebuilt_membership.items():
        if rows != payload["memberships"].get(split_name):
            raise ValueError(f"Rebuilt {split_name} membership differs from the frozen cohort manifest.")

    _preregistration, results = run_ablation_study(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        study_dir=output_dir,
        source_provenance=payload["provenance"],
        test_exposure=(
            "The test partition is never passed to this runner or scored. "
            "The prior single-session Monza test result was exposed; the current "
            "cohort's test partition is reserved and remains unevaluated."
        ),
        seeds=(selected_seed,),
        mode="pilot",
        max_runs=1,
        epoch_cap=epochs,
        candidate_indices=[candidate_index],
    )
    return {
        "output_dir": str(output_dir),
        "candidate": candidate_name,
        "seed": selected_seed,
        "epochs": epochs if epochs is not None else int(config["train"]["max_epochs"]),
        "provider": manifest["provider"],
        "session_identity": manifest["session_identity"],
        "status": results["status"],
        "completed_runs": results.get("completed_runs", len(results.get("runs", []))),
        "test_metrics_evaluated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path,
                        help="prepared_cohort.pkl produced by src/prepare_cohort.py")
    parser.add_argument("--config", required=True, type=Path,
                        help="Exact YAML config frozen into the prepared cohort")
    parser.add_argument("--output", required=True, type=Path,
                        help="New validation-only run directory; existing output is never replaced")
    parser.add_argument("--candidate", required=True,
                        help="Declared ablation candidate name (for example gated_dilated_full_history)")
    parser.add_argument("--epochs", type=int,
                        help="Optional epoch cap; capped runs are explicitly recorded as pilots")
    parser.add_argument("--seed", type=int,
                        help="Seed frozen in the cohort config (defaults to data.random_seed)")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    args = parser.parse_args(argv)
    result = train_prepared(args.prepared, args.config, args.output, args.candidate,
                            args.epochs, args.seed, args.device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
