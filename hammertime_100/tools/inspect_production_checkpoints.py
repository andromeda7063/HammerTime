"""Read-only inventory of best/last checkpoints in one specified season-run directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _checkpoint_metadata(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            return {"error": "checkpoint root is not a mapping"}
        fields = ("epoch", "completed_epoch", "best_epoch", "val_mae", "best_val_mae",
                  "run_id", "resume_signature")
        return {key: state[key] for key in fields if key in state and isinstance(
            state[key], (str, int, float, bool, type(None)))}
    except Exception as exc:  # report only class; never dump checkpoint contents
        return {"error": type(exc).__name__}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path,
                        help="One specific prior runner output containing training/<fingerprint>/best.pt.")
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve(strict=True)
    candidates = sorted(root.glob("training/*/best.pt"))
    if not candidates:
        parser.error(f"No training/*/best.pt checkpoints found under {root}")
    rows = []
    for best in candidates:
        last = best.with_name("best.last.pt")
        metrics = []
        for name in ("test_metrics.json", "validation_metrics.json", "metrics.json",
                     "training_complete.json"):
            path = best.parent / name
            if path.is_file():
                try:
                    metrics.append({"file": str(path), "content": json.loads(path.read_text(encoding="utf-8"))})
                except (OSError, ValueError):
                    metrics.append({"file": str(path), "error": "invalid json"})
        rows.append({"best_checkpoint": str(best), "best_metadata": _checkpoint_metadata(best),
                     "last_checkpoint": str(last) if last.is_file() else None,
                     "last_metadata": _checkpoint_metadata(last) if last.is_file() else None,
                     "nearby_run_metrics": metrics})
    print(json.dumps({"run_root": str(root), "read_only": True, "candidates": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
