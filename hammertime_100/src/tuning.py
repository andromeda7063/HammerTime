"""Select model/optimizer settings on fixed validation data, never test outcomes."""

import copy
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from uuid import uuid4

from config_validation import validate_config
from model import create_model
from train import train_model


def tune_hyperparameters(train_loader, val_loader, config: dict, device):
    """Compare a declared candidate list across seeds; retain its first seed for test.

    The caller creates and scales the split once. Trial seeds reset the training
    loader's shuffle generator so each candidate receives the same sample order.
    The mean best validation MAE selects the candidate. The first declared seed
    supplies its checkpoint, rather than choosing the luckiest seed.
    """
    study = config.get("tuning", {})
    if not isinstance(study, dict):
        raise ValueError("tuning must be a mapping containing candidates and seeds.")
    candidates = study.get("candidates")
    seeds = study.get("seeds")
    if (not isinstance(candidates, list) or not candidates
            or any(not isinstance(candidate, dict) or not candidate for candidate in candidates)):
        raise ValueError("tuning.candidates must be a non-empty list of parameter mappings.")
    if (not isinstance(seeds, list) or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("tuning.seeds must contain distinct integer seeds in [0, 2**32).")
    if seeds[0] != config["data"]["random_seed"]:
        raise ValueError("The first tuning seed must match data.random_seed for the retained checkpoint.")
    allowed = {"train.lr", "train.weight_decay", "model.hidden_dim", "model.dropout"}
    trial_configs = []
    for candidate in candidates:
        if set(candidate).difference(allowed):
            raise ValueError(f"Unsupported tuning candidate parameters: {sorted(set(candidate).difference(allowed))}")
        candidate_config = copy.deepcopy(config)
        for path, item in candidate.items():
            section, key = path.split(".")
            candidate_config[section][key] = item
        validate_config(candidate_config)
        trial_configs.append(candidate_config)
    generator = getattr(train_loader, "generator", None)
    if generator is None:
        raise ValueError("Tuning requires a seeded training DataLoader generator.")
    original_state = generator.get_state()
    study_dir = Path(config["session"]["processed_dir"]) / "tuning" / uuid4().hex
    study_dir.mkdir(parents=True, exist_ok=False)
    trials, retained_configs = [], []
    split = getattr(train_loader.dataset, "split_metadata", {})
    result = {"status": "running", "selection_metric": "mean_validation_mae_sec",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "configuration": copy.deepcopy(config), "candidates": copy.deepcopy(candidates),
              "seeds": list(seeds), "device": str(device),
              "split": split if isinstance(split, dict) else {}, "trials": trials,
              "study_path": str(study_dir / "study.json")}

    def save_study():
        temporary = study_dir / "study.json.tmp"
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
        os.replace(temporary, study_dir / "study.json")

    save_study()
    try:
        for index, candidate_config in enumerate(trial_configs):
            runs = []
            candidate_result = {"candidate": candidates[index], "runs": runs}
            trials.append(candidate_result)
            for seed in seeds:
                trial = copy.deepcopy(candidate_config)
                trial_dir = study_dir / f"candidate_{index:02d}" / f"seed_{seed}"
                trial["data"]["random_seed"] = seed
                trial["train"]["checkpoint_path"] = str(trial_dir / "best.pt")
                trial["train"]["history_path"] = str(trial_dir / "training_history.csv")
                result["active_trial"] = {"candidate_index": index, "seed": seed,
                                          "checkpoint": trial["train"]["checkpoint_path"]}
                save_study()
                generator.manual_seed(seed)
                model = create_model(trial).to(device)
                score, history = train_model(model, train_loader, val_loader, trial, device)
                if not math.isfinite(score):
                    raise FloatingPointError(f"Non-finite validation score for candidate {index}, seed {seed}.")
                runs.append({"seed": seed, "validation_mae_sec": float(score),
                             "epochs": len(history), "checkpoint": trial["train"]["checkpoint_path"]})
                save_study()
                if seed == seeds[0]:
                    retained_configs.append(trial)
            candidate_result["mean_validation_mae_sec"] = fmean(run["validation_mae_sec"] for run in runs)
            save_study()
    except BaseException as error:
        result.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        save_study()
        raise
    finally:
        generator.set_state(original_state)
    selected_index = min(range(len(trials)), key=lambda index: trials[index]["mean_validation_mae_sec"])
    selected_config = retained_configs[selected_index]
    result.pop("active_trial", None)
    result.update(status="complete", selected_candidate_index=selected_index,
                  mean_validation_mae_sec=trials[selected_index]["mean_validation_mae_sec"],
                  selected_validation_mae_sec=trials[selected_index]["runs"][0]["validation_mae_sec"],
                  retained_seed=seeds[0], completed_at_utc=datetime.now(timezone.utc).isoformat())
    save_study()
    return selected_config, result
