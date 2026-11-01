"""Tiny offline regression tests for complete-epoch training resume."""

import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train as train_module


class TinyLoader:
    def __init__(self, seed=81):
        self.dataset = SimpleNamespace(samples=[])
        self.dataset.feature_scaler = SimpleNamespace(
            mean_=np.asarray([2.0]), scale_=np.asarray([3.0]),
            var_=np.asarray([9.0]), n_features_in_=1,
        )
        self.generator = torch.Generator().manual_seed(seed)
        for index, target in enumerate((0.2, 0.8, 1.1, -0.2)):
            snapshot = SimpleNamespace(
                race_id="R1", time_sec=float(index),
                x=torch.tensor([[index, 1.0]], dtype=torch.float32),
                mask=torch.tensor([True]),
            )
            self.dataset.samples.append({
                "race_id": "R1", "input_start_time": float(index),
                "target_time": float(index + 1), "time_sec": float(index + 1),
                "snapshots": [snapshot],
                "target": torch.tensor([target], dtype=torch.float32),
                "mask": torch.tensor([True]),
            })

    def __len__(self):
        return 2

    def __iter__(self):
        order = torch.randperm(len(self.dataset.samples), generator=self.generator).tolist()
        for start in range(0, len(order), 2):
            samples = [self.dataset.samples[index] for index in order[start:start + 2]]
            windows = [sample["snapshots"] for sample in samples]
            targets = torch.stack([sample["target"] for sample in samples])
            masks = torch.stack([sample["mask"] for sample in samples])
            yield windows, targets, masks


class TinyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.15))
        self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, batch_windows):
        noise = self.weight.new_tensor((random.random() + np.random.random()) * 0.01)
        value = self.dropout(self.weight + noise)
        return value.expand(len(batch_windows), 1)


class TrainResumeTests(unittest.TestCase):
    def _config(self, directory):
        return {
            "train": {
                "loss": "mae", "lr": 0.01, "weight_decay": 0.0,
                "max_epochs": 4, "early_stopping_patience": 10,
                "checkpoint_path": str(Path(directory) / "best.pt"),
                "history_path": str(Path(directory) / "training.csv"),
            },
        }

    @staticmethod
    def _seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def test_interrupted_resume_matches_uninterrupted_optimizer_and_rng_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            whole_dir = Path(temporary) / "whole"
            resumed_dir = Path(temporary) / "resumed"
            whole_dir.mkdir(); resumed_dir.mkdir()

            self._seed(23)
            whole_model = TinyRegressor()
            whole_train, whole_val = TinyLoader(), TinyLoader(seed=91)
            whole_score, whole_history = train_module.train_model(
                whole_model, whole_train, whole_val, self._config(whole_dir), torch.device("cpu"))

            self._seed(23)
            partial_model = TinyRegressor()
            partial_train, partial_val = TinyLoader(), TinyLoader(seed=91)
            original_save = train_module._atomic_torch_save

            def interrupt_after_best_replace(payload, path):
                if Path(path).name == "best.last.pt" and payload.get("completed_epoch") == 4:
                    raise RuntimeError("simulated interruption between best and last checkpoint writes")
                return original_save(payload, path)

            with patch.object(train_module, "_atomic_torch_save", interrupt_after_best_replace):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train_module.train_model(
                        partial_model, partial_train, partial_val,
                        self._config(resumed_dir), torch.device("cpu"))

            last_path = resumed_dir / "best.last.pt"
            first_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
            self.assertEqual(first_checkpoint["completed_epoch"], 3)
            self.assertEqual(len(first_checkpoint["history"]), 3)
            self.assertIn("optimizer_state_dict", first_checkpoint)
            self.assertIn("rng_state", first_checkpoint)
            disk_best = torch.load(resumed_dir / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(disk_best["epoch"], 4)

            # Model construction may consume RNG; resume restores the exact saved states.
            self._seed(999)
            resumed_model = TinyRegressor()
            resumed_train, resumed_val = TinyLoader(), TinyLoader(seed=91)
            resumed_score, resumed_history = train_module.train_model(
                resumed_model, resumed_train, resumed_val, self._config(resumed_dir),
                torch.device("cpu"), resume_from=last_path)

            self.assertEqual(len(whole_history), len(resumed_history))
            for expected, actual in zip(whole_history, resumed_history):
                self.assertEqual(expected["epoch"], actual["epoch"])
                self.assertAlmostEqual(expected["train_loss"], actual["train_loss"], places=6)
                self.assertAlmostEqual(expected["val_mae"], actual["val_mae"], places=6)
            self.assertAlmostEqual(whole_score, resumed_score, places=6)
            for name, value in whole_model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=1e-6, atol=1e-7)

    def test_resume_rejects_config_data_and_scaler_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(temporary)
            self._seed(12)
            model, train_loader, val_loader = TinyRegressor(), TinyLoader(), TinyLoader(seed=91)
            original_eval = train_module.evaluate_val_mae
            calls = 0

            def stop_after_saved_epoch(model, loader, device):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("stop")
                return original_eval(model, loader, device)

            with patch.object(train_module, "evaluate_val_mae", stop_after_saved_epoch):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    train_module.train_model(model, train_loader, val_loader, config, torch.device("cpu"))
            last_path = Path(temporary) / "best.last.pt"

            changed_config = copy.deepcopy(config)
            changed_config["train"]["lr"] = 0.02
            with self.assertRaisesRegex(ValueError, "config, train/validation data, scaler"):
                train_module.train_model(TinyRegressor(), TinyLoader(), TinyLoader(seed=91),
                                         changed_config, torch.device("cpu"), resume_from=last_path)

            changed_data = TinyLoader()
            changed_data.dataset.samples[0]["target"][0] += 1.0
            with self.assertRaisesRegex(ValueError, "config, train/validation data, scaler"):
                train_module.train_model(TinyRegressor(), changed_data, TinyLoader(seed=91),
                                         config, torch.device("cpu"), resume_from=last_path)

            changed_scaler = TinyLoader()
            changed_scaler.dataset.feature_scaler.mean_[0] += 1.0
            with self.assertRaisesRegex(ValueError, "config, train/validation data, scaler"):
                train_module.train_model(TinyRegressor(), changed_scaler, TinyLoader(seed=91),
                                         config, torch.device("cpu"), resume_from=last_path)


if __name__ == "__main__":
    unittest.main()
