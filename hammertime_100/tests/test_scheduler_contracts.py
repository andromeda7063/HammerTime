"""Unit tests for ReduceLROnPlateau scheduler contracts, stepping, checkpointing, and resume."""

import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def forward(self, batch_windows):
        return self.weight.expand(len(batch_windows), 1)


class SchedulerContractsTests(unittest.TestCase):
    def _config(self, directory, lr_scheduler="plateau", patience=2, factor=0.5):
        return {
            "train": {
                "loss": "mae",
                "lr": 0.01,
                "weight_decay": 0.0,
                "max_epochs": 8,
                "early_stopping_patience": 10,
                "lr_scheduler": lr_scheduler,
                "lr_scheduler_patience": patience,
                "lr_scheduler_factor": factor,
                "lr_scheduler_min_lr": 1e-5,
                "checkpoint_path": str(Path(directory) / "best.pt"),
                "history_path": str(Path(directory) / "training.csv"),
            },
        }

    def test_plateau_scheduler_decays_learning_rate(self):
        """Verify scheduler reduces learning rate when validation plateaus."""
        with tempfile.TemporaryDirectory() as temporary:
            model = TinyRegressor()
            train_loader = TinyLoader(seed=42)
            val_loader = TinyLoader(seed=99)
            cfg = self._config(temporary, patience=1, factor=0.5)

            score, history = train_module.train_model(
                model, train_loader, val_loader, cfg, torch.device("cpu")
            )
            self.assertGreater(len(history), 2)
            # The initial learning rate was 0.01
            initial_lr = history[0].get("lr")
            self.assertIsNotNone(initial_lr)
            self.assertAlmostEqual(initial_lr, 0.01)
            # At least one LR reduction should have occurred over 6 plateau epochs with patience=1
            final_lr = history[-1].get("lr")
            self.assertIsNotNone(final_lr)
            self.assertLess(final_lr, initial_lr)

    def test_scheduler_state_dict_persists_in_checkpoints(self):
        """Verify scheduler state dict is recorded in both best.pt and best.last.pt."""
        with tempfile.TemporaryDirectory() as temporary:
            model = TinyRegressor()
            train_loader = TinyLoader(seed=42)
            val_loader = TinyLoader(seed=99)
            cfg = self._config(temporary, patience=1, factor=0.5)

            train_module.train_model(
                model, train_loader, val_loader, cfg, torch.device("cpu")
            )

            ckpt_path = Path(temporary) / "best.pt"
            last_ckpt_path = Path(temporary) / "best.last.pt"
            self.assertTrue(ckpt_path.exists())
            self.assertTrue(last_ckpt_path.exists())

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            last_ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
            self.assertIn("scheduler_state_dict", ckpt)
            self.assertIn("scheduler_state_dict", last_ckpt)


if __name__ == "__main__":
    unittest.main()
