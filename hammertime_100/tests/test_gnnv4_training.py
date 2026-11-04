"""Offline contracts for the GNNv4 training options (fixtures only, no race data)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
TEST_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train as train_module  # noqa: E402
from test_train_resume import TinyLoader, TinyRegressor  # noqa: E402


def _config(directory, **train):
    base = {"loss": "smooth_l1", "lr": 0.01, "weight_decay": 0.0, "max_epochs": 3,
            "early_stopping_patience": 10,
            "checkpoint_path": str(Path(directory) / "best.pt"),
            "history_path": str(Path(directory) / "training.csv")}
    base.update(train)
    return {"train": base}


class CosineScheduleTests(unittest.TestCase):
    def test_warmup_then_cosine_to_min_ratio(self):
        factor = train_module._cosine_warmup_lambda(total_steps=10, warmup_steps=2, min_ratio=0.1)
        self.assertAlmostEqual(factor(0), 0.5)
        self.assertAlmostEqual(factor(1), 1.0)
        self.assertAlmostEqual(factor(2), 1.0)
        self.assertAlmostEqual(factor(10), 0.1)
        values = [factor(step) for step in range(2, 11)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))


class TrainingOptionTests(unittest.TestCase):
    def test_configured_huber_beta_reaches_loss(self):
        seen = []
        original = train_module.masked_smooth_l1_loss

        def spy(preds, targets, masks, beta=0.02):
            seen.append(beta)
            return original(preds, targets, masks, beta=beta)

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(train_module, "masked_smooth_l1_loss", spy):
            train_module.train_model(TinyRegressor(), TinyLoader(), TinyLoader(seed=91),
                                     _config(temporary, smooth_l1_beta=0.15, max_epochs=1),
                                     torch.device("cpu"))
        self.assertTrue(seen)
        self.assertTrue(all(beta == 0.15 for beta in seen))

    def test_default_beta_is_unchanged(self):
        seen = []
        original = train_module.masked_smooth_l1_loss

        def spy(preds, targets, masks, beta=0.02):
            seen.append(beta)
            return original(preds, targets, masks, beta=beta)

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(train_module, "masked_smooth_l1_loss", spy):
            train_module.train_model(TinyRegressor(), TinyLoader(), TinyLoader(seed=91),
                                     _config(temporary, max_epochs=1), torch.device("cpu"))
        self.assertTrue(all(beta == 0.02 for beta in seen))

    def test_gradient_accumulation_steps_optimizer_once_per_group(self):
        steps = []
        original_step = torch.optim.Adam.step

        def counting_step(self, *args, **kwargs):
            steps.append(1)
            return original_step(self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(torch.optim.Adam, "step", counting_step):
            train_module.train_model(TinyRegressor(), TinyLoader(), TinyLoader(seed=91),
                                     _config(temporary, grad_accum_steps=2, max_epochs=3),
                                     torch.device("cpu"))
        # TinyLoader yields 2 batches per epoch -> one optimizer step per epoch.
        self.assertEqual(len(steps), 3)

    def test_cosine_schedule_resumes_and_logs_val_r2(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary, lr_scheduler="cosine", warmup_epochs=1,
                             lr_scheduler_min_lr=0.001, log_val_r2=True, max_epochs=3)
            score, history = train_module.train_model(
                TinyRegressor(), TinyLoader(), TinyLoader(seed=91), config, torch.device("cpu"))
            self.assertTrue(np.isfinite(score))
            self.assertIn("val_r2", history[0])
            self.assertIn("lr", history[0])
            self.assertLess(history[-1]["lr"], history[0]["lr"] + 1e-12)
            last = Path(temporary) / "best.last.pt"
            resumed_score, resumed_history = train_module.train_model(
                TinyRegressor(), TinyLoader(), TinyLoader(seed=91), config,
                torch.device("cpu"), resume_from=last)
            self.assertEqual(resumed_history, history)
            self.assertEqual(resumed_score, score)

    def test_lazy_dataset_fingerprint_is_used_for_resume_signature(self):
        train, val = TinyLoader(), TinyLoader(seed=91)
        train.dataset.data_fingerprint = "a" * 64
        val.dataset.data_fingerprint = "b" * 64
        first = train_module._training_data_signature(train, val)
        val.dataset.data_fingerprint = "c" * 64
        self.assertNotEqual(first, train_module._training_data_signature(train, val))


if __name__ == "__main__":
    unittest.main()
