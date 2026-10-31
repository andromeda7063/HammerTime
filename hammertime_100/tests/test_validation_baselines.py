"""Contract tests for train/validation-only isolated-car baseline reports."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import run_validation_baselines


def _sample(race, time, target, width):
    snapshots = [SimpleNamespace(x=torch.full((2, width), float(frame + 1))) for frame in range(3)]
    return {
        "snapshots": snapshots,
        "target": torch.tensor([target, 99.0], dtype=torch.float32),
        "mask": torch.tensor([True, False]),
        "driver_names": ["driver-1", "driver-2"],
        "mini_sector": np.asarray([7.0, np.nan]),
        "race_id": race,
        "target_time": time,
    }


class ValidationBaselineTests(unittest.TestCase):
    def test_wrapper_supports_variable_feature_width_and_returns_metadata(self):
        train = [_sample("train-race", 10.0, 1.0, 19), _sample("train-race", 11.0, 3.0, 19)]
        validation = [_sample("validation-race", 20.0, 2.0, 19)]
        with patch("baselines._train_validation_xgboost",
                   return_value=(np.asarray([2.5], dtype=np.float32), 1.25, None)) as xgb_train, \
             patch("baselines._train_validation_lstm",
                   return_value=(np.asarray([1.5], dtype=np.float32), 2.5)) as lstm_train:
            result = run_validation_baselines(train, validation, {}, "cpu")

        self.assertEqual(result["feature_width"], 19)
        self.assertEqual(result["train_target_count"], 2)
        self.assertEqual(result["validation_target_count"], 1)
        self.assertEqual(result["mean_train_constant"]["predictions"], [2.0])
        self.assertEqual(result["xgboost"]["predictions"], [2.5])
        self.assertEqual(result["lstm"]["predictions"], [1.5])
        self.assertEqual(result["lstm"]["metadata"][0], {
            "race_id": "validation-race", "driver": "driver-1",
            "mini_sector": 7.0, "target_time": 20.0,
        })
        self.assertEqual(xgb_train.call_args.args[0].shape[1], 3 * 19)
        self.assertEqual(lstm_train.call_args.args[0].shape, (2, 3, 19))
        self.assertEqual(lstm_train.call_args.args[2].shape, (1, 3, 19))

    def test_wrapper_preserves_legacy_width(self):
        train = [_sample("train-race", 10.0, 1.0, 9)]
        validation = [_sample("validation-race", 20.0, 2.0, 9)]
        with patch("baselines._train_validation_xgboost",
                   return_value=(np.asarray([2.0], dtype=np.float32), 0.1, None)), \
             patch("baselines._train_validation_lstm",
                   return_value=(np.asarray([2.0], dtype=np.float32), 0.1)):
            result = run_validation_baselines(train, validation, {}, torch.device("cpu"))
        self.assertEqual(result["feature_width"], 9)
        self.assertEqual(result["window_size"], 3)

    def test_wrapper_requires_nonempty_train_and_validation_targets(self):
        with self.assertRaisesRegex(ValueError, "No valid car targets"):
            run_validation_baselines([], [_sample("val", 1.0, 1.0, 9)], {}, "cpu")


if __name__ == "__main__":
    unittest.main()
