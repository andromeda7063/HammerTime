"""Unavailable baseline metrics must remain valid strict JSON report values."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import train_lstm_baseline, train_xgboost_baseline


class BaselineMetricTests(unittest.TestCase):
    def setUp(self):
        self.config = {"data": {"random_seed": 7}, "baselines": {
            "xgboost": {"n_estimators": 1, "max_depth": 1},
            "lstm": {"hidden_dim": 2, "num_layers": 1, "epochs": 1,
                     "batch_size": 2, "dropout": 0.0}}}
        self.train_x = np.array([[[0.0, 1.0]], [[1.0, 0.0]]], dtype=np.float32)
        self.train_y = np.array([0.0, 1.0], dtype=np.float32)
        self.test_x = self.train_x[:1]
        self.test_y = self.train_y[:1]

    def assert_single_target_metrics(self, result):
        self.assertTrue(result["available"])
        self.assertTrue(np.isfinite(result["mae_sec"]))
        self.assertIsNone(result["r2_score"])
        json.dumps({key: result[key] for key in ("mae_sec", "r2_score")}, allow_nan=False)

    def test_lstm_single_target_has_unavailable_r2(self):
        with patch("torch.cuda.is_available", return_value=False):
            result = train_lstm_baseline(self.train_x, self.train_y,
                                         self.test_x, self.test_y, self.config)
        self.assert_single_target_metrics(result)

    def test_xgboost_single_target_has_unavailable_r2(self):
        result = train_xgboost_baseline(self.train_x.reshape(2, 2), self.train_y,
                                        self.test_x.reshape(1, 2), self.test_y, self.config)
        self.assert_single_target_metrics(result)


if __name__ == "__main__":
    unittest.main()
