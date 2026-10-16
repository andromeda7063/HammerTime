"""Regression tests for race-specific targets and isolated-car baselines."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import extract_isolated_car_features
from target import compute_clean_air_baselines


class BaselineIsolationTests(unittest.TestCase):
    def test_clean_air_reference_keys_are_scoped_to_race(self):
        sectors = pd.DataFrame([
            {"RaceId": race, "Driver": "AA", "Team": "T", "Compound": "SOFT",
             "MiniSector": 1, "AvgGap": 4.0, "Duration": duration}
            for race, duration in (("race-a", 2.0), ("race-b", 8.0))
        ])

        baselines = compute_clean_air_baselines(sectors, clean_air_gap=3.0, min_samples=1)

        self.assertEqual(baselines["primary"][("race-a", "AA", "SOFT", 1)], 2.0)
        self.assertEqual(baselines["primary"][("race-b", "AA", "SOFT", 1)], 8.0)
        self.assertEqual(len(baselines["grid"]), 2)

    def test_isolated_baseline_uses_full_car_history_and_no_neighbor_features(self):
        first = torch.tensor([[1.0, 2.0], [100.0, 200.0]])
        second = torch.tensor([[3.0, 4.0], [300.0, 400.0]])
        sample = {
            "snapshots": [Data(x=first), Data(x=second)],
            "target": torch.tensor([0.75, 99.0]),
            "mask": torch.tensor([True, False]),
        }

        sequences, flattened, targets = extract_isolated_car_features([sample])

        np.testing.assert_array_equal(sequences, np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))
        np.testing.assert_array_equal(flattened, np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        np.testing.assert_array_equal(targets, np.array([0.75], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
