"""Unit tests for F1 aerodynamic physics constraints, quality gates, and loss functions."""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import validate_clean_air_physics, validate_drs_train_physics
from train import masked_l1_loss


class PhysicsValidationTests(unittest.TestCase):
    def test_clean_air_physics_gate_pass(self):
        """Clean air law passes when predicted delay is near zero (< 0.050s)."""
        records = [
            {"RaceId": "2023-Monza-R", "Driver": "VER", "gap_sec": 4.5, "y_pred": 0.012},
            {"RaceId": "2023-Monza-R", "Driver": "HAM", "gap_sec": 3.8, "y_pred": -0.008},
            {"RaceId": "2023-Monza-R", "Driver": "NOR", "gap_sec": 5.2, "y_pred": 0.015},
        ]
        result = validate_clean_air_physics(records, max_mae=0.05, clean_gap_sec=3.0)
        self.assertTrue(result["passed"])
        self.assertLessEqual(result["observed_mae"], 0.05)

    def test_clean_air_physics_gate_fail_on_bias(self):
        """Clean air law fails when predicted delay has an unphysical bias (> 0.050s)."""
        records = [
            {"RaceId": "2023-Monza-R", "Driver": "VER", "gap_sec": 4.5, "y_pred": 0.120},
            {"RaceId": "2023-Monza-R", "Driver": "HAM", "gap_sec": 3.8, "y_pred": 0.095},
        ]
        result = validate_clean_air_physics(records, max_mae=0.05, clean_gap_sec=3.0)
        self.assertFalse(result["passed"])
        self.assertGreater(result["observed_mae"], 0.05)

    def test_drs_train_physics_gate_pass(self):
        """DRS train law passes when full-lap delay is within [0.40s, 0.80s]."""
        records = []
        for sec in range(20):
            records.append({
                "RaceId": "2023-Monza-R",
                "Driver": "LEC",
                "LapNumber": 22.0,
                "MiniSector": float(sec),
                "gap_sec": 0.55,
                "y_pred": 0.031  # 20 * 0.031s = 0.620s per lap
            })
        result = validate_drs_train_physics(records, mini_sectors_per_lap=20, min_lap_loss=0.40, max_lap_loss=0.80)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["observed_lap_delay"], 0.620, places=3)

    def test_drs_train_physics_gate_fail_low_loss(self):
        """DRS train law fails when cumulative lap loss is below 0.40s."""
        records = []
        for sec in range(20):
            records.append({
                "RaceId": "2023-Monza-R",
                "Driver": "LEC",
                "LapNumber": 22.0,
                "MiniSector": float(sec),
                "gap_sec": 0.55,
                "y_pred": 0.005  # 20 * 0.005s = 0.100s per lap (too low)
            })
        result = validate_drs_train_physics(records, mini_sectors_per_lap=20, min_lap_loss=0.40, max_lap_loss=0.80)
        self.assertFalse(result["passed"])

    def test_masked_l1_loss_ignores_masked_nodes(self):
        """Masked L1 loss must compute error strictly over unmasked, valid nodes."""
        preds = torch.tensor([[1.0, 99.0], [2.0, 50.0]])
        targets = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        masks = torch.tensor([[True, False], [True, False]])
        loss = masked_l1_loss(preds, targets, masks)
        self.assertEqual(float(loss), 0.0)

    def test_nan_loss_guard_triggers(self):
        """Failure Case / Law 9: Non-finite loss raises ValueError."""
        preds = torch.tensor([[float("nan"), 1.0]])
        targets = torch.tensor([[0.0, 1.0]])
        masks = torch.tensor([[False, False]])
        with self.assertRaises(ValueError):
            masked_l1_loss(preds, targets, masks)

    def test_clean_air_empty_records_safe(self):
        """Clean air evaluation handles empty records without crashing."""
        result = validate_clean_air_physics([], max_mae=0.05, clean_gap_sec=3.0)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["observed_mae"])

    def test_drs_train_empty_records_safe(self):
        """DRS train evaluation handles incomplete laps safely."""
        result = validate_drs_train_physics([], mini_sectors_per_lap=20)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["observed_lap_delay"])


if __name__ == "__main__":
    unittest.main()
