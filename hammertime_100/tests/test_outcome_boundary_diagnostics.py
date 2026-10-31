"""Offline contracts for additive sector-outcome boundary diagnostics."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import _outcome_boundary_diagnostic


def sample(input_start, mask, exits=None, race_id="race-a"):
    return {
        "input_start_time": input_start,
        "race_id": race_id,
        "mask": torch.tensor(mask, dtype=torch.bool),
        "outcome_exit_time_sec": (np.asarray(exits, dtype=float) if exits is not None
                                   else np.full(len(mask), np.nan)),
    }


class OutcomeBoundaryDiagnosticTests(unittest.TestCase):
    def test_counts_outcomes_finishing_after_next_partition_input_start(self):
        result = _outcome_boundary_diagnostic(
            [sample(1.0, [True, True], [2.0, 4.0]), sample(1.25, [True, False], [3.0, np.nan])],
            [sample(5.0, [True], [6.0])],
        )
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["boundary_input_start_time_sec"], 5.0)
        self.assertEqual(result["valid_outcome_count"], 3)
        self.assertEqual(result["crossing_count"], 0)
        self.assertEqual(result["crossing_fraction"], 0.0)

    def test_missing_exit_metadata_never_reports_false_zero(self):
        result = _outcome_boundary_diagnostic(
            [sample(1.0, [True, True], [4.0, np.nan])],
            [sample(3.0, [True], [4.0])],
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["metadata_missing_count"], 1)
        self.assertIsNone(result["crossing_count"])
        self.assertIsNone(result["crossing_fraction"])

    def test_multiple_session_clocks_are_not_compared(self):
        result = _outcome_boundary_diagnostic(
            [sample(1.0, [True], [4.0], race_id="race-a")],
            [sample(3.0, [True], [4.0], race_id="race-b")],
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("timestamps are incomparable", result["reason"])
        self.assertIsNone(result["crossing_count"])

    def test_known_crossing_reports_count_and_fraction(self):
        result = _outcome_boundary_diagnostic(
            [sample(1.0, [True, True], [4.0, 4.1])],
            [sample(3.0, [True], [5.0])],
        )
        self.assertEqual(result["crossing_count"], 2)
        self.assertEqual(result["crossing_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
