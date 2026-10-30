"""Independent offline regression checks for real OpenF1 endpoint contracts."""

import sys
import unittest
from pathlib import Path

import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(SRC_DIR))

from openf1_adapter import normalize_openf1_tables
from test_openf1_adapter import bundle, config


class OpenF1SourceContractReviewTests(unittest.TestCase):
    def test_location_endpoint_rows_supply_aligned_xy_coordinates(self):
        data = bundle()
        origin = data["session"]["date_start"]
        data["tables"].pop("position", None)
        data["tables"]["location"] = pd.DataFrame({
            "driver_number": [1, 1, 1, 1, 1],
            "date": [origin + pd.Timedelta(sec, unit="s") for sec in range(5)],
            "x": [10, 20, 30, 40, 50],
            "y": [100, 200, 300, 400, 500],
            "z": [1, 2, 3, 4, 5],
        })

        aligned = normalize_openf1_tables(data, config())
        lap1 = aligned[aligned["LapNumber"] == 1]

        self.assertEqual(lap1["X"].tolist(), [10.0, 20.0])
        self.assertEqual(lap1["Y"].tolist(), [100.0, 200.0])

    def test_pit_endpoint_marks_the_lap_containing_a_stop(self):
        data = bundle()
        data["tables"]["pit"] = pd.DataFrame({
            "driver_number": [1], "lap_number": [1],
            "date": [data["session"]["date_start"] + pd.Timedelta(1, unit="s")],
        })

        aligned = normalize_openf1_tables(data, config())
        lap1 = aligned[aligned["LapNumber"] == 1]

        self.assertTrue(lap1["IsPitLap"].all())


if __name__ == "__main__":
    unittest.main()
