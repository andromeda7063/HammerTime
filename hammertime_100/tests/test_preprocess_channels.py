"""Missing or malformed distance observations obey the same validity contract."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
from preprocess import align_driver_telemetry


class DistanceChannelTests(unittest.TestCase):
    def setUp(self):
        self.telemetry = pd.DataFrame({"TimeSec": [0.0, 0.5, 1.0], "Speed": 180.0,
                                       "Throttle": 80.0, "Distance": [0.0, 50.0, 100.0]})
        self.laps = pd.DataFrame({"LapNumber": [1], "LapStartTime": [pd.Timedelta(seconds=0)],
                                  "LapTime": [pd.Timedelta(seconds=2)], "Compound": ["SOFT"], "TyreLife": [2]})

    def align(self, telemetry):
        return align_driver_telemetry("AA", telemetry, self.laps, sample_hz=2,
                                       max_interpolate_sec=1, lap_length_m=100, mini_sectors=2)

    def test_missing_distance_channel_marks_rows_invalid(self):
        result = self.align(self.telemetry.drop(columns="Distance"))
        self.assertFalse(result["IsValid"].any())
        self.assertTrue(result["Distance"].isna().all())

    def test_invalid_distance_value_uses_only_supported_observations(self):
        self.telemetry["Distance"] = ["0", "unavailable", "100"]
        result = self.align(self.telemetry)
        self.assertTrue(result["IsValid"].all())
        np.testing.assert_allclose(result["Distance"], [0, 50, 100])


if __name__ == "__main__":
    unittest.main()
