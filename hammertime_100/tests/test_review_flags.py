"""Contracts for the review-gated GNNv4 flags (synthetic fixtures only)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
TEST_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import review_flags  # noqa: E402
from cohort_v4 import review_flags as parse_flags  # noqa: E402
from target import compute_reference_lap_trends  # noqa: E402
from test_cohort_v4 import LAP, fixture_config, race_tables  # noqa: E402


class ReviewFlagTests(unittest.TestCase):
    def test_all_off_is_identity(self):
        aligned, labels = race_tables(0)
        config = fixture_config()
        config["session"]["race_id"] = "R0"
        flags = parse_flags(config)
        self.assertFalse(any(flags.values()))
        out_aligned, out_labels, report = review_flags.apply_review_flags(aligned, labels, config, flags)
        self.assertIs(out_aligned, aligned)
        self.assertIs(out_labels, labels)

    def test_unknown_flag_is_rejected(self):
        config = fixture_config()
        config["review_flags"]["typo_flag"] = True
        with self.assertRaisesRegex(ValueError, "Unknown review_flags"):
            parse_flags(config)

    def test_rescale_lap_distance_restores_circuit_length(self):
        aligned, _ = race_tables(0)
        aligned = aligned[aligned["Driver"] == "1"].copy()
        aligned["Distance"] = aligned["Distance"] * 0.98  # integrated distance reads 2% short
        rescaled, info = review_flags.rescale_lap_distance(aligned, LAP, 20)
        self.assertGreater(info["laps_rescaled"], 0)
        self.assertAlmostEqual(info["median_scale"], 1 / 0.98, places=2)
        lap1 = rescaled[rescaled["LapNumber"] == 1]["Distance"]
        self.assertGreater(lap1.max(), aligned[aligned["LapNumber"] == 1]["Distance"].max())

    def test_physical_gap_uses_circular_order(self):
        frame = pd.DataFrame({
            "TimeSec": [0.0, 0.0], "Driver": ["A", "B"], "Distance": [990.0, 10.0],
            "Speed": [180.0, 180.0], "IsValid": [True, True], "IsPitLap": [False, False],
            "DeltaT": [99.0, 99.0]})
        result = review_flags.compute_physical_time_gaps(frame, 1000.0)
        self.assertAlmostEqual(float(result.loc[0, "DeltaT"]), 20.0 / 50.0, places=6)
        self.assertEqual(float(result.loc[0, "DeltaT_RaceOrder"]), 99.0)

    def test_detrend_trend_fit(self):
        sectors = pd.DataFrame({"RaceId": "R", "Driver": "A", "Compound": "SOFT", "MiniSector": 0,
                                "LapNumber": [1, 2, 3, 4], "Duration": [3.0, 2.9, 2.8, 2.7],
                                "AvgGap": 5.0})
        trends = compute_reference_lap_trends(sectors, min_samples=3)
        slope, center = trends[("R", "A", "SOFT", 0)]
        self.assertAlmostEqual(slope, -0.1, places=6)
        self.assertAlmostEqual(center, 2.5)

    def test_exclude_lap1_and_yellow(self):
        _, labels = race_tables(0)
        config = fixture_config()
        config["session"]["race_id"] = "R0"
        flags = {**parse_flags(config), "exclude_lap1_and_yellow_targets": True}
        with tempfile.TemporaryDirectory() as temporary:
            race_dir = Path(temporary) / "R0"
            race_dir.mkdir()
            pd.DataFrame({"DriverNumber": ["11"], "LapNumber": [2.0], "TrackStatus": ["12"]}).to_parquet(
                race_dir / "raw_laps.parquet")
            _, kept, report = review_flags.apply_review_flags(
                pd.DataFrame(), labels, config, flags, raw_sessions_dir=temporary)
        self.assertFalse((kept["LapNumber"] == 1).any())
        self.assertFalse(((kept["Driver"] == "11") & (kept["LapNumber"] == 2)).any())
        expected = int(((labels["LapNumber"] == 1) | ((labels["Driver"] == "11") & (labels["LapNumber"] == 2))).sum())
        self.assertEqual(report["labels_dropped_lap1_or_yellow"], expected)


if __name__ == "__main__":
    unittest.main()
