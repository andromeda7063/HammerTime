"""Measured mini-sector duration contracts; all observations stay in memory."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
import target
from target import calculate_mini_sector_durations


def observations(times=(0.0, 0.7, 1.4, 2.1, 2.8), distances=(0, 30, 60, 80, 100)):
    return pd.DataFrame({
        "RaceId": "observed-session", "Driver": "AA", "LapNumber": 1,
        "MiniSector": [min(int(d // 50), 1) for d in distances],
        "Compound": "MEDIUM", "Team": "Team A", "TimeSec": times,
        "Distance": distances, "DeltaT": 4.0, "IsValid": True,
        "IsPitLap": False, "IsSafetyCar": False,
    })


class TargetMeasurementTests(unittest.TestCase):
    def durations(self, rows):
        return calculate_mini_sector_durations(rows, max_gap_sec=1.0,
                                               lap_length_m=100.0, mini_sectors=2)

    def test_duration_interpolates_observed_boundary_crossings(self):
        result = self.durations(observations()).sort_values("MiniSector")
        crossing = 0.7 + (50.0 - 30.0) / 30.0 * 0.7
        np.testing.assert_allclose(result["Duration"], [crossing, 2.8 - crossing])
        np.testing.assert_allclose(result["TimeSec"], [0.0, crossing])
        self.assertAlmostEqual(result["Duration"].sum(), 2.8)

    def test_original_call_measures_elapsed_time_instead_of_wake_formula(self):
        rows = observations()
        rows["Distance"] = rows["Distance"] * (5793.0 / 20 / 50)
        rows["MiniSector"] = [0, 0, 1, 1, 2]
        result = calculate_mini_sector_durations(rows, max_gap_sec=1.0)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result["Duration"].sum(), 2.8)

    def test_gap_and_driver_name_do_not_create_unmeasured_duration(self):
        rows = observations()
        expected = self.durations(rows)["Duration"].to_numpy()
        rows["Driver"] = "BB"
        rows["DeltaT"] = 0.3
        np.testing.assert_allclose(self.durations(rows)["Duration"], expected)

    def test_partial_sector_is_omitted_without_extrapolation(self):
        result = self.durations(observations().iloc[1:-1])
        self.assertTrue(result.empty)

    def test_invalid_observation_cannot_be_bridged(self):
        rows = observations()
        rows.loc[2, "IsValid"] = False
        self.assertTrue(self.durations(rows).empty)

    def test_long_gap_cannot_be_used_to_cross_boundary(self):
        rows = observations(times=(0.0, 0.7, 2.0, 2.7, 3.4))
        result = self.durations(rows)
        self.assertTrue(result.empty)

    def test_races_with_identical_driver_and_lap_remain_separate(self):
        first = observations()
        second = observations(times=(0.0, 1.4, 2.8, 4.2, 5.6))
        second["RaceId"] = "other-session"
        result = calculate_mini_sector_durations(pd.concat([first, second]),
                    max_gap_sec=2.0, lap_length_m=100.0, mini_sectors=2)
        totals = result.groupby("RaceId")["Duration"].sum()
        self.assertAlmostEqual(totals["observed-session"], 2.8)
        self.assertAlmostEqual(totals["other-session"], 5.6)

    def test_next_lap_start_supports_previous_lap_final_boundary(self):
        rows = observations()
        rows.loc[4, ["LapNumber", "Distance", "MiniSector"]] = [2, 0, 0]
        result = self.durations(rows)
        self.assertEqual(result["LapNumber"].tolist(), [1, 1])
        self.assertEqual(result["MiniSector"].tolist(), [0, 1])
        self.assertAlmostEqual(result["Duration"].sum(), 2.8)

    def test_build_targets_strips_large_pandas_attrs_on_copy_without_mutating_input(self):
        rows = pd.concat([
            observations(),
            observations(times=(10.0, 10.7, 11.4, 12.1, 12.8)).assign(Driver="BB"),
        ], ignore_index=True)
        rows["DeltaT"] = rows["Driver"].map({"AA": 1.0, "BB": 2.0})
        rows.attrs = {
            "coverage": {f"{driver}:{lap}": {"distance_error": -46.0}
                         for driver in range(20) for lap in range(1, 52)},
            "provenance": {f"request-{i}": {"sha256": "x" * 64, "status": 200}
                           for i in range(900)},
        }
        original_attrs = rows.attrs.copy()
        cfg = {
            "target": {"clean_air_gap_sec": 0.5, "min_clean_samples": 1,
                       "clip_delta_sec": [-10.0, 10.0]},
            "preprocess": {"max_interpolate_sec": 1.0},
            "track": {"lap_length_m": 100.0, "mini_sectors": 2},
            "session": {"race_id": "attr-test"},
        }
        seen_attrs = []
        calculate = target.calculate_mini_sector_durations

        def record_attrs(frame, *args, **kwargs):
            seen_attrs.append(dict(frame.attrs))
            return calculate(frame, *args, **kwargs)

        with (tempfile.TemporaryDirectory() as decorated_dir,
              patch("target.calculate_mini_sector_durations", side_effect=record_attrs)):
            decorated_cfg = {**cfg, "session": {**cfg["session"],
                                                   "processed_dir": decorated_dir}}
            decorated = target.build_targets(rows, decorated_cfg)

        self.assertEqual(seen_attrs, [{}])
        self.assertEqual(rows.attrs, original_attrs)
        plain = rows.copy()
        plain.attrs = {}
        with tempfile.TemporaryDirectory() as plain_dir:
            plain_cfg = {**cfg, "session": {**cfg["session"], "processed_dir": plain_dir}}
            expected = target.build_targets(plain, plain_cfg)
        pd.testing.assert_frame_equal(decorated, expected)


if __name__ == "__main__":
    unittest.main()
