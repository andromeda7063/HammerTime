"""Offline shared-clock and interpolation contracts for telemetry alignment."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
from preprocess import align_driver_telemetry, process_race


def lap_rows(start=100.0, duration=2.0, number=1):
    return pd.DataFrame({"LapNumber": [number], "LapStartTime": pd.to_timedelta([start], unit="s"),
                         "LapTime": pd.to_timedelta([duration], unit="s"),
                         "Compound": ["SOFT"], "TyreLife": [2.0], "Team": ["Team A"]})


def telemetry():
    return pd.DataFrame({"Time": pd.to_timedelta([0, 1, 2], unit="s"),
                         "SessionTime": pd.to_timedelta([100, 101, 102], unit="s"),
                         "Speed": [180, 180, 180], "Throttle": [80, 80, 80],
                         "Distance": [0, 50, 100]})


class SharedClockTests(unittest.TestCase):
    def test_alignment_uses_session_clock_for_lap_metadata(self):
        aligned = align_driver_telemetry("AA", telemetry(), lap_rows(), sample_hz=1,
                                         lap_length_m=100, mini_sectors=2)
        np.testing.assert_allclose(aligned["TimeSec"], [100, 101, 102])
        self.assertEqual(aligned.loc[0, "Compound"], "SOFT")

    def test_race_preparation_preserves_shared_session_clock(self):
        driver_laps = SimpleNamespace(empty=False, get_telemetry=lambda: telemetry())
        session = SimpleNamespace(drivers=["AA"], laps=SimpleNamespace(pick_driver=lambda _: driver_laps))
        config = {"preprocess": {"sample_hz": 1, "max_interpolate_sec": 1},
                  "track": {"mini_sectors": 2, "lap_length_m": 100},
                  "session": {"processed_dir": "/tmp/hammertime-test-preprocess", "race_id": "observed"}}
        frame = pd.DataFrame({"TimeSec": [100.0], "Driver": ["AA"], "IsPitLap": [False],
                              "IsValid": [True], "LapNumber": [1], "Distance": [0], "Speed": [180]})
        with patch("preprocess.align_driver_telemetry", return_value=frame) as align, \
                patch.object(pd.DataFrame, "to_parquet"):
            process_race(session, config)
        np.testing.assert_allclose(align.call_args.args[1]["TimeSec"], [100, 101, 102])

    def test_explicit_time_seconds_remains_authoritative(self):
        rows = telemetry()
        rows["TimeSec"] = [200, 201, 202]
        aligned = align_driver_telemetry("AA", rows, lap_rows(start=200), sample_hz=1)
        np.testing.assert_allclose(aligned["TimeSec"], [200, 201, 202])


class InterpolationSupportTests(unittest.TestCase):
    def test_session_time_magnitude_does_not_make_long_gap_exact(self):
        rows = pd.DataFrame({"TimeSec": [4000.0, 4002.01], "Speed": [180, 180],
                             "Throttle": [80, 80], "Distance": [0, 100]})
        aligned = align_driver_telemetry("AA", rows, lap_rows(start=4000, duration=3),
                                         sample_hz=1, max_interpolate_sec=1)
        self.assertFalse(aligned.loc[aligned["TimeSec"] == 4002.0, "IsValid"].iloc[0])


class LapDistanceTests(unittest.TestCase):
    def test_cumulative_distance_is_anchored_at_observed_lap_start(self):
        rows = pd.DataFrame({"TimeSec": np.arange(100.0, 105.0),
                             "Speed": 180.0, "Throttle": 80.0,
                             "Distance": [10, 60, 110, 160, 210]})
        laps = pd.concat([lap_rows(), lap_rows(start=102, number=2)], ignore_index=True)
        aligned = align_driver_telemetry("AA", rows, laps, sample_hz=1,
                                         lap_length_m=100, mini_sectors=2)
        np.testing.assert_allclose(aligned.loc[:3, "Distance"], [0, 50, 0, 50])
        np.testing.assert_array_equal(aligned.loc[:3, "MiniSector"], [0, 1, 0, 1])

    def test_unmatched_lap_metadata_cannot_create_default_clean_lap(self):
        aligned = align_driver_telemetry("AA", telemetry(), lap_rows(start=101), sample_hz=1)
        self.assertFalse(bool(aligned.loc[0, "IsValid"]))
        self.assertTrue(pd.isna(aligned.loc[0, "LapNumber"]))

    def test_missing_observed_lap_start_cannot_guess_distance_origin(self):
        rows = telemetry()
        rows["SessionTime"] = pd.to_timedelta([100.5, 101.0, 102.0], unit="s")
        aligned = align_driver_telemetry("AA", rows, lap_rows(), sample_hz=2)
        self.assertFalse(aligned["IsValid"].any())


if __name__ == "__main__":
    unittest.main()
