"""Offline fixtures for OpenF1 normalization; these tests never access the network."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
from openf1_adapter import _seconds_from_origin, normalize_openf1_tables


def bundle(*, gap=False, race_control=None):
    origin = pd.Timestamp("2024-09-01T12:00:00Z")
    speed_times = [0, 1, 2, 3, 4]
    if gap:
        speed_times = [0, 1, 4]
    tables = {
        "drivers": pd.DataFrame({
            "driver_number": [1], "full_name": ["Alex Driver"], "team_name": ["Team A"]
        }),
        "laps": pd.DataFrame({
            "driver_number": [1, 1], "lap_number": [1, 2],
            "date_start": [origin, origin + pd.Timedelta(2, unit="s")],
            "lap_duration": [2.0, 2.0], "is_pit_out_lap": [False, True],
        }),
        "stints": pd.DataFrame({
            "driver_number": [1], "stint_number": [1], "lap_start": [1],
            "lap_end": [2], "compound": ["SOFT"], "tyre_age_at_start": [3],
        }),
        "car_data": pd.DataFrame({
            "driver_number": [1] * len(speed_times),
            "date": [origin + pd.Timedelta(s, unit="s") for s in speed_times],
            "speed": [36.0] * len(speed_times), "throttle": [80] * len(speed_times),
        }),
        "position": pd.DataFrame(),
        "race_control": pd.DataFrame(race_control or []),
    }
    return {"session": {"date_start": origin, "session_key": 999}, "tables": tables,
            "raw_artifacts": {}, "provenance": {}}


def config():
    return {"session": {"race_id": "2024-Test-R"},
            "preprocess": {"sample_hz": 1.0, "max_interpolate_sec": 1.0,
                           "flag_pit_laps": True, "flag_safety_car": True},
            "track": {"mini_sectors": 2, "lap_length_m": 100.0}}


class OpenF1NormalizationTests(unittest.TestCase):
    def test_iso_timestamps_with_mixed_fractional_precision_share_one_clock(self):
        origin = pd.Timestamp("2023-09-03T13:00:00Z")
        values = pd.Series(["2023-09-03T13:00:00+00:00", "2023-09-03T13:00:00.250000Z",
                            "2023-09-03T15:00:00.500+02:00"])
        np.testing.assert_allclose(_seconds_from_origin(values, origin), [0.0, 0.25, 0.5])

    def test_builds_session_clock_lap_distance_metadata_and_race_id(self):
        aligned = normalize_openf1_tables(bundle(), config())
        self.assertEqual(set(aligned["RaceId"]), {"2024-Test-R"})
        self.assertEqual(set(aligned["Driver"]), {"1"})
        self.assertTrue(np.all(np.diff(aligned["TimeSec"].unique()) > 0))
        lap1 = aligned[aligned["LapNumber"] == 1]
        np.testing.assert_allclose(lap1["Distance"], [0.0, 10.0])
        self.assertTrue((lap1["Compound"] == "SOFT").all())
        self.assertTrue((lap1["TyreLife"] == 4).all())
        self.assertTrue((aligned.loc[aligned["LapNumber"] == 2, "TyreLife"] == 5).all())
        self.assertTrue(aligned.loc[aligned["LapNumber"] == 2, "IsPitLap"].all())
        self.assertIn("coverage", aligned.attrs)
        self.assertAlmostEqual(aligned.attrs["coverage"]["lap_distance_errors_m"]["1:1"], -80.0)

    def test_unsupported_speed_gap_invalidates_affected_lap(self):
        aligned = normalize_openf1_tables(bundle(gap=True), config())
        lap1 = aligned[aligned["LapNumber"] == 1]
        self.assertFalse(lap1["IsValid"].any())

    def test_location_channels_are_bounded_and_optional(self):
        data = bundle()
        origin = data["session"]["date_start"]
        data["tables"]["location"] = pd.DataFrame({
            "driver_number": [1, 1],
            "date": [origin + pd.Timedelta(0.5, unit="s"), origin + pd.Timedelta(1.5, unit="s")],
            "x": [10.0, 30.0], "y": [5.0, 15.0],
        })
        aligned = normalize_openf1_tables(data, config())
        self.assertTrue(pd.isna(aligned.loc[aligned["TimeSec"] == 0, "X"].iloc[0]))
        self.assertEqual(aligned.loc[aligned["TimeSec"] == 1, "X"].iloc[0], 20.0)
        self.assertTrue(aligned["IsValid"].all())

    def test_brake_0_100_is_normalized_to_observed_zero_one(self):
        data = bundle()
        data["tables"]["car_data"]["brake"] = [0, 100, 0, 100, 0]
        aligned = normalize_openf1_tables(data, config())
        lap1 = aligned[aligned["LapNumber"] == 1]
        self.assertEqual(lap1["Brake"].tolist(), [0.0, 1.0])

    def test_stint_age_zero_advances_from_one_on_first_lap(self):
        data = bundle()
        data["tables"]["stints"]["tyre_age_at_start"] = [0]
        aligned = normalize_openf1_tables(data, config())
        self.assertTrue((aligned.loc[aligned["LapNumber"] == 1, "TyreLife"] == 1).all())
        self.assertTrue((aligned.loc[aligned["LapNumber"] == 2, "TyreLife"] == 2).all())

    def test_safety_car_deployed_and_ending_form_event_window(self):
        origin = pd.Timestamp("2024-09-01T12:00:00Z")
        control = [
            {"date": origin + pd.Timedelta(0.5, unit="s"), "message": "Safety Car deployed"},
            {"date": origin + pd.Timedelta(1.5, unit="s"), "message": "Safety Car in this lap"},
            {"date": origin + pd.Timedelta(1.75, unit="s"), "message": "GREEN FLAG"},
        ]
        cfg = config()
        cfg["preprocess"]["sample_hz"] = 4.0
        aligned = normalize_openf1_tables(bundle(race_control=control), cfg)
        self.assertTrue(aligned.loc[aligned["LapNumber"] == 1, "IsSafetyCar"].all())
        self.assertFalse(aligned.loc[aligned["LapNumber"] == 2, "IsSafetyCar"].any())

    def test_vsc_ending_does_not_clear_until_green_message(self):
        origin = pd.Timestamp("2024-09-01T12:00:00Z")
        control = [
            {"date": origin + pd.Timedelta(0.5, unit="s"), "message": "VSC deployed"},
            {"date": origin + pd.Timedelta(1.5, unit="s"), "message": "VSC ending"},
            {"date": origin + pd.Timedelta(2.5, unit="s"), "message": "GREEN FLAG"},
        ]
        aligned = normalize_openf1_tables(bundle(race_control=control), config())
        self.assertTrue(aligned.loc[aligned["LapNumber"].isin([1, 2]), "IsSafetyCar"].all())

    def test_ambiguous_safety_text_fails_clearly(self):
        origin = pd.Timestamp("2024-09-01T12:00:00Z")
        control = [{"date": origin + pd.Timedelta(0.5, unit="s"),
                    "message": "Safety Car status uncertain"}]
        with self.assertRaisesRegex(ValueError, "Ambiguous.*safety-car"):
            normalize_openf1_tables(bundle(race_control=control), config())

    def test_local_green_flag_does_not_clear_global_safety_car(self):
        origin = pd.Timestamp("2024-09-01T12:00:00Z")
        control = [
            {"date": origin, "message": "SAFETY CAR DEPLOYED", "scope": "Track"},
            {"date": origin + pd.Timedelta(1, unit="s"), "message": "GREEN FLAG",
             "flag": "GREEN", "scope": "Sector", "sector": 2},
            {"date": origin + pd.Timedelta(3, unit="s"), "message": "GREEN FLAG",
             "flag": "GREEN", "scope": "Track"},
        ]
        aligned = normalize_openf1_tables(bundle(race_control=control), config())
        self.assertTrue(aligned.loc[aligned["LapNumber"] == 2, "IsSafetyCar"].all())

    def test_ending_without_observed_deployment_fails_clearly(self):
        control = [{"date": "2024-09-01T12:00:01Z", "message": "VSC ENDING"}]
        with self.assertRaisesRegex(ValueError, "deployment"):
            normalize_openf1_tables(bundle(race_control=control), config())

    def test_unparseable_race_control_time_cannot_hide_safety_car(self):
        control = [{"date": "unavailable", "message": "SAFETY CAR DEPLOYED"}]
        with self.assertRaisesRegex(ValueError, "race-control.*date"):
            normalize_openf1_tables(bundle(race_control=control), config())

    def test_invalid_observed_speeds_are_rejected(self):
        for invalid in (-1.0, np.inf):
            with self.subTest(speed=invalid):
                data = bundle()
                data["tables"]["car_data"].loc[0, "speed"] = invalid
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    normalize_openf1_tables(data, config())

    def test_non_integer_lap_number_is_rejected(self):
        data = bundle()
        data["tables"]["laps"]["lap_number"] = data["tables"]["laps"]["lap_number"].astype(float)
        data["tables"]["laps"].loc[0, "lap_number"] = 1.5
        with self.assertRaisesRegex(ValueError, "finite integers"):
            normalize_openf1_tables(data, config())

    def test_duplicate_driver_lap_records_are_rejected(self):
        data = bundle()
        data["tables"]["laps"] = pd.concat(
            [data["tables"]["laps"], data["tables"]["laps"].iloc[[0]]], ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "duplicate.*driver.*lap"):
            normalize_openf1_tables(data, config())

    def test_nonpositive_lap_number_is_rejected(self):
        data = bundle()
        data["tables"]["laps"].loc[0, "lap_number"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            normalize_openf1_tables(data, config())

    def test_overlapping_laps_sharing_grid_time_are_rejected(self):
        data = bundle()
        origin = data["session"]["date_start"]
        data["tables"]["laps"].loc[1, "date_start"] = origin + pd.Timedelta(1, unit="s")
        with self.assertRaisesRegex(ValueError, "overlap.*duplicate aligned rows"):
            normalize_openf1_tables(data, config())

    def test_sub_grid_overlap_without_shared_grid_time_is_retained(self):
        data = bundle()
        origin = data["session"]["date_start"]
        data["tables"]["laps"].loc[1, "date_start"] = origin + pd.Timedelta(1.5, unit="s")
        aligned = normalize_openf1_tables(data, config())
        self.assertFalse(aligned.duplicated(["Driver", "TimeSec"]).any())

    def test_missing_required_inputs_fail_clearly(self):
        data = bundle()
        data["tables"]["car_data"] = pd.DataFrame()
        with self.assertRaisesRegex(ValueError, "telemetry|speed|car_data"):
            normalize_openf1_tables(data, config())


if __name__ == "__main__":
    unittest.main()
