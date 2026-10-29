"""Dashboard artifact contracts using local JSON and in-memory telemetry only."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_DIR / "dashboard" / "app.py"


class DashboardContractTests(unittest.TestCase):
    def render(self, report=None, telemetry=None, image_content=None):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            if image_content is not None:
                image_path = artifact_dir / "card_01.png"
                image_path.write_bytes(image_content)
                report = {**(report or {}), "presentation": {
                    "cards_dir": directory, "visuals_dir": directory, "artifacts": [str(image_path)]
                }}
            if report is not None:
                (artifact_dir / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")
            if telemetry is not None:
                (artifact_dir / "aligned_telemetry.parquet").touch()
            source = APP_PATH.read_text(encoding="utf-8")
            source = source.replace("PROJECT_DIR = Path(__file__).resolve().parents[1]", f"PROJECT_DIR = Path({str(PROJECT_DIR)!r})")
            source = source.replace('PROCESSED_DIR = Path(CONFIG["session"].get("processed_dir", "data/processed"))', f"PROCESSED_DIR = Path({directory!r})")
            with patch.object(pd, "read_parquet", return_value=telemetry):
                app = AppTest.from_string(source).run(timeout=20)
            return app

    def test_missing_artifacts_do_not_invent_baseline_results(self):
        app = self.render()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(any("unavailable" in message.value.lower() for message in app.info))

    def test_partial_report_shows_only_observed_scores_including_zero(self):
        app = self.render({"metrics": {"mae_sec": 0.0, "r2_score": -0.5}, "baselines": {"lstm": {"mae_sec": None}}})
        self.assertEqual(len(app.exception), 0)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Test MAE"], "0.000 s")
        self.assertEqual(metrics["Test R²"], "-0.500")
        chart = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual(chart["data"][0]["x"], ["HammerTime ST-GNN"])
        self.assertEqual(chart["data"][0]["y"], [0.0])

    def test_physics_status_is_matched_by_test_name(self):
        app = self.render({"metrics": {"mae_sec": 0.1}, "physics_tests": [
            {"test": "DRS Train Aerodynamic Wake (observed gap < 0.8 s)", "passed": False}
        ]})
        self.assertEqual(len(app.exception), 0)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Clean-air check"], "N/A")
        self.assertEqual(metrics["DRS-train check"], "FAIL")

    def test_malformed_report_sections_are_unavailable_without_crashing(self):
        cases = [
            {"metrics": {"mae_sec": "invalid"}},
            {"metrics": {"mae_sec": True}},
            {"metrics": {"mae_sec": float("nan")}},
            {"metrics": {"mae_sec": 10 ** 400}},
            {"metrics": {"mae_sec": 0.1}, "physics_tests": None},
            {"metrics": {"mae_sec": 0.1}, "physics_tests": ["invalid"]},
            {"metrics": {"mae_sec": 0.1}, "baselines": {"lstm": None}},
            {"metrics": {"mae_sec": 0.1}, "metadata": {"split": None}},
            {"metrics": {"mae_sec": 0.1}, "predictions": ["invalid"]},
        ]
        for report in cases:
            with self.subTest(report=report):
                app = self.render(report)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("invalid" in message.value.lower() for message in app.warning))

    def test_non_numeric_laps_do_not_crash_slider(self):
        for lap in (None, "unavailable", float("inf"), float("-inf")):
            with self.subTest(lap=lap):
                telemetry = pd.DataFrame({"Driver": ["AAA"], "TimeSec": [1.0], "LapNumber": [lap]})
                app = self.render(telemetry=telemetry)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.select_slider), 0)
                self.assertTrue(any("lap" in message.value.lower() for message in app.info))

    def test_optional_car_ahead_field_is_not_required_for_gap_history(self):
        telemetry = pd.DataFrame({"Driver": ["AAA"], "TimeSec": [1.0], "LapNumber": [1], "DeltaT": [1.0]})
        app = self.render(telemetry=telemetry)
        self.assertEqual(len(app.exception), 0)

    def test_track_positions_exclude_nonfinite_coordinates(self):
        telemetry = pd.DataFrame({
            "Driver": ["AAA", "BBB"], "TimeSec": [1.0, 1.0], "LapNumber": [1, 1],
            "X": [1.0, float("inf")], "Y": [2.0, 3.0],
        })
        app = self.render(telemetry=telemetry)
        self.assertEqual(len(app.exception), 0)
        chart = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual(chart["data"][-1]["text"], ["AAA"])

    def test_legacy_images_are_not_presented_as_current_without_manifest(self):
        app = self.render({"metrics": {"mae_sec": 0.1}})
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("image")), 0)
        self.assertTrue(any("current run" in message.value.lower() for message in app.info))

    def test_replay_does_not_mix_nearby_session_timestamps(self):
        telemetry = pd.DataFrame({"Driver": ["AAA", "BBB"], "TimeSec": [6000.0, 6000.05],
                                   "LapNumber": [1, 1], "X": [1.0, 3.0], "Y": [2.0, 4.0]})
        app = self.render(telemetry=telemetry)
        self.assertEqual(len(app.exception), 0)
        chart = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual(chart["data"][-1]["text"], ["AAA"])

    def test_corrupt_current_image_does_not_crash_dashboard(self):
        app = self.render({"metrics": {"mae_sec": 0.1}}, image_content=b"not a PNG")
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("could not load" in message.value.lower() for message in app.warning))


if __name__ == "__main__":
    unittest.main()
