"""Dashboard configuration path and missing-config contracts."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_DIR / "dashboard" / "app.py"


class DashboardConfigTests(unittest.TestCase):
    def render(self, env_config: Path):
        source = APP_PATH.read_text(encoding="utf-8")
        source = source.replace(
            "PROJECT_DIR = Path(__file__).resolve().parents[1]",
            f"PROJECT_DIR = Path({str(PROJECT_DIR)!r})",
        )
        with patch.dict(os.environ, {"HAMMERTIME_CONFIG": str(env_config)}):
            with patch.object(pd, "read_parquet", return_value=pd.DataFrame({
                "Driver": ["AAA"], "TimeSec": [1.0], "LapNumber": [1]
            })):
                return AppTest.from_string(source).run(timeout=20)

    def test_environment_config_loads_custom_evaluation_report_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            processed.mkdir()
            report_path = root / "reports" / "openf1-evaluation.json"
            report_path.parent.mkdir()
            report_path.write_text(json.dumps({"metrics": {"mae_sec": 0.123, "r2_score": 0.4}}), encoding="utf-8")
            config_path = root / "openf1.yaml"
            config_path.write_text(
                f"session:\n  processed_dir: {str(processed)!r}\nevaluation:\n  report_path: {str(report_path)!r}\n",
                encoding="utf-8",
            )

            app = self.render(config_path)

        self.assertEqual(len(app.exception), 0)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics.get("Test MAE"), "0.123 s")
        self.assertEqual(metrics.get("Test R²"), "0.400")

    def test_missing_environment_config_shows_error_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.yaml"
            app = self.render(missing)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("config" in error.value.lower() for error in app.error))
        self.assertEqual(len(app.metric), 0)

    def test_custom_report_name_still_requires_valid_report_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "custom.json"
            report_path.write_text("[]", encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                f"session:\n  processed_dir: {str(root)!r}\nevaluation:\n  report_path: {str(report_path)!r}\n",
                encoding="utf-8",
            )
            app = self.render(config_path)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("invalid structure" in warning.value for warning in app.warning))

    def test_invalid_configuration_is_reported_without_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "invalid.yaml"
            config_path.write_text("session: []", encoding="utf-8")
            app = self.render(config_path)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()
