"""Provider selection must be explicit and cannot duplicate a held-out race."""

import copy
import sys
import unittest
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config_validation import validate_config, validate_session_source
from run_pipeline import _session_specs


class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((PROJECT_DIR / "configs" / "default.yaml").read_text())
        self.options = {"base_url": "https://api.openf1.org/v1", "timeout_sec": 30,
                        "max_attempts": 3, "chunk_seconds": 300,
                        "request_interval_sec": 2.1, "fetch_location": True}

    def test_legacy_configuration_keeps_fastf1(self):
        validate_config(self.config)
        self.assertEqual(_session_specs(self.config)[0]["source"], "fastf1")

    def test_openf1_requires_supported_year_and_pinned_identity(self):
        session = {**self.config["session"], "source": "openf1", "session_key": 9157}
        validate_session_source(session, self.options)
        for field, value in (("year", 2021), ("session_key", None), ("session_key", True),
                             ("source", "automatic")):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_session_source({**session, field: value}, self.options)

    def test_openf1_options_are_validated_before_acquisition(self):
        session = {**self.config["session"], "source": "openf1", "session_key": 9157}
        for field, value in (("timeout_sec", 0), ("max_attempts", 1.5),
                             ("chunk_seconds", -1), ("request_interval_sec", -1),
                             ("fetch_location", "yes"), ("base_url", "http://example.com")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_session_source(session, {**self.options, field: value})
        missing = copy.deepcopy(self.options)
        del missing["timeout_sec"]
        with self.assertRaisesRegex(ValueError, "timeout_sec"):
            validate_session_source(session, missing)

    def test_same_race_cannot_be_held_out_through_another_provider(self):
        self.config["openf1"] = self.options
        self.config["sessions"] = [
            {"year": 2023, "gp": "Monza", "source": "fastf1", "race_id": "first"},
            {"year": 2023, "gp": "Monza", "source": "openf1", "session_key": 9157,
             "race_id": "second"},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate configured session"):
            _session_specs(self.config)

    def test_per_session_provider_is_validated(self):
        self.config["sessions"] = [{"year": 2023, "gp": "Monza", "source": "unknown"}]
        with self.assertRaisesRegex(ValueError, "source"):
            _session_specs(self.config)


if __name__ == "__main__":
    unittest.main()
