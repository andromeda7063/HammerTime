"""Session identity and geometry checks prevent accidental held-out duplication."""

import copy
import sys
import unittest
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from run_pipeline import _session_specs


class SessionConfigTests(unittest.TestCase):
    def setUp(self):
        with (PROJECT_DIR / "configs" / "default.yaml").open(encoding="utf-8") as source:
            self.config = yaml.safe_load(source)

    def test_directory_name_collisions_are_rejected(self):
        self.config["sessions"] = [
            {"year": 2023, "gp": "Monza", "race_id": "race/a"},
            {"year": 2024, "gp": "Monza", "race_id": "race_a"}]
        with self.assertRaisesRegex(ValueError, "directory"):
            _session_specs(self.config)

    def test_same_source_cannot_be_held_out_under_another_id(self):
        self.config["sessions"] = [
            {"year": 2023, "gp": "Monza", "race_id": "first"},
            {"year": 2023, "gp": "monza", "race_id": "renamed"}]
        with self.assertRaisesRegex(ValueError, "Duplicate.*session"):
            _session_specs(self.config)

    def test_other_circuits_require_explicit_geometry(self):
        self.config["sessions"] = [{"year": 2023, "gp": "Silverstone"}]
        with self.assertRaisesRegex(ValueError, "lap_length_m"):
            _session_specs(self.config)
        self.config["sessions"][0]["lap_length_m"] = 6000.0
        self.assertEqual(_session_specs(self.config)[0]["lap_length_m"], 6000.0)

    def test_malformed_falsy_sessions_do_not_silently_select_default(self):
        for malformed in ({}, "", 0, False):
            with self.subTest(value=malformed):
                self.config["sessions"] = malformed
                with self.assertRaisesRegex(ValueError, "sessions"):
                    _session_specs(self.config)

    def test_same_circuit_across_years_uses_configured_default(self):
        self.config["sessions"] = [{"year": year, "gp": "Monza"} for year in (2022, 2023, 2024)]
        original = copy.deepcopy(self.config)
        self.assertEqual(len(_session_specs(self.config)), 3)
        self.assertEqual(self.config, original)


if __name__ == "__main__":
    unittest.main()
