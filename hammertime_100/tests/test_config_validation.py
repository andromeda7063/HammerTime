"""Reject unusable experiment configurations before loading any telemetry."""

import copy
import sys
import unittest
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_validation import validate_config


class ConfigValidationTests(unittest.TestCase):
    def setUp(self):
        with (SRC_DIR.parent / "configs" / "default.yaml").open(encoding="utf-8") as source:
            self.config = yaml.safe_load(source)

    def test_default_config_is_valid_and_not_mutated(self):
        original = copy.deepcopy(self.config)
        validate_config(self.config)
        self.assertEqual(self.config, original)

    def test_missing_keys_name_the_full_path(self):
        for section, key in (("session", "processed_dir"), ("track", "mini_sectors"),
                             ("data", "timestamp_tolerance_sec"), ("train", "lr"),
                             ("model", "hidden_dim"), ("physics_tests", "clean_air_max_mae")):
            with self.subTest(section=section, key=key):
                config = copy.deepcopy(self.config)
                del config[section][key]
                with self.assertRaisesRegex(ValueError, rf"{section}\.{key}"):
                    validate_config(config)

    def test_non_mapping_section_is_descriptive(self):
        self.config["model"] = []
        with self.assertRaisesRegex(ValueError, "model.*mapping"):
            validate_config(self.config)

    def test_invalid_numeric_values_are_rejected(self):
        cases = (("train", "lr", 0), ("train", "max_epochs", 0),
                 ("train", "batch_size", True), ("model", "hidden_dim", 63),
                 ("model", "dropout", 1.0), ("graph", "rbf_sigma", float("nan")),
                 ("preprocess", "sample_hz", float("inf")),
                 ("data", "horizon", -1), ("data", "window_size", 1.5),
                 ("data", "random_seed", -1), ("data", "min_valid_targets_per_sample", 21))
        for section, key, value in cases:
            with self.subTest(section=section, key=key, value=value):
                config = copy.deepcopy(self.config)
                config[section][key] = value
                with self.assertRaisesRegex(ValueError, rf"{section}\.{key}"):
                    validate_config(config)

    def test_invalid_schema_combinations_are_rejected(self):
        cases = (("model", "in_dim", 8), ("model", "edge_dim", 2),
                 ("data", "split_method", "random"), ("data", "test_ratio", 0.3),
                 ("target", "clip_delta_sec", [10, -2]),
                 ("graph", "compound_categories", ["SOFT", "SOFT"]))
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                config = copy.deepcopy(self.config)
                config[section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()
