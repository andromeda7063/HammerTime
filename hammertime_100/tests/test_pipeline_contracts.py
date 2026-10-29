"""Runner failure and artifact provenance contracts without FastF1 access."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import run_pipeline as pipeline


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        with (PROJECT_DIR / "configs" / "default.yaml").open(encoding="utf-8") as source:
            self.config = yaml.safe_load(source)

    def test_load_failure_propagates_even_when_combined_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = copy.deepcopy(self.config)
            config["session"]["processed_dir"] = str(root)
            config["session"]["cache_dir"] = str(root / "cache")
            (root / "aligned_telemetry.parquet").touch()
            (root / "target_labels.parquet").touch()
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with patch("ingest.load_race", side_effect=RuntimeError("session unavailable")), \
                    patch.object(pd, "read_parquet", side_effect=AssertionError("stale data read")) as read:
                with self.assertRaisesRegex(RuntimeError, "session unavailable"):
                    pipeline.run_100_percent_pipeline(config_path)
            read.assert_not_called()



if __name__ == "__main__":
    unittest.main()
