"""Explicit OpenF1 dispatch must never fall back to another provider."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
import run_pipeline as pipeline


class OpenF1PipelineTests(unittest.TestCase):
    def test_openf1_failure_propagates_without_fastf1_fallback(self):
        with patch("openf1_client.load_openf1_tables", side_effect=RuntimeError("upstream failed")), \
                patch("ingest.load_race", side_effect=AssertionError("provider fallback")):
            with self.assertRaisesRegex(RuntimeError, "upstream failed"):
                pipeline._load_configured_race({"source": "openf1"}, {})

    def test_normalized_outputs_and_source_coverage_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "observed.json"
            raw_path.write_text("[]", encoding="utf-8")
            bundle = {"session": {"session_key": 9157}, "tables": {},
                      "raw_artifacts": [str(raw_path)], "provenance": {"request": {"status": 200}}}
            aligned = pd.DataFrame({"TimeSec": [0.0], "Driver": ["1"]})
            aligned.attrs["coverage"] = {"valid_rows": 1}
            config = {"session": {"processed_dir": str(root)}}
            with patch("openf1_client.load_openf1_tables", return_value=bundle), \
                    patch("openf1_adapter.normalize_openf1_tables", return_value=aligned), \
                    patch("ingest.load_race", side_effect=AssertionError("provider fallback")):
                output, artifacts, metadata = pipeline._load_configured_race({"source": "openf1"}, config)
            self.assertTrue(output.equals(aligned))
            self.assertTrue((root / "aligned_telemetry.parquet").is_file())
            self.assertEqual(metadata["provider"], "openf1")
            self.assertIn(str(raw_path), artifacts)
            self.assertEqual(json.loads((root / "coverage_report.json").read_text()), {"valid_rows": 1})
            self.assertEqual(json.loads((root / "source_manifest.json").read_text())["session"]["session_key"], 9157)


if __name__ == "__main__":
    unittest.main()
