"""End-to-end runner contract on synthetic fixtures (no real data, no network)."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
for path in (HERE, HERE / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cohort_v4  # noqa: E402
import train_gnnv4  # noqa: E402
from test_cohort_v4 import build_fixture, fixture_config  # noqa: E402


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = fixture_config(train_stride=2)
        self.config["train"]["max_epochs"] = 1
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        self.fixture = build_fixture(self.root, self.config)
        self.source = self.root / "race_cache"
        self.source.mkdir()
        manifest = self.source / "normalized_source_manifest.json"
        manifest.write_text(json.dumps({"fixture": True}), encoding="utf-8")
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        (self.source / "normalized_source_manifest.json.sha256").write_text(f"{digest}  x\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _argv(self, output, *extra):
        return ["--normalized-source", str(self.source), "--config", str(self.config_path),
                "--output", str(output), "--cohort-root", str(self.root / "cohorts"),
                "--threads", "1", "--log-every-batches", "1", *extra]

    def _source(self, *_args, **_kwargs):
        specs, records, paths, labels = self.fixture
        return specs, records, paths, labels, "fixture-hash"

    def test_preflight_builds_nothing(self):
        with patch.object(cohort_v4, "_read_source", self._source):
            code = train_gnnv4.main(self._argv(self.root / "pre", "--preflight-only", "--device", "cpu"))
        self.assertEqual(code, 0)
        self.assertFalse((self.root / "cohorts").exists())
        self.assertFalse((self.root / "pre").exists())

    def test_full_fixture_run_with_cuda_fallback_then_cohort_reuse(self):
        with patch.object(cohort_v4, "_read_source", self._source), \
                patch.object(torch.cuda, "is_available", return_value=False):
            code = train_gnnv4.main(self._argv(self.root / "run1", "--device", "cuda"))
        self.assertEqual(code, 0)
        report = json.loads((self.root / "run1" / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertIn("r2_score", report["metrics"])
        self.assertEqual(report["device"], "cpu")
        self.assertIn("dense_test_stride1", report["supplementary"])
        self.assertIn("corner_loss_a", report["aero_coefficients"])
        cohorts = list((self.root / "cohorts").iterdir())
        self.assertEqual(len(cohorts), 1)
        # Second run reuses the fingerprinted cohort (the source is never read again).
        with patch.object(cohort_v4, "_read_source", side_effect=AssertionError("rebuilt")):
            code = train_gnnv4.main(self._argv(self.root / "run2", "--device", "cpu", "--skip-dense-test"))
        self.assertEqual(code, 0)
        self.assertEqual(len(list((self.root / "cohorts").iterdir())), 1)

    def test_existing_output_requires_resume(self):
        (self.root / "taken").mkdir()
        self.assertEqual(train_gnnv4.main(self._argv(self.root / "taken", "--device", "cpu")), 2)

    def test_training_memory_preflight_failure_returns_3(self):
        with patch.object(cohort_v4, "_read_source", self._source), \
                patch.object(cohort_v4, "estimate_training_bytes", return_value=10 ** 15):
            code = train_gnnv4.main(self._argv(self.root / "mem", "--device", "cpu"))
        self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()
