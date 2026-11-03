"""Isolated temp-directory checks for GNNv3 checkpoint and source-cache guards."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_2023_fastf1_season as runner
from prepare_cohort import _canonical_json, _sha256_bytes


class GNNv3RunGuardTests(unittest.TestCase):
    def test_checkpoint_copy_is_verified_and_source_remains_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "old" / "training" / "fingerprint" / "best.pt"
            output = base / "new"
            source.parent.mkdir(parents=True)
            payload = b"isolated fixture checkpoint bytes"
            source.write_bytes(payload)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            runner._archive_production_checkpoints([source], output)
            backup = output / "production_model_backup/01-best.pt"
            self.assertEqual(hashlib.sha256(backup.read_bytes()).hexdigest(), before)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            self.assertTrue((backup.parent / "production_checkpoint_manifest.json").is_file())

    def test_conflicting_backup_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "best.pt"
            source.write_bytes(b"source")
            backup = base / "output/production_model_backup/01-best.pt"
            backup.parent.mkdir(parents=True)
            backup.write_bytes(b"different")
            with self.assertRaisesRegex(RuntimeError, "backup differs"):
                runner._archive_production_checkpoints([source], base / "output")

    def test_source_output_alias_is_rejected_before_any_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "separate, disjoint"):
                runner.run(SimpleNamespace(
                    output=root, reuse_prepared_from=root, training_overrides=Path("override.yaml"),
                    preserve_production_checkpoint=None,
                    preserve_production_last_checkpoint=None,
                ))
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_cache_inventory_fails_without_writing_to_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            old_run = base / "old"
            old_run.mkdir()
            checkpoint = old_run / "best.pt"
            checkpoint.write_bytes(b"fixture production")
            output = base / "new"
            args = SimpleNamespace(
                output=output, reuse_prepared_from=old_run,
                training_overrides=Path("profile.yaml"),
                preserve_production_checkpoint=checkpoint,
                preserve_production_last_checkpoint=None,
                local_data_only=True, raw_sessions_dir=old_run / "missing-raw-sessions",
                require_existing_caches=True,
            )
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                runner.run(args)
            self.assertEqual(checkpoint.read_bytes(), b"fixture production")
            self.assertFalse((old_run / "race_cache").exists())
            self.assertTrue((output / "production_model_backup/01-best.pt").is_file())
            self.assertFalse(any((output / "training").glob("*/best.pt")))

    def test_explicit_no_production_checkpoint_does_not_create_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            old_run = base / "old"
            old_run.mkdir()
            output = base / "new"
            args = SimpleNamespace(
                output=output, reuse_prepared_from=old_run,
                training_overrides=Path("profile.yaml"),
                preserve_production_checkpoint=None,
                preserve_production_last_checkpoint=None,
                no_production_checkpoint=True,
                local_data_only=True, raw_sessions_dir=old_run / "missing-raw-sessions",
                require_existing_caches=True,
            )
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                runner.run(args)
            self.assertFalse((output / "production_model_backup").exists())

    def test_manifest_preflight_checks_artifact_without_unpickling(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary) / "cohort"
            cohort.mkdir()
            artifact = cohort / "prepared_cohort.pkl"
            artifact.write_bytes(b"opaque prepared artifact")
            config = {"data": {"split_method": "race"}}
            source_hash = "a" * 64
            manifest = {
                "provider": "fastf1",
                "sessions": [{"race_id": "2023-Test-R"}],
                "prepared_manifest_path": str((cohort / "prepared_manifest.json").resolve()),
                "prepared_artifact_path": str(artifact.resolve()),
                "prepared_artifact_bytes": artifact.stat().st_size,
                "prepared_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "effective_config_sha256": _sha256_bytes(_canonical_json(config)),
                "aligned_source_manifest_sha256": source_hash,
            }
            manifest["prepared_manifest_sha256"] = _sha256_bytes(_canonical_json(manifest))
            manifest_path = cohort / "prepared_manifest.json"
            raw = __import__("json").dumps(manifest, indent=2).encode()
            manifest_path.write_bytes(raw)
            Path(str(manifest_path) + ".sha256").write_text(
                hashlib.sha256(raw).hexdigest() + "  prepared_manifest.json\n", encoding="utf-8")

            self.assertIsNotNone(runner._cohort_manifest_cached(cohort, config, source_hash))
            artifact.write_bytes(b"tampered artifact")
            self.assertIsNone(runner._cohort_manifest_cached(cohort, config, source_hash))


if __name__ == "__main__":
    unittest.main()
