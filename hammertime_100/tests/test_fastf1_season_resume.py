"""Offline cache-contract checks; fixtures are not presented as race data."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import yaml

from hammertime_100 import run_2023_fastf1_season as runner


class FastF1SeasonResumeTests(unittest.TestCase):
    def test_completed_race_stage_is_reused_without_loading_session(self):
        event = {"year": 2023, "gp": "Fixture Grand Prix", "session_type": "R",
                 "race_id": "2023-Fixture-R", "source": "fastf1", "circuit_id": "fixture",
                 "lap_length_m": 1000.0}
        config = {"preprocess": {}, "target": {}, "track": {"lap_length_m": 1000.0}}
        with tempfile.TemporaryDirectory() as tmp:
            stage_root = Path(tmp) / "sessions"
            stage = stage_root / event["race_id"]
            stage.mkdir(parents=True)
            for name, data in (("aligned_telemetry.parquet", b"aligned fixture"),
                               ("target_labels.parquet", b"label fixture"),
                               ("raw_laps.parquet", b"authentic-source-export fixture")):
                (stage / name).write_bytes(data)
            files = {path.name: {"bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in stage.iterdir()}
            final_event = {**event, "validated_target_rows": 1}
            manifest = {"fingerprint": runner._race_fingerprint(event, config, "3.9.0"),
                        "event": final_event, "files": files,
                        "source_metadata": {"provider": "fastf1", "session": {"circuit_key": "fixture"}},
                        "coverage": {"aligned_rows": 1, "target_rows": 1}}
            (stage / "stage_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            fake_ingest = types.ModuleType("ingest")
            fake_ingest.load_race = Mock(side_effect=AssertionError("cached session must not be loaded"))
            fake_ingest.export_raw_tables = Mock(side_effect=AssertionError("cached raw tables must not be exported"))
            fake_preprocess = types.ModuleType("preprocess")
            fake_preprocess.process_race = Mock(side_effect=AssertionError("cached telemetry must not be normalized"))
            fake_target = types.ModuleType("target")
            fake_target.build_targets = Mock(side_effect=AssertionError("cached targets must not be rebuilt"))
            with patch.dict(sys.modules, {"ingest": fake_ingest, "preprocess": fake_preprocess,
                                          "target": fake_target}):
                result, error = runner.ensure_race_stage(event, Path(tmp) / "fastf1-cache", stage_root,
                                                         config, "3.9.0")
            self.assertIsNone(error)
            self.assertEqual(result["validated_target_rows"], 1)

    def test_partial_race_staging_is_ignored_and_prediction_cache_skips_inference(self):
        event = {"year": 2023, "gp": "Fixture Grand Prix", "session_type": "R",
                 "race_id": "2023-Fixture-R", "source": "fastf1", "circuit_id": "fixture",
                 "lap_length_m": 1000.0}
        config = {"preprocess": {}, "target": {}, "track": {"lap_length_m": 1000.0}}
        with tempfile.TemporaryDirectory() as tmp:
            stage_root = Path(tmp) / "sessions"
            stage_root.mkdir()
            incomplete = stage_root / ".2023-Fixture-R.staging-interrupted"
            incomplete.mkdir()
            (incomplete / "aligned_telemetry.parquet").write_bytes(b"partial")
            fake_ingest = types.ModuleType("ingest")
            fake_ingest.load_race = Mock(side_effect=RuntimeError("offline fixture; no network call"))
            fake_ingest.export_raw_tables = Mock()
            fake_preprocess = types.ModuleType("preprocess")
            fake_preprocess.process_race = Mock()
            fake_target = types.ModuleType("target")
            fake_target.build_targets = Mock()
            with patch.dict(sys.modules, {"ingest": fake_ingest, "preprocess": fake_preprocess,
                                          "target": fake_target}):
                result, error = runner.ensure_race_stage(event, Path(tmp) / "fastf1-cache", stage_root,
                                                         config, "3.9.0")
            self.assertIsNone(result)
            self.assertEqual(error, "RuntimeError")
            self.assertTrue(incomplete.exists())

            cache = Path(tmp) / "predictions.pkl"
            calls = []
            first = runner.load_or_compute_predictions(
                cache, "fixture-fingerprint", lambda: (calls.append(1) or pd.DataFrame({"y_true": [1.0], "y_pred": [0.9]})))
            second = runner.load_or_compute_predictions(
                cache, "fixture-fingerprint", lambda: (_ for _ in ()).throw(AssertionError("inference should be cached")))
            self.assertEqual(calls, [1])
            self.assertEqual(first.to_dict(), second.to_dict())

    def test_verified_cohort_cache_is_reused(self):
        from prepare_cohort import _canonical_json, _sha256_bytes

        config = {"fixture_config": "identity"}
        config_hash = _sha256_bytes(_canonical_json(config))
        memberships = {}
        membership_hashes = {}
        with tempfile.TemporaryDirectory() as tmp:
            cohort_dir = Path(tmp) / "cohort"
            cohort_dir.mkdir()
            artifact = cohort_dir / "prepared_cohort.pkl"
            payload = {"schema_version": 1, "config": config, "memberships": memberships,
                       "membership_sha256": membership_hashes,
                       "provenance": {"provider": "fastf1", "session_identity": "fixture",
                                      "effective_config_sha256": config_hash,
                                      "split_membership_sha256": membership_hashes}}
            with artifact.open("wb") as stream:
                pickle.dump(payload, stream)
            manifest = {"schema_version": 1, "provider": "fastf1", "providers": ["fastf1"],
                        "session_identity": "fixture", "effective_config_sha256": config_hash,
                        "split_membership_sha256": membership_hashes, "split_membership": memberships,
                        "prepared_artifact_path": str(artifact.resolve()),
                        "prepared_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "prepared_artifact_bytes": artifact.stat().st_size,
                        "prepared_manifest_path": str((cohort_dir / "prepared_manifest.json").resolve()),
                        "prepared_manifest_sha256": None, "source_hashes": {}, "coverage": {}, "split": {}}
            manifest["aligned_source_manifest_sha256"] = "fixture-source-hash"
            manifest["prepared_manifest_sha256"] = _sha256_bytes(_canonical_json(
                {key: value for key, value in manifest.items() if key != "prepared_manifest_sha256"}))
            manifest_path = cohort_dir / "prepared_manifest.json"
            manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
            manifest_path.write_bytes(manifest_bytes)
            (cohort_dir / "prepared_manifest.json.sha256").write_text(
                f"{hashlib.sha256(manifest_bytes).hexdigest()}  prepared_manifest.json\n", encoding="utf-8")
            first = runner._cohort_cached(cohort_dir, config, "fixture-source-hash")
            second = runner._cohort_cached(cohort_dir, config, "fixture-source-hash")
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)

    def test_normalized_source_manifest_reuses_verified_tables_without_cohort_claim(self):
        from prepare_cohort import _read_normalized_source
        from run_pipeline import _session_specs

        event = {"year": 2023, "gp": "Fixture Grand Prix", "session_type": "R",
                 "race_id": "2023-Fixture-R", "source": "fastf1", "circuit_id": "fixture",
                 "lap_length_m": 1000.0}
        config = {"session": {**event, "cache_dir": "fixture-cache", "processed_dir": "fixture-processed"},
                  "sessions": [event], "track": {"lap_length_m": 1000.0}}
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "normalized"
            stage_root = source_root / "sessions"
            stage = stage_root / event["race_id"]
            (stage / "raw").mkdir(parents=True)
            config_path = Path(tmp) / "source.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            aligned_path, labels_path = stage / "aligned_telemetry.parquet", stage / "target_labels.parquet"
            pd.DataFrame({"RaceId": [event["race_id"]], "value": [1.0]}).to_parquet(aligned_path, index=False)
            pd.DataFrame({"RaceId": [event["race_id"]], "target": [0.2]}).to_parquet(labels_path, index=False)
            raw_path = stage / "raw" / "raw_laps.parquet"
            raw_path.write_bytes(b"authentic export fixture")
            files = {path.relative_to(stage).as_posix(): {"bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in (aligned_path, labels_path, raw_path)}
            record = {"fingerprint": "fixture", "event": event, "files": files,
                      "source_metadata": {"provider": "fastf1", "session": {"circuit_key": "fixture"}},
                      "coverage": {"aligned_rows": 1, "target_rows": 1}}
            (stage / "stage_manifest.json").write_text(json.dumps(record), encoding="utf-8")
            runner.write_normalized_source(stage_root, source_root, config_path, [event])
            specs = _session_specs(config)
            aligned, labels, records, paths, manifest = _read_normalized_source(source_root, specs, "fastf1")
            self.assertEqual(len(aligned), 1)
            self.assertEqual(len(labels), 1)
            self.assertEqual(records[0]["source_metadata"]["session"]["circuit_key"], "fixture")
            self.assertEqual(manifest["artifact_type"], "hammertime-normalized-source-cache")
            self.assertNotIn("prepared_cohort.pkl", [Path(path).name for path in paths])


if __name__ == "__main__":
    unittest.main()
