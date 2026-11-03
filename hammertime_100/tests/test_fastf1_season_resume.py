"""Offline cache-contract checks; fixtures are not presented as race data."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from hammertime_100 import run_2023_fastf1_season as runner


def _offline_process_task(task):
    """Small process-pool fixture, not Formula 1 data."""
    time.sleep(task["delay"])
    if task.get("fail"):
        return None, "FixtureFailure"
    return {"race_id": task["race_id"], "pid": __import__("os").getpid()}, None


class FastF1SeasonResumeTests(unittest.TestCase):
    def test_worker_memory_cap_uses_available_bytes_and_fails_closed(self):
        gib = 1024 ** 3
        self.assertEqual(runner._worker_memory_limit(8, available_bytes=3 * gib), 1)
        self.assertEqual(runner._worker_memory_limit(3, available_bytes=16 * gib), 3)
        with patch.object(runner, "_available_memory_bytes", return_value=None):
            self.assertEqual(runner._worker_memory_limit(4), 1)

    def test_cached_calendar_location_aliases_resolve_reviewed_circuits(self):
        class FakeFastF1:
            __version__ = "3.9.0"

            @staticmethod
            def get_event_schedule(_year, include_testing=False):
                return pd.DataFrame([
                    {"EventName": "Canadian Grand Prix", "Location": "Montréal", "RoundNumber": 8},
                    {"EventName": "Singapore Grand Prix", "Location": "Marina Bay", "RoundNumber": 16},
                    {"EventName": "Abu Dhabi Grand Prix", "Location": "Yas Island", "RoundNumber": 23},
                ])

        with tempfile.TemporaryDirectory() as tmp:
            events, skips = runner.discover(FakeFastF1, Path(tmp))
        self.assertEqual(skips, [])
        self.assertEqual([event["lap_length_m"] for event in events], [4361.0, 4940.0, 5281.0])
        self.assertEqual([event["circuit_id"] for event in events], ["canada", "singapore", "abu-dhabi"])

    def test_failure_diagnostics_redact_credentials_and_tokens(self):
        secret = "fixture-password"
        try:
            raise RuntimeError(f"token=secret-token password={secret}")
        except RuntimeError as exc:
            details = runner._sanitized_error(exc, ("fixture-user", secret))
        self.assertNotIn(secret, details["message"])
        self.assertNotIn(secret, details["traceback"])
        self.assertNotIn("secret-token", details["message"])
        self.assertNotIn("secret-token", details["traceback"])

    def test_completed_normalized_stage_is_reused_and_partial_cache_rejected(self):
        event = {"year": 2023, "gp": "Fixture Grand Prix", "session_type": "R",
                 "race_id": "2023-Fixture-R", "source": "fastf1", "circuit_id": "fixture",
                 "lap_length_m": 1000.0}
        config = {"preprocess": {}, "target": {}, "track": {"lap_length_m": 1000.0}}
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw_sessions"
            raw_stage = raw_root / event["race_id"]
            raw_stage.mkdir(parents=True)
            raw_files = {}
            for name, data in (("raw_laps.parquet", b"source laps fixture"),
                               ("raw_telemetry.parquet", b"source telemetry fixture")):
                path = raw_stage / name
                path.write_bytes(data)
                raw_files[name] = {"bytes": path.stat().st_size,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            raw_manifest = {"files": raw_files}
            event_fingerprint = runner._normalization_fingerprint(event, raw_manifest, config, "3.9.0")
            stage_root = Path(tmp) / "sessions"
            stage = stage_root / event["race_id"]
            stage.mkdir(parents=True)
            for name, data in (("aligned_telemetry.parquet", b"aligned fixture"),
                               ("target_labels.parquet", b"labels fixture")):
                (stage / name).write_bytes(data)
            files = {path.name: {"bytes": path.stat().st_size,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in stage.iterdir()}
            source_files = {str((raw_stage / name).resolve()): info for name, info in raw_files.items()}
            final_event = {**event, "validated_target_rows": 1}
            manifest = {"fingerprint": event_fingerprint, "event": final_event,
                        "files": files, "source_files": source_files,
                        "source_metadata": {"provider": "fastf1", "session": {"circuit_key": "fixture"}},
                        "coverage": {"aligned_rows": 1, "target_rows": 1}}
            (stage / "stage_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIsNotNone(runner._valid_race_stage(stage, event_fingerprint))
            (stage / "aligned_telemetry.parquet").write_bytes(b"partial replacement")
            self.assertIsNone(runner._valid_race_stage(stage, event_fingerprint))

    def test_spawned_cpu_queue_uses_distinct_processes_and_isolates_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool, pids = runner._start_worker_pool(2)
            try:
                tasks = [
                    {"race_id": "round-1", "delay": 0.04},
                    {"race_id": "round-2", "delay": 0.01, "fail": True},
                    {"race_id": "round-3", "delay": 0.02},
                    {"race_id": "round-4", "delay": 0.00},
                ]
                futures = {pool.submit(_offline_process_task, task): task["race_id"] for task in tasks}
                results, failures = runner._collect_normalization_futures(futures)
            finally:
                pool.shutdown(wait=True)
            self.assertEqual(len(pids), 2)
            self.assertGreaterEqual(len({value["pid"] for value in results.values()}), 2)
            self.assertEqual(failures, {"round-2": "FixtureFailure"})
            calendar_results = [results[task["race_id"]]["race_id"] for task in tasks
                                if task["race_id"] in results]
            self.assertEqual(calendar_results, ["round-1", "round-3", "round-4"])

    def test_atomic_prediction_cache_skips_inference_on_export_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
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
            stage.mkdir(parents=True)
            raw_stage = source_root / "raw_sessions" / event["race_id"]
            raw_stage.mkdir(parents=True)
            config_path = Path(tmp) / "source.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            aligned_path, labels_path = stage / "aligned_telemetry.parquet", stage / "target_labels.parquet"
            aligned_frame = pd.DataFrame({"RaceId": [event["race_id"]], "Driver": ["AA"],
                                          "TimeSec": [0.0], "LapNumber": [1], "IsValid": [True],
                                          "value": [1.0]})
            label_frame = pd.DataFrame({"RaceId": [event["race_id"]], "Driver": ["AA"],
                                        "TimeSec": [0.0], "target": [0.2]})
            aligned_frame.to_parquet(aligned_path, index=False)
            label_frame.to_parquet(labels_path, index=False)
            raw_path = raw_stage / "raw_laps.parquet"
            raw_path.write_bytes(b"authentic export fixture")
            files = {path.relative_to(stage).as_posix(): {"bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in (aligned_path, labels_path)}
            source_files = {str(raw_path.resolve()): {"bytes": raw_path.stat().st_size,
                                                       "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}}
            record = {"fingerprint": "fixture", "event": event, "files": files,
                      "source_files": source_files,
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

            import prepare_cohort as cohort
            lazy_aligned, lazy_labels, _, _, _ = _read_normalized_source(
                source_root, specs, "fastf1", load_aligned=False)
            self.assertIsInstance(lazy_aligned[0], Path)
            eager_coverage = cohort._coverage(aligned, labels[0], [])
            lazy_coverage = cohort._coverage(lazy_aligned, lazy_labels[0], [])
            self.assertEqual(lazy_coverage, eager_coverage)
            with patch.object(cohort, "build_race_graph_snapshots",
                              return_value=([], {"AA": 0})) as build_graphs:
                snapshots, driver_map = cohort._build_snapshots_by_race(
                    lazy_aligned, {"session": {}, "track": {}, "graph": {"num_nodes": 20}}, specs)
            self.assertEqual(snapshots, [])
            self.assertEqual(driver_map, {"AA": 0})
            self.assertEqual(build_graphs.call_args.args[0]["RaceId"].unique().tolist(),
                             [event["race_id"]])

    def test_cohort_fingerprint_resolves_all_prepare_code_modules(self):
        config = {"fixture": 1}
        fingerprint = runner._cohort_fingerprint(config, "fixture-source-hash")
        self.assertEqual(len(fingerprint), 64)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cohort_prefix = out / "cohorts" / fingerprint[:20]
            candidates = [cohort_prefix, *sorted((out / "cohorts").glob(f"{fingerprint[:20]}-rebuild*"))]
            self.assertEqual(candidates, [cohort_prefix])
            rebuild_dir = out / "cohorts" / f"{fingerprint[:20]}-rebuild1"
            rebuild_dir.mkdir(parents=True)
            candidates = [cohort_prefix, *sorted((out / "cohorts").glob(f"{fingerprint[:20]}-rebuild*"))]
            self.assertEqual(candidates, [cohort_prefix, rebuild_dir])

    def test_align_cohort_num_nodes_expands_for_substitutions(self):
        config = {"graph": {"num_nodes": 20}}
        events = [{"race_id": "r1"}, {"race_id": "r2"}]
        with tempfile.TemporaryDirectory() as tmp:
            stages_root = Path(tmp)
            for r_id, drivers in [("r1", ["1", "2", "3"]), ("r2", ["1", "4", "5", "6"])]:
                stage = stages_root / r_id
                stage.mkdir()
                pd.DataFrame({"Driver": drivers, "value": [1.0] * len(drivers)}).to_parquet(
                    stage / "aligned_telemetry.parquet", index=False
                )
            num_nodes = runner._align_cohort_num_nodes(config, events, stages_root)
            self.assertEqual(num_nodes, 20)
            self.assertEqual(config["graph"]["num_nodes"], 20)

            extra_drivers = [str(i) for i in range(10, 27)]
            stage_extra = stages_root / "r3"
            stage_extra.mkdir()
            pd.DataFrame({"Driver": extra_drivers, "value": [1.0] * len(extra_drivers)}).to_parquet(
                stage_extra / "aligned_telemetry.parquet", index=False
            )
            events.append({"race_id": "r3"})
            num_nodes = runner._align_cohort_num_nodes(config, events, stages_root)
            self.assertEqual(num_nodes, 23)
            self.assertEqual(config["graph"]["num_nodes"], 23)


if __name__ == "__main__":
    unittest.main()
