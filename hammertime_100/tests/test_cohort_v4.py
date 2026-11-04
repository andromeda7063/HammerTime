"""Offline contracts for the memory-safe compact cohort (cohort_v4).

All tables are small synthetic fixtures that exercise code paths only; they are
not race data and carry no performance meaning.
"""

import copy
import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cohort_v4  # noqa: E402
from context_features import prepare_context_features  # noqa: E402
from dataset import create_dataloaders  # noqa: E402
from graph import build_race_graph_snapshots, remap_cohort_driver_nodes  # noqa: E402
from model import create_model  # noqa: E402
from train import train_model  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "train_2023_fastf1_season_gnnv4.yaml"
LAP = 1000.0
DRIVERS = ("1", "11", "16", "44")


def fixture_config(train_stride=2):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["graph"]["num_nodes"] = 4
    config["track"]["lap_length_m"] = LAP
    config["train"]["batch_size"] = 4
    config["train"]["max_epochs"] = 1
    config["train"]["grad_accum_steps"] = 1
    config["train"]["lr_scheduler"] = "none"
    config["data"]["train_window_stride"] = train_stride
    return config


def race_tables(race_index, seconds=40.0):
    rows, labels = [], []
    times = np.arange(0.0, seconds, 0.25)
    rng = np.random.default_rng(race_index)
    for car, driver in enumerate(DRIVERS):
        speed_mps = 48.0 + 2.0 * car
        offset = 30.0 * car
        for t in times:
            progress = offset + speed_mps * t
            lap = int(progress // LAP) + 1
            distance = progress % LAP
            angle = 2 * math.pi * distance / LAP
            rows.append({
                "TimeSec": float(t), "Driver": driver, "Speed": speed_mps * 3.6 + rng.normal(0, 1),
                "Throttle": 90.0, "Brake": 0.0, "DRS": 0.0, "Distance": distance,
                "X": 160.0 * math.cos(angle), "Y": 160.0 * math.sin(angle), "Z": 0.0,
                "IsValid": True, "DRSActive": float(car == 1), "DRSSupported": 1.0, "a_long": 0.0,
                "LapNumber": lap, "Compound": "MEDIUM", "Team": f"T{car}", "TyreLife": 3.0 + lap,
                "IsPitLap": False, "IsSafetyCar": False,
                "MiniSector": int(distance // (LAP / 20)), "RaceId": f"R{race_index}",
                "DriverAhead": None, "DeltaT": 0.6 if car else 99.0,
            })
        for t in times[::3]:
            progress = offset + speed_mps * t
            labels.append({"Driver": driver, "TimeSec": float(t), "SectorExitTimeSec": float(t) + 1.0,
                           "LapNumber": int(progress // LAP) + 1,
                           "MiniSector": int((progress % LAP) // (LAP / 20)),
                           "Duration": 1.0, "Baseline": 1.0,
                           "DeltaT_Sector": float(rng.normal(0.05, 0.1)),
                           "AvgGap": 0.6 if car else 4.0, "Team": f"T{car}", "RaceId": f"R{race_index}"})
    return pd.DataFrame(rows), pd.DataFrame(labels)


def build_fixture(directory: Path, config: dict, races=3):
    specs, records, paths, labels = [], [], [], []
    for index in range(races):
        aligned, label_table = race_tables(index)
        path = directory / f"aligned_{index}.parquet"
        aligned.to_parquet(path, index=False)
        specs.append({"race_id": f"R{index}", "circuit_id": f"c{index}", "lap_length_m": LAP,
                      "year": 2023, "gp": f"G{index}", "session_type": "R"})
        records.append({"race_id": f"R{index}", "provider": "fastf1", "year": 2023,
                        "gp": f"G{index}", "session_type": "R", "session_key": None})
        paths.append(path)
        labels.append(label_table)
    return specs, records, paths, labels


class CohortV4Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = fixture_config(train_stride=2)
        self.specs, self.records, self.paths, self.labels = build_fixture(self.root, self.config)
        self.out = self.root / "cohort"
        self.manifest = cohort_v4.build_compact_cohort_from_tables(
            self.specs, self.records, self.paths, self.labels, "fixture", self.config, self.out,
            headroom_fraction=0.0)

    def tearDown(self):
        self.temporary.cleanup()

    def _legacy_loaders(self):
        snapshots, maps = [], {}
        for spec, path, labels in zip(self.specs, self.paths, self.labels):
            race_config = copy.deepcopy(self.config)
            race_config["session"].update(spec)
            race_config["track"]["lap_length_m"] = LAP
            contextual = prepare_context_features(
                pd.read_parquet(path), {spec["race_id"]: spec}, schema="circuit_context_v3")
            race_snaps, driver_map = build_race_graph_snapshots(contextual, race_config)
            snapshots.extend(race_snaps)
            maps[spec["race_id"]] = driver_map
        remap_cohort_driver_nodes(snapshots, maps, 4)
        all_labels = pd.concat(self.labels, ignore_index=True)
        driver_to_idx = {driver: index for index, driver in enumerate(sorted(DRIVERS))}
        return create_dataloaders(snapshots, all_labels, driver_to_idx, self.config)

    def test_round_trip_matches_legacy_dataloaders(self):
        legacy_train, legacy_val, legacy_test, legacy_scaler = self._legacy_loaders()
        cohort = cohort_v4.CompactCohort(self.out)
        train, val, test, scaler = cohort_v4.create_compact_dataloaders(cohort, self.config)
        np.testing.assert_allclose(scaler.mean_, legacy_scaler.mean_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(scaler.scale_, legacy_scaler.scale_, rtol=1e-6, atol=1e-6)
        self.assertEqual(cohort.manifest["split"]["train"], legacy_train.dataset.split_metadata["train"]["race_ids"])
        for new_loader, old_loader in ((val, legacy_val), (test, legacy_test)):
            self.assertEqual(len(new_loader.dataset), len(old_loader.dataset))
            for new_batch, old_batch in zip(new_loader, old_loader):
                torch.testing.assert_close(new_batch[1], old_batch[1])
                torch.testing.assert_close(new_batch[2], old_batch[2])
                torch.testing.assert_close(new_batch[3], old_batch[3])
                for new_window, old_window in zip(new_batch[0], old_batch[0]):
                    for new_snap, old_snap in zip(new_window, old_window):
                        torch.testing.assert_close(new_snap.x, old_snap.x, rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(new_snap.edge_index, old_snap.edge_index)
                        torch.testing.assert_close(new_snap.edge_attr, old_snap.edge_attr, rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(new_snap.aero_raw, old_snap.aero_raw)
            old_meta = old_loader.dataset.samples[0]
            new_meta = new_loader.dataset.samples[0]
            self.assertEqual(new_meta["race_id"], old_meta["race_id"])
            np.testing.assert_allclose(new_meta["gap_sec"], old_meta["gap_sec"])

    def test_train_stride_one_uses_more_windows_and_same_eval(self):
        config = fixture_config(train_stride=1)
        out = self.root / "cohort_stride1"
        cohort_v4.build_compact_cohort_from_tables(self.specs, self.records, self.paths, self.labels,
                                                   "fixture", config, out, headroom_fraction=0.0)
        dense = cohort_v4.create_compact_dataloaders(cohort_v4.CompactCohort(out), config)
        sparse = cohort_v4.create_compact_dataloaders(cohort_v4.CompactCohort(self.out), self.config)
        self.assertGreater(len(dense[0].dataset), len(sparse[0].dataset))
        self.assertEqual(len(dense[2].dataset), len(sparse[2].dataset))

    def test_corruption_is_detected(self):
        target = next(self.out.rglob("win_target.npy"))
        array = np.load(target)
        array[0, 0] += 1.0
        np.save(target, array)
        with self.assertRaisesRegex(ValueError, "corrupt"):
            cohort_v4.CompactCohort(self.out)

    def test_dataset_pickles_without_memory_maps(self):
        cohort = cohort_v4.CompactCohort(self.out)
        train, _, _, _ = cohort_v4.create_compact_dataloaders(cohort, self.config)
        train.dataset[0]
        self.assertTrue(cohort._cache)
        restored = pickle.loads(pickle.dumps(train.dataset))
        self.assertEqual(restored.cohort._cache, {})
        torch.testing.assert_close(restored[0][1], train.dataset[0][1])

    def test_workers_and_training_step_run(self):
        cohort = cohort_v4.CompactCohort(self.out)
        train, val, _, _ = cohort_v4.create_compact_dataloaders(cohort, self.config, num_workers=1)
        config = copy.deepcopy(self.config)
        config["train"]["checkpoint_path"] = str(self.root / "train" / "best.pt")
        config["train"]["history_path"] = str(self.root / "train" / "history.csv")
        model = create_model(config)
        score, history = train_model(model, train, val, config, torch.device("cpu"))
        self.assertTrue(np.isfinite(score))
        self.assertIn("aero_corner_loss_a", history[0])

    def test_existing_output_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            cohort_v4.build_compact_cohort_from_tables(self.specs, self.records, self.paths, self.labels,
                                                       "fixture", self.config, self.out, headroom_fraction=0.0)


class MemoryPreflightTests(unittest.TestCase):
    def test_ok_and_insufficient(self):
        ok = cohort_v4.memory_preflight("stage", 1 * cohort_v4.GIB, 0.25, available_bytes=4 * cohort_v4.GIB)
        self.assertEqual(ok["status"], "ok")
        with self.assertRaises(cohort_v4.MemoryPreflightError):
            cohort_v4.memory_preflight("stage", 4 * cohort_v4.GIB, 0.25, available_bytes=4 * cohort_v4.GIB)

    def test_unknown_memory_continues(self):
        with patch.object(cohort_v4, "available_memory_bytes", return_value=None):
            self.assertEqual(cohort_v4.memory_preflight("stage", 10 ** 15)["status"], "unknown")

    def test_build_aborts_before_writing_when_memory_is_short(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = fixture_config()
            specs, records, paths, labels = build_fixture(root, config)
            with patch.object(cohort_v4, "available_memory_bytes", return_value=1024):
                with self.assertRaises(cohort_v4.MemoryPreflightError):
                    cohort_v4.build_compact_cohort_from_tables(specs, records, paths, labels, "fixture",
                                                               config, root / "cohort")
            self.assertFalse((root / "cohort").exists())
            self.assertEqual(list(root.glob(".cohort.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
