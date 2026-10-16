"""Focused regression tests for M5 alignment, isolation, and leakage controls."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_windowed_samples, create_dataloaders


def snapshot(time_sec, race_id="race-a", driver_names=("AA", "BB"), value=None, safety_car=False):
    if value is None:
        value = time_sec
    x = torch.zeros((len(driver_names), 9), dtype=torch.float32)
    x[:, 0] = float(value)
    return Data(
        x=x,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        mask=torch.ones(len(driver_names), dtype=torch.bool),
        time_sec=float(time_sec),
        race_id=race_id,
        driver_names=tuple(driver_names),
        is_safety_car=safety_car,
    )


class DatasetContractTests(unittest.TestCase):
    def test_target_uses_nearest_grid_frame_and_target_snapshot_driver_order(self):
        frames = [snapshot(i * 0.25, driver_names=("AA", "BB")) for i in range(5)]
        labels = pd.DataFrame([{
            "RaceId": "race-a", "Driver": "AA", "TimeSec": 0.51,
            "DeltaT_Sector": 1.25,
        }])

        samples = build_windowed_samples(
            frames, labels, {"AA": 1, "BB": 0}, W=2, K=1,
            timestamp_tolerance_sec=0.1275, continuity_tolerance_sec=0.05,
        )

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["target_time"], 0.5)
        self.assertTrue(sample["mask"].tolist() == [True, False])
        self.assertAlmostEqual(float(sample["target"][0]), 1.25)

    def test_windows_are_isolated_by_race_when_timestamps_repeat(self):
        frames = [snapshot(t, race_id=race) for race in ("race-a", "race-b")
                  for t in (0.0, 0.25, 0.5)]
        labels = pd.DataFrame([
            {"RaceId": race, "Driver": "AA", "TimeSec": 0.5, "DeltaT_Sector": value}
            for race, value in (("race-a", 1.0), ("race-b", 2.0))
        ])

        samples = build_windowed_samples(
            frames, labels, {"AA": 0, "BB": 1}, W=2, K=1,
            timestamp_tolerance_sec=0.1275, continuity_tolerance_sec=0.05,
        )

        self.assertEqual({sample["race_id"] for sample in samples}, {"race-a", "race-b"})
        self.assertEqual({float(sample["target"][0]) for sample in samples}, {1.0, 2.0})
        for sample in samples:
            self.assertEqual({frame.race_id for frame in sample["snapshots"]}, {sample["race_id"]})

    def test_safety_car_frames_are_removed_from_full_forecast_span(self):
        frames = [snapshot(i * 0.25, safety_car=(i == 2)) for i in range(6)]
        labels = pd.DataFrame([
            {"RaceId": "race-a", "Driver": "AA", "TimeSec": i * 0.25,
             "DeltaT_Sector": float(i)}
            for i in range(6)
        ])

        samples = build_windowed_samples(
            frames, labels, {"AA": 0, "BB": 1}, W=2, K=1,
            timestamp_tolerance_sec=0.1275, continuity_tolerance_sec=0.05,
        )

        self.assertEqual([sample["target_time"] for sample in samples], [1.25])
        self.assertFalse(any(frame.is_safety_car for frame in samples[0]["snapshots"]))

    def test_temporal_split_purges_overlap_and_scaler_fits_only_training_frames(self):
        frames = [snapshot(i * 0.25, value=float(i)) for i in range(100)]
        labels = pd.DataFrame([
            {"RaceId": "race-a", "Driver": driver, "TimeSec": i * 0.25,
             "DeltaT_Sector": float(i % 7)}
            for i in range(100) for driver in ("AA", "BB")
        ])
        config = {
            "data": {"window_size": 2, "horizon": 1, "train_ratio": 0.6,
                     "val_ratio": 0.2, "test_ratio": 0.2, "split_method": "temporal",
                     "timestamp_tolerance_sec": 0.1275, "continuity_tolerance_sec": 0.05,
                     "min_valid_targets_per_sample": 1, "random_seed": 42},
            "preprocess": {"sample_hz": 4.0, "flag_safety_car": True},
            "train": {"batch_size": 4, "num_workers": 0},
        }

        train, validation, test, scaler = create_dataloaders(
            frames, labels, {"AA": 0, "BB": 1}, config,
        )

        train_samples = train.dataset.samples
        validation_samples = validation.dataset.samples
        test_samples = test.dataset.samples
        self.assertLess(train_samples[-1]["target_time"], validation_samples[0]["input_start_time"])
        self.assertLess(validation_samples[-1]["target_time"], test_samples[0]["input_start_time"])
        # Training windows see input frames 0..55; duplicated node values leave
        # the fitted first-feature mean at the average of those frame indices.
        self.assertAlmostEqual(float(scaler.mean_[0]), 27.5)
        self.assertIs(test.dataset.feature_scaler, scaler)


if __name__ == "__main__":
    unittest.main()
