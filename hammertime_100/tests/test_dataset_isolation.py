"""Offline contracts for race-held-out scaling and valid forecast windows."""

import unittest

import numpy as np
import pandas as pd
import torch

from test_dataset_contracts import snapshot
from dataset import build_windowed_samples, create_dataloaders


def labels_at(times, race_id="race-a", drivers=("AA", "BB")):
    return pd.DataFrame([{"RaceId": race_id, "Driver": driver, "TimeSec": time,
                          "DeltaT_Sector": 0.1}
                         for time in times for driver in drivers])


class DatasetIsolationTests(unittest.TestCase):
    def test_race_split_is_disjoint_and_extreme_holdout_features_do_not_fit_scaler(self):
        frames, labels = [], []
        for race, offset in (("train-race", 10.0), ("validation-race", 1e6), ("test-race", -1e6)):
            frames.extend(snapshot(i * 0.25, race_id=race, value=offset + i) for i in range(6))
            labels.append(labels_at([i * 0.25 for i in range(6)], race))
        config = {
            "data": {"window_size": 2, "horizon": 1, "train_ratio": 0.6,
                     "val_ratio": 0.2, "test_ratio": 0.2, "split_method": "race",
                     "timestamp_tolerance_sec": 0.1275, "continuity_tolerance_sec": 0.05,
                     "min_valid_targets_per_sample": 1, "random_seed": 42},
            "preprocess": {"sample_hz": 4.0, "flag_safety_car": True},
            "train": {"batch_size": 2, "num_workers": 0},
        }
        train, validation, test, scaler = create_dataloaders(
            frames, pd.concat(labels), {"AA": 0, "BB": 1}, config)
        race_sets = [{sample["race_id"] for sample in loader.dataset.samples}
                     for loader in (train, validation, test)]
        self.assertEqual(race_sets, [{"train-race"}, {"validation-race"}, {"test-race"}])
        self.assertTrue(race_sets[0].isdisjoint(race_sets[1] | race_sets[2]))
        self.assertTrue(race_sets[1].isdisjoint(race_sets[2]))
        # Four windows use the distinct training input frames with raw values 10..14.
        self.assertAlmostEqual(scaler.mean_[0], 12.0)
        self.assertAlmostEqual(scaler.var_[0], 2.0)
        expected = (1e6 - 12.0) / np.sqrt(2.0)
        self.assertAlmostEqual(float(validation.dataset.samples[0]["snapshots"][0].x[0, 0]), expected, delta=0.1)
        self.assertEqual(scaler.n_samples_seen_, 10)
        self.assertIs(validation.dataset.feature_scaler, scaler)
        self.assertIs(test.dataset.feature_scaler, scaler)
        diagnostic = test.dataset.split_metadata["outcome_boundary_diagnostics"]["train_to_validation"]
        self.assertEqual(diagnostic["status"], "unavailable")
        self.assertIn("separate sessions", diagnostic["reason"])
        self.assertIsNone(diagnostic["crossing_count"])

    def test_missing_frame_breaks_full_forecast_span(self):
        frames = [snapshot(time) for time in (0.0, 0.25, 0.75, 1.0, 1.25)]
        samples = build_windowed_samples(frames, labels_at([0.75, 1.25]), {}, W=2, K=1)
        self.assertEqual([sample["target_time"] for sample in samples], [1.25])

    def test_label_outside_configured_timestamp_tolerance_is_not_attached(self):
        frames = [snapshot(time) for time in (0.0, 0.25, 0.5)]
        samples = build_windowed_samples(frames, labels_at([0.61]), {}, W=2, K=1,
                                          timestamp_tolerance_sec=0.1)
        self.assertEqual(samples, [])

    def test_target_snapshot_mask_excludes_inactive_labelled_driver(self):
        frames = [snapshot(time) for time in (0.0, 0.25, 0.5)]
        frames[-1].mask = torch.tensor([False, True])
        samples = build_windowed_samples(frames, labels_at([0.5]), {}, W=2, K=1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["mask"].tolist(), [False, True])

    def test_all_inactive_target_nodes_remove_window(self):
        frames = [snapshot(time) for time in (0.0, 0.25, 0.5)]
        frames[-1].mask = torch.tensor([False, False])
        self.assertEqual(build_windowed_samples(frames, labels_at([0.5]), {}, W=2, K=1), [])


if __name__ == "__main__":
    unittest.main()
