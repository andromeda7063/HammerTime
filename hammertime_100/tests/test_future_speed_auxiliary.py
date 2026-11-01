"""Source-only contracts for opt-in future-speed auxiliary supervision."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_windowed_samples, collate_st_graph_windows, create_dataloaders
from train import masked_auxiliary_speed_loss


def snapshot(index, *, aux=True):
    values = {
        "x": torch.zeros((2, 9), dtype=torch.float32),
        "mask": torch.tensor([True, True]),
        "time_sec": index * 0.25,
        "race_id": "race-a",
        "driver_names": ("AA", "BB"),
        "is_safety_car": False,
    }
    if aux:
        values["aligned_speed_kph"] = np.asarray([100.0 + index, 200.0 + index], dtype=np.float32)
        values["aligned_speed_valid"] = np.asarray([True, index != 3], dtype=bool)
    return SimpleNamespace(**values)


class FutureSpeedAuxiliaryTests(unittest.TestCase):
    def test_future_speed_labels_use_only_frames_after_input_and_mask_missing_values(self):
        frames = [snapshot(i) for i in range(4)]
        # BB is absent from the final encoder frame, so it receives no aux loss
        # even when a future aligned observation is present.
        frames[1].mask[1] = False
        labels = pd.DataFrame([{
            "RaceId": "race-a", "Driver": "AA", "TimeSec": 0.75, "DeltaT_Sector": 0.2,
        }])

        samples = build_windowed_samples(
            frames, labels, {"AA": 0, "BB": 1}, W=2, K=2,
            timestamp_tolerance_sec=0.1275, continuity_tolerance_sec=0.05,
            future_speed_enabled=True,
        )

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual([snap.time_sec for snap in sample["snapshots"]], [0.0, 0.25])
        self.assertEqual(sample["aux_speed_target"].shape, (2, 2))
        self.assertEqual(sample["aux_speed_target"][0].tolist(), [102.0, 103.0])
        self.assertEqual(sample["aux_speed_mask"].tolist(), [[True, True], [False, False]])

    def test_enabled_auxiliary_labels_fail_closed_for_legacy_cache(self):
        frames = [snapshot(i, aux=False) for i in range(4)]
        labels = pd.DataFrame([{
            "RaceId": "race-a", "Driver": "AA", "TimeSec": 0.75, "DeltaT_Sector": 0.2,
        }])

        with self.assertRaisesRegex(ValueError, "rebuild old caches"):
            build_windowed_samples(
                frames, labels, {"AA": 0, "BB": 1}, W=2, K=2,
                timestamp_tolerance_sec=0.1275, continuity_tolerance_sec=0.05,
                future_speed_enabled=True,
            )

    def test_collation_preserves_legacy_and_auxiliary_batch_shapes(self):
        legacy = ([], torch.zeros(2), torch.ones(2, dtype=torch.bool))
        self.assertEqual(len(collate_st_graph_windows([legacy])), 3)
        auxiliary = (*legacy, torch.zeros(2, 5), torch.ones(2, 5, dtype=torch.bool))
        batch = collate_st_graph_windows([auxiliary, auxiliary])
        self.assertEqual(len(batch), 5)
        self.assertEqual(tuple(batch[3].shape), (2, 2, 5))

    def test_auxiliary_loss_normalizes_kph_and_skips_missing_observations(self):
        predictions = torch.zeros((1, 2, 1), requires_grad=True)
        targets = torch.tensor([[[100.0], [0.0]]])
        masks = torch.tensor([[[True], [False]]])
        loss = masked_auxiliary_speed_loss(predictions, targets, masks, scale_kph=100.0)
        self.assertAlmostEqual(float(loss), 0.5)

    def test_auxiliary_loss_excludes_nonfinite_values_and_returns_connected_zero(self):
        predictions = torch.tensor([[[float("nan")], [0.0]]], requires_grad=True)
        targets = torch.tensor([[[100.0], [float("nan")]]])
        masks = torch.tensor([[[False], [True]]])
        loss = masked_auxiliary_speed_loss(predictions, targets, masks, scale_kph=100.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())

    def test_auxiliary_loss_fails_on_nonfinite_prediction_for_valid_target(self):
        predictions = torch.tensor([[[float("inf")]]])
        targets = torch.tensor([[[100.0]]])
        masks = torch.tensor([[[True]]])
        with self.assertRaisesRegex(FloatingPointError, "valid auxiliary target"):
            masked_auxiliary_speed_loss(predictions, targets, masks, scale_kph=100.0)

    def test_typed_physical_edges_use_train_input_scaler_once_and_preserve_raw_attrs(self):
        frames = []
        raw_attrs = []
        for index in range(100):
            raw = torch.tensor([[0.8, float(index + 1), float(index * 2 - 20), 0.9, 1.0]])
            raw_attrs.append(raw.clone())
            frames.append(SimpleNamespace(
                x=torch.full((2, 3), float(index)),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=raw.clone(),
                edge_index_physical_circuit=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr_physical_circuit=raw.clone(),
                mask=torch.ones(2, dtype=torch.bool),
                time_sec=index * 0.25,
                race_id="race-a",
                driver_names=("AA", "BB"),
                is_safety_car=False,
            ))
        labels = pd.DataFrame([
            {"RaceId": "race-a", "Driver": driver, "TimeSec": index * 0.25,
             "DeltaT_Sector": float(index % 5)}
            for index in range(100) for driver in ("AA", "BB")
        ])
        config = {
            "data": {"window_size": 2, "horizon": 1, "train_ratio": 0.6,
                     "val_ratio": 0.2, "test_ratio": 0.2, "split_method": "temporal",
                     "timestamp_tolerance_sec": 0.1275, "continuity_tolerance_sec": 0.05,
                     "min_valid_targets_per_sample": 1, "random_seed": 42},
            "preprocess": {"sample_hz": 4.0, "flag_safety_car": True},
            "graph": {"edge_feature_schema": "physical_v2"},
            "model": {"typed_edge_values": True, "mask_inactive_nodes": True},
            "train": {"batch_size": 4, "num_workers": 0},
        }

        train, validation, test, _ = create_dataloaders(frames, labels, {"AA": 0, "BB": 1}, config)
        scaler = train.dataset.edge_feature_scaler
        train_times = {float(snap.time_sec) for sample in train.dataset.samples
                       for snap in sample["snapshots"]}
        train_raw_continuous = np.asarray([
            raw_attrs[index][0, [1, 2]].numpy()
            for index, snap in enumerate(frames) if float(snap.time_sec) in train_times
        ])
        np.testing.assert_allclose(scaler.mean_, train_raw_continuous.mean(axis=0))
        self.assertNotAlmostEqual(float(scaler.mean_[0]), float(np.mean([raw[0, 1] for raw in raw_attrs])))
        for index, frame in enumerate(frames):
            torch.testing.assert_close(frame.edge_attr_physical_circuit, raw_attrs[index])
            torch.testing.assert_close(frame.edge_attr, scaler.transform(raw_attrs[index]))
        self.assertIs(validation.dataset.edge_feature_scaler, scaler)
        self.assertIs(test.dataset.edge_feature_scaler, scaler)

        # Rebuilding loaders from the same graph list must start with raw cached
        # attrs again, rather than applying the transform to transformed values.
        train_again, _, _, _ = create_dataloaders(frames, labels, {"AA": 0, "BB": 1}, config)
        for index, frame in enumerate(frames):
            torch.testing.assert_close(frame.edge_attr, train_again.dataset.edge_feature_scaler.transform(raw_attrs[index]))
            torch.testing.assert_close(frame.edge_attr_physical_circuit, raw_attrs[index])

        legacy_frames = [SimpleNamespace(
            x=torch.full((2, 3), float(index)),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=raw.clone(),
            edge_index_physical_circuit=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr_physical_circuit=raw.clone(),
            mask=torch.ones(2, dtype=torch.bool),
            time_sec=index * 0.25,
            race_id="race-a",
            driver_names=("AA", "BB"),
            is_safety_car=False,
        ) for index, raw in enumerate(raw_attrs)]
        legacy_config = dict(config)
        legacy_config["model"] = {**config["model"], "typed_edge_values": False}
        legacy_train, _, _, _ = create_dataloaders(
            legacy_frames, labels, {"AA": 0, "BB": 1}, legacy_config
        )
        for index, frame in enumerate(legacy_frames):
            torch.testing.assert_close(frame.edge_attr, raw_attrs[index])


if __name__ == "__main__":
    unittest.main()
