"""Unit tests for HammerTime v2 enhancements.

Tests DRSActive, a_long, circuit_context_v2 (20-dim), physical_v2 (5-dim),
GraphNorm, residual skip connections, edge dropout, and Smooth L1 loss.
"""

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

from config_validation import validate_config
from context_features import (
    CONTEXT_V2_SCHEMA,
    build_node_feature_vector,
    feature_names,
    feature_schema_metadata,
    prepare_context_features,
    resolve_feature_schema,
)
from model import HammerTimeSTGNN, create_model
from physical_graph import (
    EDGE_FEATURES_BY_SCHEMA,
    PhysicalEdgeFeatureScaler,
    build_physical_circuit_edges,
)
from preprocess import align_driver_telemetry
from train import masked_smooth_l1_loss, masked_l1_loss, masked_mse_loss


class TestHammerTimeV2Features(unittest.TestCase):
    def test_preprocess_drs_and_a_long(self):
        # Generate synthetic driver telemetry with DRS and acceleration
        times = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        speeds = np.array([200.0, 210.0, 225.0, 240.0, 250.0])  # accelerating
        drs = np.array([0, 0, 12, 12, 14])  # DRS opened at t=0.5
        tel_df = pd.DataFrame({
            "TimeSec": times,
            "Speed": speeds,
            "Throttle": [100.0] * 5,
            "Brake": [0.0] * 5,
            "Distance": [0.0, 14.5, 30.0, 46.5, 64.0],
            "DRS": drs,
        })
        laps_df = pd.DataFrame()
        resampled = align_driver_telemetry("VER", tel_df, laps_df, sample_hz=4.0)

        self.assertIn("DRSActive", resampled.columns)
        self.assertIn("a_long", resampled.columns)
        # DRSActive should be 0.0 at t=0.0, 0.25 and 1.0 at t=0.5, 0.75, 1.0
        self.assertEqual(resampled["DRSActive"].iloc[0], 0.0)
        self.assertEqual(resampled["DRSActive"].iloc[1], 0.0)
        self.assertEqual(resampled["DRSActive"].iloc[2], 1.0)
        self.assertEqual(resampled["DRSActive"].iloc[3], 1.0)
        # a_long should be positive (accelerating from 200 to 250 km/h)
        self.assertTrue((resampled["a_long"] >= 0.0).all())
        self.assertTrue(np.isfinite(resampled["a_long"]).all())

    def test_circuit_context_v2_schema_contract(self):
        names = feature_names(CONTEXT_V2_SCHEMA)
        self.assertEqual(len(names), 20)
        self.assertIn("DRSActive", names)
        self.assertIn("a_long", names)
        self.assertIn("SpeedRatioToTrainReference", names)
        self.assertIn("lap_phase", names)
        self.assertNotIn("lap_length_m", names)  # Track identity removed

        meta = feature_schema_metadata(CONTEXT_V2_SCHEMA)
        self.assertEqual(meta["id"], CONTEXT_V2_SCHEMA)
        self.assertEqual(len(meta["feature_names"]), 20)
        self.assertEqual(len(meta["raw_units"]), 20)

    def test_build_node_feature_vector_v2(self):
        row = {
            "Speed": 315.0,
            "Throttle": 100.0,
            "Compound": "SOFT",
            "TyreLife": 5.0,
            "MiniSector": 2,
            "SpeedRatioToTrainReference": 1.05,
            "SpeedDeltaToTrainReference": 15.0,
            "Brake": 0.0,
            "DRSActive": 1.0,
            "a_long": 2.5,
            "lap_sin": 0.587,
            "lap_cos": 0.809,
            "lap_phase": 0.125,
            "abs_curvature": 0.001,
            "lateral_accel_proxy": 1.2,
            "ReferenceAvailable": 1.0,
        }
        vec = build_node_feature_vector(row, feature_schema=CONTEXT_V2_SCHEMA)
        self.assertEqual(len(vec), 20)
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(np.isfinite(vec).all())
        # Check DRSActive is 1.0
        names = feature_names(CONTEXT_V2_SCHEMA)
        drs_idx = names.index("DRSActive")
        self.assertEqual(vec[drs_idx], 1.0)

    def test_physical_v2_edge_contract(self):
        self.assertIn("physical_v2", EDGE_FEATURES_BY_SCHEMA)
        self.assertEqual(len(EDGE_FEATURES_BY_SCHEMA["physical_v2"]), 5)
        self.assertIn("in_wake_corridor", EDGE_FEATURES_BY_SCHEMA["physical_v2"])

        # Build close following edges (gap = 20 m < 35 m -> in_wake_corridor = 1)
        res = build_physical_circuit_edges(
            distance_by_driver={"LEAD": 990.0, "CHASE": 10.0},
            speed_by_driver_kmh={"LEAD": 300.0, "CHASE": 310.0},
            position_valid_by_driver={"LEAD": True, "CHASE": True},
            node_active_by_driver={"LEAD": True, "CHASE": True},
            driver_to_idx={"LEAD": 0, "CHASE": 1},
            lap_length_m=1000.0,
            max_gap_sec=2.0,
            edge_schema="physical_v2",
        )
        self.assertEqual(res.edge_attr.shape[1], 5)
        # Gap is 20m, speed is ~86 m/s -> gap_sec = 20/86 ~ 0.23s <= 0.6s -> in_wake_corridor = 1.0
        self.assertEqual(res.edge_attr[0, 4].item(), 1.0)

    def test_physical_v2_edge_scaler(self):
        scaler = PhysicalEdgeFeatureScaler()
        snap = Data(
            x=torch.randn(2, 20),
            edge_index_physical_circuit=torch.tensor([[0], [1]]),
            edge_attr_physical_circuit=torch.tensor([[0.95, 20.0, -2.5, 1.0, 1.0]]),
        )
        scaler.fit([snap], [0])
        self.assertTrue(scaler.fitted_)
        self.assertEqual(scaler.schema_, "physical_v2")
        meta = scaler.metadata()
        self.assertEqual(meta["schema"], "physical_v2")

        transformed = scaler.transform(snap.edge_attr_physical_circuit)
        self.assertEqual(transformed.shape, (1, 5))
        # RBF (col 0), coverage (col 3), and in_wake_corridor (col 4) must remain unscaled
        self.assertAlmostEqual(transformed[0, 0].item(), 0.95, places=5)
        self.assertAlmostEqual(transformed[0, 3].item(), 1.0, places=5)
        self.assertAlmostEqual(transformed[0, 4].item(), 1.0, places=5)

    def test_model_v2_graphnorm_and_residual(self):
        model = HammerTimeSTGNN(
            in_dim=20,
            hidden_dim=32,
            gat_heads=4,
            edge_dim=5,
            edge_feature_schema="physical_v2",
            gat_type="gatv2",
            use_graph_norm=True,
            drop_edge_rate=0.15,
            residual=True,
        )
        model.eval()
        dummy_window = [
            Data(
                x=torch.randn(4, 20),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                edge_attr=torch.randn(3, 5),
                mask=torch.ones(4, dtype=torch.bool),
            )
            for _ in range(10)
        ]
        # Single window forward pass
        out = model(dummy_window)
        self.assertEqual(out.shape, torch.Size([4]))
        self.assertTrue(torch.isfinite(out).all())

        # Batched window forward pass
        batch = [dummy_window, dummy_window]
        batch_out = model(batch)
        self.assertEqual(batch_out.shape, torch.Size([2, 4]))
        self.assertTrue(torch.isfinite(batch_out).all())

        # Test training mode with drop_edge_rate active
        model.train()
        train_out = model(batch)
        self.assertEqual(train_out.shape, torch.Size([2, 4]))
        self.assertTrue(torch.isfinite(train_out).all())

    def test_smooth_l1_loss(self):
        preds = torch.tensor([0.05, -0.10, 0.20])
        targets = torch.tensor([0.04, -0.12, 0.25])
        masks = torch.tensor([True, True, True])

        loss = masked_smooth_l1_loss(preds, targets, masks, beta=0.02)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_config_validation_v2(self):
        import yaml
        config_path = Path(__file__).resolve().parents[1] / "configs/train_2023_fastf1_season_physical_v2.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Should validate without any error
        validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
