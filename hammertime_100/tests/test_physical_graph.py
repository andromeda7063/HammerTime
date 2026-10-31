"""Focused offline coverage for the opt-in physical graph representation."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physical_graph import (
    PhysicalEdgeFeatureScaler,
    build_physical_circuit_edges,
    is_position_valid,
)
from graph import remap_cohort_driver_nodes
from model import HammerTimeSTGNN


class PhysicalGraphTests(unittest.TestCase):
    def test_position_support_is_independent_of_general_validity_and_lap_counter(self):
        self.assertTrue(is_position_valid(123.0, 0.0, None, is_pit=False))
        self.assertFalse(is_position_valid(None, 100.0, 3, is_pit=False))
        self.assertFalse(is_position_valid(123.0, 100.0, 3, is_pit=True))

    def _build(self, distances, speeds=None, active=None, valid=None, lap=1000.0,
               expected=3):
        drivers = list(distances)
        speeds = speeds or {driver: 90.0 for driver in drivers}
        active = active or {driver: True for driver in drivers}
        valid = valid or {driver: True for driver in drivers}
        return build_physical_circuit_edges(
            distance_by_driver=distances,
            speed_by_driver_kmh=speeds,
            position_valid_by_driver=valid,
            node_active_by_driver=active,
            driver_to_idx={driver: index for index, driver in enumerate(drivers)},
            lap_length_m=lap,
            expected_on_track_count=expected,
        )

    def test_circular_distance_orders_physical_neighbor_across_start_finish(self):
        result = self._build({"near_start": 10.0, "near_finish": 990.0})
        edges = {tuple(edge) for edge in result.edge_index.T.tolist()}
        # Drivers are indexed in insertion order. The 990 m car leads the 10 m car.
        self.assertIn((1, 0), edges)
        self.assertEqual(result.edge_attr.shape, (1, 4))
        self.assertAlmostEqual(float(result.edge_attr[0, 1]), 20.0)

    def test_physical_order_ignores_lap_progress_and_selects_nearest_supported_car(self):
        # `lapped` would have lower cumulative race progress, but lies 50 m ahead.
        result = self._build({"follower": 100.0, "lapped": 150.0, "farther": 500.0})
        edges = {tuple(edge) for edge in result.edge_index.T.tolist()}
        self.assertIn((1, 0), edges)
        self.assertNotIn((2, 0), edges)

    def test_cutoff_is_inclusive_and_inactive_nearest_car_is_not_bridged(self):
        exact = self._build({"follower": 100.0, "leader": 150.0}, expected=2)
        self.assertEqual(exact.edge_index.size(1), 1)
        self.assertAlmostEqual(float(exact.edge_attr[0, 1]), 50.0)
        blocked = self._build(
            {"follower": 100.0, "inactive": 150.0, "farther": 200.0},
            active={"follower": True, "inactive": False, "farther": True},
        )
        self.assertEqual(blocked.blocked_neighbor_count, 2)
        self.assertEqual(blocked.edge_index.size(1), 0)

    def test_missing_position_reduces_coverage_without_changing_sample(self):
        result = self._build(
            {"known_a": 100.0, "known_b": 150.0},
            valid={"known_a": True, "known_b": True, "missing": False},
            expected=3,
        )
        self.assertAlmostEqual(result.position_coverage, 2 / 3)
        self.assertEqual(result.edge_index.size(1), 2)
        self.assertTrue(torch.isfinite(result.edge_attr).all())

    def test_scaler_uses_only_requested_training_snapshot_edges_and_preserves_raw(self):
        train = SimpleNamespace(edge_attr_physical_circuit=torch.tensor(
            [[0.8, 10.0, 2.0, 0.9], [0.7, 20.0, 4.0, 0.9]]))
        held_out = SimpleNamespace(edge_attr_physical_circuit=torch.tensor(
            [[0.6, 1000.0, 100.0, 0.1]]))
        scaler = PhysicalEdgeFeatureScaler().fit([train, held_out], [0])
        raw = train.edge_attr_physical_circuit.clone()
        transformed = scaler.transform(raw)
        self.assertTrue(torch.equal(train.edge_attr_physical_circuit, raw))
        self.assertTrue(torch.allclose(transformed[:, 1:3].mean(dim=0), torch.zeros(2)))
        self.assertTrue(torch.equal(transformed[:, [0, 3]], raw[:, [0, 3]]))
        self.assertEqual(scaler.metadata()["n_samples_seen_"], 2)
        restored = PhysicalEdgeFeatureScaler.from_metadata(scaler.metadata())
        self.assertTrue(np.allclose(restored.mean_, scaler.mean_))

    def test_cohort_remap_keeps_both_edge_views_and_position_support_aligned(self):
        snapshot = Data(
            x=torch.tensor([[10.0], [20.0]]),
            mask=torch.tensor([True, False]),
            position_valid=torch.tensor([True, False]),
            driver_names=["B", "A"],
            edge_index=torch.tensor([[0], [1]]),
            edge_index_race_progress=torch.tensor([[0], [1]]),
            edge_index_physical_circuit=torch.tensor([[1], [0]]),
            edge_attr_race_progress=torch.ones((1, 1)),
            edge_attr_physical_circuit=torch.ones((1, 4)),
            race_id="R1",
        )
        remap_cohort_driver_nodes([snapshot], {"R1": {"B": 0, "A": 1}}, 2)
        self.assertEqual(snapshot.driver_names, ["A", "B"])
        self.assertEqual(snapshot.edge_index_race_progress.tolist(), [[1], [0]])
        self.assertEqual(snapshot.edge_index_physical_circuit.tolist(), [[0], [1]])
        self.assertEqual(snapshot.position_valid.tolist(), [False, True])
        self.assertEqual(snapshot.edge_attr_physical_circuit.shape, (1, 4))

    def test_model_keeps_legacy_edge_checkpoint_contract_and_physical_self_loops(self):
        legacy = HammerTimeSTGNN(in_dim=9, hidden_dim=8, gat_heads=2, edge_dim=1)
        reloaded = HammerTimeSTGNN(in_dim=9, hidden_dim=8, gat_heads=2, edge_dim=1)
        reloaded.load_state_dict(legacy.state_dict(), strict=True)
        _, legacy_loop_attr = legacy._ensure_self_loops(
            2, torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 1)), torch.device("cpu"),
        )
        self.assertEqual(legacy_loop_attr.tolist(), [[1.0], [1.0]])

        physical = HammerTimeSTGNN(
            in_dim=9, hidden_dim=8, gat_heads=2, edge_dim=4,
            edge_feature_schema="physical_v1",
        )
        _, loop_attr = physical._ensure_self_loops(
            2, torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 4)), torch.device("cpu"), position_coverage=0.75,
        )
        self.assertEqual(loop_attr.tolist(), [[1.0, 0.0, 0.0, 0.75]] * 2)


if __name__ == "__main__":
    unittest.main()
