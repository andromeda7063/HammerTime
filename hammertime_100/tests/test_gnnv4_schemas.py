"""Offline contracts for circuit_context_v3, physical_v3 and the opt-in aero head.

Fixtures are tiny synthetic tensors/frames used only to check code contracts;
they are not race data and produce no performance evidence.
"""

import copy
import math
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

from config_validation import validate_config  # noqa: E402
from context_features import (  # noqa: E402
    CONTEXT_V3_FEATURES, add_causal_race_references, add_race_progress_features,
    build_node_feature_vector, feature_names,
)
from model import HammerTimeSTGNN, create_model  # noqa: E402
from physical_graph import (  # noqa: E402
    PhysicalEdgeFeatureScaler, WAKE_WEIGHT_TAU_SEC, build_physical_circuit_edges,
)


def _two_lap_frame():
    """One race, one driver, two laps, two mini-sectors, 2 samples per sector."""
    rows = []
    time = 0.0
    for lap, speed in ((1, 100.0), (2, 200.0)):
        for sector in (0, 1):
            for _ in range(2):
                rows.append({"RaceId": "R1", "Driver": "A", "TimeSec": time, "LapNumber": lap,
                             "MiniSector": sector, "Speed": speed + sector, "IsValid": True,
                             "IsPitLap": False, "IsSafetyCar": False})
                time += 0.25
    return pd.DataFrame(rows)


class CausalReferenceTests(unittest.TestCase):
    def test_own_reference_uses_only_completed_earlier_passes(self):
        frame = add_causal_race_references(_two_lap_frame())
        lap1 = frame[frame["LapNumber"] == 1]
        lap2 = frame[frame["LapNumber"] == 2]
        # No earlier pass of a sector exists during lap 1.
        self.assertTrue((lap1["OwnReferenceAvailable"] == 0).all())
        self.assertTrue((lap1["SpeedRatioToOwnRaceReference"] == 1.0).all())
        # Lap 2 sector 0 compares 200 km/h with lap 1 sector 0 (100 km/h).
        sector0 = lap2[lap2["MiniSector"] == 0]
        self.assertTrue((sector0["OwnReferenceAvailable"] == 1).all())
        np.testing.assert_allclose(sector0["SpeedRatioToOwnRaceReference"], 2.0)
        np.testing.assert_allclose(sector0["SpeedDeltaToOwnRaceReference"], 100.0)

    def test_reference_never_reads_future_samples(self):
        base = add_causal_race_references(_two_lap_frame())
        changed = _two_lap_frame()
        changed.loc[changed["LapNumber"] == 2, "Speed"] = 999.0
        perturbed = add_causal_race_references(changed)
        lap1 = base["LapNumber"] == 1
        for column in ("SpeedRatioToOwnRaceReference", "SpeedRatioToFieldRaceReference",
                       "OwnReferenceAvailable", "FieldReferenceAvailable"):
            np.testing.assert_array_equal(base.loc[lap1, column], perturbed.loc[lap1, column])

    def test_pit_and_safety_car_rows_do_not_enter_reference(self):
        frame = _two_lap_frame()
        frame.loc[frame["LapNumber"] == 1, "IsSafetyCar"] = True
        result = add_causal_race_references(frame)
        self.assertTrue((result["OwnReferenceAvailable"] == 0).all())

    def test_race_progress_and_stint_lap(self):
        frame = pd.DataFrame({
            "RaceId": ["R"] * 4, "Driver": ["A"] * 4, "LapNumber": [1, 2, 3, 4],
            "IsPitLap": [False, True, False, False],
        })
        result = add_race_progress_features(frame)
        np.testing.assert_allclose(result["RaceProgress"], [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(result["StintLap"], [1.0, 0.0, 1.0, 2.0])


class NodeSchemaTests(unittest.TestCase):
    def test_v3_width_and_ratio_defaults(self):
        names = feature_names("circuit_context_v3")
        self.assertEqual(len(names), 9 + len(CONTEXT_V3_FEATURES))
        vector = build_node_feature_vector({"Speed": 200.0, "Compound": "SOFT"},
                                           feature_schema="circuit_context_v3")
        lookup = dict(zip(names, vector))
        self.assertEqual(lookup["SpeedRatioToOwnRaceReference"], 1.0)
        self.assertEqual(lookup["SpeedRatioToFieldRaceReference"], 1.0)
        self.assertEqual(lookup["RaceProgress"], 0.0)

    def test_v2_vector_is_unchanged_for_missing_columns(self):
        names = feature_names("circuit_context_v2")
        vector = build_node_feature_vector({"Speed": 200.0}, feature_schema="circuit_context_v2")
        # Historical v2 behavior: an absent ratio column yields 0.0, not 1.0.
        self.assertEqual(dict(zip(names, vector))["SpeedRatioToTrainReference"], 0.0)


class PhysicalV3Tests(unittest.TestCase):
    def _edges(self, schema):
        return build_physical_circuit_edges(
            distance_by_driver={"L": 120.0, "F": 100.0},
            speed_by_driver_kmh={"L": 180.0, "F": 180.0},
            position_valid_by_driver={"L": True, "F": True},
            node_active_by_driver={"L": True, "F": True},
            driver_to_idx={"F": 0, "L": 1}, lap_length_m=5000.0, edge_schema=schema)

    def test_v3_appends_continuous_wake_weight(self):
        v2, v3 = self._edges("physical_v2"), self._edges("physical_v3")
        self.assertEqual(v3.edge_attr.shape[1], 6)
        torch.testing.assert_close(v3.edge_attr[:, :5], v2.edge_attr)
        gap_sec = 20.0 / 50.0
        self.assertAlmostEqual(float(v3.edge_attr[0, 5]), math.exp(-gap_sec / WAKE_WEIGHT_TAU_SEC), places=5)

    def test_v3_scaler_round_trip_leaves_wake_unscaled(self):
        edges = self._edges("physical_v3")
        snapshot = Data(edge_attr_physical_circuit=edges.edge_attr)
        scaler = PhysicalEdgeFeatureScaler().fit([snapshot], [0])
        restored = PhysicalEdgeFeatureScaler.from_metadata(scaler.metadata())
        self.assertEqual(restored.schema_, "physical_v3")
        transformed = restored.transform(edges.edge_attr)
        torch.testing.assert_close(transformed[:, 5], edges.edge_attr[:, 5])


def _v3_model(aero=True):
    names = feature_names("circuit_context_v3")
    return HammerTimeSTGNN(in_dim=len(names), edge_dim=6, edge_feature_schema="physical_v3",
                           typed_edge_values=True, mask_inactive_nodes=True,
                           spatial_encoder="edge_temporal", spatial_layers=0,
                           temporal_mode="gated_dilated", aero_head=aero).eval()


def _snapshot(edge=True, nodes=4):
    names = feature_names("circuit_context_v3")
    generator = torch.Generator().manual_seed(3)
    raw = torch.tensor([[250.0, 0.02, 0.0], [250.0, 0.0, 1.0], [100.0, 0.05, 0.0], [200.0, 0.0, 0.0]])
    return Data(
        x=torch.randn(nodes, len(names), generator=generator),
        edge_index=torch.tensor([[0], [1]]) if edge else torch.empty((2, 0), dtype=torch.long),
        edge_attr=(torch.tensor([[0.8, 0.1, 0.1, 1.0, 0.0, 0.6]]) if edge
                   else torch.empty((0, 6))),
        mask=torch.ones(nodes, dtype=torch.bool), aero_raw=raw, position_coverage=1.0)


class AeroHeadTests(unittest.TestCase):
    def test_zero_without_incoming_edge(self):
        torch.manual_seed(0)
        with_head, without_head = _v3_model(True), _v3_model(False)
        without_head.load_state_dict(
            {k: v for k, v in with_head.state_dict().items() if not k.startswith("aero_head.")})
        window = [_snapshot(edge=False) for _ in range(10)]
        with torch.no_grad():
            torch.testing.assert_close(with_head(window), without_head(window))

    def test_batched_and_single_window_agree(self):
        model = _v3_model(True)
        window = [_snapshot(edge=True) for _ in range(10)]
        with torch.no_grad():
            torch.testing.assert_close(model([window, window])[0], model(window), atol=1e-5, rtol=1e-5)

    def test_follower_receives_corner_loss_and_coefficients_export(self):
        model = _v3_model(True)
        wake = torch.zeros(10, 4)
        wake[:, 1] = 0.6  # node 1 follows node 0 on a straight with DRS open
        raw = torch.stack([_snapshot().aero_raw for _ in range(10)])
        raw[:, 1, 1] = 0.05  # make it a corner instead
        term = model.aero_head(wake, raw, torch.ones(10, 4, dtype=torch.bool))
        self.assertGreater(float(term[1]), 0.0)
        self.assertEqual(float(term[0]), 0.0)
        coefficients = model.aero_coefficients()
        self.assertTrue(all(coefficients[key] > 0 for key in
                            ("corner_loss_a", "straight_gain_b", "drs_gain_d", "decay_k")))

    def test_missing_aero_raw_is_a_clear_error(self):
        model = _v3_model(True)
        snapshot = _snapshot()
        del snapshot.aero_raw
        with self.assertRaisesRegex(ValueError, "aero_raw"):
            model([snapshot] * 10)

    def test_aero_requires_physical_v3(self):
        with self.assertRaisesRegex(ValueError, "physical_v3"):
            HammerTimeSTGNN(in_dim=9, edge_dim=5, edge_feature_schema="physical_v2", aero_head=True)


class GNNv4ConfigTests(unittest.TestCase):
    def test_v4_config_validates_and_builds(self):
        import yaml
        path = Path(__file__).resolve().parents[1] / "configs" / "train_2023_fastf1_season_gnnv4.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_config(copy.deepcopy(config))
        model = create_model(config)
        self.assertIsNotNone(model.aero_head)
        self.assertEqual(model.in_dim, len(feature_names("circuit_context_v3")))


if __name__ == "__main__":
    unittest.main()
