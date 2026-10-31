"""Focused unit coverage for the opt-in causal node context schema."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from context_features import (
    CONTEXT_SCHEMA,
    LEGACY_SCHEMA,
    apply_context_calibration,
    build_node_feature_vector,
    feature_names,
    feature_schema_metadata,
    fit_context_calibration,
    node_feature_schema,
    prepare_context_features,
)


class ContextFeatureTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({
            "RaceId": ["R1"] * 4,
            "Driver": ["1"] * 4,
            "TimeSec": [0.0, 0.25, 0.5, 0.75],
            "MiniSector": [0, 0, 1, 1],
            "Speed": [100.0, 110.0, 150.0, 160.0],
            "Throttle": [90.0] * 4,
            "Brake": [0.0, 1.0, np.nan, 0.0],
            "TyreLife": [3.0] * 4,
            "Compound": ["SOFT"] * 4,
            "X": [0.0, 1.0, 2.0, 3.0],
            "Y": [0.0, 0.0, 1.0, 2.0],
            "IsValid": [True] * 4,
            "IsPitLap": [False] * 4,
        })
        self.session_map = {"R1": {"circuit_id": "CircuitA", "lap_length_m": 5793.0}}

    def test_legacy_schema_and_old_call_shape_remain_nine_columns(self):
        self.assertEqual(node_feature_schema({"graph": {}}), LEGACY_SCHEMA)
        self.assertEqual(len(feature_names(LEGACY_SCHEMA)), 9)
        vector = build_node_feature_vector(self.rows.iloc[0], num_compounds=5)
        self.assertEqual(vector.shape, (9,))
        self.assertEqual(vector.dtype, np.float32)

    def test_context_schema_has_stable_order_and_declared_units(self):
        names = feature_names(CONTEXT_SCHEMA)
        self.assertEqual(len(names), 19)
        self.assertEqual(names[9:], (
            "Brake", "BrakeSupported", "lap_sin", "lap_cos", "lap_length_m",
            "abs_curvature", "lateral_accel_proxy", "GeometrySupported",
            "SpeedDeltaToTrainReference", "ReferenceAvailable",
        ))
        metadata = feature_schema_metadata(CONTEXT_SCHEMA)
        self.assertEqual(metadata["feature_names"], list(names))
        self.assertEqual(metadata["geometry_lookback_frames"], 2)
        self.assertEqual(len(metadata["raw_units"]), 19)

    def test_context_preparation_is_causal_and_preserves_rows(self):
        self.rows.attrs["provenance"] = {"marker": "keep caller attrs"}
        prepared = prepare_context_features(self.rows, self.session_map)
        self.assertEqual(self.rows.attrs["provenance"], {"marker": "keep caller attrs"})
        self.assertEqual(len(prepared), len(self.rows))
        self.assertEqual(prepared["TimeSec"].tolist(), self.rows["TimeSec"].tolist())
        self.assertEqual(prepared["GeometrySupported"].tolist(), [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(prepared.loc[2, "BrakeSupported"], 0.0)
        self.assertEqual(prepared.loc[2, "Brake"], 0.0)
        extended = pd.concat([self.rows, pd.DataFrame([{
            **self.rows.iloc[-1].to_dict(), "TimeSec": 1.0, "X": 1000.0, "Y": -500.0
        }])], ignore_index=True)
        extended_prepared = prepare_context_features(extended, self.session_map)
        np.testing.assert_allclose(
            prepared.loc[:2, ["abs_curvature", "lateral_accel_proxy"]].to_numpy(),
            extended_prepared.loc[:2, ["abs_curvature", "lateral_accel_proxy"]].to_numpy(),
        )

    def test_calibration_uses_unique_training_frames_and_marks_unseen_groups(self):
        prepared = prepare_context_features(self.rows, self.session_map)
        calibration = fit_context_calibration(
            prepared,
            [("R1", 0.0), ("R1", 0.25), ("R1", 0.25)],
        )
        self.assertEqual(calibration["training_frame_count"], 2)
        self.assertEqual(calibration["training_row_count"], 2)
        self.assertEqual(calibration["references"][0]["median_speed_kph"], 105.0)
        prepared.attrs["provenance"] = {"marker": "keep caller attrs"}
        applied = apply_context_calibration(prepared, calibration)
        self.assertEqual(prepared.attrs["provenance"], {"marker": "keep caller attrs"})
        self.assertEqual(applied.loc[2, "SpeedDeltaToTrainReference"], 0.0)
        self.assertEqual(applied.loc[2, "ReferenceAvailable"], 0.0)
        refit_unseen = prepared.copy()
        refit_unseen["CircuitId"] = "CircuitB"
        unseen = apply_context_calibration(refit_unseen, calibration)
        self.assertTrue((unseen["SpeedDeltaToTrainReference"] == 0.0).all())
        self.assertTrue((unseen["ReferenceAvailable"] == 0.0).all())

    def test_context_vector_has_nineteen_values_and_missing_geometry_is_zero_flagged(self):
        prepared = prepare_context_features(self.rows, self.session_map)
        calibration = fit_context_calibration(prepared, [("R1", 0.0), ("R1", 0.25)])
        applied = apply_context_calibration(prepared, calibration)
        vector = build_node_feature_vector(applied.iloc[0], feature_schema=CONTEXT_SCHEMA)
        self.assertEqual(vector.shape, (19,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(vector[14], 0.0)  # curvature has no two-frame history yet
        self.assertEqual(vector[15], 0.0)
        self.assertEqual(vector[16], 0.0)  # GeometrySupported
        self.assertEqual(vector[17], -5.0)  # speed minus the fitted sector reference
        self.assertEqual(vector[18], 1.0)  # sector 0 has a train-only reference


if __name__ == "__main__":
    unittest.main()
