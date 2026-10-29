"""Runner failure and artifact provenance contracts without FastF1 access."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import run_pipeline as pipeline


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        with (PROJECT_DIR / "configs" / "default.yaml").open(encoding="utf-8") as source:
            self.config = yaml.safe_load(source)

    def test_missing_positions_do_not_generate_a_track_map(self):
        module_names = (
            "v1_track_map", "v2_pred_vs_actual", "v3_delay_heatmap", "v4_drs_timeline",
            "v5_baseline_duel", "v6_physics_cards", "v7_blockers", "v8_residuals",
            "v9_clean_air_cal", "cards",
        )
        modules = {name: Mock() for name in module_names}
        modules["cards"].build_broadcast_cards.return_value = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "visuals").mkdir()
            legacy = root / "visuals" / "v1_track_map.png"
            legacy.write_bytes(b"historical artifact")
            config = copy.deepcopy(self.config)
            config["session"]["processed_dir"] = str(root / "processed")
            with patch.dict(sys.modules, modules), patch.object(pipeline, "PROJECT_DIR", root), \
                    patch.object(pipeline, "BROADCAST_DIR", root / "broadcast"):
                artifacts = pipeline._render_real_outputs(pd.DataFrame(), [], pd.DataFrame(), {}, config)
            modules["v1_track_map"].generate_track_map.assert_not_called()
            self.assertFalse(any(Path(path).name == "v1_track_map.png" for path in artifacts))
            self.assertEqual(legacy.read_bytes(), b"historical artifact")

    def test_observed_positions_use_the_observed_route(self):
        module_names = ("v1_track_map", "v2_pred_vs_actual", "v3_delay_heatmap", "v4_drs_timeline",
                        "v5_baseline_duel", "v6_physics_cards", "v7_blockers", "v8_residuals",
                        "v9_clean_air_cal", "cards")
        modules = {name: Mock() for name in module_names}
        modules["cards"].build_broadcast_cards.return_value = []
        edges, weights = Mock(), Mock()
        edges.detach.return_value.cpu.return_value.numpy.return_value = np.empty((2, 0), dtype=int)
        weights.detach.return_value.cpu.return_value.numpy.return_value = np.empty((0, 1))
        snapshot = SimpleNamespace(race_id="observed", time_sec=6000.0, driver_names=("AA",),
                                   edge_index=edges, edge_attr=weights)
        rows = pd.DataFrame({"RaceId": ["observed"] * 2, "Driver": ["AA"] * 2,
                             "TimeSec": [6000.0, 6000.05], "LapNumber": [1, 1],
                             "X": [11.0, 12.0], "Y": [21.0, 22.0], "DeltaT": [4.0, 4.0]})
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config["session"]["processed_dir"] = directory
            with patch.dict(sys.modules, modules):
                paths = pipeline._render_real_outputs(rows, [snapshot], pd.DataFrame(), {}, config)
            call = modules["v1_track_map"].generate_track_map.call_args
            self.assertEqual(call.args[0], {"AA": (11.0, 21.0)})
            self.assertEqual(call.kwargs["circuit_name"], "observed")
            np.testing.assert_array_equal(call.kwargs["track_xy"], [[11.0, 21.0], [12.0, 22.0]])
            self.assertEqual(len(paths), 1)



if __name__ == "__main__":
    unittest.main()
