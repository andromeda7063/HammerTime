"""Graph identity and active-node contracts on small in-memory rows."""

import sys
import unittest
from pathlib import Path

import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph import build_race_graph_snapshots


def row(driver, **changes):
    values = {"RaceId": "race-a", "Driver": driver, "TimeSec": 0.0, "Speed": 200.0,
              "Throttle": 50.0, "TyreLife": 3.0, "MiniSector": 1, "Compound": "SOFT",
              "IsValid": True, "IsPitLap": False, "DriverAhead": None, "DeltaT": 4.0}
    values.update(changes)
    return values


class GraphContractTests(unittest.TestCase):
    def setUp(self):
        self.config = {"graph": {"num_nodes": 2, "edge_threshold_sec": 2.0, "rbf_sigma": 1.0},
                       "model": {"in_dim": 9}}

    def test_inactive_leader_cannot_send_messages(self):
        data = pd.DataFrame([row("AA", IsPitLap=True), row("BB", DriverAhead="AA", DeltaT=0.5)])
        snapshots, _ = build_race_graph_snapshots(data, self.config)
        self.assertEqual(snapshots[0].edge_index.shape[1], 0)
        self.assertEqual(snapshots[0].mask.tolist(), [False, True])

    def test_too_many_drivers_raise_instead_of_silently_truncating(self):
        with self.assertRaisesRegex(ValueError, "graph.num_nodes"):
            build_race_graph_snapshots(pd.DataFrame([row("AA"), row("BB"), row("CC")]), self.config)

    def test_duplicate_driver_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate|Duplicate"):
            build_race_graph_snapshots(pd.DataFrame([row("AA"), row("AA")]), self.config)

    def test_group_iteration_preserves_snapshots_under_input_row_permutation(self):
        rows = [
            row("AA", TimeSec=0.0, DriverAhead=None),
            row("BB", TimeSec=0.0, DriverAhead="AA", DeltaT=0.5),
            row("AA", TimeSec=0.25, Speed=210.0, DriverAhead="BB", DeltaT=0.75),
            row("BB", TimeSec=0.25, Speed=190.0, DriverAhead="AA", DeltaT=0.4),
        ]
        ordered, _ = build_race_graph_snapshots(pd.DataFrame(rows), self.config)
        shuffled, _ = build_race_graph_snapshots(
            pd.DataFrame(rows).iloc[[3, 0, 2, 1]].reset_index(drop=True), self.config
        )
        self.assertEqual([snapshot.time_sec for snapshot in ordered], [0.0, 0.25])
        self.assertEqual(len(shuffled), len(ordered))
        for left, right in zip(ordered, shuffled):
            self.assertEqual(left.driver_names, right.driver_names)
            self.assertEqual(left.race_id, right.race_id)
            self.assertEqual(left.mask.tolist(), right.mask.tolist())
            torch.testing.assert_close(left.x, right.x)
            left_edges = sorted(zip(left.edge_index.T.tolist(), left.edge_attr.view(-1).tolist()))
            right_edges = sorted(zip(right.edge_index.T.tolist(), right.edge_attr.view(-1).tolist()))
            self.assertEqual(left_edges, right_edges)

    def test_graph_build_ignores_frame_metadata_without_mutating_it(self):
        data = pd.DataFrame([
            row("AA", TimeSec=0.0), row("BB", TimeSec=0.0, DriverAhead="AA", DeltaT=0.5),
        ])
        data.attrs["source_provenance"] = {"cache": ["hash"] * 10}
        expected, _ = build_race_graph_snapshots(data.copy(deep=True), self.config)
        actual, _ = build_race_graph_snapshots(data, self.config)
        self.assertEqual(data.attrs["source_provenance"], {"cache": ["hash"] * 10})
        self.assertEqual(actual[0].driver_names, expected[0].driver_names)
        torch.testing.assert_close(actual[0].x, expected[0].x)
        torch.testing.assert_close(actual[0].edge_index, expected[0].edge_index)


if __name__ == "__main__":
    unittest.main()
