"""Graph identity and active-node contracts on small in-memory rows."""

import sys
import unittest
from pathlib import Path

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
