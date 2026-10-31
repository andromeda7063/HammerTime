"""Offline integration contracts for isolated, multi-session cohort preparation."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import yaml
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import prepare_cohort
from graph import remap_cohort_driver_nodes


def make_graphs(race_id, drivers, times=20, num_nodes=20):
    local = {driver: index for index, driver in enumerate(drivers)}
    snapshots = []
    for step in range(times):
        x = torch.zeros((num_nodes, 9), dtype=torch.float32)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        names = [None] * num_nodes
        for driver, index in local.items():
            x[index, 0] = float(ord(driver[0]))
            mask[index] = True
            names[index] = driver
        snapshots.append(Data(
            x=x, edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.tensor([[0.5]], dtype=torch.float32), mask=mask,
            time_sec=step * 0.25, race_id=race_id, driver_names=names,
            is_safety_car=False,
        ))
    return snapshots, local


class PrepareCohortTests(unittest.TestCase):
    def test_preparation_freezes_multirace_artifact_and_global_driver_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "cohort.yaml"
            output_dir = root / "prepared"
            with (Path(__file__).resolve().parents[1] / "configs" / "default.yaml").open() as source:
                config = yaml.safe_load(source)
            specs = [
                {"year": 2023, "gp": "Monza", "race_id": "race-a"},
                {"year": 2024, "gp": "Monza", "race_id": "race-b"},
                {"year": 2025, "gp": "Monza", "race_id": "race-c"},
            ]
            config["sessions"] = specs
            config["data"]["split_method"] = "race"
            config["data"]["random_seed"] = 42
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            session_drivers = {
                "race-a": ["AA", "BB"],
                "race-b": ["BB", "CC"],
                "race-c": ["CC", "DD"],
            }
            raw_files = {}
            for race_id in session_drivers:
                raw = root / f"{race_id}.json"
                raw.write_text(json.dumps({"race_id": race_id}), encoding="utf-8")
                raw_files[race_id] = str(raw)

            def fake_load(spec, race_config):
                race_id = spec["race_id"]
                aligned = pd.DataFrame([
                    {"RaceId": race_id, "Driver": driver, "LapNumber": 1,
                     "TimeSec": 0.0, "IsValid": True}
                    for driver in session_drivers[race_id]
                ])
                return aligned, [raw_files[race_id]], {"provider": "fastf1"}

            def fake_targets(aligned, race_config):
                records = []
                for time_sec in [index * 0.25 for index in range(14, 20)]:
                    for driver in session_drivers[aligned["RaceId"].iloc[0]]:
                        records.append({"RaceId": aligned["RaceId"].iloc[0],
                                        "Driver": driver, "TimeSec": time_sec,
                                        "DeltaT_Sector": 0.1})
                return pd.DataFrame(records)

            def fake_snapshots(aligned, race_config):
                race_id = aligned["RaceId"].iloc[0]
                return make_graphs(race_id, session_drivers[race_id])

            with patch.object(prepare_cohort, "_load_configured_race", side_effect=fake_load), \
                 patch.object(prepare_cohort, "build_targets", side_effect=fake_targets), \
                 patch.object(prepare_cohort, "build_race_graph_snapshots", side_effect=fake_snapshots):
                manifest = prepare_cohort.prepare_cohort(config_path, output_dir)

            self.assertEqual(manifest["provider"], "fastf1")
            self.assertEqual(manifest["split"]["train"]["race_ids"], ["race-a"])
            self.assertEqual(manifest["split"]["validation"]["race_ids"], ["race-b"])
            self.assertEqual(manifest["split"]["test"]["race_ids"], ["race-c"])
            self.assertTrue(manifest["source_hashes"])

            payload, checked_manifest = prepare_cohort.load_prepared_cohort(
                output_dir / "prepared_cohort.pkl"
            )
            self.assertEqual(checked_manifest["prepared_manifest_sha256"],
                             manifest["prepared_manifest_sha256"])
            self.assertEqual(payload["driver_to_idx"], {"AA": 0, "BB": 1, "CC": 2, "DD": 3})
            by_race = {snapshot.race_id: snapshot for snapshot in payload["snapshots"]
                       if snapshot.time_sec == 0.0}
            self.assertEqual(by_race["race-b"].driver_names[:4], [None, "BB", "CC", None])
            self.assertEqual(by_race["race-b"].edge_index.tolist(), [[1], [2]])
            self.assertEqual(payload["provenance"]["prepared_artifact_sha256"],
                             manifest["prepared_artifact_sha256"])
            with self.assertRaises(FileExistsError):
                prepare_cohort.prepare_cohort(config_path, output_dir)

    def test_distinct_driver_identities_fit_explicit_cohort_capacity(self):
        first = [f"D{index:02d}" for index in range(20)]
        second = [f"D{index:02d}" for index in range(1, 21)]
        first_graphs, first_map = make_graphs("race-a", first, times=1, num_nodes=22)
        second_graphs, second_map = make_graphs("race-b", second, times=1, num_nodes=22)

        driver_to_idx = remap_cohort_driver_nodes(
            first_graphs + second_graphs,
            {"race-a": first_map, "race-b": second_map}, num_nodes=22,
        )

        self.assertEqual(len(driver_to_idx), 21)
        self.assertLess(max(driver_to_idx.values()), 22)
        self.assertEqual(len(set(driver_to_idx.values())), 21)
        self.assertEqual(len({driver_to_idx[driver] for driver in first}), 20)
        self.assertEqual(len({driver_to_idx[driver] for driver in second}), 20)
        self.assertEqual(len(first_graphs[0].driver_names), 22)
        self.assertEqual(len(second_graphs[0].driver_names), 22)


if __name__ == "__main__":
    unittest.main()
