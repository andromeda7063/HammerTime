"""Offline contract tests for matched circuit-context comparison views."""

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ablation import CONTEXT_GRAPH_CANDIDATES, _clone_transformed_samples, _target_mask_hash


class IdentityEdgeScaler:
    def transform(self, attrs):
        return attrs.clone()


class ContextGraphViewTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = Data(
            x=torch.arange(38, dtype=torch.float32).reshape(2, 19),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.tensor([[0.4]], dtype=torch.float32),
            edge_index_race_progress=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr_race_progress=torch.tensor([[0.4]], dtype=torch.float32),
            edge_index_physical_circuit=torch.tensor([[1], [0]], dtype=torch.long),
            edge_attr_physical_circuit=torch.tensor([[0.7, 10.0, -2.0, 0.8]], dtype=torch.float32),
            position_valid=torch.tensor([True, True]),
            mask=torch.tensor([True, True]),
        )
        self.samples = [{
            "snapshots": [self.snapshot],
            "target": torch.tensor([1.0, -1.0]),
            "mask": torch.tensor([True, False]),
        }]

    def test_five_declared_views_preserve_targets_and_masks(self):
        self.assertEqual(len(CONTEXT_GRAPH_CANDIDATES), 5)
        before = _target_mask_hash(self.samples)
        outputs = []
        for candidate in CONTEXT_GRAPH_CANDIDATES:
            transformed = _clone_transformed_samples(
                self.samples, candidate, edge_scaler=IdentityEdgeScaler()
            )
            self.assertEqual(_target_mask_hash(transformed), before)
            outputs.append(transformed[0]["snapshots"][0])

        self.assertEqual(tuple(outputs[0].x.shape), (2, 9))
        self.assertEqual(tuple(outputs[1].x.shape), (2, 19))
        self.assertEqual(outputs[0].edge_index.tolist(), [[0], [1]])
        self.assertEqual(outputs[1].edge_attr.tolist(), [[0.4]])
        self.assertEqual(outputs[2].edge_index.tolist(), [[1], [0]])
        self.assertEqual(outputs[2].edge_attr.tolist(), [[0.7]])
        self.assertEqual(outputs[3].edge_attr.tolist(), [[0.7, 10.0, -2.0, 0.8]])
        self.assertEqual(tuple(outputs[4].edge_index.shape), (2, 0))


if __name__ == "__main__":
    unittest.main()
