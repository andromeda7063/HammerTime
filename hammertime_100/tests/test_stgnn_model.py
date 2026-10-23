"""Unit tests for HammerTime ST-GNN architecture, message passing, and zero-edge invariance."""

import sys
import unittest
from pathlib import Path
import torch
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import HammerTimeSTGNN, create_model
from graph import build_node_feature_vector, build_snapshot_edges


class STGNNModelTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": {
                "in_dim": 9,
                "hidden_dim": 64,
                "gat_heads": 4,
                "edge_dim": 1,
                "conv1d_kernel": 3,
                "dropout": 0.1
            },
            "data": {"random_seed": 42}
        }
        self.model = create_model(self.config)

    def test_model_initialization_dimensions(self):
        """Test model parameter dimensions and divisibility."""
        self.assertEqual(self.model.in_dim, 9)
        self.assertEqual(self.model.hidden_dim, 64)
        self.assertEqual(self.model.gat_heads, 4)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 10000)

    def test_forward_pass_full_graph(self):
        """Test forward pass with 20 cars and active edges."""
        snaps = [
            Data(
                x=torch.randn(20, 9),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_attr=torch.tensor([[0.85], [0.72]], dtype=torch.float32),
                mask=torch.ones(20, dtype=torch.bool)
            )
            for _ in range(10)
        ]
        out = self.model(snaps)
        self.assertEqual(out.shape, torch.Size([20]))
        self.assertTrue(torch.isfinite(out).all())

    def test_zero_edge_forward_pass_stability(self):
        """Immutable Law 9: Forward pass on empty/zero-edge snapshots must not fail or produce NaN."""
        snaps = [
            Data(
                x=torch.randn(20, 9),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1), dtype=torch.float32),
                mask=torch.ones(20, dtype=torch.bool)
            )
            for _ in range(10)
        ]
        out = self.model(snaps)
        self.assertEqual(out.shape, torch.Size([20]))
        self.assertTrue(torch.isfinite(out).all())

    def test_batched_window_forward(self):
        """Verify model handles batch of windows [B, W, N]."""
        batch_windows = [
            [
                Data(
                    x=torch.randn(20, 9),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 1), dtype=torch.float32),
                )
                for _ in range(10)
            ]
            for _ in range(3)
        ]
        out = self.model(batch_windows)
        self.assertEqual(out.shape, torch.Size([3, 20]))
        self.assertTrue(torch.isfinite(out).all())

    def test_gradient_flow_and_finite_gradients(self):
        """Verify backpropagation produces finite gradients across all weights."""
        snaps = [
            Data(
                x=torch.randn(20, 9),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=torch.tensor([[0.9]], dtype=torch.float32),
            )
            for _ in range(10)
        ]
        out = self.model(snaps)
        loss = out.sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"NaN grad in {name}")

    def test_ensure_self_loops_structure(self):
        """Verify self-loops add self-edges with weight 1.0."""
        num_nodes = 5
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_attr = torch.tensor([[0.8], [0.6]], dtype=torch.float32)
        idx, attr = self.model._ensure_self_loops(num_nodes, edge_index, edge_attr, torch.device("cpu"))
        self.assertEqual(idx.size(1), 2 + num_nodes)
        self.assertEqual(attr.size(0), 2 + num_nodes)
        self.assertTrue((attr[-num_nodes:] == 1.0).all())

    def test_rbf_gaussian_weight_decay(self):
        """Verify RBF weight decreases monotonically with increasing time gap."""
        ahead = {"LEC": "VER"}
        gap1 = {"LEC": 0.4}
        gap2 = {"LEC": 1.2}
        driver_to_idx = {"VER": 0, "LEC": 1}
        idx1, attr1 = build_snapshot_edges(ahead, gap1, driver_to_idx, max_gap=2.0, sigma=1.0)
        idx2, attr2 = build_snapshot_edges(ahead, gap2, driver_to_idx, max_gap=2.0, sigma=1.0)
        self.assertGreater(float(attr1[0]), float(attr2[0]))

    def test_edge_cutoff_threshold(self):
        """Verify no edge is formed when gap exceeds edge_threshold_sec (2.0s)."""
        ahead = {"LEC": "VER"}
        gap = {"LEC": 2.5}
        driver_to_idx = {"VER": 0, "LEC": 1}
        idx, attr = build_snapshot_edges(ahead, gap, driver_to_idx, max_gap=2.0, sigma=1.0)
        self.assertEqual(idx.size(1), 0)
        self.assertEqual(attr.size(0), 0)


if __name__ == "__main__":
    unittest.main()
