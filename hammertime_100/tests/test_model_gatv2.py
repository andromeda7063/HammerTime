"""Unit tests for GATv2 dynamic attention architecture and 4D physical edge conditioning."""

import sys
import unittest
from pathlib import Path
import torch
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import HammerTimeSTGNN, create_model


class GATv2ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": {
                "name": "HammerTime_STGNN",
                "gat_type": "gatv2",
                "in_dim": 19,
                "hidden_dim": 64,
                "gat_heads": 4,
                "edge_dim": 4,
                "conv1d_kernel": 3,
                "dropout": 0.1,
                "temporal_mode": "gated_dilated",
                "temporal_dilations": [1, 2, 4],
            },
            "graph": {
                "node_feature_schema": "circuit_context_v1",
                "edge_feature_schema": "physical_v1",
                "neighbor_mode": "physical_circuit",
            },
            "data": {"random_seed": 42}
        }
        self.model = create_model(self.config)

    def test_gatv2_initialization(self):
        """Verify model initializes with GATv2Conv layers and correct dimensions."""
        self.assertEqual(self.model.gat_type, "gatv2")
        self.assertEqual(self.model.in_dim, 19)
        self.assertEqual(self.model.edge_dim, 4)
        self.assertEqual(self.model.hidden_dim, 64)
        self.assertEqual(self.model.gat_heads, 4)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 10000)

    def test_gatv2_forward_pass_4d_edges(self):
        """Test forward pass with 20 cars, 19-dim node features, and 4-dim physical edge features."""
        snaps = [
            Data(
                x=torch.randn(20, 19),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
                edge_attr=torch.randn(3, 4),
                mask=torch.ones(20, dtype=torch.bool),
            )
            for _ in range(10)
        ]
        out = self.model(snaps)
        self.assertEqual(out.shape, torch.Size([20]))
        self.assertTrue(torch.isfinite(out).all())

    def test_gatv2_zero_edge_stability(self):
        """Verify zero-edge snapshots do not crash or produce NaNs with 4-dim edge_attr."""
        snaps = [
            Data(
                x=torch.randn(20, 19),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 4), dtype=torch.float32),
                mask=torch.ones(20, dtype=torch.bool),
            )
            for _ in range(10)
        ]
        out = self.model(snaps)
        self.assertEqual(out.shape, torch.Size([20]))
        self.assertTrue(torch.isfinite(out).all())

    def test_gatv2_batched_windows(self):
        """Verify batched windows evaluate correctly and produce shape [B, N]."""
        batch_windows = [
            [
                Data(
                    x=torch.randn(20, 19),
                    edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                    edge_attr=torch.randn(2, 4),
                    mask=torch.arange(20) % 2 == 0,
                )
                for _ in range(10)
            ]
            for _ in range(4)
        ]
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch_windows)
        self.assertEqual(out.shape, torch.Size([4, 20]))
        self.assertTrue(torch.isfinite(out).all())

    def test_gatv2_backward_gradient_flow(self):
        """Verify backpropagation produces finite gradients through GATv2Conv and 4D edge inputs."""
        self.model.train()
        snaps = [
            Data(
                x=torch.randn(10, 19, requires_grad=True),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_attr=torch.randn(2, 4),
                mask=torch.ones(10, dtype=torch.bool),
            )
            for _ in range(10)
        ]
        out = self.model(snaps)
        loss = out.sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}")


if __name__ == "__main__":
    unittest.main()
