"""Device contracts for graph edges before spatial message passing."""

import sys
import unittest
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import HammerTimeSTGNN, create_model


class ModelDeviceContractTests(unittest.TestCase):
    def test_seeded_model_initialization_is_repeatable(self):
        config = {"data": {"random_seed": 17},
                  "model": {"hidden_dim": 8, "gat_heads": 2}}
        first = create_model(config)
        torch.rand(10)
        second = create_model(config)
        for name, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, second.state_dict()[name], rtol=0, atol=0)

    def test_self_loops_move_existing_edges_to_requested_device(self):
        # Meta tensors check device transfer without requiring GPU hardware.
        model = HammerTimeSTGNN(hidden_dim=8, gat_heads=2)
        edges = torch.tensor([[0], [1]], dtype=torch.long)
        weights = torch.tensor([[0.5]])
        index, attr = model._ensure_self_loops(2, edges, weights, torch.device("meta"))
        self.assertEqual(index.device.type, "meta")
        self.assertEqual(attr.device.type, "meta")
        self.assertEqual(index.shape, (2, 3))
        self.assertEqual(attr.shape, (3, 1))

    def test_cpu_self_loops_preserve_existing_edges_and_weights(self):
        model = HammerTimeSTGNN(hidden_dim=8, gat_heads=2)
        edges = torch.tensor([[0], [1]], dtype=torch.long)
        weights = torch.tensor([[0.5]])
        index, attr = model._ensure_self_loops(2, edges, weights, torch.device("cpu"))
        torch.testing.assert_close(index[:, :1], edges)
        torch.testing.assert_close(attr[:1], weights)
        torch.testing.assert_close(index[:, 1:], torch.tensor([[0, 1], [0, 1]]))
        torch.testing.assert_close(attr[1:], torch.ones((2, 1)))


if __name__ == "__main__":
    unittest.main()
