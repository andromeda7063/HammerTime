"""Shape, masking, batch-equivalence, and gradient checks for the opt-in GNNv3 encoder."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_validation import validate_config
from model import create_model


class EdgeTemporalTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / "configs/train_2023_fastf1_season_gnnv2.yaml").open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        self.config = copy.deepcopy(self.config)
        self.config["model"].update({"spatial_encoder": "edge_temporal", "spatial_layers": 0})
        self.config["model"]["dropout"] = 0.0
        validate_config(self.config)
        self.model = create_model(self.config)
        self.window_size = self.config["data"]["window_size"]
        self.nodes = self.config["graph"]["num_nodes"]
        self.width = self.config["model"]["in_dim"]

    def window(self, seed: int):
        generator = torch.Generator().manual_seed(seed)
        output = []
        for frame in range(self.window_size):
            x = torch.randn(self.nodes, self.width, generator=generator)
            mask = torch.ones(self.nodes, dtype=torch.bool)
            mask[-1] = False
            if frame % 3 == 0:
                edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
                edge_attr = torch.randn(3, 5, generator=generator)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 5), dtype=torch.float32)
            output.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                               mask=mask, position_coverage=0.8))
        return output

    def test_batch_matches_individual_forward_with_aux_and_masks(self):
        windows = [self.window(4), self.window(7)]
        self.model.eval()
        with torch.no_grad():
            singles = torch.stack([self.model(window) for window in windows])
            batched, aux = self.model(windows, return_aux=True)
        self.assertEqual(batched.shape, (2, self.nodes))
        self.assertEqual(aux["future_speed"].shape,
                         (2, self.nodes, self.config["data"]["horizon"]))
        self.assertTrue(torch.isfinite(batched).all())
        self.assertTrue(torch.allclose(batched, singles, atol=1e-6, rtol=1e-6))

    def test_gradient_flow_and_no_attention_parameters(self):
        self.assertIsNone(self.model.gat1)
        self.assertIsNone(self.model.gat2)
        self.model.train()
        prediction, aux = self.model(self.window(11), return_aux=True)
        loss = prediction.square().mean() + aux["future_speed"].square().mean()
        loss.backward()
        grads = [parameter.grad for parameter in self.model.parameters() if parameter.requires_grad]
        self.assertTrue(grads)
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                            for gradient in grads))

    def test_edge_temporal_rejects_missing_physical_message_contract(self):
        invalid = copy.deepcopy(self.config)
        invalid["model"]["typed_edge_values"] = False
        with self.assertRaisesRegex(ValueError, "edge_temporal requires"):
            validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
