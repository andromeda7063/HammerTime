"""Source-only contracts for opt-in mask-aware and typed-edge model paths."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import HammerTimeSTGNN


def snapshot(x, edge_index=None, edge_attr=None, mask=None):
    return Data(
        x=x,
        edge_index=(torch.empty((2, 0), dtype=torch.long)
                    if edge_index is None else edge_index),
        edge_attr=(torch.empty((0, 5), dtype=torch.float32)
                   if edge_attr is None else edge_attr),
        mask=(torch.ones(x.size(0), dtype=torch.bool) if mask is None else mask),
        position_coverage=1.0,
    )


class GNNv2ModelTests(unittest.TestCase):
    def make_model(self, *, mask=True, typed=False, auxiliary=False):
        return HammerTimeSTGNN(
            in_dim=3, hidden_dim=8, gat_heads=2, edge_dim=5,
            edge_feature_schema="physical_v2", dropout=0.0,
            mask_inactive_nodes=mask, typed_edge_values=typed,
            predict_future_speed=auxiliary, future_speed_horizon_steps=5,
        ).eval()

    def test_legacy_default_ignores_mask_and_adds_no_optional_checkpoint_keys(self):
        model = self.make_model(mask=False)
        x = torch.randn(3, 3)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        attrs = torch.rand(2, 5)
        first = [snapshot(x, edge_index, attrs, torch.ones(3, dtype=torch.bool))]
        masked = [snapshot(x, edge_index, attrs, torch.tensor([True, False, True]))]
        with torch.no_grad():
            torch.testing.assert_close(model(first), model(masked))
        keys = model.state_dict()
        self.assertFalse(any("typed_" in key or "future_speed_head" in key for key in keys))

    def test_active_predictions_are_invariant_to_added_padding_slots(self):
        torch.manual_seed(23)
        model = self.make_model(mask=True)
        real_x = torch.randn(3, 3)
        real_edges = torch.tensor([[0, 1], [1, 2]])
        real_attrs = torch.rand(2, 5)
        compact = [snapshot(real_x, real_edges, real_attrs)]
        padded_x = torch.cat((real_x, torch.randn(2, 3)), dim=0)
        padded_edges = torch.cat((real_edges, torch.tensor([[3], [4]])), dim=1)
        padded_attrs = torch.cat((real_attrs, torch.rand(1, 5)), dim=0)
        padded = [snapshot(padded_x, padded_edges, padded_attrs,
                           torch.tensor([True, True, True, False, False]))]
        with torch.no_grad():
            torch.testing.assert_close(model(compact), model(padded)[:3], rtol=1e-5, atol=1e-6)

    def test_edge_attributes_condition_message_values(self):
        model = self.make_model(mask=True, typed=True)
        model.typed_norm = nn.Identity()
        for gate in model.typed_gate:
            nn.init.zeros_(gate.weight)
        with torch.no_grad():
            model.typed_gate[0].bias.fill_(30.0)
            model.typed_gate[1].bias.fill_(30.0)
            for layer in model.typed_value:
                layer.weight.zero_()
            for layer in model.typed_edge_value:
                layer.weight.zero_()
            model.typed_edge_value[0].weight[0, 0] = 1.0
        h = torch.zeros((2, 8))
        edge_index = torch.tensor([[0], [1]])
        active = torch.ones(2, dtype=torch.bool)
        attrs_a = torch.zeros((1, 5))
        attrs_b = attrs_a.clone()
        attrs_b[0, 0] = 1.0
        result_a = model._typed_messages(h, edge_index, attrs_a, active)
        result_b = model._typed_messages(h, edge_index, attrs_b, active)
        self.assertFalse(torch.equal(result_a, result_b))

    def test_relation_zero_sends_from_leader_to_follower(self):
        model = self.make_model(mask=True, typed=True)
        model.typed_norm = nn.Identity()
        with torch.no_grad():
            for relation in range(2):
                model.typed_value[relation].weight.zero_()
                model.typed_edge_value[relation].weight.zero_()
                model.typed_gate[relation].weight.zero_()
                model.typed_gate[relation].bias.fill_(30.0 if relation == 0 else -30.0)
            model.typed_value[0].weight.copy_(torch.eye(8))
        h = torch.zeros((2, 8))
        h[0, 0] = 1.0
        result = model._typed_messages(
            h, torch.tensor([[0], [1]]), torch.zeros((1, 5)), torch.ones(2, dtype=torch.bool)
        )
        self.assertGreater(float(result[1, 0]), float(result[0, 0]))

    def test_batched_and_single_paths_match_and_auxiliary_shapes_are_explicit(self):
        torch.manual_seed(31)
        model = self.make_model(mask=True, typed=True, auxiliary=True)
        windows = []
        for offset in (0.0, 0.4):
            windows.append([
                snapshot(torch.randn(3, 3) + offset,
                         torch.tensor([[0, 1], [1, 2]]), torch.rand(2, 5))
                for _ in range(2)
            ])
        with torch.no_grad():
            single, single_aux = model(windows[0], return_aux=True)
            batched, batch_aux = model(windows, return_aux=True)
            legacy_default = model(windows[0])
        self.assertEqual(tuple(single.shape), (3,))
        self.assertEqual(tuple(single_aux["future_speed"].shape), (3, 5))
        self.assertEqual(tuple(batched.shape), (2, 3))
        self.assertEqual(tuple(batch_aux["future_speed"].shape), (2, 3, 5))
        torch.testing.assert_close(legacy_default, single)
        torch.testing.assert_close(batched[0], single, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(batch_aux["future_speed"][0], single_aux["future_speed"])


if __name__ == "__main__":
    unittest.main()
