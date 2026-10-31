"""Focused causality, history, and legacy checkpoint contracts."""

import copy
import sys
import unittest
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_validation import validate_config
from model import CausalGatedResidualBlock, create_model


class GatedDilatedTemporalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.blocks = torch.nn.Sequential(*[
            CausalGatedResidualBlock(4, dilation, dropout=0.0)
            for dilation in (1, 2, 4)
        ]).eval()

    def test_future_perturbations_do_not_change_earlier_temporal_outputs(self):
        sequence = torch.randn(2, 4, 10)
        changed = sequence.clone()
        changed[:, :, 6:] += 10.0
        with torch.no_grad():
            expected = self.blocks(sequence)
            actual = self.blocks(changed)
        torch.testing.assert_close(actual[:, :, :6], expected[:, :, :6])

    def test_earliest_window_snapshot_can_influence_final_output(self):
        sequence = torch.randn(1, 4, 10)
        changed = sequence.clone()
        changed[:, :, 0] += 1.0
        with torch.no_grad():
            expected = self.blocks(sequence)
            actual = self.blocks(changed)
        self.assertFalse(torch.allclose(actual[:, :, -1], expected[:, :, -1]))
        self.assertEqual(1 + 2 * sum((1, 2, 4)), 15)

    def test_default_conv_checkpoint_keys_remain_strictly_compatible(self):
        base = {"data": {"random_seed": 42}, "model": {
            "in_dim": 9, "hidden_dim": 8, "gat_heads": 2,
            "edge_dim": 1, "conv1d_kernel": 3, "dropout": 0.0,
        }}
        legacy = create_model(base)
        explicit_conv_config = copy.deepcopy(base)
        explicit_conv_config["model"]["temporal_mode"] = "conv"
        current = create_model(explicit_conv_config)
        self.assertEqual(set(legacy.state_dict()), set(current.state_dict()))
        current.load_state_dict(legacy.state_dict(), strict=True)
        self.assertIn("tconv.weight", legacy.state_dict())

    def test_gated_model_preserves_per_driver_output_shape(self):
        config = {"data": {"random_seed": 42}, "model": {
            "in_dim": 9, "hidden_dim": 8, "gat_heads": 2,
            "edge_dim": 1, "conv1d_kernel": 3, "dropout": 0.0,
            "temporal_mode": "gated_dilated", "temporal_dilations": [1, 2, 4],
        }}
        model = create_model(config).eval()
        snapshots = [Data(
            x=torch.randn(2, 9), edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1)),
        ) for _ in range(10)]
        with torch.no_grad():
            output = model(snapshots)
        self.assertEqual(tuple(output.shape), (2,))

    def test_vectorized_gated_batch_matches_per_window_reference(self):
        config = {"data": {"random_seed": 42}, "model": {
            "in_dim": 9, "hidden_dim": 8, "gat_heads": 2,
            "edge_dim": 1, "conv1d_kernel": 3, "dropout": 0.0,
            "temporal_mode": "gated_dilated", "temporal_dilations": [1, 2, 4],
        }}
        model = create_model(config).eval()
        windows = [[Data(
            x=torch.randn(4, 9),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_attr=torch.tensor([[0.6], [0.2]]),
            mask=torch.tensor([True, False, True, True]),
        ) for _ in range(10)] for _ in range(2)]
        with torch.no_grad():
            reference = torch.stack([model.forward_window(window) for window in windows])
            actual = model(windows)
        self.assertEqual(tuple(actual.shape), (2, 4))
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    def test_active_gated_config_validates_full_history_receptive_field(self):
        with (CONFIG_DIR / "supplied_fastf1_monza_2023_gated.yaml").open(encoding="utf-8") as source:
            config = yaml.safe_load(source)
        validate_config(config)
        config["model"]["temporal_dilations"] = [1]
        with self.assertRaisesRegex(ValueError, "receptive field"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
