"""Held-out record metadata must follow the same samples as predictions."""

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import STGraphDataset, collate_st_graph_windows
from evaluate import predict_test_set


class ZeroModel(torch.nn.Module):
    def forward(self, windows):
        return torch.zeros((len(windows), 1))


class EvaluationOrderTests(unittest.TestCase):
    def test_shuffled_loader_cannot_misattribute_held_out_predictions(self):
        samples = [
            {"snapshots": [index], "target": torch.tensor([float(index)]),
             "mask": torch.tensor([True]), "race_id": f"race-{index}",
             "target_time": float(index), "driver_names": [f"DR{index}"],
             "gap_sec": [float(index + 1)]}
            for index in range(3)
        ]
        loader = DataLoader(STGraphDataset(samples), batch_size=2, shuffle=True,
                            collate_fn=collate_st_graph_windows,
                            generator=torch.Generator().manual_seed(5))

        with self.assertRaisesRegex(ValueError, "sequential"):
            predict_test_set(ZeroModel(), loader, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
