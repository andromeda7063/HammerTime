"""Tiny offline integration check from train-only scaling through checkpoint scoring."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import create_dataloaders
from evaluate import run_full_evaluation
from model import create_model
from train import train_model
from test_dataset_contracts import snapshot


class TrainingIntegrationTests(unittest.TestCase):
    def test_checkpoint_and_predictions_reproduce_on_identical_cpu_runs(self):
        with (SRC_DIR.parent / "configs" / "default.yaml").open(encoding="utf-8") as source:
            base = yaml.safe_load(source)
        base["graph"]["num_nodes"] = 2
        base["model"].update(hidden_dim=8, gat_heads=2)
        base["data"].update(window_size=2, horizon=1, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        base["train"].update(max_epochs=2, batch_size=8, early_stopping_patience=2)
        labels = pd.DataFrame([
            {"RaceId": "race-a", "Driver": "AA", "TimeSec": i * 0.25,
             "DeltaT_Sector": (i % 5) / 10, "AvgGap": 4.0, "LapNumber": 1, "MiniSector": i % 20}
            for i in range(80)])
        reports, weights, histories = [], [], []
        with tempfile.TemporaryDirectory() as directory:
            for run in range(2):
                config = copy.deepcopy(base)
                run_dir = Path(directory) / str(run)
                config["session"]["processed_dir"] = str(run_dir)
                config["train"]["checkpoint_path"] = str(run_dir / "best.pt")
                frames = [snapshot(i * 0.25, value=i / 80) for i in range(80)]
                train, validation, test, scaler = create_dataloaders(frames, labels, {"AA": 0, "BB": 1}, config)
                model = create_model(config).to("cpu")
                best, history = train_model(model, train, validation, config, torch.device("cpu"))
                report = run_full_evaluation(config["train"]["checkpoint_path"], test, config)
                checkpoint = torch.load(config["train"]["checkpoint_path"], map_location="cpu", weights_only=True)
                self.assertEqual(best, checkpoint["val_mae"])
                self.assertEqual(report["metrics"]["sample_count"], len(test.dataset))
                self.assertEqual(report["metadata"]["feature_scaler"]["mean_"], scaler.mean_.tolist())
                self.assertEqual(json.loads((run_dir / "evaluation_report.json").read_text()), report)
                reports.append(report)
                weights.append(checkpoint["model_state_dict"])
                histories.append(history)
        self.assertEqual(histories[0], histories[1])
        self.assertEqual(reports[0]["predictions"], reports[1]["predictions"])
        for name in weights[0]:
            self.assertTrue(torch.equal(weights[0][name], weights[1][name]), name)


if __name__ == "__main__":
    unittest.main()
