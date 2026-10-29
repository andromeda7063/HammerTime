"""Hyperparameter selection may see only the fixed train/validation loaders."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import yaml

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tuning import tune_hyperparameters


class TuningContractTests(unittest.TestCase):
    def setUp(self):
        with (SRC_DIR.parent / "configs" / "default.yaml").open(encoding="utf-8") as source:
            self.config = yaml.safe_load(source)
        self.config["tuning"] = {"seeds": [42, 43], "candidates": [
            {"train.lr": 0.001}, {"train.lr": 0.0003}]}
        self.train_loader = Mock(generator=torch.Generator().manual_seed(123))
        self.val_loader = Mock()

    def test_selects_mean_validation_score_without_cherry_picking_seed(self):
        original = copy.deepcopy(self.config)
        state = self.train_loader.generator.get_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            self.config["session"]["processed_dir"] = directory
            # Trial 0 has the best individual seed; trial 1 has the best mean.
            scores = iter([(0.01, []), (0.31, []), (0.12, []), (0.10, [])])
            shuffle_samples = []

            def fit(*args):
                shuffle_samples.append(torch.rand(4, generator=self.train_loader.generator))
                return next(scores)

            with patch("tuning.create_model", return_value=Mock()) as create, \
                    patch("tuning.train_model", side_effect=fit) as train:
                selected, result = tune_hyperparameters(
                    self.train_loader, self.val_loader, self.config, torch.device("cpu"))
            self.assertEqual(selected["train"]["lr"], 0.0003)
            self.assertEqual(selected["data"]["random_seed"], 42)
            self.assertAlmostEqual(result["mean_validation_mae_sec"], 0.11)
            self.assertEqual(result["selected_validation_mae_sec"], 0.12)
            self.assertEqual(len(train.call_args_list), 4)
            for call in train.call_args_list:
                self.assertIs(call.args[1], self.train_loader)
                self.assertIs(call.args[2], self.val_loader)
            checkpoints = [call.args[0]["train"]["checkpoint_path"] for call in create.call_args_list]
            self.assertEqual(len(set(checkpoints)), 4)
            self.assertTrue(torch.equal(shuffle_samples[0], shuffle_samples[2]))
            self.assertTrue(torch.equal(shuffle_samples[1], shuffle_samples[3]))
            self.assertTrue(Path(result["study_path"]).is_file())
            saved = json.loads(Path(result["study_path"]).read_text())
            self.assertEqual(saved["selection_metric"], "mean_validation_mae_sec")
        self.assertTrue(torch.equal(state, self.train_loader.generator.get_state()))
        self.config["session"]["processed_dir"] = original["session"]["processed_dir"]
        self.assertEqual(self.config, original)

    def test_invalid_later_candidate_cannot_start_an_incomplete_study(self):
        self.config["tuning"]["candidates"][1]["train.lr"] = 0
        with patch("tuning.train_model") as train:
            with self.assertRaisesRegex(ValueError, "train.lr"):
                tune_hyperparameters(self.train_loader, self.val_loader, self.config, torch.device("cpu"))
        train.assert_not_called()

    def test_forbids_changing_splits_targets_metrics_or_gates(self):
        for key in ("data.horizon", "data.train_ratio", "target.clean_air_gap_sec",
                    "physics_tests.clean_air_max_mae", "preprocess.sample_hz"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                self.config["session"]["processed_dir"] = directory
                self.config["tuning"]["candidates"] = [{key: 1}]
                with patch("tuning.train_model") as train:
                    with self.assertRaisesRegex(ValueError, "candidate"):
                        tune_hyperparameters(self.train_loader, self.val_loader, self.config, torch.device("cpu"))
                train.assert_not_called()

    def test_rejects_nonfinite_validation_and_restores_shuffle_state(self):
        state = self.train_loader.generator.get_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            self.config["session"]["processed_dir"] = directory
            with patch("tuning.create_model", return_value=Mock()), \
                    patch("tuning.train_model", return_value=(float("nan"), [])):
                with self.assertRaises(FloatingPointError):
                    tune_hyperparameters(self.train_loader, self.val_loader, self.config, torch.device("cpu"))
        self.assertTrue(torch.equal(state, self.train_loader.generator.get_state()))


if __name__ == "__main__":
    unittest.main()
