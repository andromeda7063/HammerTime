"""Offline contracts for pre-registered validation-only ablation runs."""

import inspect
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ablation
from dataset import STGraphDataset, collate_st_graph_windows
from train import evaluate_val_mae


def graph(time):
    return Data(x=torch.full((2, 9), float(time)),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=torch.tensor([[0.25]], dtype=torch.float32),
                mask=torch.ones(2, dtype=torch.bool), time_sec=float(time),
                race_id="fixture-race", driver_names=("AA", "BB"))


def sample(index, base_time):
    return {"snapshots": [graph(base_time + i * 0.25) for i in range(10)],
            "target": torch.tensor([float(index), float(index + 1)]),
            "mask": torch.ones(2, dtype=torch.bool),
            "race_id": "fixture-race", "target_time": base_time + 0.25,
            "input_start_time": base_time, "driver_names": ["AA", "BB"]}


def loader(samples, shuffle):
    return DataLoader(STGraphDataset(samples), batch_size=2, shuffle=shuffle,
                      collate_fn=collate_st_graph_windows,
                      generator=torch.Generator().manual_seed(7) if shuffle else None)


def config():
    return {
        "data": {"random_seed": 42, "window_size": 10, "horizon": 5,
                 "split_method": "temporal"},
        "model": {"in_dim": 9, "hidden_dim": 8, "gat_heads": 2,
                  "edge_dim": 1, "conv1d_kernel": 3, "dropout": 0.0},
        "train": {"batch_size": 2, "num_workers": 0, "lr": 0.001,
                  "weight_decay": 0.0001, "max_epochs": 1,
                  "early_stopping_patience": 1},
    }


def source_provenance(directory):
    artifact = directory / "prepared.pt"
    artifact.write_bytes(b"offline unit-test artifact")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    split_membership = {"train": ["fixture-train"], "validation": ["fixture-val"]}
    manifest = {
        "provider": "FastF1", "session_identity": "unit-test-only",
        "prepared_artifact_sha256": artifact_hash,
        "effective_config_sha256": ablation._sha256_json(config()),
        "source_hashes": {"fixture-cache": "unit-test-only"},
        "coverage": {"fixture": True},
        "split_membership": split_membership,
        "split_membership_sha256": {
            key: ablation._sha256_json(value) for key, value in split_membership.items()
        },
    }
    manifest["prepared_manifest_sha256"] = ablation._sha256_json(manifest)
    manifest_path = directory / "prepared.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "prepared.json.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(), encoding="utf-8")
    return {
        "provider": "FastF1", "session_identity": "unit-test-only",
        "prepared_artifact_path": str(artifact), "prepared_artifact_sha256": artifact_hash,
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": manifest["prepared_manifest_sha256"],
        "effective_config_sha256": ablation._sha256_json(config()),
    }


class AblationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.train_loader = loader([sample(0, 0.0), sample(1, 1.0)], True)
        self.val_loader = loader([sample(2, 2.0), sample(3, 3.0)], False)

    def test_graph_transforms_preserve_source_and_labels(self):
        samples = [sample(0, 0.0)]
        source_edge = samples[0]["snapshots"][0].edge_index.clone()
        self_loop = ablation._clone_transformed_samples(samples, "self_loop_only")
        constant = ablation._clone_transformed_samples(samples, "constant_edge_attribute")
        self.assertEqual(self_loop[0]["snapshots"][0].edge_index.shape, (2, 0))
        torch.testing.assert_close(constant[0]["snapshots"][0].edge_index, source_edge)
        torch.testing.assert_close(constant[0]["snapshots"][0].edge_attr,
                                   torch.ones((1, 1)))
        torch.testing.assert_close(samples[0]["snapshots"][0].edge_index, source_edge)
        torch.testing.assert_close(self_loop[0]["target"], samples[0]["target"])
        torch.testing.assert_close(self_loop[0]["mask"], samples[0]["mask"])

    def test_runner_accepts_only_train_and_validation_loaders(self):
        self.assertEqual(list(inspect.signature(ablation.run_ablation_study).parameters)[:2],
                         ["train_loader", "val_loader"])

    def test_complete_study_requires_at_least_three_matched_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = source_provenance(Path(temporary))
            with self.assertRaisesRegex(ValueError, "at least three matched seeds"):
                ablation.run_ablation_study(
                    self.train_loader, self.val_loader, config(), torch.device("cpu"),
                    Path(temporary) / "complete", source, "No test access.",
                    seeds=(42, 43), mode="complete",
                )

    def test_runner_rejects_mixed_provider_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = source_provenance(Path(temporary))
            source["providers"] = ["FastF1", "OpenF1"]
            with self.assertRaisesRegex(ValueError, "cannot mix data providers"):
                ablation.run_ablation_study(
                    self.train_loader, self.val_loader, config(), torch.device("cpu"),
                    Path(temporary) / "mixed", source, "No test access.",
                    seeds=(42,), mode="pilot", max_runs=1, epoch_cap=1,
                )

    def test_pilot_preregisters_before_run_and_reloads_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            study_path = directory / "pilot"
            source = source_provenance(directory)
            exposure = "Prior report exposed held-out metrics; this pilot does not access test data."

            def fake_train(model, train_loader, val_loader, trial_config, device):
                prereg = study_path / "preregistration.json"
                self.assertTrue(prereg.is_file())
                self.assertEqual(json.loads(prereg.read_text())["status"], "preregistered")
                score = evaluate_val_mae(model, val_loader, device)
                torch.save({"model_state_dict": {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }}, trial_config["train"]["checkpoint_path"])
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                return score, [{"epoch": 1, "val_mae": score}]

            with patch.object(ablation, "train_model", side_effect=fake_train):
                prereg, result = ablation.run_ablation_study(
                    self.train_loader, self.val_loader, config(), torch.device("cpu"),
                    study_path, source, exposure, seeds=(42, 43, 44),
                    mode="pilot", max_runs=1, epoch_cap=1,
                )

            self.assertEqual(prereg["max_runs"], 1)
            self.assertEqual(len(prereg["train_validation_membership_sha256"]), 64)
            self.assertFalse(prereg["test_handling"]["test_loader_passed_to_runner"])
            self.assertEqual(prereg["validation_zero_prediction_baseline"]["sample_count"], 4)
            self.assertEqual(prereg["validation_zero_prediction_baseline_role"],
                             "context only; not a candidate and not used for selection")
            self.assertEqual(result["status"], "incomplete")
            self.assertIsNone(result["candidate_selection"])
            self.assertEqual(len(result["runs"]), 1)
            run = result["runs"][0]
            self.assertEqual(run["candidate"], "current_k3")
            self.assertEqual(run["validation_metrics"]["sample_count"], 4)
            self.assertGreater(run["parameter_count"], 0)
            self.assertGreaterEqual(run["runtime_seconds"], 0.0)
            self.assertAlmostEqual(run["validation_metrics"]["mae_sec"],
                                   run["best_checkpoint_validation_mae_sec"], places=6)
            self.assertIsNotNone(run["validation_metrics"]["r2_score"])
            json.loads((study_path / "results.json").read_text())


if __name__ == "__main__":
    unittest.main()
