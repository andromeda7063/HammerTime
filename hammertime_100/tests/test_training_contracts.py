"""Training guards and validation metrics on small in-memory batches."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import evaluate_val_mae, train_model


class TinyLoader(list):
    dataset = SimpleNamespace(feature_scaler=None)


class ConstantModel(nn.Module):
    def __init__(self, value=0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, shape):
        return self.value.expand(shape)


class NonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("nan"))


class NonfiniteGradientModel(ConstantModel):
    def forward(self, shape):
        return NonfiniteGradient.apply(self.value).expand(shape)


class TrainingContractTests(unittest.TestCase):
    device = torch.device("cpu")

    def batch(self, targets, masks):
        targets = torch.tensor(targets, dtype=torch.float32)
        return targets.shape, targets, torch.tensor(masks, dtype=torch.bool)

    def config(self, directory):
        return {"train": {"max_epochs": 1,
                          "checkpoint_path": str(Path(directory) / "best.pt")}}

    def test_validation_rejects_empty_loader(self):
        with self.assertRaisesRegex(ValueError, "valid target"):
            evaluate_val_mae(ConstantModel(), [], self.device)

    def test_validation_rejects_no_valid_targets(self):
        loader = [self.batch([[1.0, 2.0]], [[False, False]])]
        with self.assertRaisesRegex(ValueError, "valid target"):
            evaluate_val_mae(ConstantModel(), loader, self.device)

    def test_validation_weights_nodes_instead_of_batches(self):
        loader = [self.batch([[1.0, 1.0]], [[True, True]]),
                  self.batch([[4.0, float("nan")]], [[True, False]])]
        self.assertAlmostEqual(evaluate_val_mae(ConstantModel(), loader, self.device), 2.0)

    def test_training_rejects_nonfinite_forward_loss_before_checkpoint(self):
        loader = TinyLoader([self.batch([[1.0]], [[True]])])
        with tempfile.TemporaryDirectory() as directory:
            model = ConstantModel(float("nan"))
            with self.assertRaisesRegex(FloatingPointError, "loss"):
                train_model(model, loader, loader, self.config(directory), self.device)
            self.assertFalse((Path(directory) / "best.pt").exists())

    def test_training_rejects_nonfinite_gradient_before_optimizer_step(self):
        loader = TinyLoader([self.batch([[1.0]], [[True]])])
        model = NonfiniteGradientModel()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FloatingPointError, "gradient"):
                train_model(model, loader, loader, self.config(directory), self.device)
            self.assertTrue(torch.isfinite(model.value))
            self.assertEqual(model.value.item(), 0.0)
            self.assertFalse((Path(directory) / "best.pt").exists())

    def test_training_rejects_unmeasurable_validation_before_checkpoint(self):
        train_loader = TinyLoader([self.batch([[1.0]], [[True]])])
        val_loader = TinyLoader([self.batch([[1.0]], [[False]])])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "valid target"):
                train_model(ConstantModel(), train_loader, val_loader,
                            self.config(directory), self.device)
            self.assertFalse((Path(directory) / "best.pt").exists())

    def test_training_writes_finite_checkpoint_and_history(self):
        loader = TinyLoader([self.batch([[1.0]], [[True]])])
        with tempfile.TemporaryDirectory() as directory:
            score, history = train_model(ConstantModel(), loader, loader,
                                         self.config(directory), self.device)
            self.assertTrue(torch.isfinite(torch.tensor(score)))
            self.assertEqual(len(history), 1)
            checkpoint = torch.load(Path(directory) / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertTrue((Path(directory) / "training_history.csv").exists())


if __name__ == "__main__":
    unittest.main()
