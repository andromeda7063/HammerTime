"""A fresh run must train its configured split and preserve historical checkpoints."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import run_pipeline as pipeline


class CheckpointProvenanceTests(unittest.TestCase):
    def test_existing_checkpoint_never_skips_configured_training(self):
        with (PROJECT_DIR / "configs" / "default.yaml").open(encoding="utf-8") as source:
            config = yaml.safe_load(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config["session"]["processed_dir"] = str(root)
            config["session"]["cache_dir"] = str(root / "cache")
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"historical checkpoint")
            config["train"]["checkpoint_path"] = str(checkpoint)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            loader = Mock()
            loader.__len__ = Mock(return_value=1)
            loader.dataset.samples = []
            aligned = pd.DataFrame({"TimeSec": [0.0], "Driver": ["AA"]})
            labels = pd.DataFrame({"DeltaT_Sector": [0.1]})
            with patch("ingest.load_race"), patch("ingest.export_raw_tables", return_value=(root / "raw", None)), \
                    patch("preprocess.process_race", return_value=aligned), \
                    patch("target.build_targets", return_value=labels), \
                    patch("graph.build_race_graph_snapshots", return_value=([Mock()], {"AA": 0})), \
                    patch("dataset.create_dataloaders", return_value=(loader, loader, loader, None)), \
                    patch("model.create_model"), patch("train.train_model", return_value=(0.2, [])) as train, \
                    patch("baselines.run_baselines", return_value={"xgboost": {"available": False}}), \
                    patch("evaluate.run_full_evaluation", return_value={"metrics": {}, "physics_tests": []}), \
                    patch.object(pipeline, "_render_real_outputs", return_value=[]), \
                    patch.object(pd.DataFrame, "to_parquet"), \
                    patch.object(pipeline.torch, "load", side_effect=AssertionError("unverified checkpoint read")):
                result = pipeline.run_100_percent_pipeline(config_path)
            train.assert_called_once()
            self.assertEqual(result["best_validation_mae_sec"], 0.2)
            preserved = list((root / "artifact_history").rglob("best.pt"))
            self.assertEqual(len(preserved), 1)
            self.assertEqual(preserved[0].read_bytes(), b"historical checkpoint")


if __name__ == "__main__":
    unittest.main()
