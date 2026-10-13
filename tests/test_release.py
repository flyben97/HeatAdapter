"""Release regression checks for external data, saved predictions, and resume guards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from graphheat.mol_data import load_dataset
from graphheat.predict import predict_csv


ROOT = Path(__file__).resolve().parents[1]


class DataInputTests(unittest.TestCase):
    def test_custom_csv_curates_duplicates_and_nonfinite_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            pd.DataFrame(
                {
                    "structure": ["CCO", "OCC", "invalid", "CCN", "CCC", "CO", "C"],
                    "measurement": [1, 3, 6, np.inf, -np.inf, "bad", None],
                }
            ).to_csv(path, index=False)
            frame, report = load_dataset(
                "custom",
                csv_path=path,
                smiles_column="structure",
                target_column="measurement",
                return_report=True,
            )
            self.assertEqual(frame.SMILES.tolist(), ["CCO"])
            self.assertEqual(frame.target.tolist(), [2.0])
            self.assertEqual(frame.n_measurements.tolist(), [2])
            self.assertEqual(report["invalid_smiles_removed"], 1)
            self.assertEqual(report["canonical_duplicates_collapsed"], 1)

    def test_named_dataset_from_external_root(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "datasets/csv/backup/logS_esol_delaney.csv"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                {"smiles": ["CCO"], "measured log solubility in mols per litre": [1.0]}
            ).to_csv(path, index=False)
            frame = load_dataset("esol", base_dir=folder)
            self.assertEqual(frame.SMILES.tolist(), ["CCO"])

    def test_empty_curated_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            pd.DataFrame({"smiles": ["CCO"], "target": [np.inf]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "no valid"):
                load_dataset("custom", csv_path=path)


class TrainingArtifactTests(unittest.TestCase):
    def test_training_prediction_and_resume_from_another_directory(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data_path = folder / "molecules.csv"
            data_path.write_bytes((ROOT / "examples/toy_molecules.csv").read_bytes())
            output = folder / "run"
            command = [
                sys.executable,
                "-m",
                "analysis.revision_benchmark",
                "--csv",
                str(data_path),
                "--backbones",
                "gcn",
                "--variants",
                "base,mlp,heat,identity,zero,fixed,single,unweighted,topk20",
                "--folds",
                "0",
                "--seeds",
                "42",
                "--epochs",
                "2",
                "--patience",
                "2",
                "--learning-rates",
                "0.001,0.0003",
                "--hidden",
                "16",
                "--depth",
                "1",
                "--batch-size",
                "32",
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
            env = {**os.environ, "PYTHONPATH": str(ROOT)}
            first = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            results = sorted((output / "custom").glob("gcn_*.json"))
            self.assertEqual(len(results), 9)
            before = {path.name: path.stat().st_mtime_ns for path in results}
            for path in results:
                row = json.loads(path.read_text())
                self.assertFalse(Path(row["checkpoint"]).is_absolute())
                predictions = pd.read_csv(path.parent / row["predictions"])
                input_path = folder / "prediction_input.csv"
                predictions[["smiles"]].to_csv(input_path, index=False)
                restored = predict_csv(
                    path.parent / row["checkpoint"], input_path, folder / "restored.csv"
                )
                np.testing.assert_allclose(
                    restored.prediction, predictions.prediction, atol=1e-5, rtol=1e-6
                )
                self.assertTrue(np.isfinite(row["test_mae"]))
            second = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertNotIn("COMPLETE", second.stdout)
            self.assertEqual(before, {path.name: path.stat().st_mtime_ns for path in results})
            frame = pd.read_csv(data_path)
            frame.loc[0, "target"] += 1
            frame.to_csv(data_path, index=False)
            changed = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("differs from existing manifest", changed.stderr)


if __name__ == "__main__":
    unittest.main()
