"""Validation suite for hyperparameter optimization."""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np

from src.spectraguard_cv_engine.ml.optimization.tuner import HyperparameterTuner
from src.spectraguard_cv_engine.ml.models.trainer import ModelTrainer


class TestHyperparameterTuner(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

        # Synthetic classification dataset
        np.random.seed(42)
        self.X_train = pd.DataFrame(
            {"f1": np.random.rand(100), "f2": np.random.rand(100)}
        )
        self.y_train = pd.Series(np.random.choice([0, 1], size=100))

        # Small param grid for fast tests
        self.rf_params = {"n_estimators": [10, 20], "max_depth": [3, 5]}

    def tearDown(self):
        self.test_dir.cleanup()

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            HyperparameterTuner(model_type="svm", param_distributions={})

    def test_optimization_workflow(self):
        tuner = HyperparameterTuner(
            model_type="random_forest",
            param_distributions=self.rf_params,
            n_iter=2,  # Fast iteration
            cv=2,
        )

        with self.assertRaises(RuntimeError):
            tuner.get_best_trainer("dummy")

        with self.assertRaises(RuntimeError):
            tuner.save_report("dummy.json")

        best_params, best_score = tuner.optimize(self.X_train, self.y_train)

        self.assertIn("n_estimators", best_params)
        self.assertIn("max_depth", best_params)
        self.assertGreater(best_score, 0.0)

        # Test Trainer Extraction
        best_trainer = tuner.get_best_trainer(self.test_dir.name)
        self.assertIsInstance(best_trainer, ModelTrainer)
        self.assertEqual(best_trainer.config.hyperparameters, best_params)

        # Test Report Serialization
        report_path = os.path.join(self.test_dir.name, "opt_report.json")
        tuner.save_report(report_path)
        self.assertTrue(os.path.exists(report_path))


if __name__ == "__main__":
    unittest.main()
