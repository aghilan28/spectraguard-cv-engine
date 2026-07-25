"""Validation suite for model evaluation and metric extraction."""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np

from src.spectraguard_cv_engine.ml.models.config import TrainingConfig
from src.spectraguard_cv_engine.ml.models.trainer import ModelTrainer
from src.spectraguard_cv_engine.ml.evaluation.evaluator import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

        # Linearly separable data to guarantee perfect metrics for assertion testing
        np.random.seed(42)
        X_0 = np.random.normal(loc=-5.0, scale=0.5, size=(50, 2))
        X_1 = np.random.normal(loc=5.0, scale=0.5, size=(50, 2))

        self.X = pd.DataFrame(np.vstack([X_0, X_1]), columns=["f1", "f2"])
        self.y = pd.Series([0] * 50 + [1] * 50)

        # Train a basic dummy model
        config = TrainingConfig(
            model_type="random_forest",
            checkpoint_dir=self.test_dir.name,
            hyperparameters={"n_estimators": 5, "random_state": 42},
        )
        self.trainer = ModelTrainer(config)
        self.trainer.train(self.X, self.y)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_untrained_model_rejection(self):
        untrained = ModelTrainer(TrainingConfig("random_forest"))
        with self.assertRaises(RuntimeError):
            ModelEvaluator.evaluate(untrained, self.X, self.y)

    def test_empty_dataset_rejection(self):
        empty_X = pd.DataFrame(columns=["f1", "f2"])
        empty_y = pd.Series(dtype=int)
        with self.assertRaises(ValueError):
            ModelEvaluator.evaluate(self.trainer, empty_X, empty_y)

    def test_evaluation_metrics_extraction(self):
        report = ModelEvaluator.evaluate(self.trainer, self.X, self.y)

        # Verify schema
        self.assertIn("metrics", report)
        self.assertIn("confusion_matrix", report)
        self.assertIn("performance", report)

        metrics = report["metrics"]
        # Since it's perfectly separable, we expect 1.0 accuracy
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

        cm = report["confusion_matrix"]
        # Matrix should be [[50, 0], [0, 50]]
        self.assertEqual(cm["matrix"], [[50, 0], [0, 50]])
        self.assertEqual(cm["false_positive_rate"], 0.0)
        self.assertEqual(cm["false_negative_rate"], 0.0)

        perf = report["performance"]
        self.assertEqual(perf["test_samples"], 100)
        self.assertGreater(perf["total_inference_ms"], 0.0)
        self.assertGreater(perf["avg_inference_ms_per_sample"], 0.0)

    def test_report_serialization(self):
        report = ModelEvaluator.evaluate(self.trainer, self.X, self.y)
        filepath = os.path.join(self.test_dir.name, "eval.json")

        ModelEvaluator.save_report(report, filepath)
        self.assertTrue(os.path.exists(filepath))


if __name__ == "__main__":
    unittest.main()
