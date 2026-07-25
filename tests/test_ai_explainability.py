"""Validation suite for SHAP Explainability Engine."""

import unittest
import pandas as pd
import numpy as np

from src.spectraguard_cv_engine.ml.models.config import TrainingConfig
from src.spectraguard_cv_engine.ml.models.trainer import ModelTrainer
from src.spectraguard_cv_engine.ai.explainability.models import ExplanationOutput
from src.spectraguard_cv_engine.ai.explainability.engine import ExplainabilityEngine


class TestExplainabilityEngine(unittest.TestCase):
    def setUp(self):
        # Create synthetic data where f1 is highly predictive, f2 is noise
        np.random.seed(42)

        self.features = ["f1", "f2", "f3"]
        X_0 = np.random.normal(loc=0.0, scale=0.1, size=(20, 3))
        X_1 = np.random.normal(loc=5.0, scale=0.1, size=(20, 3))

        self.X_train = pd.DataFrame(np.vstack([X_0, X_1]), columns=self.features)
        self.y_train = pd.Series([0] * 20 + [1] * 20)

        config = TrainingConfig(
            model_type="random_forest",
            hyperparameters={"n_estimators": 5, "max_depth": 2},
        )
        self.trainer = ModelTrainer(config)
        self.trainer.train(self.X_train, self.y_train)

        self.engine = ExplainabilityEngine(self.trainer)

    def test_untrained_model_rejection(self):
        untrained = ModelTrainer(TrainingConfig("random_forest"))
        with self.assertRaises(RuntimeError):
            ExplainabilityEngine(untrained)

    def test_empty_matrix_rejection(self):
        empty_df = pd.DataFrame(columns=self.features)
        with self.assertRaises(ValueError):
            self.engine.explain(empty_df)

    def test_shap_explanation_extraction(self):
        # Create a single test instance clearly belonging to class 1
        X_test = pd.DataFrame({"f1": [5.0], "f2": [5.0], "f3": [5.0]})

        explanations = self.engine.explain(X_test, top_k=2)

        self.assertEqual(len(explanations), 1)
        exp = explanations[0]

        self.assertIsInstance(exp, ExplanationOutput)
        self.assertIsInstance(exp.base_value, float)

        # Verify all features have attributions
        for f in self.features:
            self.assertIn(f, exp.feature_attributions)
            self.assertIsInstance(exp.feature_attributions[f], float)

        # Verify top_k filtering applied correctly
        self.assertEqual(len(exp.top_contributors), 2)

        # Verify sorting by magnitude
        mags = [abs(v) for v in exp.top_contributors.values()]
        self.assertGreaterEqual(mags[0], mags[1])


if __name__ == "__main__":
    unittest.main()
