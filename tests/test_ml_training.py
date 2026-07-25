"""Validation suite for model training, configuration, and checkpointing."""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np

from src.spectraguard_cv_engine.ml.models.config import TrainingConfig
from src.spectraguard_cv_engine.ml.models.trainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

        # Simple synthetic dataset (linearly separable for guaranteed learning)
        np.random.seed(42)
        X_0 = np.random.normal(loc=0.0, scale=1.0, size=(50, 2))
        X_1 = np.random.normal(loc=5.0, scale=1.0, size=(50, 2))

        self.X_train = pd.DataFrame(np.vstack([X_0, X_1]), columns=["f1", "f2"])
        self.y_train = pd.Series([0] * 50 + [1] * 50)

        self.rf_config = TrainingConfig(
            model_type="random_forest",
            checkpoint_dir=self.test_dir.name,
            hyperparameters={"n_estimators": 10, "max_depth": 3},
        )

        self.xgb_config = TrainingConfig(
            model_type="xgboost",
            checkpoint_dir=self.test_dir.name,
            hyperparameters={"n_estimators": 10, "max_depth": 3},
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def test_invalid_config_type(self):
        with self.assertRaises(ValueError):
            TrainingConfig(model_type="svm")

    def test_random_forest_training_and_inference(self):
        trainer = ModelTrainer(self.rf_config)
        self.assertFalse(trainer.is_trained)

        with self.assertRaises(RuntimeError):
            trainer.predict(self.X_train)

        trainer.train(self.X_train, self.y_train)
        self.assertTrue(trainer.is_trained)

        predictions = trainer.predict(self.X_train)
        self.assertEqual(len(predictions), 100)

        # Given it's linearly separable, accuracy should be near perfect
        accuracy = (predictions == self.y_train).mean()
        self.assertGreater(accuracy, 0.9)

    def test_xgboost_training_and_inference(self):
        trainer = ModelTrainer(self.xgb_config)
        trainer.train(self.X_train, self.y_train)
        self.assertTrue(trainer.is_trained)

        probs = trainer.predict_proba(self.X_train)
        self.assertEqual(probs.shape, (100, 2))

        preds = trainer.predict(self.X_train)
        accuracy = (preds == self.y_train).mean()
        self.assertGreater(accuracy, 0.9)

    def test_checkpointing_serialization(self):
        trainer = ModelTrainer(self.rf_config)
        trainer.train(self.X_train, self.y_train)

        checkpoint_path = trainer.save_checkpoint("test_rf.joblib")
        self.assertTrue(os.path.exists(checkpoint_path))

        loaded_trainer = ModelTrainer.load_checkpoint(checkpoint_path)
        self.assertTrue(loaded_trainer.is_trained)
        self.assertEqual(loaded_trainer.config.model_type, "random_forest")

        # Predictions should match exactly
        orig_preds = trainer.predict(self.X_train)
        loaded_preds = loaded_trainer.predict(self.X_train)
        np.testing.assert_array_equal(orig_preds, loaded_preds)


if __name__ == "__main__":
    unittest.main()
