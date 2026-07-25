"""Validation suite for AI Inference Runtime operations."""

import os
import tempfile
import unittest
import pandas as pd

from src.spectraguard_cv_engine.ml.preprocessing.scaler import FeatureScaler
from src.spectraguard_cv_engine.ml.models.config import TrainingConfig
from src.spectraguard_cv_engine.ml.models.trainer import ModelTrainer
from src.spectraguard_cv_engine.ml.export.exporter import ModelExporter

from src.spectraguard_cv_engine.ai.runtime.config import RuntimeConfig
from src.spectraguard_cv_engine.ai.runtime.loader import ModelLoader
from src.spectraguard_cv_engine.ai.runtime.engine import InferenceRuntime
from src.spectraguard_cv_engine.ai.runtime.models import PredictionOutput


class TestInferenceRuntime(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

        # 1. Synthesize a dummy Phase 6 release for integration testing
        features = ["f1", "f2"]
        X_train = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y_train = pd.Series([0, 1])

        scaler = FeatureScaler(method="standard")
        X_scaled = scaler.fit_transform(X_train, features)

        config = TrainingConfig(
            model_type="random_forest",
            checkpoint_dir=self.test_dir.name,
            hyperparameters={"n_estimators": 5},
        )
        trainer = ModelTrainer(config)
        trainer.train(X_scaled, y_train)

        # 2. Export the dummy release
        self.version_dir = ModelExporter.export_pipeline(
            trainer=trainer,
            scaler=scaler,
            evaluation_report={"metrics": {"accuracy": 1.0}},
            export_dir=self.test_dir.name,
            version="v1.0.0_test",
        )

        # 3. Load using AI-1 Loader
        self.artifacts = ModelLoader.load_version(self.version_dir)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_model_loader_integrity(self):
        self.assertIsNotNone(self.artifacts.trainer)
        self.assertIsNotNone(self.artifacts.scaler)
        self.assertEqual(self.artifacts.manifest["version"], "v1.0.0_test")
        self.assertTrue(self.artifacts.trainer.is_trained)
        self.assertTrue(self.artifacts.scaler.is_fitted)

    def test_missing_version_directory(self):
        with self.assertRaises(FileNotFoundError):
            ModelLoader.load_version(os.path.join(self.test_dir.name, "invalid_path"))

    def test_inference_engine_execution(self):
        runtime = InferenceRuntime(self.artifacts, RuntimeConfig())

        # Create a new unscaled feature matrix matching schema
        X_test = pd.DataFrame({"f1": [1.5, 2.5], "f2": [3.5, 4.5]})

        results = runtime.predict(X_test)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], PredictionOutput)

        # Verify schema bounds
        for res in results:
            self.assertIn(res.prediction, [0, 1])
            self.assertIsNotNone(res.probability)
            self.assertTrue(0.0 <= res.probability <= 1.0)
            self.assertGreater(res.latency_ms, 0.0)
            self.assertIsInstance(res.timestamp_utc, str)

    def test_inference_batch_limit_enforcement(self):
        config = RuntimeConfig(max_batch_size=2)
        runtime = InferenceRuntime(self.artifacts, config)

        # Construct batch size of 3 (exceeds limit)
        X_test = pd.DataFrame({"f1": [1, 2, 3], "f2": [1, 2, 3]})

        with self.assertRaises(ValueError):
            runtime.predict(X_test)

    def test_schema_validation_rejection(self):
        runtime = InferenceRuntime(
            self.artifacts, RuntimeConfig(enforce_schema_validation=True)
        )

        # Missing "f2"
        X_invalid = pd.DataFrame({"f1": [1.0]})

        with self.assertRaises(KeyError):
            runtime.predict(X_invalid)


if __name__ == "__main__":
    unittest.main()
