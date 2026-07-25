"""Validation suite for ML Feature Scaling operations."""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np

from src.spectraguard_cv_engine.ml.preprocessing.scaler import FeatureScaler


class TestFeatureScaler(unittest.TestCase):
    def setUp(self):
        # Temporary directory for saving scaler states
        self.test_dir = tempfile.TemporaryDirectory()
        self.scaler_path = os.path.join(self.test_dir.name, "test_scaler.joblib")

        self.features = ["f1", "f2"]
        # Create a synthetic dataset with known variance
        self.df = pd.DataFrame(
            {
                "f1": [10, 20, 30, 40, 50],  # Mean: 30, Std: approx 14.14 (sample)
                "f2": [100, 200, 300, 400, 500],
                "label": [0, 1, 0, 1, 0],
            }
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def test_standard_scaling_logic(self):
        scaler = FeatureScaler(method="standard")

        with self.assertRaises(RuntimeError):
            scaler.transform(self.df)  # Cannot transform before fitting

        transformed_df = scaler.fit_transform(self.df, self.features)

        self.assertTrue(scaler.is_fitted)
        self.assertEqual(scaler.feature_names, self.features)

        # Verify Z-score standardization properties (Mean ~ 0, Variance ~ 1)
        # Using numpy var (ddof=0) matching sklearn default behavior
        self.assertAlmostEqual(transformed_df["f1"].mean(), 0.0, places=5)
        self.assertAlmostEqual(transformed_df["f1"].std(ddof=0), 1.0, places=5)

        # Verify non-feature columns are untouched
        self.assertTrue(
            np.array_equal(transformed_df["label"].values, self.df["label"].values)
        )

    def test_minmax_scaling_logic(self):
        scaler = FeatureScaler(method="minmax")
        transformed_df = scaler.fit_transform(self.df, self.features)

        # Verify Min-Max bounds [0, 1]
        self.assertAlmostEqual(transformed_df["f1"].min(), 0.0, places=5)
        self.assertAlmostEqual(transformed_df["f1"].max(), 1.0, places=5)
        self.assertAlmostEqual(transformed_df["f2"].min(), 0.0, places=5)

    def test_missing_feature_validation(self):
        scaler = FeatureScaler()
        bad_features = ["f1", "f3"]  # f3 does not exist

        with self.assertRaises(KeyError):
            scaler.fit(self.df, bad_features)

    def test_scaler_serialization(self):
        scaler = FeatureScaler(method="standard")
        scaler.fit(self.df, self.features)

        # Save state
        scaler.save(self.scaler_path)
        self.assertTrue(os.path.exists(self.scaler_path))

        # Load into a new instance
        loaded_scaler = FeatureScaler.load(self.scaler_path)

        self.assertTrue(loaded_scaler.is_fitted)
        self.assertEqual(loaded_scaler.method, "standard")
        self.assertEqual(loaded_scaler.feature_names, self.features)

        # Ensure transformation logic is perfectly reproducible
        df_original = scaler.transform(self.df)
        df_loaded = loaded_scaler.transform(self.df)

        pd.testing.assert_frame_equal(df_original, df_loaded)


if __name__ == "__main__":
    unittest.main()
