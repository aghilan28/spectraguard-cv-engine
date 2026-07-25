"""Validation suite for ML Dataset Loader and Validation."""

import os
import json
import tempfile
import unittest
import pandas as pd
import numpy as np

from src.spectraguard_cv_engine.ml.data.loader import (
    DatasetLoader,
    EXPECTED_UNIFIED_FEATURES,
)


class TestMLDatasetLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for synthetic datasets
        self.test_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.test_dir.name, "synthetic_data.csv")
        self.manifest_path = os.path.join(self.test_dir.name, "manifest.json")
        self.label_col = "is_tampered"

        # Generate valid synthetic data using the exact Phase 4 features
        data = {feat: np.random.rand(100) for feat in EXPECTED_UNIFIED_FEATURES}
        # Add labels (binary classification)
        data[self.label_col] = np.random.choice([0, 1], size=100)
        self.valid_df = pd.DataFrame(data)
        self.valid_df.to_csv(self.csv_path, index=False)

        # Write dummy manifest
        with open(self.manifest_path, "w") as f:
            json.dump({"source": "integration_test", "version": "1.0"}, f)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_successful_dataset_loading(self):
        df, stats = DatasetLoader.load_dataset(
            self.csv_path, self.label_col, self.manifest_path
        )

        self.assertEqual(len(df), 100)
        self.assertEqual(stats["total_samples"], 100)
        self.assertEqual(stats["feature_count"], len(EXPECTED_UNIFIED_FEATURES))
        self.assertIn("0", stats["class_distribution"])
        self.assertIn("1", stats["class_distribution"])
        self.assertEqual(stats["manifest_data"]["version"], "1.0")

    def test_missing_features_validation(self):
        bad_df = self.valid_df.drop(columns=[EXPECTED_UNIFIED_FEATURES[0]])
        bad_path = os.path.join(self.test_dir.name, "bad.csv")
        bad_df.to_csv(bad_path, index=False)

        with self.assertRaises(ValueError) as context:
            DatasetLoader.load_dataset(bad_path, self.label_col)
        self.assertIn("missing required feature columns", str(context.exception))

    def test_missing_values_detection(self):
        bad_df = self.valid_df.copy()
        # Inject NaN
        bad_df.loc[5, EXPECTED_UNIFIED_FEATURES[2]] = np.nan
        bad_path = os.path.join(self.test_dir.name, "nan.csv")
        bad_df.to_csv(bad_path, index=False)

        with self.assertRaises(ValueError) as context:
            DatasetLoader.load_dataset(bad_path, self.label_col)
        self.assertIn("missing values", str(context.exception))

    def test_invalid_label_variance(self):
        # Create dataset where all labels are the same (0)
        bad_df = self.valid_df.copy()
        bad_df[self.label_col] = 0
        bad_path = os.path.join(self.test_dir.name, "single_class.csv")
        bad_df.to_csv(bad_path, index=False)

        with self.assertRaises(ValueError) as context:
            DatasetLoader.load_dataset(bad_path, self.label_col)
        self.assertIn("fewer than 2 classes", str(context.exception))

    def test_duplicate_detection(self):
        # Append a duplicate row
        dup_df = pd.concat([self.valid_df, self.valid_df.iloc[[0]]], ignore_index=True)
        dup_path = os.path.join(self.test_dir.name, "dup.csv")
        dup_df.to_csv(dup_path, index=False)

        df, stats = DatasetLoader.load_dataset(dup_path, self.label_col)
        self.assertEqual(stats["duplicates_detected"], 1)


if __name__ == "__main__":
    unittest.main()
