"""Validation suite for dataset splitting and stratification."""

import unittest
import pandas as pd
import numpy as np
from src.spectraguard_cv_engine.ml.data.splitter import DatasetSplitter


class TestDatasetSplitter(unittest.TestCase):
    def setUp(self):
        # Create a heavily imbalanced dataset to strictly test stratification
        # 1000 rows: 800 class '0', 200 class '1' (80% / 20% split)
        np.random.seed(42)
        self.total_samples = 1000
        self.df = pd.DataFrame(
            {
                "feature_a": np.random.rand(self.total_samples),
                "feature_b": np.random.rand(self.total_samples),
                "label": [0] * 800 + [1] * 200,
            }
        )

    def test_split_ratios(self):
        # 70% Train / 15% Val / 15% Test
        train, val, test = DatasetSplitter.split_train_val_test(
            self.df, label_col="label", test_size=0.15, val_size=0.15, stratify=False
        )

        self.assertEqual(len(train), 700)
        self.assertEqual(len(val), 150)
        self.assertEqual(len(test), 150)

    def test_stratification_preservation(self):
        train, val, test = DatasetSplitter.split_train_val_test(
            self.df, label_col="label", test_size=0.2, val_size=0.1, stratify=True
        )

        # Verify that all splits contain roughly 20% of class '1'
        train_ratio = train["label"].mean()
        val_ratio = val["label"].mean()
        test_ratio = test["label"].mean()

        self.assertAlmostEqual(train_ratio, 0.2, places=2)
        self.assertAlmostEqual(val_ratio, 0.2, places=2)
        self.assertAlmostEqual(test_ratio, 0.2, places=2)

    def test_reproducibility_via_seed(self):
        # Splitting identically twice should yield bit-for-bit identical DataFrames
        t1, v1, x1 = DatasetSplitter.split_train_val_test(
            self.df, "label", random_state=99
        )
        t2, v2, x2 = DatasetSplitter.split_train_val_test(
            self.df, "label", random_state=99
        )

        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(v1, v2)
        pd.testing.assert_frame_equal(x1, x2)

    def test_invalid_ratio_bounds(self):
        with self.assertRaises(ValueError):
            # Sum >= 1.0 is mathematically invalid for a 3-way split
            DatasetSplitter.split_train_val_test(
                self.df, "label", test_size=0.6, val_size=0.5
            )

        with self.assertRaises(ValueError):
            # Zero or negative sizes are invalid
            DatasetSplitter.split_train_val_test(
                self.df, "label", test_size=0.0, val_size=0.2
            )

    def test_missing_label_column(self):
        with self.assertRaises(KeyError):
            DatasetSplitter.split_train_val_test(
                self.df, label_col="non_existent_column"
            )


if __name__ == "__main__":
    unittest.main()
