"""Validation suite for Unified Feature Aggregation."""

import unittest
import numpy as np
from src.spectraguard_cv_engine.features.unified.models import UnifiedFeatureVector
from src.spectraguard_cv_engine.features.unified.pipeline import (
    UnifiedExtractionPipeline,
)


class TestUnifiedFeatures(unittest.TestCase):
    def setUp(self):
        # Create synthetic valid resolution frames (e.g., 640x480)
        self.frame1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.frame2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_vector_serialization(self):
        vec = UnifiedFeatureVector(
            vector_id="VEC_01",
            timestamp_ns=1000,
            spatial_features={"mean_intensity": 100.0, "global_contrast": 50.0},
            frequency_features={"spectral_energy": 10.0},
            temporal_features={"mean_motion": 5.0},
        )

        arr = vec.to_array()
        # 9 spatial + 3 freq + 3 temp = 15 elements
        self.assertEqual(len(arr), 15)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.float32)

        # Check deterministic ordering (mean_intensity is index 0)
        self.assertEqual(arr[0], 100.0)

        d = vec.to_dict()
        self.assertEqual(d["dimensions"], 15)
        self.assertEqual(d["vector_id"], "VEC_01")

    def test_unified_pipeline_extraction(self):
        # Extract from a 2-frame sequence
        vector = UnifiedExtractionPipeline.extract_from_sequence(
            [self.frame1, self.frame2], vector_id="SEQ_TEST", timestamp_ns=12345
        )

        self.assertIsInstance(vector, UnifiedFeatureVector)

        arr = vector.to_array()
        self.assertEqual(len(arr), 15)

        # Validate values are populated
        self.assertGreater(vector.spatial_features["mean_intensity"], 0.0)
        self.assertGreater(vector.temporal_features["mean_motion"], 0.0)


if __name__ == "__main__":
    unittest.main()
