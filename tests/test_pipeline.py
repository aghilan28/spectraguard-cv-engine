import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.features import FeatureVector

class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = PreprocessingPipeline()
        self.dummy_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

    def test_pipeline_extract_returns_feature_vector(self):
        vec = self.pipeline.extract(self.dummy_frame)
        self.assertIsInstance(vec, FeatureVector)
        self.assertEqual(len(vec.to_numpy()), 8)
        self.assertEqual(len(FeatureVector.feature_names()), 8)

    def test_pipeline_extract_rolling_window(self):
        frames = [self.dummy_frame for _ in range(5)]
        vec = self.pipeline.extract(frames)
        self.assertIsInstance(vec, FeatureVector)
        self.assertEqual(vec.temporal_difference, 0.0)

if __name__ == "__main__":
    unittest.main()
