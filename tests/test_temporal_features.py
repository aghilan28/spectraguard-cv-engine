import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.temporal import extract_temporal_feature

class TestTemporalFeatures(unittest.TestCase):
    def setUp(self):
        self.frame1 = np.full((640, 640), 128, dtype=np.uint8)
        self.frame2_static = self.frame1.copy()
        self.frame2_dynamic = np.full((640, 640), 200, dtype=np.uint8)

    def test_static_frames_yield_zero_temporal_diff(self):
        diff = extract_temporal_feature([self.frame1, self.frame2_static])
        self.assertEqual(diff, 0.0)

    def test_dynamic_frames_yield_positive_temporal_diff(self):
        diff = extract_temporal_feature([self.frame1, self.frame2_dynamic])
        self.assertAlmostEqual(diff, 72.0, places=4)

if __name__ == "__main__":
    unittest.main()
