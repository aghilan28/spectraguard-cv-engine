import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.fft import extract_fft_features, create_hanning_window

class TestFFTFeatures(unittest.TestCase):
    def setUp(self):
        self.img_sharp = np.random.randint(0, 256, (640, 640), dtype=np.uint8)

    def test_hanning_window_shape(self):
        window = create_hanning_window(640, 640)
        self.assertEqual(window.shape, (640, 640))
        self.assertGreaterEqual(np.min(window), 0.0)
        self.assertLessEqual(np.max(window), 1.0)

    def test_extract_fft_features_returns_ratios_summing_to_one(self):
        low, mid, high, log_energy = extract_fft_features(self.img_sharp)
        total_ratio = low + mid + high
        self.assertAlmostEqual(total_ratio, 1.0, places=4)
        self.assertGreater(log_energy, 0.0)

if __name__ == "__main__":
    unittest.main()
