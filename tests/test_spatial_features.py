import unittest
import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.spatial import extract_spatial_features

class TestSpatialFeatures(unittest.TestCase):
    def setUp(self):
        # Create a sharp checkerboard image
        self.img_sharp = np.zeros((640, 640), dtype=np.uint8)
        self.img_sharp[::40, :] = 255
        self.img_sharp[:, ::40] = 255

        # Create a blurred image
        self.img_blur = cv2.GaussianBlur(self.img_sharp, (21, 21), 0)

    def test_laplacian_variance_drops_on_blur(self):
        lap_sharp, _, _ = extract_spatial_features(self.img_sharp)
        lap_blur, _, _ = extract_spatial_features(self.img_blur)
        self.assertGreater(lap_sharp, lap_blur)

    def test_sobel_edge_density_drops_on_blur(self):
        _, edge_sharp, _ = extract_spatial_features(self.img_sharp)
        _, edge_blur, _ = extract_spatial_features(self.img_blur)
        self.assertGreater(edge_sharp, edge_blur)

    def test_shannon_entropy(self):
        _, _, ent = extract_spatial_features(self.img_sharp)
        self.assertGreater(ent, 0.0)
        self.assertLessEqual(ent, 8.0)

if __name__ == "__main__":
    unittest.main()
