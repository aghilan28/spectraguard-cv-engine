"""Validation suite for Spatial Feature operations."""

import unittest
import numpy as np
from src.spectraguard_cv_engine.features.spatial.gradients import GradientExtractor
from src.spectraguard_cv_engine.features.spatial.texture import TextureExtractor
from src.spectraguard_cv_engine.features.spatial.statistics import SpatialStatistics


class TestSpatialFeatures(unittest.TestCase):
    def setUp(self):
        # Create a basic 64x64 grayscale test pattern with a hard edge
        self.gray_image = np.zeros((64, 64), dtype=np.uint8)
        self.gray_image[32:, :] = 255

        # Color image for validation triggers
        self.color_image = np.zeros((64, 64, 3), dtype=np.uint8)

        # Uniform image for statistical edge cases
        self.uniform_image = np.full((64, 64), 128, dtype=np.uint8)

    def test_gradient_computation(self):
        mag, direc = GradientExtractor.compute_sobel_gradients(self.gray_image)
        self.assertEqual(mag.shape, self.gray_image.shape)
        self.assertEqual(direc.shape, self.gray_image.shape)

        # Check edge statistics
        stats = GradientExtractor.extract_edge_statistics(mag, threshold=10.0)
        self.assertIn("mean_magnitude", stats)
        self.assertIn("edge_density", stats)

        with self.assertRaises(ValueError):
            GradientExtractor.compute_sobel_gradients(self.color_image)

    def test_texture_descriptors(self):
        lap_var = TextureExtractor.compute_laplacian_variance(self.gray_image)
        self.assertIsInstance(lap_var, float)

        contrast = TextureExtractor.compute_global_contrast(self.gray_image)
        self.assertIsInstance(contrast, float)

        with self.assertRaises(ValueError):
            TextureExtractor.compute_laplacian_variance(self.color_image)

    def test_statistical_descriptors(self):
        # Test normal image
        stats = SpatialStatistics.extract_intensity_stats(self.gray_image)
        self.assertIsInstance(stats["mean_intensity"], float)
        self.assertIsInstance(stats["skewness"], float)
        self.assertEqual(stats["mean_intensity"], 127.5)  # Exactly half 0, half 255

        # Test zero-variance edge case
        uniform_stats = SpatialStatistics.extract_intensity_stats(self.uniform_image)
        self.assertEqual(uniform_stats["variance_intensity"], 0.0)
        self.assertEqual(uniform_stats["skewness"], 0.0)


if __name__ == "__main__":
    unittest.main()
