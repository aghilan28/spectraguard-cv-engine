"""Validation suite for CV Preprocessing operations."""

import unittest
import numpy as np
from src.spectraguard_cv_engine.preprocessing.loader import ImageValidator
from src.spectraguard_cv_engine.preprocessing.color import ColorConverter
from src.spectraguard_cv_engine.preprocessing.normalization import ImageNormalizer
from src.spectraguard_cv_engine.preprocessing.filters import NoiseReducer, RegionMask
from src.spectraguard_cv_engine.preprocessing.pipeline import PreprocessingPipeline
from src.spectraguard_cv_engine.contracts.exceptions import CVEngineError


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a valid synthetic BGR frame (1920x1080)
        self.valid_image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        # Create an invalid synthetic frame out of bounds (320x240)
        self.invalid_image = np.zeros((240, 320, 3), dtype=np.uint8)

    def test_image_validation(self):
        # Must not raise an exception
        ImageValidator.validate_resolution(self.valid_image)

        # Must strictly raise resolution limit error
        with self.assertRaises(CVEngineError) as context:
            ImageValidator.validate_resolution(self.invalid_image)
        self.assertEqual(context.exception.code.value, "ERR_CV_001")

    def test_color_conversion(self):
        gray = ColorConverter.to_grayscale(self.valid_image)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (1080, 1920))

    def test_image_normalization(self):
        norm = ImageNormalizer.min_max_normalize(self.valid_image)
        self.assertEqual(norm.shape, self.valid_image.shape)
        self.assertLessEqual(np.max(norm), 255)

    def test_noise_reduction_and_masking(self):
        blur = NoiseReducer.apply_gaussian_blur(self.valid_image)
        self.assertEqual(blur.shape, self.valid_image.shape)

        # Test exact mask dimensional matching
        mask = np.zeros((1080, 1920), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        masked_img = RegionMask.apply_mask(self.valid_image, mask)
        self.assertEqual(masked_img.shape, self.valid_image.shape)

    def test_pipeline_orchestrator(self):
        processed = PreprocessingPipeline.process_standard_spatial_frame(
            self.valid_image
        )
        # Verify it passed validation, became grayscale, and maintained dimensions
        self.assertEqual(len(processed.shape), 2)
        self.assertEqual(processed.shape, (1080, 1920))


if __name__ == "__main__":
    unittest.main()
