"""Validation suite for Frequency Domain operations."""

import unittest
import numpy as np
from src.spectraguard_cv_engine.features.frequency.transforms import (
    FrequencyTransformer,
)
from src.spectraguard_cv_engine.features.frequency.descriptors import (
    SpectralDescriptors,
)


class TestFrequencyDomain(unittest.TestCase):
    def setUp(self):
        # Create a basic 64x64 grayscale test pattern
        self.gray_image = np.zeros((64, 64), dtype=np.uint8)
        self.gray_image[16:48, 16:48] = 255

        # Color image to test validation
        self.color_image = np.zeros((64, 64, 3), dtype=np.uint8)

    def test_fft_computation(self):
        f_shift = FrequencyTransformer.compute_fft(self.gray_image)
        self.assertEqual(f_shift.shape, self.gray_image.shape)
        self.assertTrue(np.iscomplexobj(f_shift))

        with self.assertRaises(ValueError):
            FrequencyTransformer.compute_fft(self.color_image)

    def test_magnitude_spectrum(self):
        f_shift = FrequencyTransformer.compute_fft(self.gray_image)
        mag = FrequencyTransformer.compute_magnitude_spectrum(f_shift)
        self.assertEqual(mag.shape, self.gray_image.shape)
        self.assertFalse(np.iscomplexobj(mag))

    def test_dct_computation(self):
        dct_res = FrequencyTransformer.compute_dct(self.gray_image)
        self.assertEqual(dct_res.shape, self.gray_image.shape)

    def test_spectral_descriptors(self):
        f_shift = FrequencyTransformer.compute_fft(self.gray_image)
        mag = FrequencyTransformer.compute_magnitude_spectrum(f_shift)

        energy = SpectralDescriptors.calculate_spectral_energy(mag)
        self.assertIsInstance(energy, float)
        self.assertGreater(energy, 0.0)

        entropy = SpectralDescriptors.calculate_spectral_entropy(mag)
        self.assertIsInstance(entropy, float)
        self.assertGreater(entropy, 0.0)

        flatness = SpectralDescriptors.calculate_spectral_flatness(mag)
        self.assertIsInstance(flatness, float)


if __name__ == "__main__":
    unittest.main()
