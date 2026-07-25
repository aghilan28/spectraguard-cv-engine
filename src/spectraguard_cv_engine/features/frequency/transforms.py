"""Frequency domain transformation algorithms."""

import cv2
import numpy as np


class FrequencyTransformer:
    """Handles orthogonal transformations from spatial to frequency domains."""

    @staticmethod
    def compute_fft(image: np.ndarray) -> np.ndarray:
        """Computes the 2D Fast Fourier Transform. Expects a grayscale image."""
        if len(image.shape) != 2:
            raise ValueError("FFT requires a 2D grayscale image.")

        # Compute FFT and shift zero-frequency component to the center
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        return f_shift

    @staticmethod
    def compute_magnitude_spectrum(f_shift: np.ndarray) -> np.ndarray:
        """Calculates the logarithmic magnitude spectrum from shifted FFT data."""
        # Use log scale to visualize/process wide dynamic range
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
        return magnitude_spectrum

    @staticmethod
    def compute_dct(image: np.ndarray) -> np.ndarray:
        """Computes the 2D Discrete Cosine Transform. Must be float32 grayscale."""
        if len(image.shape) != 2:
            raise ValueError("DCT requires a 2D grayscale image.")

        img_float32 = np.float32(image)
        dct_result = cv2.dct(img_float32)
        return dct_result
