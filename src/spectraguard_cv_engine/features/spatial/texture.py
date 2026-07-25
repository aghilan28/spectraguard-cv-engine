"""Texture and structural descriptor extraction."""

import cv2
import numpy as np


class TextureExtractor:
    """Computes spatial texture characteristics and structural sharpness."""

    @staticmethod
    def compute_laplacian_variance(image: np.ndarray) -> float:
        """
        Computes the variance of the Laplacian to measure global texture/sharpness.
        Low values typically indicate blur; high values indicate rich texture.
        """
        if len(image.shape) != 2:
            raise ValueError("Laplacian computation requires a 2D grayscale image.")

        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return float(np.var(laplacian))

    @staticmethod
    def compute_global_contrast(image: np.ndarray) -> float:
        """
        Computes global contrast via the standard deviation of pixel intensities.
        """
        return float(np.std(image))
