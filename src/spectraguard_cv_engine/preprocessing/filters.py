"""Noise reduction and spatial masking utilities."""

import cv2
import numpy as np


class NoiseReducer:
    """Applies kernel-based convolution filters for artifact suppression."""

    @staticmethod
    def apply_gaussian_blur(
        image: np.ndarray, kernel_size: tuple = (5, 5)
    ) -> np.ndarray:
        """Applies a standard Gaussian blur to suppress high-frequency noise."""
        return cv2.GaussianBlur(image, kernel_size, 0)


class RegionMask:
    """Manages binary occlusion masks for regions of interest."""

    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Isolates spatial data using a binary matrix."""
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "Mask dimensions must exactly match the image spatial dimensions."
            )
        return cv2.bitwise_and(image, image, mask=mask)
