"""Pixel intensity and histogram normalization."""

import cv2
import numpy as np


class ImageNormalizer:
    """Manages pixel intensity scaling and contrast distribution."""

    @staticmethod
    def min_max_normalize(image: np.ndarray) -> np.ndarray:
        """Scales pixel intensities to a strict 0-255 bound."""
        normalized = cv2.normalize(
            image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        return normalized.astype(np.uint8)

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """Standardizes contrast distribution across the frame."""
        if len(image.shape) == 3:
            # Apply equalization only to the luminance channel to preserve color hues
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(image)
