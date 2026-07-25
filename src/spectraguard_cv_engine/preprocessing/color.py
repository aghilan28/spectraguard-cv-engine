"""Color space transformation foundation."""

import cv2
import numpy as np


class ColorConverter:
    """Handles deterministic color space conversions for spatial analysis."""

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Safely converts BGR/RGB to a single-channel grayscale matrix."""
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image

    @staticmethod
    def to_rgb(image: np.ndarray) -> np.ndarray:
        """Converts BGR to RGB for standard processing models."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
