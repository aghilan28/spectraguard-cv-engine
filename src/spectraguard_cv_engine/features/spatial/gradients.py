"""Spatial gradient and edge descriptor extraction."""

import cv2
import numpy as np
from typing import Tuple, Dict


class GradientExtractor:
    """Computes edge intensities and directional gradients from spatial matrices."""

    @staticmethod
    def compute_sobel_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes 1st-order image derivatives (Sobel) in X and Y.
        Returns gradient magnitude and phase (direction in degrees).
        Expects a 2D grayscale image.
        """
        if len(image.shape) != 2:
            raise ValueError("Gradient computation requires a 2D grayscale image.")

        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = cv2.magnitude(grad_x, grad_y)
        direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)

        return magnitude, direction

    @staticmethod
    def extract_edge_statistics(
        magnitude: np.ndarray, threshold: float = 50.0
    ) -> Dict[str, float]:
        """
        Extracts structural numerical features from the gradient magnitude matrix.
        """
        mean_mag = float(np.mean(magnitude))
        max_mag = float(np.max(magnitude))

        # Calculate the density of strong edges
        edge_pixels = np.sum(magnitude > threshold)
        edge_density = (
            float(edge_pixels / magnitude.size) if magnitude.size > 0 else 0.0
        )

        return {
            "mean_magnitude": mean_mag,
            "max_magnitude": max_mag,
            "edge_density": edge_density,
        }
