"""Orchestrator for standardized frame preprocessing workflows."""

import numpy as np
from .loader import ImageValidator
from .color import ColorConverter
from .normalization import ImageNormalizer
from .filters import NoiseReducer


class PreprocessingPipeline:
    """Executes a deterministic sequence of transformations on raw incoming frames."""

    @staticmethod
    def process_standard_spatial_frame(image: np.ndarray) -> np.ndarray:
        """
        Executes standard sequential cleaning:
        Validate -> Denoise -> Grayscale -> Normalize Contrast
        """
        # 1. Enforce architectural bounds
        ImageValidator.validate_resolution(image)

        # 2. Suppress high-frequency sensor noise
        denoised = NoiseReducer.apply_gaussian_blur(image)

        # 3. Reduce dimensionality to spatial intensity
        gray = ColorConverter.to_grayscale(denoised)

        # 4. Maximize contrast definition
        normalized = ImageNormalizer.histogram_equalization(gray)

        return normalized
