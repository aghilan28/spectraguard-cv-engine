"""Image loading and dimensional validation engine."""

import numpy as np
from ..contracts.models import ImageResolution
from ..contracts.exceptions import CVEngineError
from ..contracts.enums import CVErrorCode


class ImageValidator:
    """Validates incoming numpy arrays against strict architectural CV bounds."""

    @staticmethod
    def validate_resolution(image: np.ndarray) -> None:
        """Verifies if the image dimensions fall within the defined SystemLimits."""
        if len(image.shape) < 2:
            raise CVEngineError(
                "Invalid image dimensions or empty array.",
                CVErrorCode.CORRUPT_FRAME_DATA,
            )

        h, w = image.shape[:2]
        res = ImageResolution(width=w, height=h)

        if not res.is_valid:
            raise CVEngineError(
                f"Resolution {w}x{h} violates operational bounds.",
                CVErrorCode.INVALID_RESOLUTION,
            )
