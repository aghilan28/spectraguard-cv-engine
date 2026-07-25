"""Ingestion mechanisms for loading external camera calibration data."""

import numpy as np
from typing import Dict, Any
from .models import CameraIntrinsics


class CalibrationLoader:
    """Parses standard calibration outputs into engine-compliant immutable structures."""

    @staticmethod
    def from_dictionary(data: Dict[str, Any]) -> CameraIntrinsics:
        """
        Extracts intrinsic matrices and distortion coefficients from a dictionary payload.
        Expected keys: 'camera_matrix' (3x3 array), 'dist_coeffs' (1x5 or similar array).
        """
        if "camera_matrix" not in data or "dist_coeffs" not in data:
            raise ValueError(
                "Calibration dictionary must contain 'camera_matrix' and 'dist_coeffs' keys."
            )

        # Ensure uniform precision for OpenCV operations (CV_64F)
        mtx = np.array(data["camera_matrix"], dtype=np.float64)
        dist = np.array(data["dist_coeffs"], dtype=np.float64)

        return CameraIntrinsics(camera_matrix=mtx, dist_coeffs=dist)
