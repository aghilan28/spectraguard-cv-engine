"""Immutable data models for camera calibration parameters."""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraIntrinsics:
    """Core calibration payload containing focal, principal point, and distortion data."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
