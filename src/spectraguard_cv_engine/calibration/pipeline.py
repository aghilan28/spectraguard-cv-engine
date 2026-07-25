"""Runtime application of calibration matrices for spatial normalization."""

import cv2
import numpy as np
from .models import CameraIntrinsics


class CalibrationPipeline:
    """Executes geometric undistortion on spatial frames."""

    @staticmethod
    def apply_undistortion(
        image: np.ndarray, intrinsics: CameraIntrinsics
    ) -> np.ndarray:
        """
        Corrects radial and tangential lens distortion using the camera intrinsics.
        Returns the geometrically normalized array.
        """
        if len(image.shape) < 2:
            raise ValueError("Invalid image dimensions for undistortion.")

        undistorted = cv2.undistort(
            image, intrinsics.camera_matrix, intrinsics.dist_coeffs
        )
        return undistorted
