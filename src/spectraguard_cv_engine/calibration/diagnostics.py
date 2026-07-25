"""Health checks and validation bounds for calibration integrity."""

from .models import CameraIntrinsics


class CalibrationDiagnostics:
    """Validates the mathematical sanity of intrinsic profiles."""

    @staticmethod
    def validate_intrinsics(
        intrinsics: CameraIntrinsics, img_width: int, img_height: int
    ) -> bool:
        """
        Verifies matrix dimensions, focal length positivity, and principal point bounds.
        """
        # 1. Check strict structural shapes
        if intrinsics.camera_matrix.shape != (3, 3):
            return False
        if intrinsics.dist_coeffs.ndim != 2 or intrinsics.dist_coeffs.shape[0] != 1:
            return False

        # 2. Validate focal lengths (fx, fy must be mathematically positive)
        fx = intrinsics.camera_matrix[0, 0]
        fy = intrinsics.camera_matrix[1, 1]
        if fx <= 0 or fy <= 0:
            return False

        # 3. Validate principal point (cx, cy must reside within the physical image boundary)
        cx = intrinsics.camera_matrix[0, 2]
        cy = intrinsics.camera_matrix[1, 2]
        if cx < 0 or cx > img_width or cy < 0 or cy > img_height:
            return False

        return True
