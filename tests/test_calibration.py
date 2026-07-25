"""Validation suite for Calibration Foundation operations."""

import unittest
import numpy as np
from src.spectraguard_cv_engine.calibration.models import CameraIntrinsics
from src.spectraguard_cv_engine.calibration.loader import CalibrationLoader
from src.spectraguard_cv_engine.calibration.diagnostics import CalibrationDiagnostics
from src.spectraguard_cv_engine.calibration.pipeline import CalibrationPipeline


class TestCalibrationFoundation(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 1920x1080 camera intrinsic matrix
        self.valid_dict = {
            "camera_matrix": [
                [1500.0, 0.0, 960.0],
                [0.0, 1500.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": [[-0.2, 0.1, 0.0, 0.0, 0.0]],
        }

        # Invalid matrix (negative focal length)
        self.invalid_dict_focal = {
            "camera_matrix": [
                [-1500.0, 0.0, 960.0],
                [0.0, 1500.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": [[0.0, 0.0, 0.0, 0.0, 0.0]],
        }

        # Dummy image to test pipeline execution
        self.dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_calibration_loader(self):
        intrinsics = CalibrationLoader.from_dictionary(self.valid_dict)
        self.assertIsInstance(intrinsics, CameraIntrinsics)
        self.assertEqual(intrinsics.camera_matrix.shape, (3, 3))
        self.assertEqual(intrinsics.dist_coeffs.shape, (1, 5))

        with self.assertRaises(ValueError):
            CalibrationLoader.from_dictionary({"wrong_key": []})

    def test_calibration_diagnostics(self):
        valid_intrinsics = CalibrationLoader.from_dictionary(self.valid_dict)
        invalid_intrinsics = CalibrationLoader.from_dictionary(self.invalid_dict_focal)

        self.assertTrue(
            CalibrationDiagnostics.validate_intrinsics(valid_intrinsics, 1920, 1080)
        )
        self.assertFalse(
            CalibrationDiagnostics.validate_intrinsics(invalid_intrinsics, 1920, 1080)
        )

        # Test principal point out of bounds
        self.assertFalse(
            CalibrationDiagnostics.validate_intrinsics(valid_intrinsics, 800, 600)
        )

    def test_calibration_pipeline(self):
        intrinsics = CalibrationLoader.from_dictionary(self.valid_dict)
        undistorted = CalibrationPipeline.apply_undistortion(
            self.dummy_image, intrinsics
        )

        self.assertEqual(undistorted.shape, self.dummy_image.shape)

        # Should raise error for invalid dimensions
        with self.assertRaises(ValueError):
            CalibrationPipeline.apply_undistortion(np.array([1, 2, 3]), intrinsics)


if __name__ == "__main__":
    unittest.main()
