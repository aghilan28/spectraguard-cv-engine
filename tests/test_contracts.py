"""Validation suite for CV Contracts and Metadata constraints."""

import unittest
from src.spectraguard_cv_engine.contracts.enums import (
    PixelFormat,
    ColorSpace,
    ProcessingStatus,
    CVErrorCode,
)
from src.spectraguard_cv_engine.contracts.exceptions import CVEngineError
from src.spectraguard_cv_engine.contracts.models import (
    ImageResolution,
    ImageMetadata,
    FrameSequenceMetadata,
)


class TestCVContracts(unittest.TestCase):
    def test_resolution_bounds_validation(self):
        # Valid bounds
        valid_res = ImageResolution(1920, 1080)
        self.assertTrue(valid_res.is_valid)
        self.assertEqual(valid_res.dimensions, (1920, 1080))

        # Lower bound violation
        invalid_res_low = ImageResolution(320, 240)
        self.assertFalse(invalid_res_low.is_valid)

        # Upper bound violation
        invalid_res_high = ImageResolution(4000, 3000)
        self.assertFalse(invalid_res_high.is_valid)

    def test_image_metadata_immutability(self):
        res = ImageResolution(1280, 720)
        meta = ImageMetadata(
            frame_id="FRM_01",
            camera_id="CAM_01",
            timestamp_ns=1000000,
            resolution=res,
            channels=3,
            pixel_format=PixelFormat.RGB888,
            color_space=ColorSpace.RGB,
        )

        self.assertEqual(meta.channels, 3)
        self.assertEqual(meta.color_space.value, "RGB")

        # Ensure frozen status prevents modification
        with self.assertRaises(Exception):
            meta.channels = 1

    def test_frame_sequence_tracking(self):
        seq = FrameSequenceMetadata(
            sequence_id="SEQ_01",
            start_timestamp_ns=1000000,
            end_timestamp_ns=2000000,
            frame_count=2,
            expected_fps=30.0,
        )

        self.assertEqual(seq.status, ProcessingStatus.PENDING)
        self.assertFalse(seq.is_complete)

        res = ImageResolution(1920, 1080)
        meta1 = ImageMetadata(
            "F1", "C1", 1000000, res, 3, PixelFormat.RGB888, ColorSpace.RGB
        )
        meta2 = ImageMetadata(
            "F2", "C1", 1033333, res, 3, PixelFormat.RGB888, ColorSpace.RGB
        )

        seq.frames.extend([meta1, meta2])
        self.assertTrue(seq.is_complete)

    def test_exception_mapping(self):
        err = CVEngineError("Resolution out of bounds", CVErrorCode.INVALID_RESOLUTION)
        self.assertEqual(err.code, CVErrorCode.INVALID_RESOLUTION)
        self.assertIn("ERR_CV_001", str(err))


if __name__ == "__main__":
    unittest.main()
