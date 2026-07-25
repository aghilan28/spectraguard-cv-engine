"""Enumerated types for image parameters and processing state."""

from enum import Enum


class PixelFormat(Enum):
    RGB888 = "RGB888"
    BGR888 = "BGR888"
    GRAY8 = "GRAY8"


class ColorSpace(Enum):
    RGB = "RGB"
    HSV = "HSV"
    LAB = "LAB"
    GRAYSCALE = "GRAYSCALE"


class ProcessingStatus(Enum):
    PENDING = "PENDING"
    PREPROCESSING = "PREPROCESSING"
    EXTRACTING_FEATURES = "EXTRACTING_FEATURES"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CVErrorCode(Enum):
    INVALID_RESOLUTION = "ERR_CV_001"
    UNSUPPORTED_FORMAT = "ERR_CV_002"
    CORRUPT_FRAME_DATA = "ERR_CV_003"
    SEQUENCE_MISMATCH = "ERR_CV_004"
    CALIBRATION_MISSING = "ERR_CV_005"
    PIPELINE_FAILURE = "ERR_CV_006"
