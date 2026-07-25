"""Immutable data models for validating image frames and temporal sequences."""

from dataclasses import dataclass, field
from typing import List, Tuple
from .constants import SystemLimits
from .enums import PixelFormat, ColorSpace, ProcessingStatus


@dataclass(frozen=True)
class ImageResolution:
    """Represents and validates 2D spatial dimensions."""

    width: int
    height: int

    @property
    def is_valid(self) -> bool:
        """Evaluates if dimensions fall within architectural safety bounds."""
        return (
            SystemLimits.MIN_RESOLUTION_WIDTH
            <= self.width
            <= SystemLimits.MAX_RESOLUTION_WIDTH
            and SystemLimits.MIN_RESOLUTION_HEIGHT
            <= self.height
            <= SystemLimits.MAX_RESOLUTION_HEIGHT
        )

    @property
    def dimensions(self) -> Tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class ImageMetadata:
    """Core descriptive model for a single validated visual frame."""

    frame_id: str
    camera_id: str
    timestamp_ns: int
    resolution: ImageResolution
    channels: int
    pixel_format: PixelFormat
    color_space: ColorSpace


@dataclass
class FrameSequenceMetadata:
    """Temporal tracking model mapping sequence blocks for time-series extraction."""

    sequence_id: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    frame_count: int
    expected_fps: float
    status: ProcessingStatus = ProcessingStatus.PENDING
    frames: List[ImageMetadata] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Verifies if the sequence array matches the expected structural length."""
        return len(self.frames) == self.frame_count
