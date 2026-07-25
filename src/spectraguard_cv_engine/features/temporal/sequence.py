"""Temporal window management for frame sequences."""

import numpy as np
from collections import deque
from typing import List


class TemporalWindow:
    """Manages a rolling window of frames for temporal sequence extraction."""

    def __init__(self, window_size: int = 5):
        """
        Initializes the temporal window.

        Args:
            window_size: The strict maximum capacity of the temporal sequence.
        """
        if window_size < 2:
            raise ValueError("Temporal window size must be at least 2.")
        self.window_size = window_size
        self.frames = deque(maxlen=window_size)

    def add_frame(self, frame: np.ndarray) -> None:
        """Appends a frame to the temporal sequence, automatically ejecting oldest if full."""
        self.frames.append(frame)

    @property
    def is_full(self) -> bool:
        """Returns True if the window has reached its tracking capacity."""
        return len(self.frames) == self.window_size

    @property
    def frame_count(self) -> int:
        """Returns the current number of tracked frames."""
        return len(self.frames)

    def get_window(self) -> List[np.ndarray]:
        """Returns a snapshot list of the current frames in the window sequence."""
        return list(self.frames)

    def clear(self) -> None:
        """Purges the temporal window."""
        self.frames.clear()
