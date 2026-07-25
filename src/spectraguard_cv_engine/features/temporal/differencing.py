"""Temporal frame differencing computations."""

import cv2
import numpy as np
from typing import List


class FrameDifferencing:
    """Computes pixel-wise disparities between sequential temporal matrices."""

    @staticmethod
    def compute_absolute_difference(
        frame1: np.ndarray, frame2: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the absolute pixel-wise difference between two frames.
        """
        if frame1.shape != frame2.shape:
            raise ValueError(
                "Frames must have identical spatial dimensions for differencing."
            )
        return cv2.absdiff(frame1, frame2)

    @staticmethod
    def compute_sequence_differences(frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Computes sequential differences over an entire temporal window.
        Returns a list of N-1 difference matrices for N frames.
        """
        if len(frames) < 2:
            raise ValueError(
                "Sequence must contain at least 2 frames to compute differences."
            )

        diffs = []
        for i in range(len(frames) - 1):
            diffs.append(
                FrameDifferencing.compute_absolute_difference(frames[i], frames[i + 1])
            )
        return diffs
