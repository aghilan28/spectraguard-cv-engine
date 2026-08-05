import cv2
from typing import Iterator, Tuple
import numpy as np

class FrameSampler:
    @staticmethod
    def sample(video_path: str, interval: int) -> Iterator[Tuple[int, np.ndarray]]:
        """Yields frames at the specified interval entirely in-memory."""
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                yield frame_idx, frame
            frame_idx += 1
        cap.release()
