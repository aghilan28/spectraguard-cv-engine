import numpy as np
from typing import List

def extract_temporal_feature(gray_frames: List[np.ndarray]) -> float:
    r"""
    Computes the Mean Absolute Inter-Frame Luminance Difference over N rolling frames:
    \Delta I_{temp} = \frac{1}{N-1} \sum_{k=1}^{N-1} \left( \frac{1}{H \cdot W} \sum_{x,y} |Y_k(x,y) - Y_{k-1}(x,y)| \right)

    If gray_frames has fewer than 2 frames, returns 0.0.
    For static/frozen frames, \Delta I_{temp} -> 0.0.
    """
    if len(gray_frames) < 2:
        return 0.0

    diffs = []
    for k in range(1, len(gray_frames)):
        frame_diff = np.abs(gray_frames[k].astype(np.float32) - gray_frames[k-1].astype(np.float32))
        diffs.append(float(np.mean(frame_diff)))

    return float(np.mean(diffs))
