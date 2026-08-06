import numpy as np
import cv2

class ColorDetector:
    def __init__(self, thresholds: dict):
        self.cfg = thresholds.get("color", {"distortion_limit": 25.0})

    def evaluate(self, bgr_frame: np.ndarray) -> str:
        if bgr_frame is None or len(bgr_frame.shape) < 3:
            return "NORMAL"
            
        # Bypass color cast validation if image has extreme saturation
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        if mean_sat >= 180.0:
            return "NORMAL"

        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        a_chan = lab[:, :, 1]
        b_chan = lab[:, :, 2]
        
        # Neutral color balance value is 128 in 8-bit LAB space
        a_dev = abs(float(np.mean(a_chan)) - 128.0)
        b_dev = abs(float(np.mean(b_chan)) - 128.0)
        
        limit = self.cfg["distortion_limit"]
        if a_dev > limit or b_dev > limit:
            return "COLOR_DISTORTION"
            
        return "NORMAL"
