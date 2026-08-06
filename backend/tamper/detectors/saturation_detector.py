import numpy as np
import cv2

class SaturationDetector:
    def __init__(self, thresholds: dict):
        self.cfg = thresholds.get("saturation", {"high": 200.0, "low": 15.0})

    def evaluate(self, bgr_frame: np.ndarray) -> str:
        if bgr_frame is None or len(bgr_frame.shape) < 3:
            return "NORMAL"
            
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1]
        
        mean_sat = float(np.mean(sat_channel))
        
        if mean_sat < self.cfg["low"]:
            return "SATURATION_LOW"
        elif mean_sat > self.cfg["high"]:
            return "SATURATION_HIGH"
            
        return "NORMAL"
