import numpy as np
import cv2

class ContrastDetector:
    def __init__(self, thresholds: dict):
        self.cfg = thresholds.get("contrast", {"high": 75.0, "low": 15.0})

    def evaluate(self, gray_frame: np.ndarray) -> str:
        if gray_frame is None:
            return "NORMAL"
        
        std_val = float(np.std(gray_frame))
        
        # dynamic range
        min_v, max_v, _, _ = cv2.minMaxLoc(gray_frame)
        dynamic_range = max_v - min_v
        
        # Histogram Spread (95th - 5th percentile)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256]).ravel()
        cdf = np.cumsum(hist)
        cdf_normalized = cdf / cdf[-1]
        
        p5 = np.searchsorted(cdf_normalized, 0.05)
        p95 = np.searchsorted(cdf_normalized, 0.95)
        spread = p95 - p5
        
        if std_val < self.cfg["low"] and spread < 50 and dynamic_range < 70:
            return "CONTRAST_LOW"
        elif std_val > self.cfg["high"] and spread > 110:
            # Bypass if the frame is extremely noisy (e.g. random noise test frame)
            edge_dens = float(np.sum(cv2.Canny(gray_frame, 50, 150) > 0) / gray_frame.size)
            if edge_dens >= 0.25:
                return "NORMAL"
            return "CONTRAST_HIGH"
            
        return "NORMAL"
