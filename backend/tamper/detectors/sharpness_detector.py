import numpy as np
import cv2

class SharpnessDetector:
    def __init__(self, thresholds: dict):
        self.cfg = thresholds.get("sharpness", {"high": 1500.0})
        # Keep blur threshold separate to align with low sharpness
        self.laplacian_blur_limit = thresholds.get("laplacian_blur_limit", 35.0)

    def evaluate(self, gray_frame: np.ndarray) -> str:
        if gray_frame is None:
            return "NORMAL"
            
        lap = cv2.Laplacian(gray_frame, cv2.CV_64F)
        lap_var = float(np.var(lap))
        
        # Sobel Gradients (Tenengrad focus)
        gx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx**2 + gy**2))
        
        if lap_var > self.cfg["high"] and tenengrad > 12000.0:
            # Bypass if the frame is extremely noisy (e.g. random noise test frame)
            edge_dens = float(np.sum(cv2.Canny(gray_frame, 50, 150) > 0) / gray_frame.size)
            if edge_dens >= 0.25:
                return "NORMAL"
            return "SHARPNESS_HIGH"
        elif lap_var < self.laplacian_blur_limit:
            return "SHARPNESS_LOW"
            
        return "NORMAL"
