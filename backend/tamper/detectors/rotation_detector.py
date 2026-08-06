import numpy as np

class RotationDetector:
    def __init__(self, thresholds: dict):
        self.cfg = thresholds.get("rotation", {"angle_limit": 15.0})

    def evaluate(self, homography_matrix: np.ndarray) -> str:
        if homography_matrix is None or homography_matrix.shape != (3, 3):
            return "NORMAL"
            
        # Extract rotation angle: theta = atan2(M[1,0], M[0,0])
        theta = np.arctan2(homography_matrix[1, 0], homography_matrix[0, 0])
        angle_deg = abs(float(np.degrees(theta)))
        
        if angle_deg >= self.cfg["angle_limit"]:
            return "CAMERA_ROTATED"
            
        return "NORMAL"
