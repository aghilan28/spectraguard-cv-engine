import numpy as np
from typing import Dict
# Note: Assuming existing feature extraction is accessible via a core module.
# Replace 'existing_module' with the actual path to SpectraGuard v1/v2 feature extractor.
try:
    from existing_module.features import extract_all_features
except ImportError:
    # Fallback mock for pipeline validation without modifying existing backend
    def extract_all_features(frame: np.ndarray, prev_frame: np.ndarray = None) -> Dict:
        return {
            "fft_low_ratio": 0.0, "fft_mid_ratio": 0.0, "fft_high_ratio": 0.0,
            "log_total_energy": 0.0, "laplacian_variance": 0.0, "edge_density": 0.0,
            "shannon_entropy": 0.0, "temporal_difference": 0.0
        }

class FeaturePipeline:
    def __init__(self):
        self.prev_frame = None

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Routes frame through existing feature extraction without modifying it."""
        features = extract_all_features(frame, self.prev_frame)
        self.prev_frame = frame
        return features
