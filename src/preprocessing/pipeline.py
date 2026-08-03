import os
import json
import numpy as np
import cv2
from typing import List, Union, Dict, Any

from .features import FeatureVector
from .fft import extract_fft_features
from .spatial import extract_spatial_features
from .temporal import extract_temporal_feature

def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "pipeline_config.json")
    with open(config_path, "r") as f:
        return json.load(f)

class PreprocessingPipeline:
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = load_config()
        self.config = config

        self.width = config["resize"]["width"]
        self.height = config["resize"]["height"]
        self.clahe_clip = config["clahe"]["clip_limit"]
        self.clahe_grid = tuple(config["clahe"]["tile_grid_size"])
        self.radius_ratio = config["fft"]["highpass_radius_ratio"]
        self.sobel_thresh = config["sobel"]["threshold"]

        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocesses a single frame:
        1. Resize to fixed (640, 640)
        2. Convert BGR/RGB to Grayscale (ITU-R BT.601 Y Luminance)
        3. Apply CLAHE normalization
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        normalized = self.clahe.apply(resized)
        return normalized

    def extract(self, frames: Union[np.ndarray, List[np.ndarray]]) -> FeatureVector:
        """
        Main API Entry Point:
        Accepts a single frame or a list of rolling frames.
        Returns a structured 8D FeatureVector.
        """
        if isinstance(frames, np.ndarray) and frames.ndim in [2, 3]:
            frames_list = [frames]
        elif isinstance(frames, list):
            frames_list = frames
        else:
            raise ValueError("Input to PreprocessingPipeline.extract must be a frame or list of frames.")

        # Preprocess all frames in rolling window
        processed_frames = [self.preprocess_frame(f) for f in frames_list]

        # Target frame for spatial/FFT is the latest frame
        target_frame = processed_frames[-1]

        # 1. FFT Features
        fft_low, fft_mid, fft_high, log_energy = extract_fft_features(target_frame, self.radius_ratio)

        # 2. Spatial Features
        lap_var, edge_dens, entropy = extract_spatial_features(target_frame, self.sobel_thresh)

        # 3. Temporal Feature
        temp_diff = extract_temporal_feature(processed_frames)

        return FeatureVector(
            fft_low_ratio=fft_low,
            fft_mid_ratio=fft_mid,
            fft_high_ratio=fft_high,
            log_total_energy=log_energy,
            laplacian_variance=lap_var,
            edge_density=edge_dens,
            shannon_entropy=entropy,
            temporal_difference=temp_diff
        )
