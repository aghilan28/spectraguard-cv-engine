import numpy as np
import cv2
from typing import Tuple

def extract_spatial_features(gray_image: np.ndarray, sobel_threshold: float = 100.0) -> Tuple[float, float, float]:
    """
    Computes spatial features:
    1. Laplacian Variance (Focus/Blur measure)
    2. Sobel Edge Density (Occlusion/Blockage measure)
    3. Spatial Shannon Entropy (Information density measure)
    Returns (laplacian_variance, edge_density, shannon_entropy).
    """
    # 1. Laplacian Variance
    lap = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian_variance = float(np.var(lap))

    # 2. Sobel Edge Density
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_density = float(np.mean(magnitude > sobel_threshold))

    # 3. Spatial Shannon Entropy
    hist, _ = np.histogram(gray_image.ravel(), bins=256, range=(0, 256))
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    shannon_entropy = float(-np.sum(prob * np.log2(prob + 1e-12)))

    return laplacian_variance, edge_density, shannon_entropy
