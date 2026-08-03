import numpy as np
import cv2
from typing import Tuple, Dict

def create_hanning_window(height: int, width: int) -> np.ndarray:
    hann_h = np.hanning(height)
    hann_w = np.hanning(width)
    return np.outer(hann_h, hann_w).astype(np.float32)

def extract_fft_features(gray_image: np.ndarray, radius_ratio: float = 0.05) -> Tuple[float, float, float, float]:
    """
    Computes 2D FFT with Hanning windowing and extracts 3 Band Ratios + Log Total Energy.
    Returns (fft_low_ratio, fft_mid_ratio, fft_high_ratio, log_total_energy).
    """
    h, w = gray_image.shape
    hann = create_hanning_window(h, w)
    windowed = gray_image.astype(np.float32) * hann

    dft = cv2.dft(windowed, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # Total spectral power
    total_energy = float(np.sum(magnitude))
    log_total_energy = float(np.log1p(total_energy))

    # Compute radial distance matrix
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    max_radius = np.sqrt(cx**2 + cy**2)

    # 3 Band Boundaries
    r_low = max_radius * 0.15
    r_mid = max_radius * 0.50

    mask_low = dist_from_center <= r_low
    mask_mid = (dist_from_center > r_low) & (dist_from_center <= r_mid)
    mask_high = dist_from_center > r_mid

    e_low = float(np.sum(magnitude[mask_low]))
    e_mid = float(np.sum(magnitude[mask_mid]))
    e_high = float(np.sum(magnitude[mask_high]))

    denom = total_energy + 1e-8
    fft_low_ratio = e_low / denom
    fft_mid_ratio = e_mid / denom
    fft_high_ratio = e_high / denom

    return fft_low_ratio, fft_mid_ratio, fft_high_ratio, log_total_energy
