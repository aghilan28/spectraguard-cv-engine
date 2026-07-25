"""Global intensity statistical moments."""

import numpy as np
import scipy.stats
from typing import Dict


class SpatialStatistics:
    """Extracts high-order statistical moments from spatial intensity distributions."""

    @staticmethod
    def extract_intensity_stats(image: np.ndarray) -> Dict[str, float]:
        """
        Calculates mean, variance, skewness, and kurtosis of the image intensities.
        These moments describe the shape of the pixel intensity distribution.
        """
        flattened = image.flatten()

        # Calculate moments
        mean_val = float(np.mean(flattened))
        variance_val = float(np.var(flattened))

        # Scipy handles skew/kurtosis efficiently for large arrays
        # Add a small epsilon to variance check to prevent precision division errors in pure uniform arrays
        if variance_val < 1e-8:
            skewness_val = 0.0
            kurtosis_val = -3.0  # Fisher's kurtosis for uniform flat
        else:
            skewness_val = float(scipy.stats.skew(flattened))
            kurtosis_val = float(scipy.stats.kurtosis(flattened))

        return {
            "mean_intensity": mean_val,
            "variance_intensity": variance_val,
            "skewness": skewness_val,
            "kurtosis": kurtosis_val,
        }
