"""Spectral feature calculation from frequency domain data."""

import numpy as np
import scipy.stats


class SpectralDescriptors:
    """Extracts numerical features from frequency transforms."""

    @staticmethod
    def calculate_spectral_energy(magnitude_spectrum: np.ndarray) -> float:
        """Calculates total spectral energy (sum of squared magnitudes)."""
        return float(np.sum(np.square(magnitude_spectrum)))

    @staticmethod
    def calculate_spectral_entropy(magnitude_spectrum: np.ndarray) -> float:
        """Calculates Shannon entropy of the normalized magnitude spectrum."""
        # Flatten and normalize to create a probability distribution
        flattened = magnitude_spectrum.flatten()
        # Ensure positive values and normalize
        pos_spectrum = np.abs(flattened)
        total = np.sum(pos_spectrum)

        if total == 0:
            return 0.0

        p_data = pos_spectrum / total
        # Calculate entropy (base 2)
        entropy = scipy.stats.entropy(p_data, base=2)
        return float(entropy)

    @staticmethod
    def calculate_spectral_flatness(magnitude_spectrum: np.ndarray) -> float:
        """Calculates spectral flatness (Wiener entropy)."""
        flattened = np.abs(magnitude_spectrum.flatten()) + 1e-8
        geometric_mean = np.exp(np.mean(np.log(flattened)))
        arithmetic_mean = np.mean(flattened)

        if arithmetic_mean == 0:
            return 0.0

        return float(geometric_mean / arithmetic_mean)
