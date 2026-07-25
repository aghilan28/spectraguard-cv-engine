"""Motion statistics and temporal stability descriptors."""

import numpy as np
from typing import List, Dict


class MotionStatistics:
    """Extracts numerical descriptions of motion from difference sequences."""

    @staticmethod
    def extract_motion_features(differences: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculates global motion features across a pre-computed list of sequence differences.
        """
        if not differences:
            return {
                "mean_motion": 0.0,
                "motion_variance": 0.0,
                "temporal_instability": 0.0,
            }

        # Calculate the mean intensity of each difference frame (motion magnitude)
        mean_diffs = [np.mean(d) for d in differences]

        # Overall sequence motion average
        overall_mean_motion = float(np.mean(mean_diffs))

        # Variance of the motion over the sequence
        overall_motion_variance = float(np.var(mean_diffs))

        # Standard deviation of the motion signifies temporal instability (jitter/fluctuation)
        temporal_instability = float(np.std(mean_diffs))

        return {
            "mean_motion": overall_mean_motion,
            "motion_variance": overall_motion_variance,
            "temporal_instability": temporal_instability,
        }
