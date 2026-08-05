"""
Metrics and constants configuration for the Baseline Comparison Engine.
Maintains strict weighting, safety bounds, and severity thresholds.
"""
from typing import Dict

# Epsilon to prevent division by zero in Z-score calculation
EPSILON: float = 1e-9

# Mathematical feature ordering and contribution weightings (MUST SUM TO 1.0)
FEATURE_WEIGHTS: Dict[str, float] = {
    "laplacian_variance": 0.20,
    "log_total_energy": 0.18,
    "edge_density": 0.15,
    "shannon_entropy": 0.14,
    "fft_low_ratio": 0.10,
    "fft_mid_ratio": 0.08,
    "temporal_difference": 0.08,
    "fft_high_ratio": 0.07
}

def get_severity(score: float) -> str:
    """
    Convert a continuous global drift score into a categorical severity boundary.
    
    Args:
        score (float): Global weighted drift score [0.0, 1.0].
        
    Returns:
        str: Severity classification.
    """
    if score < 0.20:
        return "VERY_LOW"
    if score < 0.40:
        return "LOW"
    if score < 0.60:
        return "MEDIUM"
    if score < 0.80:
        return "HIGH"
    return "CRITICAL"
