"""
Pure mathematical execution module for drift calculations.
Optimized for <2ms execution using NumPy array vectorization.
"""
import numpy as np
from typing import Dict, Any, Tuple, List
from backend.calibration.baseline_metrics import FEATURE_WEIGHTS, EPSILON, get_severity

def compute_drift_vector(
    live_features: Dict[str, float],
    baseline_stats: Dict[str, Dict[str, float]]
) -> Tuple[float, str, List[Dict[str, Any]]]:
    """
    Computes mathematical deviation for a live vector against an environmental baseline.
    
    Args:
        live_features (Dict[str, float]): The incoming 8D physical feature vector.
        baseline_stats (Dict[str, Dict[str, float]]): The baseline statistics matrix.
        
    Returns:
        Tuple containing:
            - global_score (float): The final weighted anomaly score [0, 1].
            - severity (str): The categorical alert level.
            - features (List[Dict]): Detailed per-feature math breakdown.
    """
    ordered_keys = list(FEATURE_WEIGHTS.keys())
    
    # Pre-allocate numpy arrays for vectorized mathematical operations
    live_arr = np.zeros(len(ordered_keys), dtype=np.float64)
    mean_arr = np.zeros(len(ordered_keys), dtype=np.float64)
    std_arr = np.zeros(len(ordered_keys), dtype=np.float64)
    weight_arr = np.zeros(len(ordered_keys), dtype=np.float64)
    
    for i, key in enumerate(ordered_keys):
        live_arr[i] = live_features.get(key, 0.0)
        mean_arr[i] = baseline_stats.get(key, {}).get("mean", 0.0)
        std_arr[i] = baseline_stats.get(key, {}).get("std", 1.0)
        weight_arr[i] = FEATURE_WEIGHTS[key]
        
    # Phase 1: Standardized Z-Score with Zero-Division Protection
    safe_std = np.maximum(std_arr, EPSILON)
    z_scores = (live_arr - mean_arr) / safe_std
    
    # Phase 2: Absolute Z-Score
    abs_z = np.abs(z_scores)
    
    # Phase 3: Normalized Drift (capped at 5.0 standard deviations)
    normalized_drift = np.minimum(abs_z / 5.0, 1.0)
    
    # Phase 4: Weighted Contributions
    weighted_scores = normalized_drift * weight_arr
    
    # Phase 5: Global Score
    global_score = float(np.sum(weighted_scores))
    severity = get_severity(global_score)
    
    # Reconstruct readable dictionary map for API consumers
    feature_breakdown = []
    for i, key in enumerate(ordered_keys):
        feature_breakdown.append({
            "name": key,
            "live": float(live_arr[i]),
            "mean": float(mean_arr[i]),
            "std": float(std_arr[i]),
            "z_score": float(z_scores[i]),
            "absolute_z": float(abs_z[i]),
            "normalized_drift": float(normalized_drift[i]),
            "weight": float(weight_arr[i]),
            "weighted_score": float(weighted_scores[i])
        })
        
    return global_score, severity, feature_breakdown
