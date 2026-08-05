"""
Deterministic rule evaluator for physical camera tamper events based on Z-Score drift matrices.
Executes purely numerical matrix evaluations against incoming deviation invariants.
"""
from typing import Dict, Tuple, Any

TAMPER_TYPES = [
    "NORMAL", "LENS_COVER", "LENS_SPRAY", "DEFOCUS", "CAMERA_MOVED",
    "FLASH_ATTACK", "DARKNESS", "OVEREXPOSURE", "VIDEO_FREEZE",
    "HEAVY_NOISE", "PARTIAL_OCCLUSION", "UNKNOWN_ANOMALY"
]

def decrease(z: float) -> float:
    """Measures magnitude of negative drift (drop in value). Capped at 1.0 (5 std devs)."""
    return max(0.0, -z) / 5.0 if z < 0 else 0.0

def increase(z: float) -> float:
    """Measures magnitude of positive drift (spike in value). Capped at 1.0 (5 std devs)."""
    return max(0.0, z) / 5.0 if z > 0 else 0.0

def anomaly(z: float) -> float:
    """Measures absolute magnitude of drift regardless of direction."""
    return min(abs(z) / 5.0, 1.0)

def evaluate_rules(feature_reports: list) -> Tuple[str, Dict[str, float]]:
    """
    Evaluates physical feature deviations against deterministic anomaly signatures.
    
    Args:
        feature_reports: List of feature deviation structures from the Deviation Engine.
        
    Returns:
        Tuple containing the winning tamper classification string and a dict of computed rule scores.
    """
    # Extract Z-scores into a fast O(1) lookup dictionary
    z_map = {f.feature: f.z_score for f in feature_reports}
    
    # Safe getters for features
    energy = z_map.get("log_total_energy", 0.0)
    entropy = z_map.get("shannon_entropy", 0.0)
    edge = z_map.get("edge_density", 0.0)
    laplacian = z_map.get("laplacian_variance", 0.0)
    temporal = z_map.get("temporal_difference", 0.0)
    fft_high = z_map.get("fft_high_ratio", 0.0)
    fft_mid = z_map.get("fft_mid_ratio", 0.0)
    fft_low = z_map.get("fft_low_ratio", 0.0)

    # Deterministic Rule Definitions
    scores = {
        "LENS_COVER": (decrease(energy) + decrease(entropy) + decrease(edge)) / 3.0,
        "LENS_SPRAY": (decrease(laplacian) + decrease(edge) + increase(entropy)) / 3.0,
        "DEFOCUS": (decrease(laplacian) + decrease(fft_high)) / 2.0,
        "CAMERA_MOVED": (increase(temporal) + anomaly(edge)) / 2.0,
        "FLASH_ATTACK": (increase(energy) + decrease(entropy)) / 2.0,
        "DARKNESS": (decrease(energy) + decrease(fft_low)) / 2.0,
        "OVEREXPOSURE": (increase(energy) + increase(fft_low)) / 2.0,
        "VIDEO_FREEZE": decrease(temporal), # Mean temporal variance dropping to zero creates negative Z
        "HEAVY_NOISE": (increase(fft_high) + increase(entropy)) / 2.0,
        "PARTIAL_OCCLUSION": (decrease(edge) + decrease(fft_mid)) / 2.0
    }

    # Clamp all calculated scores rigidly to bounds [0.0, 1.0]
    for key in scores:
        scores[key] = min(max(scores[key], 0.0), 1.0)

    # Sort rules to identify the primary triggered anomaly signature
    sorted_rules = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_rule, best_score = sorted_rules[0]
    second_best_score = sorted_rules[1][1]

    # Tie-breaking logic: If signals are mixed/conflicting but strong, it's an unknown complex anomaly
    if best_score > 0.3 and (best_score - second_best_score) < 0.10:
        winning_rule = "UNKNOWN_ANOMALY"
    elif best_score >= 0.25:
        winning_rule = best_rule
    else:
        # If no rule heavily triggers, default to NORMAL
        winning_rule = "NORMAL"

    return winning_rule, scores
