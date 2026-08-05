import numpy as np
from typing import Dict, Any, List
from backend.models.deviation_result import FeatureDeviationModel

def calculate_zscores(
    live_vector: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
    ordered_keys: List[str]
) -> List[FeatureDeviationModel]:
    """Vectorized calculation of localized Z-scores and standardized drift constraints."""
    eps = 1e-9
    
    # Vectorized execution pathways mapping Z scores
    z_scores = (live_vector - means) / (stds + eps)
    clamped_z = np.clip(z_scores, -5.0, 5.0)
    normalized_drift = np.minimum(np.abs(clamped_z) / 5.0, 1.0)
    
    reports = []
    for i, key in enumerate(ordered_keys):
        reports.append(FeatureDeviationModel(
            feature=key,
            live=float(live_vector[i]),
            mean=float(means[i]),
            std=float(stds[i]),
            z_score=float(z_scores[i]),
            normalized_drift=float(normalized_drift[i]),
            weight=float(weights[i])
        ))
    return reports
