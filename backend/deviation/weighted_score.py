import numpy as np
from typing import List, Tuple
from backend.models.deviation_result import FeatureDeviationModel

def compute_weighted_drift(feature_reports: List[FeatureDeviationModel]) -> Tuple[float, str]:
    """Calculates final combined deterministic drift averages."""
    score = sum(r.weight * r.normalized_drift for r in feature_reports)
    clamped_score = float(np.clip(score, 0.0, 1.0))
    
    if clamped_score < 0.20:
        sev = "VERY_LOW"
    elif clamped_score < 0.40:
        sev = "LOW"
    elif clamped_score < 0.60:
        sev = "MEDIUM"
    elif clamped_score < 0.80:
        sev = "HIGH"
    else:
        sev = "CRITICAL"
        
    return clamped_score, sev
