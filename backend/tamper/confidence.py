"""
Deterministic confidence and severity rating evaluators.
"""

def calculate_confidence(rule_score: float, dev_score: float, rf_prob: float) -> float:
    """
    Blends multiple deterministic metrics into a final confidence probability [0-1].
    
    Args:
        rule_score: The mathematical strength of the winning physical rule.
        dev_score: The global environment deviation score [0-1].
        rf_prob: The Random Forest prediction probability [0-1].
        
    Returns:
        float: Absolute confidence score.
    """
    # Distance of probability from uncertainty (0.5) scaled to [0,1]
    prob_certainty = abs(rf_prob - 0.5) * 2.0
    
    # Weightings: Rule Signature (50%), Random Forest Certainty (30%), Global Deviation (20%)
    confidence = (rule_score * 0.50) + (prob_certainty * 0.30) + (dev_score * 0.20)
    return min(max(confidence, 0.0), 1.0)

def calculate_severity(dev_score: float, rule_score: float) -> str:
    """
    Determines categorical severity level based on vector deviation scale and rule fit.
    """
    blend = (dev_score + rule_score) / 2.0
    
    if blend < 0.20:
        return "LOW"
    elif blend < 0.50:
        return "MEDIUM"
    elif blend < 0.80:
        return "HIGH"
    else:
        return "CRITICAL"
