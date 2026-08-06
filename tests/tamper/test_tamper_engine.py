"""
Deterministic behavior tests asserting absolute matrix rule evaluations.
"""
import pytest
from dataclasses import dataclass
from backend.tamper.rule_engine import evaluate_rules
from backend.tamper.confidence import calculate_confidence
from backend.tamper.tamper_engine import tamper_engine
from backend.models.deviation_result import DeviationReportModel, FeatureDeviationModel

@dataclass
class MockInferenceResult:
    prediction: int
    probability: float

def build_mock_deviation(z_scores: dict, global_score=0.5):
    features = []
    for k, v in z_scores.items():
        features.append(FeatureDeviationModel(feature=k, live=0.0, mean=0.0, std=1.0, z_score=v, normalized_drift=abs(v)/5.0, weight=0.1))
    return DeviationReportModel(timestamp=None, overall_score=global_score, severity="MEDIUM", mahalanobis_distance=1.0, feature_reports=features, latency_ms=1.0)

def test_rule_evaluator_lens_cover():
    # Simulate LENS_COVER: Brightness down, Entropy down, Edge Density down
    z_scores = {"log_total_energy": -5.0, "shannon_entropy": -5.0, "edge_density": -5.0, "temporal_difference": 0.0}
    report = build_mock_deviation(z_scores)
    winning_rule, scores = evaluate_rules(report.feature_reports)
    assert winning_rule == "LENS_COVER"
    assert scores["LENS_COVER"] == 1.0

def test_rule_evaluator_unknown_anomaly_tie():
    # Tie break scenario between Flash and Overexposure
    z_scores = {"log_total_energy": 5.0, "shannon_entropy": -5.0, "fft_low_ratio": 5.0}
    report = build_mock_deviation(z_scores)
    winning_rule, scores = evaluate_rules(report.feature_reports)
    assert winning_rule == "UNKNOWN_ANOMALY"

def test_confidence_calculations():
    # Perfect rule, strong RF, high deviation -> High confidence
    c1 = calculate_confidence(rule_score=1.0, dev_score=1.0, rf_prob=0.99)
    assert c1 > 0.80
    
    # Weak rule, weak RF, low dev -> Low confidence
    c2 = calculate_confidence(rule_score=0.2, dev_score=0.1, rf_prob=0.45)
    assert c2 < 0.40

def test_orchestration_engine():
    z_scores = {"temporal_difference": -5.0} # Freeze signature
    
    # Global score 0.5 + Rule score 1.0 = Blend 0.75 (<0.80 -> HIGH)
    dev_rep = build_mock_deviation(z_scores, global_score=0.5) 
    inf_rep = MockInferenceResult(prediction=1, probability=0.85)
    
    event = tamper_engine.evaluate(inf_rep, dev_rep)
    assert event.tamper_type == "VIDEO_FREEZE"
    assert event.severity == "HIGH"
    assert event.explanation != ""

def test_classification_engine_fallbacks():
    import numpy as np
    from backend.tamper.classification_engine import TamperClassificationEngine
    engine = TamperClassificationEngine()
    
    # Create a dummy normal/neutral frame with random noise to avoid triggering FULL_LENS_COVER/BLUR
    np.random.seed(42)
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # If prob is low (<0.50), classification should return NORMAL
    label_normal = engine.classify(frame, [], prob=0.10)
    assert label_normal == "NORMAL"
    
    # If prob is high (>=0.50) but rules are inconclusive, fallback should return UNKNOWN_ANOMALY
    label_unknown = engine.classify(frame, [], prob=0.85)
    assert label_unknown == "UNKNOWN_ANOMALY"

def test_evaluate_motion_stationary():
    import numpy as np
    from backend.tamper.classification_engine import TamperClassificationEngine
    engine = TamperClassificationEngine()
    
    # Test identical frames (zero shift)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    moved, shift = engine.evaluate_motion(frame, frame)
    assert not moved
    assert shift == 0.0
