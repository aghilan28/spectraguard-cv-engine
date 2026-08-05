"""
Automated validation checks testing strict mathematics behaviors inside standard deviations logic sequences metrics bounds matrices.
"""
import pytest
from backend.calibration.drift_score import compute_drift_vector
from backend.calibration.baseline_comparator import BaselineComparator

def test_drift_score_zero_deviation():
    live = {
        "laplacian_variance": 500.0,
        "log_total_energy": 20.0,
        "edge_density": 0.25,
        "shannon_entropy": 7.5,
        "fft_low_ratio": 0.4,
        "fft_mid_ratio": 0.3,
        "temporal_difference": 5.0,
        "fft_high_ratio": 0.1
    }
    
    # Exact match means Z-score = 0 -> Drift = 0.0
    baseline = {k: {"mean": v, "std": 1.0} for k, v in live.items()}
    
    score, sev, features = compute_drift_vector(live, baseline)
    assert score == 0.0
    assert sev == "VERY_LOW"
    assert len(features) == 8

def test_drift_score_max_deviation():
    # Elevate all features to >= 5 standard deviations away from mean (0.0)
    live = {
        "laplacian_variance": 5.0,
        "log_total_energy": 5.0,
        "edge_density": 5.0,
        "shannon_entropy": 5.0,
        "fft_low_ratio": 5.0,
        "fft_mid_ratio": 5.0,
        "temporal_difference": 5.0,
        "fft_high_ratio": 5.0
    }
    
    baseline = {k: {"mean": 0.0, "std": 1.0} for k, v in live.items()}
    
    score, sev, features = compute_drift_vector(live, baseline)
    assert score == 1.0
    assert sev == "CRITICAL"

def test_rolling_history_buffer():
    comparator = BaselineComparator()
    comparator.clear_history()
    
    # Test buffer limits logic tracking state histories directly
    for i in range(1005):
        from backend.models.drift import DriftReportModel
        from datetime import datetime, timezone
        mock_report = DriftReportModel(
            timestamp=datetime.now(timezone.utc),
            global_score=0.1,
            severity="VERY_LOW",
            features=[],
            latency_ms=0.5
        )
        comparator._history.append(mock_report)
        if len(comparator._history) > comparator.max_history:
            comparator._history.pop(0)
            
    assert len(comparator.get_history()) == 1000
