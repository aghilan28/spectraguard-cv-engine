import pytest
import numpy as np
from backend.deviation.zscore import calculate_zscores
from backend.deviation.mahalanobis import calculate_mahalanobis_distance
from backend.deviation.weighted_score import compute_weighted_drift
from backend.deviation.history import DeviationHistoryBuffer

def test_vectorized_zscore_math():
    live = np.array([2.0, 0.0], dtype=np.float64)
    means = np.array([0.0, 0.0], dtype=np.float64)
    stds = np.array([1.0, 0.0], dtype=np.float64) # Test zero variance tracking epsilon protection parameters bounds
    weights = np.array([0.5, 0.5], dtype=np.float64)
    keys = ["f1", "f2"]
    
    reports = calculate_zscores(live, means, stds, weights, keys)
    assert len(reports) == 2
    assert reports[0].z_score == 2.0
    assert reports[0].normalized_drift == 0.4
    assert reports[1].z_score == 0.0

def test_mahalanobis_singular_regularization():
    live = np.array([1.0, 2.0], dtype=np.float64)
    means = np.array([0.0, 0.0], dtype=np.float64)
    # Force singular matrix shape (perfectly correlated components)
    cov = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    
    dist = calculate_mahalanobis_distance(live, means, cov)
    assert dist > 0.0
    assert not np.isnan(dist)

def test_history_buffer_telemetry_aggregations():
    buf = DeviationHistoryBuffer(capacity=5)
    assert buf.statistics()["history_size"] == 0
