import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np

from backend.config.logging import logger
from backend.calibration.calibration_storage import CalibrationStorage
from backend.models.deviation_result import DeviationReportModel
from backend.deviation.zscore import calculate_zscores
from backend.deviation.mahalanobis import calculate_mahalanobis_distance
from backend.deviation.weighted_score import compute_weighted_drift

class DeviationEngine:
    """High-speed vectorized numerical mapping routine comparing live outputs against environment maps."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "DeviationEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.storage = CalibrationStorage()
        self._cached_baseline: Optional[Dict[str, Any]] = None
        
        # Invariant architectural parameter ordering weights
        self.feature_weights = {
            "laplacian_variance": 0.20, "log_total_energy": 0.18, "edge_density": 0.15,
            "shannon_entropy": 0.14, "fft_low_ratio": 0.10, "fft_mid_ratio": 0.08,
            "temporal_difference": 0.08, "fft_high_ratio": 0.07
        }
        self._ordered_keys = list(self.feature_weights.keys())
        
        # Numpy static layouts caching boundaries allocations vectors sets
        self._means = np.zeros(len(self._ordered_keys), dtype=np.float64)
        self._stds = np.zeros(len(self._ordered_keys), dtype=np.float64)
        self._weights = np.array([self.feature_weights[k] for k in self._ordered_keys], dtype=np.float64)
        self._covariance: Optional[np.ndarray] = None
        
        self._load_lock = threading.Lock()
        self._initialized = True
        self.load_baseline_cache()

    def load_baseline_cache(self) -> bool:
        """Lock context and instantiate static memory parameter configurations indices mappings."""
        with self._load_lock:
            if not self.storage.exists():
                self._cached_baseline = None
                return False
            
            data = self.storage.load()
            if not data or "features" not in data:
                return False
                
            self._cached_baseline = data
            features_map = data["features"]
            
            for i, key in enumerate(self._ordered_keys):
                stat = features_map.get(key, {})
                self._means[i] = stat.get("mean", 0.0)
                self._stds[i] = stat.get("std", 1.0)
                
            # Attempt to reconstruct baseline covariance arrays if present
            dim = len(self._ordered_keys)
            if "covariance" in data:
                self._covariance = np.array(data["covariance"], dtype=np.float64)
            else:
                self._covariance = np.eye(dim, dtype=np.float64) * 0.1
                
            logger.info("Deviation Engine target baseline vectors mapped to active memory segments cache loops cleanly.")
            return True

    def evaluate(self, live_features: Dict[str, float]) -> DeviationReportModel:
        """
        Processes real-time multi-dimensional spatial data loops under <2ms execution speeds benchmarks.
        
        Args:
            live_features: 8D physical sensor metrics snapshot parameters.
            
        Returns:
            DeviationReportModel mapping raw execution traces metrics arrays profiles.
        """
        start_t = time.perf_counter()
        
        if self._cached_baseline is None:
            # Fallback path try hot load on late registrations hooks
            if not self.load_baseline_cache():
                raise ValueError("Active baseline is completely empty or target initialization profiles missing from storage maps indices.")

        # Allocate live metric structures directly to unified vectorized matrix arrays blocks layers
        live_vector = np.array([live_features.get(k, 0.0) for k in self._ordered_keys], dtype=np.float64)
        
        # Parallel Vector processing invocations execution layers calls bounds nodes maps parameters
        reports = calculate_zscores(live_vector, self._means, self._stds, self._weights, self._ordered_keys)
        mahalanobis_dist = calculate_mahalanobis_distance(live_vector, self._means, self._covariance)
        overall_score, severity = compute_weighted_drift(reports)
        
        latency = (time.perf_counter() - start_t) * 1000
        
        report = DeviationReportModel(
            timestamp=datetime.now(timezone.utc),
            overall_score=overall_score,
            severity=severity,
            mahalanobis_distance=mahalanobis_dist,
            feature_reports=reports,
            latency_ms=round(latency, 3)
        )
        
        logger.debug(f"Deviation Processed | Score: {overall_score:.4f} | Severity: {severity} | Mahalanobis: {mahalanobis_dist:.2f} | Latency: {latency:.2f}ms")
        return report

deviation_engine = DeviationEngine()
