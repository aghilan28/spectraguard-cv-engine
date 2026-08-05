"""
Execution engine that compares real-time extraction matrices against standard baselines.
Maintains a rolling 1000-frame history of drift structures.
"""
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from backend.config.logging import logger
from backend.calibration.baseline_loader import baseline_loader
from backend.calibration.drift_score import compute_drift_vector
from backend.models.drift import DriftReportModel

class BaselineComparator:
    """Stateful singleton performing high-speed vector deviation extraction and retention."""
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "BaselineComparator":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_history: int = 1000) -> None:
        if self._initialized:
            return
        self.max_history = max_history
        self._history: List[DriftReportModel] = []
        self._history_lock = threading.Lock()
        self._initialized = True

    def compare(self, live_features: Dict[str, float]) -> DriftReportModel:
        """
        Executes <2ms mathematical comparison.
        
        Args:
            live_features: Target physical parameters dictionary.
            
        Returns:
            DriftReportModel containing fully compiled comparison payloads.
        """
        start_t = time.perf_counter()
        
        baseline_data = baseline_loader.get_data()
        if not baseline_data or "features" not in baseline_data:
            raise ValueError("Baseline is not loaded or missing internal structural matrices.")

        logger.debug("Comparison execution frame sequence started.")
        
        # Calculate Math Vectors
        global_score, severity, breakdown = compute_drift_vector(live_features, baseline_data["features"])
        
        latency_ms = round((time.perf_counter() - start_t) * 1000, 3)
        
        report = DriftReportModel(
            timestamp=datetime.now(timezone.utc),
            global_score=round(global_score, 4),
            severity=severity,
            features=breakdown,
            latency_ms=latency_ms
        )
        
        # Lock and push to Rolling History Buffer
        with self._history_lock:
            self._history.append(report)
            if len(self._history) > self.max_history:
                self._history.pop(0)
                
        logger.debug(f"Comparison Finished | Global Drift: {report.global_score:.4f} | Severity: {report.severity} | Latency: {report.latency_ms}ms")
        
        return report

    def get_latest(self) -> Optional[DriftReportModel]:
        with self._history_lock:
            return self._history[-1] if self._history else None

    def get_history(self) -> List[DriftReportModel]:
        with self._history_lock:
            return list(self._history)
            
    def clear_history(self) -> None:
        with self._history_lock:
            self._history.clear()
            logger.info("Baseline comparator rolling history map forcefully purged.")

baseline_comparator = BaselineComparator()
