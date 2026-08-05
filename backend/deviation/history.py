import threading
from typing import List, Optional, Dict, Any
from backend.models.deviation_result import DeviationReportModel

class DeviationHistoryBuffer:
    """Thread-safe rolling data stack storing up to 1,000 statistical matrices updates."""
    
    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._buffer: List[DeviationReportModel] = []
        self._lock = threading.Lock()

    def push(self, report: DeviationReportModel) -> None:
        with self._lock:
            self._buffer.append(report)
            if len(self._buffer) > self.capacity:
                self._buffer.pop(0)

    def latest(self) -> Optional[DeviationReportModel]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def history(self) -> List[DeviationReportModel]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def statistics(self) -> Dict[str, Any]:
        with self._lock:
            size = len(self._buffer)
            if size == 0:
                return {
                    "average_score": 0.0, "maximum_score": 0.0, "minimum_score": 0.0,
                    "average_mahalanobis": 0.0, "highest_feature_drift": "NONE",
                    "lowest_feature_drift": "NONE", "history_size": 0
                }
            
            scores = [r.overall_score for r in self._buffer]
            dist = [r.mahalanobis_distance for r in self._buffer]
            
            # Map tracking individual parameter drift components totals
            feature_drift_totals: Dict[str, float] = {}
            for r in self._buffer:
                for f in r.feature_reports:
                    feature_drift_totals[f.feature] = feature_drift_totals.get(f.feature, 0.0) + f.normalized_drift
            
            highest = max(feature_drift_totals, key=lambda k: feature_drift_totals[k]) if feature_drift_totals else "NONE"
            lowest = min(feature_drift_totals, key=lambda k: feature_drift_totals[k]) if feature_drift_totals else "NONE"
            
            return {
                "average_score": round(sum(scores) / size, 4),
                "maximum_score": round(max(scores), 4),
                "minimum_score": round(min(scores), 4),
                "average_mahalanobis": round(sum(dist) / size, 4),
                "highest_feature_drift": highest,
                "lowest_feature_drift": lowest,
                "history_size": size
            }

deviation_history = DeviationHistoryBuffer()
