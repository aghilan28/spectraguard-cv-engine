import threading
from typing import Dict, Any

class RealtimeMetrics:
    """Thread-safe accumulator for performance and detection statistics."""
    def __init__(self):
        self.total_processed: int = 0
        self.latency_sum: float = 0.0
        self.drift_sum: float = 0.0
        self.prob_sum: float = 0.0
        self.tamper_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def update(self, latency: float, drift: float, probability: float, tamper_type: str) -> None:
        with self._lock:
            self.total_processed += 1
            self.latency_sum += latency
            self.drift_sum += drift
            self.prob_sum += probability
            self.tamper_counts[tamper_type] = self.tamper_counts.get(tamper_type, 0) + 1

    def get_statistics(self, uptime: float) -> Dict[str, Any]:
        with self._lock:
            count = self.total_processed if self.total_processed > 0 else 1
            normal_count = self.tamper_counts.get("NORMAL", 0)
            unknown_count = self.tamper_counts.get("UNKNOWN_ANOMALY", 0)
            tamper_count = self.total_processed - normal_count - unknown_count
            fps = self.total_processed / uptime if uptime > 0 else 0.0

            return {
                "total_processed": self.total_processed,
                "average_latency": round(self.latency_sum / count, 3),
                "average_drift": round(self.drift_sum / count, 4),
                "average_probability": round(self.prob_sum / count, 4),
                "tamper_counts": tamper_count,
                "normal_counts": normal_count,
                "unknown_counts": unknown_count,
                "processing_fps": round(fps, 2),
                "uptime": round(uptime, 2)
            }
