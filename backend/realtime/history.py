import threading
from typing import List, Optional, Dict, Any
from backend.models.detection_event import DetectionEventModel

class RealTimeHistoryBuffer:
    """Thread-safe bounded memory queue logging up to 10,000 detection reports."""
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self._buffer: List[DetectionEventModel] = []
        self._lock = threading.Lock()

    def push(self, event: DetectionEventModel) -> None:
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) > self.capacity:
                self._buffer.pop(0)

    def latest(self) -> Optional[DetectionEventModel]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def history(self) -> List[DetectionEventModel]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def statistics(self, uptime: float = 0.0) -> Dict[str, Any]:
        with self._lock:
            size = len(self._buffer)
            if size == 0:
                return {
                    "total_events": 0, "normal_events": 0, "tamper_events": 0, "unknown_events": 0,
                    "average_latency": 0.0, "average_confidence": 0.0, "average_probability": 0.0,
                    "average_deviation": 0.0, "uptime": uptime
                }

            latencies = [e.latency_ms for e in self._buffer]
            confidences = [e.confidence for e in self._buffer]
            probabilities = [e.probability for e in self._buffer]
            deviations = [e.deviation_score for e in self._buffer]

            normal = sum(1 for e in self._buffer if e.tamper_type == "NORMAL")
            unknown = sum(1 for e in self._buffer if e.tamper_type == "UNKNOWN_ANOMALY")
            tamper = size - normal - unknown

            return {
                "total_events": size,
                "normal_events": normal,
                "tamper_events": tamper,
                "unknown_events": unknown,
                "average_latency": round(sum(latencies) / size, 3),
                "average_confidence": round(sum(confidences) / size, 4),
                "average_probability": round(sum(probabilities) / size, 4),
                "average_deviation": round(sum(deviations) / size, 4),
                "uptime": uptime
            }

realtime_history = RealTimeHistoryBuffer()
