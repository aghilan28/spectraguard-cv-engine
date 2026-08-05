import threading
from typing import List, Optional
from backend.inference.result import InferenceResult
from backend.inference.history import InferenceStatistics

class PredictionBuffer:
    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._buffer: List[InferenceResult] = []
        self._lock = threading.Lock()

    def push(self, result: InferenceResult) -> None:
        with self._lock:
            self._buffer.append(result)
            if len(self._buffer) > self.max_size:
                self._buffer.pop(0)

    def latest(self) -> Optional[InferenceResult]:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def history(self) -> List[InferenceResult]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def statistics(self) -> InferenceStatistics:
        with self._lock:
            total = len(self._buffer)
            if total == 0:
                return InferenceStatistics(
                    total_inferences=0, normal_count=0, tamper_count=0,
                    average_probability=0.0, average_confidence=0.0
                )
            
            tamper_count = sum(1 for r in self._buffer if r.prediction == 1)
            normal_count = total - tamper_count
            avg_prob = sum(r.probability for r in self._buffer) / total
            avg_conf = sum(r.confidence for r in self._buffer) / total
            
            return InferenceStatistics(
                total_inferences=total,
                normal_count=normal_count,
                tamper_count=tamper_count,
                average_probability=round(avg_prob, 4),
                average_confidence=round(avg_conf, 4)
            )

prediction_buffer = PredictionBuffer()
