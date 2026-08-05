"""
Thread-safe rolling buffer for historical tamper tracking and metric aggregations.
"""
import threading
from typing import List, Optional, Dict, Any
from backend.models.tamper_event import TamperEventModel
from backend.tamper.rule_engine import TAMPER_TYPES

class TamperHistoryBuffer:
    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = capacity
        self._buffer: List[TamperEventModel] = []
        self._lock = threading.Lock()

    def push(self, event: TamperEventModel) -> None:
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) > self.capacity:
                self._buffer.pop(0)

    def latest(self) -> Optional[TamperEventModel]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def history(self) -> List[TamperEventModel]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def statistics(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._buffer)
            if total == 0:
                return {
                    "total_events": 0, "normal_events": 0, "lens_cover_events": 0,
                    "spray_events": 0, "defocus_events": 0, "camera_move_events": 0,
                    "flash_events": 0, "freeze_events": 0, "noise_events": 0,
                    "partial_occlusion_events": 0, "unknown_events": 0,
                    "average_confidence": 0.0, "average_severity": "LOW"
                }

            counts = {k: 0 for k in TAMPER_TYPES}
            conf_sum = 0.0
            sev_scores = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            sev_sum = 0
            
            for evt in self._buffer:
                counts[evt.tamper_type] = counts.get(evt.tamper_type, 0) + 1
                conf_sum += evt.confidence
                sev_sum += sev_scores.get(evt.severity, 1)
                
            avg_conf = conf_sum / total
            avg_sev_num = round(sev_sum / total)
            reverse_sev = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
            
            return {
                "total_events": total,
                "normal_events": counts.get("NORMAL", 0),
                "lens_cover_events": counts.get("LENS_COVER", 0),
                "spray_events": counts.get("LENS_SPRAY", 0),
                "defocus_events": counts.get("DEFOCUS", 0),
                "camera_move_events": counts.get("CAMERA_MOVED", 0),
                "flash_events": counts.get("FLASH_ATTACK", 0),
                "freeze_events": counts.get("VIDEO_FREEZE", 0),
                "noise_events": counts.get("HEAVY_NOISE", 0),
                "partial_occlusion_events": counts.get("PARTIAL_OCCLUSION", 0),
                "unknown_events": counts.get("UNKNOWN_ANOMALY", 0),
                "average_confidence": round(avg_conf, 4),
                "average_severity": reverse_sev.get(avg_sev_num, "LOW")
            }

tamper_history = TamperHistoryBuffer()
