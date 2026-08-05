import threading
from datetime import datetime, timezone
from typing import Optional

class RealtimeState:
    """Thread-safe state tracking for the Realtime Engine."""
    def __init__(self):
        self.running: bool = False
        self.paused: bool = False
        self.frame_count: int = 0
        self.processed_count: int = 0
        self.fps: float = 0.0
        self.last_prediction: int = 0
        self.last_probability: float = 0.0
        self.last_drift_score: float = 0.0
        self.last_tamper_type: str = "NORMAL"
        self.last_timestamp: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        self._lock = threading.Lock()

    def get_uptime(self) -> float:
        with self._lock:
            if not self.running or not self.start_time:
                return 0.0
            return (datetime.now(timezone.utc) - self.start_time).total_seconds()
