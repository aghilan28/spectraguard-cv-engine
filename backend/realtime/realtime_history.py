import threading
from collections import deque
from typing import List, Dict, Optional

class RealtimeHistory:
    """Thread-safe rolling buffer for historical detection events (Max 5000)."""
    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._buffer = deque(maxlen=self.max_size)
        self._lock = threading.Lock()

    def add(self, record: dict) -> None:
        with self._lock:
            self._buffer.append(record)

    def get_latest(self) -> Optional[dict]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_all(self) -> List[dict]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
