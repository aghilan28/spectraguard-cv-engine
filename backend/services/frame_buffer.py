import threading
from typing import List, Optional
import numpy as np

class FrameBuffer:
    def __init__(self, max_size: int = 30) -> None:
        self.max_size = max_size
        self._buffer: List[np.ndarray] = []
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buffer.append(frame)
            if len(self._buffer) > self.max_size:
                self._buffer.pop(0)

    def pop(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer.pop(0)

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].copy()

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self.max_size

    def frames(self) -> List[np.ndarray]:
        with self._lock:
            return [f.copy() for f in self._buffer]

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)
