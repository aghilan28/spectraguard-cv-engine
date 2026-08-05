import threading
from typing import Any, Optional

class FrameBuffer:
    """
    A thread-safe single-frame slot designed to store only the newest frame.
    Prevents race conditions between the daemon worker and GUI thread.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[Any] = None

    def set_frame(self, frame: Any) -> None:
        """Atomically overwrites the buffer slot with the newest frame."""
        with self._lock:
            self._frame = frame

    def get_latest(self) -> Optional[Any]:
        """Atomically retrieves the newest frame from the buffer slot."""
        with self._lock:
            return self._frame

    def has_frame(self) -> bool:
        """Atomically checks if a frame is loaded inside the buffer slot."""
        with self._lock:
            return self._frame is not None

    def clear(self) -> None:
        """Atomically flushes the current frame buffer slot."""
        with self._lock:
            self._frame = None
