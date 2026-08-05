import threading
from typing import Set

class RealTimeStateManager:
    """Thread-safe state tracking machine governing background orchestration loop bounds."""
    VALID_STATES: Set[str] = {"IDLE", "STARTING", "RUNNING", "PAUSED", "STOPPING", "STOPPED", "ERROR"}

    def __init__(self) -> None:
        self._state: str = "IDLE"
        self._lock = threading.Lock()

    def set_state(self, new_state: str) -> None:
        if new_state not in self.VALID_STATES:
            raise ValueError(f"Invalid architectural state target token declaration: {new_state}")
        with self._lock:
            self._state = new_state

    def get_state(self) -> str:
        with self._lock:
            return self._state

    def is_running(self) -> bool:
        with self._lock:
            return self._state in {"STARTING", "RUNNING", "PAUSED"}

    def is_paused(self) -> bool:
        with self._lock:
            return self._state == "PAUSED"

state_manager = RealTimeStateManager()
