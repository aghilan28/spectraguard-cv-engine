import threading
from typing import List, Callable
from backend.models.detection_event import DetectionEventModel
from backend.config.logging import logger

class EventManager:
    """Implements thread-safe Observer tracking patterns enabling decoupling hooks for future layers."""
    def __init__(self) -> None:
        self._subscribers: List[Callable[[DetectionEventModel], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[DetectionEventModel], None]) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[DetectionEventModel], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def publish(self, event: DetectionEventModel) -> None:
        with self._lock:
            current_subs = list(self._subscribers)
            
        for callback in current_subs:
            try:
                callback(event)
            except Exception as err:
                logger.error(f"Event subscriber distribution execution failure: {err}")

event_manager = EventManager()
