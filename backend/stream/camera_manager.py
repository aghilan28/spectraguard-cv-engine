import cv2
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, Any
import numpy as np
from backend.config.logging import logger
from backend.services.frame_buffer import FrameBuffer

class CameraManager:
    _instances: Dict[str, "CameraManager"] = {}
    _lock = threading.Lock()

    def __new__(cls, camera_id: str = "default", source: str = "0") -> "CameraManager":
        with cls._lock:
            key = f"{camera_id}_{source}"
            if key not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(self, camera_id: str = "default", source: str = "0") -> None:
        if self._initialized:
            return
            
        self.camera_id = camera_id
        self.source = source
        self.source_index_or_url: Any = int(source) if source.isdigit() else source
        
        self.buffer = FrameBuffer(max_size=30)
        self.cap: Optional[cv2.VideoCapture] = None
        
        self.width = 0
        self.height = 0
        self.configured_fps = 0.0
        self.frame_count = 0
        self.started_at: Optional[datetime] = None
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._initialized = True

    def start(self) -> bool:
        with self._state_lock:
            if self._running:
                return True

            logger.info(f"Initializing video resource capture for source: {self.source}")
            self.cap = cv2.VideoCapture(self.source_index_or_url)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to establish hardware camera handshake for source: {self.source}")
                self.cap = None
                return False

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.configured_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.configured_fps <= 0 or self.configured_fps > 120:
                self.configured_fps = 30.0

            self._running = True
            self.frame_count = 0
            self.started_at = datetime.now(timezone.utc)
            self.buffer.clear()

            self._thread = threading.Thread(target=self._capture_loop, name=f"cam_Acq_{self.camera_id}", daemon=True)
            self._thread.start()
            return True

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        while self._running:
            # Check state lock flags natively to instantly slip away from blocking context
            if not self._running:
                break
                
            if self.cap is None:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    time.sleep(0.5)
                continue

            consecutive_failures = 0
            self.frame_count += 1
            self.buffer.push(frame)
            time.sleep(1.0 / self.configured_fps)

    def stop(self) -> None:
        with self._state_lock:
            if not self._running:
                return
            self._running = False
            
        if self._thread:
            # Short timeout gate ensuring standard tests don't freeze indefinitely
            self._thread.join(timeout=0.2)
            self._thread = None

        with self._state_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.buffer.clear()
            self.started_at = None
            logger.info(f"Camera hardware pipeline [{self.camera_id}] released safely.")

    def restart(self) -> bool:
        self.stop()
        return self.start()

    def is_running(self) -> bool:
        with self._state_lock:
            return self._running and self.cap is not None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        return self.buffer.latest()

    def get_fps(self) -> float:
        return self.configured_fps if self.is_running() else 0.0

    def get_uptime(self) -> float:
        if not self.started_at:
            return 0.0
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def get_info(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "opencv_backend": "LOCAL_LOOP",
            "camera_source": str(self.source),
            "max_fps": self.configured_fps,
            "resolution": f"{self.width}x{self.height}"
        }
