import time
import threading
import cv2
from typing import Optional, Any
from camera.camera_config import CameraConfig
from camera.rtsp_builder import RTSPBuilder, CameraBrand
from camera.frame_buffer import FrameBuffer

class CameraManager:
    """
    Manages an isolated background thread that connects to a CCTV stream.
    Includes active auto-reconnect tracking and 180-degree ceiling mount correction.
    """
    def __init__(self, config: CameraConfig, brand: CameraBrand = CameraBrand.GENERIC):
        self.config = config
        self.brand = brand
        
        ip_str = config.ip_address.strip()
        if ip_str.isdigit():
            self.rtsp_url = int(ip_str)
        else:
            self.rtsp_url = RTSPBuilder.build_url(config, brand)
        
        self._buffer = FrameBuffer()
        self._cap: Optional[cv2.VideoCapture] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self._is_connected = False
        self.is_reconnecting = False
        self.reconnect_attempts = 0
        
        self._frame_count = 0
        self._fps = 0.0
        self._start_time = None

    def connect(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._is_connected = False
        self.is_reconnecting = False
        self.reconnect_attempts = 0
        self._frame_count = 0
        self._start_time = time.time()
        
        self._worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker_thread.start()

    def _capture_loop(self) -> None:
        last_fps_check = time.time()
        frames_since_check = 0
        
        self._cap = cv2.VideoCapture(self.rtsp_url)
        
        if not self._cap.isOpened():
            self._is_connected = False
        else:
            self._is_connected = True

        while not self._stop_event.is_set():
            # STREAM RESILIENCE: Auto-Reconnect Engine
            if not self._is_connected or self._cap is None or not self._cap.isOpened():
                self.is_reconnecting = True
                self.reconnect_attempts += 1
                
                if self._cap:
                    self._cap.release()
                
                time.sleep(2.0) # Prevent CPU thrashing during network outage
                if self._stop_event.is_set():
                    break
                    
                self._cap = cv2.VideoCapture(self.rtsp_url)
                if self._cap.isOpened():
                    self._is_connected = True
                    self.is_reconnecting = False
                    self.reconnect_attempts = 0
                continue

            ret, frame = self._cap.read()
            if not ret:
                self._is_connected = False
                continue
            
            # Ceiling mount rotation correction disabled (keep upright)
            corrected_frame = frame
                
            self._buffer.set_frame(corrected_frame)
            self._frame_count += 1
            frames_since_check += 1
            
            now = time.time()
            duration = now - last_fps_check
            if duration >= 1.0:
                self._fps = frames_since_check / duration
                frames_since_check = 0
                last_fps_check = now

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=3.0)
            
        if self._cap:
            self._cap.release()
            self._cap = None
            
        self._buffer.clear()
        self._is_connected = False
        self.is_reconnecting = False
        self._fps = 0.0

    def is_connected(self) -> bool:
        return self._is_connected

    def get_latest_frame(self) -> Optional[Any]:
        return self._buffer.get_latest()

    def get_fps(self) -> float:
        return self._fps

    def get_frame_count(self) -> int:
        return self._frame_count

    def get_uptime(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
