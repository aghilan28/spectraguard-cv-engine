import os
import json
import uuid
import cv2
import queue
import threading
import time
import collections
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

def to_standard_taxonomy(name: str) -> str:
    """Maps internal backend physical classifications to the strict 12-class taxonomy vocabulary."""
    mapping = {
        "LENS_COVER": "FULL_LENS_COVER",
        "LENS_SPRAY": "BLUR_ATTACK",
        "DEFOCUS": "DEFOCUS",
        "CAMERA_MOVED": "CAMERA_MOVED",
        "FLASH_ATTACK": "BRIGHTNESS_ATTACK",
        "DARKNESS": "DARKNESS_ATTACK",
        "OVEREXPOSURE": "BRIGHTNESS_ATTACK",
        "VIDEO_FREEZE": "VIDEO_FREEZE",
        "HEAVY_NOISE": "NOISE_ATTACK",
        "PARTIAL_OCCLUSION": "PARTIAL_LENS_COVER",
        "UNKNOWN_ANOMALY": "FULL_LENS_COVER",
        "NORMAL": "NORMAL",
        "HAND_COVER": "HAND_COVER",
        "CAMERA_REDIRECTED": "CAMERA_REDIRECTED",
        "BLUR_ATTACK": "BLUR_ATTACK",
        "BRIGHTNESS_ATTACK": "BRIGHTNESS_ATTACK",
        "DARKNESS_ATTACK": "DARKNESS_ATTACK",
        "NOISE_ATTACK": "NOISE_ATTACK",
        "FULL_LENS_COVER": "FULL_LENS_COVER",
        "PARTIAL_LENS_COVER": "PARTIAL_LENS_COVER"
    }
    return mapping.get(name, "NORMAL")

class DetectionEvent(BaseModel):
    uuid: str
    camera_name: str
    timestamp: str
    prediction: str
    probability: float
    severity: str
    snapshot_path: Optional[str]
    drift_score: float
    rule: str
    
    # Standardized Taxonomy additions
    event_id: str
    tamper_type: str
    confidence: float

    # Notification fields
    screenshot_path: Optional[str] = ""
    notification_status: str = "Pending"
    notification_provider: str = "Telegram"
    notification_delivery_state: str = "PENDING"
    notification_timestamp: str = ""
    notification_attempts: int = 0
    notification_error: str = ""
    notification_latency_ms: float = 0.0
    message_sid: str = ""
    recipient_number: str = ""
    retry_count: int = 0
    last_error: str = ""

class NotificationProvider:
    def send(self, message: str): 
        raise NotImplementedError

class ConsoleProvider(NotificationProvider):
    def send(self, message: str): 
        print(f"[TELEGRAM ALERT] {message}")


class EventService:
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls, *args, **kwargs)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.snapshots_dir = "storage/snapshots"
        self.events_root_dir = "storage/events"
        self.notifier = ConsoleProvider()
        
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.events_root_dir, exist_ok=True)

        # In-memory history deque
        self.history_deque = collections.deque(maxlen=50)
        self.deque_lock = threading.Lock()

        # Deduplication cache
        self.last_triggered = {}  # key: (camera_name, rule) -> value: timestamp
        self.cache_lock = threading.Lock()

        # Threading queue & background worker
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True, name="EventWriterThread")
        self.worker_thread.start()

        # Import NotificationManager lazily to prevent circular imports
        from backend.notifications.notification_manager import NotificationManager
        self.notification_manager = NotificationManager()

        self._initialized = True

    def handle_detection(self, camera_name, frame, prob, severity, drift, rule):
        """
        Public API: Thread-safe non-blocking event handler. 
        Filters duplicates within 5 seconds and queues the frame/metadata for background disk writes.
        """
        if not isinstance(camera_name, str):
            camera_name = str(camera_name)
        if not isinstance(severity, str):
            severity = str(severity)
        if not isinstance(rule, str):
            rule = str(rule)

        # Map to standard taxonomy immediately
        std_rule = to_standard_taxonomy(rule)

        now = time.time()
        cache_key = (camera_name, std_rule)

        with self.cache_lock:
            last_time = self.last_triggered.get(cache_key, 0.0)
            if now - last_time < 5.0:
                # Deduplicate and silent skip to prevent spamming
                return None
            self.last_triggered[cache_key] = now

        event_id = str(uuid.uuid4())
        # Capture current timestamps on the calling thread
        ts_utc = datetime.utcnow()
        ts_str = ts_utc.strftime("%Y-%m-%d_%H-%M-%S")
        date_folder_name = ts_utc.strftime("%Y-%m-%d")

        # Snapshot path metadata (generated now, written in background)
        snap_name = f"{camera_name}_{ts_str}_tamper_{event_id[:8]}.jpg"
        snap_path = os.path.join(self.snapshots_dir, snap_name)

        # Create DetectionEvent object
        event = DetectionEvent(
            uuid=event_id,
            camera_name=camera_name,
            timestamp=ts_str,
            prediction="Tamper",
            probability=prob,
            severity=severity,
            snapshot_path=snap_path,
            drift_score=drift,
            rule=std_rule,
            event_id=event_id,
            tamper_type=std_rule,
            confidence=round(prob * 100.0, 2),
            screenshot_path=snap_path
        )

        # Add to in-memory deque immediately for instant GUI display
        with self.deque_lock:
            self.history_deque.append(event.dict())

        # Queue data for background disk IO (JPEG compression + JSON serialization)
        # We pass a copy of the frame to prevent the caller thread from modifying it
        frame_copy = frame.copy() if frame is not None else None
        self.task_queue.put({
            "event": event,
            "frame": frame_copy,
            "date_folder": date_folder_name,
            "snap_path": snap_path
        })

        # Non-blocking return of event details
        msg = f"TAMPER ALERT | Cam: {camera_name} | Time: {ts_str} | Sev: {severity} | Rule: {std_rule}"
        self.notifier.send(msg)
        return event

    def _process_queue(self):
        """Infinite worker loop executing file writes asynchronously."""
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    break

                event = task["event"]
                frame = task["frame"]
                date_folder = task["date_folder"]
                snap_path = task["snap_path"]

                # 1. Asynchronously write JPEG frame
                if frame is not None:
                    try:
                        success, img_buf = cv2.imencode(".jpg", frame)
                        if success:
                            with open(snap_path, "wb") as f:
                                f.write(img_buf.tobytes())
                        else:
                            print("[EventWriter] cv2.imencode failed")
                    except Exception as e:
                        print(f"[EventWriter] Failed to write snapshot frame: {e}")


                # 2. Asynchronously write structured individual JSON
                try:
                    folder_path = os.path.join(self.events_root_dir, date_folder)
                    os.makedirs(folder_path, exist_ok=True)
                    
                    event_json_name = f"event_{event.timestamp}_{event.uuid[:8]}.json"
                    event_json_path = os.path.join(folder_path, event_json_name)

                    with open(event_json_path, "w", encoding="utf-8") as f:
                        json.dump(event.dict(), f, indent=2)

                    # 3. Asynchronously update latest_event.json shortcut
                    latest_path = os.path.join(self.events_root_dir, "latest_event.json")
                    with open(latest_path, "w", encoding="utf-8") as f:
                        json.dump(event.dict(), f, indent=2)

                except Exception as e:
                    print(f"[EventWriter] Failed to write event JSON: {e}")

                # 4. Process SMS alerts asynchronously via NotificationManager
                try:
                    self.notification_manager.handle_event(event)
                except Exception as e:
                    print(f"[EventWriter] Notification dispatch error: {e}")

                self.task_queue.task_done()
            except Exception as e:
                print(f"[EventWriter] Fatal error in loop: {e}")
                time.sleep(0.5)

    def get_history(self) -> list:
        """Thread-safe getter for in-memory deque history."""
        with self.deque_lock:
            return list(self.history_deque)
