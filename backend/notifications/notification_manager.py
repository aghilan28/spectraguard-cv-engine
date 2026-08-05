import os
import queue
import threading
import logging
import time
import json
import shutil
from datetime import datetime
from backend.notifications.rate_limiter import RateLimiter
from backend.notifications.telegram_settings import TelegramSettings
from backend.notifications.telegram_service import TelegramService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NotificationManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NotificationManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        self.settings_path = "config/user_settings.json"
        self.telegram_service = TelegramService()
        self.rate_limiter = RateLimiter(cooldown_seconds=30)
        self.notification_history = []
        
        # Duplicate prevention cache (thread-safe set of event IDs)
        self.processed_events = set()
        self.events_lock = threading.Lock()
        
        # Asynchronous Queue with max size 1000
        self.queue = queue.Queue(maxsize=1000)
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True, name="NotificationWorker")
        self.worker_thread.start()

    def _process_queue(self):
        while True:
            try:
                event = self.queue.get()
                event_id = event.get("event_id") or event.get("uuid")
                print(f"[Notification] Worker picked event: {event_id}")
                self._execute_notification(event)
                self.queue.task_done()
            except Exception as e:
                logging.error(f"[NotificationManager] Queue worker error: {e}")

    def handle_event(self, event):
        """
        Public entry point: Processes a DetectionEvent object or dict.
        Validates, rate limits, dispatches, and updates state.
        """
        if hasattr(event, "model_dump"):
            event_dict = event.model_dump()
        elif hasattr(event, "__dict__"):
            event_dict = event.__dict__
        elif isinstance(event, dict):
            event_dict = event
        else:
            event_dict = dict(event)
            
        self.dispatch_event(event_dict)

    def dispatch_event(self, event: dict):
        event_id = event.get("event_id") or event.get("uuid")
        if not event_id:
            return
            
        # Duplicate Prevention
        with self.events_lock:
            if event_id in self.processed_events:
                logging.info(f"[NotificationManager] Duplicate event {event_id} blocked.")
                return
            self.processed_events.add(event_id)

        # 1. Determine if the frame is tampered
        prediction = event.get("prediction", "").upper()
        is_frame_tampered = (prediction == "TAMPERED" or event.get("rule", "NORMAL") != "NORMAL" or prediction == "TAMPER")
        
        # Check settings
        telegram_enabled = TelegramSettings.is_telegram_enabled()

        if not telegram_enabled:
            updates = {
                "notification_status": "Disabled",
                "notification_provider": "Telegram",
                "notification_delivery_state": "DISABLED",
                "notification_timestamp": datetime.now().isoformat(),
                "notification_attempts": 0,
                "notification_error": "Telegram notifications disabled",
                "notification_latency_ms": 0.0
            }
            self._update_event_state(event_id, event, updates)
            return

        if not is_frame_tampered:
            updates = {
                "notification_status": "Suppressed (Normal Feed)",
                "notification_delivery_state": "SUPPRESSED",
                "notification_timestamp": datetime.now().isoformat()
            }
            self._update_event_state(event_id, event, updates)
            return

        # 2. Check Rate Limiter
        camera = event.get("camera_name", "Unknown")
        tamper_type = event.get("tamper_type", "Unknown")
        
        if self.rate_limiter.allow_notification(camera, tamper_type):
            updates = {
                "notification_status": "Sending",
                "notification_delivery_state": "SENDING",
                "notification_timestamp": datetime.now().isoformat()
            }
            self._update_event_state(event_id, event, updates)
            
            try:
                self.queue.put(event, block=False)
                print(f"[Notification] Event queued: {event_id}")
            except queue.Full:
                logging.error("[NotificationManager] Notification queue is full. Dropping notification.")
        else:
            logging.info(f"Notification Suppressed (Rate Limited): {camera} - {tamper_type}")
            updates = {
                "notification_status": "Suppressed",
                "notification_delivery_state": "SUPPRESSED",
                "notification_timestamp": datetime.now().isoformat(),
                "notification_attempts": 0,
                "notification_error": "Rate limit cooldown active",
                "notification_latency_ms": 0.0
            }
            self._update_event_state(event_id, event, updates)

    def _execute_notification(self, event: dict):
        event_id = event.get("event_id") or event.get("uuid")
        
        # Resolve target paths and copy the snapshot as requested
        camera = event.get("camera_name", "Gate-1")
        ts = event.get("timestamp", "Unknown")
        if "_" in ts:
            date_folder = ts.split("_")[0]
        else:
            date_folder = datetime.now().strftime("%Y-%m-%d")

        target_dir = os.path.join("storage", "snapshots", date_folder, camera)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "event.jpg")

        src_path = event.get("snapshot_path") or event.get("screenshot_path")
        if src_path and os.path.exists(src_path):
            try:
                shutil.copy2(src_path, target_path)
                logging.info(f"[NotificationManager] Copied snapshot to {target_path}")
            except Exception as e:
                logging.error(f"[NotificationManager] Failed to copy snapshot: {e}")

        # Send Telegram
        self._execute_telegram(event_id, event, target_path)

    def _execute_telegram(self, event_id: str, event: dict, photo_path: str):
        try:
            self.telegram_service.initialize()
        except Exception as e:
            logging.error(f"[NotificationManager] Telegram Init Failed: {e}")
            updates = {
                "notification_status": "Telegram Failed",
                "notification_provider": "Telegram",
                "notification_delivery_state": "FAILED",
                "notification_timestamp": datetime.now().isoformat(),
                "notification_error": f"Telegram Init Failed: {e}"
            }
            self._update_event_state(event_id, event, updates)
            return

        caption = self.telegram_service.format_message(event)
        
        retries = 3
        last_err = ""
        success = False
        res_json = {}
        start_time = time.perf_counter()

        for attempt in range(retries + 1):
            try:
                if os.path.exists(photo_path):
                    print(f"[Notification] Telegram sendPhoto() for event {event_id}")
                    res_json = self.telegram_service.send_photo(photo_path=photo_path, caption=caption)
                else:
                    print(f"[Notification] Telegram sendMessage() for event {event_id}")
                    res_json = self.telegram_service.send_message(text=caption)
                success = True
                break
            except Exception as e:
                last_err = str(e)
                logging.warning(f"[NotificationManager] Telegram attempt {attempt + 1} failed: {e}")
                if attempt < retries:
                    time.sleep(1)

        latency_ms = (time.perf_counter() - start_time) * 1000
        msg_id = ""
        if success and "result" in res_json:
            msg_id = str(res_json["result"].get("message_id", ""))
            print(f"[Notification] HTTP 200 | Message ID: {msg_id}")
        else:
            print(f"[Notification] Telegram Request Failed. Response: {last_err}")

        updates = {
            "notification_status": "Delivered" if success else "Failed",
            "notification_provider": "Telegram",
            "notification_delivery_state": "DELIVERED" if success else "FAILED",
            "notification_timestamp": datetime.now().isoformat(),
            "notification_attempts": retries + 1 if not success else 1,
            "notification_error": last_err or "None",
            "notification_latency_ms": round(latency_ms, 2),
            "message_sid": msg_id,
            "retry_count": retries if not success else 0,
            "last_error": last_err or "None"
        }
        self._update_event_state(event_id, event, updates)
        print("[Notification] History Updated")
        print("[Notification] Sidebar Updated")

    def _update_event_state(self, event_id: str, event: dict, updates: dict):
        """Updates in-memory history deques and disk JSON outputs."""
        from backend.services.event_service import EventService
        event_service = EventService()
        
        # 1. Update memory deque
        with event_service.deque_lock:
            for item in event_service.history_deque:
                if item.get("uuid") == event_id or item.get("event_id") == event_id:
                    item.update(updates)
                    
        # 2. Update individual event JSON file
        timestamp = event.get("timestamp")
        if timestamp and event_id:
            try:
                date_folder = timestamp.split("_")[0]
                json_path = os.path.join("storage", "events", date_folder, f"event_{timestamp}_{event_id[:8]}.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    data.update(updates)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
            except Exception as e:
                logging.error(f"[NotificationManager] JSON update error: {e}")
                    
        # Also update the latest_event.json shortcut
        latest_path = os.path.join(event_service.events_root_dir, "latest_event.json")
        if os.path.exists(latest_path):
            try:
                with open(latest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get("uuid") == event_id or data.get("event_id") == event_id:
                    data.update(updates)
                    with open(latest_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
            except Exception as e:
                logging.error(f"[NotificationManager] latest_event.json update error: {e}")
