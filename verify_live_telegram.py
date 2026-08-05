import os
import sys
import time
import uuid
import json
from datetime import datetime
import numpy as np
import cv2

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from backend.services.event_service import EventService
from backend.notifications.notification_manager import NotificationManager
from backend.notifications.telegram_settings import TelegramSettings

def verify():
    print("--- STARTING LIVE TELEGRAM INTEGRATION AUDIT ---")
    
    # 1. Start NotificationManager & EventService
    print("[Audit] Initializing services...")
    event_service = EventService()
    nm = NotificationManager()
    
    # Ensure Telegram configuration is valid
    if not TelegramSettings.validate_configuration():
        print("[FAIL] Telegram credentials missing or invalid in environment.")
        sys.exit(1)
        
    print("[Audit] Telegram Settings: VALID")

    # Create a synthetic tamper frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "SpectraGuard LIVE TEST", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Trigger synthetic tamper event
    camera_name = "Gate-1"
    tamper_type = "FULL_LENS_COVER"
    prob = 0.9984
    severity = "CRITICAL"
    
    print("[Audit] Dispatching synthetic tamper event through EventService...")
    # Clean the deduplication cache key for this camera/rule to prevent cache blocks
    std_rule = "FULL_LENS_COVER"
    with event_service.cache_lock:
        event_service.last_triggered[(camera_name, std_rule)] = 0.0

    event = event_service.handle_detection(
        camera_name=camera_name,
        frame=frame,
        prob=prob,
        severity=severity,
        drift=float(prob),
        rule=tamper_type
    )
    
    if not event:
        print("[FAIL] EventService discarded the detection event.")
        sys.exit(1)
        
    event_id = event.uuid
    print(f"[Audit] Synthetic event created: {event_id}")

    # 4. Wait for background worker to complete
    print("[Audit] Waiting for background NotificationWorker to send message...")
    max_wait = 20
    status_ok = False
    for i in range(max_wait):
        time.sleep(1)
        # Check in memory history
        history = event_service.get_history()
        for hist_event in history:
            if hist_event.get("uuid") == event_id:
                state = hist_event.get("notification_delivery_state")
                if state in ["DELIVERED", "FAILED"]:
                    status_ok = True
                    break
        if status_ok:
            break
            
    if not status_ok:
        print(f"[FAIL] Notification delivery state remained pending after {max_wait} seconds.")
    
    # 5. Verify snapshot exists on disk
    ts_utc = datetime.utcnow()
    date_folder = ts_utc.strftime("%Y-%m-%d")
    expected_jpg = os.path.join("storage", "snapshots", date_folder, camera_name, "event.jpg")
    
    print(f"[Audit] Verifying screenshot exists at: {expected_jpg}")
    if not os.path.exists(expected_jpg):
        print(f"[FAIL] Screenshot was not copied to target path: {expected_jpg}")
        sys.exit(1)
    print("PASS: Screenshot exists.")

    # 6. Verify Event JSON updated
    event_json_name = f"event_{event.timestamp}_{event_id[:8]}.json"
    expected_json = os.path.join("storage", "events", date_folder, event_json_name)
    
    print(f"[Audit] Verifying Event JSON exists at: {expected_json}")
    if not os.path.exists(expected_json):
        print(f"[FAIL] Event JSON was not written to: {expected_json}")
        sys.exit(1)
        
    with open(expected_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"[Audit] Verifying updated JSON states...")
    status = data.get("notification_status", "")
    provider = data.get("notification_provider", "")
    state = data.get("notification_delivery_state", "")
    message_id = data.get("message_sid", "")

    print(f"  notification_status: {status}")
    print(f"  notification_provider: {provider}")
    print(f"  notification_delivery_state: {state}")
    print(f"  message_id: {message_id}")

    if status.upper() != "DELIVERED":
        print(f"[FAIL] Expected notification_status to be DELIVERED, got: {status}")
        sys.exit(1)
        
    if provider != "Telegram":
        print(f"[FAIL] Expected notification_provider to be Telegram, got: {provider}")
        sys.exit(1)
        
    if state != "DELIVERED":
        print(f"[FAIL] Expected notification_delivery_state to be DELIVERED, got: {state}")
        sys.exit(1)
        
    if not message_id:
        print("[FAIL] Telegram message ID is missing or empty in JSON.")
        sys.exit(1)

    # 7. Verify in-memory sidebar updates
    print("[Audit] Verifying in-memory EventService history deque updates...")
    history = event_service.get_history()
    found = False
    for hist_event in history:
        if hist_event.get("uuid") == event_id:
            found = True
            if hist_event.get("notification_delivery_state") != "DELIVERED":
                print("[FAIL] In-memory event history was not updated to DELIVERED status.")
                sys.exit(1)
            break
            
    if not found:
        print("[FAIL] Synthetic event not found in memory history.")
        sys.exit(1)

    print("\n--- PHASE 8 TELEGRAM ENGINE VERIFIED ---")

if __name__ == "__main__":
    verify()
