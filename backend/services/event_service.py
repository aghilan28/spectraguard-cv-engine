import os
import json
import uuid
import cv2
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

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

class NotificationProvider:
    def send(self, message: str): raise NotImplementedError

class ConsoleProvider(NotificationProvider):
    def send(self, message: str): print(f"[SMS ALERT] {message}")

class EventService:
    def __init__(self):
        self.snapshots_dir = "storage/snapshots"
        self.events_file = "storage/events/history.json"
        self.notifier = ConsoleProvider()
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.events_file), exist_ok=True)
        
    def handle_detection(self, camera_name, frame, prob, severity, drift, rule):
        event_id = str(uuid.uuid4())
        ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save Snapshot
        snap_name = f"{camera_name}_{ts}_tamper_{event_id[:8]}.jpg"
        snap_path = os.path.join(self.snapshots_dir, snap_name)
        cv2.imwrite(snap_path, frame)
        
        # Create Event
        event = DetectionEvent(
            uuid=event_id, camera_name=camera_name, timestamp=ts, prediction="Tamper",
            probability=prob, severity=severity, snapshot_path=snap_path, drift_score=drift, rule=rule
        )
        
        # Persist
        events = []
        if os.path.exists(self.events_file):
            with open(self.events_file, 'r') as f: events = json.load(f)
        events.append(event.dict())
        with open(self.events_file, 'w') as f: json.dump(events, f, indent=2)
            
        # Notify
        msg = f"TAMPER ALERT | Cam: {camera_name} | Time: {ts} | Sev: {severity}"
        self.notifier.send(msg)
        return event
