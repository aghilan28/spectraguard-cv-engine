import json
import os
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/v1/events", tags=["Events & Alerts"])
EVENTS_FILE = "storage/events/history.json"

@router.get("/latest")
def get_latest_events(limit: int = 10):
    if not os.path.exists(EVENTS_FILE): return []
    with open(EVENTS_FILE, 'r') as f:
        events = json.load(f)
    return sorted(events, key=lambda x: x["timestamp"], reverse=True)[:limit]

@router.get("/snapshot/{uuid}")
def get_snapshot(uuid: str):
    if not os.path.exists(EVENTS_FILE): return {"error": "No events found"}
    with open(EVENTS_FILE, 'r') as f: events = json.load(f)
    for e in events:
        if e["uuid"] == uuid and e["snapshot_path"]:
            return FileResponse(e["snapshot_path"])
    return {"error": "Snapshot not found"}
