"""
Events & Snapshots API
======================
Serves the *real* tamper events persisted by the EventService.

The EventService writes one JSON per detection into
``storage/events/YYYY-MM-DD/event_*.json`` plus a ``latest_event.json``
shortcut, and JPEG snapshots into ``storage/snapshots/``.

This router previously read a non-existent flat ``history.json``; it now walks
the actual on-disk event store so the web dashboard shows genuine detections
and can load the exact tamper screenshot for the Predict page.
"""
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/v1/events", tags=["Events & Alerts"])

EVENTS_ROOT = "storage/events"
SNAPSHOTS_ROOT = "storage/snapshots"


def _discover_event_files() -> List[str]:
    """Walk storage/events recursively for real event JSON files."""
    files: List[str] = []
    if not os.path.isdir(EVENTS_ROOT):
        return files
    for root, _dirs, names in os.walk(EVENTS_ROOT):
        for name in names:
            if name.startswith("event_") and name.endswith(".json"):
                files.append(os.path.join(root, name))
    return files


def _normalize_event(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a stored DetectionEvent into the frontend event contract."""
    event_uuid = str(raw.get("uuid") or raw.get("event_id") or "")
    snapshot_path = raw.get("snapshot_path") or raw.get("screenshot_path") or ""

    # Attach a resolvable URL for the screenshot so the frontend can <img> it.
    snapshot_url = ""
    if snapshot_path and os.path.exists(snapshot_path):
        snapshot_url = f"/api/v1/events/snapshot/{event_uuid}" if event_uuid else (
            f"/api/v1/events/snapshot/file?path={os.path.basename(snapshot_path)}"
        )

    timestamp = str(raw.get("timestamp") or "")
    iso_timestamp = timestamp.replace("_", "T") if timestamp else ""

    tamper_type = str(raw.get("tamper_type") or raw.get("rule") or "UNKNOWN_TAMPER")
    confidence = float(raw.get("confidence") or raw.get("probability") or 0.0)

    severity = str(raw.get("severity") or "MEDIUM")
    if severity == "HIGH":
        status = "anomalous"
    elif tamper_type == "NORMAL":
        status = "online"
    else:
        status = "anomalous"

    return {
        "id": event_uuid,
        "uuid": event_uuid,
        "camera": raw.get("camera_name") or "Unnamed Camera",
        "cameraName": raw.get("camera_name") or "Unnamed Camera",
        "event": f"{tamper_type} Detected",
        "description": f"Tamper signature {tamper_type} detected with {confidence:.1f}% confidence.",
        "tamper_type": tamper_type,
        "severity": severity,
        "confidence": round(confidence, 4),
        "probability": raw.get("probability"),
        "status": status,
        "timestamp": iso_timestamp or raw.get("timestamp"),
        "relativeTime": timestamp,
        "snapshot_path": snapshot_path,
        "snapshot_url": snapshot_url,
        "imageUrl": snapshot_url,
        "drift_score": raw.get("drift_score"),
        "notification_delivery_state": raw.get("notification_delivery_state", "PENDING"),
    }


def _load_all_events() -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for path in _discover_event_files():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict):
                continue
            # Filter out any leftover synthetic/test rows.
            if str(raw.get("camera_name", "")).lower() in {"test_cam", "test", "synthetic"}:
                continue
            events.append(_normalize_event(raw))
        except Exception:
            continue

    # Always include the latest shortcut first if it is not already covered.
    latest_path = os.path.join(EVENTS_ROOT, "latest_event.json")
    if os.path.exists(latest_path):
        try:
            with open(latest_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                event_uuid = str(raw.get("uuid") or raw.get("event_id") or "")
                if event_uuid and not any(e.get("uuid") == event_uuid for e in events):
                    events.append(_normalize_event(raw))
        except Exception:
            pass

    # Sort newest first (timestamps may be either ISO or underscore-separated).
    def _sort_key(evt: Dict[str, Any]) -> str:
        ts = str(evt.get("timestamp") or "")
        return ts.replace("_", " ")

    return sorted(events, key=_sort_key, reverse=True)


@router.get("/latest")
def get_latest_events(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the most recent real tamper detections (newest first)."""
    limit = max(1, min(limit, 100))
    return _load_all_events()[:limit]


@router.get("/snapshot/{event_uuid}")
def get_snapshot(event_uuid: str):
    """Return the JPEG screenshot captured at the moment tampering was detected."""
    events = _load_all_events()
    for evt in events:
        if evt.get("uuid") == event_uuid and evt.get("snapshot_path"):
            snap = evt["snapshot_path"]
            if os.path.exists(snap):
                return FileResponse(snap, media_type="image/jpeg")
    return {"error": "Snapshot not found for the requested event."}


@router.get("/snapshot/file")
def get_snapshot_by_name(filename: str = ""):
    """Fallback: resolve a snapshot purely by its filename."""
    if not filename or ".." in filename or "/" in filename:
        return {"error": "Invalid snapshot filename."}
    if not os.path.isdir(SNAPSHOTS_ROOT):
        return {"error": "No snapshots stored yet."}
    for name in os.listdir(SNAPSHOTS_ROOT):
        if name == filename:
            path = os.path.join(SNAPSHOTS_ROOT, name)
            if os.path.exists(path):
                return FileResponse(path, media_type="image/jpeg")
    return {"error": "Snapshot not found."}


@router.get("/count")
def get_event_count() -> Dict[str, int]:
    return {"count": len(_load_all_events())}
