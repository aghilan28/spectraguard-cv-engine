"""
Camera Registry API
===================
Real camera inventory endpoints consumed by the web dashboard.

``GET /api/v1/cameras``            -> live registry (real names/telemetry)
``POST /api/v1/cameras/register``  -> upsert a camera (called by GUI + web)
``POST /api/v1/cameras/{id}/heartbeat`` -> live telemetry updates
``DELETE /api/v1/cameras/{id}``    -> remove a camera record
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body

from backend.services.camera_registry import camera_registry
from backend.stream.camera_manager import CameraManager

router = APIRouter(prefix="/cameras", tags=["Camera Registry"])


def _live_state() -> Dict[str, Any]:
    """Snapshot the currently running CV camera manager (if any)."""
    try:
        manager = CameraManager(camera_id="default")
        running = manager.is_running()
        return {
            "camera_id": "default",
            "is_opened": running,
            "fps": manager.get_fps() if running else 0.0,
            "width": manager.width,
            "height": manager.height,
            "resolution": f"{manager.width}x{manager.height}" if manager.width else None,
            "uptime_seconds": manager.get_uptime() if running else 0.0,
        }
    except Exception:
        return {}


def _latest_probability() -> Optional[float]:
    """Pull the most recent inference probability so registry shows real integrity."""
    try:
        from backend.services.prediction_buffer import prediction_buffer
        latest = prediction_buffer.latest()
        if latest is not None:
            return float(getattr(latest, "probability", None) or 0.0)
    except Exception:
        pass
    return None


@router.get("")
def list_cameras() -> List[Dict[str, Any]]:
    """Return the full camera registry hydrated with live telemetry."""
    return camera_registry.list_all(
        live_state=_live_state(),
        latest_probability=_latest_probability(),
    )


@router.get("/all")
def list_cameras_alias() -> List[Dict[str, Any]]:
    """Alias for frontends hitting ``/cameras`` without trailing path normalization."""
    return list_cameras()


@router.post("/register")
def register_camera(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Create or update a camera record.

    The PyQt GUI calls this with the operator-given name on connect, e.g.::

        {
          "name": "Lobby Entrance",
          "location": "Main Lobby Port A",
          "vendor": "hikvision",
          "ip_address": "192.168.1.64",
          "port": 554,
          "source": "0"
        }
    """
    try:
        record = camera_registry.register(payload)
        return {"success": True, "camera": record}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Camera registration failed: {exc}")


@router.post("/{camera_id}/heartbeat")
def heartbeat_camera(camera_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Update live status/integrity for an existing camera."""
    status = payload.get("status", "online")
    integrity = payload.get("integrity_score")
    resolution = payload.get("resolution")
    fps = payload.get("fps")
    last_event = payload.get("last_event")
    tamper_count = payload.get("tamper_count")

    updated = camera_registry.heartbeat(
        camera_id=camera_id,
        status=status,
        integrity_score=integrity,
        resolution=resolution,
        fps=fps,
        last_event=last_event,
        tamper_count=tamper_count,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Camera not found in registry.")
    return {"success": True, "camera": updated}


@router.delete("/{camera_id}")
def delete_camera(camera_id: str) -> Dict[str, bool]:
    if not camera_registry.delete(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found in registry.")
    return {"success": True}
