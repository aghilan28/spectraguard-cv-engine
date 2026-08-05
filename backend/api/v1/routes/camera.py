from fastapi import APIRouter, HTTPException, Response, Body
from typing import Optional
from backend.schemas.camera_schema import (
    StartCameraRequest, StopCameraRequest, CameraStatusResponse, 
    CameraInfoResponse, CameraActionResponse
)
from backend.stream.camera_manager import CameraManager
from backend.utils.image_utils import preprocess_frame, encode_jpeg
from backend.config.settings import settings

router = APIRouter(prefix="/camera", tags=["Camera Operations"])

@router.post("/start", response_model=CameraActionResponse)
def start_camera(payload: Optional[StartCameraRequest] = Body(default=None)) -> CameraActionResponse:
    # Resolve source: Payload input -> Settings file fallback
    source_target = str(settings.camera_index)
    if payload and payload.camera_source is not None:
        source_target = payload.camera_source
        
    manager = CameraManager(camera_id="default", source=source_target)
    if manager.start():
        return CameraActionResponse(
            success=True, 
            message=f"Camera subsystem active loop spawned using source: {source_target}", 
            camera_id="default"
        )
    raise HTTPException(status_code=500, detail="Hardware integration interface allocation failed.")

@router.post("/stop", response_model=CameraActionResponse)
def stop_camera(payload: Optional[StopCameraRequest] = Body(default=None)) -> CameraActionResponse:
    target_id = "default"
    if payload and payload.camera_id is not None:
        target_id = payload.camera_id
        
    manager = CameraManager(camera_id=target_id)
    manager.stop()
    return CameraActionResponse(success=True, message="Camera subsystem safely terminated.", camera_id=target_id)

@router.post("/restart", response_model=CameraActionResponse)
def restart_camera() -> CameraActionResponse:
    manager = CameraManager(camera_id="default")
    if manager.restart():
        return CameraActionResponse(success=True, message="Subsystem context reinitialized cleanly.", camera_id="default")
    raise HTTPException(status_code=500, detail="Reconnection routine failed to execute.")

@router.get("/status", response_model=CameraStatusResponse)
def get_status() -> CameraStatusResponse:
    manager = CameraManager(camera_id="default")
    return CameraStatusResponse(
        camera_id="default",
        is_opened=manager.is_running(),
        status="active" if manager.is_running() else "inactive",
        fps=manager.get_fps(),
        width=manager.width,
        height=manager.height,
        frame_count=manager.frame_count,
        uptime_seconds=manager.get_uptime()
    )

@router.get("/info", response_model=CameraInfoResponse)
def get_info() -> CameraInfoResponse:
    manager = CameraManager(camera_id="default")
    info = manager.get_info()
    return CameraInfoResponse(**info)

@router.get("/frame")
def get_raw_frame() -> Response:
    manager = CameraManager(camera_id="default")
    frame = manager.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="No valid capture data stored in matrix buffer stack yet.")
    
    processed = preprocess_frame(frame, target_size=None, add_timestamp=True)
    jpeg_bytes = encode_jpeg(processed, quality=80)
    return Response(content=jpeg_bytes, media_type="image/jpeg")
