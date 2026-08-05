from pydantic import BaseModel, Field
from typing import Optional

class StartCameraRequest(BaseModel):
    camera_source: Optional[str] = Field(default=None, description="Optional camera index or RTSP URL. Defaults to configuration file if null.")

class StopCameraRequest(BaseModel):
    camera_id: Optional[str] = Field(default=None, description="Optional identification context of target camera.")

class CameraStatusResponse(BaseModel):
    camera_id: str
    is_opened: bool
    status: str
    fps: float
    width: int
    height: int
    frame_count: int
    uptime_seconds: float

class CameraInfoResponse(BaseModel):
    camera_id: str
    opencv_backend: str
    camera_source: str
    max_fps: float
    resolution: str

class CameraActionResponse(BaseModel):
    success: bool
    message: str
    camera_id: str
