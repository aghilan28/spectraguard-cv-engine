import time
from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel
from backend.core.startup import BOOT_TIME
from backend.core.constants import APP_VERSION, HEALTH_HEALTHY

router = APIRouter(tags=["Health"])

class HealthResponse(BaseModel):
    status: str
    uptime: float
    version: str
    timestamp: str
    camera_status: bool
    storage_status: bool
    backend_ready: bool

@router.get("/health", response_model=HealthResponse, status_code=200)
def check_health() -> HealthResponse:
    uptime = time.time() - BOOT_TIME
    return HealthResponse(
        status=HEALTH_HEALTHY,
        uptime=round(uptime, 2),
        version=APP_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        camera_status=False,
        storage_status=True,
        backend_ready=True
    )
