import platform
import sys
import os
import psutil
from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel
from backend.core.constants import APP_NAME, APP_VERSION, AUTHOR
from backend.config.settings import settings

router = APIRouter(tags=["System"])

class VersionResponse(BaseModel):
    app_name: str
    version: str
    author: str
    environment: str
    python_version: str
    fastapi_version: str
    operating_system: str
    current_time: str

class SystemInfoResponse(BaseModel):
    cpu_count: int
    ram_total_gb: float
    platform: str
    architecture: str
    hostname: str
    python_executable: str
    working_directory: str
    disk_free_gb: float
    memory_usage_percent: float

@router.get("/version", response_model=VersionResponse, status_code=200)
def get_version() -> VersionResponse:
    import fastapi
    return VersionResponse(
        app_name=APP_NAME,
        version=APP_VERSION,
        author=AUTHOR,
        environment="DEBUG" if settings.debug else "PRODUCTION",
        python_version=sys.version.split(" ")[0],
        fastapi_version=fastapi.__version__,
        operating_system=platform.system(),
        current_time=datetime.now(timezone.utc).isoformat()
    )

@router.get("/system/info", response_model=SystemInfoResponse, status_code=200)
def get_system_info() -> SystemInfoResponse:
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())
    return SystemInfoResponse(
        cpu_count=psutil.cpu_count(logical=True) or 0,
        ram_total_gb=round(ram.total / (1024**3), 2),
        platform=platform.system(),
        architecture=platform.machine(),
        hostname=platform.node(),
        python_executable=sys.executable,
        working_directory=os.getcwd(),
        disk_free_gb=round(disk.free / (1024**3), 2),
        memory_usage_percent=ram.percent
    )
