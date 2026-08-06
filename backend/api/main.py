from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from backend.config.settings import settings
from backend.config.logging import logger
from backend.core.startup import on_startup, on_shutdown
from backend.core.constants import (
    APP_NAME, APP_VERSION, API_PREFIX, STATUS_OK,
    MSG_NOT_FOUND, MSG_INTERNAL_ERROR, MSG_VALIDATION_ERROR
)

from backend.api.v1.routes import health, system, camera, cameras, inference, calibration, baseline, deviation, tamper, realtime
from backend.api.v1.websocket import live_stream

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    on_startup()
    yield
    on_shutdown()

tags_metadata = [
    {"name": "Health", "description": "System health and readiness checks."},
    {"name": "System", "description": "System information and versioning."},
    {"name": "Root", "description": "API entrypoint."}
]

app = FastAPI(
    title=APP_NAME,
    description="SpectraGuard v2 Core API",
    version=APP_VERSION,
    openapi_tags=tags_metadata,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "SpectraGuard Team"},
    license_info={"name": "Proprietary"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

class RootResponse(BaseModel):
    app: str
    version: str
    status: str
    documentation: str
    health_check: str

@app.get("/", response_model=RootResponse, tags=["Root"])
async def root_endpoint() -> RootResponse:
    return RootResponse(
        app=APP_NAME,
        version=APP_VERSION,
        status=STATUS_OK,
        documentation="/docs",
        health_check=f"{API_PREFIX}/health"
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Any) -> JSONResponse:
    logger.warning(f"404 Not Found: {request.url}")
    return JSONResponse(status_code=404, content={"error": MSG_NOT_FOUND})

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Any) -> JSONResponse:
    logger.error(f"500 Internal Error: {request.url} - {str(exc)}")
    return JSONResponse(status_code=500, content={"error": MSG_INTERNAL_ERROR})

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning(f"Validation Error at {request.url}: {exc.errors()}")
    return JSONResponse(status_code=422, content={"error": MSG_VALIDATION_ERROR, "details": exc.errors()})

app.include_router(health.router, prefix=API_PREFIX)
app.include_router(system.router, prefix=API_PREFIX)
app.include_router(camera.router, prefix=API_PREFIX)
app.include_router(cameras.router, prefix=API_PREFIX)
app.include_router(inference.router, prefix=API_PREFIX)
app.include_router(calibration.router, prefix=API_PREFIX)
app.include_router(baseline.router, prefix=API_PREFIX)
app.include_router(deviation.router, prefix=API_PREFIX)
app.include_router(tamper.router, prefix=API_PREFIX)
app.include_router(realtime.router, prefix=API_PREFIX)
app.include_router(live_stream.router)

from backend.api.routers import model_validation
app.include_router(model_validation.router)


from backend.api.routers import model_adaptation
app.include_router(model_adaptation.router)


from backend.api.routers import events as event_router
app.include_router(event_router.router)

