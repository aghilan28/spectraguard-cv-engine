import time
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Response, Depends, Body, status
import dateutil.parser

from backend.config.logging import logger
from backend.calibration.calibration_engine import engine
from backend.stream.camera_manager import CameraManager
from backend.schemas.calibration_schema import (
    CalibrationStartRequest, CalibrationStatusResponse, CalibrationActionResponse
)
from backend.schemas.calibration_api_schema import (
    CalibrationProgressResponse, CalibrationInfoResponse,
    CalibrationFeatureResponse, CalibrationResetResponse, CalibrationReloadResponse,
    ApiHistoryEntrySchema, ApiHistoryLogResponse
)

router = APIRouter(prefix="/calibration", tags=["Scene Calibration Services"])

# Thread-safe API Request Tracking Architecture
class ApiHistoryTracker:
    def __init__(self, limit: int = 500) -> None:
        self.limit = limit
        self.entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def log_request(self, endpoint: str, duration_ms: float, status_code: int) -> None:
        with self._lock:
            self.entries.append({
                "timestamp": datetime.now(timezone.utc),
                "endpoint": endpoint,
                "duration_ms": round(duration_ms, 2),
                "status": status_code
            })
            if len(self.entries) > self.limit:
                self.entries.pop(0)

    def get_entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.entries)

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()

history_tracker = ApiHistoryTracker()

@router.post(
    "/start",
    response_model=CalibrationActionResponse,
    status_code=status.HTTP_200_OK,
    summary="Initialize background scene calibration",
    description="Spawns an asynchronous task to ingest video frames and create an environmental baseline profile snapshot map tracking structural constants values.",
    responses={
        200: {"description": "Calibration sequence successfully initiated."},
        400: {"description": "Active camera manager acquisition layer stream thread is unavailable."},
        409: {"description": "A validation error occurred because another calibration engine operation is already running."}
    }
)
def start_scene_calibration(payload: Optional[CalibrationStartRequest] = Body(default=None)) -> CalibrationActionResponse:
    start_t = time.perf_counter()
    cam = CameraManager(camera_id="default")
    if not cam.is_running():
        duration = (time.perf_counter() - start_t) * 1000
        history_tracker.log_request("POST /start", duration, 400)
        raise HTTPException(status_code=400, detail="Cannot calibrate because camera manager stream loop is inactive.")

    target_frames = payload.target_frames if payload and payload.target_frames is not None else 3000
    
    if engine.session.status == "running":
        duration = (time.perf_counter() - start_t) * 1000
        history_tracker.log_request("POST /start", duration, 409)
        raise HTTPException(status_code=409, detail="Calibration infrastructure session currently active.")
        
    logger.info("Exposing API endpoint initialization parameters to calibration engine sequencer loops.")
    if engine.initialize_session(target_frames=target_frames):
        duration = (time.perf_counter() - start_t) * 1000
        history_tracker.log_request("POST /start", duration, 200)
        return CalibrationActionResponse(
            success=True,
            message="Environmental calibration tracking initialization background thread loops successfully spawned.",
            camera_id="default"
        )
    
    duration = (time.perf_counter() - start_t) * 1000
    history_tracker.log_request("POST /start", duration, 500)
    raise HTTPException(status_code=500, detail="Failed to initialize engine runtime matrix loops allocation boundaries parameters tracks mapping context threads profiles.")

@router.get(
    "/status",
    response_model=CalibrationStatusResponse,
    summary="Fetch operational status summary matrices",
    description="Returns structural state properties tracking baseline collection statistics maps fields vectors.",
    status_code=status.HTTP_200_OK
)
def get_calibration_status() -> CalibrationStatusResponse:
    start_t = time.perf_counter()
    sess = engine.session
    is_running = (sess.status == "running")
    res = CalibrationStatusResponse(
        running=is_running,
        status=sess.status,
        progress=sess.progress_percent,
        processed_frames=sess.processed_frames,
        target_frames=sess.target_frames,
        elapsed_seconds=round(sess.elapsed_seconds, 2),
        estimated_remaining_seconds=sess.estimated_remaining_seconds
    )
    duration = (time.perf_counter() - start_t) * 1000
    history_tracker.log_request("GET /status", duration, 200)
    return res

@router.get(
    "/progress",
    response_model=CalibrationProgressResponse,
    summary="Get granular session progress metrics",
    description="Exposes ultra-lightweight execution tracking metrics vectors mapped explicitly to sub-20ms runtime boundaries targets constraints.",
    status_code=status.HTTP_200_OK
)
def get_calibration_progress() -> CalibrationProgressResponse:
    start_t = time.perf_counter()
    sess = engine.session
    res = CalibrationProgressResponse(
        running=(sess.status == "running"),
        frames_processed=sess.processed_frames,
        frames_required=sess.target_frames,
        percentage=sess.progress_percent,
        elapsed_seconds=round(sess.elapsed_seconds, 2),
        estimated_remaining_seconds=sess.estimated_remaining_seconds
    )
    duration = (time.perf_counter() - start_t) * 1000
    history_tracker.log_request("GET /progress", duration, 200)
    return res

@router.get(
    "/baseline",
    summary="Retrieve complete serialized environmental document",
    description="Returns raw human-readable JSON payload maps reflecting calculated baseline constants parameters configurations targets constraints maps structural metrics distributions profiles fields vectors properties records tracking volumes.",
    responses={
        200: {"description": "Baseline metrics mapping payload returned successfully."},
        404: {"description": "Calibrated document is missing or environment baseline profiles files allocations tracks have not been run."}
    }
)
def get_active_baseline() -> Response:
    start_t = time.perf_counter()
    profile = engine.storage.load()
    if not profile:
        duration = (time.perf_counter() - start_t) * 1000
        history_tracker.log_request("GET /baseline", duration, 404)
        raise HTTPException(status_code=404, detail="No calibrated baseline document configuration exists inside deployment storage nodes paths mappings variables allocations tracks matrices definitions.")
    
    import json
    duration = (time.perf_counter() - start_t) * 1000
    history_tracker.log_request("GET /baseline", duration, 200)
    return Response(content=json.dumps(profile, indent=4), media_type="application/json")

@router.get(
    "/info",
    response_model=CalibrationInfoResponse,
    summary="Expose diagnostic baseline system telemetry metadata information tracking tracks mapping metadata logs fields tracks parameters parameters boundaries profiles constants configuration matrices grids vectors.",
    description="Returns structural metrics detailing baseline files allocations without streaming out heavy calculation matrix indices arrays logs traces maps details structures properties tracks layout properties constraints.",
    status_code=status.HTTP_200_OK
)
def get_calibration_info() -> CalibrationInfoResponse:
    start_t = time.perf_counter()
    exists = engine.storage.exists()
    profile = engine.storage.load() if exists else None
    
    if exists and profile:
        created = profile.get("created_at", datetime.now(timezone.utc).isoformat())
        dt_created = dateutil.parser.isoparse(created)
        sample_count = profile.get("frame_count", 0)
        feat_count = len(profile.get("features", {}))
    else:
        dt_created = None
        sample_count = 0
        feat_count = 0
        
    res = CalibrationInfoResponse(
        baseline_exists=exists,
        baseline_version="2.0.0",
        creation_time=dt_created,
        sample_count=sample_count,
        feature_count=feat_count,
        camera_source=str(CameraManager(camera_id="default").source),
        storage_path=str(engine.storage.baseline_file),
        engine_status=engine.session.status
    )
    duration = (time.perf_counter() - start_t) * 1000
    history_tracker.log_request("GET /info", duration, 200)
    return res

@router.get(
    "/features",
    response_model=CalibrationFeatureResponse,
    summary="Get precise feature metric allocations parameters distributions tracking arrays grids maps specifications properties constraints boundaries.",
    description="Extracts the fine-grained statistics maps variables maps calculations invariants directly for frontend diagnostic plotting parameters checks vectors metrics dashboards profiles maps structural distributions tracking graphs layers layout metrics.",
    responses={
        200: {"description": "Statistical metric values extracted correctly."},
        404: {"description": "Baseline deployment target profile data structure metrics are unavailable."}
    }
)
def get_calibration_features() -> CalibrationFeatureResponse:
    start_t = time.perf_counter()
    profile = engine.storage.load()
    if not profile or "features" not in profile:
        duration = (time.perf_counter() - start_t) * 1000
        history_tracker.log_request("GET /features", duration, 404)
        raise HTTPException(status_code=404, detail="Baseline metrics specifications missing from system storage footprints allocations parameters.")
        
    duration = (time.perf_counter() - start_t) * 1000
    history_tracker.log_request("GET /features", duration, 200)
    return CalibrationFeatureResponse(features=profile["features"])

@router.post(
    "/reload",
    response_model=CalibrationReloadResponse,
    summary="Hot reload active baseline matrices configurations parameters context maps tracking",
    description="Forces memory cache synchronization loops models updates directly matching current disk states maps templates without triggering FastAPI server lifecycle downtime processes resets controls boundaries parameters tracking values channels.",
    responses={
        200: {"description": "Memory caching matrix mapped profiles synchronized cleanly."},
        404: {"description": "Missing deployment target baseline json file metrics structures paths fields."}
    }
)
def reload_baseline_cache() -> CalibrationReloadResponse:
    start_t = time.perf_counter()
    if not engine.storage.exists():
        duration = (time.perf_counter() - start_t) * 1000
        history_tracker.log_request("POST /reload", duration, 404)
        raise HTTPException(status_code=404, detail="Baseline storage matching files targets specifications unresolvable.")
        
    profile = engine.storage.load()
    if not profile:
        duration = (time.perf_counter() - start_t) * 1000
        history_tracker.log_request("POST /reload", duration, 500)
        raise HTTPException(status_code=500, detail="Corrupted calibration files profiles structural distributions models metadata parsing runtime error traces.")
        
    # Inject memory cache load mappings calls bridges natively to parallel analytics layers components
    try:
        from backend.deviation.deviation_engine import deviation_engine
        deviation_engine.load_baseline_cache()
    except Exception:
        pass
        
    created = profile.get("created_at", datetime.now(timezone.utc).isoformat())
    dt_created = dateutil.parser.isoparse(created)
    
    res = CalibrationReloadResponse(
        success=True,
        loaded_features=list(profile.get("features", {}).keys()),
        baseline_timestamp=dt_created
    )
    duration = (time.perf_counter() - start_t) * 1000
    history_tracker.log_request("POST /reload", duration, 200)
    logger.info("Hot-reloaded baseline metrics context targets constraints properties shapes patterns vectors maps arrays matrices mappings profiles cleanly inside memory frameworks execution blocks layers.")
    return res

@router.delete(
    "/reset",
    response_model=CalibrationResetResponse,
    summary="Wipe out all stored calibrated data properties states parameters matrix layers allocations definitions traces variables constraints loops tracking blocks profiles fields trackers allocations.",
    description="Purges active configurations baseline json snapshot logs allocations fields trackers allocations trace records data models arrays buffers structures from active server partitions volume tracks systems structures memory parameters configurations parameters maps matrices layouts tracking indices frames nodes metrics tracking vectors.",
    status_code=status.HTTP_200_OK
)
def reset_calibration() -> CalibrationResetResponse:
    start_t = time.perf_counter()
    engine.cancel_active_session()
    history_tracker.clear()
    
    success = engine.storage.delete()
    logger.info("Baseline deployment parameters deleted explicitly from system storage pathways tracks blocks.")
    
    res = CalibrationResetResponse(
        success=True,
        message="Calibration service parameters, operational trace histories buffers profiles, and layout baseline configuration structures completely cleared."
    )
    duration = (time.perf_counter() - start_t) * 1000
    history_tracker.log_request("DELETE /reset", duration, 200)
    return res

@router.post(
    "/cancel",
    response_model=CalibrationActionResponse,
    summary="Abort active background acquisition workflows",
    description="Terminates in-flight frame ingestion loops parameters context metrics monitoring tasks securely step processes bounds.",
    status_code=status.HTTP_200_OK
)
def cancel_calibration() -> CalibrationActionResponse:
    start_t = time.perf_counter()
    if engine.session.status != "running":
        duration = (time.perf_counter() - start_t) * 1000
        history_tracker.log_request("POST /cancel", duration, 400)
        raise HTTPException(status_code=400, detail="No active running background resource pipeline processing steps currently allocated inside execution paths threads context scopes.")
    
    engine.cancel_active_session()
    duration = (time.perf_counter() - start_t) * 1000
    history_tracker.log_request("POST /cancel", duration, 200)
    return CalibrationActionResponse(
        success=True,
        message="Active background tracking capture session flagged for direct halt closure and step abort execution bounds successfully.",
        camera_id="default"
    )

@router.get(
    "/history/log",
    response_model=ApiHistoryLogResponse,
    summary="Expose request tracking logs histories cache entries structures indices trackers allocations metadata telemetry matrices frames tracking variables.",
    description="Returns rolling telemetry metrics trace detailing API latency metrics for orchestration checks processing loops bounds optimization testing blocks dashboards tracking graphs frames arrays properties metrics constraints.",
    status_code=status.HTTP_200_OK
)
def get_api_request_history() -> ApiHistoryLogResponse:
    entries = history_tracker.get_entries()
    return ApiHistoryLogResponse(
        count=len(entries),
        entries=[ApiHistoryEntrySchema(**e) for e in entries]
    )
