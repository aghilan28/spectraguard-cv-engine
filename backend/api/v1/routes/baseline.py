"""
REST Endpoints for the Baseline Comparison Engine subsystem.
"""
from fastapi import APIRouter, HTTPException, Response
from typing import List, Dict, Any
import json
import dateutil.parser

from backend.calibration.baseline_loader import baseline_loader
from backend.calibration.baseline_comparator import baseline_comparator
from backend.schemas.drift_schema import (
    DriftReportResponse, LiveFeatureInput, BaselineStatusResponse
)

router = APIRouter(prefix="/baseline", tags=["Baseline Deviation Metrics"])

@router.get("", response_class=Response)
def get_raw_baseline() -> Response:
    """Return raw memory layout of environmental constraints."""
    data = baseline_loader.get_data()
    if not data:
        raise HTTPException(status_code=404, detail="No active baseline matrix cached in operational memory layers.")
    return Response(content=json.dumps(data, indent=4), media_type="application/json")

@router.get("/status", response_model=BaselineStatusResponse)
def get_baseline_status() -> BaselineStatusResponse:
    """Expose telemetry tracking statistics mappings on operational state configuration."""
    data = baseline_loader.get_data()
    if not data:
        return BaselineStatusResponse(loaded=False, feature_count=0)
        
    created = data.get("created_at")
    dt_created = dateutil.parser.isoparse(created) if created else None
    
    return BaselineStatusResponse(
        loaded=True,
        camera_id=data.get("camera_id"),
        creation_time=dt_created,
        feature_count=len(data.get("features", {}))
    )

@router.post("/compare", response_model=DriftReportResponse)
def run_live_comparison(payload: LiveFeatureInput) -> DriftReportResponse:
    """Pass live physical attributes vectors through absolute deviation math layers."""
    if not baseline_loader.exists():
        raise HTTPException(status_code=400, detail="Cannot run deviation checks because environment is uncalibrated.")
    
    try:
        report = baseline_comparator.compare(payload.features)
        return DriftReportResponse(
            timestamp=report.timestamp,
            global_score=report.global_score,
            severity=report.severity,
            latency_ms=report.latency_ms,
            features=report.features
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison evaluation crash faults: {e}")

@router.get("/latest", response_model=DriftReportResponse)
def get_latest_comparison() -> DriftReportResponse:
    """Return the absolute last mathematical divergence array map."""
    report = baseline_comparator.get_latest()
    if not report:
        raise HTTPException(status_code=404, detail="Historical queue bounds are entirely empty.")
    return DriftReportResponse(
        timestamp=report.timestamp,
        global_score=report.global_score,
        severity=report.severity,
        latency_ms=report.latency_ms,
        features=report.features
    )

@router.get("/history", response_model=List[DriftReportResponse])
def get_comparison_history() -> List[DriftReportResponse]:
    """Retrieve full rolling buffer arrays mappings allocations tracks configurations layers vectors sets sequences scopes flows targets lists paths indices variables fields bounds nodes matrices constraints limits."""
    reports = baseline_comparator.get_history()
    return [
        DriftReportResponse(
            timestamp=r.timestamp,
            global_score=r.global_score,
            severity=r.severity,
            latency_ms=r.latency_ms,
            features=r.features
        ) for r in reports
    ]

@router.delete("/history")
def clear_comparison_history() -> Dict[str, bool]:
    """Flush the memory rolling trace buffers cleanly."""
    baseline_comparator.clear_history()
    return {"success": True}
