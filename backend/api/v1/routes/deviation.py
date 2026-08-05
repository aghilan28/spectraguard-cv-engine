from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from backend.schemas.deviation_schema import (
    DeviationReportResponse, DeviationStatisticsResponse, DeviationEvaluationRequest
)
from backend.deviation.deviation_engine import deviation_engine
from backend.deviation.history import deviation_history

router = APIRouter(prefix="/deviation", tags=["System Deviation Analytics"])

@router.post("/evaluate", response_model=DeviationReportResponse)
def evaluate_deviation(payload: DeviationEvaluationRequest) -> DeviationReportResponse:
    """Run real-time numerical spatial variance matrices tracking pipelines checks directly against environment configurations."""
    try:
        report = deviation_engine.evaluate(payload.feature_vector)
        deviation_history.push(report)
        return DeviationReportResponse(
            timestamp=report.timestamp,
            overall_score=report.overall_score,
            severity=report.severity,
            mahalanobis_distance=report.mahalanobis_distance,
            feature_reports=[r.__dict__ for r in report.feature_reports],
            latency_ms=report.latency_ms
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Multivariate deviation process matrix analysis crash: {err}")

@router.get("/latest", response_model=DeviationReportResponse)
def get_latest_deviation() -> DeviationReportResponse:
    """Fetch absolute latest stored anomaly trace record context maps fields tracker allocations."""
    report = deviation_history.latest()
    if not report:
        raise HTTPException(status_code=404, detail="No deviation metric history files compiled inside stack array frames yet.")
    return DeviationReportResponse(
        timestamp=report.timestamp,
        overall_score=report.overall_score,
        severity=report.severity,
        mahalanobis_distance=report.mahalanobis_distance,
        feature_reports=[r.__dict__ for r in report.feature_reports],
        latency_ms=report.latency_ms
    )

@router.get("/history", response_model=List[DeviationReportResponse])
def get_deviation_history() -> List[DeviationReportResponse]:
    """Retrieve full history listings collection metrics mappings data parameters trace indexes volumes logs tracks."""
    records = deviation_history.history()
    return [
        DeviationReportResponse(
            timestamp=r.timestamp,
            overall_score=r.overall_score,
            severity=r.severity,
            mahalanobis_distance=r.mahalanobis_distance,
            feature_reports=[f.__dict__ for f in r.feature_reports],
            latency_ms=r.latency_ms
        ) for r in records
    ]

@router.get("/statistics", response_model=DeviationStatisticsResponse)
def get_deviation_statistics() -> DeviationStatisticsResponse:
    """Compile aggregated runtime performance tracking coefficients trace values summary report matrix grids."""
    return DeviationStatisticsResponse(**deviation_history.statistics())

@router.delete("/history")
def clear_deviation_history() -> Dict[str, bool]:
    """Flush and empty the rolling analytical queue caches structures parameters completely."""
    deviation_history.clear()
    return {"success": True}
