from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

class CalibrationProgressResponse(BaseModel):
    running: bool = Field(description="Operational status execution marker")
    frames_processed: int = Field(description="Total valid frame windows ingested")
    frames_required: int = Field(description="Target matrix sizing criteria depth parameter")
    percentage: float = Field(description="Ingestion convergence limit completion curve percentage")
    elapsed_seconds: float = Field(description="Temporal processing offset window duration metric")
    estimated_remaining_seconds: float = Field(description="Calculated computational cycle delta bounds remainder")

class CalibrationInfoResponse(BaseModel):
    baseline_exists: bool = Field(description="True if an environmental baseline exists on disk")
    baseline_version: str = Field(description="Semantic layout structural tracking iteration value")
    creation_time: Optional[datetime] = Field(default=None, description="System creation clock trace")
    sample_count: int = Field(description="Total baseline calibration training population size")
    feature_count: int = Field(description="Total invariant properties registered in baseline array shape")
    camera_source: str = Field(description="Primary camera identifier target path metric")
    storage_path: str = Field(description="Failsafe local volume system path tracking reference")
    engine_status: str = Field(description="Current internal sequence state tracker status")

class PerFeatureStatMetrics(BaseModel):
    mean: float
    std: float
    median: float
    min: float
    max: float
    variance: float
    p05: float
    p95: float
    sample_count: int

class CalibrationFeatureResponse(BaseModel):
    features: Dict[str, PerFeatureStatMetrics] = Field(description="Granular component mapping profiles arrays maps structures")

class CalibrationResetResponse(BaseModel):
    success: bool = Field(description="True if deletion operation completed successfully")
    message: str = Field(description="Operational status resolution execution summary text")

class CalibrationReloadResponse(BaseModel):
    success: bool = Field(description="True if reloading internal caches finished successfully")
    loaded_features: List[str] = Field(description="List of invariant feature names verified in memory")
    baseline_timestamp: datetime = Field(description="Original deployment snapshot creation trace")

class ApiHistoryEntrySchema(BaseModel):
    timestamp: datetime
    endpoint: str
    duration_ms: float
    status: int

class ApiHistoryLogResponse(BaseModel):
    count: int
    entries: List[ApiHistoryEntrySchema]
