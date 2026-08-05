from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class CalibrationStartRequest(BaseModel):
    """API request schema driving calibration initialization configurations."""
    target_frames: Optional[int] = Field(default=3000, ge=10, le=50000, description="Number of sample target vectors to pull for profile calculation")

class CalibrationStatusResponse(BaseModel):
    """API operational response status metadata payload schema."""
    running: bool = Field(description="True if the background ingestion thread is active")
    status: str = Field(description="Verbatim state descriptor string value")
    progress: float = Field(description="Formatted calibration window progress percentage profile")
    processed_frames: int = Field(description="Accumulated frame window arrays counts")
    target_frames: int = Field(description="Required window frames limit boundaries")
    elapsed_seconds: float = Field(description="Total operational clock time spent inside session")
    estimated_remaining_seconds: float = Field(description="Calculated execution loop duration remaining")

class FeatureStatMetrics(BaseModel):
    """Rigid statistical aggregation profile container schema for single feature invariants."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    p05: float
    p95: float
    variance: float
    sample_count: int

class BaselineProfileResponse(BaseModel):
    """System container metadata tracking the serialized calculated environment maps."""
    camera_id: str
    created_at: datetime
    frame_count: int
    features: Dict[str, FeatureStatMetrics]

class CalibrationActionResponse(BaseModel):
    """Generic payload return state tracking discrete service boundary changes requests."""
    success: bool
    message: str
    camera_id: str
