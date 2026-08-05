"""
API response models enforcing data validation on external endpoints.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class FeatureDeviationSchema(BaseModel):
    name: str = Field(description="Exact physical feature name mapped to metadata")
    live: float = Field(description="Real-time feature scalar extraction")
    mean: float = Field(description="Calibrated environmental mean baseline")
    std: float = Field(description="Calibrated standard deviation mapping")
    z_score: float = Field(description="Mathematical variance bounds (Z-Score)")
    absolute_z: float = Field(description="Absolute vector magnitude of variance")
    normalized_drift: float = Field(description="Scaled anomaly severity [0-1] limited to 5 std devs")
    weight: float = Field(description="Architectural importance modifier")
    weighted_score: float = Field(description="Final normalized contribution to global metric")

class DriftReportResponse(BaseModel):
    timestamp: datetime = Field(description="Time of mathematical evaluation")
    global_score: float = Field(description="Composite weighted deviation across all sensors [0-1]")
    severity: str = Field(description="Categorical threat mapping (VERY_LOW to CRITICAL)")
    latency_ms: float = Field(description="Execution time mapped in milliseconds")
    features: List[FeatureDeviationSchema] = Field(description="Per-vector sub-metric profiling breakdown")

class BaselineStatusResponse(BaseModel):
    loaded: bool = Field(description="True if memory holds active baseline configuration")
    camera_id: Optional[str] = Field(default=None, description="Mapped camera source target")
    creation_time: Optional[datetime] = Field(default=None, description="Baseline snapshot genesis clock")
    feature_count: int = Field(description="Count of active invariant features in matrix")

class LiveFeatureInput(BaseModel):
    features: Dict[str, float] = Field(description="8D physical matrix payload")
