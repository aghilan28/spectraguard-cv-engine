from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import List, Optional

class RealTimeEventSchema(BaseModel):
    timestamp: datetime = Field(description="UTC execution confirmation timestamp")
    event_id: UUID = Field(description="Unique deterministic identifier (UUID4) generated for this verification cycle")
    prediction: int = Field(description="Binary classification result (0=Normal, 1=Tampered)")
    probability: float = Field(description="Raw probability scale from the inference pipeline")
    tamper_type: str = Field(description="Specific threat profile category classification matched by the rule analyzer")
    severity: str = Field(description="Categorical threat classification mapping")
    confidence: float = Field(description="Blended evaluation trust boundary score [0-1]")
    deviation_score: float = Field(description="Global environmental baseline variance metric")
    mahalanobis_distance: float = Field(description="Multivariate covariate-adjusted spatial distance indicator")
    latency_ms: float = Field(description="End-to-end execution latency mapping profile")

class RealTimeEngineStatusResponse(BaseModel):
    state: str = Field(description="Current engine execution lifecycle enum token")
    running: bool = Field(description="True if background scheduler evaluation loops are processing frames")
    cycle_interval_ms: int = Field(description="Sleep interval threshold bounded between evaluation loops")
    uptime: float = Field(description="Total execution running delta window in seconds")
    events_processed: int = Field(description="Total cycles processed during active life iteration")
    last_cycle_latency: float = Field(description="Latency baseline tracking of the last execution frame processing cycle")
    last_event_time: Optional[datetime] = Field(default=None, description="Timestamp tracking absolute last snapshot generation event")
    camera_connected: bool = Field(description="True if core video capture layer reports an open loop handle state")

class RealTimeStatisticsResponse(BaseModel):
    total_events: int
    normal_events: int
    tamper_events: int
    unknown_events: int
    average_latency: float
    average_confidence: float
    average_probability: float
    average_deviation: float
    uptime: float
