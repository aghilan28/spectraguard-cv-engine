from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict

class TamperEventResponse(BaseModel):
    timestamp: datetime = Field(description="Exact UTC confirmation execution clock mark")
    tamper_type: str = Field(description="Categorical physical classification of the event")
    severity: str = Field(description="Standardized severity categorical designation string")
    confidence: float = Field(description="Deterministic confidence score [0-1] of the classification")
    triggered_rules: Dict[str, float] = Field(description="Map of evaluated rule scores that triggered the decision")
    explanation: str = Field(description="Human-readable deterministic explanation of the anomaly")
    deviation_score: float = Field(description="Underlying global environmental drift metric")
    mahalanobis_distance: float = Field(description="Multivariate statistical distance from baseline")
    random_forest_prediction: int = Field(description="Binary classification from the inference engine")
    random_forest_probability: float = Field(description="Raw probability output from the inference engine")
    latency_ms: float = Field(description="Execution time of the tamper logic layer in milliseconds")

class TamperStatisticsResponse(BaseModel):
    total_events: int
    normal_events: int
    lens_cover_events: int
    spray_events: int
    defocus_events: int
    camera_move_events: int
    flash_events: int
    freeze_events: int
    noise_events: int
    partial_occlusion_events: int
    unknown_events: int
    average_confidence: float
    average_severity: str
