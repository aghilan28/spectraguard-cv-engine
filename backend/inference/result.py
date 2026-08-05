from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict

class InferenceResult(BaseModel):
    timestamp: datetime = Field(description="Exact UTC time of the inference execution")
    probability: float = Field(description="Raw model probability for the positive (tampered) class")
    prediction: int = Field(description="Binary classification result (0 = Normal, 1 = Tampered)")
    confidence: float = Field(description="Absolute confidence of the prediction (distance from threshold)")
    threshold: float = Field(description="Operating threshold used for this decision")
    latency_ms: float = Field(description="End-to-end execution time in milliseconds")
    feature_vector: Dict[str, float] = Field(description="Raw extracted physics features mapped by name")
    camera_id: str = Field(description="Originating camera stream identifier")
