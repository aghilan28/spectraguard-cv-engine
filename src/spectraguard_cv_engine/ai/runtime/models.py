"""Data schemas for inference inputs and outputs."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PredictionOutput:
    """Standardized and deterministic prediction payload."""

    prediction: int
    probability: Optional[float]
    latency_ms: float
    timestamp_utc: str
