"""Data schemas for AI Confidence scoring."""

from enum import Enum
from dataclasses import dataclass


class ConfidenceTier(str, Enum):
    """Operational confidence classifications."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class ConfidenceOutput:
    """Standardized payload for prediction confidence evaluation."""

    raw_probability: float
    calibrated_score: float
    tier: ConfidenceTier
    is_ambiguous: bool
