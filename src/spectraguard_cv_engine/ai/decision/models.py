"""Data schemas for AI Decision states."""

from enum import Enum
from dataclasses import dataclass


class SeverityLevel(str, Enum):
    """Final operational alert states."""

    CRITICAL = "CRITICAL"  # High-confidence tamper detection
    ELEVATED = "ELEVATED"  # Low-confidence tamper detection
    REVIEW = "REVIEW"  # Ambiguous boundary case requiring human oversight
    CLEAR = "CLEAR"  # High-confidence normal feed


@dataclass(frozen=True)
class DecisionOutput:
    """Standardized payload for the final deterministic decision."""

    severity: SeverityLevel
    action_required: bool
    rationale: str
