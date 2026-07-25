"""Data schemas for aggregated AI Event Evidence."""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from ..runtime.models import PredictionOutput
from ..explainability.models import ExplanationOutput
from ..confidence.models import ConfidenceOutput
from ..decision.models import DecisionOutput


@dataclass(frozen=True)
class EventEvidence:
    """
    The immutable, complete historical record of an AI inference cycle,
    ready for backend ingestion or frontend display.
    """

    event_id: str
    timestamp_utc: str
    decision: DecisionOutput
    confidence: ConfidenceOutput
    prediction: PredictionOutput
    explainability: Optional[ExplanationOutput]
    feature_snapshot: Dict[str, float]

    def to_json(self) -> str:
        """Serializes the complete evidence package to a JSON string."""
        return json.dumps(asdict(self), default=str)
