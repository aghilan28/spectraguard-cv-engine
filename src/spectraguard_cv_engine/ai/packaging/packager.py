"""Aggregation and serialization logic for AI inferences."""

import uuid
from datetime import datetime, timezone
import pandas as pd
from typing import Optional, List

from ..runtime.models import PredictionOutput
from ..explainability.models import ExplanationOutput
from ..confidence.models import ConfidenceOutput
from ..decision.models import DecisionOutput
from .models import EventEvidence


class EvidencePackager:
    """
    Compiles individual AI layer outputs into a single, cohesive EventEvidence payload.
    """

    @staticmethod
    def package_event(
        prediction: PredictionOutput,
        confidence: ConfidenceOutput,
        decision: DecisionOutput,
        raw_features: pd.Series,
        explanation: Optional[ExplanationOutput] = None,
    ) -> EventEvidence:
        """
        Aggregates outputs into an immutable evidence record.

        Args:
            prediction: Raw inference output and latency.
            confidence: Calibrated probability and tier.
            decision: Final severity state and rationale.
            raw_features: The unscaled, original feature vector captured at inference.
            explanation: Optional SHAP feature attributions.

        Returns:
            A populated EventEvidence object.
        """
        # Generate a unique tracking ID for this specific inference event
        event_id = f"evt_{uuid.uuid4().hex[:12]}"

        # Convert Pandas Series to native dict for JSON serialization
        feature_dict = raw_features.to_dict()

        return EventEvidence(
            event_id=event_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            decision=decision,
            confidence=confidence,
            prediction=prediction,
            explainability=explanation,
            feature_snapshot={str(k): float(v) for k, v in feature_dict.items()},
        )

    @staticmethod
    def package_batch(
        predictions: List[PredictionOutput],
        confidences: List[ConfidenceOutput],
        decisions: List[DecisionOutput],
        raw_feature_matrix: pd.DataFrame,
        explanations: Optional[List[ExplanationOutput]] = None,
    ) -> List[EventEvidence]:
        """
        Helper method to aggregate a batch of inferences.
        """
        batch_size = len(predictions)
        if not (
            batch_size == len(confidences) == len(decisions) == len(raw_feature_matrix)
        ):
            raise ValueError(
                "All input lists and the feature matrix must have identical lengths."
            )

        if explanations and len(explanations) != batch_size:
            raise ValueError(
                "Explanations list length must match batch size if provided."
            )

        results = []
        for i in range(batch_size):
            exp = explanations[i] if explanations else None
            evt = EvidencePackager.package_event(
                prediction=predictions[i],
                confidence=confidences[i],
                decision=decisions[i],
                raw_features=raw_feature_matrix.iloc[i],
                explanation=exp,
            )
            results.append(evt)

        return results
