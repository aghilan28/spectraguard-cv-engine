"""Deterministic rule evaluation and severity mapping engine."""

from ..runtime.models import PredictionOutput
from ..confidence.models import ConfidenceOutput, ConfidenceTier
from .models import SeverityLevel, DecisionOutput


class DecisionEngine:
    """
    Synthesizes inference predictions and confidence tiers into final,
    actionable security severity levels based on strict deterministic rules.
    """

    @staticmethod
    def evaluate(
        prediction: PredictionOutput, confidence: ConfidenceOutput
    ) -> DecisionOutput:
        """
        Maps prediction and confidence state to a final operational decision.
        Assumes prediction == 1 implies "Tampered/Anomaly", prediction == 0 implies "Normal".

        Args:
            prediction: Standardized prediction payload from AI-1.
            confidence: Standardized confidence payload from AI-3.

        Returns:
            A deterministic DecisionOutput payload.
        """
        is_tampered = prediction.prediction == 1

        # 1. Handle Ambiguity Override
        # If the confidence engine flagged it as ambiguous, force a manual review state
        if confidence.is_ambiguous:
            return DecisionOutput(
                severity=SeverityLevel.REVIEW,
                action_required=True,
                rationale="Prediction falls within the statistical ambiguity margin.",
            )

        # 2. Handle Tampered Detections
        if is_tampered:
            if confidence.tier == ConfidenceTier.HIGH:
                return DecisionOutput(
                    severity=SeverityLevel.CRITICAL,
                    action_required=True,
                    rationale="High-confidence tamper signature detected.",
                )
            else:
                return DecisionOutput(
                    severity=SeverityLevel.ELEVATED,
                    action_required=True,
                    rationale=f"Tamper signature detected with {confidence.tier.value} confidence.",
                )

        # 3. Handle Normal Feed Detections
        else:
            if confidence.tier == ConfidenceTier.HIGH:
                return DecisionOutput(
                    severity=SeverityLevel.CLEAR,
                    action_required=False,
                    rationale="High-confidence normal feed. No action required.",
                )
            else:
                return DecisionOutput(
                    severity=SeverityLevel.REVIEW,
                    action_required=True,
                    rationale=f"Normal feed predicted, but with {confidence.tier.value} confidence. Requires verification.",
                )
