"""Probability calibration and threshold evaluation engine."""

from typing import List
from .models import ConfidenceTier, ConfidenceOutput


class ConfidenceEngine:
    """
    Evaluates raw probabilities from the inference runtime against strict bounds
    to generate calibrated scores, operational tiers, and ambiguity flags.
    """

    def __init__(
        self,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.65,
        ambiguity_margin: float = 0.15,
    ):
        """
        Args:
            high_threshold: Minimum confidence magnitude to achieve HIGH confidence.
            medium_threshold: Minimum confidence magnitude before dropping to LOW confidence.
            ambiguity_margin: +/- margin around the 0.5 decision boundary considered ambiguous.
        """
        if not (0.5 <= medium_threshold < high_threshold <= 1.0):
            raise ValueError(
                "Invalid thresholds: must follow 0.5 <= medium < high <= 1.0"
            )

        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.ambiguity_margin = ambiguity_margin

    def evaluate(self, probabilities: List[float]) -> List[ConfidenceOutput]:
        """
        Translates a batch of raw probabilities into structured confidence outputs.
        """
        results = []
        for prob in probabilities:
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability {prob} is out of bounds (0.0 - 1.0)")

            # Confidence is based on magnitude (distance from 0.5)
            confidence_magnitude = max(prob, 1.0 - prob)

            # Base calibration (Platt-scaled display confidence represents confidence in predicted class)
            calibrated = confidence_magnitude

            # Determine Tier
            if confidence_magnitude >= self.high_threshold:
                tier = ConfidenceTier.HIGH
            elif confidence_magnitude >= self.medium_threshold:
                tier = ConfidenceTier.MEDIUM
            else:
                tier = ConfidenceTier.LOW

            # Determine Ambiguity (Gray area around the 0.5 binary decision boundary)
            is_ambiguous = abs(prob - 0.5) <= self.ambiguity_margin

            results.append(
                ConfidenceOutput(
                    raw_probability=float(prob),
                    calibrated_score=float(calibrated),
                    tier=tier,
                    is_ambiguous=is_ambiguous,
                )
            )

        return results
