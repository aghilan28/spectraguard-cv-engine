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
        low_threshold: float = 0.35,
        ambiguity_margin: float = 0.15,
    ):
        """
        Args:
            high_threshold: Minimum probability to achieve HIGH confidence.
            low_threshold: Maximum probability before dropping to LOW confidence.
            ambiguity_margin: +/- margin around the 0.5 decision boundary considered ambiguous.
        """
        if not (0.0 <= low_threshold < high_threshold <= 1.0):
            raise ValueError("Invalid thresholds: must follow 0.0 <= low < high <= 1.0")

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.ambiguity_margin = ambiguity_margin

    def evaluate(self, probabilities: List[float]) -> List[ConfidenceOutput]:
        """
        Translates a batch of raw probabilities into structured confidence outputs.

        Args:
            probabilities: List of raw probability floats (0.0 to 1.0).

        Returns:
            List of structured ConfidenceOutput instances.
        """
        results = []
        for prob in probabilities:
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability {prob} is out of bounds (0.0 - 1.0)")

            # Base calibration (passthrough for now, extensible for Platt scaling)
            calibrated = prob

            # Determine Tier
            if calibrated >= self.high_threshold:
                tier = ConfidenceTier.HIGH
            elif calibrated <= self.low_threshold:
                tier = ConfidenceTier.LOW
            else:
                tier = ConfidenceTier.MEDIUM

            # Determine Ambiguity (Gray area around the 0.5 binary decision boundary)
            is_ambiguous = abs(calibrated - 0.5) <= self.ambiguity_margin

            results.append(
                ConfidenceOutput(
                    raw_probability=float(prob),
                    calibrated_score=float(calibrated),
                    tier=tier,
                    is_ambiguous=is_ambiguous,
                )
            )

        return results
