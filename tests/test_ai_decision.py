"""Validation suite for AI Decision Engine rules and mapping."""

import unittest

from src.spectraguard_cv_engine.ai.runtime.models import PredictionOutput
from src.spectraguard_cv_engine.ai.confidence.models import (
    ConfidenceOutput,
    ConfidenceTier,
)
from src.spectraguard_cv_engine.ai.decision.models import SeverityLevel
from src.spectraguard_cv_engine.ai.decision.engine import DecisionEngine


class TestDecisionEngine(unittest.TestCase):

    def test_critical_severity_mapping(self):
        # Tampered + High Confidence = CRITICAL
        pred = PredictionOutput(
            prediction=1, probability=0.95, latency_ms=1.0, timestamp_utc="z"
        )
        conf = ConfidenceOutput(
            raw_probability=0.95,
            calibrated_score=0.95,
            tier=ConfidenceTier.HIGH,
            is_ambiguous=False,
        )

        decision = DecisionEngine.evaluate(pred, conf)

        self.assertEqual(decision.severity, SeverityLevel.CRITICAL)
        self.assertTrue(decision.action_required)
        self.assertIn("High-confidence", decision.rationale)

    def test_elevated_severity_mapping(self):
        # Tampered + Medium/Low Confidence = ELEVATED
        pred = PredictionOutput(
            prediction=1, probability=0.75, latency_ms=1.0, timestamp_utc="z"
        )
        conf = ConfidenceOutput(
            raw_probability=0.75,
            calibrated_score=0.75,
            tier=ConfidenceTier.MEDIUM,
            is_ambiguous=False,
        )

        decision = DecisionEngine.evaluate(pred, conf)

        self.assertEqual(decision.severity, SeverityLevel.ELEVATED)
        self.assertTrue(decision.action_required)
        self.assertIn("MEDIUM", decision.rationale)

    def test_ambiguity_override_to_review(self):
        # Regardless of prediction, ambiguity forces REVIEW
        pred = PredictionOutput(
            prediction=1, probability=0.55, latency_ms=1.0, timestamp_utc="z"
        )
        conf = ConfidenceOutput(
            raw_probability=0.55,
            calibrated_score=0.55,
            tier=ConfidenceTier.MEDIUM,
            is_ambiguous=True,
        )

        decision = DecisionEngine.evaluate(pred, conf)

        self.assertEqual(decision.severity, SeverityLevel.REVIEW)
        self.assertTrue(decision.action_required)
        self.assertIn("ambiguity", decision.rationale.lower())

    def test_clear_severity_mapping(self):
        # Normal + High Confidence = CLEAR
        pred = PredictionOutput(
            prediction=0, probability=0.05, latency_ms=1.0, timestamp_utc="z"
        )
        conf = ConfidenceOutput(
            raw_probability=0.05,
            calibrated_score=0.05,
            tier=ConfidenceTier.HIGH,
            is_ambiguous=False,
        )

        decision = DecisionEngine.evaluate(pred, conf)

        self.assertEqual(decision.severity, SeverityLevel.CLEAR)
        self.assertFalse(decision.action_required)
        self.assertIn("No action required", decision.rationale)

    def test_low_confidence_normal_mapping(self):
        # Normal + Low/Medium Confidence = REVIEW
        pred = PredictionOutput(
            prediction=0, probability=0.25, latency_ms=1.0, timestamp_utc="z"
        )
        conf = ConfidenceOutput(
            raw_probability=0.25,
            calibrated_score=0.25,
            tier=ConfidenceTier.LOW,
            is_ambiguous=False,
        )

        decision = DecisionEngine.evaluate(pred, conf)

        self.assertEqual(decision.severity, SeverityLevel.REVIEW)
        self.assertTrue(decision.action_required)
        self.assertIn("LOW confidence", decision.rationale)


if __name__ == "__main__":
    unittest.main()
