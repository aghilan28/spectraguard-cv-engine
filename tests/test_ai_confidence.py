"""Validation suite for AI Confidence Engine."""

import unittest

from src.spectraguard_cv_engine.ai.confidence.models import (
    ConfidenceTier,
)
from src.spectraguard_cv_engine.ai.confidence.engine import ConfidenceEngine


class TestConfidenceEngine(unittest.TestCase):
    def setUp(self):
        # Ambiguity is 0.5 +/- 0.1 (0.4 to 0.6)
        self.engine = ConfidenceEngine(
            high_threshold=0.8, medium_threshold=0.6, ambiguity_margin=0.1
        )

    def test_invalid_initialization_bounds(self):
        with self.assertRaises(ValueError):
            ConfidenceEngine(high_threshold=0.7, medium_threshold=0.8)  # inverted

        with self.assertRaises(ValueError):
            ConfidenceEngine(high_threshold=1.5, medium_threshold=0.6)  # out of bounds

    def test_probability_out_of_bounds_rejection(self):
        with self.assertRaises(ValueError):
            self.engine.evaluate([1.1])

        with self.assertRaises(ValueError):
            self.engine.evaluate([-0.1])

    def test_confidence_tiers(self):
        # 0.95 -> Mag 0.95 (HIGH)
        # 0.70 -> Mag 0.70 (MEDIUM)
        # 0.55 -> Mag 0.55 (LOW)
        # 0.05 -> Mag 0.95 (HIGH Class 0)
        probs = [0.95, 0.70, 0.55, 0.05]
        results = self.engine.evaluate(probs)

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].tier, ConfidenceTier.HIGH)
        self.assertEqual(results[1].tier, ConfidenceTier.MEDIUM)
        self.assertEqual(results[2].tier, ConfidenceTier.LOW)
        self.assertEqual(results[3].tier, ConfidenceTier.HIGH)

    def test_ambiguity_detection(self):
        # 0.45 is within ambiguity margin (0.4 - 0.6)
        probs = [0.45, 0.70]
        results = self.engine.evaluate(probs)

        self.assertTrue(results[0].is_ambiguous)
        self.assertFalse(results[1].is_ambiguous)


if __name__ == "__main__":
    unittest.main()
