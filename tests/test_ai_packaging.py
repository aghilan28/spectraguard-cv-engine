"""Validation suite for AI Evidence Packaging."""

import json
import unittest
import pandas as pd

from src.spectraguard_cv_engine.ai.runtime.models import PredictionOutput
from src.spectraguard_cv_engine.ai.explainability.models import ExplanationOutput
from src.spectraguard_cv_engine.ai.confidence.models import (
    ConfidenceOutput,
    ConfidenceTier,
)
from src.spectraguard_cv_engine.ai.decision.models import SeverityLevel, DecisionOutput
from src.spectraguard_cv_engine.ai.packaging.models import EventEvidence
from src.spectraguard_cv_engine.ai.packaging.packager import EvidencePackager


class TestEvidencePackager(unittest.TestCase):
    def setUp(self):
        self.pred = PredictionOutput(
            prediction=1, probability=0.9, latency_ms=5.0, timestamp_utc="t"
        )
        self.conf = ConfidenceOutput(
            raw_probability=0.9,
            calibrated_score=0.9,
            tier=ConfidenceTier.HIGH,
            is_ambiguous=False,
        )
        self.dec = DecisionOutput(
            severity=SeverityLevel.CRITICAL, action_required=True, rationale="Test"
        )
        self.exp = ExplanationOutput(
            base_value=0.5,
            feature_attributions={"f1": 0.4},
            top_contributors={"f1": 0.4},
        )
        self.features = pd.Series({"f1": 10.5, "f2": 2.2})

    def test_single_event_packaging_and_serialization(self):
        event = EvidencePackager.package_event(
            prediction=self.pred,
            confidence=self.conf,
            decision=self.dec,
            raw_features=self.features,
            explanation=self.exp,
        )

        self.assertIsInstance(event, EventEvidence)
        self.assertTrue(event.event_id.startswith("evt_"))
        self.assertEqual(event.feature_snapshot["f1"], 10.5)

        # Test JSON Serialization
        json_str = event.to_json()
        payload = json.loads(json_str)

        self.assertEqual(payload["decision"]["severity"], "CRITICAL")
        self.assertEqual(payload["confidence"]["tier"], "HIGH")
        self.assertEqual(payload["prediction"]["prediction"], 1)
        self.assertEqual(payload["explainability"]["top_contributors"]["f1"], 0.4)

    def test_batch_packaging(self):
        df = pd.DataFrame({"f1": [10.5, 11.5], "f2": [2.2, 3.3]})

        events = EvidencePackager.package_batch(
            predictions=[self.pred, self.pred],
            confidences=[self.conf, self.conf],
            decisions=[self.dec, self.dec],
            raw_feature_matrix=df,
            explanations=[self.exp, self.exp],
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[1].feature_snapshot["f1"], 11.5)

    def test_batch_mismatch_rejection(self):
        df = pd.DataFrame({"f1": [10.5], "f2": [2.2]})

        with self.assertRaises(ValueError):
            EvidencePackager.package_batch(
                predictions=[self.pred, self.pred],  # Length 2
                confidences=[self.conf],  # Length 1
                decisions=[self.dec],
                raw_feature_matrix=df,
            )


if __name__ == "__main__":
    unittest.main()
