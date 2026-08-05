"""
Orchestrator synthesizing Inference and Deviation metrics into deterministic physical classifications.
Targets <2ms execution block using purely derived existing mathematics arrays.
"""
import time
from datetime import datetime, timezone
from backend.config.logging import logger
from backend.models.tamper_event import TamperEventModel
from backend.tamper.rule_engine import evaluate_rules
from backend.tamper.confidence import calculate_confidence, calculate_severity
from backend.tamper.explanation import generate_explanation

class TamperLogicEngine:
    def evaluate(self, inference_result, deviation_report) -> TamperEventModel:
        """
        Synthesizes outputs into final deterministic physical tamper classifications.
        
        Args:
            inference_result: Output from Phase 3 Inference Engine (Prediction + Prob)
            deviation_report: Output from Phase 4C Deviation Engine (Z-scores + Mahalanobis)
            
        Returns:
            TamperEventModel defining exact conditions and root causes.
        """
        start_t = time.perf_counter()
        
        # 1. Base Triage Rules evaluation
        winning_rule, rule_scores = evaluate_rules(deviation_report.feature_reports)
        best_rule_score = rule_scores.get(winning_rule, 0.0)
        
        # Override to NORMAL if both physical deviation and RF models indicate safe environments
        if winning_rule == "UNKNOWN_ANOMALY" and deviation_report.overall_score < 0.20 and inference_result.prediction == 0:
            winning_rule = "NORMAL"
            best_rule_score = 1.0

        if winning_rule == "NORMAL" and (inference_result.prediction == 1 or deviation_report.overall_score >= 0.40):
             winning_rule = "UNKNOWN_ANOMALY"

        # 2. Compile Confidence & Severity bounds
        confidence = calculate_confidence(
            rule_score=best_rule_score,
            dev_score=deviation_report.overall_score,
            rf_prob=inference_result.probability
        )
        
        severity = calculate_severity(deviation_report.overall_score, best_rule_score)
        
        if winning_rule == "NORMAL":
            severity = "LOW"
            
        # 3. Contextualize Explanation Strings
        explanation = generate_explanation(winning_rule)
        
        latency_ms = round((time.perf_counter() - start_t) * 1000, 3)

        event = TamperEventModel(
            timestamp=datetime.now(timezone.utc),
            tamper_type=winning_rule,
            severity=severity,
            confidence=round(confidence, 4),
            triggered_rules={k: round(v, 4) for k, v in rule_scores.items() if v > 0.0},
            explanation=explanation,
            deviation_score=deviation_report.overall_score,
            mahalanobis_distance=deviation_report.mahalanobis_distance,
            random_forest_prediction=inference_result.prediction,
            random_forest_probability=inference_result.probability,
            latency_ms=latency_ms
        )

        logger.debug(f"Tamper Engine | Type: {winning_rule} | Conf: {confidence:.2f} | Sev: {severity} | Latency: {latency_ms}ms")
        return event

tamper_engine = TamperLogicEngine()
