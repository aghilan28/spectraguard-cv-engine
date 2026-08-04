"""Master AI Intelligence Validation and Benchmarking Suite (Phase 7 Final)."""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from spectraguard_cv_engine.ml.data.loader import EXPECTED_UNIFIED_FEATURES
from spectraguard_cv_engine.ai.runtime.loader import ModelLoader
from spectraguard_cv_engine.ai.runtime.config import RuntimeConfig
from spectraguard_cv_engine.ai.runtime.engine import InferenceRuntime
from spectraguard_cv_engine.ai.explainability.engine import ExplainabilityEngine
from spectraguard_cv_engine.ai.confidence.engine import ConfidenceEngine
from spectraguard_cv_engine.ai.decision.engine import DecisionEngine
from spectraguard_cv_engine.ai.state.tracker import StateTracker
from spectraguard_cv_engine.ai.packaging.packager import EvidencePackager


def run_ai_integration_test() -> dict:
    print("\n[BENCHMARK] Executing Phase 7 AI Intelligence Validation...")

    # 1. Load the Phase 6 Production Model
    version_dir = os.path.normpath("data/models/releases/v0.7.5")
    if not os.path.exists(version_dir):
        raise FileNotFoundError(f"Required Phase 6 model missing at: {version_dir}")

    artifacts = ModelLoader.load_version(version_dir)
    print("[SUCCESS] Phase 6 Artifacts loaded successfully.")

    # 2. Initialize AI Subsystems
    runtime = InferenceRuntime(artifacts, RuntimeConfig())
    explainer = ExplainabilityEngine(artifacts.trainer)
    confidence_engine = ConfidenceEngine()
    state_tracker = StateTracker(cooldown_frames=5)

    def generate_realistic_features(size, is_tampered):
        data = []
        for _ in range(size):
            if not is_tampered:
                row = {
                    "mean_intensity": np.random.normal(loc=128.27, scale=10.0),
                    "variance_intensity": np.random.normal(loc=5427.49, scale=100.0),
                    "skewness": np.random.normal(loc=-0.0038, scale=0.01),
                    "kurtosis": np.random.normal(loc=-1.21, scale=0.05),
                    "mean_magnitude": np.random.normal(loc=22.37, scale=2.0),
                    "max_magnitude": np.random.normal(loc=586.49, scale=30.0),
                    "edge_density": np.random.normal(loc=0.124, scale=0.01),
                    "laplacian_variance": np.random.normal(loc=21.34, scale=2.0),
                    "global_contrast": np.random.normal(loc=73.67, scale=5.0),
                    "spectral_energy": np.random.normal(loc=4.34e10, scale=2.0e9),
                    "spectral_entropy": np.random.normal(loc=20.96, scale=0.5),
                    "spectral_flatness": np.random.normal(loc=0.9837, scale=0.005),
                    "mean_motion": np.random.normal(loc=4.09, scale=1.0),
                    "motion_variance": np.random.normal(loc=0.20, scale=0.05),
                    "temporal_instability": np.random.normal(loc=0.45, scale=0.1),
                }
            else:
                row = {
                    "mean_intensity": np.random.normal(loc=129.36, scale=10.0),
                    "variance_intensity": np.random.normal(loc=5256.73, scale=100.0),
                    "skewness": np.random.normal(loc=-0.0036, scale=0.01),
                    "kurtosis": np.random.normal(loc=-1.15, scale=0.05),
                    "mean_magnitude": np.random.normal(loc=2.86, scale=0.5),
                    "max_magnitude": np.random.normal(loc=108.0, scale=10.0),
                    "edge_density": np.random.normal(loc=0.005, scale=0.002),
                    "laplacian_variance": np.random.normal(loc=5.64, scale=1.0),
                    "global_contrast": np.random.normal(loc=72.50, scale=5.0),
                    "spectral_energy": np.random.normal(loc=3.58e10, scale=2.0e9),
                    "spectral_entropy": np.random.normal(loc=20.97, scale=0.5),
                    "spectral_flatness": np.random.normal(loc=0.9873, scale=0.005),
                    "mean_motion": np.random.normal(loc=1.05, scale=0.5),
                    "motion_variance": np.random.normal(loc=0.34, scale=0.05),
                    "temporal_instability": np.random.normal(loc=0.59, scale=0.1),
                }
            data.append(row)
        return pd.DataFrame(data, columns=EXPECTED_UNIFIED_FEATURES)

    # 3. Simulate a temporal stream of frames (Clear -> Attack -> Clear)
    np.random.seed(42)
    stream_size = 20
    # Simulate first 5 clear, 5 tampered, 10 clear
    X_clear1 = generate_realistic_features(5, is_tampered=False)
    X_attack = generate_realistic_features(5, is_tampered=True)
    X_clear2 = generate_realistic_features(10, is_tampered=False)

    stream_df = pd.concat([X_clear1, X_attack, X_clear2], ignore_index=True)
    stream_df["log_spectral_energy"] = np.log1p(np.abs(stream_df["spectral_energy"]))

    events_packaged = 0
    state_transitions = []

    print("\n[SIMULATION] Processing simulated video stream...")

    # 4. Process stream sequentially
    for i in range(stream_size):
        frame_df = stream_df.iloc[[i]]

        # Inference & Explainability
        pred_outputs = runtime.predict(frame_df)
        explanations = explainer.explain(artifacts.scaler.transform(frame_df), top_k=3)

        # Probability Calibration
        probs = [p.probability for p in pred_outputs]
        conf_outputs = confidence_engine.evaluate(probs)

        # Decision Mapping
        decision = DecisionEngine.evaluate(pred_outputs[0], conf_outputs[0])

        # State Tracking
        transition = state_tracker.process_decision(decision, event_id=f"sim_evt_{i}")
        if transition:
            state_transitions.append(
                {
                    "frame": i,
                    "from": transition.previous_state.value,
                    "to": transition.new_state.value,
                    "rationale": transition.rationale,
                }
            )

        # Packaging (Assigned to _ to satisfy ruff F841)
        _ = EvidencePackager.package_event(
            prediction=pred_outputs[0],
            confidence=conf_outputs[0],
            decision=decision,
            raw_features=frame_df.iloc[0],
            explanation=explanations[0],
        )
        events_packaged += 1

    print(f"[SUCCESS] Stream processed. Packaged {events_packaged} event records.")

    return {
        "events_processed": events_packaged,
        "final_system_state": state_tracker.current_state.value,
        "recorded_transitions": state_transitions,
    }


def main():
    print("================================================================")
    print(" SPECTRAGUARD PHASE 7: AI INTELLIGENCE VALIDATION               ")
    print("================================================================")

    try:
        report = run_ai_integration_test()

        # Verification Gates: We expect an Attack transition, a Cooldown transition, and a Nominal recovery.
        transitions = [t["to"] for t in report["recorded_transitions"]]
        passed = all(
            [
                "ACTIVE_EVENT" in transitions,
                "COOLDOWN" in transitions,
                "NOMINAL" in transitions,
                report["final_system_state"] == "NOMINAL",
            ]
        )

        status = "PASS" if passed else "FAIL_LIFECYCLE_VERIFICATION"

    except Exception as e:
        print(f"\n[ERROR] AI validation failed: {str(e)}")
        report = {"error": str(e)}
        status = "FAIL_EXECUTION"

    final_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE 7",
        "subsystem": "spectraguard-cv-engine-ai",
        "validation_report": report,
        "overall_status": "READY" if status == "PASS" else status,
    }

    report_path = os.path.normpath("data/reports/ai_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nEngineering Report generated at: {report_path}")
    print("================================================================")

    if final_report["overall_status"] == "READY":
        print("PHASE 7 STATUS: PASSED. AI Intelligence Layer is READY.")
        sys.exit(0)
    else:
        print(f"PHASE 7 STATUS: FAILED ({status}).")
        sys.exit(1)


if __name__ == "__main__":
    main()
