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
    version_dir = os.path.normpath("data/models/releases/v0.6.0")
    if not os.path.exists(version_dir):
        raise FileNotFoundError(f"Required Phase 6 model missing at: {version_dir}")

    artifacts = ModelLoader.load_version(version_dir)
    print("[SUCCESS] Phase 6 Artifacts loaded successfully.")

    # 2. Initialize AI Subsystems
    runtime = InferenceRuntime(artifacts, RuntimeConfig())
    explainer = ExplainabilityEngine(artifacts.trainer)
    confidence_engine = ConfidenceEngine()
    state_tracker = StateTracker(cooldown_frames=5)

    # 3. Simulate a temporal stream of frames (Clear -> Attack -> Clear)
    np.random.seed(42)
    stream_size = 20
    # Simulate first 5 clear, 5 tampered, 10 clear
    X_clear1 = pd.DataFrame(
        np.random.normal(loc=-1.0, scale=0.5, size=(5, len(EXPECTED_UNIFIED_FEATURES))),
        columns=EXPECTED_UNIFIED_FEATURES,
    )
    X_attack = pd.DataFrame(
        np.random.normal(loc=1.5, scale=0.5, size=(5, len(EXPECTED_UNIFIED_FEATURES))),
        columns=EXPECTED_UNIFIED_FEATURES,
    )
    X_clear2 = pd.DataFrame(
        np.random.normal(
            loc=-1.0, scale=0.5, size=(10, len(EXPECTED_UNIFIED_FEATURES))
        ),
        columns=EXPECTED_UNIFIED_FEATURES,
    )

    stream_df = pd.concat([X_clear1, X_attack, X_clear2], ignore_index=True)

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
