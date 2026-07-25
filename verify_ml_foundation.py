"""Master ML Foundation Validation and Benchmarking Suite (Milestone A)."""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from spectraguard_cv_engine.ml.preprocessing.scaler import FeatureScaler
from spectraguard_cv_engine.ml.models.config import TrainingConfig
from spectraguard_cv_engine.ml.models.trainer import ModelTrainer
from spectraguard_cv_engine.ml.evaluation.evaluator import ModelEvaluator
from spectraguard_cv_engine.ml.export.exporter import ModelExporter
from spectraguard_cv_engine.ml.data.loader import EXPECTED_UNIFIED_FEATURES


def run_ml_benchmark() -> dict:
    print("\n[BENCHMARK] Executing Phase 6 ML Foundation Validation (Milestone A)...")

    # 1. Generate highly realistic synthetic feature matrix conforming to Phase 4
    np.random.seed(42)
    sample_size = 1000

    # Simulate Class 0 (Normal) and Class 1 (Tampered)
    X_0 = np.random.normal(
        loc=0.0, scale=1.0, size=(sample_size // 2, len(EXPECTED_UNIFIED_FEATURES))
    )
    X_1 = np.random.normal(
        loc=1.5, scale=1.0, size=(sample_size // 2, len(EXPECTED_UNIFIED_FEATURES))
    )

    X_df = pd.DataFrame(np.vstack([X_0, X_1]), columns=EXPECTED_UNIFIED_FEATURES)
    y_series = pd.Series([0] * (sample_size // 2) + [1] * (sample_size // 2))

    # 2. Fit Scaler
    scaler = FeatureScaler(method="standard")
    X_scaled = scaler.fit_transform(X_df, EXPECTED_UNIFIED_FEATURES)

    # 3. Train Baseline XGBoost
    config = TrainingConfig(
        model_type="xgboost",
        random_seed=42,
        hyperparameters={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
    )
    trainer = ModelTrainer(config)
    trainer.train(X_scaled, y_series)

    # 4. Evaluate (Using training data just for pipeline integrity validation)
    report = ModelEvaluator.evaluate(trainer, X_scaled, y_series)

    # 5. Export Version
    export_dir = os.path.normpath("data/models/releases")
    version_dir = ModelExporter.export_pipeline(
        trainer=trainer,
        scaler=scaler,
        evaluation_report=report,
        export_dir=export_dir,
        version="v0.6.0",
    )

    print(
        f"BENCHMARK RESULT: Accuracy = {report['metrics']['accuracy']:.4f} | F1 = {report['metrics']['f1_score']:.4f}"
    )
    print(
        f"INFERENCE LATENCY: {report['performance']['avg_inference_ms_per_sample']:.4f} ms per frame"
    )
    print(f"MODEL EXPORTED TO: {version_dir}")

    return report


def main():
    print("================================================================")
    print(" SPECTRAGUARD PHASE 6: ML FOUNDATION VALIDATION (MILESTONE A)   ")
    print("================================================================")

    try:
        report = run_ml_benchmark()

        # MILESTONE A GATES
        metrics = report["metrics"]
        perf = report["performance"]

        passed = all(
            [
                metrics["accuracy"] > 0.70,
                metrics["f1_score"] > 0.70,
                perf["avg_inference_ms_per_sample"]
                < 10.0,  # Must be extremely fast for live processing
            ]
        )
        status = "PASS" if passed else "FAIL_PERFORMANCE"

    except Exception as e:
        print(f"\n[ERROR] ML validation failed: {str(e)}")
        report = {"error": str(e)}
        status = "FAIL_EXECUTION"

    final_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE 6",
        "subsystem": "spectraguard-cv-engine-ml",
        "validation_report": report,
        "overall_status": "READY" if status == "PASS" else status,
    }

    reports_dir = os.path.normpath("data/reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "ml_validation_report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nEngineering Report generated at: {report_path}")
    print("================================================================")

    if final_report["overall_status"] == "READY":
        print("PHASE 6 STATUS: PASSED. Machine Learning Foundation READY for Phase 7.")
        sys.exit(0)
    else:
        print(
            f"PHASE 6 STATUS: FAILED ({status}). Performance or execution bounds not met."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
