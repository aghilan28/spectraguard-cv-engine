import os
import sys
import json
import joblib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import PreprocessingPipeline, FeatureVector

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RELEASE_DIR = os.path.join(BASE_DIR, "data", "models", "releases", "v0.9.0-audit")

def run_smoke_test():
    print("=== M0.3F: SMOKE TEST VERIFYING v0.9.0-audit PRODUCTION ARTIFACTS ===")

    model_path = os.path.join(RELEASE_DIR, "production_model.joblib")
    scaler_path = os.path.join(RELEASE_DIR, "feature_scaler.joblib")
    thresh_path = os.path.join(RELEASE_DIR, "threshold.json")
    meta_path = os.path.join(RELEASE_DIR, "feature_metadata.json")

    assert os.path.exists(model_path), f"Missing: {model_path}"
    assert os.path.exists(scaler_path), f"Missing: {scaler_path}"
    assert os.path.exists(thresh_path), f"Missing: {thresh_path}"
    assert os.path.exists(meta_path), f"Missing: {meta_path}"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(thresh_path, "r") as f:
        thresh_info = json.load(f)
        optimal_tau = thresh_info["optimal_threshold"]

    with open(meta_path, "r") as f:
        meta_info = json.load(f)
        feature_names = meta_info["feature_names"]

    print(f"Loaded Production Model: {type(model).__name__}")
    print(f"Loaded Scaler: {type(scaler).__name__}")
    print(f"Loaded Optimal Decision Threshold (tau): {optimal_tau}")
    print(f"Feature Names ({len(feature_names)}): {feature_names}")

    # Initialize PreprocessingPipeline
    pipeline = PreprocessingPipeline()

    # Simulate live frame feed
    dummy_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    frames_window = [dummy_frame for _ in range(15)]

    # 1. Feature Extraction
    feat_vec = pipeline.extract(frames_window)
    X_raw = feat_vec.to_numpy().reshape(1, -1)
    
    # 2. Scaling
    X_scaled = scaler.transform(X_raw)

    # 3. Inference & Calibration
    prob = model.predict_proba(X_scaled)[0, 1]
    is_tampered = bool(prob >= optimal_tau)

    print("\n--- INFERENCE TEST RESULTS ---")
    print(f"Raw 8D Feature Vector: {np.round(X_raw[0], 4)}")
    print(f"Calibrated Prob(Tampered): {prob:.4f}")
    print(f"Decision Threshold (tau):   {optimal_tau:.4f}")
    print(f"Predicted Status:           {'TAMPERED ALERT' if is_tampered else 'OK VERIFIED'}")

    print("\nSMOKE TEST PASSED: 100% End-to-End Pipeline & Artifact Integration Verified!")

if __name__ == "__main__":
    run_smoke_test()
