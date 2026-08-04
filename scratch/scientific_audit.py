import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import xgboost as xgb

# Ensure CV Engine src is in path
CV_ENGINE_SRC = r"C:\Users\AKILA\OneDrive\ドキュメント\SPECTRAGUARD\spectraguard-cv-engine\src"
if CV_ENGINE_SRC not in sys.path:
    sys.path.insert(0, CV_ENGINE_SRC)

from inference.predictor import SpectraGuardPredictor
from spectraguard_cv_engine.ai.confidence.engine import ConfidenceEngine
from spectraguard_cv_engine.ai.decision.engine import DecisionEngine

def compute_hash(obj):
    # Deterministic hashing of values/lists/dicts
    if isinstance(obj, np.ndarray):
        s = ",".join(f"{v:.8f}" for v in obj.flatten())
    elif isinstance(obj, list):
        s = ",".join(f"{v}" for v in obj)
    elif isinstance(obj, dict):
        s = json.dumps(obj, sort_keys=True)
    else:
        s = str(obj)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def audit():
    videos = [
        "TEST VIDEO.mp4",
        "TEST VIDEO EXTREME BLUR.mp4",
        "TEST VIDEO COVERED HAND.mp4",
        "TEST VIDEO - 1.mp4",
        "TEST VIDEO - 2.mp4"
    ]
    uploads_dir = r"C:\Users\AKILA\OneDrive\ドキュメント\SPECTRAGUARD\spectraguard-core-infra\data\uploads"
    
    predictor = SpectraGuardPredictor(release_version="v1.0.0")
    
    results = {}
    
    for video_name in videos:
        video_path = os.path.join(uploads_dir, video_name)
        if not os.path.exists(video_path):
            print(f"Error: {video_path} does not exist!")
            continue
            
        print(f"\n================================================")
        print(f"VIDEO: {video_name}")
        print(f"================================================")
        
        # 1. Video metadata and raw FFT
        X, total_frames, resolution, frames_sampled = predictor.extract_features_with_metadata(video_path)
        video_sha = hashlib.sha256(open(video_path, 'rb').read()).hexdigest()
        
        raw_list = [float(X.iloc[0][f"fft_{i}"]) for i in range(10)]
        raw_hash = compute_hash(raw_list)
        
        # 2. Scaling
        scaled_df = predictor.artifacts.scaler.transform(X)
        scaled_list = [float(scaled_df.iloc[0][f"fft_{i}"]) for i in range(10)]
        scaled_hash = compute_hash(scaled_list)
        
        # 3. Model predict and predict_proba (Step 3: Direct model output)
        direct_pred = predictor.model.predict(scaled_df)[0]
        direct_proba = predictor.model.predict_proba(scaled_df)[0]
        
        # Margin and leaf indices
        booster = predictor.model.get_booster()
        dmat = xgb.DMatrix(scaled_df)
        margin = booster.predict(dmat, output_margin=True)[0]
        leaf_indices = booster.predict(dmat, pred_leaf=True)[0].tolist()
        leaf_hash = compute_hash(leaf_indices)
        
        # Tree path hash (let's check how tree path hash is calculated, or we hash leaf_indices and margin)
        tree_path_hash = compute_hash(f"{leaf_indices}-{margin}")
        
        # 4. SHAP
        explanations = predictor.explainer.explain(scaled_df, top_k=3)
        shap_out = explanations[0]
        shap_hash = compute_hash(shap_out.feature_attributions)
        
        # 5. Confidence Engine
        conf_out = predictor.confidence_engine.evaluate([direct_proba[1]])[0]
        
        # 6. Decision Engine
        decision_pred_out = predictor.runtime.predict(X)[0] # prediction output wrapper
        decision = DecisionEngine.evaluate(decision_pred_out, conf_out)
        
        # 7. Predictor's full response
        full_res = predictor.predict_video(video_path, prediction_id=f"pred_{video_name.replace(' ', '_')[:10]}")
        response_hash = compute_hash(full_res)
        
        print(f"Prediction ID: {full_res['prediction_id']}")
        print(f"Video SHA256: {video_sha}")
        print(f"Resolution: {resolution}")
        print(f"FPS: 30 (Assume)")
        print(f"Frame Count: {total_frames}")
        print(f"Frames Sampled: {frames_sampled}")
        print(f"Sampling Step: {total_frames // 30 if total_frames > 0 else 1}")
        print(f"Model Release: {predictor.release_version}")
        print(f"Model Type: {predictor.model_type}")
        print(f"Expected Features: {predictor.expected_features}")
        print(f"------------------------------------------------")
        print(f"Raw FFT Feature Vector:")
        for i, val in enumerate(raw_list):
            print(f"  fft_{i}: {val:.6f}")
        print(f"Raw FFT Hash: {raw_hash}")
        print(f"------------------------------------------------")
        print(f"Scaled Feature Vector:")
        for i, val in enumerate(scaled_list):
            print(f"  fft_{i}: {val:.6f}")
        print(f"Scaled Hash: {scaled_hash}")
        print(f"------------------------------------------------")
        print(f"Model Prediction:")
        print(f"  predict(): {direct_pred}")
        print(f"  predict_proba(): {direct_proba.tolist()}")
        print(f"  Raw Margin: {margin:.6f}")
        print(f"  Leaf Indices: {leaf_indices}")
        print(f"  Leaf Hash: {leaf_hash}")
        print(f"  Tree Path Hash: {tree_path_hash}")
        print(f"------------------------------------------------")
        print(f"SHAP:")
        print(f"  Expected Value: {shap_out.base_value:.6f}")
        print(f"  SHAP Values: {shap_out.feature_attributions}")
        print(f"  Top Contributors: {shap_out.top_contributors}")
        print(f"  SHAP Hash: {shap_hash}")
        print(f"------------------------------------------------")
        print(f"Confidence:")
        print(f"  Raw Probability: {conf_out.raw_probability:.6f}")
        print(f"  Calibrated Probability: {conf_out.calibrated_score:.6f}")
        print(f"  Calibration Function: Passthrough (calibrated = prob)")
        print(f"  Confidence Tier: {conf_out.tier.value}")
        print(f"------------------------------------------------")
        print(f"Decision Engine:")
        print(f"  Severity: {decision.severity.value}")
        print(f"  Action Required: {decision.action_required}")
        print(f"  Recommendation: {decision.severity.value}") # or similar mapping
        print(f"  Reason: {decision.rationale}")
        print(f"------------------------------------------------")
        print(f"Final API Response:")
        print(json.dumps(full_res, indent=2))
        print(f"Response Hash: {response_hash}")
        
        results[video_name] = {
            "raw_hash": raw_hash,
            "scaled_hash": scaled_hash,
            "leaf_hash": leaf_hash,
            "tree_path_hash": tree_path_hash,
            "shap_hash": shap_hash,
            "response_hash": response_hash,
            "prediction": direct_pred,
            "probabilities": direct_proba.tolist(),
            "confidence": conf_out.calibrated_score,
            "severity": decision.severity.value,
            "shap_values": shap_out.feature_attributions
        }

    # STEP 2 Verification
    print("\n================================================")
    print("STEP 2 HASH COMPARISON")
    print("================================================")
    stage_keys = ["raw_hash", "scaled_hash", "leaf_hash", "tree_path_hash", "shap_hash", "response_hash"]
    
    bugs_found = False
    for key in stage_keys:
        values = [results[v][key] for v in results]
        unique_values = set(values)
        print(f"{key}: {len(unique_values)} unique hashes out of {len(values)}")
        for v in results:
            print(f"  {v}: {results[v][key]}")
        if len(unique_values) < len(values):
            print(f"!!! WARNING: Identical {key} found for different videos !!!")
            bugs_found = True
            
    if bugs_found:
        print("\nBUG FOUND")

if __name__ == "__main__":
    audit()
