import os
import sys
import hashlib
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Configure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

CV_ENGINE_SRC = r"C:\Users\AKILA\OneDrive\ドキュメント\SPECTRAGUARD\spectraguard-cv-engine\src"
if CV_ENGINE_SRC not in sys.path:
    sys.path.insert(0, CV_ENGINE_SRC)

from spectraguard_cv_engine.ml.data.loader import EXPECTED_UNIFIED_FEATURES
from spectraguard_cv_engine.features.unified.pipeline import UnifiedExtractionPipeline
from spectraguard_cv_engine.ai.runtime.loader import ModelLoader
from spectraguard_cv_engine.ai.runtime.config import RuntimeConfig
from spectraguard_cv_engine.ai.runtime.engine import InferenceRuntime
from spectraguard_cv_engine.ai.confidence.engine import ConfidenceEngine
from spectraguard_cv_engine.ai.decision.engine import DecisionEngine
from spectraguard_cv_engine.ai.explainability.engine import ExplainabilityEngine

def file_hash(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"Error: {e}"

def run_all_verifications():
    print("======================================================================")
    
    # 9.1: Model Artifact Discovery
    print("[P9.1] Checking Model Release Directories...")
    v06_dir = r"C:\Users\AKILA\OneDrive\ドキュメント\SPECTRAGUARD\spectraguard-cv-engine\data\models\releases\v0.6.0"
    v07_dir = r"C:\Users\AKILA\OneDrive\ドキュメント\SPECTRAGUARD\spectraguard-cv-engine\data\models\releases\v0.7.5"
    
    v06_exists = os.path.exists(v06_dir)
    v07_exists = os.path.exists(v07_dir)
    
    print(f"  v0.6.0 directory exists: {v06_exists}")
    print(f"  v0.7.5 directory exists: {v07_exists}")
    
    for v_dir in [v06_dir, v07_dir]:
        if os.path.exists(v_dir):
            print(f"\n  Listing files in: {os.path.abspath(v_dir)}")
            for f in os.listdir(v_dir):
                f_path = os.path.join(v_dir, f)
                print(f"    - File: {f} | Size: {os.path.getsize(f_path)} bytes | Hash: {file_hash(f_path)}")
                
    # 9.2: Training Pipeline & Dataset Provenance
    print("\n[P9.2] Verifying Datasets on Filesystem...")
    dataset_path = r"C:\Users\AKILA\OneDrive\ドキュメント\SPECTRAGUARD\spectraguard-cv-engine\datasets\core\uhctd\raw\uhctd_features.csv"
    dataset_exists = os.path.exists(dataset_path)
    print(f"  Target training dataset path: {os.path.abspath(dataset_path)}")
    print(f"  Target training dataset exists: {dataset_exists}")
    
    # 9.3: Model Loading Verification
    print("\n[P9.3] Verifying Model Deserialization...")
    try:
        artifacts = ModelLoader.load_version(v06_dir)
        print("  Successfully loaded Model Version v0.6.0")
        print("  Model Class:", artifacts.trainer.model.__class__.__name__)
        print("  Scaler Class:", artifacts.scaler.scaler.__class__.__name__)
        print("  Is Fitted:", artifacts.scaler.is_fitted)
        print("  Is Trained:", artifacts.trainer.is_trained)
    except Exception as e:
        print("  Failed to load Model Version v0.6.0:", e)
        return
        
    # 9.4 & 9.6: Inference and Feature Integrity Verification
    print("\n[P9.4 & P9.6] Executing Inference on Mock Sequence...")
    np.random.seed(0)
    mock_frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(15)]
    
    try:
        vector = UnifiedExtractionPipeline.extract_from_sequence(
            raw_frames=mock_frames,
            vector_id="verification_vector",
            timestamp_ns=1700000000000
        )
        arr = vector.to_array()
        print("  Successfully extracted 15-dimensional features:")
        for k in EXPECTED_UNIFIED_FEATURES:
            val = vector.spatial_features.get(k) or vector.frequency_features.get(k) or vector.temporal_features.get(k) or 0.0
            print(f"    {k}: {val}")
            
        X = pd.DataFrame([arr], columns=EXPECTED_UNIFIED_FEATURES)
        scaled_df = artifacts.scaler.transform(X)
        print("\n  Successfully transformed features using scaler:")
        for k in EXPECTED_UNIFIED_FEATURES:
            print(f"    {k} Scaled: {scaled_df.iloc[0][k]} (raw={X.iloc[0][k]})")
            
        runtime = InferenceRuntime(artifacts, RuntimeConfig())
        pred_outputs = runtime.predict(X)
        explainer = ExplainabilityEngine(artifacts.trainer)
        explanations = explainer.explain(artifacts.scaler.transform(X), top_k=3)
        confidence_engine = ConfidenceEngine()
        probs = [p.probability for p in pred_outputs]
        conf_outputs = confidence_engine.evaluate(probs)
        decision = DecisionEngine.evaluate(pred_outputs[0], conf_outputs[0])
        
        print("\n  Prediction outputs:")
        print(f"    Prediction Class: {pred_outputs[0].prediction}")
        print(f"    Raw Probability: {probs[0]}")
        print(f"    Calibrated Conf Score: {conf_outputs[0].calibrated_score}")
        print(f"    Decision Severity: {decision.severity.value}")
        print(f"    SHAP Attributions: {explanations[0].feature_attributions}")
    except Exception as e:
        print("  Inference verification crashed:", e)
        
    print("\n======================================================================")

if __name__ == "__main__":
    run_all_verifications()
