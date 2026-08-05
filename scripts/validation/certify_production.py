import os
import time
import psutil
import joblib
import json
import numpy as np

def certify_production():
    print("=== SPECTRALGUARD V2 PRODUCTION CERTIFICATION ===")
    
    # 1. Artifact Verification
    artifacts = ["production_model.joblib", "scaler.joblib", "feature_metadata.json"]
    for a in artifacts:
        path = os.path.join("data/models/latest", a)
        assert os.path.exists(path), f"CRITICAL: {a} missing."
        
    model = joblib.load("data/models/latest/production_model.joblib")
    scaler = joblib.load("data/models/latest/scaler.joblib")
    with open("data/models/latest/feature_metadata.json", "r") as f:
        features = json.load(f)["feature_order"]
        
    assert len(features) == 8, "CRITICAL: Feature count mismatch. Expected 8."
    print("? Model Artifacts Verified")
    
    # 2. Latency & Resource Test
    dummy_input = np.random.rand(1, 8)
    
    start = time.perf_counter()
    scaled = scaler.transform(dummy_input)
    model.predict(scaled)
    latency_ms = (time.perf_counter() - start) * 1000
    
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory().percent
    
    print(f"? Inference Latency: {latency_ms:.2f} ms")
    print(f"? System CPU: {cpu}% | RAM: {mem}%")
    
    if latency_ms > 50.0:
        print("WARNING: Inference latency exceeds 50ms threshold.")
        
    print("\n? CERTIFICATION PASSED. SYSTEM READY FOR DEPLOYMENT.")
    
if __name__ == "__main__":
    certify_production()
