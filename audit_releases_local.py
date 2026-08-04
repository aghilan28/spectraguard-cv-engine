import os
import json
import joblib
from pathlib import Path

def audit_releases():
    release_dir = Path("data/models/releases")
    releases = [d for d in release_dir.iterdir() if d.is_dir()]
    
    for release in releases:
        if release.name == "v1.0.0":
            print(f"\n>> AUDITING RELEASE: {release.name} (LAPTOP)")
            print("-" * 50)
            
            model_path = release / "production_model.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
                print(f"[Model] Type: {type(model).__name__}")
                feature_count = model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown"
                print(f"[Model] Expected Features (Internal): {feature_count}")
            
            scaler_path = release / "feature_scaler.joblib"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                s_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else "Unknown"
                print(f"[Scaler] Expected Features (Internal): {s_features}")
                
            meta_path = release / "feature_metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                print(f"[Metadata] Claimed Feature Count: {meta.get('feature_count', 'Missing')}")
            print("-" * 50)

if __name__ == "__main__":
    audit_releases()
