import os
import json
import joblib
from pathlib import Path

def audit_releases():
    print("==========================================================")
    print("         SPECTRA GUARD RELEASE INTEGRITY AUDIT            ")
    print("==========================================================")
    
    release_dir = Path("data/models/releases")
    if not release_dir.exists():
        print("[-] Release directory not found.")
        return

    releases = [d for d in release_dir.iterdir() if d.is_dir()]
    
    if not releases:
        print("[-] No release folders found.")
        return

    for release in releases:
        print(f"\n>> AUDITING RELEASE: {release.name}")
        print("-" * 50)
        
        # 1. Check Model
        model_path = release / "production_model.joblib"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                model_type = type(model).__name__
                
                feature_count = "Unknown"
                if hasattr(model, 'n_features_in_'):
                    feature_count = model.n_features_in_
                elif hasattr(model, 'feature_importances_'):
                    feature_count = len(model.feature_importances_)
                elif hasattr(model, 'n_features_'):
                    feature_count = model.n_features_
                
                print(f"[Model] Type: {model_type}")
                print(f"[Model] Expected Features (Internal): {feature_count}")
                
            except Exception as e:
                print(f"[Model] Error loading model: {e}")
        else:
            print("[Model] Missing")

        # 2. Check Scaler
        scaler_path = release / "feature_scaler.joblib"
        if scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
                scaler_features = "Unknown"
                if hasattr(scaler, 'n_features_in_'):
                    scaler_features = scaler.n_features_in_
                
                print(f"[Scaler] Expected Features (Internal): {scaler_features}")
            except Exception as e:
                print(f"[Scaler] Error loading scaler: {e}")
        else:
            print("[Scaler] Missing")

        # 3. Check Metadata
        meta_path = release / "feature_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                meta_features = meta.get("feature_count", "Missing")
                data_source = meta.get("data_source", "Unknown")
                print(f"[Metadata] Claimed Feature Count: {meta_features}")
                print(f"[Metadata] Claimed Data Source: {data_source}")
            except Exception as e:
                print(f"[Metadata] Error loading metadata: {e}")
        else:
            print("[Metadata] Missing")

        print("-" * 50)

if __name__ == "__main__":
    audit_releases()
