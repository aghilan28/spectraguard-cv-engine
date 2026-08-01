import os
import json
import joblib
import numpy as np

class SpectraGuardPredictor:
    def __init__(self, release_version="v1.0.0"):
        self.base_dir = f"data/models/releases/{release_version}"
        self.model_path = os.path.join(self.base_dir, "production_model.joblib")
        self.scaler_path = os.path.join(self.base_dir, "feature_scaler.joblib")
        self.meta_path = os.path.join(self.base_dir, "feature_metadata.json")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Missing production artifacts at release destination: {self.base_dir}")
            
        self.initialize_engine()

    def initialize_engine(self):
        # Load production assets
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        with open(self.meta_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Hard alignment to the real 10-channel feature specification
        self.expected_dim = self.metadata.get("feature_count", 10)
        print(f"[✓] Inference Engine loaded version v1.0.0 successfully. (Expects {self.expected_dim} features)")

    def predict_video_features(self, feature_array):
        """
        Executes structural model inference on real extracted FFT feature data matrices.
        """
        feats = np.array(feature_array, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
            
        if feats.shape[1] != self.expected_dim:
            raise ValueError(f"Feature dimension mismatch. Model requires {self.expected_dim} features, received {feats.shape[1]}")
            
        # Standardize using the production scaler instance
        feats_scaled = self.scaler.transform(feats)
        
        predictions = self.model.predict(feats_scaled)
        probabilities = self.model.predict_proba(feats_scaled)[:, 1]
        
        return {
            "predictions": [int(p) for p in predictions],
            "probabilities": [float(p) for p in probabilities]
        }

if __name__ == "__main__":
    print("[*] Performing operational verification for production inference module...")
    try:
        engine = SpectraGuardPredictor()
        
        # Verify validation pass with base vector matching the verified 10-channel shape
        sample_vector = np.zeros((1, 10), dtype=np.float32)
        output = engine.predict_video_features(sample_vector)
        print(f"[✓] Core forward pass verification success: {output}")
    except Exception as e:
        print(f"[-] Runtime operational validation failed: {e}")
