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
            raise FileNotFoundError(f"Production artifacts missing in release path: {self.base_dir}")
            
        self.load_engine()

    def load_engine(self):
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        with open(self.meta_path, 'r') as f:
            self.metadata = json.load(f)
        self.expected_dim = self.metadata.get("feature_count", 64)

    def predict(self, feature_vector):
        """
        Executes real-time inference on an input feature matrix/vector.
        Safely applies the pre-trained standard scaling transformation.
        """
        features = np.array(feature_vector)
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        if features.shape[1] != self.expected_dim:
            raise ValueError(f"Dimension mismatch. Expected {self.expected_dim} features, got {features.shape[1]}")
            
        scaled_features = self.scaler.transform(features)
        predictions = self.model.predict(scaled_features)
        probabilities = self.model.predict_proba(scaled_features)[:, 1]
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }

if __name__ == "__main__":
    print("[*] Initializing SpectraGuard Predictor Runtime Engine...")
    predictor = SpectraGuardPredictor()
    print("[✓] Runtime Engine loaded successfully.")
    
    # Simple operational check with synthetic runtime data matching training schema
    test_vector = np.random.randn(1, 64)
    result = predictor.predict(test_vector)
    print(f"[✓] Runtime Engine Operational Verification Result: {result}")
