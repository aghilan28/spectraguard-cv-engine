import os
import json
import joblib
import threading
from typing import Tuple, Any, List
from backend.config.logging import logger

class ModelLoader:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelLoader":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.model = None
        self.scaler = None
        self.threshold: float = 0.5
        self.feature_names: List[str] = []
        self._load_lock = threading.Lock()
        self._initialized = True

    def load(self, artifacts_dir: str = "data/models/latest") -> Tuple[Any, Any, float, List[str]]:
        with self._load_lock:
            if self.is_loaded():
                return self.model, self.scaler, self.threshold, self.feature_names

            logger.info(f"Initiating production artifact load sequence from {artifacts_dir}...")
            
            model_path = os.path.join(artifacts_dir, "production_model.joblib")
            scaler_path = os.path.join(artifacts_dir, "feature_scaler.joblib")
            threshold_path = os.path.join(artifacts_dir, "threshold.json")
            metadata_path = os.path.join(artifacts_dir, "feature_metadata.json")

            if not all(os.path.exists(p) for p in [model_path, scaler_path, threshold_path, metadata_path]):
                logger.error("Missing one or more required production artifacts.")
                raise FileNotFoundError("Incomplete ML artifact deployment.")

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

            with open(threshold_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                self.threshold = float(t_data.get("optimal_threshold", 0.5))

            with open(metadata_path, "r", encoding="utf-8") as f:
                m_data = json.load(f)
                self.feature_names = m_data.get("feature_names", [])
                self.validate_metadata()

            logger.info(f"Artifacts successfully locked into memory. Operating Threshold: {self.threshold}")
            return self.model, self.scaler, self.threshold, self.feature_names

    def validate_metadata(self) -> None:
        expected_features = getattr(self.model, "n_features_in_", None)
        if expected_features is not None and len(self.feature_names) != expected_features:
            logger.error(f"Structural Mismatch: Model requires {expected_features} features, but metadata supplies {len(self.feature_names)}.")
            raise ValueError("Feature dimension mismatch between model and metadata.")
        if not self.feature_names:
            raise ValueError("Feature ordering missing from metadata.")

    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

model_loader = ModelLoader()
