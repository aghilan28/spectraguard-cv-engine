import time
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timezone
from typing import List
from backend.config.logging import logger
from backend.inference.model_loader import model_loader
from backend.inference.result import InferenceResult
from backend.utils.serialization import convert_numpy_types

# Dynamically append src to ensure verified pipeline imports correctly
sys.path.insert(0, os.path.abspath('src'))
try:
    from preprocessing.pipeline import PreprocessingPipeline
except ImportError as e:
    logger.critical(f"FATAL: Existing PreprocessingPipeline could not be found. {e}")
    raise

class InferenceEngine:
    def __init__(self) -> None:
        self.pipeline = PreprocessingPipeline()
        # Pre-warm the model loader
        model_loader.load()

    def run(self, frames: List[np.ndarray], camera_id: str = "default") -> InferenceResult:
        start_t = time.time()
        
        if len(frames) != 15:
            raise ValueError(f"Inference Engine strictly requires exactly 15 frames. Received {len(frames)}.")

        model, scaler, threshold, feature_names = model_loader.load()

        # Step 1: Verified Feature Extraction
        try:
            feat_vec = self.pipeline.extract(frames)
            feat_dict = convert_numpy_types(feat_vec.to_dict())
        except Exception as e:
            logger.error(f"Physics feature extraction crash: {e}")
            raise RuntimeError(f"Pipeline failure: {e}")

        # Step 2: Strict Feature Ordering via Metadata
        try:
            ordered_features = [feat_dict[fname] for fname in feature_names]
            # Convert to DataFrame with original feature names to avoid StandardScaler warning
            df = pd.DataFrame([ordered_features], columns=feature_names)
        except KeyError as e:
            logger.error(f"Metadata feature mapping mismatch. Missing: {e}")
            raise ValueError(f"Feature extraction missing required column: {e}")

        # Step 3: Verified Scaler Execution
        try:
            scaled_array = scaler.transform(df)
        except Exception as e:
            logger.error(f"Scaler transformation failure: {e}")
            raise RuntimeError(f"Standardization failure: {e}")

        # Step 4: Live Probability Prediction
        try:
            prob = float(model.predict_proba(scaled_array)[:, 1][0])
            pred = 1 if prob >= threshold else 0
            conf = float(prob if pred == 1 else (1.0 - prob))
        except Exception as e:
            logger.error(f"Model prediction inference graph failure: {e}")
            raise RuntimeError(f"Inference failure: {e}")

        latency_ms = round((time.time() - start_t) * 1000, 2)

        result = InferenceResult(
            timestamp=datetime.now(timezone.utc),
            probability=round(prob, 6),
            prediction=pred,
            confidence=round(conf, 6),
            threshold=threshold,
            latency_ms=latency_ms,
            feature_vector=feat_dict,
            camera_id=camera_id
        )

        logger.debug(f"Inference Complete | Pred: {pred} | Prob: {prob:.4f} | Latency: {latency_ms}ms")
        return result

inference_engine = InferenceEngine()
