"""Deterministic inference execution engine."""

import time
import pandas as pd
from typing import List
from datetime import datetime, timezone

from .config import RuntimeConfig
from .models import PredictionOutput
from .loader import RuntimeArtifacts


class InferenceRuntime:
    """
    Executes pre-processing and model inference on incoming feature vectors.
    Enforces schemas, batch limits, and extracts diagnostic latency.
    """

    def __init__(self, artifacts: RuntimeArtifacts, config: RuntimeConfig):
        self.artifacts = artifacts
        self.config = config
        self.last_raw_feature_matrix: pd.DataFrame | None = None
        self.last_scaled_feature_matrix: pd.DataFrame | None = None

    def predict(self, feature_matrix: pd.DataFrame) -> List[PredictionOutput]:
        """
        Executes the end-to-end inference pipeline on a feature batch.

        Args:
            feature_matrix: Pandas DataFrame containing raw, unscaled features.

        Returns:
            List of standardized PredictionOutput instances corresponding to the rows.
        """
        if feature_matrix.empty:
            raise ValueError("Cannot run inference on an empty feature matrix.")

        if len(feature_matrix) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size {len(feature_matrix)} exceeds maximum allowed "
                f"limit of {self.config.max_batch_size}."
            )

        # 1. Input Validation
        if self.config.enforce_schema_validation:
            expected_features = list(self.artifacts.scaler.feature_names or [])
            missing = [f for f in expected_features if f not in feature_matrix.columns]
            if missing:
                raise KeyError(f"Feature matrix is missing required columns: {missing}")

            ordered_matrix = feature_matrix.reindex(columns=expected_features, copy=False)
        else:
            ordered_matrix = feature_matrix.copy()

        self.last_raw_feature_matrix = ordered_matrix.copy()
        start_time = time.perf_counter()

        # 2. Pre-processing (Scaling)
        scaled_df = self.artifacts.scaler.transform(ordered_matrix)
        self.last_scaled_feature_matrix = scaled_df.copy()

        # 3. Model Inference
        predictions = self.artifacts.trainer.predict(scaled_df)

        probabilities = None
        if self.config.require_probabilities:
            probs_matrix = self.artifacts.trainer.predict_proba(scaled_df)
            # Assuming binary classification mapping, index 1 represents positive/tampered class
            probabilities = [
                float(p[1]) if len(p) > 1 else float(p[0]) for p in probs_matrix
            ]

        end_time = time.perf_counter()

        # 4. Diagnostics & Packaging
        # Calculate per-sample average latency in milliseconds
        latency_ms_per_sample = ((end_time - start_time) * 1000) / len(feature_matrix)

        results = []
        for idx in range(len(predictions)):
            results.append(
                PredictionOutput(
                    prediction=int(predictions[idx]),
                    probability=probabilities[idx] if probabilities else None,
                    latency_ms=float(f"{latency_ms_per_sample:.4f}"),
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )

        return results
