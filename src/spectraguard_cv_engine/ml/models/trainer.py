"""Core training orchestrator for SpectraGuard predictive models."""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Any
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import TrainingConfig


class ModelTrainer:
    """Instantiates, trains, and serializes machine learning models."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: Optional[Any] = None
        self.is_trained = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Configures the underlying algorithm based on the immutable config."""
        params = self.config.hyperparameters.copy()
        params["random_state"] = self.config.random_seed

        if self.config.model_type == "random_forest":
            # Default fallback for RF if not specified
            if "n_estimators" not in params:
                params["n_estimators"] = 100
            self.model = RandomForestClassifier(**params)

        elif self.config.model_type == "xgboost":
            # Ensure safe cross-platform threading for XGBoost
            if "n_jobs" not in params:
                params["n_jobs"] = -1
            # Suppress default XGBoost warnings
            if "eval_metric" not in params:
                params["eval_metric"] = "logloss"
            self.model = XGBClassifier(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Executes the fitting process on the provided feature matrix and labels."""
        if X_train.empty or y_train.empty:
            raise ValueError("Training data matrices cannot be empty.")

        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Executes basic inference using the trained model weights."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict().")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Executes probabilistic inference using the trained model weights."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict_proba().")
        return self.model.predict_proba(X)

    def save_checkpoint(self, filename: str) -> str:
        """Serializes the trained model to disk for persistence."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filepath = os.path.normpath(os.path.join(self.config.checkpoint_dir, filename))

        payload = {"config": self.config, "model": self.model}
        joblib.dump(payload, filepath)
        return filepath

    @classmethod
    def load_checkpoint(cls, filepath: str) -> "ModelTrainer":
        """Reconstructs a ModelTrainer instance from a serialized checkpoint."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        payload = joblib.load(filepath)

        instance = cls(config=payload["config"])
        instance.model = payload["model"]
        instance.is_trained = True
        return instance
