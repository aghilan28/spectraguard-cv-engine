"""Feature matrix scaling and transformation pipelines."""

import os
import joblib
import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureScaler:
    """
    Manages the fitting, transformation, and serialization of feature scalers.
    Ensures identical spatial/frequency/temporal transformations during training and live inference.
    """

    def __init__(self, method: str = "standard"):
        """
        Args:
            method: 'standard' (Z-score) or 'minmax' (0 to 1 scaling).
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method. Use 'standard' or 'minmax'.")

        self.method = method
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, features: List[str]) -> None:
        """Calculates scaling parameters (mean, std, min, max) from the dataset."""
        if df.empty or not features:
            raise ValueError("Cannot fit scaler on empty data or empty feature list.")

        # Verify all features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise KeyError(f"Missing features in DataFrame during fit: {missing}")

        self.scaler.fit(df[features])
        self.feature_names = features
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted scaling to the feature matrix."""
        if not self.is_fitted or self.feature_names is None:
            raise RuntimeError("Scaler must be fitted before calling transform().")

        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise KeyError(f"Missing features in DataFrame during transform: {missing}")

        # Create a copy to avoid SettingWithCopyWarning
        transformed_df = df.copy()

        # Apply transformation only to the designated feature columns
        scaled_values = self.scaler.transform(df[self.feature_names])
        transformed_df[self.feature_names] = scaled_values

        return transformed_df

    def fit_transform(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Convenience method to execute fit and transform sequentially."""
        self.fit(df, features)
        return self.transform(df)

    def save(self, filepath: str) -> None:
        """Serializes the fitted scaler state to disk for production inference."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted scaler.")

        payload = {
            "method": self.method,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FeatureScaler":
        """Instantiates a FeatureScaler from a serialized joblib payload."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")

        payload = joblib.load(filepath)

        instance = cls(method=payload["method"])
        instance.scaler = payload["scaler"]
        instance.feature_names = payload["feature_names"]
        instance.is_fitted = True

        return instance
