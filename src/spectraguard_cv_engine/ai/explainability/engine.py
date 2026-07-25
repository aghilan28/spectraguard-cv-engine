"""SHAP integration for local feature attribution."""

import shap
import pandas as pd
import numpy as np
from typing import List

from ...ml.models.trainer import ModelTrainer
from .models import ExplanationOutput


class ExplainabilityEngine:
    """
    Computes local feature attributions using SHAP (TreeExplainer) to explain
    individual model predictions deterministically.
    """

    def __init__(self, trainer: ModelTrainer):
        """Initializes the explainer using the underlying model from the trainer."""
        if not trainer.is_trained:
            raise RuntimeError(
                "Cannot initialize ExplainabilityEngine with an untrained model."
            )

        self.model = trainer.model
        self.model_type = trainer.config.model_type

        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SHAP TreeExplainer for {self.model_type}: {e}"
            )

    def explain(
        self, scaled_feature_matrix: pd.DataFrame, top_k: int = 5
    ) -> List[ExplanationOutput]:
        """
        Calculates SHAP values for a batch of predictions and packages them.
        """
        if scaled_feature_matrix.empty:
            raise ValueError("Cannot explain an empty feature matrix.")

        features = scaled_feature_matrix.columns.tolist()

        # Calculate SHAP values
        shap_values_raw = self.explainer.shap_values(scaled_feature_matrix)
        expected_value = self.explainer.expected_value

        # Normalize shapes for Binary Classification (focusing on positive class [1])
        if isinstance(shap_values_raw, list):
            # Older SHAP or certain models return a list of arrays
            shap_values = shap_values_raw[1]
            base_val = (
                expected_value[1]
                if isinstance(expected_value, (list, np.ndarray))
                else expected_value
            )
        elif (
            isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3
        ):
            # Newer SHAP with RandomForest returns 3D array: (n_samples, n_features, n_classes)
            shap_values = shap_values_raw[:, :, 1]
            base_val = (
                expected_value[1]
                if isinstance(expected_value, (list, np.ndarray))
                else expected_value
            )
        else:
            # XGBoost typically returns (n_samples, n_features) directly for binary
            shap_values = shap_values_raw
            base_val = (
                expected_value[0]
                if isinstance(expected_value, (list, np.ndarray))
                else expected_value
            )

        results = []
        for i in range(len(scaled_feature_matrix)):
            row_shaps = shap_values[i]

            # Map features to their specific SHAP values for this instance
            attr_dict = {feat: float(val) for feat, val in zip(features, row_shaps)}

            # Isolate top K contributors based on absolute magnitude
            sorted_attrs = sorted(
                attr_dict.items(), key=lambda item: abs(item[1]), reverse=True
            )
            top_dict = {k: v for k, v in sorted_attrs[:top_k]}

            results.append(
                ExplanationOutput(
                    base_value=float(base_val),
                    feature_attributions=attr_dict,
                    top_contributors=top_dict,
                )
            )

        return results
