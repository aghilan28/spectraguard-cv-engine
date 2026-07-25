"""Data integrity and schema validation for ML datasets."""

import pandas as pd
from typing import List


class DatasetValidator:
    """Validates unified feature matrices and target labels for ML ingestion."""

    @staticmethod
    def validate_schema(
        df: pd.DataFrame, expected_features: List[str], label_col: str
    ) -> None:
        """Verifies that all expected feature columns and the label column exist."""
        missing_features = [col for col in expected_features if col not in df.columns]
        if missing_features:
            raise ValueError(
                f"Dataset is missing required feature columns: {missing_features}"
            )

        if label_col not in df.columns:
            raise ValueError(
                f"Dataset is missing the target label column: '{label_col}'"
            )

    @staticmethod
    def check_missing_values(
        df: pd.DataFrame, expected_features: List[str], label_col: str
    ) -> None:
        """Detects NaNs or nulls in the critical feature matrix and label vectors."""
        subset = expected_features + [label_col]
        if df[subset].isnull().any().any():
            raise ValueError(
                "Dataset contains missing values (NaN/Null) in features or labels."
            )

    @staticmethod
    def validate_data_types(
        df: pd.DataFrame, expected_features: List[str], label_col: str
    ) -> None:
        """Ensures all features are numeric and labels are structurally sound."""
        for col in expected_features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Feature column '{col}' contains non-numeric data.")

        if not pd.api.types.is_numeric_dtype(
            df[label_col]
        ) and not pd.api.types.is_bool_dtype(df[label_col]):
            raise TypeError(f"Label column '{label_col}' must be numeric or boolean.")

    @staticmethod
    def detect_duplicates(df: pd.DataFrame, expected_features: List[str]) -> int:
        """Counts exact duplicate feature rows, which may indicate data leakage or sampling errors."""
        return int(df.duplicated(subset=expected_features).sum())

    @staticmethod
    def validate_labels(df: pd.DataFrame, label_col: str) -> None:
        """Ensures the label column contains at least two distinct classes for classification."""
        unique_classes = df[label_col].nunique()
        if unique_classes < 2:
            raise ValueError(
                f"Dataset target '{label_col}' contains fewer than 2 classes. ML requires variation."
            )
