"""Reproducible and stratified dataset splitting for model training pipelines."""

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """
    Handles partitioning of feature matrices into Train, Validation, and Test subsets
    while enforcing class stratification and reproducibility.
    """

    @staticmethod
    def split_train_val_test(
        df: pd.DataFrame,
        label_col: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame into three discrete sets: Training, Validation, and Testing.

        Args:
            df: The complete, validated dataset.
            label_col: The target classification column used for stratification.
            test_size: Proportion of the dataset allocated to the final test set.
            val_size: Proportion of the dataset allocated to the validation set.
            random_state: Seed for the random number generator to ensure reproducibility.
            stratify: If True, ensures class balance is maintained across all splits.

        Returns:
            Tuple containing (train_df, val_df, test_df).

        Raises:
            ValueError: If the defined split ratios are mathematically impossible.
            KeyError: If the label column does not exist in the DataFrame.
        """
        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not found in the DataFrame.")

        if test_size <= 0.0 or val_size <= 0.0 or (test_size + val_size) >= 1.0:
            raise ValueError(
                "Invalid split ratios. test_size and val_size must be > 0.0, "
                "and their sum must be strictly less than 1.0."
            )

        # Base stratification target
        stratify_array = df[label_col] if stratify else None

        # Split 1: Extract Test Set from the entire dataset
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=stratify_array
        )

        # Calculate relative validation size from the remaining (Train + Val) pool
        relative_val_size = val_size / (1.0 - test_size)

        # Intermediate stratification target
        stratify_array_val = train_val_df[label_col] if stratify else None

        # Split 2: Extract Validation Set from the (Train + Val) pool
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=stratify_array_val,
        )

        return train_df, val_df, test_df
