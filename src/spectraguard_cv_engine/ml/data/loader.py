"""Dataset ingestion, manifest parsing, and diagnostics extraction."""

import json
import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path

from .validator import DatasetValidator
from ...features.unified.models import SPATIAL_KEYS, FREQUENCY_KEYS, TEMPORAL_KEYS

# The authoritative list of unified features established in Phase 4
EXPECTED_UNIFIED_FEATURES = SPATIAL_KEYS + FREQUENCY_KEYS + TEMPORAL_KEYS


class DatasetLoader:
    """Handles the ingestion of static datasets and extraction of diagnostic metadata."""

    @staticmethod
    def load_dataset(
        csv_path: str, label_col: str = "label", manifest_path: str = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads a CSV dataset, enforces strict validation bounds, and calculates integrity stats.

        Args:
            csv_path: Path to the tabular dataset.
            label_col: The target classification column.
            manifest_path: Optional JSON metadata file describing the dataset origin.

        Returns:
            Tuple of (Validated DataFrame, Diagnostics Dictionary)
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        # 1. Load Data
        df = pd.read_csv(path)

        # 2. Strict Validation Pipeline
        DatasetValidator.validate_schema(df, EXPECTED_UNIFIED_FEATURES, label_col)
        DatasetValidator.check_missing_values(df, EXPECTED_UNIFIED_FEATURES, label_col)
        DatasetValidator.validate_data_types(df, EXPECTED_UNIFIED_FEATURES, label_col)
        DatasetValidator.validate_labels(df, label_col)

        # 3. Diagnostics & Statistics Calculation
        duplicates_count = DatasetValidator.detect_duplicates(
            df, EXPECTED_UNIFIED_FEATURES
        )
        class_distribution = df[label_col].value_counts().to_dict()

        stats = {
            "total_samples": len(df),
            "feature_count": len(EXPECTED_UNIFIED_FEATURES),
            "duplicates_detected": duplicates_count,
            "class_distribution": {
                str(k): int(v) for k, v in class_distribution.items()
            },
            "manifest_data": None,
        }

        # 4. Optional Manifest Loading
        if manifest_path:
            m_path = Path(manifest_path)
            if m_path.exists():
                with open(m_path, "r", encoding="utf-8") as f:
                    stats["manifest_data"] = json.load(f)

        return df, stats
