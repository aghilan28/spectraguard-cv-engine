"""Data schemas for Explainability outputs."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ExplanationOutput:
    """Standardized payload mapping feature names to their relative SHAP attribution scores."""

    base_value: float
    feature_attributions: Dict[str, float]
    top_contributors: Dict[str, float]
