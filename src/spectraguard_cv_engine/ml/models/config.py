"""Immutable configurations for Machine Learning model training."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class TrainingConfig:
    """Strict configuration boundaries for reproducible model training."""

    model_type: str  # 'random_forest' or 'xgboost'
    random_seed: int = 42
    checkpoint_dir: str = "data/models/checkpoints"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.model_type not in ["random_forest", "xgboost"]:
            raise ValueError(
                "Unsupported model_type. Must be 'random_forest' or 'xgboost'."
            )
