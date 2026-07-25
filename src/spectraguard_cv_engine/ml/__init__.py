"""Machine Learning Foundation for SpectraGuard."""

from .data.loader import DatasetLoader, EXPECTED_UNIFIED_FEATURES
from .data.validator import DatasetValidator
from .data.splitter import DatasetSplitter
from .preprocessing.scaler import FeatureScaler
from .models.config import TrainingConfig
from .models.trainer import ModelTrainer
from .optimization.tuner import HyperparameterTuner
from .evaluation.evaluator import ModelEvaluator

__all__ = [
    "DatasetLoader",
    "DatasetValidator",
    "DatasetSplitter",
    "EXPECTED_UNIFIED_FEATURES",
    "FeatureScaler",
    "TrainingConfig",
    "ModelTrainer",
    "HyperparameterTuner",
    "ModelEvaluator",
]
