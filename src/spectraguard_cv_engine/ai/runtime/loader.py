"""Secure deserialization of versioned Machine Learning artifacts."""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any

from ...ml.preprocessing.scaler import FeatureScaler
from ...ml.models.trainer import ModelTrainer


@dataclass(frozen=True)
class RuntimeArtifacts:
    """Immutable container for loaded production artifacts."""

    trainer: ModelTrainer
    scaler: FeatureScaler
    manifest: Dict[str, Any]


class ModelLoader:
    """Handles the safe loading of versioned ML releases."""

    @staticmethod
    def load_version(version_dir: str) -> RuntimeArtifacts:
        """
        Validates and loads the canonical model/scaler metadata for the active release.
        """
        if not os.path.exists(version_dir):
            raise FileNotFoundError(f"Version directory not found: {version_dir}")

        manifest_path = os.path.join(version_dir, "training_manifest.json")
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join(version_dir, "manifest.json")

        scaler_path = os.path.join(version_dir, "feature_scaler.joblib")
        model_candidates = [
            os.path.join(version_dir, "production_model.joblib"),
            os.path.join(version_dir, "classifier.joblib"),
        ]
        model_path = next((path for path in model_candidates if os.path.exists(path)), model_candidates[0])

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest missing at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                "Missing joblib artifacts declared in version directory."
            )

        scaler = FeatureScaler.load(scaler_path)
        trainer = ModelTrainer.load_checkpoint(model_path)

        if not trainer.is_trained or not scaler.is_fitted:
            raise RuntimeError(
                "Loaded artifacts are incomplete or corrupt (untrained state)."
            )

        return RuntimeArtifacts(trainer=trainer, scaler=scaler, manifest=manifest)
