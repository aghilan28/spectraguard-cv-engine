"""Production model versioning and export packaging."""

import os
import json
import joblib
from datetime import datetime, timezone
from typing import Dict, Any

from ..models.trainer import ModelTrainer
from ..preprocessing.scaler import FeatureScaler


class ModelExporter:
    """
    Packages trained models, preprocessors, and metadata into a
    versioned, production-ready release artifact.
    """

    @staticmethod
    def export_pipeline(
        trainer: ModelTrainer,
        scaler: FeatureScaler,
        evaluation_report: Dict[str, Any],
        export_dir: str,
        version: str,
    ) -> str:
        """
        Serializes all components required for live inference into a single versioned directory.
        """
        if not trainer.is_trained:
            raise RuntimeError("Cannot export an untrained model.")
        if not scaler.is_fitted:
            raise RuntimeError("Cannot export an unfitted scaler.")

        version_dir = os.path.normpath(os.path.join(export_dir, version))
        os.makedirs(version_dir, exist_ok=True)

        # 1. Save Preprocessor
        scaler_path = os.path.join(version_dir, "feature_scaler.joblib")
        scaler.save(scaler_path)

        # 2. Save Model (Directly into the version directory)
        model_path = os.path.join(version_dir, "classifier.joblib")
        payload = {"config": trainer.config, "model": trainer.model}
        joblib.dump(payload, model_path)

        # 3. Generate Manifest
        manifest = {
            "version": version,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": trainer.config.model_type,
            "scaling_method": scaler.method,
            "features": scaler.feature_names,
            "evaluation_metrics": evaluation_report["metrics"],
            "artifacts": {
                "scaler": "feature_scaler.joblib",
                "model": "classifier.joblib",
            },
        }

        manifest_path = os.path.join(version_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

        return version_dir
