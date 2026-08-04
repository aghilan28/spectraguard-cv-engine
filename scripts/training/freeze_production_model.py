"""Production Model Freeze Script - 12-Feature Alignment (using CV Engine Classes)."""

import os
import sys
import json
import joblib
import pandas as pd
from datetime import datetime, timezone

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from spectraguard_cv_engine.ml.preprocessing.scaler import FeatureScaler
from spectraguard_cv_engine.ml.models.config import TrainingConfig
from spectraguard_cv_engine.ml.models.trainer import ModelTrainer

def freeze_synchronized_model():
    data_path = os.path.normpath("datasets/core/uhctd/raw/uhctd_features.csv")
    print(f"[FREEZE] Ingesting validated features from {data_path}...")
    df = pd.read_csv(data_path)

    # Dynamically extract exactly what features exist in the CSV (excluding labels/meta)
    feature_cols = [
        c for c in df.columns if c not in ["label", "video_id", "is_synthetic"]
    ]
    print(
        f"[FREEZE] Found {len(feature_cols)} active features for training: {feature_cols}"
    )

    X = df[feature_cols]
    y = df["label"]

    # 1. Fit scaler
    print("[FREEZE] Standardizing features using CV Engine FeatureScaler...")
    scaler = FeatureScaler(method="standard")
    X_scaled = scaler.fit_transform(X, feature_cols)

    # 2. Configure training
    config = TrainingConfig(
        model_type="random_forest",
        random_seed=42,
        hyperparameters={
            "n_estimators": 150,
            "min_samples_split": 10,
            "min_samples_leaf": 1,
            "max_depth": None,
            "class_weight": "balanced",
            "n_jobs": -1
        }
    )

    # 3. Train model
    print("[FREEZE] Training RandomForest classifier on standard scaler features...")
    trainer = ModelTrainer(config)
    trainer.train(pd.DataFrame(X_scaled, columns=feature_cols), y)

    # 4. Prepare release folder
    release_dir = os.path.normpath("data/models/releases/v0.7.5")
    os.makedirs(release_dir, exist_ok=True)

    # Write files with names expected by ModelLoader.load_version
    classifier_path = os.path.join(release_dir, "classifier.joblib")
    feature_scaler_path = os.path.join(release_dir, "feature_scaler.joblib")
    manifest_path = os.path.join(release_dir, "manifest.json")

    # Save scaler and trainer checkpoint
    scaler.save(feature_scaler_path)
    trainer.save_checkpoint("classifier.joblib")
    # Move classifier.joblib from default checkpoints folder to releases directory
    default_ckpt = os.path.normpath(os.path.join("data/models/checkpoints", "classifier.joblib"))
    if os.path.exists(default_ckpt):
        if os.path.exists(classifier_path):
            os.remove(classifier_path)
        os.rename(default_ckpt, classifier_path)

    # Also keep original filenames to preserve backward compatibility/expectations
    joblib.dump(trainer.model, os.path.join(release_dir, "production_model.joblib"))
    joblib.dump(scaler.scaler, os.path.join(release_dir, "scaler.joblib"))

    # Compile manifests
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "MODEL REBUILD - 12 FEATURE FREEZE SYNCED",
        "model_type": "RandomForestClassifier",
        "feature_count": len(feature_cols),
        "features_list": feature_cols,
        "hyperparameters": {
            "n_estimators": 150,
            "min_samples_split": 10,
            "min_samples_leaf": 1,
            "max_depth": None,
            "class_weight": "balanced",
        },
        "release_version": "v0.7.5-production-12f-synchronized",
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    with open(os.path.join(release_dir, "model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(
        "[SUCCESS] Production model and scaler synchronized to the 12-feature dataset and saved to v0.7.5."
    )

if __name__ == "__main__":
    freeze_synchronized_model()
