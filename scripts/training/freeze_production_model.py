"""Production Model Freeze Script - 12-Feature Alignment."""

import os
import json
import joblib
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(
        "[FREEZE] Training final production Random Forest on the clean 12-feature matrix..."
    )
    clf = RandomForestClassifier(
        n_estimators=150,
        min_samples_split=10,
        min_samples_leaf=1,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_scaled, y)

    release_dir = os.path.normpath("data/models/releases/v0.7.5")
    os.makedirs(release_dir, exist_ok=True)

    joblib.dump(clf, os.path.join(release_dir, "production_model.joblib"))
    joblib.dump(scaler, os.path.join(release_dir, "scaler.joblib"))

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

    with open(
        os.path.join(release_dir, "model_manifest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=4)

    print(
        "[SUCCESS] Production model and scaler synchronized to the 12-feature dataset."
    )


if __name__ == "__main__":
    freeze_synchronized_model()
