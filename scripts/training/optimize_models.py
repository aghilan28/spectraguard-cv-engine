"""Episode-Aware Hyperparameter Optimization for SpectraGuard."""

import os
import json
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def run_group_optimization():
    data_path = os.path.normpath("datasets/core/uhctd/raw/uhctd_features.csv")
    print(f"[OPTIMIZATION] Loading features for tuning from {data_path}...")

    df = pd.read_csv(data_path)

    y = df["label"]
    groups = df["video_id"]
    feature_cols = [
        c for c in df.columns if c not in ["label", "video_id", "is_synthetic"]
    ]
    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define primary model and hyperparameter space
    clf = RandomForestClassifier(class_weight="balanced", random_state=42)
    param_dist = {
        "n_estimators": [100, 150, 200, 250],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Enforce strict episode isolation during cross-validation
    gkf = GroupKFold(n_splits=3)

    print(
        "[OPTIMIZATION] Running RandomizedSearchCV for Random Forest Primary (GroupKFold)..."
    )
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_macro",
        cv=gkf,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    # CRITICAL: groups parameter must be passed explicitly to fit
    search.fit(X_scaled, y, groups=groups)

    best_f1 = search.best_score_
    best_params = search.best_params_

    print(f" -> Best Cross-Validated Macro F1: {best_f1:.4f}")
    print(f" -> Best Parameters: {best_params}")

    # Export optimization report
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "MODEL REBUILD - OPTIMIZATION",
        "model": "RandomForest_Primary",
        "best_cv_macro_f1": best_f1,
        "best_hyperparameters": best_params,
        "validation_strategy": "GroupKFold(n_splits=3)",
    }

    out_dir = os.path.normpath("data/manifests")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "model_optimization_report.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Model optimization report saved to: {out_file}")


if __name__ == "__main__":
    run_group_optimization()
