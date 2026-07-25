"""Hyperparameter Optimization Suite conforming to R5 / R6."""

import os
import json
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


def optimize_random_forest():
    data_path = os.path.normpath("datasets/core/uhctd/raw/uhctd_features.csv")
    print(f"[OPTIMIZATION] Loading features for tuning from {data_path}...")
    df = pd.read_csv(data_path)

    feature_cols = [c for c in df.columns if c not in ["label", "is_synthetic"]]
    X = df[feature_cols]
    y = df["label"]

    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("[OPTIMIZATION] Running RandomizedSearchCV for Random Forest Primary...")
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=5,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X, y)

    best_params = search.best_params_
    best_score = float(search.best_score_)

    print(f" -> Best Macro F1: {best_score:.4f}")
    print(f" -> Best Parameters: {best_params}")

    os.makedirs("data/manifests", exist_ok=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE 7.5 - TASK 7 (MODEL OPTIMIZATION)",
        "model": "RandomForest_Primary",
        "best_cv_macro_f1": best_score,
        "best_hyperparameters": best_params,
    }

    report_path = os.path.normpath("data/manifests/model_optimization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Model optimization report saved to: {report_path}")


if __name__ == "__main__":
    optimize_random_forest()
