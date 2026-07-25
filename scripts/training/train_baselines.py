"""Baseline Model Training Suite conforming to R5 / R6."""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_and_evaluate_baselines():
    data_path = os.path.normpath("datasets/core/uhctd/raw/uhctd_features.csv")
    print(f"[TRAINING] Loading features from {data_path}...")
    df = pd.read_csv(data_path)

    feature_cols = [c for c in df.columns if c not in ["label", "is_synthetic"]]
    X = df[feature_cols]
    y = df["label"]

    # Define models per R5 recommendations
    models = {
        "LogisticRegression_Baseline": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest_Primary": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "ExtraTrees_Validator": ExtraTreesClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost_Challenger": XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss", n_jobs=-1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        print(f"[TRAINING] Evaluating {name} via 5-fold Stratified CV...")
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        mean_f1 = float(np.mean(scores))
        std_f1 = float(np.std(scores))
        results[name] = {
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
            "cv_scores": [round(float(s), 4) for s in scores],
        }
        print(f" -> {name}: Macro F1 = {mean_f1:.4f} (±{std_f1:.4f})")

    os.makedirs("data/manifests", exist_ok=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE 7.5 - TASK 6 (BASELINE TRAINING)",
        "evaluation_metric": "Macro F1 (5-Fold Stratified CV)",
        "model_results": results,
    }

    report_path = os.path.normpath("data/manifests/baseline_training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Baseline training report saved to: {report_path}")


if __name__ == "__main__":
    train_and_evaluate_baselines()
