"""Episode-Aware Baseline Training Infrastructure for SpectraGuard."""

import os
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def run_group_baseline_training():
    data_path = os.path.normpath("datasets/core/uhctd/raw/uhctd_features.csv")
    print(f"[TRAIN] Loading genuine pipeline features from {data_path}...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Target dataset file missing at: {data_path}")

    df = pd.read_csv(data_path)

    # Isolate targets, explicit groups, and numeric features
    y = df["label"]
    groups = df["video_id"]

    # Strip non-numeric metadata columns from the feature matrix
    feature_cols = [
        c for c in df.columns if c not in ["label", "video_id", "is_synthetic"]
    ]
    X = df[feature_cols]

    print(
        f"[TRAIN] Extracted {len(feature_cols)} active features across {len(df)} samples."
    )
    print(f"[TRAIN] Discovered {groups.nunique()} unique video episode tracks.")

    # Execute strict Group-Based Train/Test Partitioning (80/20 Group Split)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(
        f" -> Train Set: {len(X_train)} samples across groups: {groups.iloc[train_idx].unique()}"
    )
    print(
        f" -> Test Set:  {len(X_test)} samples across groups: {groups.iloc[test_idx].unique()}"
    )

    # Scale feature dimensions uniformly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define production benchmark models
    models = {
        "LogisticRegression_Baseline": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "ExtraTrees_Baseline": ExtraTreesClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "RandomForest_Primary": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42, class_weight="balanced"
        ),
    }

    report_summary = {}

    for name, clf in models.items():
        print(f"\n[MODEL] Fitting {name}...")
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)

        report = classification_report(y_test, preds, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]
        print(f" -> Evaluation Macro F1: {macro_f1:.4f}")
        report_summary[name] = {"macro_f1": macro_f1}

    # Export metrics manifest file
    out_dir = os.path.normpath("data/manifests")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "baseline_training_report.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_summary, f, indent=4)

    print(f"\n[SUCCESS] Baseline training completed. Manifest updated at: {out_file}")


if __name__ == "__main__":
    run_group_baseline_training()
