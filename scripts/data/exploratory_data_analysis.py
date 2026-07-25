"""Exploratory Data Analysis Script for SpectraGuard Features (R2/R6)."""

import os
import json
import pandas as pd
from datetime import datetime, timezone


def analyze_dataset(file_path: str) -> dict:
    df = pd.read_csv(file_path)

    stats = {}
    feature_cols = [c for c in df.columns if c != "label"]

    for col in feature_cols:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }

    class_counts = df["label"].value_counts().to_dict()
    total_rows = len(df)
    class_imbalance_ratio = float(
        max(class_counts.values()) / min(class_counts.values())
    )

    return {
        "file": file_path,
        "total_samples": total_rows,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "max_to_min_class_imbalance_ratio": round(class_imbalance_ratio, 2),
        "feature_statistics": stats,
    }


def main():
    os.makedirs("data/manifests", exist_ok=True)

    targets = [
        os.path.normpath("datasets/core/uhctd/raw/uhctd_features.csv"),
        os.path.normpath(
            "datasets/raw_self_collected/forensic_10k/forensic_features.csv"
        ),
    ]

    analysis_results = []
    for t in targets:
        print(f"[EDA] Analyzing distribution for {t}...")
        res = analyze_dataset(t)
        analysis_results.append(res)
        print(f" -> Imbalance Ratio: {res['max_to_min_class_imbalance_ratio']}x")

    master_eda = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE 7.5 - TASK 3 (EDA)",
        "analysis": analysis_results,
    }

    report_path = os.path.normpath("data/manifests/eda_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(master_eda, f, indent=4)

    print(f"[SUCCESS] EDA Report generated at: {report_path}")


if __name__ == "__main__":
    main()
