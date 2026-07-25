"""Rigorous Dataset Validation & Manifest Generation (R2/R6 Compliance)."""

import os
import json
import pandas as pd
from datetime import datetime, timezone


def validate_csv(file_path: str) -> dict:
    print(f"[VALIDATION] Inspecting {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)

    # 1. Null / NaN check
    null_counts = df.isnull().sum().to_dict()
    has_nulls = df.isnull().values.any()

    # 2. Duplicate detection
    duplicates = int(df.duplicated().sum())

    # 3. Label variance check
    if "label" not in df.columns:
        raise ValueError("Missing mandatory 'label' column.")
    unique_labels = df["label"].nunique()
    label_counts = df["label"].value_counts().to_dict()

    # 4. Schema dimension check
    row_count, col_count = df.shape

    status = "PASS" if not has_nulls and unique_labels > 1 else "FAIL"

    return {
        "file": file_path,
        "status": status,
        "rows": row_count,
        "columns": col_count,
        "null_values_present": bool(has_nulls),
        "null_breakdown": null_counts,
        "duplicate_rows": duplicates,
        "unique_classes": unique_labels,
        "class_distribution": {str(k): int(v) for k, v in label_counts.items()},
    }


def main():
    os.makedirs("data/manifests", exist_ok=True)

    targets = [
        os.path.normpath("datasets/core/uhctd/raw/uhctd_features.csv"),
        os.path.normpath(
            "datasets/raw_self_collected/forensic_10k/forensic_features.csv"
        ),
    ]

    reports = []
    for t in targets:
        report = validate_csv(t)
        reports.append(report)
        print(
            f" -> Status: {report['status']} | Rows: {report['rows']} | Classes: {report['unique_classes']}"
        )

    master_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE 7.5 - TASK 2",
        "datasets_validated": reports,
    }

    manifest_path = os.path.normpath("data/manifests/dataset_validation_report.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=4)

    print(f"[SUCCESS] Validation manifest generated at: {manifest_path}")


if __name__ == "__main__":
    main()
