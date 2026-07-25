"""Feature Extraction Validation Suite conforming to R4 / R6."""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

EXPECTED_FEATURES = [
    "d_hfer_median",
    "d_rpss_median",
    "spectral_flatness",
    "block_dct_mean",
    "block_dct_var",
    "block_dct_max",
    "optical_flow_disp",
    "immerkaer_sigma",
    "p_hash_match",
    "noise_autocorr_peak",
    "diurnal_mismatch",
    "tenengrad_focus",
    "lbp_texture_dist",
]


def validate_features(file_path: str) -> dict:
    df = pd.read_csv(file_path)

    missing_cols = [f for f in EXPECTED_FEATURES if f not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset {file_path} is missing expected R4 features: {missing_cols}"
        )

    # Check bounds and finiteness
    finite_checks = {}
    for f in EXPECTED_FEATURES:
        is_finite = bool(pd.Series(df[f]).isin([np.inf, -np.inf]).sum() == 0)
        finite_checks[f] = is_finite

    all_finite = all(finite_checks.values())

    return {
        "file": file_path,
        "schema_status": "PASS" if not missing_cols else "FAIL",
        "missing_columns": missing_cols,
        "all_features_finite": all_finite,
        "total_records": len(df),
    }


def main():
    os.makedirs("data/manifests", exist_ok=True)

    targets = [
        os.path.normpath("datasets/core/uhctd/raw/uhctd_features.csv"),
        os.path.normpath(
            "datasets/raw_self_collected/forensic_10k/forensic_features.csv"
        ),
        os.path.normpath("datasets/synthetic/synthetic_augmented_features.csv"),
    ]

    results = []
    for t in targets:
        print(f"[FEATURE-VAL] Validating R4 contract for {t}...")
        res = validate_features(t)
        results.append(res)
        print(
            f" -> Schema Status: {res['schema_status']} | All Finite: {res['all_features_finite']}"
        )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE 7.5 - TASK 5 (FEATURE VALIDATION)",
        "feature_validation_results": results,
    }

    report_path = os.path.normpath("data/manifests/feature_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Feature validation report saved to: {report_path}")


if __name__ == "__main__":
    main()
