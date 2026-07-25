"""Synthetic Data Generation Suite conforming to R3 Strategy."""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone


def generate_r3_synthetic_augmentation(base_csv_path: str, output_csv_path: str):
    print(f"[SYNTHESIS] Loading base features from {base_csv_path}...")
    if not os.path.exists(base_csv_path):
        raise FileNotFoundError(f"Base dataset missing at {base_csv_path}")

    df = pd.read_csv(base_csv_path)

    # Apply R3 synthetic augmentations (e.g. simulating codec compression noise & weather drift)
    print("[SYNTHESIS] Applying R3 codec degradation & procedural noise jitter...")
    np.random.seed(100)

    augmented_df = df.copy()

    # Simulate codec quantization noise on D-HFER and block-DCT features
    codec_noise = np.random.normal(0, 0.05, size=len(augmented_df))
    augmented_df["d_hfer_median"] = np.clip(
        augmented_df["d_hfer_median"] + codec_noise, 0.0, 1.0
    )

    # Simulate lighting drift / diurnal offset
    augmented_df["diurnal_mismatch"] += np.random.normal(
        0.2, 0.1, size=len(augmented_df)
    )

    # Tag synthetic rows explicitly per R3 / R6 data governance
    augmented_df["is_synthetic"] = 1

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    augmented_df.to_csv(output_csv_path, index=False)
    print(
        f"[SUCCESS] Synthetic augmented dataset saved to: {output_csv_path} ({len(augmented_df)} records)"
    )

    return len(augmented_df)


def main():
    base_path = os.path.normpath("datasets/core/uhctd/raw/uhctd_features.csv")
    out_path = os.path.normpath("datasets/synthetic/synthetic_augmented_features.csv")

    count = generate_r3_synthetic_augmentation(base_path, out_path)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE 7.5 - TASK 4 (SYNTHETIC)",
        "synthetic_dataset": out_path,
        "total_synthetic_records": count,
        "governance": "Excluded from E1 real-world generalization claims per R3/R6",
    }

    manifest_path = os.path.normpath("data/manifests/synthetic_generation_report.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print(f"[SUCCESS] Synthetic generation manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
