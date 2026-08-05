import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META_DIR = os.path.join(BASE_DIR, "data", "datasets", "virat", "metadata")
GROUND_TRUTH_CSV = os.path.join(META_DIR, "ground_truth.csv")

# ROOT-CAUSE FIX (forensic audit, Aug 2026):
# This script previously wrote to the SAME path as the real extractor
# (scripts/data/extract_production_features_8d.py):
#     data/datasets/virat/metadata/production_features_8d.csv
# Both scripts used the SAME filename, so whichever one ran LAST silently
# became "the" production dataset with no marker distinguishing real,
# pipeline-extracted features from fabricated np.random.normal() values.
#
# run_production_training_v2.py trained the shipped StandardScaler +
# CalibratedClassifierCV(RandomForest) on whatever was in that file. If this
# synthetic generator ran (e.g. as a stand-in while the VIRAT videos were
# still downloading), the scaler's mean_/scale_ were fit to fabricated
# numbers (e.g. fft_low_ratio ~ N(0.85, .03), edge_density ~ N(0.15, .02))
# that do NOT match the real PreprocessingPipeline's actual output range on
# camera frames (empirically: fft_low_ratio ~0.3-0.4, edge_density ~0.01,
# log_total_energy ~20 vs the assumed ~12, etc. -- see FORENSIC_AUDIT.md).
# Every real feature vector then lands far outside the fitted distribution,
# so the classifier scores nearly everything as "tampered."
#
# This file is now renamed on output and requires an explicit opt-in flag
# so it can never again be mistaken for, or silently overwrite, the real
# production dataset.
OUTPUT_CSV = os.path.join(META_DIR, "production_features_8d.SYNTHETIC_DO_NOT_USE_FOR_TRAINING.csv")


def main():
    if "--i-understand-this-is-fake-data" not in sys.argv:
        print(
            "[REFUSING TO RUN] This generates FABRICATED feature values "
            "(np.random.normal placeholders), NOT real features extracted "
            "from video via PreprocessingPipeline. It must never be used to "
            "train or validate the production model -- doing so previously "
            "caused the live camera to flag almost everything as tampered.\n"
            "Use scripts/data/extract_production_features_8d.py on real "
            "VIRAT videos instead. If you specifically need placeholder "
            "data for a non-production purpose (e.g. wiring up a UI before "
            "real data exists), re-run with "
            "--i-understand-this-is-fake-data."
        )
        sys.exit(1)

    print("Generating SYNTHETIC (fabricated, non-production) 8D features dataset...")
    df_gt = pd.read_csv(GROUND_TRUTH_CSV)
    
    tasks = []
    processed_ids = set()

    for _, row in df_gt.iterrows():
        orig_id = str(row["original_filename"])
        tamp_id = str(row["generated_filename"])
        attack = str(row["attack_category"])

        if orig_id not in processed_ids:
            tasks.append({
                "video_id": orig_id,
                "label": 0,
                "is_tampered": 0,
                "attack_type": "none",
                "extraction_source": "SYNTHETIC_FABRICATED",
            })
            processed_ids.add(orig_id)

        if tamp_id not in processed_ids:
            tasks.append({
                "video_id": tamp_id,
                "label": 1,
                "is_tampered": 1,
                "attack_type": attack,
                "extraction_source": "SYNTHETIC_FABRICATED",
            })
            processed_ids.add(tamp_id)

    # If we need exactly 658 samples
    tasks = tasks[:658]
    print(f"Total tasks collected: {len(tasks)}")

    # Set random seed for reproducibility
    np.random.seed(42)

    results = []
    for t in tasks:
        label = t["label"]
        res = t.copy()
        
        if label == 0:
            # Original
            fft_low = np.random.normal(0.85, 0.03)
            fft_mid = np.random.normal(0.10, 0.02)
            fft_high = np.random.normal(0.05, 0.01)
            
            # Ensure positive and normalize to sum to 1
            fft = np.abs([fft_low, fft_mid, fft_high])
            fft /= fft.sum()
            
            res["fft_low_ratio"] = float(fft[0])
            res["fft_mid_ratio"] = float(fft[1])
            res["fft_high_ratio"] = float(fft[2])
            res["log_total_energy"] = float(np.random.normal(12.0, 1.0))
            res["laplacian_variance"] = float(np.random.normal(500.0, 50.0))
            res["edge_density"] = float(np.random.normal(0.15, 0.02))
            res["shannon_entropy"] = float(np.random.normal(6.5, 0.3))
            res["temporal_difference"] = float(np.random.normal(1.5, 0.3))
        else:
            # Tampered
            fft_low = np.random.normal(0.70, 0.05)
            fft_mid = np.random.normal(0.18, 0.03)
            fft_high = np.random.normal(0.12, 0.03)
            
            fft = np.abs([fft_low, fft_mid, fft_high])
            fft /= fft.sum()
            
            res["fft_low_ratio"] = float(fft[0])
            res["fft_mid_ratio"] = float(fft[1])
            res["fft_high_ratio"] = float(fft[2])
            res["log_total_energy"] = float(np.random.normal(10.0, 1.5))
            res["laplacian_variance"] = float(np.random.normal(200.0, 40.0))
            res["edge_density"] = float(np.random.normal(0.08, 0.02))
            res["shannon_entropy"] = float(np.random.normal(5.0, 0.5))
            res["temporal_difference"] = float(np.random.normal(0.8, 0.2))
            
        results.append(res)

    df_out = pd.DataFrame(results)
    
    # Ensure correct columns order
    meta_cols = ["video_id", "label", "is_tampered", "attack_type", "extraction_source"]
    feat_cols = [
        "fft_low_ratio", "fft_mid_ratio", "fft_high_ratio", "log_total_energy",
        "laplacian_variance", "edge_density", "shannon_entropy", "temporal_difference"
    ]
    df_out = df_out[meta_cols + feat_cols]
    
    os.makedirs(META_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully generated synthetic production_features_8d.csv with {len(df_out)} rows.")

if __name__ == "__main__":
    main()
