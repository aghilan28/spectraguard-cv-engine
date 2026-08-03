import os
import sys
import pandas as pd
import numpy as np

# Add project root to python path
sys.path.insert(0, os.path.abspath("."))

from scripts.evaluation.inference_adapter import InferenceAdapter

def main():
    print("[EXTRACTION] Starting real FFT feature extraction from VIRAT benchmark...")
    
    meta_dir = os.path.join("data", "datasets", "virat", "metadata")
    gt_path = os.path.join(meta_dir, "ground_truth.csv")
    
    if not os.path.exists(gt_path):
        print(f"[ERROR] Ground truth CSV not found at: {gt_path}")
        sys.exit(1)
        
    df_gt = pd.read_csv(gt_path)
    print(f"[EXTRACTION] Loaded ground truth manifest with {len(df_gt)} rows.")
    
    records = []
    
    # Track metrics
    extracted_clean = 0
    extracted_tampered = 0
    
    for idx, row in df_gt.iterrows():
        orig_file = row["original_filename"]
        gen_file = row["generated_filename"]
        attack = row["attack_category"]
        
        # 1. Clean video extraction (Label 0)
        clean_path = os.path.join("data", "datasets", "virat", "original", orig_file)
        if os.path.exists(clean_path):
            try:
                features, _ = InferenceAdapter.extract_fft_features(clean_path)
                record = {f"fft_{i}": features[i] for i in range(10)}
                record.update({
                    "label": 0,
                    "video_id": orig_file,
                    "is_synthetic": 0
                })
                records.append(record)
                extracted_clean += 1
            except Exception as e:
                print(f"[WARNING] Failed to extract from clean video {orig_file}: {e}")
        else:
            print(f"[WARNING] Clean video not found: {clean_path}")
            
        # 2. Tampered video extraction (Label 1)
        if attack != "none":
            tamp_path = os.path.join("data", "datasets", "virat", "tampered", attack, gen_file)
            if os.path.exists(tamp_path):
                try:
                    features, _ = InferenceAdapter.extract_fft_features(tamp_path)
                    record = {f"fft_{i}": features[i] for i in range(10)}
                    record.update({
                        "label": 1,
                        "video_id": gen_file,
                        "is_synthetic": 0
                    })
                    records.append(record)
                    extracted_tampered += 1
                except Exception as e:
                    print(f"[WARNING] Failed to extract from tampered video {gen_file}: {e}")
            else:
                print(f"[WARNING] Tampered video not found: {tamp_path}")

    # Build and save feature matrix
    if not records:
        print("[ERROR] No features extracted. Aborting.")
        sys.exit(1)
        
    df_features = pd.DataFrame(records)
    out_path = os.path.join(meta_dir, "extracted_fft_features.csv")
    df_features.to_csv(out_path, index=False)
    
    print(f"[SUCCESS] Feature extraction completed.")
    print(f" -> Clean samples extracted: {extracted_clean}")
    print(f" -> Tampered samples extracted: {extracted_tampered}")
    print(f" -> Total sample count in feature matrix: {len(df_features)}")
    print(f" -> Saved feature matrix to: {out_path}")

if __name__ == "__main__":
    main()
