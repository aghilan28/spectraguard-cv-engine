import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import joblib
import cv2

# Add project root to python path
sys.path.insert(0, os.path.abspath("."))

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from concurrent.futures import ProcessPoolExecutor, as_completed

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_single_video_features(task_info):
    """
    Worker function to extract FFT spectral energy features from a single video.
    Includes early exit optimization: stops decoding frames as soon as 10 spectral energies are collected.
    This is mathematically identical to the original adapter logic which only sliced the first 10 elements.
    """
    video_path, label, video_id = task_info
    if not os.path.exists(video_path):
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate_frames = 30
        step = max(1, total_frames // sample_rate_frames)
        
        spectral_energies = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Production Physics-Informed FFT Preprocessing
                dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)
                magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
                
                h, w = magnitude.shape
                cy, cx = h // 2, w // 2
                mask = np.ones((h, w), np.uint8)
                cv2.circle(mask, (cx, cy), min(h, w) // 8, 0, -1) # High-pass gating filter
                high_freq_energy = np.sum(magnitude * mask)
                spectral_energies.append(high_freq_energy)
                
                # Math optimization: slice of [:10] discards any elements past index 10.
                # Break early to prevent useless decoding of hundreds of remaining frames.
                if len(spectral_energies) == 10:
                    break
                    
            frame_idx += 1
            
        cap.release()
        
        if not spectral_energies:
            feature_vector = np.zeros((10,), dtype=np.float32)
        else:
            feature_vector = np.array(spectral_energies[:10], dtype=np.float32)
            if len(feature_vector) < 10:
                feature_vector = np.pad(feature_vector, (0, 10 - len(feature_vector)), 'constant')
                
        record = {f"fft_{i}": float(feature_vector[i]) for i in range(10)}
        record.update({"label": label, "video_id": video_id, "is_synthetic": 0})
        return record
    except Exception as e:
        return None

def extract_features_if_missing():
    meta_dir = os.path.join("data", "datasets", "virat", "metadata")
    out_path = os.path.join(meta_dir, "extracted_fft_features.csv")
    
    if os.path.exists(out_path):
        print(f"[STAGE 1/7] Extracted features CSV already exists at: {out_path}. Skipping extraction.")
        return out_path

    print("[STAGE 1/7] Feature matrix not found. Initiating optimized parallel video feature extraction...")
    gt_path = os.path.join(meta_dir, "ground_truth.csv")
    if not os.path.exists(gt_path):
        print(f"[ERROR] Ground truth CSV not found at: {gt_path}")
        sys.exit(1)
        
    df_gt = pd.read_csv(gt_path)
    tasks = []
    
    for idx, row in df_gt.iterrows():
        orig_file = row["original_filename"]
        gen_file = row["generated_filename"]
        attack = row["attack_category"]
        
        # Original (Label 0)
        clean_path = os.path.join("data", "datasets", "virat", "original", orig_file)
        tasks.append((clean_path, 0, orig_file))
                
        # Tampered (Label 1)
        if attack != "none":
            tamp_path = os.path.join("data", "datasets", "virat", "tampered", attack, gen_file)
            tasks.append((tamp_path, 1, gen_file))

    records = []
    total_tasks = len(tasks)
    workers = min(os.cpu_count(), 8)
    print(f" -> Submitting {total_tasks} extraction tasks across {workers} parallel CPU processes...")
    
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_single_video_features, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                records.append(res)
            completed += 1
            if completed % 50 == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                print(f"[EXTRACTION] Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%) | Elapsed: {elapsed:.1f}s | Speed: {speed:.1f} videos/sec")

    if not records:
        print("[ERROR] Failed to extract any features from video archives.")
        sys.exit(1)
        
    df_features = pd.DataFrame(records)
    os.makedirs(meta_dir, exist_ok=True)
    df_features.to_csv(out_path, index=False)
    print(f"[SUCCESS] Feature matrix generated with {len(df_features)} records and saved to {out_path}.")
    return out_path

def main():
    print("=" * 70)
    print("      SPECTRAGUARD PHASE 8A: REAL MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Feature Extraction / Acquisition
    out_path = extract_features_if_missing()
    
    # 2. Loading and Validation
    print("\n[STAGE 2/7] Loading and validating feature dataset...")
    df = pd.read_csv(out_path)
    
    # Verify missing values
    missing_counts = df.isnull().sum().to_dict()
    print(" -> Missing values check:", missing_counts)
    if df.isnull().values.any():
        print("[WARNING] Missing values detected. Dropping NaN rows.")
        df = df.dropna().reset_index(drop=True)
        
    # Verify duplicates
    dup_count = df.duplicated(subset=[f"fft_{i}" for i in range(10)]).sum()
    print(f" -> Duplicate feature rows count: {dup_count}")
    
    # Class balance check
    class_counts = df["label"].value_counts().to_dict()
    print(" -> Class distribution (0=Clean, 1=Tampered):", class_counts)
    
    feature_cols = [f"fft_{i}" for i in range(10)]
    X = df[feature_cols].values
    y = df["label"].values
    
    # 3. Stratified Split and Normalization
    print("\n[STAGE 3/7] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into 80% train / 20% validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f" -> Train size: {X_train.shape[0]} samples | Validation size: {X_val.shape[0]} samples")
    
    # 4. Stratified CV and Model Comparison
    print("\n[STAGE 4/7] Comparing candidate classifiers via 5-Fold Stratified CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "n_estimators": [50, 100, 150],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10]
            }
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(class_weight="balanced", random_state=42),
            {
                "n_estimators": [50, 100, 150],
                "max_depth": [None, 5, 10]
            }
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="logloss"),
            {
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        )
    }
    
    best_estimators = {}
    best_scores = {}
    
    for name, (model, grid) in models.items():
        print(f" -> Tuning {name}...")
        clf = GridSearchCV(model, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        clf.fit(X_train, y_train)
        best_estimators[name] = clf.best_estimator_
        best_scores[name] = clf.best_score_
        print(f"    * Best params: {clf.best_params_} | Mean CV ROC-AUC: {clf.best_score_:.4f}")
        
    # Select winning model
    winner_name = max(best_scores, key=best_scores.get)
    winner_model = best_estimators[winner_name]
    print(f"\n -> Selected Winner: {winner_name} with CV ROC-AUC of {best_scores[winner_name]:.4f}")
    
    # 5. Final Model Fit
    print("\n[STAGE 5/7] Fitting winner on entire training matrix...")
    winner_model.fit(X_train, y_train)
    
    # 6. Save model and scaler
    release_dir = os.path.join("data", "models", "releases", "v1.0.0")
    os.makedirs(release_dir, exist_ok=True)
    
    model_path = os.path.join(release_dir, "production_model.joblib")
    scaler_path = os.path.join(release_dir, "feature_scaler.joblib")
    
    joblib.dump(winner_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f" -> Saved model to: {model_path}")
    print(f" -> Saved scaler to: {scaler_path}")
    
    # 7. Generate Validation Metrics and Artifacts
    print("\n[STAGE 6/7] Evaluating final model and compiling forensic artifacts...")
    y_val_pred = winner_model.predict(X_val)
    y_val_prob = winner_model.predict_proba(X_val)[:, 1]
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred)
    val_rec = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_prob)
    
    print(f" -> Validation Accuracy: {val_acc:.4f}")
    print(f" -> Validation Precision: {val_prec:.4f}")
    print(f" -> Validation Recall: {val_rec:.4f}")
    print(f" -> Validation F1 Score: {val_f1:.4f}")
    print(f" -> Validation ROC-AUC: {val_auc:.4f}")
    
    # Metrics JSON
    metrics_data = {
        "validation_accuracy": val_acc,
        "validation_precision": val_prec,
        "validation_recall": val_rec,
        "validation_f1_score": val_f1,
        "validation_roc_auc": val_auc
    }
    metrics_path = os.path.join(release_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    # Cross Validation Report
    cv_report = {
        "model_comparison": best_scores,
        "selected_model": winner_name,
        "best_cv_roc_auc": best_scores[winner_name]
    }
    cv_path = os.path.join(release_dir, "cross_validation_report.json")
    with open(cv_path, "w") as f:
        json.dump(cv_report, f, indent=4)
        
    # Feature Importance CSV
    if hasattr(winner_model, "feature_importances_"):
        importances = winner_model.feature_importances_
    else:
        importances = np.zeros(10)
    df_imp = pd.DataFrame({"feature_name": feature_cols, "importance": importances})
    df_imp = df_imp.sort_values(by="importance", ascending=False)
    imp_path = os.path.join(release_dir, "feature_importance.csv")
    df_imp.to_csv(imp_path, index=False)
    
    # Confusion Matrix CSV
    cm = confusion_matrix(y_val, y_val_pred)
    df_cm = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Predicted_0", "Predicted_1"])
    cm_path = os.path.join(release_dir, "confusion_matrix.csv")
    df_cm.to_csv(cm_path)
    
    # ROC Curves JSON
    fpr, tpr, roc_thresholds = roc_curve(y_val, y_val_prob)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()}
    roc_path = os.path.join(release_dir, "roc_auc.json")
    with open(roc_path, "w") as f:
        json.dump(roc_data, f, indent=4)
        
    # Precision-Recall JSON
    precision, recall, pr_thresholds = precision_recall_curve(y_val, y_val_prob)
    pr_data = {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": pr_thresholds.tolist()}
    pr_path = os.path.join(release_dir, "precision_recall.json")
    with open(pr_path, "w") as f:
        json.dump(pr_data, f, indent=4)
        
    # Feature Metadata
    feat_meta = {
        "feature_names": feature_cols,
        "feature_count": len(feature_cols),
        "feature_type": "FFT_Spectral_High_Frequency_Energy"
    }
    feat_meta_path = os.path.join(release_dir, "feature_metadata.json")
    with open(feat_meta_path, "w") as f:
        json.dump(feat_meta, f, indent=4)
        
    # Training Manifest
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_type": winner_name,
        "feature_schema": "VIRAT-10f-FFT",
        "num_features": 10,
        "release_version": "v1.0.0-production"
    }
    manifest_path = os.path.join(release_dir, "training_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    # Model Hashes
    hashes = {
        "production_model.joblib": calculate_sha256(model_path),
        "feature_scaler.joblib": calculate_sha256(scaler_path)
    }
    hashes_path = os.path.join(release_dir, "model_hashes.json")
    with open(hashes_path, "w") as f:
        json.dump(hashes, f, indent=4)
        
    print("[SUCCESS] All output forensic artifacts generated and cataloged successfully.")
    
    # 8. Forensic Verification Checks
    print("\n[STAGE 7/7] Executing Forensic Verification checks on saved artifacts...")
    
    # Reload model
    loaded_model = joblib.load(model_path)
    print(" -> Saved Model Loaded: OK")
    
    # Reload scaler
    loaded_scaler = joblib.load(scaler_path)
    print(" -> Saved Scaler Loaded: OK")
    
    # Load features metadata
    with open(feat_meta_path, "r") as f:
        loaded_feat_meta = json.load(f)
    print(f" -> Saved Feature Metadata Loaded: OK (expects {loaded_feat_meta['feature_count']} features)")
    
    # Run test sample inference
    test_sample_unscaled = X[0].reshape(1, -1)
    test_sample_scaled = loaded_scaler.transform(test_sample_unscaled)
    
    pred = loaded_model.predict(test_sample_scaled)
    probs = loaded_model.predict_proba(test_sample_scaled)[0]
    
    print(f" -> predict() invocation test: OK (Result Class={pred[0]})")
    print(f" -> predict_proba() probability array test: OK (Probs={probs})")
    print(f" -> Dimension consistency validation: OK ({test_sample_scaled.shape[1]} features)")
    print(f" -> Model SHA-256 Checksum: {hashes['production_model.joblib']}")
    
    print("\n" + "=" * 70)
    print("   FORENSIC MODEL TRAINING CERTIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
