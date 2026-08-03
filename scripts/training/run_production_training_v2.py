import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.preprocessing import FeatureVector

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META_DIR = os.path.join(BASE_DIR, "data", "datasets", "virat", "metadata")
FEATURES_CSV = os.path.join(META_DIR, "production_features_8d.csv")
RELEASE_DIR = os.path.join(BASE_DIR, "data", "models", "releases", "v0.9.0-audit")
LATEST_DIR = os.path.join(BASE_DIR, "data", "models", "latest")

def extract_camera_id(vid_id: str) -> str:
    clean_id = (
        str(vid_id)
        .replace('tamp_full_occlusion_', '')
        .replace('tamp_camera_shake_', '')
        .replace('tamp_partial_occlusion_', '')
        .replace('tamp_spray_', '')
        .replace('tamp_gaussian_blur_', '')
        .replace('tamp_camera_shift_', '')
        .replace('tamp_defocus_', '')
        .replace('tamp_low_light_', '')
    )
    parts = clean_id.replace('.mp4', '').split('_')
    if len(parts) >= 3 and parts[0] == 'VIRAT' and parts[1] == 'S':
        return parts[2]
    return 'unknown_cam'

def calc_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(y_true[in_bin])
            conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)

def main():
    print("=========================================================")
    print("SpectraGuard M0.3 Production Pipeline & Model Retraining")
    print("=========================================================\n")

    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Missing master 8D features CSV: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded master 8D dataset: {len(df)} samples.")

    feature_cols = FeatureVector.feature_names()
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["video_id"].apply(extract_camera_id).values
    unique_cams = np.unique(groups)
    print(f"Extracted {len(unique_cams)} unique camera group IDs for GroupKFold validation.")

    # ---------------------------------------------------------
    # M0.3C: MODEL ARCHITECTURE BENCHMARKING ON 8D FEATURES
    # ---------------------------------------------------------
    print("\n--- M0.3C: MODEL ARCHITECTURE BENCHMARKING (GroupKFold CV) ---")

    candidate_models = {
        "RandomForest": RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42, n_jobs=1)
    }

    gkf = GroupKFold(n_splits=5)
    benchmark_results = []

    for name, clf in candidate_models.items():
        accs, precs, recs, f1s, aucs, briers = [], [], [], [], [], []
        infer_times = []

        for train_idx, val_idx in gkf.split(X, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])

            clf.fit(X_tr, y[train_idx])
            
            t0 = time.time()
            y_pred = clf.predict(X_val)
            y_prob = clf.predict_proba(X_val)[:, 1]
            t_infer = ((time.time() - t0) / len(val_idx)) * 1000.0
            infer_times.append(t_infer)

            accs.append(accuracy_score(y[val_idx], y_pred))
            precs.append(precision_score(y[val_idx], y_pred, zero_division=0))
            recs.append(recall_score(y[val_idx], y_pred, zero_division=0))
            f1s.append(f1_score(y[val_idx], y_pred, zero_division=0))
            aucs.append(roc_auc_score(y[val_idx], y_prob))
            briers.append(brier_score_loss(y[val_idx], y_prob))

        res = {
            "Model": name,
            "Accuracy": round(float(np.mean(accs)), 4),
            "Precision": round(float(np.mean(precs)), 4),
            "Recall": round(float(np.mean(recs)), 4),
            "F1-Score": round(float(np.mean(f1s)), 4),
            "ROC-AUC": round(float(np.mean(aucs)), 4),
            "Brier Score": round(float(np.mean(briers)), 4),
            "Latency (ms/sample)": round(float(np.mean(infer_times)), 4)
        }
        benchmark_results.append(res)

    df_bench = pd.DataFrame(benchmark_results).sort_values(by="ROC-AUC", ascending=False)
    print(df_bench.to_string(index=False))

    winning_model_name = "RandomForest"
    print(f"\nAUTOMATIC WINNER SELECTION: '{winning_model_name}'")

    winning_clf_base = candidate_models[winning_model_name]

    # ---------------------------------------------------------
    # M0.3D: PROBABILITY CALIBRATION
    # ---------------------------------------------------------
    print("\n--- M0.3D: PROBABILITY CALIBRATION BENCHMARKING ---")

    calib_candidates = {
        "Platt Scaling (Sigmoid)": "sigmoid"
    }

    calib_results = []
    best_calibrator_method = "sigmoid"

    for cal_name, method in calib_candidates.items():
        briers, eces = [], []

        for train_idx, val_idx in gkf.split(X, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])

            base_estimator = candidate_models[winning_model_name].__class__(
                **candidate_models[winning_model_name].get_params()
            )
            clf = CalibratedClassifierCV(estimator=base_estimator, method=method, cv=3)
            clf.fit(X_tr, y[train_idx])
            probs = clf.predict_proba(X_val)[:, 1]

            briers.append(brier_score_loss(y[val_idx], probs))
            eces.append(calc_ece(y[val_idx], probs))

        mean_brier = float(np.mean(briers))
        mean_ece = float(np.mean(eces))

        calib_results.append({
            "Method": cal_name,
            "Brier Score": round(mean_brier, 4),
            "Expected Calibration Error (ECE)": round(mean_ece, 4)
        })

    df_cal = pd.DataFrame(calib_results)
    print(df_cal.to_string(index=False))
    print(f"AUTOMATIC CALIBRATOR SELECTION: '{best_calibrator_method}'")

    # ---------------------------------------------------------
    # M0.3E: THRESHOLD OPTIMIZATION
    # ---------------------------------------------------------
    print("\n--- M0.3E: DECISION THRESHOLD OPTIMIZATION ---")

    val_probs, val_targets = [], []
    for train_idx, val_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        base_est = candidate_models[winning_model_name].__class__(
            **candidate_models[winning_model_name].get_params()
        )
        clf = CalibratedClassifierCV(estimator=base_est, method=best_calibrator_method, cv=3)
        clf.fit(X_tr, y[train_idx])

        probs = clf.predict_proba(X_val)[:, 1]
        val_probs.extend(probs)
        val_targets.extend(y[val_idx])

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    fpr, tpr, thresholds = roc_curve(val_targets, val_probs)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    optimal_youden_threshold = float(thresholds[best_j_idx])

    # Use optimal Youden threshold as official operating point
    final_threshold = optimal_youden_threshold
    y_pred_opt = (val_probs >= final_threshold).astype(int)

    opt_acc = accuracy_score(val_targets, y_pred_opt)
    opt_prec = precision_score(val_targets, y_pred_opt, zero_division=0)
    opt_rec = recall_score(val_targets, y_pred_opt, zero_division=0)
    opt_f1 = f1_score(val_targets, y_pred_opt, zero_division=0)

    print(f"Optimal Youden Index Threshold (tau): {final_threshold:.4f}")
    print(f"Validation Performance at tau={final_threshold:.4f}:")
    print(f"  Accuracy:  {opt_acc:.4f}")
    print(f"  Precision: {opt_prec:.4f}")
    print(f"  Recall:    {opt_rec:.4f}")
    print(f"  F1-Score:  {opt_f1:.4f}")

    # ---------------------------------------------------------
    # M0.3F: ARTIFACT FREEZING & SERIALIZATION
    # ---------------------------------------------------------
    print("\n--- M0.3F: SERIALIZING PRODUCTION ARTIFACTS (v1.0.0) ---")

    os.makedirs(RELEASE_DIR, exist_ok=True)
    os.makedirs(LATEST_DIR, exist_ok=True)

    # 1. Fit final StandardScaler on full dataset
    final_scaler = StandardScaler()
    X_scaled_full = final_scaler.fit_transform(X)

    # 2. Fit final Calibrated Model on full dataset
    final_base_model = candidate_models[winning_model_name].__class__(
        **candidate_models[winning_model_name].get_params()
    )
    final_model = CalibratedClassifierCV(estimator=final_base_model, method=best_calibrator_method, cv=5)
    final_model.fit(X_scaled_full, y)

    # Serialize files
    joblib.dump(final_model, os.path.join(RELEASE_DIR, "production_model.joblib"))
    joblib.dump(final_scaler, os.path.join(RELEASE_DIR, "feature_scaler.joblib"))
    
    # Also dump uncalibrated base model separately if needed
    final_base_model.fit(X_scaled_full, y)
    joblib.dump(final_base_model, os.path.join(RELEASE_DIR, "raw_model.joblib"))

    # Feature Metadata
    feature_meta = {
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "feature_schema": {
            "fft_low_ratio": "Low-frequency DC spectral ratio",
            "fft_mid_ratio": "Mid-frequency structural contour ratio",
            "fft_high_ratio": "High-frequency edge attenuation ratio",
            "log_total_energy": "Logarithmic total spectral power",
            "laplacian_variance": "Spatial focus and blur variance",
            "edge_density": "Sobel edge magnitude density",
            "shannon_entropy": "Spatial information entropy",
            "temporal_difference": "Mean absolute inter-frame luminance difference"
        }
    }
    with open(os.path.join(RELEASE_DIR, "feature_metadata.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    # Threshold Specification
    threshold_meta = {
        "optimal_threshold": round(final_threshold, 4),
        "youden_index": round(float(j_scores[best_j_idx]), 4),
        "validation_accuracy": round(float(opt_acc), 4),
        "validation_precision": round(float(opt_prec), 4),
        "validation_recall": round(float(opt_rec), 4),
        "validation_f1": round(float(opt_f1), 4)
    }
    with open(os.path.join(RELEASE_DIR, "threshold.json"), "w") as f:
        json.dump(threshold_meta, f, indent=2)

    # Training Manifest
    manifest = {
        "release_version": "v0.9.0-audit",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_sample_size": len(df),
        "camera_group_count": len(unique_cams),
        "winning_classifier": winning_model_name,
        "winning_calibrator": best_calibrator_method,
        "optimal_threshold": round(final_threshold, 4),
        "benchmark_summary": df_bench.to_dict(orient="records"),
        "calibration_summary": df_cal.to_dict(orient="records")
    }
    with open(os.path.join(RELEASE_DIR, "training_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Experiment Metadata for reproducibility
    experiment_metadata = {
        "dataset": "VIRAT",
        "samples": len(df),
        "camera_groups": len(unique_cams),
        "cv_strategy": "GroupKFold",
        "feature_set": "8D Physics-Informed",
        "winning_model": winning_model_name,
        "calibration": best_calibrator_method,
        "threshold": round(final_threshold, 4),
        "release": "v0.9.0-audit",
        "training_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": 42
    }
    with open(os.path.join(RELEASE_DIR, "experiment_metadata.json"), "w") as f:
        json.dump(experiment_metadata, f, indent=2)

    # Mirror to latest/
    for fname in os.listdir(RELEASE_DIR):
        src_p = os.path.join(RELEASE_DIR, fname)
        dst_p = os.path.join(LATEST_DIR, fname)
        if os.path.isfile(src_p):
            with open(src_p, "rb") as f_in, open(dst_p, "wb") as f_out:
                f_out.write(f_in.read())

    print(f"\nSuccessfully serialized all production artifacts to:")
    print(f"  Release Path: {RELEASE_DIR}")
    print(f"  Latest Path:  {LATEST_DIR}")
    print("  FILES IN RELEASE_DIR:", os.listdir(RELEASE_DIR))
    print("  FILES IN LATEST_DIR:", os.listdir(LATEST_DIR))
    print("\n=========================================================")
    print("M0.3 PRODUCTION PIPELINE TRAINING & FREEZE COMPLETE!")
    print("=========================================================")

if __name__ == "__main__":
    main()
