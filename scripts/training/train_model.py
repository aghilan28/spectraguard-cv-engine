import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)

# Ensure root path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def train_and_export():
    print("=========================================================")
    print("SpectraGuard Random Forest Training & Optimization")
    print("=========================================================\n")

    dataset_path = "data/training_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"ERROR: Training dataset not found at {dataset_path}. Run generate_dataset.py first.")
        return

    df = pd.read_csv(dataset_path)
    
    # Filter only features we support
    feature_cols = ["fft_low_ratio", "fft_mid_ratio", "fft_high_ratio", "log_total_energy", 
                    "laplacian_variance", "edge_density", "shannon_entropy", "temporal_difference"]
    
    for col in feature_cols:
        if col not in df.columns:
            print(f"ERROR: Feature column '{col}' missing from dataset CSV.")
            return

    X = df[feature_cols]
    y = df["label"]
    
    # Class balancing check
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"Dataset Class Distribution: {class_dist}")
    
    if len(class_dist) < 2:
        print("ERROR: Cannot train model on a single class. Make sure you have both Normal and Tamper videos.")
        return

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Grid Search CV for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    print("Running GridSearchCV for hyperparameter optimization...")
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid = GridSearchCV(rf_base, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Parameters Found: {grid.best_params_}")
    
    # Validation predictions and probability extraction
    y_pred_default = best_model.predict(X_val_scaled)
    probs = best_model.predict_proba(X_val_scaled)[:, 1]
    
    # Threshold Optimization using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_val, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx])
    
    # Restrict threshold to a reasonable range to prevent extreme edge cases
    optimal_threshold = max(0.15, min(0.85, optimal_threshold))
    print(f"Optimized Decision Threshold: {optimal_threshold:.4f}")
    
    # Apply optimized threshold
    y_pred_opt = (probs >= optimal_threshold).astype(int)
    
    # Generate Metrics
    acc = float(accuracy_score(y_val, y_pred_opt))
    prec = float(precision_score(y_val, y_pred_opt, zero_division=0))
    rec = float(recall_score(y_val, y_pred_opt, zero_division=0))
    f1 = float(f1_score(y_val, y_pred_opt, zero_division=0))
    roc_auc = float(roc_auc_score(y_val, probs))
    
    cm = confusion_matrix(y_val, y_pred_opt).tolist()
    
    # PR Curve
    p_curve, r_curve, pr_thresh = precision_recall_curve(y_val, probs)
    
    # Feature Importance
    importances = best_model.feature_importances_
    feat_importances = {name: float(imp) for name, imp in zip(feature_cols, importances)}
    
    # Identify False Positives, False Negatives, and Misclassified Samples
    false_positives = []
    false_negatives = []
    misclassified = []
    
    X_val_list = X_val.values.tolist()
    y_val_list = y_val.tolist()
    
    for idx in range(len(y_val_list)):
        true_lbl = y_val_list[idx]
        pred_lbl = int(y_pred_opt[idx])
        prob_val = float(probs[idx])
        features_dict = dict(zip(feature_cols, X_val_list[idx]))
        
        sample_info = {
            "index": idx,
            "true_label": true_lbl,
            "predicted_label": pred_lbl,
            "probability": prob_val,
            "features": features_dict
        }
        
        if true_lbl != pred_lbl:
            misclassified.append(sample_info)
            if true_lbl == 0 and pred_lbl == 1:
                false_positives.append(sample_info)
            elif true_lbl == 1 and pred_lbl == 0:
                false_negatives.append(sample_info)

    # Compile Validation Summary Report
    metrics_summary = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "optimized_threshold": optimal_threshold,
        "confusion_matrix": cm,
        "feature_importance": feat_importances,
        "false_positives_count": len(false_positives),
        "false_negatives_count": len(false_negatives),
        "misclassified_count": len(misclassified),
        "misclassified_samples": misclassified[:20], # limit sample listings to 20
        "pr_curve": {
            "precision": p_curve.tolist(),
            "recall": r_curve.tolist(),
            "thresholds": pr_thresh.tolist()
        }
    }
    
    # Save Metrics File
    os.makedirs("reports/metrics", exist_ok=True)
    metrics_path = "reports/metrics/training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics Report Saved to {metrics_path}")

    # Step 6: Deploy Model (Save to data/models/latest/)
    out_dir = "data/models/latest"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save model
    joblib.dump(best_model, os.path.join(out_dir, "production_model.joblib"))
    # Save scaler under BOTH common filenames to ensure compatibility
    joblib.dump(scaler, os.path.join(out_dir, "feature_scaler.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    
    # Save feature metadata
    metadata = {
        "feature_names": feature_cols,
        "feature_order": feature_cols,
        "feature_count": len(feature_cols),
        "feature_type": "FFT_Spectral_High_Frequency_Energy"
    }
    with open(os.path.join(out_dir, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Save threshold
    with open(os.path.join(out_dir, "threshold.json"), "w") as f:
        json.dump({"optimal_threshold": optimal_threshold, "bounds": 0.5}, f, indent=4)
        
    print(f"\nModel deployed successfully to {out_dir}/")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    train_and_export()
