import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

class FeatureDistributionAnalyzer:
    def __init__(self, features_dir, reports_dir):
        self.features_dir = features_dir
        self.reports_dir = reports_dir

    def run(self):
        print("[DistributionAnalyzer] Checking dataset distributions...")
        train_path = os.path.join(self.features_dir, "train_features.csv")
        val_path = os.path.join(self.features_dir, "validation_features.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print("[DistributionAnalyzer] Training or validation feature CSV files not found. Skipping analysis.")
            return

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        
        feature_cols = [c for c in df_train.columns if c != 'label']
        comparison = {}

        for col in feature_cols:
            train_vals = df_train[col].values
            val_vals = df_val[col].values
            
            # Kolmogorov-Smirnov statistic
            ks_stat, p_val = ks_2samp(train_vals, val_vals)
            
            comparison[col] = {
                "train_mean": float(np.mean(train_vals)),
                "train_std": float(np.std(train_vals)),
                "val_mean": float(np.mean(val_vals)),
                "val_std": float(np.std(val_vals)),
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val),
                "drift_detected": bool(p_val < 0.05)
            }

        os.makedirs(self.reports_dir, exist_ok=True)
        comparison_path = os.path.join(self.reports_dir, "feature_comparison.json")
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=4)
        print(f"[DistributionAnalyzer] Saved distribution comparison report to {comparison_path}")
        return comparison
