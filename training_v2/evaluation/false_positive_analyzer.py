import os
import shutil
import joblib
import json
import numpy as np
import cv2
from training_v2.features.feature_extractor import FeatureExtractor

class FalsePositiveAnalyzer:
    def __init__(self, dataset_dir, model_dir, reports_dir):
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self.reports_dir = reports_dir
        self.extractor = FeatureExtractor(dataset_dir, model_dir)

    def run(self):
        print("[FalsePositiveAnalyzer] Starting False Positive diagnostics...")
        model_path = os.path.join(self.model_dir, "production_model.joblib")
        scaler_path = os.path.join(self.model_dir, "feature_scaler.joblib")
        threshold_path = os.path.join(self.model_dir, "threshold.json")
        dist_path = os.path.join(self.model_dir, "feature_distribution.json")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(threshold_path)):
            print("[FalsePositiveAnalyzer] Candidate model dependencies missing. Skipping.")
            return

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(threshold_path, "r", encoding="utf-8") as f:
            thresh_val = json.load(f).get("optimal_threshold", 0.5)

        # Load training distributions
        dist_normal = {}
        if os.path.exists(dist_path):
            with open(dist_path, "r", encoding="utf-8") as f:
                dist_normal = json.load(f).get("Normal", {})

        # Load feature metadata
        meta_path = os.path.join(self.model_dir, "feature_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                feature_names = meta.get("feature_names") or meta.get("feature_order")
        else:
            feature_names = self.extractor.feature_order

        from training_v2.utils.feature_dataframe import build_feature_dataframe

        fp_dir = os.path.join(self.reports_dir, "false_positives")
        os.makedirs(fp_dir, exist_ok=True)
        
        log_entries = []
        fp_count = 0

        # Scan validation/test directories for 'Normal' files
        for split in ["validation", "test"]:
            normal_dir = os.path.join(self.dataset_dir, split, "Normal")
            if not os.path.exists(normal_dir):
                continue
            
            for file in os.listdir(normal_dir):
                filepath = os.path.join(normal_dir, file)
                if not os.path.isfile(filepath):
                    continue
                
                try:
                    feat_dict = self.extractor.extract_from_image(filepath)
                    row = [feat_dict.get(feat, 0.0) for feat in feature_names]
                    
                    df_row = build_feature_dataframe(row, feature_names)
                    row_scaled = scaler.transform(df_row)
                    prob = float(model.predict_proba(row_scaled)[:, 1][0])

                    
                    if prob >= thresh_val:
                        fp_count += 1
                        dest_path = os.path.join(fp_dir, f"{split}_fp_{file}")
                        shutil.copy2(filepath, dest_path)
                        
                        # Diagnostics: find anomalous features
                        anomalies = []
                        for idx, feat_name in enumerate(self.extractor.feature_order):
                            val = row[idx]
                            feat_stat = dist_normal.get(feat_name, {})
                            mean = feat_stat.get("mean", 0.0)
                            std = feat_stat.get("std", 1.0)
                            
                            # If feature is more than 2 std dev away
                            if std > 0:
                                z_score = (val - mean) / std
                                if abs(z_score) > 2.0:
                                    anomalies.append(f"{feat_name} deviates by {z_score:.2f} std dev (value: {val:.4f}, expected mean: {mean:.4f})")
                        
                        log_entries.append({
                            "file": file,
                            "split": split,
                            "tamper_probability": prob,
                            "threshold": thresh_val,
                            "anomalous_features": anomalies
                        })
                except Exception as e:
                    pass

        # Write reports
        report_path = os.path.join(self.reports_dir, "false_positive_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "false_positive_count": fp_count,
                "diagnostics": log_entries
            }, f, indent=4)

        print(f"[FalsePositiveAnalyzer] Diagnosed {fp_count} False Positives. Report saved to {report_path}")
        return fp_count
