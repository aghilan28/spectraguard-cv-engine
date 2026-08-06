import os
import joblib
import pandas as pd
import json
import numpy as np
from sklearn.metrics import precision_recall_curve

class ThresholdOptimizer:
    def __init__(self, features_dir, model_dir):
        self.features_dir = features_dir
        self.model_dir = model_dir

    def run(self):
        print("[ThresholdOptimizer] Optimizing decision threshold...")
        val_path = os.path.join(self.features_dir, "validation_features.csv")
        model_path = os.path.join(self.model_dir, "production_model.joblib")
        scaler_path = os.path.join(self.model_dir, "feature_scaler.joblib")

        if not (os.path.exists(val_path) and os.path.exists(model_path) and os.path.exists(scaler_path)):
            print("[ThresholdOptimizer] Dependencies missing. Skipping threshold optimization.")
            return

        df = pd.read_csv(val_path)
        X = df.drop(columns=["label"])
        y = df["label"]

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Get feature names from metadata
        meta_path = os.path.join(self.model_dir, "feature_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            feature_names = meta.get("feature_names") or meta.get("feature_order")

        from training_v2.utils.feature_dataframe import build_feature_dataframe
        X_reordered = X[feature_names]
        X_df = build_feature_dataframe(X_reordered.values.tolist(), feature_names)

        X_scaled = scaler.transform(X_df)
        probs = model.predict_proba(X_scaled)[:, 1]


        # Calculate PR Curve
        precisions, recalls, thresholds = precision_recall_curve(y, probs)
        
        # Avoid division by zero
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        # Check if thresholds list is empty (e.g. all predictions identical)
        if len(thresholds) > 0:
            best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
        else:
            best_threshold = 0.5

        threshold_data = {
            "optimal_threshold": best_threshold,
            "bounds": best_threshold,
            "validation_f1": float(f1_scores[best_idx]),
            "validation_precision": float(precisions[best_idx]),
            "validation_recall": float(recalls[best_idx])
        }

        out_path = os.path.join(self.model_dir, "threshold.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(threshold_data, f, indent=4)

        print(f"[ThresholdOptimizer] Optimal threshold determined: {best_threshold:.4f}")
        print(f"[ThresholdOptimizer] Threshold file saved to {out_path}")
        return threshold_data
