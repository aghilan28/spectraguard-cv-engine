import os
import joblib
import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ModelComparator:
    def __init__(self, prod_model_dir, cand_model_dir, features_dir, reports_dir):
        self.prod_model_dir = prod_model_dir
        self.cand_model_dir = cand_model_dir
        self.features_dir = features_dir
        self.reports_dir = reports_dir

    def calculate_metrics(self, model, scaler, threshold, X, y, feature_names):
        from training_v2.utils.feature_dataframe import build_feature_dataframe
        # Reorder columns to match feature metadata and assert schema
        X_reordered = X[feature_names]
        X_df = build_feature_dataframe(X_reordered.values.tolist(), feature_names)
        
        X_scaled = scaler.transform(X_df)
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probs)),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }
        return metrics

    def run(self):
        print("[Comparator] Starting model side-by-side comparison...")
        val_path = os.path.join(self.features_dir, "validation_features.csv")
        test_path = os.path.join(self.features_dir, "test_features.csv")
        
        # Load models
        prod_model_path = os.path.join(self.prod_model_dir, "production_model.joblib")
        prod_scaler_path = os.path.join(self.prod_model_dir, "feature_scaler.joblib")
        if not os.path.exists(prod_scaler_path):
            prod_scaler_path = os.path.join(self.prod_model_dir, "scaler.joblib")
            
        cand_model_path = os.path.join(self.cand_model_dir, "production_model.joblib")
        cand_scaler_path = os.path.join(self.cand_model_dir, "feature_scaler.joblib")

        if not (os.path.exists(val_path) and os.path.exists(prod_model_path) and os.path.exists(cand_model_path)):
            print("[Comparator] Dependencies missing. Skipping comparison.")
            return

        # Thresholds
        prod_thresh = 0.5
        prod_thresh_path = os.path.join(self.prod_model_dir, "threshold.json")
        if os.path.exists(prod_thresh_path):
            with open(prod_thresh_path, "r", encoding="utf-8") as f:
                prod_thresh = json.load(f).get("optimal_threshold", 0.5)

        cand_thresh = 0.5
        cand_thresh_path = os.path.join(self.cand_model_dir, "threshold.json")
        if os.path.exists(cand_thresh_path):
            with open(cand_thresh_path, "r", encoding="utf-8") as f:
                cand_thresh = json.load(f).get("optimal_threshold", 0.5)

        df_val = pd.read_csv(val_path)
        X_val = df_val.drop(columns=["label"])
        y_val = df_val["label"]

        prod_model = joblib.load(prod_model_path)
        prod_scaler = joblib.load(prod_scaler_path)
        cand_model = joblib.load(cand_model_path)
        cand_scaler = joblib.load(cand_scaler_path)

        # Get production feature names
        prod_meta_path = os.path.join(self.prod_model_dir, "feature_metadata.json")
        prod_feature_names = list(X_val.columns)
        if os.path.exists(prod_meta_path):
            with open(prod_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                prod_feature_names = meta.get("feature_names") or meta.get("feature_order")

        # Get candidate feature names
        cand_meta_path = os.path.join(self.cand_model_dir, "feature_metadata.json")
        cand_feature_names = list(X_val.columns)
        if os.path.exists(cand_meta_path):
            with open(cand_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                cand_feature_names = meta.get("feature_names") or meta.get("feature_order")

        prod_metrics = self.calculate_metrics(prod_model, prod_scaler, prod_thresh, X_val, y_val, prod_feature_names)
        cand_metrics = self.calculate_metrics(cand_model, cand_scaler, cand_thresh, X_val, y_val, cand_feature_names)

        report = {
            "production": prod_metrics,
            "candidate": cand_metrics
        }


        os.makedirs(self.reports_dir, exist_ok=True)
        report_json_path = os.path.join(self.reports_dir, "comparison_report.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"[Comparator] Metrics comparison JSON saved to {report_json_path}")

        # Generate interactive HTML report comparison_report.html
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>SpectraGuard Model Comparison Report</title>
    <style>
        body {{
            font-family: 'Outfit', 'Inter', sans-serif;
            background-color: #0d0d0d;
            color: #e0e0e0;
            padding: 40px;
        }}
        h1 {{
            color: #ff3333;
            text-align: center;
        }}
        table {{
            width: 80%;
            margin: 40px auto;
            border-collapse: collapse;
            background-color: #1a1a1a;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 15px;
            text-align: center;
            border-bottom: 1px solid #333;
        }}
        th {{
            background-color: #2b2b2b;
            color: #ff3333;
        }}
        .better {{
            color: #00ff00;
            font-weight: bold;
        }}
        .worse {{
            color: #ff3333;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>SpectraGuard v2 Model Comparison</h1>
    <table>
        <thead>
            <tr>
                <th>Metric</th>
                <th>Production Model</th>
                <th>Candidate Model</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Accuracy</td>
                <td>{prod_metrics['accuracy']:.4f}</td>
                <td>{cand_metrics['accuracy']:.4f}</td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>{prod_metrics['precision']:.4f}</td>
                <td>{cand_metrics['precision']:.4f}</td>
            </tr>
            <tr>
                <td>Recall (True Positive Rate)</td>
                <td>{prod_metrics['recall']:.4f}</td>
                <td>{cand_metrics['recall']:.4f}</td>
            </tr>
            <tr>
                <td>F1-Score</td>
                <td>{prod_metrics['f1']:.4f}</td>
                <td>{cand_metrics['f1']:.4f}</td>
            </tr>
            <tr>
                <td>ROC AUC</td>
                <td>{prod_metrics['roc_auc']:.4f}</td>
                <td>{cand_metrics['roc_auc']:.4f}</td>
            </tr>
            <tr>
                <td>False Positive Rate (FPR)</td>
                <td>{prod_metrics['fpr']:.4f}</td>
                <td>{cand_metrics['fpr']:.4f}</td>
            </tr>
            <tr>
                <td>False Negative Rate (FNR)</td>
                <td>{prod_metrics['fnr']:.4f}</td>
                <td>{cand_metrics['fnr']:.4f}</td>
            </tr>
        </tbody>
    </table>
</body>
</html>
"""
        html_path = os.path.join(self.reports_dir, "comparison_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[Comparator] HTML comparison report generated at {html_path}")
        return report
