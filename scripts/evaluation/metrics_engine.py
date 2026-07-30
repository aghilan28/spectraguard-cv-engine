import numpy as np
from typing import List, Dict, Any, Tuple

class MetricsEngine:
    @staticmethod
    def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (fp + tn) if (fp + tn) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        balanced_acc = (tpr + tnr) / 2.0
        
        # Matthews Correlation Coefficient computation
        mcc_denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0
        
        return {
            "overall_accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "matthews_correlation_coefficient": round(mcc, 4),
            "true_positive_rate": round(tpr, 4),
            "false_positive_rate": round(fpr, 4),
            "true_negative_rate": round(tnr, 4),
            "false_negative_rate": round(fnr, 4),
            "confusion_matrix_raw": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        }

    @staticmethod
    def generate_curves(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        # Deterministic discrete thresholds iteration setup
        thresholds = np.linspace(0.0, 1.0, 50)
        roc_points = []
        pr_points = []
        
        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            tp = np.sum((y_true == 1) & (preds == 1))
            tn = np.sum((y_true == 0) & (preds == 0))
            fp = np.sum((y_true == 0) & (preds == 1))
            fn = np.sum((y_true == 1) & (preds == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            roc_points.append((float(fpr), float(tpr)))
            pr_points.append((float(tpr), float(precision)))
            
        # Mathematical AUC Riemann Integration approximation
        roc_points = sorted(roc_points, key=lambda x: x[0])
        roc_auc = 0.0
        for i in range(len(roc_points) - 1):
            roc_auc += 0.5 * (roc_points[i+1][0] - roc_points[i][0]) * (roc_points[i+1][1] + roc_points[i][1])
            
        return {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(np.mean([p[1] for p in pr_points])), 4),
            "roc_curve_coordinates": roc_points,
            "pr_curve_coordinates": pr_points
        }
