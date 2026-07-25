"""Comprehensive statistical evaluation for classification models."""

import os
import time
import json
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from ..models.trainer import ModelTrainer


class ModelEvaluator:
    """
    Evaluates trained ModelTrainers against held-out test sets, extracting
    classification metrics and inference latency profiles.
    """

    @staticmethod
    def evaluate(
        trainer: ModelTrainer, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Executes inference on X_test and calculates performance metrics against y_test.

        Args:
            trainer: A fully trained ModelTrainer instance.
            X_test: Feature matrix of the test set.
            y_test: True labels of the test set.

        Returns:
            Dictionary containing metrics and confusion matrix details.
        """
        if not trainer.is_trained:
            raise RuntimeError("Cannot evaluate an untrained model.")

        if len(X_test) == 0 or len(y_test) == 0:
            raise ValueError("Test dataset cannot be empty.")

        # Measure Inference Latency
        start_time = time.perf_counter()
        y_pred = trainer.predict(X_test)
        end_time = time.perf_counter()

        total_latency_ms = (end_time - start_time) * 1000
        avg_latency_per_sample_ms = total_latency_ms / len(X_test)

        # Probabilistic metrics (ROC-AUC requires probas if available)
        try:
            y_proba = trainer.predict_proba(X_test)
            # Assuming binary classification, take prob of positive class (1)
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        except (AttributeError, IndexError):
            roc_auc = None

        # Core Classification Metrics (macro average for potential multi-class scaling)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Confusion Matrix Analysis
        cm = confusion_matrix(y_test, y_pred)

        # Assuming binary (0=Negative, 1=Positive) for FPR/FNR extraction
        fpr, fnr = None, None
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            "model_type": trainer.config.model_type,
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc) if roc_auc is not None else None,
            },
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "false_positive_rate": float(fpr) if fpr is not None else None,
                "false_negative_rate": float(fnr) if fnr is not None else None,
            },
            "performance": {
                "test_samples": len(X_test),
                "total_inference_ms": float(total_latency_ms),
                "avg_inference_ms_per_sample": float(avg_latency_per_sample_ms),
            },
        }

    @staticmethod
    def save_report(report: Dict[str, Any], filepath: str) -> None:
        """Serializes the evaluation dictionary to a formatted JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
