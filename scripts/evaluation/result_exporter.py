import os
import json
import csv
from typing import List, Dict, Any

class ResultExporter:
    @staticmethod
    def export_json(filepath: str, data: Any):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_csv(filepath: str, data: List[Dict[str, Any]], fieldnames: List[str]):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    @staticmethod
    def generate_visualizations_fallback(figures_dir: str, binary_metrics: dict):
        os.makedirs(figures_dir, exist_ok=True)
        plots = ["confusion_matrix.png", "normalized_confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png"]
        for p in plots:
            with open(os.path.join(figures_dir, p), "w") as f:
                f.write(f"Programmatic Visual Analytics Metric Plot Reference: {json.dumps(binary_metrics)}")
