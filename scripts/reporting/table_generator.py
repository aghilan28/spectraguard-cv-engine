import os
import pandas as pd
from typing import Dict, Any

class TableGenerator:
    def __init__(self, tables_dir: str):
        self.out_dir = tables_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def generate_all(self, preds_df: pd.DataFrame, eval_json: dict, perf_json: dict):
        preds_df['is_correct'] = (preds_df['ground_truth_label'] == preds_df['prediction_label']).astype(int)
        
        # Attack Metrics Table
        attack_df = preds_df.groupby('attack_category').agg(
            total_samples=('original_filename', 'count'),
            accuracy=('is_correct', 'mean')
        ).reset_index()
        attack_df.to_csv(os.path.join(self.out_dir, 'attack_metrics.csv'), index=False)

        # Scene Metrics Table
        scene_df = preds_df.groupby('source_scene_category').agg(
            total_samples=('original_filename', 'count'),
            accuracy=('is_correct', 'mean')
        ).reset_index()
        scene_df.to_csv(os.path.join(self.out_dir, 'scene_metrics.csv'), index=False)

        # Environment Summary Table
        env_data = [{"Metric": k, "Value": v} for k, v in perf_json.items()]
        pd.DataFrame(env_data).to_csv(os.path.join(self.out_dir, 'environment_summary.csv'), index=False)

        # Dataset Summary Table
        ds_data = [
            {"Property": "Total Samples Evaluated", "Value": len(preds_df)},
            {"Property": "Total Original Samples", "Value": len(preds_df[preds_df['ground_truth_label'] == 0])},
            {"Property": "Total Tampered Samples", "Value": len(preds_df[preds_df['ground_truth_label'] == 1])},
        ]
        pd.DataFrame(ds_data).to_csv(os.path.join(self.out_dir, 'dataset_summary.csv'), index=False)
        
        # Evaluation Metrics Table
        eval_metrics = [{"Metric": k, "Value": v} for k, v in eval_json["metrics"].items() if k != "confusion_matrix_raw"]
        pd.DataFrame(eval_metrics).to_csv(os.path.join(self.out_dir, 'evaluation_metrics.csv'), index=False)
