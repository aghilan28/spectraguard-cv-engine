import os
import json
import shutil

class ModelExporter:
    def __init__(self, comparison_report_path, candidate_dir, promotable_dir):
        self.comparison_report_path = comparison_report_path
        self.candidate_dir = candidate_dir
        self.promotable_dir = promotable_dir

    def run(self):
        print("[ModelExporter] Checking promotion criteria...")
        if not os.path.exists(self.comparison_report_path):
            print(f"[ModelExporter] Comparison report not found: {self.comparison_report_path}")
            return False

        with open(self.comparison_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        prod = report["production"]
        cand = report["candidate"]

        # Promotion Conditions
        better_recall = cand["recall"] >= prod["recall"]
        better_fpr = cand["fpr"] <= prod["fpr"]
        better_f1 = cand["f1"] >= prod["f1"]
        better_accuracy = cand["accuracy"] >= prod["accuracy"]

        is_promotable = better_recall and better_fpr and better_f1 and better_accuracy

        print(f"[ModelExporter] Metrics Comparison:")
        print(f"  Recall: Candidate={cand['recall']:.4f}, Production={prod['recall']:.4f} (Better/Equal: {better_recall})")
        print(f"  FPR: Candidate={cand['fpr']:.4f}, Production={prod['fpr']:.4f} (Better/Equal: {better_fpr})")
        print(f"  F1: Candidate={cand['f1']:.4f}, Production={prod['f1']:.4f} (Better/Equal: {better_f1})")
        print(f"  Accuracy: Candidate={cand['accuracy']:.4f}, Production={prod['accuracy']:.4f} (Better/Equal: {better_accuracy})")

        if is_promotable:
            print("[ModelExporter] PROMOTION CRITERIA MET. Exporting candidate to promotable directory...")
            os.makedirs(self.promotable_dir, exist_ok=True)
            
            files_to_copy = ["production_model.joblib", "feature_scaler.joblib", "threshold.json", "feature_metadata.json"]
            for file in files_to_copy:
                src = os.path.join(self.candidate_dir, file)
                dst = os.path.join(self.promotable_dir, file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            print(f"[ModelExporter] Model successfully exported to {self.promotable_dir}")
            return True
        else:
            print("[ModelExporter] PROMOTION CRITERIA NOT MET. Candidate model rejected.")
            return False
