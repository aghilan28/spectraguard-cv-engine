import os
import csv
import logging
from typing import List, Dict, Any

logger = logging.getLogger("BenchmarkLoader")

class BenchmarkLoader:
    @staticmethod
    def load_dataset(meta_dir: str) -> List[Dict[str, Any]]:
        gt_path = os.path.join(meta_dir, "ground_truth.csv")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Prerequisite Ground Truth missing: {gt_path}")
            
        dataset_records = []
        with open(gt_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("original_filename") or not row.get("generated_filename"):
                    logger.warning(f"Skipping structurally malformed manifest entry row: {row}")
                    continue
                dataset_records.append(dict(row))
        
        logger.info(f"Successfully verified and ingested {len(dataset_records)} target benchmark rows.")
        return dataset_records
