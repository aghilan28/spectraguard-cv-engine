import json
from datetime import datetime
from typing import Dict, Any

class ManifestGenerator:
    @staticmethod
    def create(total_videos: int) -> Dict[str, Any]:
        return {
            "dataset_name": "VIRAT_Surveillance_Benchmark_Core",
            "generation_timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset_version": "1.0.0",
            "supported_formats": [".mp4", ".avi", ".mov", ".mkv"],
            "total_videos": total_videos,
            "inventory_file_reference": "data/datasets/virat/metadata/inventory.csv"
        }
