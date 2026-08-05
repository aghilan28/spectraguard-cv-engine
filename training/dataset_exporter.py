import pandas as pd
import json
from pathlib import Path
from typing import Dict

class DatasetExporter:
    @staticmethod
    def export(df: pd.DataFrame, report: Dict, csv_path: Path, json_path: Path):
        df.to_csv(csv_path, index=False)
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"[EXPORT] Dataset saved to {csv_path}")
        print(f"[EXPORT] Report saved to {json_path}")
