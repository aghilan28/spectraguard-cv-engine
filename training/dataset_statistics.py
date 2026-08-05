import pandas as pd
from typing import Dict

class DatasetStatistics:
    @staticmethod
    def generate_report(df: pd.DataFrame) -> Dict:
        return {
            "total_frames": len(df),
            "total_videos": df['video_name'].nunique(),
            "class_balance": {
                "normal_samples": int((df['label'] == 0).sum()),
                "tampered_samples": int((df['label'] == 1).sum())
            },
            "feature_means": df.drop(columns=['video_name', 'frame_number', 'label']).mean().to_dict()
        }
