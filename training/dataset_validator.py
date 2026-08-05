import pandas as pd
import numpy as np

class DatasetValidator:
    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """Ensures dataset integrity before export."""
        initial_len = len(df)
        
        # Check for missing and infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Check for duplicate frame processing
        df = df.drop_duplicates(subset=['video_name', 'frame_number'])
        
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"[VALIDATION] Dropped {dropped} invalid/duplicate rows.")
            
        return df
