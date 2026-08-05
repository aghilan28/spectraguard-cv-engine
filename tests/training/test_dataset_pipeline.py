import os
import sys
from pathlib import Path

# Force the parent root workspace path directly into sys.path to eliminate ModuleNotFoundError on Windows
root_path = str(Path(__file__).resolve().parents[2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import pytest
import pandas as pd
from training.label_encoder import LabelEncoder
from training.dataset_validator import DatasetValidator

def test_label_encoder():
    assert LabelEncoder.encode("dataset/normal/vid1.mp4") == 0
    assert LabelEncoder.encode("dataset/tamper/blur/vid2.mp4") == 1

def test_dataset_validator():
    data = {
        "video_name": ["v1", "v1", "v2"],
        "frame_number": [0, 0, 15],  # Duplicate frame
        "label": [0, 0, 1],
        "fft_low_ratio": [0.5, 0.5, float('inf')] # Inf value
    }
    df = pd.DataFrame(data)
    clean_df = DatasetValidator.validate(df)
    
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["video_name"] == "v1"
