import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from backend.services.model_adaptation_service import ModelAdaptationService

@pytest.fixture
def service_instance():
    return ModelAdaptationService()

def test_dataset_discovery_parsing(service_instance):
    with patch('os.walk') as mock_walk:
        mock_walk.return_value = [
            (r"C:\Users\AKILA\Downloads\TAMPERING DATASET", [], []),
            (r"C:\Users\AKILA\Downloads\TAMPERING DATASET\Normal", [], ["cam1.mp4"]),
            (r"C:\Users\AKILA\Downloads\TAMPERING DATASET\Flash", [], ["cam2.avi"])
        ]
        dataset = service_instance.discover_dataset()
        assert len(dataset) == 2
        # Fixed: Verifying class_name maps to folder and binary label maps to 1 for tampering folders
        assert any(item["class_name"] == "Flash" and item["label"] == 1 for item in dataset)
        assert any(item["class_name"] == "Normal" and item["label"] == 0 for item in dataset)

def test_training_pipeline_data_generation(service_instance, tmp_path):
    report_path = tmp_path / "reports"
    report_path.mkdir()
    service_instance.report_dir = str(report_path)
    
    dummy_records = [{"Video": "v1.mp4", "Frame": 1, "Label": 0, "feat1": 0.1, "feat2": 0.2, "feat3": 0.3, "feat4": 0.4, "feat5": 0.5, "feat6": 0.6, "feat7": 0.7, "feat8": 0.8}]
    df = pd.DataFrame(dummy_records)
    df.to_csv(os.path.join(service_instance.report_dir, "training_dataset.csv"), index=False)
    
    assert os.path.exists(os.path.join(service_instance.report_dir, "training_dataset.csv"))
