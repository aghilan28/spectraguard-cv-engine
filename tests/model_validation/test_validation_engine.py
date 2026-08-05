import os
import json
import pytest
import joblib
import numpy as np
from unittest.mock import patch, MagicMock
from backend.services.model_validation_service import ModelValidationService

# Define plain Python classes that can be pickled by joblib safely
class DummyProductionModel:
    def __init__(self):
        self.classes_ = np.array(["normal", "blur"])
    def predict(self, X):
        return np.array(["normal"])
    def predict_proba(self, X):
        return np.array([[0.85, 0.15]])

class DummyStandardScaler:
    def transform(self, X):
        return X

@pytest.fixture
def mock_artifacts(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    model_path = model_dir / "production_model.joblib"
    scaler_path = model_dir / "StandardScaler.joblib"
    meta_path = model_dir / "feature_metadata.json"
    threshold_path = model_dir / "threshold.json"
    
    # Serialize real picklable objects instead of dynamic MagicMocks
    joblib.dump(DummyProductionModel(), model_path)
    joblib.dump(DummyStandardScaler(), scaler_path)
    
    meta_data = {"feature_order": ["feat1", "feat2"]}
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f)
        
    threshold_data = {"bounds": 0.5}
    with open(threshold_path, 'w') as f:
        json.dump(threshold_data, f)
        
    return str(model_path), str(scaler_path), str(meta_path), str(threshold_path)

@patch('glob.glob')
def test_artifact_discovery_and_loading(mock_glob, mock_artifacts):
    mock_glob.side_effect = [[mock_artifacts[0]], [mock_artifacts[1]], [mock_artifacts[2]], [mock_artifacts[3]]]
    
    service = ModelValidationService()
    service.load_artifacts()
    
    assert service.model is not None
    assert service.feature_metadata["feature_order"] == ["feat1", "feat2"]

@patch('os.walk')
def test_dataset_recursive_discovery(mock_walk):
    mock_walk.return_value = [
        (r"C:\Users\AKILA\Downloads\TAMPERING DATASET", [], []),
        (r"C:\Users\AKILA\Downloads\TAMPERING DATASET\Normal", [], ["cam1.mp4"]),
        (r"C:\Users\AKILA\Downloads\TAMPERING DATASET\Blur", [], ["cam2.avi"])
    ]
    
    service = ModelValidationService()
    dataset = service.discover_dataset()
    
    assert len(dataset) == 2
    assert dataset[0]["label"] == "Normal"
    assert dataset[1]["filename"] == "cam2.avi"
