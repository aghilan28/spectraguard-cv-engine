import pytest
from datetime import datetime, timezone
from backend.services.prediction_buffer import PredictionBuffer
from backend.inference.result import InferenceResult
from backend.inference.model_loader import ModelLoader

def test_prediction_buffer():
    buf = PredictionBuffer(max_size=3)
    res_base = {
        "probability": 0.9, "prediction": 1, "confidence": 0.4,
        "threshold": 0.5, "latency_ms": 10.0, "feature_vector": {}, "camera_id": "test"
    }
    
    for i in range(5):
        buf.push(InferenceResult(timestamp=datetime.now(timezone.utc), **res_base))
        
    assert len(buf.history()) == 3
    assert buf.statistics().total_inferences == 3
    assert buf.statistics().tamper_count == 3
    
def test_model_loader_missing_artifacts():
    loader = ModelLoader()
    # Force test against a dummy path to trigger strict FileNotFoundError validation
    with pytest.raises(FileNotFoundError):
        loader.load(artifacts_dir="storage/invalid_path_mock")
