import pytest
import time
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.realtime.realtime_engine import realtime_engine

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_engine():
    realtime_engine.stop()
    realtime_engine.history.clear()
    yield
    realtime_engine.stop()

def test_engine_start_stop():
    assert not realtime_engine.is_running()
    res = client.post("/api/v1/realtime/start")
    assert res.status_code == 200
    assert realtime_engine.is_running()

    res_stop = client.post("/api/v1/realtime/stop")
    assert res_stop.status_code == 200
    assert not realtime_engine.is_running()

def test_duplicate_start_protection():
    client.post("/api/v1/realtime/start")
    res2 = client.post("/api/v1/realtime/start")
    assert res2.status_code == 409

def test_pause_resume():
    client.post("/api/v1/realtime/start")
    
    res = client.post("/api/v1/realtime/pause")
    assert res.status_code == 200
    with realtime_engine.state._lock:
        assert realtime_engine.state.paused
        
    res2 = client.post("/api/v1/realtime/resume")
    assert res2.status_code == 200
    with realtime_engine.state._lock:
        assert not realtime_engine.state.paused

@patch("backend.realtime.realtime_engine.CameraManager")
@patch("backend.realtime.realtime_engine.inference_engine")
@patch("backend.realtime.realtime_engine.deviation_engine")
@patch("backend.realtime.realtime_engine.tamper_engine")
def test_pipeline_execution_and_statistics(mock_tamper, mock_dev, mock_inf, mock_cam):
    # Setup mocks to bypass OpenCV and deep math for deterministic testing
    mock_cam_instance = MagicMock()
    mock_cam_instance.is_running.return_value = True
    mock_cam_instance.buffer.frames.return_value = [None] * 15
    mock_cam.return_value = mock_cam_instance
    
    mock_tamper_res = MagicMock()
    mock_tamper_res.random_forest_prediction = 1
    mock_tamper_res.random_forest_probability = 0.95
    mock_tamper_res.tamper_type = "LENS_COVER"
    mock_tamper_res.deviation_score = 0.8
    mock_tamper.evaluate.return_value = mock_tamper_res

    # Start loop
    realtime_engine.start()
    time.sleep(1.0) # Allow worker to process a few 500ms cycles
    realtime_engine.stop()

    # Verify State
    history = client.get("/api/v1/realtime/history").json()
    assert len(history) > 0
    assert history[-1]["tamper_type"] == "LENS_COVER"

    # Verify Stats
    stats = client.get("/api/v1/realtime/statistics").json()
    assert stats["total_processed"] > 0
    assert stats["tamper_counts"] > 0
    assert stats["average_probability"] == 0.95

    # Verify Latest
    latest = client.get("/api/v1/realtime/latest").json()
    assert latest["prediction"] == "TAMPER"

def test_clean_history():
    realtime_engine.history.add({"test": "data"})
    res = client.delete("/api/v1/realtime/history")
    assert res.status_code == 200
    assert len(realtime_engine.history.get_all()) == 0
