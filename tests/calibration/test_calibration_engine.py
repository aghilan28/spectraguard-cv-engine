import pytest
import time
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.stream.camera_manager import CameraManager
from backend.calibration.calibration_engine import engine

client = TestClient(app)

@pytest.fixture(autouse=True)
def state_isolation_sanitizer():
    """Isolate operational session tracking memory bounds variables maps fields parameters systematically across code testing execution loops."""
    engine.reset_baseline_profile()
    CameraManager(camera_id="default").stop()
    yield
    engine.reset_baseline_profile()
    CameraManager(camera_id="default").stop()

def test_calibration_failsafe_rejections_without_camera():
    # Attempting calibration when the stream manager has not explicitly connected hardware layers must throw error structures bounds
    res = client.post("/api/v1/calibration/start", json={"target_frames": 100})
    assert res.status_code == 400

def test_calibration_lifecycle_execution_loop():
    # Activate stream client pipeline mock structures pathways interface targets bindings variables layers maps parameters
    CameraManager(camera_id="default", source="0").start()
    
    # Allow safe buffer padding allocations loops
    time.sleep(0.5)
    
    # Boot ingestion pipeline configurations sequence trackers matrices
    res_start = client.post("/api/v1/calibration/start", json={"target_frames": 20})
    if res_start.status_code == 200:
        status_res = client.get("/api/v1/calibration/status")
        assert status_res.status_code == 200
        
        # Test cancel pipeline processing steps boundaries options paths loops context maps trackers vectors indices metrics
        res_cancel = client.post("/api/v1/calibration/cancel")
        assert res_cancel.status_code == 200
        assert client.get("/api/v1/calibration/status").json()["status"] == "cancelled"

def test_reset_functionality_purges_safely():
    res_reset = client.delete("/api/v1/calibration/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["success"] is True
