import pytest
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.calibration.calibration_engine import engine
from backend.stream.camera_manager import CameraManager

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_system_state():
    """Ensure baseline states isolation parameters scopes vectors across continuous integration unit testing boundaries tracking."""
    engine.reset_baseline_profile()
    CameraManager(camera_id="default").stop()
    yield
    engine.reset_baseline_profile()
    CameraManager(camera_id="default").stop()

def test_calibration_api_missing_baseline_error_boundaries():
    # Fetch requests before calibration profiles exist must drop back with explicit standard status constraints
    res_base = client.get("/api/v1/calibration/baseline")
    assert res_base.status_code == 404
    
    res_reload = client.post("/api/v1/calibration/reload")
    assert res_reload.status_code == 404
    
    res_features = client.get("/api/v1/calibration/features")
    assert res_features.status_code == 404

def test_calibration_api_diagnostics_progress_and_info_endpoint_contracts():
    res_progress = client.get("/api/v1/calibration/progress")
    assert res_progress.status_code == 200
    assert "percentage" in res_progress.json()
    assert res_progress.json()["running"] is False
    
    res_info = client.get("/api/v1/calibration/info")
    assert res_info.status_code == 200
    assert res_info.json()["baseline_exists"] is False

def test_calibration_api_conflict_double_start_protections():
    # Activate streaming device layer dependencies mapping references trackers framework options
    CameraManager(camera_id="default", source="0").start()
    
    res_first = client.post("/api/v1/calibration/start", json={"target_frames": 100})
    if res_first.status_code == 200:
        # Enforce parallel structural race block validation rejection
        res_second = client.post("/api/v1/calibration/start", json={"target_frames": 200})
        assert res_second.status_code == 409
        
    client.delete("/api/v1/calibration/reset")

def test_openapi_schema_generation_conformance():
    openapi_res = client.get("/openapi.json")
    assert openapi_res.status_code == 200
    paths = openapi_res.json()["paths"]
    assert "/api/v1/calibration/progress" in paths
    assert "/api/v1/calibration/reload" in paths
    assert "/api/v1/calibration/info" in paths
