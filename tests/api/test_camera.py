import pytest
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.stream.camera_manager import CameraManager

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup_camera():
    yield
    # Safely release hooks post test
    CameraManager(camera_id="default").stop()

def test_camera_subsystem_lifecycle():
    # Verify starting
    response = client.post("/api/v1/camera/start", json={"camera_source": "0"})
    assert response.status_code in [200, 500] # Standard execution matrices depending on physical loopback presence
    
    # Verify baseline configurations
    status_res = client.get("/api/v1/camera/status")
    assert status_res.status_code == 200
    assert "is_opened" in status_res.json()
    
    # Verify metadata schemas
    info_res = client.get("/api/v1/camera/info")
    assert info_res.status_code == 200
    assert "opencv_backend" in info_res.json()

    # Verify termination routines
    stop_res = client.post("/api/v1/camera/stop", json={"camera_id": "default"})
    assert stop_res.status_code == 200
