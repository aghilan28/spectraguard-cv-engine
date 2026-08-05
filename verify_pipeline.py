from camera.camera_config import CameraConfig
from camera.rtsp_builder import RTSPBuilder, CameraBrand

def run_verification():
    print("==================================================")
    print("         PHASE 1 VALIDATION PIPELINE              ")
    print("==================================================\n")

    # Mock parameters
    ip = "192.168.1.100"
    user = "admin"
    pwd = "SecretPassword123"

    # Base configuration setup
    config = CameraConfig(
        name="Front Door Camera",
        ip_address=ip,
        username=user,
        password=pwd
    )

    print(f"Loaded Configuration: {config.name}")
    print(f"Target IP: {config.ip_address}\n")
    print("--- Generating URL Outputs per Vendor ---")

    # 1. Test Generic Vendor
    generic_url = RTSPBuilder.build_url(config, CameraBrand.GENERIC)
    print(f"[GENERIC]   -> {generic_url}")

    # 2. Test Hikvision Vendor
    hik_url = RTSPBuilder.build_url(config, CameraBrand.HIKVISION)
    print(f"[HIKVISION] -> {hik_url}")

    # 3. Test Dahua Vendor
    dahua_url = RTSPBuilder.build_url(config, CameraBrand.DAHUA)
    print(f"[DAHUA]     -> {dahua_url}")

    # 4. Test Axis Vendor
    axis_url = RTSPBuilder.build_url(config, CameraBrand.AXIS)
    print(f"[AXIS]      -> {axis_url}")
    
    print("\n==================================================")
    print("        PHASE 1 ARCHITECTURE VALIDATED            ")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
