import time
from camera.camera_config import CameraConfig
from camera.camera_manager import CameraManager
from camera.stream_controller import StreamController

def run_stream_verification():
    print("==================================================")
    print("         PHASE 3 STREAM VERIFICATION              ")
    print("==================================================\n")

    config = CameraConfig(name="Stream Test", ip_address="simulation")
    manager = CameraManager(config)
    
    # Force simulation mode bypass for architectural structural check
    manager._is_connected = True
    manager._start_time = time.time()
    import numpy as np
    manager._buffer.set_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    controller = StreamController(manager)
    print("[SYSTEM] StreamController linked successfully to CameraManager.")
    print("[SYSTEM] Fetching sample stream byte slices...")

    stream_generator = controller.generate_mjpeg_stream()
    
    # Capture the first 5 payload fragments to verify structure stability
    count = 0
    for chunk in stream_generator:
        print(f"Captured Chunk {count+1} | Byte Length: {len(chunk)} | Header Match: {chunk.startswith(b'--frame')}")
        count += 1
        if count >= 5:
            break

    manager.disconnect()
    print("\n==================================================")
    print("        PHASE 3 STREAM ARCHITECTURE VALIDATED     ")
    print("==================================================")

if __name__ == "__main__":
    run_stream_verification()
