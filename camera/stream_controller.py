import time
import cv2
from typing import Generator, Optional
from camera.camera_manager import CameraManager

class StreamController:
    """
    Consumes frames from a CameraManager instance and encodes them into 
    standard stream formats suitable for UI consumption or network transmission.
    """
    def __init__(self, camera_manager: CameraManager):
        self.manager = camera_manager

    def generate_mjpeg_stream(self, quality: int = 80) -> Generator[bytes, None, None]:
        """
        Generates an MJPEG byte stream wrapper string payload.
        Ideal for updating UI surfaces or basic HTTP video streaming layers.
        """
        while self.manager.is_connected():
            frame = self.manager.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Encode frame to JPEG binary format
            ret, encoded_jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ret:
                continue

            # Yield data matching standard multipart boundary formatting rules
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_jpeg.tobytes() + b'\r\n')
            
            # Match the pacing to target frame capture rates (~30 FPS max check frequency)
            time.sleep(0.033)
