import time
import cv2
from typing import Tuple
from camera.camera_manager import CameraManager

class LiveViewer:
    """
    Consumes frames from a CameraManager instance and displays them inside
    a highly performant OpenCV window embedded with real-time operational metrics.
    """
    def __init__(self, manager: CameraManager, window_title: str = "SpectraGuard CCTV Viewer"):
        self.manager = manager
        self.window_title = window_title
        self._is_running = False

    def start(self) -> None:
        """Launches the main window context loop and captures user window-interaction commands."""
        if not self.manager.is_connected():
            print("[VIEWER ERROR] Cannot start viewer: CameraManager is not connected.")
            return

        self._is_running = True
        
        # Configure native resizable window properties
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        print(f"[VIEWER] Display window '{self.window_title}' opened.")
        print("[VIEWER] Controls: Press 'Q' or 'ESC' inside the window frame to exit.")

        try:
            while self._is_running:
                # Retrieve the latest atomic frame array
                frame = self.manager.get_latest_frame()
                
                if frame is not None:
                    # Create a writeable copy to overlay metrics without altering raw source data
                    display_frame = frame.copy()
                    
                    # Apply real-time telemetry HUD overlay
                    self._draw_hud(display_frame)
                    
                    # Push buffer out to the OpenCV native graphics engine window
                    cv2.imshow(self.window_title, display_frame)
                
                # Check keyboard bitmask for user terminations
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27): # 'q', 'Q', or Escape key
                    print("[VIEWER] Termination key caught by window interface thread loop.")
                    self._is_running = False
                    break
                    
                # Graceful termination hook if the background window is closed by standard OS 'X' widget
                if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                    print("[VIEWER] Operating System window close action detected.")
                    self._is_running = False
                    break
                
                # CRITICAL PERFORMANCE FIX: Prevent CPU thrashing by yielding thread execution.
                # Paces the loop cleanly around ~30 iterations per second to keep OS window smooth.
                time.sleep(0.03)
        finally:
            self._cleanup()

    def _draw_hud(self, frame) -> None:
        """Applies high-visibility telemetry info directly into frame arrays."""
        h, w = frame.shape[:2]
        
        # Gather metrics from configuration and runtime manager parameters
        cam_name = f"CAM: {self.manager.config.name}"
        fps_stat = f"FPS: {self.manager.get_fps():.2f}"
        frame_stat = f"FRAMES: {self.manager.get_frame_count()}"
        res_stat = f"RES: {w}x{h}"
        status_text = "STATUS: ACTIVE" if self.manager.is_connected() else "STATUS: LOSS"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Construct textual composite blocks
        left_telemetry = f"{cam_name} | {status_text} | {res_stat}"
        right_telemetry = f"{fps_stat} | {frame_stat} | {timestamp}"

        # Render background drop-shadow text followed by primary white overlay text for high legibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        shadow_thick = 3
        color_white = (255, 255, 255)
        color_black = (0, 0, 0)
        
        # Top-Left HUD placement
        cv2.putText(frame, left_telemetry, (10, 25), font, scale, color_black, shadow_thick, cv2.LINE_AA)
        cv2.putText(frame, left_telemetry, (10, 25), font, scale, color_white, thickness, cv2.LINE_AA)
        
        # Bottom-Left HUD placement
        cv2.putText(frame, right_telemetry, (10, h - 15), font, scale, color_black, shadow_thick, cv2.LINE_AA)
        cv2.putText(frame, right_telemetry, (10, h - 15), font, scale, color_white, thickness, cv2.LINE_AA)

    def _cleanup(self) -> None:
        """Destroys window layer allocations safely."""
        try:
            cv2.destroyWindow(self.window_title)
        except cv2.error:
            # Catch the Null pointer exception if the OS already destroyed the window via the 'X' button
            pass
        print("[VIEWER] Window context dropped completely.")
