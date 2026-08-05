from pathlib import Path
from typing import Iterator, Dict
import cv2

class VideoLoader:
    def __init__(self, config):
        self.config = config

    def scan_directory(self) -> Iterator[Dict]:
        """Recursively scans for supported video files."""
        if not self.config.dataset_dir.exists():
            raise FileNotFoundError(f"Directory {self.config.dataset_dir} does not exist.")
            
        for ext in self.config.supported_extensions:
            for filepath in self.config.dataset_dir.rglob(f"*{ext}"):
                cap = cv2.VideoCapture(str(filepath))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    yield {
                        "path": str(filepath),
                        "filename": filepath.name,
                        "fps": fps,
                        "duration": frame_count / fps if fps > 0 else 0,
                        "resolution": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    }
                cap.release()
