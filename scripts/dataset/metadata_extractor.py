import os
import cv2
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class VideoMetadata:
    filename: str
    absolute_path: str
    file_size: int
    duration: float
    frame_count: int
    fps: float
    width: int
    height: int
    aspect_ratio: str
    codec: str
    creation_timestamp: str
    modification_timestamp: str

class MetadataExtractor:
    @staticmethod
    def extract(file_path: str) -> Optional[VideoMetadata]:
        if not os.path.exists(file_path):
            return None
        
        abs_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        stat = os.stat(file_path)
        mod_time = datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z"
        # Fallback to mtime if ctime represents metadata change rather than creation on certain filesystems
        try:
            cre_time = datetime.fromtimestamp(stat.st_birthtime).isoformat() + "Z"
        except AttributeError:
            cre_time = datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z"

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            cap.release()
            return None
            
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Decode fourcc bytes into readable string
        fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)]).strip()
        
        cap.release()
        
        duration = (frame_count / fps) if fps > 0 else 0.0
        
        if width > 0 and height > 0:
            def gcd(a: int, b: int) -> int:
                while b:
                    a, b = b, a % b
                return a
            divisor = gcd(width, height)
            aspect_ratio = f"{width // divisor}:{height // divisor}"
        else:
            aspect_ratio = "0:0"
            
        return VideoMetadata(
            filename=filename,
            absolute_path=abs_path,
            file_size=file_size,
            duration=round(duration, 4),
            frame_count=frame_count,
            fps=round(fps, 4),
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            codec=codec if codec else "UNKNOWN",
            creation_timestamp=cre_time,
            modification_timestamp=mod_time
        )
