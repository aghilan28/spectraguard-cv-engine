import cv2
import time
import numpy as np
from typing import Tuple, Optional

def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    add_timestamp: bool = True
) -> np.ndarray:
    if frame is None:
        raise ValueError("Cannot process an empty frame matrix.")
        
    out_frame = frame.copy()
    
    if target_size:
        out_frame = cv2.resize(out_frame, target_size, interpolation=cv2.INTER_LINEAR)
        
    if add_timestamp:
        ts_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(
            out_frame,
            ts_text,
            (10, out_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
        
    return out_frame

def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    if frame is None:
        raise ValueError("Cannot encode empty frame.")
    success, encoded_img = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError("JPEG conversion framework error.")
    return encoded_img.tobytes()
