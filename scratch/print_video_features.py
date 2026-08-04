import os
import sys
import cv2
import numpy as np

# Add project src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from spectraguard_cv_engine.features.unified.pipeline import UnifiedExtractionPipeline
from spectraguard_cv_engine.ml.data.loader import EXPECTED_UNIFIED_FEATURES

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 3:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (1920, 1080))
        frames.append(resized)
    cap.release()
    
    if len(frames) < 3:
        print(f"Error: Not enough frames in {video_path}")
        return None
        
    vec = UnifiedExtractionPipeline.extract_from_sequence(frames, "test", 0)
    arr = vec.to_array()
    return dict(zip(EXPECTED_UNIFIED_FEATURES, arr))

def main():
    clear_path = "../spectraguard-core-infra/data/uploads/TEST VIDEO.mp4"
    blur_path = "../spectraguard-core-infra/data/uploads/TEST VIDEO EXTREME BLUR.mp4"
    
    print("CLEAR VIDEO:")
    clear_feat = extract_features(clear_path)
    if clear_feat:
        for k, v in clear_feat.items():
            print(f"  {k}: {v}")
            
    print("\nBLURRY VIDEO:")
    blur_feat = extract_features(blur_path)
    if blur_feat:
        for k, v in blur_feat.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
