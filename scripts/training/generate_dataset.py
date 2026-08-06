import os
import csv
import glob
import cv2
import sys
import numpy as np
from typing import List, Tuple

# Ensure root path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from training.feature_extractor import FeatureExtractor
except ImportError:
    from backend.services.feature_extractor import FeatureExtractor

FEATURES = ["fft_low_ratio", "fft_mid_ratio", "fft_high_ratio", "log_total_energy", 
            "laplacian_variance", "edge_density", "shannon_entropy", "temporal_difference"]

def print_progress(current: int, total: int, prefix: str = '', suffix: str = '', bar_length: int = 40):
    percent = f"{100 * (current / float(total)):.1f}"
    filled_length = int(round(bar_length * current / float(total)))
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if current == total:
        print()

def process_video(video_path: str, label: int, target_fps: float = 2.0, max_frames: int = 15) -> Tuple[str, List[List[float]]]:
    """
    Extracts frames from a video, computes the 8D feature vector for each,
    and returns features. Skips duplicates and handles corruption safely.
    """
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nWarning: Could not open video file: {video_path}")
        return video_path, []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    
    # Calculate skip interval based on target_fps
    interval = max(1, int(round(fps / target_fps)))
    records = []
    
    from training_v2.augmentation.augmentation import ImageAugmentor
    augmentor = ImageAugmentor()
    prev_small = None
    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame is None or frame.size == 0:
                continue # ignore corrupted frames
            
            if frame_count % interval != 0:
                continue
                
            # Skip duplicate frames by checking downsampled grayscale difference
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64))
            if prev_small is not None:
                diff = float(np.mean(np.abs(small.astype(np.float32) - prev_small.astype(np.float32))))
                if diff < 0.8: # very low difference means duplicate frame
                    continue
            
            prev_small = small
            
            # Extract features for original frame
            feats = extractor.extract(frame)
            if not feats:
                continue
                
            row = [feats.get(f, 0.0) for f in FEATURES] + [label]
            records.append(row)
            extracted_count += 1
            
            # If this is a normal/background video, synthesize matched tamper variants
            if label == 0:
                # 1. Simulate Hand Cover (glove/skin)
                hand_img = augmentor.simulate_hand_cover(frame)
                hand_feats = extractor.extract(hand_img)
                if hand_feats:
                    records.append([hand_feats.get(f, 0.0) for f in FEATURES] + [1])
                    
                # 2. Simulate Lens Cover (paper/blackout/fabric)
                lens_img = augmentor.simulate_paper_cover(frame)
                lens_feats = extractor.extract(lens_img)
                if lens_feats:
                    records.append([lens_feats.get(f, 0.0) for f in FEATURES] + [1])
                    
                # 3. Simulate Camera Moved
                moved_img = augmentor.simulate_camera_moved(frame)
                moved_feats = extractor.extract(moved_img)
                if moved_feats:
                    records.append([moved_feats.get(f, 0.0) for f in FEATURES] + [1])
            
            if extracted_count >= max_frames:
                break
            
        except Exception as e:
            # Safely catch CV errors or index errors to ignore corrupted frames
            continue
            
    cap.release()
    return video_path, records

def generate_dataset(dataset_path: str, output_csv: str, target_fps: float = 2.0):
    print("=========================================================")
    print("Processing specific targeted tampering and normal videos...")
    
    labeled_videos = []
    
    # Exact paths of target tampering videos
    tamper_candidates = [
        r"C:\Users\AKILA\Downloads\TAMPERING DATASET\TAMPER 1 - PAPER COVER.mp4",
        r"C:\Users\AKILA\Downloads\TEST VIDEO COVERED HAND.mp4",
        r"C:\Users\AKILA\Downloads\TEST VIDEO HALF COVERED.mp4"
    ]
    for path in tamper_candidates:
        if os.path.exists(path):
            labeled_videos.append((path, 1))
            
    # Inject exactly 3 background/normal videos
    virat_dir = os.path.join(PROJECT_ROOT, "data", "datasets", "virat", "videos_original")
    normal_candidates = [
        "VIRAT_S_000200_05_001525_001575.mp4",
        "VIRAT_S_010115_01_000399_000467.mp4",
        "VIRAT_S_010005_02_000177_000203.mp4"
    ]
    for filename in normal_candidates:
        path = os.path.join(virat_dir, filename)
        if os.path.exists(path):
            labeled_videos.append((path, 0))
            
    print(f"Target dataset compiled: {len(labeled_videos)} videos")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    total_records = []
    print("\nStarting Frame Extraction and Feature Generation:")
    
    for count, (v_path, label) in enumerate(labeled_videos, 1):
        lbl_str = "TAMPER" if label == 1 else "NORMAL"
        v_name = os.path.basename(v_path)
        print_progress(count - 1, len(labeled_videos), prefix='Processing', suffix=f'Scanning {v_name} ({lbl_str})')
        
        _, records = process_video(v_path, label, target_fps=target_fps)
        total_records.extend(records)
        
        print_progress(count, len(labeled_videos), prefix='Processing', suffix=f'Processed {v_name} | Frames: {len(records)}')

    # Write training dataset CSV with extraction provenance stamp
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FEATURES + ["label", "extraction_source"])
        for r in total_records:
            writer.writerow(r + ["real_pipeline_v1"])

    print(f"\nDataset generation complete. Saved {len(total_records)} samples to {output_csv}")

if __name__ == "__main__":
    generate_dataset(r"C:\Users\AKILA\Downloads", "data/training_dataset.csv", target_fps=2.0)
