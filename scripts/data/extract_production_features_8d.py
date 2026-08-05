import os
import sys
import json
import time
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.preprocessing import PreprocessingPipeline, FeatureVector

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "datasets", "virat")
META_DIR = os.path.join(DATA_DIR, "metadata")
GROUND_TRUTH_CSV = os.path.join(META_DIR, "ground_truth.csv")
OUTPUT_CSV = os.path.join(META_DIR, "production_features_8d.csv")

def build_video_index():
    print(f"Indexing all .mp4 files under {DATA_DIR}...")
    index = {}
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".mp4"):
                index[f] = os.path.join(root, f)
    print(f"Indexed {len(index)} video files on disk.")
    return index

def extract_features_from_video(video_path: str, max_frames: int = 15) -> FeatureVector:
    pipeline = PreprocessingPipeline()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 300  # Fallback if frame count cannot be read

    # ROOT-CAUSE FIX (forensic audit, Aug 2026), Phase C window-consistency
    # requirement: live_camera_demo.py's `rolling_window` is a buffer of the
    # last `window_size` (15) CONSECUTIVE real-time camera frames -- i.e. a
    # tight ~0.5s window at 30fps. This function previously sampled 15
    # frames spread with `step = total_frames // 15` across the ENTIRE clip
    # (often gaps of 10-20+ frames apart), which is a materially different
    # temporal window and drives temporal_difference (and, via CLAHE/FFT
    # sensitivity to whichever frame lands last, the other 7 features) to a
    # different distribution than what live inference ever sees. We now
    # sample `max_frames` CONSECUTIVE frames -- matching the live rolling
    # window's semantics -- starting from the clip midpoint (where staged
    # tamper events in this dataset are centered) so training features are
    # drawn from the same kind of window as production inference.
    start_idx = max(0, (total_frames - max_frames) // 2)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from video: {video_path}")

    return pipeline.extract(frames)

def process_video_task(task):
    vid_id = task["video_id"]
    vid_path = task["video_path"]
    is_tampered = task["is_tampered"]
    label = task["label"]
    attack_type = task["attack_type"]

    try:
        f_vec = extract_features_from_video(vid_path, max_frames=15)
        res = {
            "video_id": vid_id,
            "label": label,
            "is_tampered": is_tampered,
            "attack_type": attack_type,
            # Provenance marker: proves these 8 values came from a real
            # video run through the actual PreprocessingPipeline, not from
            # scripts/data/generate_synthetic_production_features.py.
            # run_production_training_v2.py refuses to train unless every
            # row carries this exact marker. See FORENSIC_AUDIT.md.
            "extraction_source": "real_pipeline_v1",
        }
        res.update(f_vec.to_dict())
        return res
    except Exception as e:
        print(f"Error processing {vid_id}: {e}")
        return None

def main():
    print("=== M0.3B: EXTRACTING 8D MASTER FEATURE DATASET ===")
    t0 = time.time()

    video_index = build_video_index()
    df_gt = pd.read_csv(GROUND_TRUTH_CSV)
    
    tasks = []
    processed_ids = set()

    for _, row in df_gt.iterrows():
        orig_id = str(row["original_filename"])
        tamp_id = str(row["generated_filename"])
        attack = str(row["attack_category"])

        if orig_id not in processed_ids and orig_id in video_index:
            tasks.append({
                "video_id": orig_id,
                "video_path": video_index[orig_id],
                "label": 0,
                "is_tampered": 0,
                "attack_type": "none"
            })
            processed_ids.add(orig_id)

        if tamp_id not in processed_ids and tamp_id in video_index:
            tasks.append({
                "video_id": tamp_id,
                "video_path": video_index[tamp_id],
                "label": 1,
                "is_tampered": 1,
                "attack_type": attack
            })
            processed_ids.add(tamp_id)

    print(f"Prepared {len(tasks)} valid video extraction tasks out of 658 target videos. Processing...")

    results = []
    total_tasks = len(tasks)
    workers = min(os.cpu_count(), 8)
    print(f" -> Processing in parallel using {workers} worker processes...")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_video_task, t): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
            completed += 1
            if completed % 50 == 0 or completed == total_tasks:
                print(f"Processed {completed}/{total_tasks} videos...")

    df_out = pd.DataFrame(results)
    
    meta_cols = ["video_id", "label", "is_tampered", "attack_type", "extraction_source"]
    feat_cols = FeatureVector.feature_names()
    all_cols = meta_cols + feat_cols
    df_out = df_out[all_cols]

    df_out.to_csv(OUTPUT_CSV, index=False)

    t_elapsed = time.time() - t0
    print(f"Successfully extracted 8D features for {len(df_out)} videos in {t_elapsed:.2f}s!")
    print(f"Saved master dataset to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
