import os
import cv2
import csv
import numpy as np
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

logger = logging.getLogger("QualityAnalyzer")

@dataclass
class QualityMetrics:
    filename: str
    blur_score: float
    sharpness_score: float
    brightness_score: float
    contrast_score: float
    noise_estimation: float
    camera_stability_score: float
    motion_intensity: float
    corrupted_frame_ratio: float
    frame_readability: float
    resolution_validation: int
    normalized_quality_score: float

class QualityAnalyzer:
    def __init__(self, sample_frames: int = 8):
        self.sample_frames = sample_frames

    def analyze_dataset(self, inventory_path: str) -> List[Dict[str, Any]]:
        results = []
        if not os.path.exists(inventory_path):
            return results

        with open(inventory_path, mode="r", encoding="utf-8") as f:
            total_videos = sum(1 for _ in f) - 1
            
        with open(inventory_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx % 10 == 0 or idx == total_videos - 1:
                    logger.info(f"Analyzing video {idx + 1}/{total_videos} [{row.get('filename', 'Unknown')}]")
                    
                metrics = self._analyze_video(row)
                if metrics:
                    results.append(asdict(metrics))
        return results

    def _analyze_video(self, video_meta: Dict[str, str]) -> QualityMetrics:
        filepath = video_meta.get("absolute_path", "")
        filename = video_meta.get("filename", "")
        
        if not os.path.exists(filepath):
            return self._default_metrics(filename, 1.0)
            
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return self._default_metrics(filename, 1.0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return self._default_metrics(filename, 1.0)
        
        sharpness_list = []
        brightness_list = []
        contrast_list = []
        motion_list = []
        
        prev_gray = None
        corrupted = 0
        frames_read = 0
        
        # FAST READ OPTIMIZATION: 
        # Avoid cap.set() entirely. It hangs FFmpeg decoders on compressed MP4s.
        # Instead, read sequential frames at the beginning of the file.
        target_reads = min(self.sample_frames, total_frames)
        
        while frames_read < target_reads:
            ret, frame = cap.read()
            if not ret or frame is None:
                corrupted += 1
                frames_read += 1
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_list.append(laplacian_var)
            
            mean, stddev = cv2.meanStdDev(gray)
            brightness_list.append(mean[0][0])
            contrast_list.append(stddev[0][0])
            
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_list.append(np.mean(diff))
            prev_gray = gray
            
            frames_read += 1

        cap.release()
        
        valid_samples = target_reads - corrupted
        corrupted_ratio = corrupted / target_reads if target_reads > 0 else 1.0
        
        if valid_samples == 0:
            return self._default_metrics(filename, 1.0)

        avg_sharp = float(np.mean(sharpness_list))
        avg_bright = float(np.mean(brightness_list))
        avg_contrast = float(np.mean(contrast_list))
        avg_motion = float(np.mean(motion_list)) if motion_list else 0.0
        
        blur_score = max(0.0, 1.0 - (avg_sharp / 1000.0))
        noise_est = max(0.0, 1.0 - (avg_contrast / 128.0))
        stability = max(0.0, 1.0 - (avg_motion / 50.0))
        readability = 1.0 - corrupted_ratio
        
        width, height = int(video_meta.get("width", 0)), int(video_meta.get("height", 0))
        resolution_valid = 1 if (width >= 640 and height >= 480) else 0

        q_score = (
            min(avg_sharp / 500.0, 1.0) * 0.3 +
            (1.0 - abs(avg_bright - 128) / 128.0) * 0.2 +
            (min(avg_contrast / 64.0, 1.0)) * 0.2 +
            stability * 0.15 +
            readability * 0.15
        ) * resolution_valid

        return QualityMetrics(
            filename=filename,
            blur_score=round(blur_score, 4),
            sharpness_score=round(avg_sharp, 4),
            brightness_score=round(avg_bright, 4),
            contrast_score=round(avg_contrast, 4),
            noise_estimation=round(noise_est, 4),
            camera_stability_score=round(stability, 4),
            motion_intensity=round(avg_motion, 4),
            corrupted_frame_ratio=round(corrupted_ratio, 4),
            frame_readability=round(readability, 4),
            resolution_validation=resolution_valid,
            normalized_quality_score=round(max(0.0, min(1.0, q_score)), 4)
        )

    def _default_metrics(self, filename: str, corruption_ratio: float) -> QualityMetrics:
        return QualityMetrics(
            filename=filename, blur_score=1.0, sharpness_score=0.0, brightness_score=0.0,
            contrast_score=0.0, noise_estimation=1.0, camera_stability_score=0.0,
            motion_intensity=0.0, corrupted_frame_ratio=corruption_ratio,
            frame_readability=0.0, resolution_validation=0, normalized_quality_score=0.0
        )
