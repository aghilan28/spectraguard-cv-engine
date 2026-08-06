import cv2
import numpy as np
import time
import os
import json

from backend.tamper.detectors.contrast_detector import ContrastDetector
from backend.tamper.detectors.saturation_detector import SaturationDetector
from backend.tamper.detectors.sharpness_detector import SharpnessDetector
from backend.tamper.detectors.color_detector import ColorDetector
from backend.tamper.detectors.rotation_detector import RotationDetector

class TamperClassificationEngine:
    def __init__(self):
        self.thresholds = {
            "entropy_limit": 4.2,
            "edge_density_limit": 0.02,
            "laplacian_blur_limit": 35.0,
            "laplacian_defocus_limit": 95.0,
            "black_ratio_limit": 0.85,
            "white_ratio_limit": 0.85
        }
        config_path = os.path.normpath("config/tamper_thresholds.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.thresholds.update(cfg)
            except Exception as e:
                print(f"[ClassificationEngine] Warning: Failed to parse thresholds config: {e}")
        
        # Initialize modular detectors
        self.contrast_detector = ContrastDetector(self.thresholds)
        self.saturation_detector = SaturationDetector(self.thresholds)
        self.sharpness_detector = SharpnessDetector(self.thresholds)
        self.color_detector = ColorDetector(self.thresholds)
        self.rotation_detector = RotationDetector(self.thresholds)

    def compute_hash(self, frame):
        """Computes a simple 64-bit Average Hash for freeze detection."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            avg = np.mean(small)
            return "".join(["1" if p > avg else "0" for p in small.ravel()])
        except Exception:
            return ""

    def evaluate_motion(self, prev_frame, curr_frame, return_homography=False):
        """Uses ORB features and Homography to detect camera movement shift."""
        try:
            if prev_frame is None or curr_frame is None:
                return (False, 0.0, None) if return_homography else (False, 0.0)
            
            # Resize to 480x360 to ensure high-performance ORB feature matching
            f1_resized = cv2.resize(prev_frame, (480, 360), interpolation=cv2.INTER_AREA)
            f2_resized = cv2.resize(curr_frame, (480, 360), interpolation=cv2.INTER_AREA)
            
            gray1 = cv2.cvtColor(f1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(f2_resized, cv2.COLOR_BGR2GRAY)
            
            orb = cv2.ORB_create(nfeatures=300)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return (False, 0.0, None) if return_homography else (False, 0.0)
                
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) < 8:
                return (False, 0.0, None) if return_homography else (False, 0.0)
                
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None and mask is not None:
                # 1. Background consistency: check RANSAC inlier ratio and count
                inlier_ratio = np.sum(mask) / len(mask)
                num_inliers = int(np.sum(mask))
                if inlier_ratio < 0.50 or num_inliers < 20:
                    return (False, 0.0, None) if return_homography else (False, 0.0)

                # 2. Homography consistency: verify scale factors (preserves 2D geometry)
                scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
                scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)
                if not (0.85 < scale_x < 1.15 and 0.85 < scale_y < 1.15):
                    return (False, 0.0, M) if return_homography else (False, 0.0)

                # 3. Global feature displacement check
                inlier_mask = mask.ravel() == 1
                disps = dst_pts[inlier_mask] - src_pts[inlier_mask]
                disps = disps.reshape(-1, 2)
                
                mags = np.linalg.norm(disps, axis=1)
                avg_disp = float(np.mean(mags))
                
                # Require significant movement magnitude to trigger camera moved
                if avg_disp < 10.0:
                    return (False, 0.0, M) if return_homography else (False, 0.0)

                # Verify direction consistency (global shift)
                x_mean, y_mean = np.mean(disps[:, 0]), np.mean(disps[:, 1])
                mean_mag = np.sqrt(x_mean**2 + y_mean**2)
                if avg_disp > 0 and (mean_mag / avg_disp) < 0.80:
                    return (False, 0.0, M) if return_homography else (False, 0.0)

                tx = M[0, 2]
                ty = M[1, 2]
                shift = np.sqrt(tx**2 + ty**2)
                if shift > 15.0:  # Shift threshold in pixels on 480x360 plane
                    return (True, shift, M) if return_homography else (True, shift)
            return (False, 0.0, M) if return_homography else (False, 0.0)
        except Exception as e:
            print(f"[ORB Error] {e}")
            return (False, 0.0, None) if return_homography else (False, 0.0)

    def classify(self, frame, history_frames, prob=0.5) -> str:
        """
        Analyzes the current frame and 15-frame history to classify the exact tamper type.
        Returns one of the official 12 core classes:
        NORMAL, FULL_LENS_COVER, PARTIAL_LENS_COVER, HAND_COVER, BLUR_ATTACK,
        DEFOCUS, CAMERA_MOVED, CAMERA_REDIRECTED, VIDEO_FREEZE, BRIGHTNESS_ATTACK,
        DARKNESS_ATTACK, NOISE_ATTACK.
        """
        if frame is None or prob < 0.50:
            return "NORMAL"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        mean_intensity = float(np.mean(gray))
        black_ratio = float(np.sum(gray < 25) / gray.size)
        white_ratio = float(np.sum(gray > 230) / gray.size)

        # Shannon Entropy
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        prob_dist = hist / (hist.sum() + 1e-12)
        prob_dist = prob_dist[prob_dist > 0]
        entropy = float(-np.sum(prob_dist * np.log2(prob_dist + 1e-12)))

        # Laplacian focus and Edge metrics
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge_density = float(np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size)

        # Skin density for Hand Cover
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([20, 180, 255]))
        skin_ratio = float(np.sum(skin_mask > 0) / skin_mask.size)

        # 8x8 grid contrast check for localized occlusion
        h_grid, w_grid = h // 8, w // 8
        flat_blocks = 0
        for r in range(8):
            for c in range(8):
                block = gray[r*h_grid : (r+1)*h_grid, c*w_grid : (c+1)*w_grid]
                block_std = np.std(block)
                if block_std < 10.0:
                    flat_blocks += 1
        flat_ratio = flat_blocks / 64.0

        # Left/Right half blackness
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        left_black = float(np.sum(left_half < 25) / left_half.size)
        right_black = float(np.sum(right_half < 25) / right_half.size)
        
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        top_black = float(np.sum(top_half < 25) / top_half.size)
        bottom_black = float(np.sum(bottom_half < 25) / bottom_half.size)

        is_half_covered = (
            (left_black > 0.70 and right_black < 0.08) or 
            (right_black > 0.70 and left_black < 0.08) or
            (top_black > 0.70 and bottom_black < 0.08) or
            (bottom_black > 0.70 and top_black < 0.08)
        )

        is_frozen_stream = False
        if len(history_frames) >= 15:
            diffs = []
            for k in range(1, len(history_frames)):
                d = np.mean(np.abs(history_frames[k].astype(np.float32) - history_frames[k-1].astype(np.float32)))
                diffs.append(d)
            if len(diffs) > 0 and np.max(diffs) < 0.02:
                is_frozen_stream = True

        # ==========================================
        # SIMPLE, DIRECT CLASSIFICATION RULES
        # ==========================================
        
        # 1. Video Freeze (Only when structured; prevents dark/grey freezes from overriding covers)
        if is_frozen_stream and laplacian_var >= 15.0 and entropy >= 3.0:
            return "VIDEO_FREEZE"

        # 2. Darkness Attack (True blackout)
        if mean_intensity < 8.0 or black_ratio > self.thresholds["black_ratio_limit"]:
            return "DARKNESS_ATTACK"

        # 3. Brightness Attack or Glare (includes full brightness and partial flashlight glare)
        if mean_intensity > 235.0 or (white_ratio > self.thresholds["white_ratio_limit"] and mean_intensity > 80.0) or (white_ratio > 0.15 and mean_intensity > 130.0):
            return "BRIGHTNESS_ATTACK"

        # 4. Full Lens Cover (low focus details and low entropy)
        if (black_ratio > self.thresholds["black_ratio_limit"] or white_ratio > self.thresholds["white_ratio_limit"]) and entropy < self.thresholds["entropy_limit"] and edge_density < self.thresholds["edge_density_limit"]:
            return "FULL_LENS_COVER"

        # 5. Hand Cover (8x8 grid localized occlusion + edge density loss + no camera shift)
        is_occluded = (0.25 <= flat_ratio <= 0.85)
        moved = False
        homography_matrix = None
        if len(history_frames) >= 15:
            moved, _, homography_matrix = self.evaluate_motion(history_frames[0], history_frames[-1], return_homography=True)
            
        # 6. Camera Rotation (prevent rotation hijack under glare/blackouts)
        if homography_matrix is not None and white_ratio < 0.15 and black_ratio < 0.15:
            rot_res = self.rotation_detector.evaluate(homography_matrix)
            if rot_res == "CAMERA_ROTATED":
                return "CAMERA_ROTATED"

        # 7. Contrast Attack
        contrast_res = self.contrast_detector.evaluate(gray)
        if contrast_res != "NORMAL":
            return contrast_res

        # 8. Saturation Attack
        sat_res = self.saturation_detector.evaluate(frame)
        if sat_res != "NORMAL":
            return sat_res

        # 9. Color Distortion
        color_res = self.color_detector.evaluate(frame)
        if color_res != "NORMAL":
            return color_res

        # 10. Sharpness Attack
        sharpness_res = self.sharpness_detector.evaluate(gray)
        if sharpness_res == "SHARPNESS_HIGH":
            return "SHARPNESS_HIGH"
        elif sharpness_res == "SHARPNESS_LOW":
            return "SHARPNESS_LOW"

        # Prevent hand cover triggers under active flashlight glare
        if is_occluded and edge_density < 0.03 and not moved and white_ratio < 0.15:
            return "HAND_COVER"

        # 11. Blur Attack (High blur)
        if laplacian_var < self.thresholds["laplacian_blur_limit"]:
            return "BLUR_ATTACK"

        # 12. Defocus (Moderate blur)
        if laplacian_var < self.thresholds["laplacian_defocus_limit"]:
            return "DEFOCUS"

        # 13. Partial Lens Cover (Half Cover fallback)
        if is_half_covered or is_occluded:
            return "PARTIAL_LENS_COVER"

        # 14. Camera Moved (only when evaluated as moved)
        if moved:
            if edge_density < 0.05 and entropy < 4.5:
                return "CAMERA_REDIRECTED"
            return "CAMERA_MOVED"

        # 15. Noise Attack
        if edge_density > 0.75 and entropy > 7.85:
            return "NOISE_ATTACK"

        # Fallback to UNKNOWN_ANOMALY if we are confidently in the tampered state but rules didn't match
        if prob >= 0.50:
            return "UNKNOWN_ANOMALY"

        return "NORMAL"
