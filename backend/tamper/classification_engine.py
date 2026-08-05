import cv2
import numpy as np
import time

class TamperClassificationEngine:
    def __init__(self):
        pass

    def compute_hash(self, frame):
        """Computes a simple 64-bit Average Hash for freeze detection."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            avg = np.mean(small)
            return "".join(["1" if p > avg else "0" for p in small.ravel()])
        except Exception:
            return ""

    def evaluate_motion(self, prev_frame, curr_frame):
        """Uses ORB features and Homography to detect camera movement shift."""
        try:
            if prev_frame is None or curr_frame is None:
                return False, 0.0
            
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            orb = cv2.ORB_create(nfeatures=300)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return False, 0.0
                
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) < 8:
                return False, 0.0
                
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                tx = M[0, 2]
                ty = M[1, 2]
                shift = np.sqrt(tx**2 + ty**2)
                if shift > 35.0:  # Shift threshold in pixels
                    return True, shift
            return False, 0.0
        except Exception as e:
            print(f"[ORB Error] {e}")
            return False, 0.0

    def classify(self, frame, history_frames, prob=0.5) -> str:
        """
        Analyzes the current frame and 15-frame history to classify the exact tamper type.
        Returns one of the official 12 core classes:
        NORMAL, FULL_LENS_COVER, PARTIAL_LENS_COVER, HAND_COVER, BLUR_ATTACK,
        DEFOCUS, CAMERA_MOVED, CAMERA_REDIRECTED, VIDEO_FREEZE, BRIGHTNESS_ATTACK,
        DARKNESS_ATTACK, NOISE_ATTACK.
        """
        if frame is None:
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
        if mean_intensity < 8.0:
            return "DARKNESS_ATTACK"

        # 3. Brightness Attack or Glare
        if mean_intensity > 235.0 or (white_ratio > 0.15 and mean_intensity > 80.0):
            return "BRIGHTNESS_ATTACK"

        # 4. Full Lens Cover (low focus details and low entropy)
        if laplacian_var < 15.0 and entropy < 3.0:
            return "FULL_LENS_COVER"

        # 5. Darkness Attack fallback
        if mean_intensity < 18.0:
            return "DARKNESS_ATTACK"

        # 6. Hand Cover (Skin tone matching)
        if skin_ratio > 0.24:
            return "HAND_COVER"

        # 7. Blur Attack (High blur)
        if laplacian_var < 15.0:
            return "BLUR_ATTACK"

        # 8. Defocus (Moderate blur)
        if laplacian_var < 150.0:
            return "DEFOCUS"

        # 9. Partial Lens Cover (Half Cover)
        if is_half_covered:
            return "PARTIAL_LENS_COVER"

        # 10. Camera Moved / Redirected (only when in focus)
        if laplacian_var >= 95.0 and entropy >= 4.0:
            if len(history_frames) >= 15:
                moved, shift_val = self.evaluate_motion(history_frames[0], history_frames[-1])
                if moved:
                    if edge_density < 0.05 and entropy < 4.5:
                        return "CAMERA_REDIRECTED"
                    return "CAMERA_MOVED"

        # 11. Noise Attack
        if edge_density > 0.75 and entropy > 7.85:
            return "NOISE_ATTACK"

        return "NORMAL"
