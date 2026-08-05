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
                if shift > 12.0:  # Shift threshold in pixels
                    return True, shift
            return False, 0.0
        except Exception as e:
            print(f"[ORB Error] {e}")
            return False, 0.0

    def classify(self, frame, history_frames) -> str:
        """
        Analyzes the current frame and 15-frame history to classify the exact tamper type.
        Returns one of: PAPER_COVER, HAND_COVER, HALF_COVER, BLUR, DEFOCUS, 
        BRIGHTNESS_ATTACK, DARKNESS_ATTACK, CAMERA_MOVED, VIDEO_FREEZE, NOISE_ATTACK, COLOR_DISTORTION.
        """
        if frame is None:
            return "NORMAL"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # General frame statistics
        mean_intensity = float(np.mean(gray))
        var_intensity = float(np.var(gray))
        black_ratio = float(np.sum(gray < 25) / gray.size)
        white_ratio = float(np.sum(gray > 230) / gray.size)

        # Shannon Entropy
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        prob = hist / (hist.sum() + 1e-12)
        prob = prob[prob > 0]
        entropy = float(-np.sum(prob * np.log2(prob + 1e-12)))

        # Focus/Blur metrics: Laplacian, Tenengrad, Brenner
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # Tenengrad
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(sobelx**2 + sobely**2))
        
        # Brenner (Cast to float32 before subtraction to avoid uint8 overflow/wrap-around)
        diff_x = gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)
        brenner = float(np.sum(diff_x ** 2) / (h * (w - 2)))

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)

        # Skin color detection for Hand Cover (Strict saturation bounds to avoid desks/walls)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([20, 180, 255]))
        skin_ratio = float(np.sum(skin_mask > 0) / skin_mask.size)

        # Spatial Imbalance (Left/Right half, Top/Bottom half)
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        left_black = float(np.sum(left_half < 25) / left_half.size)
        right_black = float(np.sum(right_half < 25) / right_half.size)

        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        top_black = float(np.sum(top_half < 25) / top_half.size)
        bottom_black = float(np.sum(bottom_half < 25) / bottom_half.size)

        # Strict half covered check (must be heavily dark on one side and clean on the other)
        is_half_covered = (
            (left_black > 0.70 and right_black < 0.08) or 
            (right_black > 0.70 and left_black < 0.08) or
            (top_black > 0.70 and bottom_black < 0.08) or
            (bottom_black > 0.70 and top_black < 0.08)
        )

        # consecutive frame differences to detect freeze
        is_frozen_stream = False
        if len(history_frames) >= 15:
            diffs = []
            for k in range(1, len(history_frames)):
                d = np.mean(np.abs(history_frames[k].astype(np.float32) - history_frames[k-1].astype(np.float32)))
                diffs.append(d)
            max_diff = np.max(diffs) if len(diffs) > 0 else 999.0
            
            # Real freeze (identical buffers)
            if max_diff < 0.02:
                is_frozen_stream = True
            else:
                # Average hash comparison if noise is low but hashes match
                hashes = [self.compute_hash(f) for f in history_frames[-15:]]
                if len(set(hashes)) == 1 and hashes[0] != "" and max_diff < 0.15:
                    is_frozen_stream = True

        # ==========================================
        # RULE HIERARCHY
        # ==========================================

        # 1. Total blackout Darkness Attack (Case 8)
        if mean_intensity < 10.0:
            return "DARKNESS_ATTACK"

        # 2. Total blinded Brightness Attack (Case 7)
        if mean_intensity > 240.0:
            return "BRIGHTNESS_ATTACK"

        # 3. Paper Cover (Case 2: low entropy, low Laplacian)
        if (black_ratio > 0.85 or white_ratio > 0.85) and entropy < 2.5 and laplacian_var < 15.0:
            return "PAPER_COVER"

        # 4. Darkness Attack fallback (for general dark environments)
        if mean_intensity < 18.0:
            return "DARKNESS_ATTACK"

        # 5. Brightness Attack fallback (for general bright environments)
        if mean_intensity > 235.0:
            return "BRIGHTNESS_ATTACK"

        # 6. Half Cover
        if is_half_covered:
            return "HALF_COVER"

        # 7. Video Freeze
        if is_frozen_stream:
            return "VIDEO_FREEZE"

        # 8. Hand Cover (Triggered only when significant skin density is observed)
        if skin_ratio > 0.24:
            return "HAND_COVER"

        # 9. Camera Moved (ORB Homography translation)
        if len(history_frames) >= 15:
            moved, shift_val = self.evaluate_motion(history_frames[0], history_frames[-1])
            if moved:
                return "CAMERA_MOVED"

        # 10. Blur
        if laplacian_var < 35.0 and tenengrad < 150.0 and brenner < 150.0:
            return "BLUR"

        # 11. Defocus
        if laplacian_var < 95.0 and tenengrad < 300.0 and brenner < 300.0:
            return "DEFOCUS"

        # 12. Noise Attack
        if edge_density > 0.75 and entropy > 7.85:
            return "NOISE_ATTACK"

        # 13. Color Distortion
        b, g, r_ch = cv2.split(frame)
        channel_means = [np.mean(b), np.mean(g), np.mean(r_ch)]
        if np.std(channel_means) > 65.0:
            return "COLOR_DISTORTION"

        return "NORMAL"
