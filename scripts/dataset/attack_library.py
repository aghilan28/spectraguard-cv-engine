import cv2
import numpy as np
import random
from typing import Tuple, Dict, Any

class AttackLibrary:
    _spray_mask_cache = {}

    @staticmethod
    def defocus_blur(frame: np.ndarray, ksize: int = 25) -> np.ndarray:
        return cv2.blur(frame, (ksize, ksize))

    @staticmethod
    def gaussian_blur(frame: np.ndarray, ksize: int = 31) -> np.ndarray:
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(frame, (ksize, ksize), 0)

    @staticmethod
    def partial_occlusion(frame: np.ndarray, percentage: float = 0.25, color: Tuple[int, int, int] = (0, 0, 0), seed: int = 42) -> np.ndarray:
        random.seed(seed)
        h, w = frame.shape[:2]
        area = w * h * percentage
        rect_w = int(np.sqrt(area))
        rect_h = int(np.sqrt(area))
        x = random.randint(0, max(1, w - rect_w))
        y = random.randint(0, max(1, h - rect_h))
        out = frame.copy()
        cv2.rectangle(out, (x, y), (x + rect_w, y + rect_h), color, -1)
        return out

    @staticmethod
    def full_occlusion(frame: np.ndarray) -> np.ndarray:
        return np.zeros_like(frame)

    @staticmethod
    def spray_smudge(frame: np.ndarray, seed: int = 42) -> np.ndarray:
        h, w = frame.shape[:2]
        cache_key = (w, h, seed)
        
        if cache_key not in AttackLibrary._spray_mask_cache:
            random.seed(seed)
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(5):
                cx = random.randint(0, w)
                cy = random.randint(0, h)
                r = random.randint(20, max(21, min(w, h) // 4))
                cv2.circle(mask, (cx, cy), r, (255, 255, 255), -1)
            mask = cv2.GaussianBlur(mask, (99, 99), 0)
            AttackLibrary._spray_mask_cache[cache_key] = mask.astype(np.float32) / 255.0
            
        mask_f = AttackLibrary._spray_mask_cache[cache_key]
        out = frame.astype(np.float32)
        out = out * (1.0 - mask_f) + (255.0 * mask_f * 0.6)
        return out.astype(np.uint8)

    @staticmethod
    def camera_shift(frame: np.ndarray, tx: int = 30, ty: int = 30) -> np.ndarray:
        h, w = frame.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(frame, M, (w, h))

    @staticmethod
    def camera_shake(frame: np.ndarray, max_shift: int = 15) -> np.ndarray:
        tx = random.randint(-max_shift, max_shift)
        ty = random.randint(-max_shift, max_shift)
        return AttackLibrary.camera_shift(frame, tx, ty)

    @staticmethod
    def low_light(frame: np.ndarray, alpha: float = 0.5, beta: int = -40) -> np.ndarray:
        out = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        noise = np.zeros_like(out)
        cv2.randn(noise, 128, 10)
        return cv2.addWeighted(out, 1.0, noise, 1.0, -128.0)
