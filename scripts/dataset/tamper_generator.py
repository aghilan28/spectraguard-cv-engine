import cv2
import os
import random
import numpy as np
import logging
from typing import Dict, Any
from scripts.dataset.attack_library import AttackLibrary

logger = logging.getLogger("TamperGenerator")

class TamperGenerator:
    @staticmethod
    def process_video(input_path: str, output_path: str, attack_type: str, params: Dict[str, Any], seed: int = 42) -> bool:
        if os.path.exists(output_path):
            return True

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {input_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        np.random.seed(seed)
        random.seed(seed)
        
        success = True
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if attack_type == 'defocus':
                    res = AttackLibrary.defocus_blur(frame, **params)
                elif attack_type == 'gaussian_blur':
                    res = AttackLibrary.gaussian_blur(frame, **params)
                elif attack_type == 'partial_occlusion':
                    res = AttackLibrary.partial_occlusion(frame, seed=seed, **params)
                elif attack_type == 'full_occlusion':
                    res = AttackLibrary.full_occlusion(frame)
                elif attack_type == 'spray':
                    res = AttackLibrary.spray_smudge(frame, seed=seed)
                elif attack_type == 'camera_shift':
                    res = AttackLibrary.camera_shift(frame, **params)
                elif attack_type == 'camera_shake':
                    res = AttackLibrary.camera_shake(frame, **params)
                elif attack_type == 'low_light':
                    res = AttackLibrary.low_light(frame, **params)
                else:
                    res = frame

                out.write(res)
        except Exception as e:
            logger.error(f"Frame processing error on {input_path}: {str(e)}")
            success = False
        finally:
            cap.release()
            out.release()
            
        return success
