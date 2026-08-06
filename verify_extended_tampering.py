import os
import sys
import numpy as np
import cv2
import json

# Ensure project path resolution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.tamper.classification_engine import TamperClassificationEngine
from backend.services.event_service import EventService

def create_neutral_frame():
    # Create a rich colored frame with distinct shapes to ensure high contrast, saturation, and sharpness
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    
    # Red rectangle
    cv2.rectangle(img, (50, 50), (250, 220), (50, 50, 220), -1)
    # Green circle
    cv2.circle(img, (480, 160), 90, (50, 220, 50), -1)
    # Blue rectangle
    cv2.rectangle(img, (100, 260), (550, 420), (220, 50, 50), -1)
    # High-frequency text
    cv2.putText(img, "SpectraGuard V2", (140, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    
    return img

def main():
    print("==================================================")
    print(" SpectraGuard Extended Tampering Verification Suite")
    print("==================================================")

    # Initialize Engine and EventService
    engine = TamperClassificationEngine()
    event_service = EventService()
    
    # Track results
    results = {}

    normal_frame = create_neutral_frame()

    # 0. Base normal verification (must not trigger any tamper)
    print("\n[TEST] Verifying NORMAL (negative test)...")
    event_service.last_triggered.clear()
    neg_res = engine.classify(normal_frame, [], prob=0.1)
    if neg_res == "NORMAL":
        results["NORMAL_NEGATIVE"] = "PASS"
    else:
        results["NORMAL_NEGATIVE"] = f"FAIL (Got: {neg_res})"

    # 1. CONTRAST_LOW verification
    print("\n[TEST] Verifying CONTRAST_LOW...")
    event_service.last_triggered.clear()
    # Scale intensities toward mean to reduce standard deviation under 15
    low_contrast_frame = np.clip(128.0 + (normal_frame.astype(float) - 128.0) * 0.05, 0, 255).astype(np.uint8)
    pos_res = engine.classify(low_contrast_frame, [], prob=0.6)
    evt = event_service.handle_detection("test_cam", low_contrast_frame, 0.95, "MEDIUM", 0.95, pos_res)
    
    if pos_res == "CONTRAST_LOW" and evt is not None and evt.tamper_type == "CONTRAST_LOW":
        results["CONTRAST_LOW"] = "PASS"
    else:
        results["CONTRAST_LOW"] = f"FAIL (Got: {pos_res})"

    # 2. CONTRAST_HIGH verification
    print("[TEST] Verifying CONTRAST_HIGH...")
    event_service.last_triggered.clear()
    # Stretch intensities from mean to boost standard deviation above 45
    high_contrast_frame = np.clip(128.0 + (normal_frame.astype(float) - 128.0) * 5.0, 0, 255).astype(np.uint8)
    pos_res = engine.classify(high_contrast_frame, [], prob=0.6)
    evt = event_service.handle_detection("test_cam", high_contrast_frame, 0.95, "MEDIUM", 0.95, pos_res)
    
    if pos_res == "CONTRAST_HIGH" and evt is not None and evt.tamper_type == "CONTRAST_HIGH":
        results["CONTRAST_HIGH"] = "PASS"
    else:
        results["CONTRAST_HIGH"] = f"FAIL (Got: {pos_res})"

    # 3. SATURATION_LOW verification
    print("[TEST] Verifying SATURATION_LOW...")
    event_service.last_triggered.clear()
    # Convert BGR to Grayscale then back to BGR (removes all saturation)
    gray_frame_bgr = cv2.cvtColor(cv2.cvtColor(normal_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    pos_res = engine.classify(gray_frame_bgr, [], prob=0.6)
    evt = event_service.handle_detection("test_cam", gray_frame_bgr, 0.95, "MEDIUM", 0.95, pos_res)
    
    if pos_res == "SATURATION_LOW" and evt is not None and evt.tamper_type == "SATURATION_LOW":
        results["SATURATION_LOW"] = "PASS"
    else:
        results["SATURATION_LOW"] = f"FAIL (Got: {pos_res})"

    # 4. SATURATION_HIGH verification
    print("[TEST] Verifying SATURATION_HIGH...")
    event_service.last_triggered.clear()
    # Convert to HSV and boost Saturation channel directly to 240
    hsv = cv2.cvtColor(normal_frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = 240
    high_sat_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    pos_res = engine.classify(high_sat_frame, [], prob=0.6)
    evt = event_service.handle_detection("test_cam", high_sat_frame, 0.95, "MEDIUM", 0.95, pos_res)
    
    if pos_res == "SATURATION_HIGH" and evt is not None and evt.tamper_type == "SATURATION_HIGH":
        results["SATURATION_HIGH"] = "PASS"
    else:
        results["SATURATION_HIGH"] = f"FAIL (Got: {pos_res})"

    # 5. SHARPNESS_HIGH verification
    print("[TEST] Verifying SHARPNESS_HIGH...")
    event_service.last_triggered.clear()
    # Generate high frequency checkerboard pattern with intermediate colors (4x4 blocks)
    sharp_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        for x in range(640):
            bx = x // 4
            by = y // 4
            if (bx + by) % 2 == 0:
                sharp_frame[y, x] = [200, 100, 100]
            else:
                sharp_frame[y, x] = [100, 200, 100]
                
    gray_sh = cv2.cvtColor(sharp_frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray_sh, cv2.CV_64F)
    lap_var = float(np.var(lap))
    gx = cv2.Sobel(gray_sh, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_sh, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))
    print(f"DEBUG: Sharpness High checkerboard has lap_var={lap_var}, tenengrad={tenengrad}")
    
    pos_res = engine.classify(sharp_frame, [], prob=0.6)
    evt = event_service.handle_detection("test_cam", sharp_frame, 0.95, "MEDIUM", 0.95, pos_res)
    
    if pos_res == "SHARPNESS_HIGH" and evt is not None and evt.tamper_type == "SHARPNESS_HIGH":
        results["SHARPNESS_HIGH"] = "PASS"
    else:
        results["SHARPNESS_HIGH"] = f"FAIL (Got: {pos_res})"

    # 6. SHARPNESS_LOW verification
    print("[TEST] Verifying SHARPNESS_LOW...")
    event_service.last_triggered.clear()
    # Apply a strong Gaussian blur (removes high frequencies)
    blurred_frame = cv2.GaussianBlur(normal_frame, (51, 51), 0)
    pos_res = engine.classify(blurred_frame, [], prob=0.6)
    evt = event_service.handle_detection("test_cam", blurred_frame, 0.95, "MEDIUM", 0.95, pos_res)
    
    if pos_res == "SHARPNESS_LOW" and evt is not None and evt.tamper_type == "SHARPNESS_LOW":
        results["SHARPNESS_LOW"] = "PASS"
    else:
        results["SHARPNESS_LOW"] = f"FAIL (Got: {pos_res})"

    # 7. COLOR_DISTORTION verification
    print("[TEST] Verifying COLOR_DISTORTION...")
    event_service.last_triggered.clear()
    # Add a strong blue cast to shift balance
    tinted_frame = np.clip(normal_frame.astype(float) + [120.0, 0.0, 0.0], 0, 255).astype(np.uint8)
    pos_res = engine.classify(tinted_frame, [], prob=0.6)
    evt = event_service.handle_detection("test_cam", tinted_frame, 0.95, "MEDIUM", 0.95, pos_res)
    
    if pos_res == "COLOR_DISTORTION" and evt is not None and evt.tamper_type == "COLOR_DISTORTION":
        results["COLOR_DISTORTION"] = "PASS"
    else:
        results["COLOR_DISTORTION"] = f"FAIL (Got: {pos_res})"

    # 8. CAMERA_ROTATED verification
    print("[TEST] Verifying CAMERA_ROTATED...")
    event_service.last_triggered.clear()
    # Evaluate camera rotation with a 35 degree homography matrix
    theta = np.radians(35.0)
    M = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0]
    ])
    pos_res = engine.rotation_detector.evaluate(M)
    evt = event_service.handle_detection("test_cam", normal_frame, 0.95, "MEDIUM", 0.95, pos_res)
    
    if pos_res == "CAMERA_ROTATED" and evt is not None and evt.tamper_type == "CAMERA_ROTATED":
        results["CAMERA_ROTATED"] = "PASS"
    else:
        results["CAMERA_ROTATED"] = f"FAIL (Got: {pos_res})"

    # Summary table
    print("\n" + "="*50)
    print("          EXTENDED TAMPER VERIFICATION RESULTS")
    print("="*50)
    print(f"{'Tampering Category':<25} | {'Verification Status':<20}")
    print("-"*50)
    for name, status in sorted(results.items()):
        print(f"{name:<25} | {status:<20}")
    print("="*50)

if __name__ == "__main__":
    main()
