import os
import sys
import json
import time
import cv2
import joblib
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import PreprocessingPipeline, FeatureVector

def parse_args():
    parser = argparse.ArgumentParser(description="SpectraGuard Live Camera Demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--simulate", action="store_true", help="Simulate camera input with random noise frames")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=== SpectraGuard Live Camera Demo ===")

    # Define paths
    release_dir = os.path.join("data", "models", "releases", "v0.9.0-audit")
    model_path = os.path.join(release_dir, "production_model.joblib")
    scaler_path = os.path.join(release_dir, "feature_scaler.joblib")
    thresh_path = os.path.join(release_dir, "threshold.json")

    # Load artifacts
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(thresh_path)):
        print(f"[ERROR] Production model artifacts not found in {release_dir}.")
        print("Please run training first.")
        sys.exit(1)

    print("Loading production model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(thresh_path, "r") as f:
        threshold_info = json.load(f)
    threshold = threshold_info["optimal_threshold"]
    print(f"Model loaded successfully. Operating threshold (tau) = {threshold:.4f}")

    # Initialize preprocessing pipeline
    pipeline = PreprocessingPipeline()
    rolling_window = []
    window_size = 15  # standard from pipeline_config.json

    simulate = args.simulate
    cap = None

    if not simulate:
        print(f"Opening webcam index {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[WARNING] Could not open webcam index {args.camera}. Falling back to simulation mode...")
            simulate = True
            cap = None

    if simulate:
        print("Starting in SIMULATION MODE. Generating mock frames...")
    else:
        print("Webcam opened. Press 'q' or 'ESC' to exit.")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            t0 = time.time()

            if simulate:
                # Generate random synthetic BGR frame
                frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
                # Periodically simulate tampering on visual features (e.g., add high noise or blur)
                if (frame_count // 60) % 2 == 1:
                    # Tampering block simulation
                    frame[200:500, 200:1000] = 0
                time.sleep(0.033) # limit loop to ~30 FPS
                ret = True
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame. Exiting.")
                    break

            # Maintain rolling window of frames
            rolling_window.append(frame)
            if len(rolling_window) > window_size:
                rolling_window.pop(0)

            # Feature extraction and prediction
            is_tampered = False
            prob = 0.0
            latency = 0.0

            if len(rolling_window) == window_size:
                t_extract = time.time()
                # Run preprocessing and feature extraction
                feat_vec = pipeline.extract(rolling_window)
                X_raw = feat_vec.to_numpy().reshape(1, -1)
                # Scaling
                X_scaled = scaler.transform(X_raw)
                # Inference
                prob = model.predict_proba(X_scaled)[0, 1]
                is_tampered = bool(prob >= threshold)
                latency = (time.time() - t_extract) * 1000.0

            # Render overlay info on the frame
            display_frame = frame.copy()
            status_text = f"Status: {'TAMPERING DETECTED' if is_tampered else 'OK'}"
            color = (0, 0, 255) if is_tampered else (0, 255, 0)
            
            cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display_frame, f"Tampering Prob: {prob:.4f} (Threshold: {threshold:.4f})", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Latency: {latency:.1f}ms | Frame: {frame_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if simulate:
                cv2.putText(display_frame, "SIMULATION MODE", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            try:
                cv2.imshow("SpectraGuard Live Camera Inference", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            except cv2.error as e:
                print(f"[INFO] Headless environment detected: cv2.imshow not supported. Running headless demo mode...")
                print(f"Frame {frame_count}: Prob(Tampered)={prob:.4f} | Status={'TAMPERING' if is_tampered else 'OK'}")
                if frame_count >= window_size + 5: # run a few frames to verify and then exit
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("Demo closed.")

if __name__ == "__main__":
    main()
