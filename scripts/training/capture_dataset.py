import os
import sys
import csv
import cv2
import numpy as np

# Ensure root path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.services.feature_extractor import FeatureExtractor
except ImportError:
    from training.feature_extractor import FeatureExtractor

FEATURES = ["fft_low_ratio", "fft_mid_ratio", "fft_high_ratio", "log_total_energy", 
            "laplacian_variance", "edge_density", "shannon_entropy", "temporal_difference"]

def augment_frame(frame: np.ndarray, mode: str) -> list:
    """
    Applies synthetic augmentations to a frame to simulate different settings.
    - 'normal': slight brightness changes, mild noise.
    - 'tamper': extreme brightness changes (darkness/overexposure), heavy blur, noise.
    Returns a list of augmented frames.
    """
    augmented = []
    
    if mode == "normal":
        # 1. Original
        augmented.append(frame.copy())
        # 2. Lower brightness (-20%)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, -30)
        v = np.clip(v, 0, 255)
        augmented.append(cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR))
        # 3. Higher brightness (+20%)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 30)
        v = np.clip(v, 0, 255)
        augmented.append(cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR))
    
    elif mode == "tamper":
        # 1. Original
        augmented.append(frame.copy())
        # 2. Extreme Darkness (simulating light block / black tape)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.zeros_like(v) # pure dark
        augmented.append(cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR))
        # 3. Heavy Defocus / Blur
        blurred = cv2.GaussianBlur(frame, (25, 25), 0)
        augmented.append(blurred)
        # 4. Overexposure (simulating flash attack)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.full_like(v, 255) # pure white
        augmented.append(cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR))
        
    return augmented

def run_interactive_capture():
    print("=========================================================")
    print("SpectraGuard Interactive Live Dataset Capturer")
    print("=========================================================\n")
    print("Instructions:")
    print("  - Point camera at normal scene and press [N] to capture Normal frames.")
    print("  - Cover camera / change brightness and press [T] to capture Tamper frames.")
    print("  - Press [Q] to quit and compile the dataset CSV.\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open live camera.")
        return
        
    extractor = FeatureExtractor()
    records = []
    
    normal_count = 0
    tamper_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        
        # Display overlay instructions
        cv2.putText(display_frame, f"Normal Samples: {normal_count} | Tamper Samples: {tamper_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'N' for Normal | 'T' for Tamper | 'Q' to Quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        cv2.imshow("SpectraGuard Live Dataset Capturer", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('n') or key == ord('N'):
            # Capture normal frame + augmentations
            aug_frames = augment_frame(frame, "normal")
            for af in aug_frames:
                feats = extractor.extract(af)
                if feats:
                    row = [feats.get(f, 0.0) for f in FEATURES] + [0, "real_pipeline_v1"]
                    records.append(row)
                    normal_count += 1
            print(f"Captured {len(aug_frames)} Normal samples (including augmentations). Total Normal: {normal_count}")
            
        elif key == ord('t') or key == ord('T'):
            # Capture tamper frame + augmentations
            aug_frames = augment_frame(frame, "tamper")
            for af in aug_frames:
                feats = extractor.extract(af)
                if feats:
                    row = [feats.get(f, 0.0) for f in FEATURES] + [1, "real_pipeline_v1"]
                    records.append(row)
                    tamper_count += 1
            print(f"Captured {len(aug_frames)} Tamper samples (including augmentations). Total Tamper: {tamper_count}")
            
        elif key == ord('q') or key == ord('Q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if len(records) == 0:
        print("No samples captured. Exiting.")
        return
        
    output_csv = "data/training_dataset.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FEATURES + ["label", "extraction_source"])
        writer.writerows(records)
        
    print(f"\n✔ Live dataset compilation complete! Saved {len(records)} samples to {output_csv}")
    print(f"Normal: {normal_count} | Tamper: {tamper_count}")
    print("\nNext step: Run 'python scripts/training/train_model.py' to train the Random Forest.")

if __name__ == "__main__":
    run_interactive_capture()
