import os, sys, json, joblib, time
import cv2
import numpy as np
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath('src'))

try:
    from preprocessing.pipeline import PreprocessingPipeline
except ImportError as e:
    print("[FATAL] Failed to import PreprocessingPipeline: {}".format(e))
    sys.exit(1)

def main():
    print("[INFO] Booting SpectraGuard Live Inference Engine (REPAIRED)...")
    
    model_dir = os.path.normpath('data/models/latest')
    try:
        model = joblib.load(os.path.join(model_dir, 'production_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.joblib'))
        with open(os.path.join(model_dir, 'threshold.json'), 'r') as f:
            threshold = json.load(f).get('optimal_threshold', 0.4285)
        with open(os.path.join(model_dir, 'feature_metadata.json'), 'r') as f:
            feat_cols = json.load(f).get('feature_names', [])
        print("[INFO] Artifacts loaded. Operating strictly at threshold {:.4f}".format(threshold))
    except Exception as e:
        print("[FATAL] Artifact load failed: {}".format(e))
        sys.exit(1)

    try:
        pipeline = PreprocessingPipeline()
    except Exception as e:
        print("[FATAL] Pipeline initialization failed: {}".format(e))
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] Cannot open physical webcam on index 0.")
        sys.exit(1)

    buffer = []
    buffer_size = 15
    frame_count = 0
    duplicate_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("[INFO] Camera stream active. Awaiting buffer fill. Press 'q' to terminate.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Hardware frame read failure.")
            break
            
        start_t = time.time()
        frame_count += 1
        
        # --- BUG 2 REPAIR: Hardware-aware duplicate filtering ---
        if len(buffer) > 0 and np.array_equal(frame, buffer[-1]):
            duplicate_count += 1
            continue
            
        buffer.append(frame.copy())
        if len(buffer) > buffer_size:
            buffer.pop(0)

        overlay = frame.copy()
        status = "Buffering Frame Data..."
        color = (0, 255, 255)
        prob = 0.0
        pred = 0
        conf = 0.0
        
        if len(buffer) == buffer_size:
            try:
                feat_vec = pipeline.extract(buffer)
                feat_dict = feat_vec.to_dict()
                raw_features = np.array([[feat_dict[c] for c in feat_cols]])
                
                # --- BUG 1 REPAIR: Verified sklearn Scaling ---
                scaled = scaler.transform(raw_features)
                
                prob = model.predict_proba(scaled)[:, 1][0]
                pred = int(prob >= threshold)
                conf = prob if pred == 1 else (1.0 - prob)
                
                status = "TAMPERING DETECTED" if pred == 1 else "ENVIRONMENT SECURE"
                color = (0, 0, 255) if pred == 1 else (0, 255, 0)

                # Diagnostic print for live forensic verification (Outputs every ~2 seconds)
                if frame_count % 60 == 0: 
                    print("\n--- INFERENCE AUDIT (Frame {}) ---".format(frame_count))
                    print("Raw LapVar: {:.2f} | Scaled LapVar: {:.2f}".format(raw_features[0][feat_cols.index('laplacian_variance')], scaled[0][feat_cols.index('laplacian_variance')]))
                    print("Raw TempDiff: {:.2f} | Scaled TempDiff: {:.2f}".format(raw_features[0][feat_cols.index('temporal_difference')], scaled[0][feat_cols.index('temporal_difference')]))
                    print("Prob: {:.4f} | Pred: {} | Duplicates Blocked: {}".format(prob, pred, duplicate_count))
                
            except Exception as e:
                status = "Pipeline Extraction Error"
                color = (0, 165, 255)
        
        latency = (time.time() - start_t) * 1000
        fps = 1000.0 / latency if latency > 0 else 0

        cv2.rectangle(overlay, (5, 5), (420, 170), (0, 0, 0), -1)
        cv2.putText(overlay, "STATUS: {}".format(status), (15, 30), font, 0.7, color, 2)
        cv2.putText(overlay, "Predictive Probability: {:.4f}".format(prob), (15, 60), font, 0.5, (255,255,255), 1)
        cv2.putText(overlay, "Inference Confidence: {:.2f}".format(conf), (15, 85), font, 0.5, (255,255,255), 1)
        cv2.putText(overlay, "Operating Threshold: {:.4f}".format(threshold), (15, 110), font, 0.5, (255,255,255), 1)
        cv2.putText(overlay, "Engine Latency: {:.1f}ms | FPS: {:.1f}".format(latency, fps), (15, 135), font, 0.5, (255,255,255), 1)
        cv2.putText(overlay, "Frames: {} | Buffer: {}/{} | Dupes: {}".format(frame_count, len(buffer), buffer_size, duplicate_count), (15, 160), font, 0.5, (255,255,255), 1)
        
        cv2.imshow('SpectraGuard Live Execution', overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Terminating live feed.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
