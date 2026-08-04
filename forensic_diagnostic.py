import os, sys, json, joblib, time
import cv2
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath('src'))

from preprocessing.pipeline import PreprocessingPipeline

def main():
    print("==================================================")
    print("TASK 1: PIPELINE EXECUTION FLOW AUDIT")
    print("==================================================")
    
    # Load Artifacts
    model_dir = os.path.normpath('data/models/latest')
    model = joblib.load(os.path.join(model_dir, 'production_model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.joblib'))
    with open(os.path.join(model_dir, 'threshold.json'), 'r') as f:
        threshold = json.load(f).get('optimal_threshold', 0.4285)
    with open(os.path.join(model_dir, 'feature_metadata.json'), 'r') as f:
        feat_cols = json.load(f).get('feature_names', [])
        
    pipeline = PreprocessingPipeline()
    print("[PASS] Artifacts & Pipeline Initialized.")

    print("\n==================================================")
    print("TASK 3: TRAINING FEATURES STATISTICS")
    print("==================================================")
    csv_path = os.path.normpath('data/datasets/virat/metadata/production_features_8d.csv')
    df = pd.read_csv(csv_path)
    train_stats = df[feat_cols].describe().T
    print(train_stats[['min', 'max', 'mean', 'std']])

    print("\n==================================================")
    print("TASK 7: FRAME BUFFER AUDIT")
    print("==================================================")
    cap = cv2.VideoCapture(0)
    
    # Warmup camera (skip auto-exposure initialization frames)
    for _ in range(10): cap.read()
    
    buffer = []
    timestamps = []
    buffer_size = 15
    
    for i in range(buffer_size):
        ret, frame = cap.read()
        timestamps.append(time.time())
        buffer.append(frame.copy())
        # Add slight delay to simulate actual runtime pacing
        time.sleep(0.03) 
        
    cap.release()
    
    print("Buffer Size: {} frames".format(len(buffer)))
    fps = 1.0 / np.mean(np.diff(timestamps))
    print("Capture FPS: {:.2f}".format(fps))
    
    dupes = 0
    for i in range(1, len(buffer)):
        if np.array_equal(buffer[i], buffer[i-1]):
            dupes += 1
    print("Duplicated Frames: {}".format(dupes))

    print("\n==================================================")
    print("TASK 2 & 5: RAW FEATURES & SCALER AUDIT")
    print("==================================================")
    # Extract
    feat_vec = pipeline.extract(buffer)
    feat_dict = feat_vec.to_dict()
    raw_array = np.array([[feat_dict[c] for c in feat_cols]])
    
    # Scale
    if isinstance(scaler, dict) and 'mean' in scaler:
        scaled_array = (raw_array - scaler['mean']) / scaler['scale']
    else:
        scaled_array = raw_array
        
    for i, col in enumerate(feat_cols):
        print("{:<25} | RAW: {:<12.6f} | SCALED: {:<12.6f}".format(col, raw_array[0][i], scaled_array[0][i]))

    print("\n==================================================")
    print("TASK 4 & 8: FEATURE DRIFT & Z-SCORE ANALYSIS")
    print("==================================================")
    for i, col in enumerate(feat_cols):
        val = raw_array[0][i]
        t_mean = train_stats.loc[col, 'mean']
        t_std = train_stats.loc[col, 'std']
        t_max = train_stats.loc[col, 'max']
        z_score = (val - t_mean) / (t_std if t_std > 0 else 1e-9)
        
        flag = "*** OUT OF BOUNDS ***" if abs(z_score) > 3.0 else ""
        print("{:<25} | Z-Score: {:>6.2f} | Max Allowed: {:.4f} | {}".format(col, z_score, t_max, flag))

    print("\n==================================================")
    print("TASK 6: PREDICTION AUDIT")
    print("==================================================")
    prob = model.predict_proba(scaled_array)[:, 1][0]
    pred = int(prob >= threshold)
    conf = prob if pred == 1 else (1.0 - prob)
    print("Raw Probability  : {:.6f}".format(prob))
    print("Threshold        : {:.6f}".format(threshold))
    print("Prediction       : {}".format(pred))
    print("Confidence       : {:.6f}".format(conf))

if __name__ == '__main__':
    main()
