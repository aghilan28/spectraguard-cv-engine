import cv2
import joblib
import json
import os
import numpy as np
from backend.services.feature_extractor import FeatureExtractor
from backend.services.event_service import EventService

def run_realtime():
    model = joblib.load("data/models/latest/production_model.joblib")
    scaler = joblib.load("data/models/latest/scaler.joblib")
    with open("data/models/latest/feature_metadata.json", "r") as f:
        features = json.load(f)["feature_order"]
        
    extractor = FeatureExtractor()
    event_mgr = EventService()
    cap = cv2.VideoCapture(0)
    
    print("Real-time engine active. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        feats = extractor.extract(frame)
        f_vec = [feats.get(f, 0.0) for f in features]
        
        f_scaled = scaler.transform([f_vec])
        pred = model.predict(f_scaled)[0]
        prob = np.max(model.predict_proba(f_scaled)[0])
        
        is_tamper = pred == 1
        color = (0, 0, 255) if is_tamper else (0, 255, 0)
        label = "TAMPER" if is_tamper else "NORMAL"
        
        if is_tamper and prob > 0.8:
            event_mgr.handle_detection("Camera01", frame, float(prob), "CRITICAL", 0.95, "Physics Drift")
            
        cv2.putText(frame, f"Prediction: {label} ({prob:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("SpectraGuard Real-Time Validation", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()
