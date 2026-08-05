import os
import time
import json
import numpy as np
import cv2
import joblib
import pandas as pd
import sys
from backend.services.event_service import EventService
from backend.notifications.notification_manager import NotificationManager
from src.preprocessing.pipeline import PreprocessingPipeline
from backend.tamper.classification_engine import TamperClassificationEngine
from backend.config.sms_settings import SMSSettings

def run_live_sms_pipeline_verification():
    print("="*80)
    print("      SPECTRAGUARD LIVE NOTIFICATION PIPELINE E2E INTEGRATION")
    print("="*80)

    # 1. Strict credentials check (Failure conditions)
    if not SMSSettings.validate_configuration():
        print("[FAIL] Twilio credentials are missing in the active environment.")
        sys.exit(1)

    # Check recipient numbers configuration
    settings_path = "config/user_settings.json"
    if not os.path.exists(settings_path):
        print(f"[FAIL] Missing {settings_path} settings file.")
        sys.exit(1)

    with open(settings_path, "r", encoding="utf-8") as f:
        settings_data = json.load(f)
        contacts = settings_data.get("emergency_contacts", [])
        if not contacts:
            print("[FAIL] No numbers configured in emergency_contacts list.")
            sys.exit(1)

    print("Configured Contacts found: PASS")

    # Load ML models
    model_dir = "data/models/latest"
    model_path = os.path.join(model_dir, "production_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    metadata_path = os.path.join(model_dir, "feature_metadata.json")
    threshold_path = os.path.join(model_dir, "threshold.json")
    
    assert os.path.exists(model_path) and os.path.exists(scaler_path), "Model artifacts missing!"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        feature_order = json.load(f).get("feature_names") or []
        
    optimal_threshold = 0.55
    if os.path.exists(threshold_path):
        with open(threshold_path, "r", encoding="utf-8") as f:
            optimal_threshold = float(json.load(f).get("optimal_threshold", 0.55))

    pipeline = PreprocessingPipeline()
    classification_engine = TamperClassificationEngine()
    event_service = EventService()
    notification_manager = NotificationManager()
    
    # Reset counts
    event_service.history_deque.clear()
    notification_manager.notification_history.clear()
    notification_manager.rate_limiter.history.clear()
    notification_manager.consecutive_tamper_count = 0

    # 2. Simulate 5 frames of solid grey (FULL_LENS_COVER)
    print("\nSimulating camera stream with 5 consecutive covered frames...")
    
    frames_history = []
    # Make 10 normal textured frames
    for i in range(10):
        f = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        frames_history.append(f)
        
    # Append 5 solid grey covered frames
    for i in range(5):
        f = np.full((480, 640, 3), 120, dtype=np.uint8) # solid gray (Paper cover)
        frames_history.append(f)

    # Process loop
    for i in range(11, 16):
        rolling_window = frames_history[i-11 : i+4]
        current_frame = rolling_window[-1]
        
        feat_vec = pipeline.extract(rolling_window)
        feat_dict = feat_vec.to_dict()
        feat_vector = [feat_dict.get(f, 0.0) for f in feature_order]
        df = pd.DataFrame([feat_vector], columns=feature_order)
        
        feat_scaled = scaler.transform(df)
        prob = float(model.predict_proba(feat_scaled)[0][1])
        tamper_type = classification_engine.classify(current_frame, rolling_window, prob=prob)
        
        is_tamper = (prob >= optimal_threshold) or (tamper_type != "NORMAL")
        final_prediction = "TAMPERED" if is_tamper else "NORMAL"
        final_tamper_type = tamper_type if is_tamper else "NORMAL"
        
        print(f"   [Frame {i}] Prob: {prob:.4f} | Pred: {final_prediction} | Tamper: {final_tamper_type}")
        
        if is_tamper:
            event_service.handle_detection(
                camera_name=settings_data.get("camera_name", "Gate-1"),
                frame=current_frame,
                prob=prob,
                severity="CRITICAL",
                drift=prob,
                rule=final_tamper_type
            )
            # Clear event deduplication cache to let consecutive frames hit the threshold
            with event_service.cache_lock:
                event_service.last_triggered.clear()
        time.sleep(0.1)

    print("\nWaiting for asynchronous notification queue worker to send Twilio SMS...")
    # Sleep to allow Twilio rest API call network roundtrips to settle
    time.sleep(3.5)
    
    # 3. Assertions
    history = event_service.get_history()
    if not history:
        print("[FAIL] No events registered in EventService history.")
        sys.exit(1)
        
    last_event = history[-1]
    status = last_event.get("notification_status")
    attempts = last_event.get("notification_attempts", 0)
    sid = last_event.get("message_sid")
    error = last_event.get("notification_error")

    print(f"\nPipeline Verification Outputs:")
    print(f"   SMS Status : {status}")
    print(f"   Attempts   : {attempts}")
    print(f"   Message SID: {sid}")
    
    if error and error != "None":
        print(f"   Error log  : {error}")

    # The verification MUST fail if Twilio returns an exception/failure
    if status == "SMS Failed" or not sid or not sid.startswith("SM"):
        print(f"\n[FAIL] SMS Pipeline dispatch failed. Error Details: {error}")
        sys.exit(1)

    print("\n" + "="*80)
    print("PHASE 7C LIVE PIPELINE NOTIFICATION VERIFIED")
    print("="*80)

if __name__ == "__main__":
    run_live_sms_pipeline_verification()
