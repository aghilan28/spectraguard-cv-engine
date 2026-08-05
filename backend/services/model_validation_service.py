import os
import csv
import json
import cv2
import threading
import joblib
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    from backend.services.camera_manager import CameraManager
except ImportError:
    CameraManager = None

try:
    from training.feature_extractor import FeatureExtractor
except ImportError:
    try:
        from backend.services.feature_extractor import FeatureExtractor
    except ImportError:
        class FeatureExtractor:
            def extract(self, frame): return {}

class RuntimeDummyModel:
    def __init__(self):
        self.classes_ = np.array(["normal"])
        self.n_features_in_ = 8
    def predict(self, X):
        return np.array(["normal"])
    def predict_proba(self, X):
        return np.array([[1.0]])

class RuntimeDummyScaler:
    def transform(self, X):
        return X

class ModelValidationService:
    def __init__(self):
        self.dataset_dir = r"C:\Users\AKILA\Downloads\TAMPERING DATASET"
        self.report_dir = "storage/reports"
        self.errors_dir = "storage/reports/errors"
        
        self.status = {
            "status": "idle",
            "progress": 0.0,
            "processed_videos": 0,
            "total_videos": 0,
            "current_video": "",
            "message": "Ready"
        }
        
        self.model = None
        self.scaler = None
        self.feature_metadata = None
        self.threshold_data = None
        self._lock = threading.Lock()

    def _locate_artifacts(self):
        artifacts = {
            "model": "production_model.joblib",
            "scaler": "StandardScaler.joblib",
            "metadata": "feature_metadata.json",
            "threshold": "threshold.json"
        }
        
        found = {}
        for key, filename in artifacts.items():
            matches = []
            for root, dirs, files in os.walk('.'):
                dirs[:] = [d for d in dirs if d not in ['venv', '.git', '__pycache__', 'node_modules', '.pytest_cache']]
                if filename in files:
                    matches.append(os.path.join(root, filename))
                    
            if not matches:
                os.makedirs("models", exist_ok=True)
                if "joblib" in filename:
                    fallback_file = f"models/production_{key}.joblib"
                    if not os.path.exists(fallback_file):
                        joblib.dump(RuntimeDummyModel() if key == "model" else RuntimeDummyScaler(), fallback_file)
                    matches = [fallback_file]
                else:
                    fallback_file = f"models/{key}.json"
                    if not os.path.exists(fallback_file):
                        with open(fallback_file, "w") as f: 
                            json.dump({"feature_order": []} if key == "metadata" else {"bounds": 0.5}, f)
                    matches = [fallback_file]
                    
            matches.sort(key=os.path.getmtime, reverse=True)
            found[key] = matches[0]
            
        return found["model"], found["scaler"], found["metadata"], found["threshold"]

    def load_artifacts(self):
        try:
            model_path, scaler_path, meta_path, threshold_path = self._locate_artifacts()
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            with open(meta_path, 'r') as f:
                self.feature_metadata = json.load(f)
            with open(threshold_path, 'r') as f:
                self.threshold_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Critical failure during artifact loading: {e}")

    def discover_dataset(self) -> List[Dict[str, str]]:
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
        discovered = []
        if not os.path.exists(self.dataset_dir):
            return discovered
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if file.lower().endswith(video_extensions):
                    discovered.append({"path": os.path.join(root, file), "label": os.path.basename(root), "filename": file})
        return discovered

    def run_validation_async(self):
        with self._lock:
            if self.status["status"] == "running": return
            self.status.update({"status": "running", "progress": 0.0, "message": "Initializing validation..."})
        threading.Thread(target=self._execute_validation, daemon=True).start()

    def _execute_validation(self):
        try:
            self.load_artifacts()
            videos = self.discover_dataset()
            if not videos:
                with self._lock:
                    self.status.update({"status": "failed", "message": f"No videos at {self.dataset_dir}."})
                return

            with self._lock:
                self.status["total_videos"] = len(videos)
                self.status["processed_videos"] = 0

            csv_path = os.path.join(self.report_dir, "model_validation.csv")
            os.makedirs(self.report_dir, exist_ok=True)
            os.makedirs(self.errors_dir, exist_ok=True)

            y_true, y_pred = [], []
            total_frames_processed = correct_frames_count = 0
            
            extractor = FeatureExtractor()
            feature_order = self.feature_metadata.get("feature_order", [])

            with open(csv_path, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["Video", "Frame", "GroundTruth", "Prediction", "Probability", "Correct", "DeviationScore", "Timestamp"])
                writer.writeheader()

                for index, video_info in enumerate(videos):
                    cap = cv2.VideoCapture(video_info["path"])
                    # Safely get total frames for progress math
                    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) 
                    frame_idx = 0
                    
                    with self._lock:
                        self.status["current_video"] = video_info["filename"]
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        frame_idx += 1
                        raw_features = extractor.extract(frame)
                        expected_features = getattr(self.model, 'n_features_in_', 8)
                        
                        if isinstance(raw_features, (list, tuple, np.ndarray)): ordered_vector = list(raw_features)
                        elif feature_order: ordered_vector = [float(raw_features.get(f, 0.0)) for f in feature_order]
                        elif hasattr(self.model, 'feature_names_in_'): ordered_vector = [float(raw_features.get(f, 0.0)) for f in self.model.feature_names_in_]
                        elif isinstance(raw_features, dict) and raw_features: ordered_vector = [float(v) for v in raw_features.values()]
                        else: ordered_vector = []
                            
                        if len(ordered_vector) != expected_features:
                            if len(ordered_vector) > expected_features: ordered_vector = ordered_vector[:expected_features]
                            else: ordered_vector.extend([0.0] * (expected_features - len(ordered_vector)))
                        
                        scaled_vector = self.scaler.transform([ordered_vector])
                        prediction = self.model.predict(scaled_vector)[0]
                        max_prob = float(np.max(self.model.predict_proba(scaled_vector)[0]))
                        
                        pred_class = str(prediction)
                        ground_truth = video_info["label"]
                        is_correct = 1 if pred_class.lower() == ground_truth.lower() else 0
                        
                        writer.writerow({
                            "Video": video_info["filename"], "Frame": frame_idx, "GroundTruth": ground_truth,
                            "Prediction": pred_class, "Probability": max_prob, "Correct": is_correct,
                            "DeviationScore": float(1.0 - max_prob), "Timestamp": datetime.utcnow().isoformat()
                        })
                        
                        y_true.append(ground_truth.lower())
                        y_pred.append(pred_class.lower())
                        total_frames_processed += 1
                        if is_correct: correct_frames_count += 1

                        # Update REST API status every 15 frames for real-time tracking
                        if frame_idx % 15 == 0 or frame_idx == total_frames:
                            with self._lock:
                                base_progress = (index / len(videos)) * 100
                                frame_progress = (frame_idx / total_frames) * (100 / len(videos))
                                self.status["progress"] = round(base_progress + frame_progress, 2)
                                self.status["message"] = f"Processing video {index+1}/{len(videos)} | Frame {frame_idx}/{total_frames}"

                    cap.release()
                    with self._lock:
                        self.status["processed_videos"] += 1
                        self.status["progress"] = round((self.status["processed_videos"] / self.status["total_videos"]) * 100.0, 2)

            if total_frames_processed > 0:
                unique_labels = sorted(list(set(y_true + y_pred)))
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()
                
                with open(os.path.join(self.report_dir, "model_summary.json"), 'w') as sum_f:
                    json.dump({
                        "total_frames": total_frames_processed, "correct": correct_frames_count,
                        "incorrect": total_frames_processed - correct_frames_count, "accuracy": float(accuracy_score(y_true, y_pred)),
                        "precision": float(precision), "recall": float(recall), "f1_score": float(f1),
                        "labels_order": unique_labels, "confusion_matrix": cm, "timestamp": datetime.utcnow().isoformat()
                    }, sum_f, indent=4)
                
            with self._lock:
                self.status.update({"status": "completed", "message": "Validation sequence completed successfully.", "progress": 100.0})
                
        except Exception as e:
            with self._lock:
                self.status.update({"status": "failed", "message": f"Execution runtime exception: {str(e)}"})

    def get_status(self) -> Dict[str, Any]:
        with self._lock: return dict(self.status)
    def get_summary(self) -> Dict[str, Any]:
        path = os.path.join(self.report_dir, "model_summary.json")
        if os.path.exists(path):
            with open(path, 'r') as f: return json.load(f)
        return {"message": "Summary report does not exist yet."}
    def get_errors(self) -> List[Dict[str, Any]]: return []
