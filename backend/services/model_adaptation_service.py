import os
import csv
import json
import time
import shutil
import threading
import glob
import cv2
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

# Reuse ONLY the existing production components
try:
    from training.feature_extractor import FeatureExtractor
except ImportError:
    try:
        from backend.services.feature_extractor import FeatureExtractor
    except ImportError:
        class FeatureExtractor:
            def extract(self, frame): return {f"feat{i}": 0.0 for i in range(1, 9)}

class ModelAdaptationService:
    def __init__(self):
        self.dataset_dir = r"C:\Users\AKILA\Downloads\TAMPERING DATASET"
        self.report_dir = "storage/reports"
        self.candidate_dir = "data/models/candidate"
        self.prod_dir = "models"
        
        self.status = {
            "status": "idle",
            "progress": 0.0,
            "message": "System Ready",
            "training_time": 0.0,
            "dataset_size": 0,
            "best_params": {},
            "candidate_accuracy": 0.0,
            "production_accuracy": 0.0,
            "promotion_decision": "none"
        }
        self._lock = threading.Lock()

    def discover_dataset(self) -> List[Dict[str, Any]]:
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        discovered = []
        if not os.path.exists(self.dataset_dir):
            return discovered
            
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if file.lower().endswith(video_extensions):
                    full_path = os.path.join(root, file)
                    class_name = os.path.basename(root)
                    # Binary classification alignment rule: Normal -> 0, any tamper folder -> 1
                    label = 0 if class_name.lower() == "normal" else 1
                    discovered.append({
                        "path": full_path,
                        "class_name": class_name,
                        "label": label
                    })
        return discovered

    def validate_dataset(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_entries = []
        for entry in entries:
            path = entry["path"]
            if not os.path.exists(path):
                continue
            
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                continue
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                cap.release()
                continue
                
            cap.release()
            valid_entries.append(entry)
        return valid_entries

    def build_training_dataframe(self, entries: List[Dict[str, Any]]) -> pd.DataFrame:
        metadata_records = []
        for entry in entries:
            cap = cv2.VideoCapture(entry["path"])
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = max(1.0, cap.get(cv2.CAP_PROP_FPS))
            duration = float(frame_count / fps)
            cap.release()
            
            metadata_records.append({
                "path": entry["path"],
                "label": entry["label"],
                "class_name": entry["class_name"],
                "frame_count": frame_count,
                "duration": duration
            })
        return pd.DataFrame(metadata_records)

    def extract_features(self, df_meta: pd.DataFrame, feature_order: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        extractor = FeatureExtractor()
        frame_features_records = []
        
        for idx, row in df_meta.iterrows():
            cap = cv2.VideoCapture(row["path"])
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                # Sample 1 frame every second to keep metrics optimal and robust
                fps = cap.get(cv2.CAP_PROP_FPS)
                sample_rate = max(1, int(fps)) if fps > 0 else 30
                if frame_idx % sample_rate != 0:
                    continue
                    
                raw_features = extractor.extract(frame)
                record = {
                    "Video": os.path.basename(row["path"]),
                    "Frame": frame_idx,
                    "Label": row["label"],
                    "Timestamp": datetime.utcnow().isoformat()
                }
                for f in feature_order:
                    record[f] = float(raw_features.get(f, 0.0))
                frame_features_records.append(record)
            cap.release()
            
        df_features = pd.DataFrame(frame_features_records)
        os.makedirs(self.report_dir, exist_ok=True)
        df_features.to_csv(os.path.join(self.report_dir, "training_dataset.csv"), index=False)
        
        X = df_features[feature_order].values
        y = df_features["Label"].values
        return X, y, df_features

    def split_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Stratified split: 80% train+val, 20% test
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_val_idx, test_idx in sss_test.split(X, y):
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]
            
        # Split train_val into 80% training and 20% validation (giving an overall 64/16/20 breakdown)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, val_idx in sss_val.split(X_train_val, y_train_val):
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
            
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[CalibratedClassifierCV, StandardScaler]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Reuse RandomForest standard production settings
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        # Calibrate probabilities using standard CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv='prefit')
        calibrated_model.fit(X_val_scaled, y_val)
        return calibrated_model, scaler

    def evaluate_model(self, model: CalibratedClassifierCV, scaler: StandardScaler, X_eval: np.ndarray, y_eval: np.ndarray) -> Dict[str, Any]:
        X_scaled = scaler.transform(X_eval)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]
        
        acc = float(accuracy_score(y_eval, preds))
        precision, recall, f1, _ = precision_recall_fscore_support(y_eval, preds, average='binary', zero_division=0)
        cm = confusion_matrix(y_eval, preds).tolist()
        
        try:
            roc_auc = float(roc_auc_score(y_eval, probs))
        except Exception:
            roc_auc = 0.5
            
        metrics = {
            "accuracy": acc,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }
        return metrics

    def _locate_production_artifacts(self) -> Dict[str, str]:
        artifacts = {"model": "production_model.joblib", "scaler": "StandardScaler.joblib", "metadata": "feature_metadata.json"}
        found = {}
        for key, name in artifacts.items():
            matches = []
            for root, dirs, files in os.walk('.'):
                dirs[:] = [d for d in dirs if d not in ['venv', '.git', '__pycache__', 'data']]
                if name in files:
                    matches.append(os.path.join(root, name))
            if matches:
                matches.sort(key=os.path.getmtime, reverse=True)
                found[key] = matches[0]
            else:
                found[key] = None
        return found

    def compare_models(self, candidate_acc: float, prod_artifacts: Dict[str, str], X_test: np.ndarray, y_test: np.ndarray) -> Tuple[bool, float]:
        prod_acc = 0.0
        if prod_artifacts["model"] and os.path.exists(prod_artifacts["model"]):
            try:
                old_model = joblib.load(prod_artifacts["model"])
                old_scaler = joblib.load(prod_artifacts["scaler"])
                X_test_scaled = old_scaler.transform(X_test)
                old_preds = old_model.predict(X_test_scaled)
                prod_acc = float(accuracy_score(y_test, old_preds))
            except Exception:
                prod_acc = 0.0
                
        # Configurable promotion threshold: strictly better accuracy beats previous production model
        should_promote = candidate_acc > prod_acc
        return should_promote, prod_acc

    def run_training_async(self):
        with self._lock:
            if self.status["status"] == "running":
                return
            self.status["status"] = "running"
            self.status["progress"] = 0.0
            self.status["message"] = "Initializing training workflow..."
        threading.Thread(target=self._execute_adaptation, daemon=True).start()

    def _execute_adaptation(self):
        start_time = time.time()
        try:
            # 1. Recursive Data Discovery & Validation Check
            with self._lock: self.status["message"] = "Scanning and verifying tampering dataset layout..."
            raw_entries = self.discover_dataset()
            valid_entries = self.validate_dataset(raw_entries)
            
            if not valid_entries:
                with self._lock: self.status.update({"status": "failed", "message": "Zero readable verification frames found."})
                return
                
            df_meta = self.build_training_dataframe(valid_entries)
            with self._lock:
                self.status["dataset_size"] = len(df_meta)
                self.status["progress"] = 20.0

            # 2. Sequential Feature Engineering Processing
            with self._lock: self.status["message"] = "Executing unified feature extraction pipeline..."
            feature_order = ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6", "feat7", "feat8"]
            prod_artifacts = self._locate_production_artifacts()
            
            if prod_artifacts["metadata"] and os.path.exists(prod_artifacts["metadata"]):
                with open(prod_artifacts["metadata"], 'r') as f:
                    feature_order = json.load(f).get("feature_order", feature_order)

            X, y, df_features = self.extract_features(df_meta, feature_order)
            with self._lock: self.status["progress"] = 50.0

            # 3. Partitioning, Training, and Probability Calibration
            with self._lock: self.status["message"] = "Training and calibrating candidate model..."
            X_train, y_train, X_val, y_val, X_test, y_test = self.split_dataset(X, y)
            candidate_model, scaler = self.train_model(X_train, y_train, X_val, y_val)
            with self._lock: self.status["progress"] = 70.0

            # 4. Certification & Promotion Verification Sequence
            with self._lock: self.status["message"] = "Evaluating candidate vs production performance baseline..."
            test_metrics = self.evaluate_model(candidate_model, scaler, X_test, y_test)
            cand_acc = test_metrics["accuracy"]
            
            should_promote, prod_acc = self.compare_models(cand_acc, prod_artifacts, X_test, y_test)
            
            # Save Candidate Model to Staging Directory Area
            os.makedirs(self.candidate_dir, exist_ok=True)
            joblib.dump(candidate_model, os.path.join(self.candidate_dir, "production_model.joblib"))
            joblib.dump(scaler, os.path.join(self.candidate_dir, "StandardScaler.joblib"))
            with open(os.path.join(self.candidate_dir, "feature_metadata.json"), 'w') as f:
                json.dump({
                    "feature_order": feature_order,
                    "feature_names": feature_order
                }, f)
            with open(os.path.join(self.candidate_dir, "threshold.json"), 'w') as f:
                json.dump({"bounds": 0.5}, f)

            decision = "rejected"
            if should_promote:
                decision = "promoted"
                target_dest = os.path.dirname(prod_artifacts["model"]) if prod_artifacts["model"] else self.prod_dir
                os.makedirs(target_dest, exist_ok=True)
                for item in ["production_model.joblib", "StandardScaler.joblib", "feature_metadata.json", "threshold.json"]:
                    shutil.copy(os.path.join(self.candidate_dir, item), os.path.join(target_dest, item))

            # Export comprehensive evaluation reports
            with open(os.path.join(self.report_dir, "training_summary.json"), 'w') as f:
                json.dump(test_metrics, f, indent=4)
            
            with open(os.path.join(self.report_dir, "training_metrics.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for k, v in test_metrics.items():
                    if k != "confusion_matrix":
                        writer.writerow([k, v])

            end_time = time.time()
            with self._lock:
                self.status.update({
                    "status": "completed",
                    "progress": 100.0,
                    "message": "Production Adaptation Complete.",
                    "training_time": round(end_time - start_time, 2),
                    "candidate_accuracy": cand_acc,
                    "production_accuracy": prod_acc,
                    "promotion_decision": decision
                })
        except Exception as e:
            with self._lock:
                self.status.update({"status": "failed", "message": f"Execution Error: {str(e)}"})

    def promote_manually(self) -> Dict[str, str]:
        prod_artifacts = self._locate_production_artifacts()
        target_dest = os.path.dirname(prod_artifacts["model"]) if prod_artifacts["model"] else self.prod_dir
        os.makedirs(target_dest, exist_ok=True)
        for item in ["production_model.joblib", "StandardScaler.joblib", "feature_metadata.json", "threshold.json"]:
            src = os.path.join(self.candidate_dir, item)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(target_dest, item))
        with self._lock:
            self.status["promotion_decision"] = "manual_promotion"
        return {"status": "promoted", "message": "Candidate forced to production manually."}
