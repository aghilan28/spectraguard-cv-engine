import sys
import os
import cv2
import json
import joblib
import time
import numpy as np
import pandas as pd
import traceback
from datetime import datetime, timezone, timedelta
from collections import deque

from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QLabel, QFrame, QMessageBox, QListWidget, QPushButton)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from gui.login_panel import LoginPanel
from gui.video_widget import VideoWidget
from camera.camera_config import CameraConfig
from camera.camera_manager import CameraManager
from camera.rtsp_builder import CameraBrand, RTSPBuilder
from camera.exceptions import CameraError

# Ensure root path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from backend.services.feature_extractor import FeatureExtractor
    from backend.services.event_service import EventService
except ImportError:
    FeatureExtractor = None
    EventService = None


class PredictionThread(QThread):
    """
    Runs in the background, consuming frames from the CameraManager frame buffer,
    running feature extraction and model inference, applying decision fusion,
    and triggering asynchronous event storage.
    """
    prediction_ready = pyqtSignal(dict)

    def __init__(self, manager, model, scaler, feature_order, optimal_threshold):
        super().__init__()
        self.manager = manager
        self.model = model
        self.scaler = scaler
        self.feature_order = feature_order
        self.optimal_threshold = optimal_threshold
        
        self.running = True
        
        from src.preprocessing.pipeline import PreprocessingPipeline
        self.pipeline = PreprocessingPipeline()
        self.frame_history = []
        
        from backend.tamper.classification_engine import TamperClassificationEngine
        self.classification_engine = TamperClassificationEngine()
        
        self.event_service = EventService() if EventService else None

    def run(self):
        last_prediction_time = 0.0
        while self.running:
            if not self.manager or not self.manager.is_connected():
                self.msleep(100)
                continue
                
            frame = self.manager.get_latest_frame()
            if frame is None:
                self.msleep(30)
                continue
            
            # Maintain the 15-frame history for temporal/ORB rules
            if len(self.frame_history) == 0 or not np.array_equal(frame, self.frame_history[-1]):
                self.frame_history.append(frame.copy())
                if len(self.frame_history) > 15:
                    self.frame_history.pop(0)

            now = time.time()
            # Predict every 300ms to remain highly responsive without overloading CPU
            if len(self.frame_history) == 15 and (now - last_prediction_time) >= 0.3:
                last_prediction_time = now
                cycle_start = time.perf_counter()
                try:
                    # 1. Physics Feature Extraction
                    feat_vec = self.pipeline.extract(self.frame_history)
                    feat_dict = feat_vec.to_dict()
                    
                    # 2. Strict feature ordering DataFrame creation
                    feat_vector = [feat_dict.get(f, 0.0) for f in self.feature_order]
                    df = pd.DataFrame([feat_vector], columns=self.feature_order)
                    
                    # 3. Scaler transform
                    feat_scaled = self.scaler.transform(df)
                    
                    # 4. ML Model Probability
                    prob = float(self.model.predict_proba(feat_scaled)[0][1])
                    
                    # 5. Deterministic Rules
                    tamper_type = self.classification_engine.classify(frame, self.frame_history, prob=prob)
                    
                    # Decision Fusion: triggers TAMPER if RF model agrees OR any custom rule triggers
                    is_tamper = (prob >= self.optimal_threshold) or (tamper_type != "NORMAL")
                    
                    final_prediction = "TAMPERED" if is_tamper else "NORMAL"
                    final_tamper_type = tamper_type if is_tamper else "NORMAL"
                    
                    if is_tamper and final_tamper_type in ["NORMAL", "UNKNOWN_ANOMALY"]:
                        if prob < 0.70:
                            # Downgrade low-probability alerts with no physical rule triggers to NORMAL
                            final_prediction = "NORMAL"
                            final_tamper_type = "NORMAL"
                            is_tamper = False
                        else:
                            lap_var = feat_dict.get("laplacian_variance", 999.0)
                            ent = feat_dict.get("shannon_entropy", 8.0)
                            edg = feat_dict.get("edge_density", 1.0)
                            mean_val = float(np.mean(frame))
                            white_ratio = float(np.sum(frame > 230) / frame.size)
                            
                            if lap_var < 35.0:
                                final_tamper_type = "BLUR_ATTACK"
                            elif lap_var < 95.0:
                                final_tamper_type = "DEFOCUS"
                            elif mean_val < 30.0:
                                final_tamper_type = "DARKNESS_ATTACK"
                            elif mean_val > 180.0 or white_ratio > 0.15:
                                final_tamper_type = "BRIGHTNESS_ATTACK"
                            elif ent < 3.8 or edg < 0.05:
                                final_tamper_type = "FULL_LENS_COVER"
                            else:
                                final_tamper_type = "PARTIAL_LENS_COVER"
                    
                    # 6. Event persisting (non-blocking, queues background writes)
                    screenshot_saved = "No"
                    sidebar_updated = "No"
                    if is_tamper and self.event_service:
                        res = self.event_service.handle_detection(
                            camera_name=self.manager.config.name,
                            frame=frame,
                            prob=prob,
                            severity="HIGH" if prob > 0.9 else "MEDIUM",
                            drift=float(prob),
                            rule=final_tamper_type
                        )
                        if res is not None:
                            screenshot_saved = "Yes"
                            sidebar_updated = "Yes"

                    latency_ms = (time.perf_counter() - cycle_start) * 1000
                    
                    # Structure logging output
                    print(
                        f"[Frame {self.manager.get_frame_count()}] | "
                        f"Feats: {list(np.round(feat_vector, 4))} | "
                        f"Scaled: {list(np.round(feat_scaled[0], 4))} | "
                        f"Prob: {prob:.4f} | "
                        f"Pred: {final_prediction} | "
                        f"Tamper: {final_tamper_type} | "
                        f"Latency: {latency_ms:.2f}ms | "
                        f"Saved: {screenshot_saved} | "
                        f"Sidebar: {sidebar_updated}"
                    )

                    result = {
                        "prediction": final_prediction,
                        "tamper_type": final_tamper_type,
                        "probability": prob,
                        "confidence": (prob if is_tamper else (1.0 - prob)) * 100.0,
                        "frame": frame.copy()
                    }
                    self.prediction_ready.emit(result)
                    
                except Exception as e:
                    print(f"[PredictionThread] Pipeline crash: {e}")
                    traceback.print_exc()
            
            self.msleep(50)

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraGuard CCTV Viewer")
        self.resize(1100, 650)
        
        self.manager = None
        self.predict_thread = None
        self.frame_counter = 0
        
        self.last_status_text = "CONNECTED"
        self.last_prob_text = ""
        
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._process_ui_frame)
        
        self.event_service = EventService() if EventService else None
        self.model = None
        self.scaler = None
        self.optimal_threshold = 0.5
        self.feature_order = []
        
        # In-memory history cache
        self.recent_events_deque = deque(maxlen=50)
        
        self._load_ml_model()
        self.init_ui()

    def _load_ml_model(self):
        try:
            model_dir = "data/models/latest"
            model_path = os.path.join(model_dir, "production_model.joblib")
            scaler_path = os.path.join(model_dir, "feature_scaler.joblib")
            if not os.path.exists(scaler_path):
                scaler_path = os.path.join(model_dir, "scaler.joblib")
            metadata_path = os.path.join(model_dir, "feature_metadata.json")
            threshold_path = os.path.join(model_dir, "threshold.json")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.feature_order = meta.get("feature_names") or meta.get("feature_order") or []
                
                if os.path.exists(threshold_path):
                    with open(threshold_path, "r", encoding="utf-8") as f:
                        t_data = json.load(f)
                        self.optimal_threshold = float(t_data.get("optimal_threshold", t_data.get("bounds", 0.5)))
                
                print("GUI successfully loaded active ML model and scaler!")
            else:
                print("WARNING: GUI could not find active ML model artifacts. Tampering detection disabled.")
        except Exception as e:
            print(f"Error loading ML model in GUI: {e}")

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.control_deck = LoginPanel()
        self.control_deck.connect_requested.connect(self._handle_connect)
        self.control_deck.disconnect_requested.connect(self._handle_disconnect)
        main_layout.addWidget(self.control_deck, stretch=1)

        video_wrapper = QVBoxLayout()
        self.video_display = VideoWidget()
        video_wrapper.addWidget(self.video_display, stretch=5)
        
        self.telemetry_label = QLabel("Status: DISCONNECTED | FPS: 0.00 | Resolution: N/A | Uptime: 0.0s | Time: --")
        self.telemetry_label.setStyleSheet("font-family: Consolas, monospace; padding: 5px; color: #aaa;")
        video_wrapper.addWidget(self.telemetry_label, stretch=0)
        
        main_layout.addLayout(video_wrapper, stretch=3)

        # Right-hand Sidebar Layout
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_widget.setFixedWidth(280)
        sidebar_widget.setStyleSheet("background-color: #1a1a1a; border-left: 1px solid #333;")
        
        lbl_snap = QLabel("LATEST SNAPSHOT")
        lbl_snap.setStyleSheet("font-weight: bold; color: #ff3333; font-family: Arial; font-size: 13px;")
        sidebar_layout.addWidget(lbl_snap)
        
        self.screenshot_label = QLabel()
        self.screenshot_label.setFixedSize(260, 150)
        self.screenshot_label.setStyleSheet("background-color: #0d0d0d; border: 1px solid #444;")
        self.screenshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screenshot_label.setText("No snapshot yet")
        sidebar_layout.addWidget(self.screenshot_label)
        
        lbl_events = QLabel("RECENT EVENTS HISTORY")
        lbl_events.setStyleSheet("font-weight: bold; color: #aaa; font-family: Arial; font-size: 13px; margin-top: 15px;")
        sidebar_layout.addWidget(lbl_events)
        
        self.event_list = QListWidget()
        self.event_list.setStyleSheet("""
            QListWidget {
                background-color: #0d0d0d;
                border: 1px solid #333;
                color: #ddd;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
            QListWidget::item {
                border-bottom: 1px solid #222;
                padding: 4px;
            }
        """)
        self.event_list.itemClicked.connect(self._handle_event_click)
        sidebar_layout.addWidget(self.event_list)

        btn_open = QPushButton("Open Snapshots Folder")
        btn_open.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                border: 1px solid #444;
                color: #ddd;
                padding: 6px;
                font-family: Arial;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #383838;
            }
        """)
        btn_open.clicked.connect(self._open_snapshots_folder)
        sidebar_layout.addWidget(btn_open)
        
        main_layout.addWidget(sidebar_widget, stretch=1)

    def _handle_connect(self, payload: dict):
        try:
            config = CameraConfig(
                name=payload["name"],
                ip_address=payload["ip_address"],
                port=payload["port"],
                username=payload["username"],
                password=payload["password"]
            )
            
            brand_map = {
                "generic": CameraBrand.GENERIC, "hikvision": CameraBrand.HIKVISION,
                "dahua": CameraBrand.DAHUA, "cp plus": CameraBrand.CP_PLUS, "axis": CameraBrand.AXIS
            }
            target_brand = brand_map.get(payload["vendor"], CameraBrand.GENERIC)
            
            self.manager = CameraManager(config, target_brand)
            
            try:
                self.manager.connect()
            except Exception:
                if not config.ip_address.strip().isdigit():
                    raw_url = f"rtsp://{config.username}:{config.password}@{config.ip_address}:{config.port}"
                    if target_brand == CameraBrand.HIKVISION: raw_url += "/Streaming/Channels/101"
                    elif target_brand in (CameraBrand.DAHUA, CameraBrand.CP_PLUS): raw_url += "/cam/realmonitor?channel=1&subtype=0"
                    elif target_brand == CameraBrand.AXIS: raw_url += "/axis-media/media.amp"
                    else: raw_url += config.stream_path
                    
                    self.manager.rtsp_url = raw_url
                self.manager.connect()
            
            self.control_deck.set_connected_state(True)
            self.frame_counter = 0
            self.last_status_text = "NORMAL"
            self.last_prob_text = " | Confidence: 100.0%"
            
            # Start background Prediction Thread
            if self.model and self.scaler and self.feature_order:
                self.predict_thread = PredictionThread(
                    self.manager, self.model, self.scaler, self.feature_order, self.optimal_threshold
                )
                self.predict_thread.prediction_ready.connect(self._handle_prediction_update)
                self.predict_thread.start()

            self.update_timer.start(33)
            
        except Exception as e:
            QMessageBox.critical(self, "Connection Failure", f"Error: {str(e)}")
            self._handle_disconnect()

    def _handle_prediction_update(self, result: dict):
        """Processes outputs from the background PredictionThread."""
        pred = result["prediction"]
        tamper_type = result["tamper_type"]
        prob = result["probability"]
        conf = result["confidence"]
        frame = result["frame"]

        self.last_status_text = f"{pred} ({tamper_type})"
        self.last_prob_text = f" | Confidence: {conf:.1f}% (Prob: {prob:.2f})"

        if pred == "TAMPERED":
            self.telemetry_label.setStyleSheet("font-family: Consolas, monospace; padding: 5px; color: #ff3333; font-weight: bold; background-color: #330000;")
            
            # Add to local events list and update sidebar UI
            ist_tz = timezone(timedelta(hours=5, minutes=30))
            ts = datetime.now(ist_tz).strftime("%H:%M:%S")
            evt_str = f"[{ts}] {tamper_type} (Conf: {conf:.1f}%)"
            
            if len(self.recent_events_deque) == 0 or self.recent_events_deque[-1] != evt_str:
                self.recent_events_deque.append(evt_str)
                self.event_list.clear()
                # Display in reverse chronological order (newest on top)
                for item in reversed(self.recent_events_deque):
                    self.event_list.addItem(item)
            
            # Update snapshot thumbnail preview
            try:
                h_snap, w_snap = frame.shape[:2]
                rgb_snap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_snap.data, w_snap, h_snap, 3 * w_snap, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(q_img)
                self.screenshot_label.setPixmap(
                    pix.scaled(260, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                )
            except Exception as err:
                print(f"Failed to update thumbnail: {err}")
        else:
            self.telemetry_label.setStyleSheet("font-family: Consolas, monospace; padding: 5px; color: #33ff33;")

    def _process_ui_frame(self):
        """Runs in the main thread on QTimer ticks, performing fast UI updates."""
        if not self.manager:
            return

        frame = self.manager.get_latest_frame()
        self.frame_counter += 1
        
        if frame is not None:
            # Display frame immediately (already rotated in CameraManager worker thread)
            self.video_display.update_frame(frame)
            h, w = frame.shape[:2]
            res_str = f"{w}x{h}"
        else:
            res_str = "N/A"

        ist_tz = timezone(timedelta(hours=5, minutes=30))
        current_time = datetime.now(ist_tz).strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine status text from connection state
        if not self.manager.is_connected():
            if self.manager.is_reconnecting:
                status_text = f"RECONNECTING (Attempt {self.manager.reconnect_attempts})..."
            else:
                status_text = "LOSS OF SIGNAL"
            self.telemetry_label.setStyleSheet("font-family: Consolas, monospace; padding: 5px; color: #ff3333; font-weight: bold; background-color: #330000;")
        else:
            status_text = self.last_status_text
            
        self.telemetry_label.setText(
            f"Status: {status_text} | FPS: {self.manager.get_fps():.2f} | Resolution: {res_str} | Uptime: {self.manager.get_uptime():.1f}s{self.last_prob_text} | Time: {current_time}"
        )

    def _handle_disconnect(self):
        self.update_timer.stop()
        
        # Safely shut down background PredictThread
        if self.predict_thread:
            self.predict_thread.stop()
            self.predict_thread = None

        if self.manager:
            self.manager.disconnect()
            self.manager = None
            
        self.video_display.clear_frame()
        self.control_deck.set_connected_state(False)
        self.screenshot_label.clear()
        self.screenshot_label.setText("No snapshot yet")
        self.event_list.clear()
        self.recent_events_deque.clear()
        self.telemetry_label.setStyleSheet("font-family: Consolas, monospace; padding: 5px; color: #aaa;")
        self.telemetry_label.setText("Status: DISCONNECTED | FPS: 0.00 | Resolution: N/A | Uptime: 0.0s | Time: --")

    def _handle_event_click(self, item):
        try:
            if not self.event_service:
                return
            row = self.event_list.row(item)
            events = self.event_service.get_history()
            if not events:
                return
            idx = len(events) - 1 - row
            if 0 <= idx < len(events):
                evt = events[idx]
                snap_path = evt.get("snapshot_path")
                if snap_path and os.path.exists(snap_path):
                    pix = QPixmap(snap_path)
                    self.screenshot_label.setPixmap(
                        pix.scaled(260, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    )
        except Exception as e:
            print(f"Failed to display event snapshot: {e}")

    def _open_snapshots_folder(self):
        try:
            os.makedirs("storage/snapshots", exist_ok=True)
            os.startfile(os.path.abspath("storage/snapshots"))
        except Exception as e:
            print(f"Failed to open snapshots directory: {e}")

    def closeEvent(self, event):
        self._handle_disconnect()
        super().closeEvent(event)
