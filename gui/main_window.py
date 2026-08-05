import sys
import os
import cv2
import json
import joblib
import uuid
from datetime import datetime, timezone, timedelta
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QMessageBox
from PyQt6.QtCore import Qt, QTimer
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraGuard CCTV Viewer")
        self.resize(1100, 650)
        
        self.manager = None
        self.frame_counter = 0
        self.last_status_text = "CONNECTED"
        self.last_prob_text = ""
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._process_ui_frame)
        
        # Initialize FeatureExtractor and ML models
        self.extractor = FeatureExtractor() if FeatureExtractor else None
        self.event_service = EventService() if EventService else None
        self.model = None
        self.scaler = None
        self.optimal_threshold = 0.5
        self.feature_order = []
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

    def _save_tamper_snapshot(self, frame, prob, tamper_mode="LENS_COVER"):
        if not self.event_service or frame is None:
            return
        try:
            # Determine severity and rule based on simple checks
            severity = "HIGH"
            rule = tamper_mode
            drift = 0.8
            self.event_service.handle_detection(
                camera_name=self.manager.config.name if self.manager else "GUI_Camera",
                frame=frame,
                prob=prob,
                severity=severity,
                drift=drift,
                rule=rule
            )
        except Exception as e:
            print(f"Failed to generate event snapshot: {e}")

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
                raw_url = f"rtsp://{config.username}:{config.password}@{config.ip_address}:{config.port}"
                if target_brand == CameraBrand.HIKVISION: raw_url += "/Streaming/Channels/101"
                elif target_brand in (CameraBrand.DAHUA, CameraBrand.CP_PLUS): raw_url += "/cam/realmonitor?channel=1&subtype=0"
                elif target_brand == CameraBrand.AXIS: raw_url += "/axis-media/media.amp"
                else: raw_url += config.stream_path
                
                self.manager.rtsp_url = raw_url
                self.manager.connect()
            
            self.control_deck.set_connected_state(True)
            self.frame_counter = 0
            self.update_timer.start(33)
            
        except Exception as e:
            QMessageBox.critical(self, "Connection Failure", f"Error: {str(e)}")
            self._handle_disconnect()

    def _process_ui_frame(self):
        if not self.manager:
            return

        frame = self.manager.get_latest_frame()
        self.frame_counter += 1
        
        if frame is not None:
            self.video_display.update_frame(frame)
            h, w = frame.shape[:2]
            res_str = f"{w}x{h}"
            
            # Predict every 15 frames to prevent UI lag (approx 2 FPS)
            if self.frame_counter % 15 == 0 and self.extractor and self.model and self.scaler and self.feature_order:
                try:
                    feats = self.extractor.extract(frame)
                    if feats:
                        feat_vector = [feats.get(f, 0.0) for f in self.feature_order]
                        feat_scaled = self.scaler.transform([feat_vector])
                        prob = float(self.model.predict_proba(feat_scaled)[0][1])
                        
                        print(f"[GUI Predict] Features: {feat_vector}")
                        print(f"[GUI Predict] Scaled: {feat_scaled[0].tolist()}")
                        print(f"[GUI Predict] Prob: {prob:.4f} | Thresh: {self.optimal_threshold:.4f}")
                        
                        self.last_prob_text = f" | Prob: {prob:.2f} (Thresh: {self.optimal_threshold:.2f})"
                        
                        if prob >= self.optimal_threshold:
                            lap = feats.get("laplacian_variance", 0.0)
                            edge = feats.get("edge_density", 0.0)
                            if lap < 15.0:
                                tamper_mode = "HAND_COVER"
                            elif lap < 350.0 and edge < 0.05:
                                tamper_mode = "PAPER_COVER"
                            elif lap < 600.0:
                                tamper_mode = "HALF_COVER / BLUR"
                            else:
                                tamper_mode = "LENS_COVER"
                                
                            self.last_status_text = f"⚠️ TAMPER DETECTED! ({tamper_mode}) ⚠️"
                            self.telemetry_label.setStyleSheet("font-family: Consolas, monospace; padding: 5px; color: #ff3333; font-weight: bold; background-color: #330000;")
                            print(f"[ALERT] CAM TAMPER DETECTED! Mode: {tamper_mode} | Probability: {prob:.4f}")
                            self._save_tamper_snapshot(frame, prob, tamper_mode)
                        else:
                            self.last_status_text = "NORMAL"
                            self.telemetry_label.setStyleSheet("font-family: Consolas, monospace; padding: 5px; color: #33ff33;")
                except Exception as e:
                    print(f"Error in GUI feature extraction/prediction: {e}")
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
            self.telemetry_label.setStyleSheet("font-family: Consolas, monospace; padding: 5px; color: #ff9900;")
        else:
            status_text = self.last_status_text
            
        self.telemetry_label.setText(
            f"Status: {status_text} | FPS: {self.manager.get_fps():.2f} | Resolution: {res_str} | Uptime: {self.manager.get_uptime():.1f}s{self.last_prob_text} | Time: {current_time}"
        )

    def _handle_disconnect(self):
        self.update_timer.stop()
        if self.manager:
            self.manager.disconnect()
            self.manager = None
            
        self.video_display.clear_frame()
        self.control_deck.set_connected_state(False)
        self.telemetry_label.setStyleSheet("font-family: Consolas, monospace; padding: 5px; color: #aaa;")
        self.telemetry_label.setText("Status: DISCONNECTED | FPS: 0.00 | Resolution: N/A | Uptime: 0.0s | Time: --")

    def closeEvent(self, event):
        self._handle_disconnect()
        super().closeEvent(event)
