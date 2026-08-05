import time
import threading
import sys
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import numpy as np

from backend.config.logging import logger
from backend.stream.camera_manager import CameraManager
from backend.models.calibration import CalibrationSessionModel
from backend.calibration.calibration_storage import CalibrationStorage
from backend.utils.serialization import convert_numpy_types

sys.path.insert(0, os.path.abspath('src'))
try:
    from preprocessing.pipeline import PreprocessingPipeline
except ImportError as e:
    logger.critical(f"Inference dependency bridge failed to load pipeline configurations: {e}")
    raise

class CalibrationEngine:
    """Thread-safe environmental calibration sequence engine that maps raw frame vectors into statistical baselines."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "CalibrationEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.session = CalibrationSessionModel(camera_id="default")
        self.storage = CalibrationStorage()
        self.pipeline = PreprocessingPipeline()
        self._feature_store: Dict[str, List[float]] = {}
        self._cancel_flag = threading.Event()
        self._engine_lock = threading.RLock()
        self._initialized = True

    def initialize_session(self, target_frames: int) -> bool:
        """Configure session metadata maps memory layouts allocations parameters."""
        with self._engine_lock:
            if self.session.status == "running":
                logger.warning("Calibration execution attempt requested while background loops are already running.")
                return False
                
            self.session = CalibrationSessionModel(
                camera_id="default",
                status="running",
                target_frames=target_frames,
                start_time=datetime.now(timezone.utc)
            )
            self._feature_store = {
                "fft_low_ratio": [], "fft_mid_ratio": [], "fft_high_ratio": [],
                "log_total_energy": [], "laplacian_variance": [], "edge_density": [],
                "shannon_entropy": [], "temporal_difference": []
            }
            self._cancel_flag.clear()
            
            worker = threading.Thread(target=self._execution_loop, name="CalibrationAcquisitionWorker", daemon=True)
            worker.start()
            logger.info(f"Calibration session initialized. Target sample frames size defined to: {target_frames}")
            return True

    def _execution_loop(self) -> None:
        """Non-blocking background feature harvesting ingestion worker logic sequence."""
        cam = CameraManager(camera_id="default")
        
        if not cam.is_running():
            self._terminate_with_fault("Camera interface manager tracking loop is not active. Subsystem unavailable.")
            return

        logger.info("Background scene calibration frame gathering capture worker loop running.")
        frame_window_depth = 15
        
        while self.session.processed_frames < self.session.target_frames:
            if self._cancel_flag.is_set():
                logger.info("Calibration sequence termination flags thrown. Halting context operations loops safely.")
                return

            if not cam.is_running():
                self._terminate_with_fault("Camera stream link crashed during background acquisition cycle loops.")
                return

            frames = cam.buffer.frames()
            if len(frames) < frame_window_depth:
                time.sleep(0.1)
                continue

            target_window = frames[-frame_window_depth:]
            
            try:
                feat_vec = self.pipeline.extract(target_window)
                feat_dict = feat_vec.to_dict()
                
                for k in self._feature_store.keys():
                    self._feature_store[k].append(float(feat_dict[k]))
                    
            except Exception as e:
                logger.error(f"In-flight physics frame decomposition calculation failure inside calibration worker loop: {e}")
                time.sleep(0.033)
                continue

            self.session.processed_frames += 1
            self.session.elapsed_seconds = (datetime.now(timezone.utc) - self.session.start_time).total_seconds()
            self.session.progress_percent = round((self.session.processed_frames / self.session.target_frames) * 100, 2)
            
            avg_frame_rate = self.session.processed_frames / max(self.session.elapsed_seconds, 0.001)
            remaining_frames = self.session.target_frames - self.session.processed_frames
            self.session.estimated_remaining_seconds = round(remaining_frames / max(avg_frame_rate, 0.001), 1)

            if self.session.processed_frames % 100 == 0 or self.session.processed_frames == self.session.target_frames:
                logger.info(f"Calibration Ingestion Progress Trace: {self.session.processed_frames}/{self.session.target_frames} frames ({self.session.progress_percent}%)")

            time.sleep(0.033)

        self._finalize_calibration_matrices()

    def _finalize_calibration_matrices(self) -> None:
        """Aggregate compiled variables lists down into structural baseline distribution profiles maps files."""
        with self._engine_lock:
            logger.info("Ingestion limit bounds hit. Aggregating environment physics invariants statistics matrices...")
            try:
                baseline_features = {}
                
                for key, vectors in self._feature_store.items():
                    np_array = np.array(vectors, dtype=np.float64)
                    
                    metrics = {
                        "mean": float(np.mean(np_array)),
                        "std": float(np.std(np_array)),
                        "min": float(np.min(np_array)),
                        "max": float(np.max(np_array)),
                        "median": float(np.median(np_array)),
                        "p05": float(np.percentile(np_array, 5)),
                        "p95": float(np.percentile(np_array, 95)),
                        "variance": float(np.var(np_array)),
                        "sample_count": int(len(vectors))
                    }
                    baseline_features[key] = metrics
                
                baseline_payload = {
                    "camera_id": self.session.camera_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "frame_count": self.session.processed_frames,
                    "features": convert_numpy_types(baseline_features)
                }
                
                self.storage.save(baseline_payload)
                
                self.session.status = "completed"
                self.session.end_time = datetime.now(timezone.utc)
                self.session.estimated_remaining_seconds = 0.0
                logger.info(f"Calibration system sequence completed successfully in {self.session.elapsed_seconds:.2f}s.")
                
            except Exception as e:
                self._terminate_with_fault(f"Statistical validation processing optimization matrix math crash: {e}")

    def _terminate_with_fault(self, message: str) -> None:
        with self._engine_lock:
            self.session.status = "failed"
            self.session.error_message = message
            self.session.end_time = datetime.now(timezone.utc)
            logger.error(f"Calibration session faulted out context parameters: {message}")

    def cancel_active_session(self) -> None:
        with self._engine_lock:
            if self.session.status == "running":
                self._cancel_flag.set()
                self.session.status = "cancelled"
                self.session.end_time = datetime.now(timezone.utc)
                logger.warning("Active environmental processing calibration session explicitly terminated by external command signal.")

    def reset_baseline_profile(self) -> bool:
        with self._engine_lock:
            self.cancel_active_session()
            self.session = CalibrationSessionModel(camera_id="default")
            return self.storage.delete()

engine = CalibrationEngine()
