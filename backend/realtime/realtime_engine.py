import threading
import time
from datetime import datetime, timezone

from backend.config.logging import logger
from backend.realtime.realtime_state import RealtimeState
from backend.realtime.realtime_history import RealtimeHistory
from backend.realtime.realtime_metrics import RealtimeMetrics

# Existing Engines
from backend.stream.camera_manager import CameraManager
from backend.inference.inference_engine import inference_engine
from backend.deviation.deviation_engine import deviation_engine
from backend.tamper.tamper_engine import tamper_engine

class RealtimeEngine:
    """Daemonized orchestration loop integrating all SpectraGuard subsystems."""
    def __init__(self):
        self.state = RealtimeState()
        self.history = RealtimeHistory(max_size=5000)
        self.metrics = RealtimeMetrics()
        self._worker: threading.Thread = None
        self._engine_lock = threading.Lock()
        
        from backend.tamper.classification_engine import TamperClassificationEngine
        self.classification_engine = TamperClassificationEngine()
        
        # State tracking for hysteresis and temporal confirmation
        self.consecutive_tamper_count = 0
        self.is_currently_tampered = False
        self.active_tamper_event_sent = False
        self.last_sent_tamper_type = None

    def start(self) -> bool:
        with self._engine_lock:
            if self.is_running():
                return False
            with self.state._lock:
                self.state.running = True
                self.state.paused = False
                self.state.start_time = datetime.now(timezone.utc)
            
            self._worker = threading.Thread(target=self._orchestration_loop, daemon=True, name="RealtimeEngineWorker")
            self._worker.start()
            logger.info("Realtime Engine started.")
            return True

    def stop(self) -> bool:
        with self._engine_lock:
            with self.state._lock:
                if not self.state.running:
                    return False
                self.state.running = False
        
        if self._worker:
            self._worker.join(timeout=2.0)
            self._worker = None
        logger.info("Realtime Engine stopped.")
        return True

    def pause(self) -> bool:
        with self.state._lock:
            if not self.state.running or self.state.paused:
                return False
            self.state.paused = True
            return True

    def resume(self) -> bool:
        with self.state._lock:
            if not self.state.running or not self.state.paused:
                return False
            self.state.paused = False
            return True

    def is_running(self) -> bool:
        with self.state._lock:
            return self.state.running

    def _orchestration_loop(self):
        cam = CameraManager(camera_id="default")
        
        while True:
            with self.state._lock:
                if not self.state.running:
                    break
                paused = self.state.paused

            if paused:
                time.sleep(0.1)
                continue

            cycle_start = time.perf_counter()

            if not cam.is_running():
                time.sleep(0.5)
                continue

            frames = cam.buffer.frames()
            if len(frames) < 15:
                time.sleep(0.1)
                continue

            try:
                window = frames[-15:]
                
                # Full Pipeline Orchestration
                inf_res = inference_engine.run(window)
                dev_res = deviation_engine.evaluate(inf_res.feature_vector)
                tamper_res = tamper_engine.evaluate(inf_res, dev_res)

                prob = tamper_res.random_forest_probability
                threshold = inf_res.threshold
                margin = 0.10
                exit_threshold = threshold - margin

                # 1. State transition using hysteresis and physical validation
                candidate_type = self.classification_engine.classify(window[-1], window, prob=prob)

                if self.is_currently_tampered:
                    if prob < exit_threshold or candidate_type == "NORMAL":
                        self.is_currently_tampered = False
                        self.consecutive_tamper_count = 0
                        self.active_tamper_event_sent = False
                        self.last_sent_tamper_type = None
                        final_tamper_type = "NORMAL"
                        final_prediction = "NORMAL"
                        final_severity = "LOW"
                    else:
                        self.consecutive_tamper_count += 1
                        final_tamper_type = candidate_type
                        final_prediction = "TAMPER"
                        final_severity = "HIGH" if prob > 0.90 else "MEDIUM"
                else:
                    if prob >= threshold and candidate_type != "NORMAL":
                        self.is_currently_tampered = True
                        self.consecutive_tamper_count = 1
                        final_tamper_type = candidate_type
                        final_prediction = "TAMPER"
                        final_severity = "HIGH" if prob > 0.90 else "MEDIUM"
                    else:
                        self.consecutive_tamper_count = 0
                        self.active_tamper_event_sent = False
                        self.last_sent_tamper_type = None
                        final_tamper_type = "NORMAL"
                        final_prediction = "NORMAL"
                        final_severity = "LOW"

                latency_ms = (time.perf_counter() - cycle_start) * 1000
                current_time = datetime.now(timezone.utc)

                # Calculate metric checks for explainable print
                import cv2
                import numpy as np
                gray_dbg = cv2.cvtColor(window[-1], cv2.COLOR_BGR2GRAY)
                black_ratio_dbg = float(np.sum(gray_dbg < 25) / gray_dbg.size)
                white_ratio_dbg = float(np.sum(gray_dbg > 230) / gray_dbg.size)
                hist_dbg, _ = np.histogram(gray_dbg.ravel(), bins=256, range=(0, 256))
                prob_dist_dbg = hist_dbg / (hist_dbg.sum() + 1e-12)
                prob_dist_dbg = prob_dist_dbg[prob_dist_dbg > 0]
                entropy_dbg = float(-np.sum(prob_dist_dbg * np.log2(prob_dist_dbg + 1e-12))) if len(prob_dist_dbg) > 0 else 0.0
                laplacian_var_dbg = float(cv2.Laplacian(gray_dbg, cv2.CV_64F).var())
                edge_density_dbg = float(np.sum(cv2.Canny(gray_dbg, 50, 150) > 0) / gray_dbg.size)
                
                h_dbg, w_dbg = gray_dbg.shape
                h_grid_dbg, w_grid_dbg = h_dbg // 8, w_dbg // 8
                flat_blocks_dbg = 0
                for r in range(8):
                    for c in range(8):
                        block = gray_dbg[r*h_grid_dbg : (r+1)*h_grid_dbg, c*w_grid_dbg : (c+1)*w_grid_dbg]
                        if np.std(block) < 10.0:
                            flat_blocks_dbg += 1
                flat_ratio_dbg = flat_blocks_dbg / 64.0
                
                t = self.classification_engine.thresholds
                
                entropy_pass = "PASS" if entropy_dbg < t["entropy_limit"] else "FAIL"
                edge_pass = "PASS" if edge_density_dbg < t["edge_density_limit"] else "FAIL"
                blur_pass = "PASS" if laplacian_var_dbg < t["laplacian_blur_limit"] else "FAIL"
                occl_pass = "PASS" if (0.25 <= flat_ratio_dbg <= 0.85) else "FAIL"
                
                logger.info(
                    f"\n=== BACKEND REALTIME ENGINE DEBUG OVERLAY ==="
                    f"\nRF Probability: {prob:.4f} (Threshold: {threshold:.4f})"
                    f"\n  - Entropy: {entropy_dbg:.2f} (Limit < {t['entropy_limit']}) -> {entropy_pass}"
                    f"\n  - Edge Density: {edge_density_dbg:.4f} (Limit < {t['edge_density_limit']}) -> {edge_pass}"
                    f"\n  - Blur (Laplacian Var): {laplacian_var_dbg:.2f} (Limit < {t['laplacian_blur_limit']}) -> {blur_pass}"
                    f"\n  - Occlusion (8x8 Grid Flat): {flat_ratio_dbg:.2f} (Range 0.25-0.85) -> {occl_pass}"
                    f"\n  - Black Ratio: {black_ratio_dbg:.4f} | White Ratio: {white_ratio_dbg:.4f}"
                    f"\nDecision: {final_tamper_type} | Consecutive Count: {self.consecutive_tamper_count}"
                    f"\n=================================\n"
                )

                record = {
                    "timestamp": current_time.isoformat(),
                    "prediction": final_prediction,
                    "probability": prob,
                    "tamper_type": final_tamper_type,
                    "severity": final_severity,
                    "rule": final_tamper_type,
                    "deviation_score": tamper_res.deviation_score,
                    "latency_ms": latency_ms
                }

                # 2. Require 5 consecutive tampered frames before generating ONE user-visible event
                should_trigger_event = (self.is_currently_tampered and 
                                        self.consecutive_tamper_count >= 5 and 
                                        (not self.active_tamper_event_sent or final_tamper_type != self.last_sent_tamper_type))

                if should_trigger_event:
                    self.active_tamper_event_sent = True
                    self.last_sent_tamper_type = final_tamper_type
                    try:
                        from backend.services.event_service import EventService
                        EventService().handle_detection(
                            camera_name=cam.camera_id,
                            frame=window[-1],
                            prob=prob,
                            severity=final_severity,
                            drift=tamper_res.deviation_score,
                            rule=final_tamper_type
                        )
                    except Exception as ev_err:
                        logger.error(f"Event persistence failed: {ev_err}")

                self.history.add(record)
                self.metrics.update(
                    latency=latency_ms,
                    drift=tamper_res.deviation_score,
                    probability=final_prob,
                    tamper_type=final_tamper_type
                )

                with self.state._lock:
                    self.state.processed_count += 1
                    self.state.last_prediction = 1 if final_prediction == "TAMPER" else 0
                    self.state.last_probability = final_prob
                    self.state.last_drift_score = tamper_res.deviation_score
                    self.state.last_tamper_type = final_tamper_type
                    self.state.last_timestamp = current_time

            except Exception as e:
                logger.error(f"Realtime pipeline execution failure: {e}")

            # Enforce 500ms cycle minimum
            elapsed = time.perf_counter() - cycle_start
            sleep_time = max(0.0, 0.5 - elapsed)
            time.sleep(sleep_time)

realtime_engine = RealtimeEngine()
