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

                latency_ms = (time.perf_counter() - cycle_start) * 1000
                current_time = datetime.now(timezone.utc)

                pred_str = "TAMPER" if tamper_res.random_forest_prediction == 1 else "NORMAL"
                record = {
                    "timestamp": current_time.isoformat(),
                    "prediction": pred_str,
                    "probability": tamper_res.random_forest_probability,
                    "tamper_type": tamper_res.tamper_type,
                    "severity": tamper_res.severity,
                    "rule": tamper_res.tamper_type,
                    "deviation_score": tamper_res.deviation_score,
                    "latency_ms": latency_ms
                }

                # Persist snapshot event on tamper prediction
                if tamper_res.random_forest_prediction == 1:
                    try:
                        from backend.services.event_service import EventService
                        EventService().handle_detection(
                            camera_name=cam.camera_id,
                            frame=window[-1],
                            prob=tamper_res.random_forest_probability,
                            severity=tamper_res.severity,
                            drift=tamper_res.deviation_score,
                            rule=tamper_res.tamper_type
                        )
                    except Exception as ev_err:
                        logger.error(f"Event persistence failed: {ev_err}")

                self.history.add(record)
                self.metrics.update(
                    latency=latency_ms,
                    drift=tamper_res.deviation_score,
                    probability=tamper_res.random_forest_probability,
                    tamper_type=tamper_res.tamper_type
                )

                with self.state._lock:
                    self.state.processed_count += 1
                    self.state.last_prediction = tamper_res.random_forest_prediction
                    self.state.last_probability = tamper_res.random_forest_probability
                    self.state.last_drift_score = tamper_res.deviation_score
                    self.state.last_tamper_type = tamper_res.tamper_type
                    self.state.last_timestamp = current_time

            except Exception as e:
                logger.error(f"Realtime pipeline execution failure: {e}")

            # Enforce 500ms cycle minimum
            elapsed = time.perf_counter() - cycle_start
            sleep_time = max(0.0, 0.5 - elapsed)
            time.sleep(sleep_time)

realtime_engine = RealtimeEngine()
