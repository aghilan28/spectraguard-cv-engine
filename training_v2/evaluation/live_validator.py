import os
import time
import joblib
import json
import numpy as np
import cv2
import psutil
from src.preprocessing.pipeline import PreprocessingPipeline

class LiveValidator:
    def __init__(self, prod_model_dir, cand_model_dir, reports_dir):
        self.prod_model_dir = prod_model_dir
        self.cand_model_dir = cand_model_dir
        self.reports_dir = reports_dir
        self.pipeline = PreprocessingPipeline()
        self.feature_order = [
            "fft_low_ratio",
            "fft_mid_ratio",
            "fft_high_ratio",
            "log_total_energy",
            "laplacian_variance",
            "edge_density",
            "shannon_entropy",
            "temporal_difference"
        ]

    def run(self, source=0, max_frames=50):
        print(f"[LiveValidator] Initializing models for concurrent benchmarking on source {source}...")
        
        # Load Production
        prod_model_path = os.path.join(self.prod_model_dir, "production_model.joblib")
        prod_scaler_path = os.path.join(self.prod_model_dir, "feature_scaler.joblib")
        if not os.path.exists(prod_scaler_path):
            prod_scaler_path = os.path.join(self.prod_model_dir, "scaler.joblib")
        
        # Load Candidate
        cand_model_path = os.path.join(self.cand_model_dir, "production_model.joblib")
        cand_scaler_path = os.path.join(self.cand_model_dir, "feature_scaler.joblib")

        if not (os.path.exists(prod_model_path) and os.path.exists(cand_model_path)):
            print("[LiveValidator] Models missing. Skipping benchmark.")
            return

        prod_model = joblib.load(prod_model_path)
        prod_scaler = joblib.load(prod_scaler_path)
        cand_model = joblib.load(cand_model_path)
        cand_scaler = joblib.load(cand_scaler_path)

        # Get production feature names
        prod_meta_path = os.path.join(self.prod_model_dir, "feature_metadata.json")
        prod_feature_names = self.feature_order
        if os.path.exists(prod_meta_path):
            with open(prod_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                prod_feature_names = meta.get("feature_names") or meta.get("feature_order")

        # Get candidate feature names
        cand_meta_path = os.path.join(self.cand_model_dir, "feature_metadata.json")
        cand_feature_names = self.feature_order
        if os.path.exists(cand_meta_path):
            with open(cand_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                cand_feature_names = meta.get("feature_names") or meta.get("feature_order")


        # Open video capture (webcam or synthetic)
        cap = None
        if source is not None:
            if isinstance(source, int) or os.path.exists(source):
                cap = cv2.VideoCapture(source)

            
        frame_history = []
        latencies_prod = []
        latencies_cand = []
        fps_start = time.perf_counter()
        
        process = psutil.Process(os.getpid())
        cpu_usage = []
        memory_usage = []

        print(f"[LiveValidator] Running benchmark for {max_frames} frames...")
        for i in range(max_frames):
            frame = None
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    frame = None
            
            # If no physical frame, use a synthetic frame
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Benchmarking Frame {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            frame_history.append(frame)
            if len(frame_history) > 15:
                frame_history.pop(0)

            if len(frame_history) == 15:
                # Extract features
                feat_vec = self.pipeline.extract(frame_history)
                feat_dict = feat_vec.to_dict()

                from training_v2.utils.feature_dataframe import build_feature_dataframe

                # Benchmark Production
                t0 = time.perf_counter()
                prod_vector = [feat_dict.get(f, 0.0) for f in prod_feature_names]
                prod_df = build_feature_dataframe(prod_vector, prod_feature_names)
                prod_scaled = prod_scaler.transform(prod_df)
                prod_model.predict_proba(prod_scaled)
                latencies_prod.append((time.perf_counter() - t0) * 1000)

                # Benchmark Candidate
                t0 = time.perf_counter()
                cand_vector = [feat_dict.get(f, 0.0) for f in cand_feature_names]
                cand_df = build_feature_dataframe(cand_vector, cand_feature_names)
                cand_scaled = cand_scaler.transform(cand_df)
                cand_model.predict_proba(cand_scaled)
                latencies_cand.append((time.perf_counter() - t0) * 1000)

                
            cpu_usage.append(psutil.cpu_percent())
            memory_usage.append(process.memory_info().rss / (1024 * 1024))
            time.sleep(0.03) # simulate 30 FPS input delay

        if cap:
            cap.release()

        total_time = time.perf_counter() - fps_start
        fps = max_frames / total_time

        benchmark_report = {
            "fps": fps,
            "total_time_seconds": total_time,
            "production_latency_ms": {
                "mean": float(np.mean(latencies_prod)) if latencies_prod else 0.0,
                "std": float(np.std(latencies_prod)) if latencies_prod else 0.0
            },
            "candidate_latency_ms": {
                "mean": float(np.mean(latencies_cand)) if latencies_cand else 0.0,
                "std": float(np.std(latencies_cand)) if latencies_cand else 0.0
            },
            "resource_utilization": {
                "cpu_percent_mean": float(np.mean(cpu_usage)),
                "memory_rss_mb_mean": float(np.mean(memory_usage))
            }
        }

        os.makedirs(self.reports_dir, exist_ok=True)
        report_path = os.path.join(self.reports_dir, "live_benchmark_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=4)
            
        print(f"[LiveValidator] Live benchmark finished. Report saved to {report_path}")
        return benchmark_report
