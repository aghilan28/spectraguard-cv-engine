"""Performance Benchmark Suite for SpectraGuard Inference Pipeline."""

import os
import sys
import time
import psutil
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath("src"))
from spectraguard_cv_engine.inference.pipeline import VideoInferencePipeline
from spectraguard_cv_engine.inference.loader import ProductionModelLoader


def run_benchmark(iterations: int = 50):
    print("==================================================")
    print("    SPECTRAGUARD INFERENCE PERFORMANCE PROFILE    ")
    print("==================================================")

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # Initialize CPU monitor
    base_mem = process.memory_info().rss / (1024 * 1024)

    # 1. Measure Model Loading Time & Memory Footprint
    print("\n[1] Artifact Initialization (Singleton Loader)")
    start_load = time.time()
    ProductionModelLoader.force_reload()  # Force clear for accurate timing
    _ = ProductionModelLoader()
    load_time_ms = (time.time() - start_load) * 1000
    mem_after_load = process.memory_info().rss / (1024 * 1024)
    model_mem_footprint = mem_after_load - base_mem

    print(f" -> Loader Instantiation Time : {load_time_ms:.2f} ms")
    print(f" -> ML Model Memory Footprint : {model_mem_footprint:.2f} MB")

    # Initialize pipeline
    pipeline = VideoInferencePipeline()

    # 2. Measure Inference Latency & FPS via Memory Mocking
    print(f"\n[2] Inference Pipeline (Evaluating {iterations} iterations)")
    latencies_ms = []

    # Mock VideoCapture to isolate CPU pipeline performance from Disk I/O limits
    with patch(
        "spectraguard_cv_engine.inference.pipeline.cv2.VideoCapture"
    ) as mock_cap:
        for _ in range(iterations):
            instance = MagicMock()
            instance.isOpened.return_value = True
            # Simulate 3 continuous frames (480x640 BGR)
            instance.read.side_effect = [
                (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
                for _ in range(3)
            ] + [(False, None)]
            mock_cap.return_value = instance

            start_inf = time.time()
            pipeline.process_video("mock_memory_stream.mp4", max_frames=3)
            latencies_ms.append((time.time() - start_inf) * 1000)

    avg_latency = np.mean(latencies_ms)
    p95_latency = np.percentile(latencies_ms, 95)

    # FPS Calculation: (3 frames per sequence * iterations) / total seconds
    total_frames = 3 * iterations
    total_time_sec = sum(latencies_ms) / 1000
    fps = total_frames / total_time_sec

    print(f" -> Average Pipeline Latency  : {avg_latency:.2f} ms")
    print(f" -> P95 Pipeline Latency      : {p95_latency:.2f} ms")
    print(f" -> Processed Throughput      : {fps:.2f} FPS")

    # 3. CPU / Environment Profile
    print("\n[3] System Utilization")
    cpu_usage = process.cpu_percent(interval=None)
    peak_mem = process.memory_info().rss / (1024 * 1024)
    print(f" -> Peak CPU Utilization      : {cpu_usage:.2f} %")
    print(f" -> Peak Memory Utilization   : {peak_mem:.2f} MB")
    print("==================================================\n")


if __name__ == "__main__":
    run_benchmark(iterations=50)
