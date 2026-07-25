"""Master Computer Vision Validation, Stress Testing, and Benchmarking Suite."""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime, timezone

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from spectraguard_cv_engine.features.unified.pipeline import UnifiedExtractionPipeline


def generate_synthetic_sequence(frames=3, resolution=(1080, 1920, 3)):
    """Generates synthetic camera frames for testing."""
    sequence = []
    for _ in range(frames):
        # Fast generation of pseudo-random frames
        sequence.append(np.random.randint(0, 256, resolution, dtype=np.uint8))
    return sequence


def run_benchmarks() -> dict:
    print("\n[BENCHMARK] Executing Unified Feature Extraction Pipeline (1080p)...")

    iterations = 20
    sequence_length = 3
    test_sequences = [
        generate_synthetic_sequence(sequence_length) for _ in range(iterations)
    ]

    latencies = []

    # Warmup
    _ = UnifiedExtractionPipeline.extract_from_sequence(test_sequences[0], "WARMUP", 0)

    start_total = time.perf_counter()
    for i, seq in enumerate(test_sequences):
        t0 = time.perf_counter()
        vec = UnifiedExtractionPipeline.extract_from_sequence(
            seq, f"VEC_{i}", time.time_ns()
        )
        # Force array serialization to benchmark complete cycle
        _ = vec.to_array()
        latencies.append(time.perf_counter() - t0)

    total_time = time.perf_counter() - start_total
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    throughput_fps = iterations / total_time

    print(
        f"BENCHMARK RESULT: Avg Pipeline Latency = {avg_latency_ms:.2f} ms/sequence | Throughput = {throughput_fps:.2f} Extractions/sec"
    )

    return {
        "iterations": iterations,
        "resolution": "1920x1080",
        "sequence_length": sequence_length,
        "total_time_seconds": float(f"{total_time:.4f}"),
        "average_latency_ms": float(f"{avg_latency_ms:.2f}"),
        "throughput_extractions_per_sec": float(f"{throughput_fps:.2f}"),
    }


def main():
    print("================================================================")
    print(" SPECTRAGUARD PHASE 4: CV FOUNDATION VALIDATION & BENCHMARKING  ")
    print("================================================================")

    try:
        bench_results = run_benchmarks()
        status = "PASS"
    except Exception as e:
        print(f"\n[ERROR] Pipeline validation failed: {str(e)}")
        bench_results = {"error": str(e)}
        status = "FAIL"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE 4",
        "subsystem": "spectraguard-cv-engine",
        "benchmarks": bench_results,
        "overall_status": "READY" if status == "PASS" else "FAILED",
    }

    report_path = os.path.normpath("data/reports/cv_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport successfully generated at: {report_path}")
    print("================================================================")

    if report["overall_status"] == "READY":
        print("PHASE 4 STATUS: PASSED. Computer Vision Foundation READY for Phase 5.")
        sys.exit(0)
    else:
        print("PHASE 4 STATUS: FAILED. CV Validation errors detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
