import os

class PresentationExporter:
    def __init__(self, assets_dir: str):
        self.out_dir = assets_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def export(self, eval_json: dict, perf_json: dict):
        acc = eval_json['metrics'].get('overall_accuracy', 0) * 100
        fps = perf_json.get('processing_fps_throughput', 0)
        lat = perf_json.get('average_inference_latency_ms', 0)

        exec_summary = f"""# SpectraGuard Executive Summary
**Project:** Physics-Informed Frequency-Domain Camera Integrity Intelligence
**Status:** Scientifically Validated

### Key Performance Indicators
- **Overall Accuracy:** {acc:.2f}%
- **Inference Latency:** {lat} ms per frame
- **Processing Throughput:** {fps} FPS

*The model successfully validates physical camera stream integrity using high-frequency spectral rolloff parameters, bypassing traditional deep-learning latency constraints.*
"""
        with open(os.path.join(self.out_dir, 'executive_summary.md'), 'w') as f:
            f.write(exec_summary)

        poster_summary = f"""# SpectraGuard Poster Reference
## Core Innovation
Transforming raw surveillance pixels into spatial frequency spectrums (2D-FFT) to detect physical camera tampering deterministically.

## Results
Achieved **{acc:.2f}%** accuracy in detecting occlusion, defocus, and spray attacks in real-time ({lat} ms latency).
"""
        with open(os.path.join(self.out_dir, 'poster_summary.md'), 'w') as f:
            f.write(poster_summary)

        demo_summary = f"""# Demo Talking Points
1. **The Problem:** Modern camera tampering is subtle (clear sprays, slight defocus).
2. **The Physics Solution:** We don't look at pixels; we look at frequencies.
3. **The Proof:** Our model processes at {fps} FPS with {acc:.2f}% accuracy natively.
"""
        with open(os.path.join(self.out_dir, 'demo_summary.md'), 'w') as f:
            f.write(demo_summary)
