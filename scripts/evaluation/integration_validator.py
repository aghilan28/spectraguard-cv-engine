import os
import sys
import json

def run_audit():
    print("Initiating Scientific Validation Integrity Check...")
    perf_report_path = "data/datasets/virat/reports/performance_report.json"
    eval_pipe_path = "scripts/evaluation/evaluation_pipeline.py"
    
    if not os.path.exists(perf_report_path):
        print("ERROR: Performance report missing. Pipeline was not run.")
        sys.exit(1)
        
    with open(perf_report_path, "r") as f:
        data = json.load(f)
        
    # Validation Rule 1: Verify placeholder bypass via runtime mapping flags
    if not data.get("inference_stub_bypass_verified", False):
        print("FAIL: The execution logic is still routed through the temporary mock model engine.")
        sys.exit(1)
        
    # Validation Rule 2: Ensure actual extraction time telemetry has been registered
    if data.get("total_feature_extraction_time_seconds", 0.0) <= 0.0:
        print("FAIL: FFT process recorded zero execution cycles. Feature extraction was bypassed.")
        sys.exit(1)
        
    # Validation Rule 3: Inspect pipeline code structure for dead mock methods
    with open(eval_pipe_path, "r") as f:
        code = f.read()
    if "ExistingSpectraGuardEngineStub" in code:
        print("FAIL: Stray code references to the mock framework engine still exist inside orchestrator.")
        sys.exit(1)
        
    print(f"SUCCESS: Scientific validation passed. Average Inference Latency: {data.get('average_inference_latency_ms')} ms per video target.")

if __name__ == "__main__":
    run_audit()
