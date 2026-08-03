# SpectraGuard M0.3 Portable Release

This repository is in a complete and portable state.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - On Windows: `.venv\Scripts\activate`
   - On Linux/macOS: `source .venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If running the live camera demo, ensure you install `opencv-python` (with GUI support) rather than `opencv-python-headless`.*

## How to Run Training

To retrain the 8D production model, run:
```bash
python scripts/training/run_production_training_v2.py
```
This script will read the dataset `data/datasets/virat/metadata/production_features_8d.csv`, run GroupKFold cross-validation, train a calibrated classifier, optimize the decision threshold, and save all artifacts under `data/models/releases/v0.9.0-audit`.

## How to Run Webcam Demo

To start the real-time inference webcam demo:
```bash
python live_camera_demo.py
```
If you do not have a physical webcam attached, you can run in simulation mode:
```bash
python live_camera_demo.py --simulate
```

## How to Load Model

To load and use the production model in Python:
```python
import joblib
import json
import numpy as np

# Load artifacts
model = joblib.load("data/models/releases/v0.9.0-audit/production_model.joblib")
scaler = joblib.load("data/models/releases/v0.9.0-audit/feature_scaler.joblib")

with open("data/models/releases/v0.9.0-audit/threshold.json", "r") as f:
    threshold_info = json.load(f)
threshold = threshold_info["optimal_threshold"]

# Run inference on an 8D feature vector
# raw_features structure: [fft_low_ratio, fft_mid_ratio, fft_high_ratio, log_total_energy, laplacian_variance, edge_density, shannon_entropy, temporal_difference]
raw_features = np.array([0.85, 0.10, 0.05, 12.0, 500.0, 0.15, 6.5, 1.5]).reshape(1, -1)
scaled_features = scaler.transform(raw_features)
prob = model.predict_proba(scaled_features)[0, 1]
is_tampered = prob >= threshold

print(f"Tampering Probability: {prob:.4f}")
print(f"Is Tampered: {is_tampered}")
```
