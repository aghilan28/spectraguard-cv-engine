import os
import sys
import json
import base64
import subprocess

# Run extraction
print("Running extraction...", file=sys.stderr)
subprocess.run([sys.executable, "scripts/data/extract_production_features_8d.py"], check=True)

# Run training
print("Running training...", file=sys.stderr)
subprocess.run([sys.executable, "scripts/training/run_production_training_v2.py"], check=True)

# Gather and encode artifacts
print("Gathering artifacts...", file=sys.stderr)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
meta_dir = os.path.join(base_dir, "data", "datasets", "virat", "metadata")
release_dir = os.path.join(base_dir, "data", "models", "releases", "v0.9.0-audit")

files_to_encode = {
    "features_csv": os.path.join(meta_dir, "production_features_8d.csv"),
    "production_model": os.path.join(release_dir, "production_model.joblib"),
    "feature_scaler": os.path.join(release_dir, "feature_scaler.joblib"),
    "raw_model": os.path.join(release_dir, "raw_model.joblib"),
    "threshold": os.path.join(release_dir, "threshold.json"),
    "feature_metadata": os.path.join(release_dir, "feature_metadata.json"),
    "training_manifest": os.path.join(release_dir, "training_manifest.json"),
    "experiment_metadata": os.path.join(release_dir, "experiment_metadata.json")
}

output_data = {}
for name, path in files_to_encode.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            content = f.read()
            if path.endswith(".json"):
                output_data[name] = json.loads(content.decode("utf-8"))
            else:
                output_data[name] = base64.b64encode(content).decode("utf-8")
    else:
        print(f"Warning: {path} not found!", file=sys.stderr)

# Print as JSON to stdout
print(json.dumps(output_data))
