import os
import binascii
import subprocess

print("Running extraction...")
subprocess.run(["python", "scripts/data/extract_production_features_8d.py"], check=True)

print("Running training...")
subprocess.run(["python", "scripts/training/run_production_training_v2.py"], check=True)

print("Reading and converting to hex...")
d = os.path.join("data", "models", "releases", "v0.9.0-audit")
files = [
    "production_model.joblib",
    "feature_scaler.joblib",
    "raw_model.joblib",
    "threshold.json",
    "feature_metadata.json",
    "training_manifest.json",
    "experiment_metadata.json"
]

out = []
out.append("HEX_DATA = {")
for f in files:
    path = os.path.join(d, f)
    with open(path, "rb") as file_in:
        hex_data = binascii.hexlify(file_in.read()).decode("utf-8")
        out.append(f'  "{f}": "{hex_data}",')
out.append("}")

# Include features CSV too
features_path = os.path.join("data", "datasets", "virat", "metadata", "production_features_8d.csv")
if os.path.exists(features_path):
    with open(features_path, "rb") as file_in:
        hex_data = binascii.hexlify(file_in.read()).decode("utf-8")
        out.append(f'FEATURES_CSV_HEX = "{hex_data}"')

with open("temp_hex.py", "w") as file_out:
    file_out.write("\n".join(out))

print("temp_hex.py written successfully!")
