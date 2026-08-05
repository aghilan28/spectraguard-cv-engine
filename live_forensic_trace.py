import cv2
import json
import joblib
import numpy as np
import time

from pathlib import Path

from src.preprocessing.pipeline import PreprocessingPipeline

ROOT = Path(".")

MODEL = ROOT / "data/models/latest/production_model.joblib"
SCALER = ROOT / "data/models/latest/feature_scaler.joblib"
META = ROOT / "data/models/latest/feature_metadata.json"
THRESH = ROOT / "data/models/latest/threshold.json"

print("="*80)
print("SPECTRAGUARD LIVE FORENSIC TRACE")
print("="*80)

model = joblib.load(MODEL)
scaler = joblib.load(SCALER)

with open(META,"r") as f:
    meta=json.load(f)

with open(THRESH,"r") as f:
    threshold=json.load(f)["optimal_threshold"]

feature_names=meta["feature_names"]

print()
print("Model:",type(model))
print("Scaler:",type(scaler))
print("Threshold:",threshold)
print()

pipe=PreprocessingPipeline()

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not opened")

frames=[]

print("Collecting 15 frames...")
while len(frames)<15:

    ok,frame=cap.read()

    if not ok:
        continue

    frames.append(frame.copy())

    cv2.imshow("Live Feed",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()

print()
print("="*80)
print("FEATURE EXTRACTION")
print("="*80)

features=pipe.extract(frames)

if isinstance(features,dict):

    raw=np.array([features[x] for x in feature_names],dtype=np.float32)

else:

    raw=np.asarray(features,dtype=np.float32).reshape(-1)

print()

for n,v in zip(feature_names,raw):

    print(f"{n:30s} {v:15.6f}")

print()

print("="*80)
print("SCALER CHECK")
print("="*80)

scaled=scaler.transform(raw.reshape(1,-1))

scaled=scaled[0]

for n,r,s in zip(feature_names,raw,scaled):

    print(f"{n:30s} RAW={r:12.5f}   SCALED={s:12.5f}")

print()

print("="*80)
print("MODEL")
print("="*80)

prob=model.predict_proba(scaled.reshape(1,-1))[0,1]

print()

print("Probability :",round(float(prob),6))

print("Threshold   :",threshold)

print()

if prob>=threshold:
    print("Prediction  : TAMPERED")
else:
    print("Prediction  : NORMAL")

print()

print("="*80)
print("FEATURE RANGE CHECK")
print("="*80)

for n,s in zip(feature_names,scaled):

    if abs(s)>5:

        print(f"[WARNING] {n}  Z={s:.2f}")

print()

print("="*80)
print("FORENSIC TRACE COMPLETE")
print("="*80)
