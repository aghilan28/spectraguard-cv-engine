# SpectraGuard Phase 7.5 Authoritative Execution Checklist
**Derived from R1-R6 and Tech Audit**

## TASK 1 & 2: Dataset Acquisition & Validation (R2)
- [ ] Setup canonical dirs: `data/raw`, `data/processed`, `data/manifests`.
- [ ] Acquire Mandatory: **UHCTD** (Agreement form required), **Kylberg Textures** (without-rotation, 0.9GB).
- [ ] Acquire Highly Recommended: **CUHK Blur**, **DUT-DBD**, **RainDrop**, **WoodScape** (Soiling subset only, ~5k images).
- [ ] Validate integrity, missing values, and target labels.

## TASK 3: Exploratory Data Analysis (EDA)
- [ ] Class distribution, resolution, video duration, blur/noise baseline stats.

## TASK 4: Synthetic Dataset Generation (R3)
- [ ] T1: Replay/Loop/Frozen-frame (FFmpeg/OpenCV).
- [ ] T2: Codec/Compression matrix (FFmpeg H.264/H.265 sweeps).
- [ ] T3: Organic lens-accumulation (Dust/Spiderweb progressive overlay).
- [ ] T4: Lens dirt/mud/spray/tape/paper (Alpha-compositing).
- [ ] T5: Hand-obstruction set.
- [ ] T6: Camera shake/vibration (Homography/affine perturbation).
- [ ] T7: Weather/lighting nuisance (imgaug/Albumentations rain/flicker/drift).
- [ ] T8: Motion-blur discrimination fixture (Linear kernel).
- [ ] T9: Combined tampering composite set.

## TASK 5: Feature Extraction (R4)
- [ ] Implement robust regression (RANSAC/Huber) for D-RPSS fit.
- [ ] Calculate original 11-dim features.
- [ ] Add **Tenengrad** focus measure.
- [ ] Add **Local Binary Patterns (LBP)** (gated on Block-DCT anomaly).
- [ ] Verify output: 13-dimensional vector per frame.

## TASK 6 & 7: Baseline Training & Optimization (R5)
- [ ] **Strict Constraint:** XGBoost version MUST be `< 3.0` (Tech Audit).
- [ ] **Strict Constraint:** Episode-level stratified splitting ONLY. NO frame-level splits (R6).
- [ ] Primary Model: Random Forest.
- [ ] Validator Models: XGBoost, Extra Trees.
- [ ] Baseline Falsification Model: Logistic Regression.
- [ ] Balancing: Use `class_weight='balanced'` (inverse-frequency). Do NOT use SMOTE by default.
- [ ] Calibration: Platt scaling for display confidence ONLY.

## TASK 8 & 9: Error Analysis & Production Freeze (R6)
- [ ] Generate 95% Bootstrap Confidence Intervals for F1/Precision/Recall.
- [ ] Run McNemar's test with continuity correction for model promotion (+3 F1 point rule).
- [ ] Run E1-E6 ablation and robustness matrices.
- [ ] Ensure Exact TreeSHAP functions correctly for chosen model.
- [ ] Export via `joblib`.
- [ ] Tag `v0.7.5-production-model`.
