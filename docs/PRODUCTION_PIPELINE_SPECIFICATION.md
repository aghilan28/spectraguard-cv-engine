# SpectraGuard — Production Pipeline Specification & M0.3 Execution Blueprint

**Project:** SpectraGuard — Physics-Informed Frequency-Domain Camera Integrity Intelligence  
**Component:** CV Engine (`spectraguard-cv-engine`)  
**Specification:** M0.3 Implementation & Experimentation Blueprint  
**Date:** August 3, 2026  
**Status:** ARCHITECTURE APPROVED & STAGED FOR M0.3 VALIDATION  

---

## 1. Selected Feature Vector Design (To Be Validated in M0.3)

> **DESIGN SPECIFICATION: 8D Physics-Informed Feature Vector**  
> *"The production feature vector will use the 8D representation because it addresses the weaknesses identified in M0.1/M0.2. Full empirical validation will be performed during M0.3."*

$$\mathbf{x} = \begin{bmatrix} \text{Ratio}_{\text{Low}} & \text{Ratio}_{\text{Mid}} & \text{Ratio}_{\text{High}} & \log(1 + E_{\text{total}}) & \sigma^2_{\text{Lap}} & D_{\text{edge}} & H_{\text{spatial}} & \Delta I_{\text{temp}} \end{bmatrix}^T \in \mathbb{R}^8$$

### 1.1 Formal Mathematical Definitions

| Index | Feature Name | Mathematical Formulation | Physical Camera Phenomenon Measured |
| :--- | :--- | :--- | :--- |
| **0** | $\text{Ratio}_{\text{Low}}$ | $\sum_{r=0}^{2} E_r / E_{\text{total}}$ | Low-frequency background illumination & DC spectral power |
| **1** | $\text{Ratio}_{\text{Mid}}$ | $\sum_{r=3}^{6} E_r / E_{\text{total}}$ | Mid-frequency structural contours |
| **2** | $\text{Ratio}_{\text{High}}$ | $\sum_{r=7}^{9} E_r / E_{\text{total}}$ | High-frequency sharp edge & noise ratio (Blur/Defocus indicator) |
| **3** | $\log(1 + E_{\text{total}})$ | $\log\left(1 + \sum_{y=1}^{H}\sum_{x=1}^{W} |\mathcal{F}(Y)(x,y)|\right)$ | Logarithmic total spectral energy (Full Occlusion/Blackout indicator) |
| **4** | $\sigma^2_{\text{Lap}}$ | $\text{Var}\left(\nabla^2 Y\right) = \text{Var}\left(\frac{\partial^2 Y}{\partial x^2} + \frac{\partial^2 Y}{\partial y^2}\right)$ | Spatial focus & sharpness loss |
| **5** | $D_{\text{edge}}$ | $\frac{1}{HW} \sum_{x,y} \mathbb{I}\left(\sqrt{G_x^2 + G_y^2} > T_{\text{Sobel}}\right)$ | Physical blockage & occlusion edge density ($T_{\text{Sobel}}=100$) |
| **6** | $H_{\text{spatial}}$ | $-\sum_{k=0}^{255} p_k \log_2(p_k + 1\text{e-}12)$ | Spatial information entropy (Spray/Smudge indicator) |
| **7** | **$\Delta I_{\text{temp}}$ (Temporal)**| **$\frac{1}{N-1} \sum_{k=1}^{N-1} \left( \frac{1}{HW} \sum_{x,y} |Y_k(x,y) - Y_{k-1}(x,y)| \right)$** | **Mean Absolute Inter-Frame Luminance Difference over $N=15$ rolling frames ($\Delta I_{\text{temp}} \to 0$ indicates Video Replay/Freeze)** |

---

## 2. Classifier Selection Protocol (To Be Benchmarked in M0.3C)

> **POLICY: Model Selection Will Be Benchmarked on New 8D Features in M0.3C**  
> *"While XGBoost won on the legacy 10-bin dataset, the classifier model is NOT permanently frozen. The final model choice will be scientifically selected after benchmarking Random Forest, ExtraTrees, XGBoost, HistGradientBoosting, and SVM on the new 8D feature matrix."*

Candidate models to be evaluated in M0.3C using `GroupKFold` cross-validation:
1. `XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)`
2. `RandomForestClassifier(n_estimators=100, max_depth=10)`
3. `ExtraTreesClassifier(n_estimators=100, max_depth=10)`
4. `HistGradientBoostingClassifier(max_depth=5)`
5. `CalibratedClassifierCV(SVC(kernel="rbf"))`

---

## 3. Official Validation Protocol

> **OFFICIAL PROTOCOL: GroupKFold Cross-Validation by Camera ID (LOCO-CV as Primary)**

- **Primary Evaluation Protocol (Official):** **GroupKFold (Leave-One-Camera-Out CV)** across the 55 unique physical camera setups.
- **Supplementary Protocol:** Stratified 5-Fold Cross-Validation.
- **Transparency Policy:** Random-split performance is NEVER claimed as generalizable accuracy due to spatial background camera leakage. LOCO-CV is our primary reported metric.

---

## 4. Locked Preprocessing Specification

The training pipeline and live RTSP inference engine MUST execute identical processing:

```
Stream ──> [1] Sampler ──> [2] Resize ──> [3] Gray ──> [4] CLAHE ──> [5] Window ──> [6] 2D FFT
                                                                                       │
Alert <── [10] Thresh <── [9] Calib/Model <── [8] Scaler <── [7] 8D Vector Extraction <┘
```

1. **Frame Sampler:** Rolling window $N=15$ frames.
2. **Fixed Resize:** $640 \times 640$ Bilinear Interpolation (`cv2.INTER_LINEAR`).
3. **Color Space Conversion:** ITU-R BT.601 Single-channel Y Luminance ($Y = 0.299R + 0.587G + 0.114B$).
4. **Contrast Normalization:** CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`).
5. **Windowing:** 2D Hanning Window ($w(n) = 0.5 - 0.5 \cos(\frac{2\pi n}{N-1})$).
6. **2D FFT & Masking:** High-Pass Filter with radius $r = \lfloor 0.05 \times 640 \rfloor = 32\text{px}$.
7. **Feature Extraction:** Concatenate 8D Feature Vector ($\text{Ratio}_{\text{Low}}, \text{Ratio}_{\text{Mid}}, \text{Ratio}_{\text{High}}, \log E_{\text{total}}, \sigma^2_{\text{Lap}}, D_{\text{edge}}, H_{\text{spatial}}, \Delta I_{\text{temp}}$).
8. **Feature Normalization:** `StandardScaler` (fitted EXCLUSIVELY on training folds, zero whole-dataset leakage).
9. **Inference & Calibration:** Winning Model + `CalibratedClassifierCV` (Isotonic Regression vs Platt Scaling, selected in M0.3D).
10. **Decision Threshold:** Threshold $\tau$ (To be optimized in M0.3E after final model fit).

---

## 5. Decision Threshold Policy (To Be Optimized in M0.3E)

> **POLICY: Decision Threshold Will Be Optimized After Final Production Fit**  
> *"The decision threshold $\tau$ will be optimized during M0.3E after the final production model is trained and calibrated on the 8D feature matrix."*

- Threshold Optimization Objective: Maximize Youden Index $J = \text{TPR} - \text{FPR}$ and F1-score on held-out validation folds.

---

## 6. Confirmed Training Datasets & Limitations

### 6.1 Confirmed Dataset
- **VIRAT Video Surveillance Core ($N=658$ total clips across 55 cameras)**: 329 clean baseline clips + 329 tampered counterparts.

### 6.2 Documented System Limitations (Honest Scoping)
1. **Synthetic Laser Simulation:** Laser dazzle testing currently uses synthetic saturation simulation; real 532nm/650nm laser pointer validation requires physical hardware data collection.
2. **Synthetic Stream Replay:** 3D Temporal inter-frame difference logic is specified ($\Delta I_{\text{temp}}$), but full network RTSP stream replay training requires dedicated network capture datasets.
3. **Daylight Outdoor Dominance:** VIRAT is 100% daylight outdoor surveillance; night-vision and indoor corridor performance requires planned dataset fine-tuning.
4. **Unseen Camera Variance:** Accuracy across unseen camera viewpoints exhibits variance under LOCO-CV, requiring zero-shot background calibration upon initial stream connection.

---

## 7. M0.3 Structured Implementation Blueprint

M0.3 will proceed through 6 clean sequential sub-phases:

```
[M0.3A] Preprocessing Pipeline Module
   │
   ▼
[M0.3B] Extract 8D Feature Matrix (658 videos)
   │
   ▼
[M0.3C] Benchmark Classifiers (RF, ExtraTrees, XGBoost, HistGB, SVM)
   │
   ▼
[M0.3D] Probability Calibration (Uncalibrated vs Platt vs Isotonic)
   │
   ▼
[M0.3E] Optimize Threshold τ (Youden / F1 Optimal)
   │
   ▼
[M0.3F] Freeze & Serialize Artifacts + Smoke Tests + Runtime Update
```

- **M0.3A:** Implement `src/preprocessing/pipeline.py` (Unified 8D feature extractor; no training).
- **M0.3B:** Run feature extraction on all 658 video clips $\to$ output `data/datasets/virat/metadata/production_features_8d.csv`.
- **M0.3C:** Benchmark candidate models on `production_features_8d.csv` using `GroupKFold` by camera ID $\to$ select winning classifier.
- **M0.3D:** Calibrate winning model (Sigmoid vs Isotonic) $\to$ select winning calibrator.
- **M0.3E:** Compute optimal threshold $\tau$ on calibrated probabilities.
- **M0.3F:** Serialize final production artifacts (`production_model.joblib`, `feature_scaler.joblib`, `calibrator.joblib`, `feature_metadata.json`, `training_manifest.json`, `threshold.json`), run automated unit/smoke tests, and update runtime inference engine.

---
*Specification approved and staged for M0.3 implementation by SpectraGuard Lead AI Architect.*
