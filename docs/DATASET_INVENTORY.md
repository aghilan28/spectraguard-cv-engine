# SpectraGuard Dataset Inventory

## 1. Core Training Data
* **UHCTD (Feature Extracted):** 12,000 episodes (Train/Val). Classes: Normal (0), Covered (1), Defocused (2), Moved (3).

## 2. Synthetic Data
* **Kylberg Textures:** Used for synthetic 'covered' augmentation.
* **Cyber-Tamper (Replay/Loop):** Synthesized via noise-modulated duplication.

## 3. R4 Feature Schema (13 Dimensions)
1. `d_hfer_median` (Continuous [0,1])
2. `d_rpss_median` (Continuous)
3. `spectral_flatness` (Continuous [0,1])
4. `block_dct_mean` (Continuous)
5. `block_dct_var` (Continuous)
6. `block_dct_max` (Continuous)
7. `optical_flow_disp` (Continuous, -1.0 = INCONCLUSIVE)
8. `immerkaer_sigma` (Continuous)
9. `p_hash_match` (Binary {0,1})
10. `noise_autocorr_peak` (Continuous)
11. `diurnal_mismatch` (Continuous)
12. `tenengrad_focus` (Continuous)
13. `lbp_texture_dist` (Continuous, 0.0 = Absent)
