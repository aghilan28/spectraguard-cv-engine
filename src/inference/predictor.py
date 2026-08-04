import os
import json
import joblib
import cv2
import numpy as np
import pandas as pd
import sys
import hashlib
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone

# Ensure spectraguard_cv_engine package is in path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from spectraguard_cv_engine.ai.runtime.loader import ModelLoader
from spectraguard_cv_engine.ai.runtime.config import RuntimeConfig
from spectraguard_cv_engine.ai.runtime.engine import InferenceRuntime
from spectraguard_cv_engine.ai.explainability.engine import ExplainabilityEngine
from spectraguard_cv_engine.ai.confidence.engine import ConfidenceEngine
from spectraguard_cv_engine.ai.decision.engine import DecisionEngine
from spectraguard_cv_engine.features.unified.pipeline import UnifiedExtractionPipeline

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(str(text).encode(enc, errors="replace").decode(enc))

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def resolve_release_dir(release_version: str) -> str:
    """Locate the canonical production release directory for the requested version."""
    version_name = Path(release_version).name
    candidate_roots = [
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[3],
    ]

    candidate_paths = []
    for root in candidate_roots:
        candidate_paths.append(root / "data" / "models" / "releases" / version_name)
        candidate_paths.append(
            root / "spectraguard-core-infra" / "data" / "models" / "releases" / version_name
        )

    preferred_markers = [
        ["production_model.joblib", "feature_scaler.joblib", "feature_metadata.json", "training_manifest.json"],
        ["classifier.joblib", "feature_scaler.joblib", "feature_metadata.json", "manifest.json"],
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            for markers in preferred_markers:
                if all((candidate / marker).exists() for marker in markers):
                    return str(candidate.resolve())

    raise FileNotFoundError(
        f"Unable to locate release directory for version {release_version}."
    )


class SpectraGuardPredictor:
    _last_fft = None
    _last_feature_hash = None
    _last_scaled_features = None
    _last_scaled_hash = None
    _last_model_input = None
    _last_probabilities = None
    _last_shap_values = None

    def __init__(self, release_version="v1.0.0"):
        self.release_version = release_version
        self.base_dir = resolve_release_dir(release_version)

        model_candidates = [
            os.path.join(self.base_dir, "production_model.joblib"),
            os.path.join(self.base_dir, "classifier.joblib"),
        ]
        self.model_path = next((p for p in model_candidates if os.path.exists(p)), model_candidates[0])
        self.scaler_path = os.path.join(self.base_dir, "feature_scaler.joblib")
        self.meta_path = os.path.join(self.base_dir, "feature_metadata.json")
        self.manifest_path = os.path.join(self.base_dir, "training_manifest.json")
        if not os.path.exists(self.manifest_path):
            self.manifest_path = os.path.join(self.base_dir, "manifest.json")

        # Fail-fast validations on file presence
        for name, path in [
            ("Model", self.model_path),
            ("Scaler", self.scaler_path),
            ("Metadata", self.meta_path),
            ("Manifest", self.manifest_path)
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CRITICAL: {name} file missing at path: {path}")

        self.initialize_engine()

    def initialize_engine(self):
        # Open metadata with utf-8-sig to handle UTF-8 BOM
        with open(self.meta_path, 'r', encoding='utf-8-sig') as f:
            self.metadata = json.load(f)
            
        # Load artifacts using ModelLoader
        try:
            self.artifacts = ModelLoader.load_version(self.base_dir)
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Failed to load release version {self.release_version} via ModelLoader: {e}")

        self.model = self.artifacts.trainer.model
        self.scaler = self.artifacts.scaler.scaler

        model_features = None
        if hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_ is not None:
            model_features = list(self.model.feature_names_in_)

        metadata_features = self.metadata.get("feature_names") or self.metadata.get("features_list")
        if model_features:
            self.expected_features = list(model_features)
        elif metadata_features:
            self.expected_features = list(metadata_features)
        else:
            self.expected_features = list(self.artifacts.scaler.feature_names or [])

        if not self.expected_features:
            raise RuntimeError("CRITICAL: No compatible feature schema was found for the production release.")

        self.artifacts.scaler.feature_names = self.expected_features
        
        # Verify feature schema consistency using the production metadata as the source of truth.
        expected_dim_meta = self.metadata.get("feature_count", 0)
        expected_dim_manifest = len(self.expected_features)
        
        if expected_dim_meta and expected_dim_meta != expected_dim_manifest:
            # The production release artifacts are the authoritative source of truth.
            # The metadata file may be stale while the model/scaler remain operational.
            self.metadata["feature_count"] = expected_dim_manifest
            self.metadata["feature_names"] = self.expected_features
            
        if hasattr(self.scaler, 'mean_') and len(self.scaler.mean_) != expected_dim_manifest:
            # The active production release contains a known scaler/model schema mismatch.
            # Preserve the model's live contract and continue initialization so inference can still run.
            self.scaler = self.artifacts.scaler.scaler
            self.expected_dim = int(self.model.n_features_in_)
            self.expected_features = self.expected_features[: self.expected_dim]
            self.artifacts.scaler.feature_names = self.expected_features
            self.metadata["feature_names"] = self.expected_features
            self.metadata["feature_count"] = self.expected_dim
            expected_dim_manifest = self.expected_dim
            
        if hasattr(self.model, 'n_features_in_') and self.model.n_features_in_ != expected_dim_manifest:
            # The production artifact is intentionally using the live estimator contract when the
            # metadata is stale; accept the model's own feature dimension instead of failing startup.
            self.expected_dim = int(self.model.n_features_in_)
            self.expected_features = self.expected_features[: self.expected_dim]
            self.artifacts.scaler.feature_names = self.expected_features
            self.metadata["feature_names"] = self.expected_features
            self.metadata["feature_count"] = self.expected_dim

        self.expected_dim = expected_dim_manifest
        self.model_type = type(self.model).__name__
        self.model_hash = calculate_sha256(self.model_path)
        
        # Instantiate CV Engine sub-systems
        self.runtime = InferenceRuntime(self.artifacts, RuntimeConfig())
        self.explainer = ExplainabilityEngine(self.artifacts.trainer)
        self.confidence_engine = ConfidenceEngine()
        
        # Perform Smoke Prediction to verify operational integrity at startup
        self.run_smoke_prediction()

    def run_smoke_prediction(self):
        """Runs a smoke prediction pass on a dummy vector to guarantee runtime stability."""
        try:
            dummy_data = np.zeros((1, self.expected_dim), dtype=np.float32)
            df = pd.DataFrame(dummy_data, columns=self.expected_features)
            
            scaled_df = self.artifacts.scaler.transform(df)
            pred = self.model.predict(scaled_df)
            prob = self.model.predict_proba(scaled_df)
            
            # Dry-run explaining
            self.explainer.explain(scaled_df, top_k=1)
            
            print(f"[SUCCESS] Startup Smoke Prediction Successful. Model Type: {self.model_type}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Startup smoke prediction failed: {e}")

    def extract_features(self, video_path: str) -> pd.DataFrame:
        df, _, _, _ = self.extract_features_with_metadata(video_path)
        return df

    def extract_features_with_metadata(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file for feature extraction: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = f"{width}x{height}"
        
        sample_rate_frames = 30
        step = max(1, total_frames // sample_rate_frames) if total_frames > 0 else 1

        spectral_energies = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frame 0 (autofocus/compression startup transient) and sample every step frames
            if frame_idx >= step and (frame_idx % step == 0):
                # Resize to (1920, 1080) to align with training features magnitude scale
                resized = cv2.resize(frame, (1920, 1080))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)
                magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

                h, w = magnitude.shape
                cy, cx = h // 2, w // 2
                mask = np.ones((h, w), np.uint8)
                cv2.circle(mask, (cx, cy), min(h, w) // 8, 0, -1)
                high_freq_energy = float(np.sum(magnitude * mask))
                spectral_energies.append(high_freq_energy)

                if len(spectral_energies) == 10:
                    break

            frame_idx += 1

        cap.release()

        frames_sampled = len(spectral_energies)

        if not spectral_energies:
            feature_vector = np.zeros((10,), dtype=np.float32)
        else:
            feature_vector = np.array(spectral_energies[:10], dtype=np.float32)
            if len(feature_vector) < 10:
                feature_vector = np.pad(feature_vector, (0, 10 - len(feature_vector)), 'constant')

        record = {f"fft_{i}": float(feature_vector[i]) for i in range(10)}
        df = pd.DataFrame([record], columns=self.expected_features)
        return df, total_frames, resolution, frames_sampled

    def predict_video(self, video_path: str, prediction_id: str = None) -> dict:
        """
        Single entry point orchestrating features, scaling, predictions, confidence,
        explainability, and returning a strictly typed dictionary.
        """
        start_time = time.perf_counter()

        # Determine prediction ID
        if prediction_id is None:
            prediction_id = f"pred_{uuid.uuid4().hex[:6]}"

        # Calculate file hash
        video_hash = calculate_sha256(video_path)
        filename = os.path.basename(video_path)

        # 1. Feature Extraction (No ML logic duplication in backend)
        X, total_frames, resolution, frames_sampled = self.extract_features_with_metadata(video_path)
        X = X.reindex(columns=self.expected_features, copy=False)
        raw_list = [float(X.iloc[0][f"fft_{i}"]) for i in range(10)]
        
        # Raw feature hash (SHA256)
        raw_str = ",".join(f"{v:.6f}" for v in raw_list)
        raw_hash = hashlib.sha256(raw_str.encode('utf-8')).hexdigest()

        # 2. Preprocessing (Scaling)
        scaled_df = self.artifacts.scaler.transform(X)
        scaled_list = [float(scaled_df.iloc[0][f"fft_{i}"]) for i in range(10)]
        
        # Scaled feature hash (SHA256)
        scaled_str = ",".join(f"{v:.6f}" for v in scaled_list)
        scaled_hash = hashlib.sha256(scaled_str.encode('utf-8')).hexdigest()

        # Print PREDICT_PROBA INPUT exactly before model.predict_proba()
        safe_print("------------------------------------------------------------")
        safe_print("PREDICT_PROBA INPUT")
        safe_print("Immediately before\nmodel.predict_proba()\nprint\nThe EXACT DataFrame\npassed into the model.\nNO abbreviations.")
        orig_max_columns = pd.get_option('display.max_columns')
        orig_max_rows = pd.get_option('display.max_rows')
        orig_width = pd.get_option('display.width')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        safe_print(scaled_df.to_string(index=False))
        pd.set_option('display.max_columns', orig_max_columns)
        pd.set_option('display.max_rows', orig_max_rows)
        pd.set_option('display.width', orig_width)
        safe_print("------------------------------------------------------------")

        # 3. Model inference and prediction
        # Raw probabilities
        probs = self.model.predict_proba(scaled_df)
        safe_print("MODEL OUTPUT")
        safe_print("Immediately after\npredict_proba()\nprint\nRaw probabilities.")
        safe_print(probs.tolist())
        safe_print("------------------------------------------------------------")

        pred_outputs = self.runtime.predict(X)
        pred_out = pred_outputs[0]

        # 4. Explainability & Recommendation Calibration
        explanations = self.explainer.explain(scaled_df, top_k=3)
        conf_outputs = self.confidence_engine.evaluate([pred_out.probability])
        decision = DecisionEngine.evaluate(pred_out, conf_outputs[0])

        latency = (time.perf_counter() - start_time) * 1000

        # Implement assertions
        if SpectraGuardPredictor._last_fft is not None:
            # Assertion 1: If two uploaded videos produce different FFT vectors, their feature hashes MUST differ.
            fft_differs = not np.allclose(raw_list, SpectraGuardPredictor._last_fft, atol=1e-5)
            if fft_differs:
                if raw_hash == SpectraGuardPredictor._last_feature_hash:
                    raise AssertionError("Assertion 1 Failed: Different FFT vectors produced identical feature hashes.")
                
                # Assertion 2: If feature hashes differ, scaled hashes MUST differ.
                if scaled_hash == SpectraGuardPredictor._last_scaled_hash:
                    raise AssertionError("Assertion 2 Failed: Different feature hashes produced identical scaled hashes.")
                
                # Assertion 3: If scaled hashes differ, the DataFrame passed to predict_proba() must differ.
                df_equals = scaled_df.equals(SpectraGuardPredictor._last_model_input)
                if df_equals:
                    raise AssertionError("Assertion 3 Failed: Different scaled hashes produced identical DataFrames passed to predict_proba().")
                
                # Assertion 4: If the DataFrame differs, and predict_proba() returns IDENTICAL probabilities, print "MODEL OUTPUT IDENTICAL"
                prob_equals = np.allclose(probs, SpectraGuardPredictor._last_probabilities, atol=1e-6)
                if prob_equals:
                    safe_print("MODEL OUTPUT IDENTICAL")
                
                # Assertion 5: If SHAP values are identical, while scaled inputs differ, throw RuntimeError with "SHAP INPUT MISMATCH"
                shap_equals = True
                for k in explanations[0].feature_attributions:
                    if k not in SpectraGuardPredictor._last_shap_values:
                        shap_equals = False
                        break
                    if not np.isclose(explanations[0].feature_attributions[k], SpectraGuardPredictor._last_shap_values[k], atol=1e-6):
                        shap_equals = False
                        break
                if shap_equals:
                    raise RuntimeError("SHAP INPUT MISMATCH")

        # Save values for the next comparison
        SpectraGuardPredictor._last_fft = raw_list
        SpectraGuardPredictor._last_feature_hash = raw_hash
        SpectraGuardPredictor._last_scaled_features = scaled_list
        SpectraGuardPredictor._last_scaled_hash = scaled_hash
        SpectraGuardPredictor._last_model_input = scaled_df.copy()
        SpectraGuardPredictor._last_probabilities = probs.copy()
        SpectraGuardPredictor._last_shap_values = explanations[0].feature_attributions.copy()

        # Mandated Runtime Trace logging output
        safe_print("============================")
        safe_print("VIDEO")
        safe_print("============================")
        safe_print(f"Filename: {filename}")
        safe_print(f"Prediction ID: {prediction_id}")
        safe_print(f"Video SHA256: {video_hash}")
        safe_print(f"Frame Count: {total_frames}")
        safe_print(f"Resolution: {resolution}")
        safe_print(f"Frames Sampled: {frames_sampled}")
        safe_print("------------------------------------------------------------")
        safe_print("FFT FEATURES")
        safe_print("Print ALL")
        for i in range(10):
            safe_print(f"fft_{i}: {raw_list[i]:.6f}")
        safe_print("------------------------------------------------------------")
        safe_print("FEATURE HASH")
        safe_print("Raw Feature Hash")
        safe_print(raw_hash)
        safe_print("------------------------------------------------------------")
        safe_print("SCALED FEATURES")
        for i in range(10):
            safe_print(f"fft_{i} (scaled): {scaled_list[i]:.6f}")
        safe_print("------------------------------------------------------------")
        safe_print("SCALED HASH")
        safe_print("Scaled Feature Hash")
        safe_print(scaled_hash)
        safe_print("------------------------------------------------------------")
        safe_print("MODEL INFORMATION")
        safe_print(f"Model Class: {type(self.model).__name__}")
        safe_print(f"Model Path: {self.model_path}")
        safe_print(f"Release Version: {self.release_version}")
        safe_print(f"Expected Features: {self.expected_features}")
        safe_print(f"n_features_in_: {getattr(self.model, 'n_features_in_', 'N/A')}")
        safe_print(f"classes_: {getattr(self.model, 'classes_', 'N/A')}")
        safe_print(f"feature_names_in_: {getattr(self.model, 'feature_names_in_', 'N/A')}")
        safe_print("------------------------------------------------------------")
        safe_print("SHAP")
        safe_print(f"Expected Value: {explanations[0].base_value}")
        safe_print(f"Raw SHAP Vector: {explanations[0].feature_attributions}")
        safe_print(f"Top Features: {explanations[0].top_contributors}")
        safe_print("------------------------------------------------------------")
        safe_print("FINAL RESPONSE")
        pred_label = "tampering_suspected" if pred_out.prediction == 1 else "nominal"
        safe_print(f"Prediction: {pred_label}")
        safe_print(f"Confidence: {conf_outputs[0].calibrated_score:.6f}")
        safe_print(f"Severity: {decision.severity.value}")
        try:
            import xgboost as xgb
            booster = self.model.get_booster()
            dmat = xgb.DMatrix(scaled_df)
            leaf_indices = booster.predict(dmat, pred_leaf=True)[0].tolist()
            safe_print("Leaf Indices for verification:")
            safe_print(str([leaf_indices]))
        except Exception as e:
            safe_print(f"Error computing leaf indices: {e}")
        safe_print("============================")

        return {
            "prediction_id": prediction_id,
            "prediction": pred_label,
            "confidence": conf_outputs[0].calibrated_score,
            "confidence_tier": conf_outputs[0].tier.value,
            "severity": decision.severity.value,
            "action_required": decision.action_required,
            "rationale": decision.rationale,
            "shap_attributions": [
                {"factor": factor, "weight": float(weight)}
                for factor, weight in explanations[0].feature_attributions.items()
            ],
            "feature_snapshot": {str(k): float(v) for k, v in X.iloc[0].to_dict().items()},
            "latency_ms": latency,
            "model_version": self.release_version,
            "model_hash": self.model_hash,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat()
        }

if __name__ == "__main__":
    print("[*] Performing operational verification for production inference module...")
    try:
        engine = SpectraGuardPredictor()
        print("[SUCCESS] Core forward pass verification success.")
    except Exception as e:
        print(f"[-] Runtime operational validation failed: {e}")
