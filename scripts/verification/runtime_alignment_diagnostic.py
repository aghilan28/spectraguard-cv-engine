import os
import sys
import json
import math
import joblib
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.predictor import SpectraGuardPredictor

RELEASE_DIR = os.path.join(ROOT, 'data', 'models', 'releases', 'v1.0.0')
CSV_PATH = os.path.join(ROOT, 'data', 'datasets', 'virat', 'metadata', 'extracted_fft_features.csv')


def load_offline_reference():
    model = joblib.load(os.path.join(RELEASE_DIR, 'production_model.joblib'))
    scaler = joblib.load(os.path.join(RELEASE_DIR, 'feature_scaler.joblib'))
    metadata = json.load(open(os.path.join(RELEASE_DIR, 'feature_metadata.json'), encoding='utf-8'))
    feature_cols = metadata['feature_names']
    df = pd.read_csv(CSV_PATH)
    return model, scaler, feature_cols, df


def run_offline_rows(model, scaler, feature_cols, df, sample_count=10):
    rows = df.head(sample_count).copy()
    X = rows[feature_cols]
    scaled = scaler.transform(X)
    probs = model.predict_proba(scaled)[:, 1]
    preds = model.predict(scaled)
    return rows, X, scaled, probs, preds


def run_runtime_rows(predictor, feature_cols, df, sample_count=10):
    rows = df.head(sample_count).copy()
    X = rows[feature_cols]
    scaled_df = predictor.artifacts.scaler.transform(X)
    pred_outputs = predictor.runtime.predict(X)
    probs = [p.probability for p in pred_outputs]
    preds = [p.prediction for p in pred_outputs]
    return rows, X, scaled_df, probs, preds


def main():
    model, scaler, feature_cols, df = load_offline_reference()
    predictor = SpectraGuardPredictor(release_version='v1.0.0')

    print('=== RUNTIME ARTIFACT TRACE ===')
    print('predictor.base_dir=', predictor.base_dir)
    print('predictor.model_path=', predictor.model_path)
    print('predictor.scaler_path=', predictor.scaler_path)
    print('predictor.meta_path=', predictor.meta_path)
    print('predictor.model_type=', predictor.model_type)
    print('predictor.release_version=', predictor.release_version)
    print('predictor.expected_features=', predictor.expected_features)
    print('predictor.expected_dim=', predictor.expected_dim)
    print('model_sha256=', predictor.model_hash)
    print('runtime scaler feature count=', len(predictor.artifacts.scaler.feature_names or []))

    sample_count = 10
    rows_offline, X_offline, scaled_offline, offline_probs, offline_preds = run_offline_rows(model, scaler, feature_cols, df, sample_count=sample_count)
    rows_runtime, X_runtime, scaled_runtime, runtime_probs, runtime_preds = run_runtime_rows(predictor, feature_cols, df, sample_count=sample_count)

    print(f'\n=== SAMPLE 0..{sample_count - 1}: OFFLINE MODEL vs RUNTIME PREDICTOR ===')
    for idx in range(sample_count):
        print(f'\n[Sample {idx}]')
        print('raw_features=', rows_offline.iloc[idx][feature_cols].to_dict())
        print('scaled_features=', scaled_offline[idx].tolist())
        print('offline_prob=', float(offline_probs[idx]))
        print('offline_pred=', int(offline_preds[idx]))
        print('runtime_scaled_features=', scaled_runtime.iloc[idx][feature_cols].to_list())
        print('runtime_prob=', float(runtime_probs[idx]))
        print('runtime_pred=', int(runtime_preds[idx]))

        raw_diff = float(np.max(np.abs(X_offline.iloc[idx].to_numpy() - X_runtime.iloc[idx].to_numpy())))
        scaled_diff = float(np.max(np.abs(scaled_offline[idx] - scaled_runtime.iloc[idx][feature_cols].to_numpy())))
        prob_diff = abs(float(offline_probs[idx]) - float(runtime_probs[idx]))
        pred_diff = int(offline_preds[idx]) != int(runtime_preds[idx])
        print(f'diff_raw_max={raw_diff:.12g}, diff_scaled_max={scaled_diff:.12g}, diff_prob={prob_diff:.12g}, pred_match={not pred_diff}')


if __name__ == '__main__':
    main()
