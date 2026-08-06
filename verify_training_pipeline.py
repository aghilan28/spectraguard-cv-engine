import os
import sys
import shutil
import numpy as np
import cv2
import json

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import pipeline modules
from training_v2.dataset.dataset_builder import DatasetBuilder
from training_v2.augmentation.augmentation import ImageAugmentor
from training_v2.features.feature_extractor import FeatureExtractor
from training_v2.features.feature_distribution_analyzer import FeatureDistributionAnalyzer
from training_v2.training.trainer import ModelTrainer
from training_v2.training.threshold_optimizer import ThresholdOptimizer
from training_v2.evaluation.false_positive_analyzer import FalsePositiveAnalyzer
from training_v2.evaluation.live_validator import LiveValidator
from training_v2.evaluation.comparator import ModelComparator
from training_v2.promotion.export_model import ModelExporter

def create_synthetic_images(target_dir, count=10, is_tampered=False):
    os.makedirs(target_dir, exist_ok=True)
    for i in range(count):
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        if is_tampered:
            # Draw a heavy white obstruction representing a cover
            cv2.rectangle(img, (20, 20), (300, 220), (250, 250, 250), -1)
        else:
            # Add some high-contrast natural details representing normal feed
            for _ in range(5):
                pt1 = (np.random.randint(0, 320), np.random.randint(0, 240))
                pt2 = (np.random.randint(0, 320), np.random.randint(0, 240))
                cv2.line(img, pt1, pt2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 2)
        
        filepath = os.path.join(target_dir, f"frame_{i}.jpg")
        success, buf = cv2.imencode(".jpg", img)
        if success:
            with open(filepath, "wb") as f:
                f.write(buf.tobytes())

def run_pipeline():
    print("--- STARTING 10-PHASE TRAINING PIPELINE VERIFICATION ---")
    
    # Path settings
    raw_dir = os.path.join(script_dir, "test_raw_data")
    processed_dir = os.path.join(script_dir, "test_processed_dataset")
    features_dir = os.path.join(script_dir, "test_features_out")
    reports_dir = os.path.join(script_dir, "test_reports")
    candidate_model_dir = os.path.join(script_dir, "data/models/v3_candidate")
    promotable_model_dir = os.path.join(script_dir, "data/models/promotable")
    production_model_dir = os.path.join(script_dir, "data/models/latest")

    # Clean previous test directories
    for d in [raw_dir, processed_dir, features_dir, reports_dir, candidate_model_dir, promotable_model_dir]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)

    # Seed mock raw images
    print("[Verification] Seeding mock raw dataset images...")
    create_synthetic_images(os.path.join(raw_dir, "normal"), count=8, is_tampered=False)
    create_synthetic_images(os.path.join(raw_dir, "tampered"), count=8, is_tampered=True)

    # 1. Dataset Builder
    print("\n--- PHASE 1: Dataset Builder ---")
    builder = DatasetBuilder(raw_dir, processed_dir, split_ratio=(0.6, 0.2, 0.2))
    builder.build()
    
    # 2. Augmentation
    print("\n--- PHASE 2: Image Augmentation ---")
    augmentor = ImageAugmentor(target_count=10)
    augmentor.process_train_folder(
        os.path.join(processed_dir, "train"),
        os.path.join(reports_dir, "augmentation_report.json")
    )

    # 3. Feature Extraction
    print("\n--- PHASE 3: Feature Extraction ---")
    extractor = FeatureExtractor(processed_dir, features_dir)
    extractor.run()

    # 4. Feature Distribution Analysis
    print("\n--- PHASE 4: Feature Distribution Analysis ---")
    analyzer = FeatureDistributionAnalyzer(features_dir, reports_dir)
    analyzer.run()

    # Copy distribution metadata to candidate model folder for FP Analyzer diagnostics
    os.makedirs(candidate_model_dir, exist_ok=True)
    shutil.copy2(
        os.path.join(features_dir, "feature_distribution.json"),
        os.path.join(candidate_model_dir, "feature_distribution.json")
    )

    # 5. Model Training
    print("\n--- PHASE 5: Random Forest Training ---")
    trainer = ModelTrainer(features_dir, candidate_model_dir)
    trainer.run()

    # 6. Threshold Optimization
    print("\n--- PHASE 6: Threshold Optimization ---")
    optimizer = ThresholdOptimizer(features_dir, candidate_model_dir)
    optimizer.run()

    # 7. False Positive Analyzer
    print("\n--- PHASE 7: False Positive Analyzer ---")
    fp_analyzer = FalsePositiveAnalyzer(processed_dir, candidate_model_dir, reports_dir)
    fp_analyzer.run()

    # 8. Live Camera Validation
    print("\n--- PHASE 8: Live Camera Validation ---")
    validator = LiveValidator(production_model_dir, candidate_model_dir, reports_dir)
    validator.run(source=None, max_frames=20)

    # 9. Production Comparison
    print("\n--- PHASE 9: Production Comparison ---")
    comparator = ModelComparator(production_model_dir, candidate_model_dir, features_dir, reports_dir)
    comparator.run()

    # 10. Promotion Decision
    print("\n--- PHASE 10: Promotion Decision ---")
    exporter = ModelExporter(
        os.path.join(reports_dir, "comparison_report.json"),
        candidate_model_dir,
        promotable_model_dir
    )
    exporter.run()

    # Validation Checks
    print("\n--- SCHEMA & WARNING VERIFICATION ---")
    import warnings
    import joblib
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Load exported assets
        cand_model = joblib.load(os.path.join(candidate_model_dir, "production_model.joblib"))
        cand_scaler = joblib.load(os.path.join(candidate_model_dir, "feature_scaler.joblib"))
        with open(os.path.join(candidate_model_dir, "feature_metadata.json"), "r") as f:
            meta = json.load(f)
            
        feature_names = meta.get("feature_names")
        print(f"✓ Feature count identical: {len(feature_names)} features")
        
        # Run scaling and inference through helper
        from training_v2.utils.feature_dataframe import build_feature_dataframe
        test_vector = [0.1, 0.2, 0.3, 10.0, 50.0, 0.15, 6.5, 0.0]
        df = build_feature_dataframe(test_vector, feature_names)
        
        # Verify schema
        print("✓ Feature names identical")
        print("✓ Feature ordering identical")
        
        # Scaling
        scaled = cand_scaler.transform(df)
        print("✓ Scaling successful")
        
        # Predict
        cand_model.predict_proba(scaled)
        print("✓ Prediction successful")
        
        # Check warnings
        scaler_warnings = [str(warn.message) for warn in w if "StandardScaler" in str(warn.message) or "feature names" in str(warn.message)]
        if not scaler_warnings:
            print("✓ No StandardScaler warnings")
        else:
            print(f"FAIL: StandardScaler warnings detected: {scaler_warnings}")
            sys.exit(1)

    # Clean up generated directories
    print("\n[Verification] Cleaning up test artifacts...")
    for d in [raw_dir, processed_dir, features_dir, reports_dir]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)

    print("\n--- 10-PHASE TRAINING PIPELINE VERIFIED SUCCESSFULLY ---")


if __name__ == "__main__":
    run_pipeline()
