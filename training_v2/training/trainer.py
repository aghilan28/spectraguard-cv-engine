import os
import joblib
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

class ModelTrainer:
    def __init__(self, features_dir, model_dir):
        self.features_dir = features_dir
        self.model_dir = model_dir

    def run(self):
        print("[Trainer] Starting model training...")
        train_path = os.path.join(self.features_dir, "train_features.csv")
        if not os.path.exists(train_path):
            print("[Trainer] Training features CSV not found. Skipping training.")
            return

        df = pd.read_csv(train_path)
        X = df.drop(columns=["label"])
        y = df["label"]

        # Scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Base RF Classifier with Class Weight Balancing
        rf = RandomForestClassifier(class_weight="balanced", random_state=42)

        # GridSearchCV / Cross Validation
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        }
        grid = GridSearchCV(rf, param_grid, cv=3, scoring="f1", n_jobs=-1)
        grid.fit(X_scaled, y)
        best_rf = grid.best_estimator_
        print(f"[Trainer] GridSearchCV complete. Best params: {grid.best_params_}")

        # Probability Calibration
        calibrated_model = CalibratedClassifierCV(best_rf, method="sigmoid", cv=3)
        calibrated_model.fit(X_scaled, y)


        # Export Candidate Model and Scaler
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "production_model.joblib")
        scaler_path = os.path.join(self.model_dir, "feature_scaler.joblib")
        
        joblib.dump(calibrated_model, model_path)
        joblib.dump(scaler, scaler_path)

        # Save metadata
        meta = {
            "feature_names": list(X.columns),
            "feature_order": list(X.columns),
            "feature_count": len(X.columns),
            "best_params": grid.best_params_,
            "model_type": "Binary_Random_Forest_Calibrated"
        }
        meta_path = os.path.join(self.model_dir, "feature_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

        print(f"[Trainer] Candidate model and metadata exported to {self.model_dir}")
