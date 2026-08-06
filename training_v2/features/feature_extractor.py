import os
import csv
import json
import numpy as np
import cv2
from src.preprocessing.pipeline import PreprocessingPipeline

class FeatureExtractor:
    def __init__(self, dataset_dir, output_dir):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.pipeline = PreprocessingPipeline()
        self.feature_order = [
            "fft_low_ratio",
            "fft_mid_ratio",
            "fft_high_ratio",
            "log_total_energy",
            "laplacian_variance",
            "edge_density",
            "shannon_entropy",
            "temporal_difference"
        ]

    def extract_from_image(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            img = None
            
        if img is None:
            raise ValueError(f"Could not read image: {filepath}")
        # Pass a list of 15 duplicates of the frame to get a temporal difference of 0.0
        feat_vector = self.pipeline.extract([img] * 15)
        return feat_vector.to_dict()


    def process_split(self, split_name):
        split_path = os.path.join(self.dataset_dir, split_name)
        features_data = []
        
        if not os.path.exists(split_path):
            print(f"[FeatureExtractor] Split folder {split_name} does not exist. Skipping.")
            return []

        classes = {"Normal": 0, "Tampered": 1}
        for cls_name, label_val in classes.items():
            cls_dir = os.path.join(split_path, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for file in os.listdir(cls_dir):
                filepath = os.path.join(cls_dir, file)
                if not os.path.isfile(filepath):
                    continue
                try:
                    feat_dict = self.extract_from_image(filepath)
                    row = [feat_dict.get(feat, 0.0) for feat in self.feature_order]
                    row.append(label_val)
                    features_data.append(row)
                except Exception as e:
                    print(f"[FeatureExtractor] Failed to extract from {file}: {e}")

        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{split_name}_features.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.feature_order + ["label"])
            writer.writerows(features_data)
        print(f"[FeatureExtractor] Saved features to {csv_path}")
        return features_data

    def run(self):
        train_data = self.process_split("train")
        val_data = self.process_split("validation")
        test_data = self.process_split("test")

        if not train_data:
            print("[FeatureExtractor] No training data processed. Exiting distribution analysis.")
            return

        # Calculate feature distributions for training split
        train_np = np.array(train_data)
        features_np = train_np[:, :-1]
        labels_np = train_np[:, -1]

        dist = {}
        for label_val, label_name in [(0, "Normal"), (1, "Tampered")]:
            mask = (labels_np == label_val)
            dist[label_name] = {}
            if np.any(mask):
                cls_features = features_np[mask]
                for idx, feat_name in enumerate(self.feature_order):
                    col_data = cls_features[:, idx]
                    dist[label_name][feat_name] = {
                        "mean": float(np.mean(col_data)),
                        "std": float(np.std(col_data)),
                        "min": float(np.min(col_data)),
                        "max": float(np.max(col_data))
                    }
            else:
                for feat_name in self.feature_order:
                    dist[label_name][feat_name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        dist_path = os.path.join(self.output_dir, "feature_distribution.json")
        with open(dist_path, "w", encoding="utf-8") as f:
            json.dump(dist, f, indent=4)
        print(f"[FeatureExtractor] Saved training feature distributions to {dist_path}")
