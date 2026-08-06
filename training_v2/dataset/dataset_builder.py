import os
import shutil
import json
import hashlib
import cv2
import numpy as np
from PIL import Image


class DatasetBuilder:
    def __init__(self, raw_dir, output_dir, split_ratio=(0.7, 0.15, 0.15)):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.split_ratio = split_ratio
        
    def check_md5(self, filepath):
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def validate_image(self, filepath):
        # Format check
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            return False, "Incorrect format"
        
        # Corrupted check / Dimension check
        try:
            with Image.open(filepath) as img:
                img.verify()
            
            # Read dimension
            with open(filepath, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                return False, "Corrupted image reader"
            h, w = img.shape[:2]
            return True, {"dimensions": f"{w}x{h}"}
        except Exception as e:
            return False, f"Corruption: {e}"


    def build(self):
        print("[DatasetBuilder] Scanning raw source files...")
        image_list = []
        md5_seen = set()
        
        stats = {
            "total_raw": 0,
            "corrupted": 0,
            "duplicates": 0,
            "incorrect_format": 0,
            "normal_raw_count": 0,
            "tamper_raw_count": 0,
            "final_dataset": {}
        }

        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir, exist_ok=True)
            print(f"[DatasetBuilder] Created raw folder at {self.raw_dir}. Add dataset images there.")
            # Create subfolders for mock seed
            os.makedirs(os.path.join(self.raw_dir, "normal"), exist_ok=True)
            os.makedirs(os.path.join(self.raw_dir, "tampered"), exist_ok=True)

        for root, dirs, files in os.walk(self.raw_dir):
            for file in files:
                filepath = os.path.join(root, file)
                stats["total_raw"] += 1
                
                # Check format
                valid, info = self.validate_image(filepath)
                if not valid:
                    print(f"[DatasetBuilder] Validation failed for {file}: {info}")
                    if "format" in str(info):
                        stats["incorrect_format"] += 1
                    else:
                        stats["corrupted"] += 1
                    continue

                
                # Check duplicate
                file_hash = self.check_md5(filepath)
                if file_hash in md5_seen:
                    stats["duplicates"] += 1
                    continue
                md5_seen.add(file_hash)
                
                # Label correctness mapping
                # Map directories to Normal vs Tampered (binary classifier)
                parent_folder = os.path.basename(os.path.dirname(filepath)).lower()
                if "normal" in parent_folder:
                    label = "Normal"
                    stats["normal_raw_count"] += 1
                else:
                    label = "Tampered"
                    stats["tamper_raw_count"] += 1
                    
                image_list.append({
                    "src": filepath,
                    "filename": file,
                    "label": label,
                    "dims": info["dimensions"]
                })

        # Split and write
        import random
        random.seed(42)
        random.shuffle(image_list)
        
        n_total = len(image_list)
        n_train = int(n_total * self.split_ratio[0])
        n_val = int(n_total * self.split_ratio[1])
        
        splits = {
            "train": image_list[:n_train],
            "validation": image_list[n_train:n_train+n_val],
            "test": image_list[n_train+n_val:]
        }
        
        # Write to folders
        for split_name, split_files in splits.items():
            stats["final_dataset"][split_name] = {"total": len(split_files), "Normal": 0, "Tampered": 0}
            for item in split_files:
                label_dir = os.path.join(self.output_dir, split_name, item["label"])
                os.makedirs(label_dir, exist_ok=True)
                dest = os.path.join(label_dir, item["filename"])
                shutil.copy2(item["src"], dest)
                stats["final_dataset"][split_name][item["label"]] += 1

        # Save stats
        stats_path = os.path.join(self.output_dir, "dataset_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
            
        print(f"[DatasetBuilder] Dataset statistics saved to {stats_path}")
        return stats
