import os
import csv
import json
import logging
import hashlib
import time
from datetime import datetime, UTC

from scripts.dataset.tamper_generator import TamperGenerator
from scripts.dataset.ground_truth_generator import GroundTruthGenerator, GroundTruthRecord

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
logger = logging.getLogger("BenchmarkBuilder")

ATTACK_CONFIG = [
    ('defocus', {'ksize': 25}, 'high'),
    ('gaussian_blur', {'ksize': 31}, 'high'),
    ('partial_occlusion', {'percentage': 0.25}, 'medium'),
    ('full_occlusion', {}, 'critical'),
    ('spray', {}, 'medium'),
    ('camera_shift', {'tx': 30, 'ty': 30}, 'low'),
    ('camera_shake', {'max_shift': 15}, 'high'),
    ('low_light', {'alpha': 0.5, 'beta': -40}, 'medium')
]

def ensure_directories(base_tampered_dir: str):
    dirs = [
        "defocus", "gaussian_blur", "partial_occlusion", "full_occlusion",
        "spray", "camera_shift", "camera_shake", "low_light"
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_tampered_dir, d), exist_ok=True)

def load_csv(filepath: str) -> dict:
    data = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                data[row["filename"]] = row
    return data

def main():
    meta_dir = os.path.join("data", "datasets", "virat", "metadata")
    original_dir = os.path.join("data", "datasets", "virat", "original")
    tampered_dir = os.path.join("data", "datasets", "virat", "tampered")
    reports_dir = os.path.join("data", "datasets", "virat", "reports")
    
    ensure_directories(tampered_dir)
    
    selection_path = os.path.join(meta_dir, "benchmark_selection.csv")
    if not os.path.exists(selection_path):
        logger.error("Phase 2 prerequisite missing. Aborting.")
        return

    selections = load_csv(selection_path)
    
    candidates = [v for k, v in selections.items() if v.get("accepted") == 'True']
    total_candidates = len(candidates)
    logger.info(f"Loaded {total_candidates} accepted benchmark candidates.")

    gt_generator = GroundTruthGenerator()
    success_count = 0
    failure_count = 0

    for idx, candidate in enumerate(candidates):
        orig_filename = candidate["filename"]
        source_path = os.path.join(original_dir, orig_filename)
        
        if not os.path.exists(source_path):
            found = False
            for root, _, files in os.walk(original_dir):
                if orig_filename in files:
                    source_path = os.path.join(root, orig_filename)
                    found = True
                    break
            if not found:
                logger.warning(f"Source file not found: {orig_filename}")
                failure_count += 1
                continue

        file_hash = int(hashlib.md5(orig_filename.encode('utf-8')).hexdigest(), 16)
        attack_idx = file_hash % len(ATTACK_CONFIG)
        attack_type, params, severity = ATTACK_CONFIG[attack_idx]
        
        gen_filename = f"tamp_{attack_type}_{orig_filename}"
        output_path = os.path.join(tampered_dir, attack_type, gen_filename)
        seed = file_hash % (2**32 - 1)

        start_time = time.time()
        logger.info(f"[{idx + 1}/{total_candidates}] Generating {attack_type} -> {orig_filename}...")
        
        is_success = TamperGenerator.process_video(source_path, output_path, attack_type, params, seed=seed)
        
        elapsed = time.time() - start_time
        
        if is_success:
            logger.info(f"[{idx + 1}/{total_candidates}] Completed in {elapsed:.2f}s")
            success_count += 1
            gt_record = GroundTruthRecord(
                original_filename=orig_filename,
                generated_filename=gen_filename,
                attack_category=attack_type,
                attack_subtype="standard",
                attack_severity=severity,
                attack_parameters=json.dumps(params),
                source_scene_category=candidate.get("category", "Unknown"),
                quality_score=float(candidate.get("quality_score", 0.0)),
                benchmark_identifier=f"BMK-{seed:08x}"
            )
            gt_generator.add_record(gt_record)
        else:
            logger.error(f"[{idx + 1}/{total_candidates}] Failed to generate attack.")
            failure_count += 1

    gt_csv_path = os.path.join(meta_dir, "ground_truth.csv")
    gt_generator.write_csv(gt_csv_path)

    manifest_data = {
        "dataset_name": "SpectraGuard_Tampered_Benchmark",
        "generation_timestamp": datetime.now(UTC).isoformat(),
        "total_attempted": total_candidates,
        "total_generated": success_count,
        "total_failed": failure_count,
        "ground_truth_file": gt_csv_path
    }
    
    with open(os.path.join(meta_dir, "tampering_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for k, v in manifest_data.items():
            writer.writerow([k, v])

    with open(os.path.join(reports_dir, "tampering_generation_report.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Generation complete. Success: {success_count} | Failed: {failure_count}")

if __name__ == "__main__":
    main()
