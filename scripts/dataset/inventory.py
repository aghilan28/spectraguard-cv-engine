import os
import csv
import json
import logging
from dataclasses import asdict

from scripts.dataset.dataset_scanner import DatasetScanner
from scripts.dataset.metadata_extractor import MetadataExtractor
from scripts.dataset.dataset_validator import DatasetValidator
from scripts.dataset.statistics_generator import StatisticsGenerator
from scripts.dataset.manifest_generator import ManifestGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
logger = logging.getLogger("DatasetInventoryPipeline")

def main():
    scan_target = os.path.join("data", "datasets", "virat", "original")
    meta_output = os.path.join("data", "datasets", "virat", "metadata")
    repr_output = os.path.join("data", "datasets", "virat", "reports")
    
    os.makedirs(meta_output, exist_ok=True)
    os.makedirs(repr_output, exist_ok=True)
    
    logger.info(f"Targeting discovery cycle sequence inside: {scan_target}")
    scanner = DatasetScanner(scan_target)
    discovered_files = scanner.scan()
    logger.info(f"Discovered matching supported target array entries: {len(discovered_files)}")
    
    validator = DatasetValidator()
    valid_records = []
    
    csv_file = os.path.join(meta_output, "inventory.csv")
    csv_headers = [
        "filename", "absolute_path", "file_size", "duration", 
        "frame_count", "fps", "width", "height", "aspect_ratio", 
        "codec", "creation_timestamp", "modification_timestamp"
    ]
    
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        
        for file_path in discovered_files:
            meta = MetadataExtractor.extract(file_path)
            if validator.validate_record(file_path, meta) and meta is not None:
                valid_records.append(meta)
                writer.writerow(asdict(meta))
                
    logger.info("CSV inventory matrix serialization complete.")
    
    stats = StatisticsGenerator.calculate(valid_records)
    with open(os.path.join(meta_output, "dataset_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    manifest = ManifestGenerator.create(len(valid_records))
    with open(os.path.join(meta_output, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    validation_report = validator.generate_report()
    with open(os.path.join(repr_output, "dataset_validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
        
    logger.info("Foundation manifest infrastructure reporting generated without runtime failure.")

if __name__ == "__main__":
    main()
