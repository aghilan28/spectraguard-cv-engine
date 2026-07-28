import os
import csv
import json
import logging
from datetime import datetime

from scripts.dataset.quality_analyzer import QualityAnalyzer
from scripts.dataset.scene_classifier import SceneClassifier
from scripts.dataset.benchmark_selector import BenchmarkSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
logger = logging.getLogger("CurationPipeline")

def write_csv(filepath: str, data: list, fieldnames: list):
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def main():
    meta_dir = os.path.join("data", "datasets", "virat", "metadata")
    reports_dir = os.path.join("data", "datasets", "virat", "reports")
    inventory_path = os.path.join(meta_dir, "inventory.csv")

    if not os.path.exists(inventory_path):
        logger.error(f"Phase 1 prerequisite missing: {inventory_path}")
        return

    logger.info("Initializing Quality Assessment phase...")
    analyzer = QualityAnalyzer(sample_frames=8)
    quality_results = analyzer.analyze_dataset(inventory_path)
    
    q_fields = [
        "filename", "blur_score", "sharpness_score", "brightness_score", "contrast_score",
        "noise_estimation", "camera_stability_score", "motion_intensity", "corrupted_frame_ratio",
        "frame_readability", "resolution_validation", "normalized_quality_score"
    ]
    write_csv(os.path.join(meta_dir, "quality_scores.csv"), quality_results, q_fields)

    logger.info("Initializing Scene Categorization phase...")
    classifier = SceneClassifier()
    scene_results = classifier.classify_dataset(inventory_path)
    write_csv(os.path.join(meta_dir, "scene_labels.csv"), scene_results, ["filename", "scene_category", "classification_method"])

    logger.info("Executing Benchmark Candidate Selection logic...")
    inventory_data = []
    with open(inventory_path, "r", encoding="utf-8") as f:
        inventory_data = list(csv.DictReader(f))

    selector = BenchmarkSelector(min_quality=0.35, min_duration=2.0)
    selection_results = selector.select(inventory_data, quality_results, scene_results)
    
    sel_fields = ["filename", "category", "quality_score", "duration", "accepted", "rejection_reason", "benchmark_rank"]
    write_csv(os.path.join(meta_dir, "benchmark_selection.csv"), selection_results, sel_fields)

    # Compile reporting metadata
    accepted_count = sum(1 for x in selection_results if x["accepted"])
    rejected_count = len(selection_results) - accepted_count
    
    category_dist = {}
    for item in selection_results:
        if item["accepted"]:
            cat = item["category"]
            category_dist[cat] = category_dist.get(cat, 0) + 1

    manifest = {
        "dataset_name": "SpectraGuard_Curated_Benchmark",
        "curation_timestamp": datetime.utcnow().isoformat() + "Z",
        "total_candidates_processed": len(selection_results),
        "total_accepted_benchmarks": accepted_count,
        "total_rejected": rejected_count,
        "category_distribution": category_dist
    }

    with open(os.path.join(meta_dir, "benchmark_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    curation_report = {
        "execution_date": datetime.utcnow().isoformat() + "Z",
        "pipeline_status": "SUCCESS",
        "rejection_summary": {}
    }
    
    for item in selection_results:
        if not item["accepted"]:
            reason = item["rejection_reason"]
            curation_report["rejection_summary"][reason] = curation_report["rejection_summary"].get(reason, 0) + 1
            
    with open(os.path.join(reports_dir, "dataset_curation_report.json"), "w", encoding="utf-8") as f:
        json.dump(curation_report, f, indent=2)

    logger.info(f"Curation complete. Accepted: {accepted_count}, Rejected: {rejected_count}")

if __name__ == "__main__":
    main()
