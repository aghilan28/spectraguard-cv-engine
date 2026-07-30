import os
import json
import logging
import pandas as pd

from scripts.reporting.visualization_generator import VisualizationGenerator
from scripts.reporting.table_generator import TableGenerator
from scripts.reporting.reproducibility_generator import ReproducibilityGenerator
from scripts.reporting.presentation_exporter import PresentationExporter
from scripts.reporting.report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')
logger = logging.getLogger("PublicationBuilder")

def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    logger.info("Initializing Phase 5 Publication & Reporting Pipeline...")
    
    # Input directories
    meta_dir = os.path.join("data", "datasets", "virat", "metadata")
    eval_dir = os.path.join("data", "datasets", "virat", "evaluation")
    reports_in_dir = os.path.join("data", "datasets", "virat", "reports")
    
    # Output directories
    figures_dir = os.path.join("data", "datasets", "virat", "figures")
    tables_dir = os.path.join("data", "datasets", "virat", "tables")
    pres_dir = os.path.join("data", "datasets", "virat", "presentation_assets")
    reports_out_dir = reports_in_dir  # Save final_report here

    # Load artifacts
    try:
        preds_df = pd.read_csv(os.path.join(eval_dir, "prediction_results.csv"))
        eval_json = load_json(os.path.join(reports_in_dir, "evaluation_report.json"))
        perf_json = load_json(os.path.join(reports_in_dir, "performance_report.json"))
        fail_json = load_json(os.path.join(reports_in_dir, "failure_analysis.json"))
    except Exception as e:
        logger.error(f"Missing required Phase 4.5 artifact: {e}")
        return

    # 1. Generate Visualizations
    logger.info("Generating publication-quality figures...")
    viz_gen = VisualizationGenerator(figures_dir)
    viz_gen.generate_all(preds_df, eval_json)

    # 2. Generate Tables
    logger.info("Generating statistical publication tables...")
    tab_gen = TableGenerator(tables_dir)
    tab_gen.generate_all(preds_df, eval_json, perf_json)

    # 3. Generate Reproducibility Package
    logger.info("Packaging reproducibility and environment manifests...")
    ReproducibilityGenerator.generate(reports_out_dir)

    # 4. Export Presentation Assets
    logger.info("Synthesizing Markdown presentation assets...")
    pres_gen = PresentationExporter(pres_dir)
    pres_gen.export(eval_json, perf_json)

    # 5. Generate Final Technical Report
    logger.info("Assembling multi-format Master Technical Report (Markdown/PDF)...")
    rep_gen = ReportGenerator(reports_out_dir)
    rep_gen.generate(eval_json, perf_json, fail_json)
    
    # Benchmark Summary Map
    summary = {
        "total_videos": len(preds_df),
        "attack_distribution": preds_df['attack_category'].value_counts().to_dict(),
        "scene_distribution": preds_df['source_scene_category'].value_counts().to_dict()
    }
    with open(os.path.join(reports_out_dir, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase 5 Reporting Pipeline Complete. Publication assets successfully persisted.")

if __name__ == "__main__":
    main()
