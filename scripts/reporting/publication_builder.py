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
    logger.info("Initializing PHASE CLEAN Publication & Reporting Pipeline...")
    
    meta_dir = os.path.join("data", "datasets", "virat", "metadata")
    eval_dir = os.path.join("data", "datasets", "virat", "evaluation")
    reports_in_dir = os.path.join("data", "datasets", "virat", "reports")
    
    figures_dir = os.path.join("data", "datasets", "virat", "figures")
    tables_dir = os.path.join("data", "datasets", "virat", "tables")
    pres_dir = os.path.join("data", "datasets", "virat", "presentation_assets")
    reports_out_dir = reports_in_dir

    try:
        preds_df = pd.read_csv(os.path.join(eval_dir, "prediction_results.csv"))
        eval_json = load_json(os.path.join(reports_in_dir, "evaluation_report.json"))
        perf_json = load_json(os.path.join(reports_in_dir, "performance_report.json"))
        fail_json = load_json(os.path.join(reports_in_dir, "failure_analysis.json"))
    except Exception as e:
        logger.error(f"Missing required artifact: {e}")
        return

    logger.info("Generating publication-quality figures (Including corrected ROC/PR)...")
    viz_gen = VisualizationGenerator(figures_dir)
    viz_gen.generate_all(preds_df, eval_json)

    logger.info("Generating statistical publication tables (Including missing JSON/HW)...")
    tab_gen = TableGenerator(tables_dir)
    tab_gen.generate_all(preds_df, eval_json, perf_json, reports_out_dir)

    logger.info("Packaging reproducibility and environment manifests...")
    ReproducibilityGenerator.generate(reports_out_dir)

    logger.info("Synthesizing Markdown presentation assets...")
    pres_gen = PresentationExporter(pres_dir)
    pres_gen.export(eval_json, perf_json)

    logger.info("Assembling expanded Technical Report (Markdown/PDF)...")
    rep_gen = ReportGenerator(reports_out_dir)
    rep_gen.generate(eval_json, perf_json, fail_json, preds_df)
    
    summary = {
        "total_videos": len(preds_df),
        "attack_distribution": preds_df['attack_category'].value_counts().to_dict(),
        "scene_distribution": preds_df['source_scene_category'].value_counts().to_dict()
    }
    with open(os.path.join(reports_out_dir, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("PHASE CLEAN Complete. Release-ready publication assets successfully persisted.")

if __name__ == "__main__":
    main()
