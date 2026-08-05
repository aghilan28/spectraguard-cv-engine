import pandas as pd
from training.config import DatasetConfig
from training.video_loader import VideoLoader
from training.frame_sampler import FrameSampler
from training.feature_pipeline import FeaturePipeline
from training.label_encoder import LabelEncoder
from training.dataset_validator import DatasetValidator
from training.dataset_statistics import DatasetStatistics
from training.dataset_exporter import DatasetExporter

class DatasetBuilder:
    def __init__(self, config: DatasetConfig = DatasetConfig()):
        self.config = config
        self.loader = VideoLoader(config)
        self.validator = DatasetValidator()
        self.stats = DatasetStatistics()
        self.exporter = DatasetExporter()

    def build(self):
        print(f"[BUILDER] Starting dataset generation from {self.config.dataset_dir}...")
        dataset_records = []
        
        for video_info in self.loader.scan_directory():
            path = video_info["path"]
            filename = video_info["filename"]
            label = LabelEncoder.encode(path)
            
            print(f"[BUILDER] Processing: {filename} (Label: {label})")
            feature_pipeline = FeaturePipeline()
            
            for frame_idx, frame in FrameSampler.sample(path, self.config.sampling_interval):
                features = feature_pipeline.process_frame(frame)
                record = {
                    "video_name": filename,
                    "frame_number": frame_idx,
                    "label": label,
                    **features
                }
                dataset_records.append(record)
                
        if not dataset_records:
            print("[BUILDER] No frames processed. Check dataset directory.")
            return

        df = pd.DataFrame(dataset_records)
        df = self.validator.validate(df)
        
        report = self.stats.generate_report(df)
        self.exporter.export(df, report, self.config.output_csv, self.config.output_report)
        print("[BUILDER] Pipeline complete.")

if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()
