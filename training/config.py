from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetConfig:
    dataset_dir: Path = Path("dataset")
    sampling_interval: int = 15  # Extract every 15th frame
    output_csv: Path = Path("training_dataset.csv")
    output_report: Path = Path("dataset_report.json")
    supported_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv')
