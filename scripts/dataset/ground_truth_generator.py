import csv
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class GroundTruthRecord:
    original_filename: str
    generated_filename: str
    attack_category: str
    attack_subtype: str
    attack_severity: str
    attack_parameters: str
    source_scene_category: str
    quality_score: float
    benchmark_identifier: str

class GroundTruthGenerator:
    def __init__(self):
        self.records: List[GroundTruthRecord] = []

    def add_record(self, record: GroundTruthRecord):
        self.records.append(record)

    def write_csv(self, filepath: str):
        if not self.records:
            return
        fieldnames = list(asdict(self.records[0]).keys())
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))
