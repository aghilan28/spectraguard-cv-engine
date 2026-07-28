import os
from typing import List

class DatasetScanner:
    def __init__(self, target_dir: str):
        self.target_dir = target_dir
        self.supported_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    def scan(self) -> List[str]:
        if not os.path.exists(self.target_dir):
            return []
            
        discovered_files = []
        for root, _, files in os.walk(self.target_dir):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in self.supported_extensions:
                    discovered_files.append(os.path.join(root, file))
                    
        return sorted(discovered_files)
