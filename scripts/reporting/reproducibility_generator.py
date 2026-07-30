import os
import sys
import json
import psutil
import platform
import subprocess
from datetime import datetime, UTC

class ReproducibilityGenerator:
    @staticmethod
    def generate(out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        
        commit_hash = "unknown"
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
        except Exception:
            pass

        data = {
            "execution_metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "pipeline_version": "1.0.0",
                "benchmark_version": "1.0.0",
                "random_seed": 42
            },
            "environment_summary": {
                "os": platform.system(),
                "os_release": platform.release(),
                "python_version": sys.version,
                "cpu_cores": psutil.cpu_count(logical=True),
                "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            },
            "repository_state": {
                "commit_hash": commit_hash
            }
        }
        
        with open(os.path.join(out_dir, 'reproducibility.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
