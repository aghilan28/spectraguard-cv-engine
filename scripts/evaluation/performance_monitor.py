import time
import psutil
import os
from typing import Dict, Any

class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = 0.0
        self.start_cpu = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        self.start_cpu = self.process.cpu_percent()

    def sample(self) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self.start_time
        mem_info = self.process.memory_info()
        return {
            "elapsed_seconds": elapsed,
            "cpu_utilization_pct": self.process.cpu_percent(),
            "memory_usage_bytes": mem_info.rss,
            "peak_memory_bytes": mem_info.peak_wset if hasattr(mem_info, 'peak_wset') else mem_info.vms
        }
