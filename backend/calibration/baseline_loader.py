"""
Thread-safe memory loader for the baseline.json serialization.
"""
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from backend.calibration.calibration_storage import CalibrationStorage
from backend.config.logging import logger

class BaselineLoader:
    """Manages hot-loading and memory isolation for the active baseline schema."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "BaselineLoader":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.storage = CalibrationStorage()
        self._data: Optional[Dict[str, Any]] = None
        self._read_lock = threading.Lock()
        self._initialized = True
        self.reload()

    def reload(self) -> bool:
        """Force a fresh disk read of the baseline configurations maps."""
        with self._read_lock:
            try:
                if self.storage.exists():
                    self._data = self.storage.load()
                    logger.info("Baseline environmental metrics cleanly loaded into memory caches.")
                    return True
                else:
                    self._data = None
                    logger.warning("No baseline configuration discovered on disk paths.")
                    return False
            except Exception as e:
                logger.error(f"Failed to read baseline configurations structures: {e}")
                self._data = None
                return False

    def get_data(self) -> Optional[Dict[str, Any]]:
        with self._read_lock:
            return self._data

    def exists(self) -> bool:
        with self._read_lock:
            return self._data is not None

    def camera_id(self) -> Optional[str]:
        with self._read_lock:
            if self._data:
                return self._data.get("camera_id")
            return None
            
    def feature_names(self) -> List[str]:
        with self._read_lock:
            if self._data and "features" in self._data:
                return list(self._data["features"].keys())
            return []

baseline_loader = BaselineLoader()
