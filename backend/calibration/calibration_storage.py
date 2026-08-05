import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from backend.config.settings import settings
from backend.config.logging import logger

class CalibrationStorage:
    """Production-grade filesystem interface managing serialization mechanics for structural baseline distributions."""
    
    def __init__(self, target_dir: Optional[str] = None) -> None:
        self.target_dir = Path(target_dir or settings.baseline_path)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_file = self.target_dir / "baseline.json"

    def save(self, data: Dict[str, Any]) -> None:
        """Serialize baseline schema metadata matrix maps to disk cleanly formatted."""
        try:
            temp_file = self.baseline_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, default=str)
            if self.baseline_file.exists():
                self.baseline_file.unlink()
            temp_file.rename(self.baseline_file)
            logger.info(f"Environmental physical baseline safely serialized to path location: {self.baseline_file}")
        except Exception as e:
            logger.error(f"Failed to execute structural baseline file validation block write: {e}")
            raise IOError(f"Baseline disk serialization serialization architecture error: {e}")

    def load(self) -> Optional[Dict[str, Any]]:
        """Load and parse the active configuration matrix profiles from disk maps hierarchy structures."""
        if not self.exists():
            return None
        try:
            with open(self.baseline_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Baseline storage mapping parse execution corruption caught: {e}")
            return None

    def exists(self) -> bool:
        """True if standard initialization file markers are located inside system maps directories."""
        return self.baseline_file.exists()

    def delete(self) -> bool:
        """Purge configurations map files patterns from system volumes."""
        if self.exists():
            try:
                self.baseline_file.unlink()
                logger.info("Baseline profile structure cleared from physical storage tracks arrays.")
                return True
            except Exception as e:
                logger.error(f"Failsafe baseline tracking clean delete pipeline failure: {e}")
                return False
        return False
