from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class ValidationFailure:
    file_path: str
    issue: str
    details: str

class DatasetValidator:
    def __init__(self):
        self.failures: List[ValidationFailure] = []

    def validate_record(self, file_path: str, meta: Any) -> bool:
        if meta is None:
            self.failures.append(ValidationFailure(file_path, "UNREADABLE_OR_CORRUPTED", "OpenCV failed to bind target file pointer."))
            return False
            
        is_valid = True
        if meta.frame_count <= 0:
            self.failures.append(ValidationFailure(file_path, "ZERO_FRAME_VIDEO", "Video metadata reports 0 frames."))
            is_valid = False
            
        if meta.fps <= 0 or meta.fps > 240:
            self.failures.append(ValidationFailure(file_path, "INVALID_FPS", f"Extracted frame timing profile is invalid: {meta.fps}"))
            is_valid = False
            
        if meta.width <= 0 or meta.height <= 0:
            self.failures.append(ValidationFailure(file_path, "MISSING_METADATA", f"Dimensions resolution parsing error: {meta.width}x{meta.height}"))
            is_valid = False
            
        return is_valid

    def generate_report(self) -> Dict[str, Any]:
        return {
            "valid": len(self.failures) == 0,
            "total_violations": len(self.failures),
            "violations": [asdict(f) for f in self.failures]
        }
