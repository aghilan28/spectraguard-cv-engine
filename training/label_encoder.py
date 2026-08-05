class LabelEncoder:
    @staticmethod
    def encode(video_path: str) -> int:
        """Assigns 0 for normal, 1 for any tampering based on directory structure."""
        path_lower = video_path.lower()
        if "normal" in path_lower or "virat" in path_lower:
            return 0
        return 1
