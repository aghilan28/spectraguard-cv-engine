import numpy as np
from src.preprocessing.pipeline import PreprocessingPipeline

class FeatureExtractor:
    def __init__(self, max_history: int = 5):
        self.pipeline = PreprocessingPipeline()
        self.history = []
        self.max_history = max_history

    def extract(self, frame: np.ndarray) -> dict:
        """
        Extracts an 8D feature dictionary from a single video frame.
        Maintains an internal rolling history window for temporal feature computation.
        """
        if frame is None:
            return {}
        self.history.append(frame)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        feature_vector = self.pipeline.extract(self.history)
        return feature_vector.to_dict()
