"""Unified feature vector models and serialization logic."""

import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

# Enforce deterministic ordering for 1D array conversion (Critical for ML consistency)
SPATIAL_KEYS = [
    "mean_intensity",
    "variance_intensity",
    "skewness",
    "kurtosis",
    "mean_magnitude",
    "max_magnitude",
    "edge_density",
    "laplacian_variance",
    "global_contrast",
]
FREQUENCY_KEYS = ["spectral_energy", "spectral_entropy", "spectral_flatness"]
TEMPORAL_KEYS = ["mean_motion", "motion_variance", "temporal_instability"]


@dataclass
class UnifiedFeatureVector:
    """Consolidated 1D numerical representation of a frame sequence's physical properties."""

    vector_id: str
    timestamp_ns: int
    spatial_features: Dict[str, float]
    frequency_features: Dict[str, float]
    temporal_features: Dict[str, float]

    def to_array(self) -> np.ndarray:
        """Serializes features into a strictly ordered 1D NumPy float array."""
        vector = []
        # Spatial
        for key in SPATIAL_KEYS:
            vector.append(self.spatial_features.get(key, 0.0))
        # Frequency
        for key in FREQUENCY_KEYS:
            vector.append(self.frequency_features.get(key, 0.0))
        # Temporal
        for key in TEMPORAL_KEYS:
            vector.append(self.temporal_features.get(key, 0.0))

        return np.array(vector, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the vector payload to a JSON-compatible dictionary."""
        return {
            "vector_id": self.vector_id,
            "timestamp_ns": self.timestamp_ns,
            "features": {
                "spatial": self.spatial_features,
                "frequency": self.frequency_features,
                "temporal": self.temporal_features,
            },
            "dimensions": len(SPATIAL_KEYS) + len(FREQUENCY_KEYS) + len(TEMPORAL_KEYS),
        }

    def to_json(self) -> str:
        """Returns the JSON string representation of the unified features."""
        return json.dumps(self.to_dict())
