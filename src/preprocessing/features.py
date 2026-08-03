import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List

@dataclass
class FeatureVector:
    fft_low_ratio: float
    fft_mid_ratio: float
    fft_high_ratio: float
    log_total_energy: float
    laplacian_variance: float
    edge_density: float
    shannon_entropy: float
    temporal_difference: float

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "fft_low_ratio",
            "fft_mid_ratio",
            "fft_high_ratio",
            "log_total_energy",
            "laplacian_variance",
            "edge_density",
            "shannon_entropy",
            "temporal_difference"
        ]

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.fft_low_ratio,
            self.fft_mid_ratio,
            self.fft_high_ratio,
            self.log_total_energy,
            self.laplacian_variance,
            self.edge_density,
            self.shannon_entropy,
            self.temporal_difference
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
