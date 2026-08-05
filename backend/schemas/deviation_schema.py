from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict

class FeatureDeviationSchema(BaseModel):
    feature: str = Field(description="Name of the evaluated physics feature mapping")
    live: float = Field(description="Real-time feature scalar extraction value")
    mean: float = Field(description="Calibrated environmental mean baseline marker")
    std: float = Field(description="Calibrated feature standard deviation metric")
    z_score: float = Field(description="Raw statistical deviation variance scale bounds")
    normalized_drift: float = Field(description="Standardized feature drift [0-1] scaled to 5 sigma")
    weight: float = Field(description="Architectural importance modifier factor weight")

class DeviationReportResponse(BaseModel):
    timestamp: datetime = Field(description="Exact UTC confirmation execution clock mark")
    overall_score: float = Field(description="Weighted global deviation metric indicator bounded [0-1]")
    severity: str = Field(description="Standardized severity categorical designation string")
    mahalanobis_distance: float = Field(description="Full multivariate cov-adjusted distance footprint calculation")
    feature_reports: List[FeatureDeviationSchema] = Field(description="Granular component data vector listings breakdown")

class DeviationStatisticsResponse(BaseModel):
    average_score: float
    maximum_score: float
    minimum_score: float
    average_mahalanobis: float
    highest_feature_drift: str
    lowest_feature_drift: str
    history_size: int

class DeviationEvaluationRequest(BaseModel):
    feature_vector: Dict[str, float] = Field(
        description="Incoming 8D real-time parameter tracking vector array dictionary map",
        examples=[{
            "laplacian_variance": 1200.5, "log_total_energy": 20.4, "edge_density": 0.22,
            "shannon_entropy": 6.8, "fft_low_ratio": 0.38, "fft_mid_ratio": 0.41,
            "temporal_difference": 4.2, "fft_high_ratio": 0.21
        }]
    )
