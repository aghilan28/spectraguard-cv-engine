from pydantic import BaseModel
from typing import List
from backend.inference.result import InferenceResult

class InferenceHistory(BaseModel):
    count: int
    history: List[InferenceResult]

class InferenceStatistics(BaseModel):
    total_inferences: int
    normal_count: int
    tamper_count: int
    average_probability: float
    average_confidence: float
