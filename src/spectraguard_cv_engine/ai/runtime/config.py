"""Immutable configurations for the Inference Runtime."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Strict configuration boundaries for inference execution."""

    require_probabilities: bool = True
    enforce_schema_validation: bool = True
    max_batch_size: int = 256
