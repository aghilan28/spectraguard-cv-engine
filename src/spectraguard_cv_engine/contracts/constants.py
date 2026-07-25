"""System-wide limits, dimensional constraints, and feature namespaces."""


class SystemLimits:
    """Resolution and operational constraints for the CV engine."""

    MIN_RESOLUTION_WIDTH = 640
    MIN_RESOLUTION_HEIGHT = 480
    MAX_RESOLUTION_WIDTH = 3840  # 4K limit for performance bounds
    MAX_RESOLUTION_HEIGHT = 2160
    EXPECTED_CHANNELS_RGB = 3
    EXPECTED_CHANNELS_GRAY = 1
    MAX_SEQUENCE_LENGTH = 120  # Max frames per temporal window


class FeatureNamespaces:
    """Keys for namespacing unified feature vectors."""

    SPATIAL = "spatial"
    FREQUENCY = "frequency"
    TEMPORAL = "temporal"
    META = "meta"
