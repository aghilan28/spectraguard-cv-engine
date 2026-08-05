"""
Template-driven deterministic explanation generator for UI contextualization.
No AI or LLMs involved. Static mapping guarantees stability.
"""

def generate_explanation(tamper_type: str) -> str:
    """Provides a human-readable deterministic justification for the triggered classification."""
    templates = {
        "NORMAL": "All physical parameters are operating within standard calibrated baseline tolerances.",
        "LENS_COVER": "Brightness, entropy, and edge density simultaneously decreased beyond calibrated limits, consistent with severe lens obstruction.",
        "LENS_SPRAY": "Laplacian variance and edge density collapsed while entropy spiked, indicating particle adhesion or liquid spray on the lens.",
        "DEFOCUS": "Laplacian variance decreased significantly while high-frequency FFT energy collapsed, indicating severe optical defocus.",
        "CAMERA_MOVED": "Temporal frame difference spiked rapidly alongside structural edge shifts, consistent with physical camera realignment or impact.",
        "FLASH_ATTACK": "Absolute image brightness surged violently while scene entropy collapsed, indicating blinding directed light.",
        "DARKNESS": "Low-frequency illumination and total energy dropped below minimum visibility thresholds.",
        "OVEREXPOSURE": "Total spectral energy and low-frequency saturation exceeded standard exposure bounds.",
        "VIDEO_FREEZE": "Temporal pixel variation dropped to zero against historical bounds, indicating a frozen sensor or network replication loop.",
        "HEAVY_NOISE": "High-frequency spectral bands and spatial entropy spiked abnormally, indicating digital interference or sensor degradation.",
        "PARTIAL_OCCLUSION": "Structural edge complexity and mid-frequency textures dropped locally, indicating a partial physical block.",
        "UNKNOWN_ANOMALY": "Multiple physical parameters exceeded baseline bounds with conflicting signatures, indicating a complex environmental disturbance."
    }
    return templates.get(tamper_type, "Physical anomaly detected without matching defined signatures.")
