"""Master orchestrator for end-to-end feature extraction."""

import numpy as np
from typing import List

from .models import UnifiedFeatureVector
from ...preprocessing.pipeline import PreprocessingPipeline
from ..spatial.gradients import GradientExtractor
from ..spatial.texture import TextureExtractor
from ..spatial.statistics import SpatialStatistics
from ..frequency.transforms import FrequencyTransformer
from ..frequency.descriptors import SpectralDescriptors
from ..temporal.differencing import FrameDifferencing
from ..temporal.motion import MotionStatistics


class UnifiedExtractionPipeline:
    """Aggregates all CV subsystems to process raw sequences into UnifiedFeatureVectors."""

    @staticmethod
    def extract_from_sequence(
        raw_frames: List[np.ndarray], vector_id: str, timestamp_ns: int
    ) -> UnifiedFeatureVector:
        """
        Executes the complete CV pipeline on a temporal sequence of raw BGR frames.
        Spatial and Frequency features are extracted from the most recent (target) frame.
        Temporal features are extracted across the sequence.
        """
        if not raw_frames:
            raise ValueError("Cannot extract features from an empty sequence.")

        # 1. Preprocessing (Validate, Denoise, Grayscale, Normalize)
        clean_frames = [
            PreprocessingPipeline.process_standard_spatial_frame(f) for f in raw_frames
        ]
        target_frame = clean_frames[-1]

        # 2. Spatial Domain Extraction
        stats = SpatialStatistics.extract_intensity_stats(target_frame)
        mag, _ = GradientExtractor.compute_sobel_gradients(target_frame)
        edges = GradientExtractor.extract_edge_statistics(mag)
        lap_var = TextureExtractor.compute_laplacian_variance(target_frame)
        contrast = TextureExtractor.compute_global_contrast(target_frame)

        spatial_dict = {
            **stats,
            **edges,
            "laplacian_variance": lap_var,
            "global_contrast": contrast,
        }

        # 3. Frequency Domain Extraction
        f_shift = FrequencyTransformer.compute_fft(target_frame)
        mag_spec = FrequencyTransformer.compute_magnitude_spectrum(f_shift)
        freq_dict = {
            "spectral_energy": SpectralDescriptors.calculate_spectral_energy(mag_spec),
            "spectral_entropy": SpectralDescriptors.calculate_spectral_entropy(
                mag_spec
            ),
            "spectral_flatness": SpectralDescriptors.calculate_spectral_flatness(
                mag_spec
            ),
        }

        # 4. Temporal Domain Extraction
        if len(clean_frames) >= 2:
            diffs = FrameDifferencing.compute_sequence_differences(clean_frames)
            temp_dict = MotionStatistics.extract_motion_features(diffs)
        else:
            temp_dict = MotionStatistics.extract_motion_features([])

        return UnifiedFeatureVector(
            vector_id=vector_id,
            timestamp_ns=timestamp_ns,
            spatial_features=spatial_dict,
            frequency_features=freq_dict,
            temporal_features=temp_dict,
        )
