"""Validation suite for Temporal Feature operations."""

import unittest
import numpy as np
from src.spectraguard_cv_engine.features.temporal.sequence import TemporalWindow
from src.spectraguard_cv_engine.features.temporal.differencing import FrameDifferencing
from src.spectraguard_cv_engine.features.temporal.motion import MotionStatistics


class TestTemporalFeatures(unittest.TestCase):
    def setUp(self):
        # Create dummy sequence of 64x64 frames simulating motion
        self.frame1 = np.zeros((64, 64), dtype=np.uint8)
        self.frame2 = np.zeros((64, 64), dtype=np.uint8)
        self.frame3 = np.zeros((64, 64), dtype=np.uint8)

        # Introduce a moving "block"
        self.frame1[10:20, 10:20] = 255
        self.frame2[15:25, 15:25] = 255
        self.frame3[20:30, 20:30] = 255

        # Invalid dimension frame
        self.invalid_frame = np.zeros((32, 32), dtype=np.uint8)

    def test_temporal_window_management(self):
        window = TemporalWindow(window_size=2)
        self.assertFalse(window.is_full)
        self.assertEqual(window.frame_count, 0)

        window.add_frame(self.frame1)
        self.assertFalse(window.is_full)

        window.add_frame(self.frame2)
        self.assertTrue(window.is_full)
        self.assertEqual(window.frame_count, 2)

        # Test rolling window boundary (ejects frame1)
        window.add_frame(self.frame3)
        frames = window.get_window()
        self.assertEqual(len(frames), 2)
        self.assertTrue(np.array_equal(frames[0], self.frame2))
        self.assertTrue(np.array_equal(frames[1], self.frame3))

        with self.assertRaises(ValueError):
            TemporalWindow(window_size=1)

    def test_frame_differencing(self):
        diff = FrameDifferencing.compute_absolute_difference(self.frame1, self.frame2)
        self.assertEqual(diff.shape, (64, 64))
        # Where the block moved, there should be difference values
        self.assertTrue(np.any(diff > 0))

        # Test shape mismatch rejection
        with self.assertRaises(ValueError):
            FrameDifferencing.compute_absolute_difference(
                self.frame1, self.invalid_frame
            )

        # Test sequence differencing
        seq_diffs = FrameDifferencing.compute_sequence_differences(
            [self.frame1, self.frame2, self.frame3]
        )
        self.assertEqual(len(seq_diffs), 2)

        with self.assertRaises(ValueError):
            FrameDifferencing.compute_sequence_differences([self.frame1])

    def test_motion_statistics(self):
        seq_diffs = FrameDifferencing.compute_sequence_differences(
            [self.frame1, self.frame2, self.frame3]
        )
        stats = MotionStatistics.extract_motion_features(seq_diffs)

        self.assertIn("mean_motion", stats)
        self.assertIn("motion_variance", stats)
        self.assertIn("temporal_instability", stats)

        # Motion should be > 0 since blocks moved
        self.assertGreater(stats["mean_motion"], 0.0)

        # Empty difference list test
        empty_stats = MotionStatistics.extract_motion_features([])
        self.assertEqual(empty_stats["mean_motion"], 0.0)


if __name__ == "__main__":
    unittest.main()
