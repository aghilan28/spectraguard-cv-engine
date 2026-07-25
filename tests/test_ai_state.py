"""Validation suite for AI State Tracking and Recovery."""

import unittest

from src.spectraguard_cv_engine.ai.decision.models import DecisionOutput, SeverityLevel
from src.spectraguard_cv_engine.ai.state.models import SystemState
from src.spectraguard_cv_engine.ai.state.tracker import StateTracker


class TestStateTracker(unittest.TestCase):
    def setUp(self):
        # Extremely short cooldown for fast testing
        self.tracker = StateTracker(cooldown_frames=3)
        self.evt_id = "test_evt_1"

        self.threat_dec = DecisionOutput(
            severity=SeverityLevel.CRITICAL, action_required=True, rationale=""
        )
        self.clear_dec = DecisionOutput(
            severity=SeverityLevel.CLEAR, action_required=False, rationale=""
        )

    def test_initial_state(self):
        self.assertEqual(self.tracker.current_state, SystemState.NOMINAL)
        self.assertIsNone(self.tracker.active_event_id)

    def test_transition_to_active_event(self):
        trans = self.tracker.process_decision(self.threat_dec, self.evt_id)

        self.assertIsNotNone(trans)
        self.assertEqual(trans.previous_state, SystemState.NOMINAL)
        self.assertEqual(trans.new_state, SystemState.ACTIVE_EVENT)
        self.assertEqual(self.tracker.current_state, SystemState.ACTIVE_EVENT)
        self.assertEqual(self.tracker.active_event_id, self.evt_id)

    def test_transition_to_cooldown_and_interruption(self):
        # Trigger event
        self.tracker.process_decision(self.threat_dec, self.evt_id)

        # Enter cooldown
        trans1 = self.tracker.process_decision(self.clear_dec, self.evt_id)
        self.assertEqual(trans1.new_state, SystemState.COOLDOWN)

        # Interrupt cooldown with another threat
        trans2 = self.tracker.process_decision(self.threat_dec, self.evt_id)
        self.assertEqual(trans2.new_state, SystemState.ACTIVE_EVENT)
        self.assertEqual(self.tracker.consecutive_clear_count, 0)

    def test_successful_cooldown_recovery(self):
        # Trigger event
        self.tracker.process_decision(self.threat_dec, self.evt_id)

        # Feed exactly 3 clear frames (cooldown_frames limit)
        trans1 = self.tracker.process_decision(self.clear_dec, self.evt_id)  # 1
        self.assertEqual(trans1.new_state, SystemState.COOLDOWN)

        trans2 = self.tracker.process_decision(self.clear_dec, self.evt_id)  # 2
        self.assertIsNone(trans2)  # Still in cooldown

        trans3 = self.tracker.process_decision(self.clear_dec, self.evt_id)  # 3
        self.assertEqual(trans3.new_state, SystemState.NOMINAL)  # Recovered!
        self.assertIsNone(self.tracker.active_event_id)


if __name__ == "__main__":
    unittest.main()
