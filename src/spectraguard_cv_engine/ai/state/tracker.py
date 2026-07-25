"""Temporal event tracking and recovery lifecycle management."""

from typing import Optional
from .models import SystemState, StateTransition
from ..decision.models import SeverityLevel, DecisionOutput


class StateTracker:
    """
    Maintains continuity across individual frame decisions, managing the
    lifecycle of security events and enforcing recovery cooldowns.
    """

    def __init__(self, cooldown_frames: int = 15):
        """
        Args:
            cooldown_frames: Number of consecutive CLEAR decisions required
                             to transition from ACTIVE_EVENT back to NOMINAL.
        """
        if cooldown_frames < 1:
            raise ValueError("Cooldown frames must be at least 1.")

        self.cooldown_frames = cooldown_frames

        self.current_state = SystemState.NOMINAL
        self.frames_in_current_state = 0
        self.consecutive_clear_count = 0
        self.active_event_id: Optional[str] = None

    def process_decision(
        self, decision: DecisionOutput, event_id: str
    ) -> Optional[StateTransition]:
        """
        Ingests a frame-level decision, updates the internal state machine,
        and returns a StateTransition object if a state change occurred.
        """
        self.frames_in_current_state += 1
        previous_state = self.current_state

        is_threat = decision.severity in [
            SeverityLevel.CRITICAL,
            SeverityLevel.ELEVATED,
            SeverityLevel.REVIEW,
        ]
        is_clear = decision.severity == SeverityLevel.CLEAR

        if self.current_state == SystemState.NOMINAL:
            if is_threat:
                self._transition_to(SystemState.ACTIVE_EVENT, event_id)
                return StateTransition(
                    previous_state,
                    self.current_state,
                    self.frames_in_current_state,
                    f"Threat detected ({decision.severity.value}). Initiating active event.",
                )

        elif self.current_state == SystemState.ACTIVE_EVENT:
            if is_clear:
                self._transition_to(SystemState.COOLDOWN, self.active_event_id)
                self.consecutive_clear_count = 1
                return StateTransition(
                    previous_state,
                    self.current_state,
                    self.frames_in_current_state,
                    "Clear frame detected. Entering recovery cooldown phase.",
                )

        elif self.current_state == SystemState.COOLDOWN:
            if is_threat:
                # Cooldown interrupted, snap back to active event
                self._transition_to(SystemState.ACTIVE_EVENT, self.active_event_id)
                self.consecutive_clear_count = 0
                return StateTransition(
                    previous_state,
                    self.current_state,
                    self.frames_in_current_state,
                    f"Cooldown interrupted by threat ({decision.severity.value}). Resuming active event.",
                )
            elif is_clear:
                self.consecutive_clear_count += 1
                if self.consecutive_clear_count >= self.cooldown_frames:
                    # Cooldown complete, recover system
                    self._transition_to(SystemState.NOMINAL, None)
                    return StateTransition(
                        previous_state,
                        self.current_state,
                        self.frames_in_current_state,
                        f"System fully recovered after {self.cooldown_frames} consecutive clear frames.",
                    )

        # No transition occurred
        return None

    def _transition_to(
        self, new_state: SystemState, active_event_id: Optional[str]
    ) -> None:
        """Internal helper to safely mutate state trackers."""
        self.current_state = new_state
        self.frames_in_current_state = 0
        self.active_event_id = active_event_id

        if new_state == SystemState.NOMINAL:
            self.consecutive_clear_count = 0
