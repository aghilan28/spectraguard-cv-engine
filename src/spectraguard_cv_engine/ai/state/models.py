"""Data schemas for AI temporal state tracking."""

from enum import Enum
from dataclasses import dataclass


class SystemState(str, Enum):
    """The overarching temporal state of the monitored stream."""

    NOMINAL = "NOMINAL"  # Normal operation, no active event
    ACTIVE_EVENT = "ACTIVE_EVENT"  # Currently tracking a tamper or anomaly
    COOLDOWN = "COOLDOWN"  # Potential recovery detected, awaiting confirmation


@dataclass(frozen=True)
class StateTransition:
    """Payload emitted when the system changes its temporal state."""

    previous_state: SystemState
    new_state: SystemState
    frames_in_state: int
    rationale: str
