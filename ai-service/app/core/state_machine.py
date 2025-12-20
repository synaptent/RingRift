"""Unified State Machine for RingRift AI Service.

Provides a reusable state machine implementation with:
- Type-safe state transitions
- Transition validation and guards
- Automatic logging of state changes
- Metrics integration
- Common state sets for coordinators/orchestrators

Usage:
    from app.core.state_machine import StateMachine, State, Transition

    class MyService(StateMachine):
        # Define states
        IDLE = State("idle", initial=True)
        RUNNING = State("running")
        STOPPED = State("stopped", terminal=True)

        # Define transitions
        TRANSITIONS = [
            Transition(IDLE, RUNNING, guard=lambda s: s.is_ready()),
            Transition(RUNNING, STOPPED),
            Transition(IDLE, STOPPED),
        ]

        def start(self):
            self.transition_to(self.RUNNING)

Pre-built State Sets:
    - CoordinatorStates: INITIALIZING, ACTIVE, PAUSED, STOPPING, STOPPED, FAILED
    - OrchestratorStates: IDLE, STARTING, RUNNING, DRAINING, STOPPED, FAILED
    - PipelineStates: IDLE, SELFPLAY, TRAINING, EVALUATION, PROMOTION, FAILED
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Union

logger = logging.getLogger(__name__)

__all__ = [
    # Pre-built state sets
    "CoordinatorStates",
    "InvalidTransitionError",
    "OrchestratorStates",
    "PipelineStates",
    "State",
    "StateMachine",
    "StateMachineError",
    "Transition",
]


class StateMachineError(Exception):
    """Base exception for state machine errors."""


class InvalidTransitionError(StateMachineError):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, from_state: str, to_state: str, reason: str = ""):
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        message = f"Invalid transition from '{from_state}' to '{to_state}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


@dataclass
class State:
    """Represents a state in a state machine.

    Attributes:
        name: Unique state identifier
        initial: Whether this is the initial state
        terminal: Whether this is a terminal/final state
        on_enter: Callback when entering this state
        on_exit: Callback when exiting this state
        metadata: Additional state metadata
    """
    name: str
    initial: bool = False
    terminal: bool = False
    on_enter: Callable[[StateMachine], None] | None = None
    on_exit: Callable[[StateMachine], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, State):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"State({self.name!r})"


@dataclass
class Transition:
    """Represents a valid state transition.

    Attributes:
        from_state: Source state
        to_state: Target state
        guard: Optional condition that must be true for transition
        action: Optional action to perform during transition
        name: Optional transition name for logging
    """
    from_state: State
    to_state: State
    guard: Callable[[StateMachine], bool] | None = None
    action: Callable[[StateMachine], None] | None = None
    name: str | None = None

    def can_execute(self, machine: StateMachine) -> bool:
        """Check if this transition can be executed."""
        if self.guard is None:
            return True
        try:
            return self.guard(machine)
        except Exception as e:
            logger.warning(f"Transition guard failed: {e}")
            return False

    def execute(self, machine: StateMachine) -> None:
        """Execute the transition action."""
        if self.action is not None:
            self.action(machine)


@dataclass
class StateHistory:
    """Record of a state change."""
    from_state: str
    to_state: str
    timestamp: float
    transition_name: str | None = None
    duration_in_previous: float = 0.0


class StateMachine:
    """Base class for state machines.

    Provides state transition management with validation, logging,
    and optional metrics integration.

    Subclasses should:
    1. Define State class attributes for each state
    2. Define TRANSITIONS class attribute with valid transitions
    3. Call super().__init__() in their __init__

    Example:
        class MyService(StateMachine):
            IDLE = State("idle", initial=True)
            RUNNING = State("running")
            STOPPED = State("stopped", terminal=True)

            TRANSITIONS = [
                Transition(IDLE, RUNNING),
                Transition(RUNNING, STOPPED),
            ]

            def __init__(self):
                super().__init__()

            def start(self):
                self.transition_to(self.RUNNING)
    """

    # Subclasses should override these
    TRANSITIONS: list[Transition] = []

    def __init__(
        self,
        initial_state: State | None = None,
        record_history: bool = True,
        max_history: int = 100,
    ):
        """Initialize the state machine.

        Args:
            initial_state: Override initial state (else uses State with initial=True)
            record_history: Whether to record state change history
            max_history: Maximum history entries to keep
        """
        self._states = self._collect_states()
        self._transitions = self._build_transition_map()
        self._current_state = initial_state or self._find_initial_state()
        self._state_entered_at = time.time()
        self._record_history = record_history
        self._max_history = max_history
        self._history: list[StateHistory] = []

    def _collect_states(self) -> dict[str, State]:
        """Collect all State attributes from the class."""
        states = {}
        for name in dir(self.__class__):
            attr = getattr(self.__class__, name)
            if isinstance(attr, State):
                states[attr.name] = attr
        return states

    def _build_transition_map(self) -> dict[str, list[Transition]]:
        """Build a map of from_state -> valid transitions."""
        transition_map: dict[str, list[Transition]] = {}
        for t in self.TRANSITIONS:
            from_name = t.from_state.name
            if from_name not in transition_map:
                transition_map[from_name] = []
            transition_map[from_name].append(t)
        return transition_map

    def _find_initial_state(self) -> State:
        """Find the initial state."""
        for state in self._states.values():
            if state.initial:
                return state
        # Default to first state if none marked initial
        if self._states:
            return next(iter(self._states.values()))
        raise StateMachineError("No states defined")

    @property
    def state(self) -> State:
        """Get the current state."""
        return self._current_state

    @property
    def state_name(self) -> str:
        """Get the current state name."""
        return self._current_state.name

    @property
    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return self._current_state.terminal

    @property
    def time_in_state(self) -> float:
        """Get time spent in current state (seconds)."""
        return time.time() - self._state_entered_at

    @property
    def history(self) -> list[StateHistory]:
        """Get state change history."""
        return list(self._history)

    def can_transition_to(self, target: Union[State, str]) -> bool:
        """Check if transition to target state is valid.

        Args:
            target: Target state or state name

        Returns:
            True if transition is valid and guards pass
        """
        target_name = target.name if isinstance(target, State) else target

        if self._current_state.terminal:
            return False

        transitions = self._transitions.get(self._current_state.name, [])
        for t in transitions:
            if t.to_state.name == target_name:
                return t.can_execute(self)

        return False

    def transition_to(
        self,
        target: Union[State, str],
        *,
        force: bool = False,
    ) -> bool:
        """Transition to a new state.

        Args:
            target: Target state or state name
            force: Skip validation (use with caution)

        Returns:
            True if transition succeeded

        Raises:
            InvalidTransitionError: If transition is invalid and not forced
        """
        target_state = self._resolve_state(target)
        from_state = self._current_state

        # Check if transition is valid
        if not force:
            if from_state.terminal:
                raise InvalidTransitionError(
                    from_state.name, target_state.name,
                    "Cannot transition from terminal state"
                )

            transition = self._find_transition(from_state, target_state)
            if transition is None:
                raise InvalidTransitionError(
                    from_state.name, target_state.name,
                    "No valid transition defined"
                )

            if not transition.can_execute(self):
                raise InvalidTransitionError(
                    from_state.name, target_state.name,
                    "Transition guard rejected"
                )
        else:
            transition = self._find_transition(from_state, target_state)

        # Execute transition
        duration = self.time_in_state

        # Exit callback
        if from_state.on_exit:
            try:
                from_state.on_exit(self)
            except Exception as e:
                logger.warning(f"State exit callback failed: {e}")

        # Transition action
        if transition and transition.action:
            try:
                transition.execute(self)
            except Exception as e:
                logger.warning(f"Transition action failed: {e}")

        # Update state
        self._current_state = target_state
        self._state_entered_at = time.time()

        # Enter callback
        if target_state.on_enter:
            try:
                target_state.on_enter(self)
            except Exception as e:
                logger.warning(f"State enter callback failed: {e}")

        # Record history
        if self._record_history:
            self._record_transition(from_state, target_state, transition, duration)

        logger.debug(
            f"State transition: {from_state.name} -> {target_state.name}"
        )

        return True

    def _resolve_state(self, state: Union[State, str]) -> State:
        """Resolve a state reference to a State object."""
        if isinstance(state, State):
            return state
        if state in self._states:
            return self._states[state]
        raise StateMachineError(f"Unknown state: {state}")

    def _find_transition(
        self,
        from_state: State,
        to_state: State,
    ) -> Transition | None:
        """Find a transition between two states."""
        transitions = self._transitions.get(from_state.name, [])
        for t in transitions:
            if t.to_state.name == to_state.name:
                return t
        return None

    def _record_transition(
        self,
        from_state: State,
        to_state: State,
        transition: Transition | None,
        duration: float,
    ) -> None:
        """Record a state transition in history."""
        record = StateHistory(
            from_state=from_state.name,
            to_state=to_state.name,
            timestamp=time.time(),
            transition_name=transition.name if transition else None,
            duration_in_previous=duration,
        )
        self._history.append(record)

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_valid_transitions(self) -> list[str]:
        """Get list of valid target states from current state."""
        if self._current_state.terminal:
            return []

        valid = []
        for t in self._transitions.get(self._current_state.name, []):
            if t.can_execute(self):
                valid.append(t.to_state.name)
        return valid

    def reset(self) -> None:
        """Reset to initial state."""
        initial = self._find_initial_state()
        self._current_state = initial
        self._state_entered_at = time.time()
        self._history.clear()


# =============================================================================
# Pre-built State Sets
# =============================================================================

class CoordinatorStates:
    """Common states for coordinators."""
    INITIALIZING = State("initializing", initial=True)
    ACTIVE = State("active")
    PAUSED = State("paused")
    STOPPING = State("stopping")
    STOPPED = State("stopped", terminal=True)
    FAILED = State("failed", terminal=True)

    TRANSITIONS = [
        Transition(INITIALIZING, ACTIVE),
        Transition(INITIALIZING, FAILED),
        Transition(ACTIVE, PAUSED),
        Transition(ACTIVE, STOPPING),
        Transition(ACTIVE, FAILED),
        Transition(PAUSED, ACTIVE),
        Transition(PAUSED, STOPPING),
        Transition(STOPPING, STOPPED),
        Transition(STOPPING, FAILED),
    ]


class OrchestratorStates:
    """Common states for orchestrators."""
    IDLE = State("idle", initial=True)
    STARTING = State("starting")
    RUNNING = State("running")
    DRAINING = State("draining")
    STOPPED = State("stopped", terminal=True)
    FAILED = State("failed", terminal=True)

    TRANSITIONS = [
        Transition(IDLE, STARTING),
        Transition(STARTING, RUNNING),
        Transition(STARTING, FAILED),
        Transition(RUNNING, DRAINING),
        Transition(RUNNING, FAILED),
        Transition(DRAINING, STOPPED),
        Transition(DRAINING, FAILED),
        Transition(IDLE, STOPPED),
    ]


class PipelineStates:
    """Common states for training pipeline."""
    IDLE = State("idle", initial=True)
    SELFPLAY = State("selfplay")
    DATA_SYNC = State("data_sync")
    TRAINING = State("training")
    EVALUATION = State("evaluation")
    PROMOTION = State("promotion")
    FAILED = State("failed")

    TRANSITIONS = [
        Transition(IDLE, SELFPLAY),
        Transition(SELFPLAY, DATA_SYNC),
        Transition(SELFPLAY, FAILED),
        Transition(DATA_SYNC, TRAINING),
        Transition(DATA_SYNC, FAILED),
        Transition(TRAINING, EVALUATION),
        Transition(TRAINING, FAILED),
        Transition(EVALUATION, PROMOTION),
        Transition(EVALUATION, IDLE),
        Transition(EVALUATION, FAILED),
        Transition(PROMOTION, IDLE),
        Transition(PROMOTION, FAILED),
        Transition(FAILED, IDLE),
    ]
