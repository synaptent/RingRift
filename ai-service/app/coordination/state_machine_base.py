"""State machine base classes for coordination infrastructure.

December 29, 2025: Created to consolidate state management patterns across:
- SelfplayScheduler (priority state, allocation state)
- DaemonManager (daemon lifecycle state)
- FeedbackLoopController (feedback signal state)

This module provides:
- StateMachineBase: Generic state machine with transitions and validation
- StateTransition: Typed transition definition with guards
- StateHistory: Bounded history of state changes for debugging
- state_machine decorator for simple enum-based state machines

Usage:
    from app.coordination.state_machine_base import (
        StateMachineBase,
        StateTransition,
        StateHistory,
        state_machine,
    )

    # Simple usage with decorator
    @state_machine
    class MyStateMachine(StateMachineBase[MyState]):
        INITIAL_STATE = MyState.IDLE

        TRANSITIONS = {
            MyState.IDLE: [MyState.RUNNING, MyState.ERROR],
            MyState.RUNNING: [MyState.IDLE, MyState.PAUSED, MyState.ERROR],
            MyState.PAUSED: [MyState.RUNNING, MyState.IDLE],
            MyState.ERROR: [MyState.IDLE],
        }

    # Advanced usage with guards and callbacks
    class AdvancedMachine(StateMachineBase[JobState]):
        def __init__(self):
            super().__init__(
                initial_state=JobState.PENDING,
                transitions={
                    JobState.PENDING: [
                        StateTransition(JobState.RUNNING, guard=self._can_start),
                        StateTransition(JobState.CANCELLED),
                    ],
                    JobState.RUNNING: [
                        StateTransition(JobState.COMPLETED, guard=self._is_done),
                        StateTransition(JobState.FAILED),
                    ],
                },
                on_enter={
                    JobState.RUNNING: self._on_start,
                    JobState.COMPLETED: self._on_complete,
                },
                on_exit={
                    JobState.RUNNING: self._on_stop,
                },
            )
"""

from __future__ import annotations

import logging
import time
from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# Type variable for state enums
S = TypeVar("S", bound=Enum)


# =============================================================================
# State Transition Definition
# =============================================================================


@dataclass
class StateTransition(Generic[S]):
    """Definition of a valid state transition.

    Attributes:
        target: Target state for this transition
        guard: Optional callable that returns True if transition is allowed
        on_transition: Optional callback invoked during transition
        name: Optional name for logging/debugging
    """

    target: S
    guard: Callable[[], bool] | None = None
    on_transition: Callable[[S, S], None] | None = None
    name: str | None = None

    def is_allowed(self) -> bool:
        """Check if this transition is currently allowed."""
        if self.guard is None:
            return True
        try:
            return self.guard()
        except Exception as e:
            logger.warning(f"Guard check failed for transition to {self.target}: {e}")
            return False


# =============================================================================
# State History
# =============================================================================


@dataclass
class StateHistoryEntry(Generic[S]):
    """Single entry in state history."""

    from_state: S
    to_state: S
    timestamp: float
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StateHistory(Generic[S]):
    """Bounded history of state changes for debugging and analysis.

    Attributes:
        max_entries: Maximum entries to keep (default: 100)
        entries: Deque of StateHistoryEntry objects
    """

    def __init__(self, max_entries: int = 100) -> None:
        self.max_entries = max_entries
        self.entries: deque[StateHistoryEntry[S]] = deque(maxlen=max_entries)
        self._transition_counts: dict[tuple[S, S], int] = {}

    def record(
        self,
        from_state: S,
        to_state: S,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a state transition."""
        entry = StateHistoryEntry(
            from_state=from_state,
            to_state=to_state,
            timestamp=time.time(),
            reason=reason,
            metadata=metadata or {},
        )
        self.entries.append(entry)

        # Track transition counts
        key = (from_state, to_state)
        self._transition_counts[key] = self._transition_counts.get(key, 0) + 1

    def get_recent(self, n: int = 10) -> list[StateHistoryEntry[S]]:
        """Get the N most recent entries."""
        return list(self.entries)[-n:]

    def get_time_in_state(self, state: S) -> float:
        """Calculate total time spent in a given state."""
        total = 0.0
        entries = list(self.entries)

        for i, entry in enumerate(entries):
            if entry.to_state == state:
                # Find when we left this state
                exit_time = None
                for j in range(i + 1, len(entries)):
                    if entries[j].from_state == state:
                        exit_time = entries[j].timestamp
                        break
                if exit_time is None:
                    # Still in this state
                    exit_time = time.time()
                total += exit_time - entry.timestamp

        return total

    def get_transition_count(self, from_state: S, to_state: S) -> int:
        """Get count of specific transition."""
        return self._transition_counts.get((from_state, to_state), 0)

    def clear(self) -> None:
        """Clear all history."""
        self.entries.clear()
        self._transition_counts.clear()

    def to_dict(self) -> dict[str, Any]:
        """Export history as dictionary for serialization."""
        return {
            "entries": [
                {
                    "from": e.from_state.name if hasattr(e.from_state, "name") else str(e.from_state),
                    "to": e.to_state.name if hasattr(e.to_state, "name") else str(e.to_state),
                    "timestamp": e.timestamp,
                    "reason": e.reason,
                    "metadata": e.metadata,
                }
                for e in self.entries
            ],
            "transition_counts": {
                f"{k[0].name}->{k[1].name}": v
                for k, v in self._transition_counts.items()
            },
        }


# =============================================================================
# State Machine Base Class
# =============================================================================


class StateMachineBase(ABC, Generic[S]):
    """Base class for state machines with validated transitions.

    Provides:
    - State transition validation against allowed transitions
    - Optional guard functions for conditional transitions
    - On-enter and on-exit callbacks
    - State history tracking
    - Event emission on state changes

    Subclasses should define:
    - INITIAL_STATE: The starting state
    - TRANSITIONS: Dict mapping states to allowed transitions

    Example:
        class JobStateMachine(StateMachineBase[JobState]):
            INITIAL_STATE = JobState.PENDING
            TRANSITIONS = {
                JobState.PENDING: [JobState.RUNNING, JobState.CANCELLED],
                JobState.RUNNING: [JobState.COMPLETED, JobState.FAILED],
                JobState.COMPLETED: [],
                JobState.FAILED: [JobState.PENDING],
                JobState.CANCELLED: [],
            }
    """

    # Subclasses should override these
    INITIAL_STATE: S = None  # type: ignore
    TRANSITIONS: dict[S, list[S | StateTransition[S]]] = {}

    def __init__(
        self,
        initial_state: S | None = None,
        transitions: dict[S, list[S | StateTransition[S]]] | None = None,
        on_enter: dict[S, Callable[[S], None]] | None = None,
        on_exit: dict[S, Callable[[S], None]] | None = None,
        history_size: int = 100,
        emit_events: bool = True,
    ) -> None:
        """Initialize state machine.

        Args:
            initial_state: Starting state (defaults to class INITIAL_STATE)
            transitions: Transition map (defaults to class TRANSITIONS)
            on_enter: Callbacks when entering states
            on_exit: Callbacks when exiting states
            history_size: Max history entries to keep
            emit_events: Whether to emit events on state changes
        """
        self._current_state: S = initial_state or self.INITIAL_STATE
        self._transitions = transitions or self.TRANSITIONS
        self._on_enter = on_enter or {}
        self._on_exit = on_exit or {}
        self._history = StateHistory[S](max_entries=history_size)
        self._emit_events = emit_events
        self._state_enter_time: float = time.time()
        self._locked = False

        if self._current_state is None:
            raise ValueError("Initial state must be specified")

    @property
    def state(self) -> S:
        """Current state."""
        return self._current_state

    @property
    def state_name(self) -> str:
        """Current state name as string."""
        return self._current_state.name if hasattr(self._current_state, "name") else str(self._current_state)

    @property
    def time_in_current_state(self) -> float:
        """Seconds spent in current state."""
        return time.time() - self._state_enter_time

    @property
    def history(self) -> StateHistory[S]:
        """State transition history."""
        return self._history

    def can_transition_to(self, target: S) -> bool:
        """Check if transition to target state is allowed.

        Args:
            target: Target state to check

        Returns:
            True if transition is allowed and any guards pass
        """
        allowed = self._transitions.get(self._current_state, [])

        for transition in allowed:
            if isinstance(transition, StateTransition):
                if transition.target == target:
                    return transition.is_allowed()
            elif transition == target:
                return True

        return False

    def get_allowed_transitions(self) -> list[S]:
        """Get list of states we can currently transition to."""
        allowed = self._transitions.get(self._current_state, [])
        result = []

        for transition in allowed:
            if isinstance(transition, StateTransition):
                if transition.is_allowed():
                    result.append(transition.target)
            else:
                result.append(transition)

        return result

    def transition_to(
        self,
        target: S,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """Attempt to transition to target state.

        Args:
            target: Target state
            reason: Optional reason for transition (for logging/history)
            metadata: Optional metadata to store in history
            force: If True, skip validation (use with caution)

        Returns:
            True if transition succeeded, False otherwise
        """
        if self._locked:
            logger.warning(
                f"State machine locked, cannot transition from {self.state_name} to {target}"
            )
            return False

        from_state = self._current_state

        # Already in target state
        if from_state == target:
            return True

        # Check if transition is allowed
        if not force and not self.can_transition_to(target):
            logger.warning(
                f"Invalid transition: {from_state} -> {target}. "
                f"Allowed: {self.get_allowed_transitions()}"
            )
            return False

        # Find transition definition for callbacks
        transition_def: StateTransition[S] | None = None
        allowed = self._transitions.get(from_state, [])
        for t in allowed:
            if isinstance(t, StateTransition) and t.target == target:
                transition_def = t
                break

        try:
            # Exit callback
            if from_state in self._on_exit:
                self._on_exit[from_state](target)

            # Transition callback
            if transition_def and transition_def.on_transition:
                transition_def.on_transition(from_state, target)

            # Update state
            self._current_state = target
            self._state_enter_time = time.time()

            # Record history
            self._history.record(
                from_state=from_state,
                to_state=target,
                reason=reason,
                metadata=metadata,
            )

            # Enter callback
            if target in self._on_enter:
                self._on_enter[target](from_state)

            # Emit event
            if self._emit_events:
                self._emit_state_change_event(from_state, target, reason)

            logger.debug(
                f"State transition: {from_state} -> {target}"
                + (f" (reason: {reason})" if reason else "")
            )
            return True

        except Exception as e:
            logger.error(f"Error during transition {from_state} -> {target}: {e}")
            # Rollback
            self._current_state = from_state
            return False

    def _emit_state_change_event(
        self,
        from_state: S,
        to_state: S,
        reason: str | None,
    ) -> None:
        """Emit event on state change. Override in subclasses for custom events."""
        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus:
                bus.publish(
                    "STATE_CHANGED",
                    {
                        "machine": self.__class__.__name__,
                        "from_state": from_state.name if hasattr(from_state, "name") else str(from_state),
                        "to_state": to_state.name if hasattr(to_state, "name") else str(to_state),
                        "reason": reason,
                        "timestamp": time.time(),
                    },
                )
        except Exception:
            pass  # Event emission is optional

    def lock(self) -> None:
        """Lock state machine to prevent transitions."""
        self._locked = True

    def unlock(self) -> None:
        """Unlock state machine to allow transitions."""
        self._locked = False

    def reset(self, to_state: S | None = None) -> None:
        """Reset state machine to initial or specified state.

        Args:
            to_state: State to reset to (defaults to INITIAL_STATE)
        """
        target = to_state or self.INITIAL_STATE
        self._current_state = target
        self._state_enter_time = time.time()
        self._locked = False
        self._history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get state machine statistics."""
        return {
            "current_state": self.state_name,
            "time_in_state_seconds": self.time_in_current_state,
            "locked": self._locked,
            "history_size": len(self._history.entries),
            "allowed_transitions": [
                s.name if hasattr(s, "name") else str(s)
                for s in self.get_allowed_transitions()
            ],
        }


# =============================================================================
# Composite State Machine
# =============================================================================


class CompositeStateMachine(Generic[S]):
    """Manages multiple related state machines as a group.

    Useful when a system has multiple independent state dimensions that
    need to be tracked together (e.g., job state + resource state).

    Example:
        composite = CompositeStateMachine()
        composite.add("job", JobStateMachine())
        composite.add("resource", ResourceStateMachine())

        # Transition individual machines
        composite.transition("job", JobState.RUNNING)

        # Get combined state
        combined = composite.get_combined_state()
    """

    def __init__(self) -> None:
        self._machines: dict[str, StateMachineBase[Any]] = {}

    def add(self, name: str, machine: StateMachineBase[Any]) -> None:
        """Add a state machine to the composite."""
        self._machines[name] = machine

    def get(self, name: str) -> StateMachineBase[Any] | None:
        """Get a state machine by name."""
        return self._machines.get(name)

    def transition(
        self,
        machine_name: str,
        target: Any,
        reason: str | None = None,
    ) -> bool:
        """Transition a specific machine."""
        machine = self._machines.get(machine_name)
        if machine is None:
            logger.warning(f"Unknown state machine: {machine_name}")
            return False
        return machine.transition_to(target, reason=reason)

    def get_combined_state(self) -> dict[str, str]:
        """Get current state of all machines."""
        return {
            name: machine.state_name
            for name, machine in self._machines.items()
        }

    def get_combined_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all machines."""
        return {
            name: machine.get_stats()
            for name, machine in self._machines.items()
        }


# =============================================================================
# Decorator for Simple State Machines
# =============================================================================


def state_machine(cls: type) -> type:
    """Decorator to add state machine boilerplate to a class.

    The decorated class should define:
    - INITIAL_STATE: Starting state enum value
    - TRANSITIONS: Dict mapping states to allowed target states

    Example:
        @state_machine
        class MyMachine(StateMachineBase[MyState]):
            INITIAL_STATE = MyState.IDLE
            TRANSITIONS = {
                MyState.IDLE: [MyState.RUNNING],
                MyState.RUNNING: [MyState.IDLE, MyState.DONE],
                MyState.DONE: [],
            }
    """
    original_init = cls.__init__

    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Call StateMachineBase init with class-level config
        StateMachineBase.__init__(
            self,
            initial_state=getattr(cls, "INITIAL_STATE", None),
            transitions=getattr(cls, "TRANSITIONS", {}),
        )
        # Call original init
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


# =============================================================================
# Priority State (used by SelfplayScheduler)
# =============================================================================


class PriorityState(Enum):
    """Priority levels for selfplay allocation."""

    CRITICAL = "critical"      # Immediate attention needed (regression, data starvation)
    HIGH = "high"              # Above-average priority (low Elo velocity, stale data)
    NORMAL = "normal"          # Standard priority
    LOW = "low"                # Below-average priority (already performing well)
    PAUSED = "paused"          # Temporarily paused (backpressure, cooldown)


class AllocationState(Enum):
    """Allocation states for selfplay scheduling."""

    IDLE = "idle"              # No selfplay running for this config
    SCHEDULED = "scheduled"    # Selfplay job scheduled, not yet started
    RUNNING = "running"        # Selfplay actively running
    COMPLETING = "completing"  # Selfplay finishing, data being collected
    COOLDOWN = "cooldown"      # Post-selfplay cooldown period


class PriorityStateMachine(StateMachineBase[PriorityState]):
    """State machine for config priority management.

    Used by SelfplayScheduler to track config priority transitions.
    """

    INITIAL_STATE = PriorityState.NORMAL

    TRANSITIONS = {
        PriorityState.CRITICAL: [
            PriorityState.HIGH,
            PriorityState.NORMAL,
            PriorityState.PAUSED,
        ],
        PriorityState.HIGH: [
            PriorityState.CRITICAL,
            PriorityState.NORMAL,
            PriorityState.LOW,
            PriorityState.PAUSED,
        ],
        PriorityState.NORMAL: [
            PriorityState.CRITICAL,
            PriorityState.HIGH,
            PriorityState.LOW,
            PriorityState.PAUSED,
        ],
        PriorityState.LOW: [
            PriorityState.NORMAL,
            PriorityState.HIGH,
            PriorityState.CRITICAL,
            PriorityState.PAUSED,
        ],
        PriorityState.PAUSED: [
            PriorityState.NORMAL,
            PriorityState.LOW,
        ],
    }


class AllocationStateMachine(StateMachineBase[AllocationState]):
    """State machine for selfplay allocation lifecycle.

    Used by SelfplayScheduler to track job allocation states.
    """

    INITIAL_STATE = AllocationState.IDLE

    TRANSITIONS = {
        AllocationState.IDLE: [
            AllocationState.SCHEDULED,
        ],
        AllocationState.SCHEDULED: [
            AllocationState.RUNNING,
            AllocationState.IDLE,  # Cancelled before start
        ],
        AllocationState.RUNNING: [
            AllocationState.COMPLETING,
            AllocationState.IDLE,  # Failed/cancelled
        ],
        AllocationState.COMPLETING: [
            AllocationState.COOLDOWN,
            AllocationState.IDLE,  # Error during completion
        ],
        AllocationState.COOLDOWN: [
            AllocationState.IDLE,
        ],
    }


# =============================================================================
# Daemon Lifecycle State (used by DaemonManager)
# =============================================================================


class DaemonLifecycleState(Enum):
    """Lifecycle states for daemon management."""

    UNINITIALIZED = "uninitialized"  # Not yet created
    INITIALIZED = "initialized"       # Created but not started
    STARTING = "starting"             # Start in progress
    RUNNING = "running"               # Actively running
    PAUSING = "pausing"               # Pause in progress
    PAUSED = "paused"                 # Temporarily paused
    STOPPING = "stopping"             # Stop in progress
    STOPPED = "stopped"               # Cleanly stopped
    FAILED = "failed"                 # Failed with error
    RESTARTING = "restarting"         # Restart in progress


class DaemonLifecycleMachine(StateMachineBase[DaemonLifecycleState]):
    """State machine for daemon lifecycle management.

    Used by DaemonManager to track daemon lifecycle transitions.
    """

    INITIAL_STATE = DaemonLifecycleState.UNINITIALIZED

    TRANSITIONS = {
        DaemonLifecycleState.UNINITIALIZED: [
            DaemonLifecycleState.INITIALIZED,
        ],
        DaemonLifecycleState.INITIALIZED: [
            DaemonLifecycleState.STARTING,
        ],
        DaemonLifecycleState.STARTING: [
            DaemonLifecycleState.RUNNING,
            DaemonLifecycleState.FAILED,
        ],
        DaemonLifecycleState.RUNNING: [
            DaemonLifecycleState.PAUSING,
            DaemonLifecycleState.STOPPING,
            DaemonLifecycleState.FAILED,
            DaemonLifecycleState.RESTARTING,
        ],
        DaemonLifecycleState.PAUSING: [
            DaemonLifecycleState.PAUSED,
            DaemonLifecycleState.FAILED,
        ],
        DaemonLifecycleState.PAUSED: [
            DaemonLifecycleState.STARTING,
            DaemonLifecycleState.STOPPING,
        ],
        DaemonLifecycleState.STOPPING: [
            DaemonLifecycleState.STOPPED,
            DaemonLifecycleState.FAILED,
        ],
        DaemonLifecycleState.STOPPED: [
            DaemonLifecycleState.INITIALIZED,
        ],
        DaemonLifecycleState.FAILED: [
            DaemonLifecycleState.INITIALIZED,
            DaemonLifecycleState.RESTARTING,
        ],
        DaemonLifecycleState.RESTARTING: [
            DaemonLifecycleState.STARTING,
            DaemonLifecycleState.FAILED,
        ],
    }


# =============================================================================
# Feedback Signal State (used by FeedbackLoopController)
# =============================================================================


class FeedbackSignalState(Enum):
    """States for feedback signal processing."""

    COLLECTING = "collecting"        # Gathering metrics
    ANALYZING = "analyzing"          # Computing feedback signals
    READY = "ready"                  # Signals computed, ready to apply
    APPLYING = "applying"            # Applying feedback to training
    APPLIED = "applied"              # Feedback successfully applied
    COOLDOWN = "cooldown"            # Waiting before next feedback cycle


class FeedbackSignalMachine(StateMachineBase[FeedbackSignalState]):
    """State machine for feedback signal processing.

    Used by FeedbackLoopController to track feedback cycle states.
    """

    INITIAL_STATE = FeedbackSignalState.COLLECTING

    TRANSITIONS = {
        FeedbackSignalState.COLLECTING: [
            FeedbackSignalState.ANALYZING,
        ],
        FeedbackSignalState.ANALYZING: [
            FeedbackSignalState.READY,
            FeedbackSignalState.COLLECTING,  # Not enough data
        ],
        FeedbackSignalState.READY: [
            FeedbackSignalState.APPLYING,
            FeedbackSignalState.COLLECTING,  # Skipped
        ],
        FeedbackSignalState.APPLYING: [
            FeedbackSignalState.APPLIED,
            FeedbackSignalState.COLLECTING,  # Failed
        ],
        FeedbackSignalState.APPLIED: [
            FeedbackSignalState.COOLDOWN,
        ],
        FeedbackSignalState.COOLDOWN: [
            FeedbackSignalState.COLLECTING,
        ],
    }


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Base classes
    "StateMachineBase",
    "StateTransition",
    "StateHistory",
    "StateHistoryEntry",
    "CompositeStateMachine",
    # Decorator
    "state_machine",
    # Priority states (SelfplayScheduler)
    "PriorityState",
    "AllocationState",
    "PriorityStateMachine",
    "AllocationStateMachine",
    # Daemon states (DaemonManager)
    "DaemonLifecycleState",
    "DaemonLifecycleMachine",
    # Feedback states (FeedbackLoopController)
    "FeedbackSignalState",
    "FeedbackSignalMachine",
]
