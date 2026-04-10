"""Job State Machine - Centralized job lifecycle state management.

January 2026 - Sprint 17: Extracted from job_loops.py to provide a reusable
state machine for job lifecycle management across all P2P components.

This module provides:
- Valid state transition definitions
- Transition validation
- Event emission on state changes
- Consistent error handling for invalid transitions

State Transition Diagram:
    PENDING -> CLAIMED -> STARTING -> RUNNING -> COMPLETED
                           |           |
                           v           v
                         STALE     ORPHANED / STUCK
                           |           |
                           v           v
                         FAILED      FAILED

    Any state -> CANCELLED (explicit cancellation)
    Any non-terminal state -> FAILED (on error)

Usage:
    from scripts.p2p.job_state_machine import (
        JobStateMachine,
        TransitionResult,
        get_job_state_machine,
    )

    # Get singleton instance
    sm = get_job_state_machine()

    # Transition a job
    result = sm.transition(
        job_id="job-123",
        current_state=JobLifecycleState.PENDING,
        new_state=JobLifecycleState.CLAIMED,
        node_id="worker-1",
        reason="worker claimed job",
    )

    if result.success:
        print(f"Transitioned to {result.new_state}")
    else:
        print(f"Invalid transition: {result.error}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .types import JobLifecycleState

logger = logging.getLogger(__name__)


# Valid state transitions: from_state -> [allowed_to_states]
VALID_TRANSITIONS: dict[JobLifecycleState, set[JobLifecycleState]] = {
    # Initial state - can be claimed or cancelled
    JobLifecycleState.PENDING: {
        JobLifecycleState.CLAIMED,
        JobLifecycleState.CANCELLED,
        JobLifecycleState.FAILED,  # Queue expired
    },
    # Claimed - can start, go stale, or be cancelled
    JobLifecycleState.CLAIMED: {
        JobLifecycleState.STARTING,
        JobLifecycleState.STALE,
        JobLifecycleState.CANCELLED,
        JobLifecycleState.FAILED,
    },
    # Starting - can run, fail, or be cancelled
    JobLifecycleState.STARTING: {
        JobLifecycleState.RUNNING,
        JobLifecycleState.VERIFIED,  # Spawn confirmed
        JobLifecycleState.FAILED,
        JobLifecycleState.CANCELLED,
    },
    # Running - can complete, fail, get stuck/orphaned
    JobLifecycleState.RUNNING: {
        JobLifecycleState.COMPLETED,
        JobLifecycleState.FAILED,
        JobLifecycleState.STUCK,
        JobLifecycleState.ORPHANED,
        JobLifecycleState.CANCELLED,
        JobLifecycleState.VERIFIED,  # Re-verification
    },
    # Verified - same as running (just confirmed)
    JobLifecycleState.VERIFIED: {
        JobLifecycleState.COMPLETED,
        JobLifecycleState.FAILED,
        JobLifecycleState.STUCK,
        JobLifecycleState.ORPHANED,
        JobLifecycleState.CANCELLED,
    },
    # Problem states - can only fail or be reassigned back to pending
    JobLifecycleState.STALE: {
        JobLifecycleState.FAILED,
        JobLifecycleState.PENDING,  # Reassignment
        JobLifecycleState.CANCELLED,
    },
    JobLifecycleState.STUCK: {
        JobLifecycleState.FAILED,
        JobLifecycleState.RUNNING,  # Unstuck (recovered)
        JobLifecycleState.CANCELLED,
    },
    JobLifecycleState.ORPHANED: {
        JobLifecycleState.FAILED,
        JobLifecycleState.PENDING,  # Reassignment
        JobLifecycleState.CANCELLED,
    },
    # Terminal states - no transitions allowed
    JobLifecycleState.COMPLETED: set(),
    JobLifecycleState.FAILED: set(),
    JobLifecycleState.CANCELLED: set(),
}


class TransitionType(str, Enum):
    """Types of state transitions for event categorization."""

    PROGRESS = "progress"  # Normal forward progress (pending -> claimed -> running)
    COMPLETION = "completion"  # Successful completion
    FAILURE = "failure"  # Any failure transition
    CANCELLATION = "cancellation"  # Explicit cancellation
    PROBLEM = "problem"  # Transition to problem state (stale, stuck, orphaned)
    RECOVERY = "recovery"  # Recovery from problem state


@dataclass
class TransitionResult:
    """Result of a state transition attempt."""

    success: bool
    job_id: str
    old_state: JobLifecycleState
    new_state: JobLifecycleState | None
    transition_type: TransitionType | None = None
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success


@dataclass
class TransitionEvent:
    """Event emitted when a job state transitions."""

    job_id: str
    old_state: JobLifecycleState
    new_state: JobLifecycleState
    transition_type: TransitionType
    node_id: str | None
    reason: str | None
    timestamp: float
    metadata: dict[str, Any]


class JobStateMachine:
    """Centralized state machine for job lifecycle management.

    Provides:
    - Valid transition enforcement
    - Event emission on transitions
    - Transition history tracking
    - Statistics for monitoring

    Thread-safe for concurrent job operations.
    """

    def __init__(
        self,
        emit_events: bool = True,
        event_emitter: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize the state machine.

        Args:
            emit_events: Whether to emit events on state transitions.
            event_emitter: Custom event emitter function. If None, uses
                the default event router.
        """
        self._emit_events = emit_events
        self._event_emitter = event_emitter

        # Statistics tracking
        self._transition_counts: dict[str, int] = {}
        self._failure_counts: dict[str, int] = {}
        self._last_transition_time: dict[str, float] = {}

        # Lazy-loaded event emission
        self._emit_fn: Callable[[str, dict[str, Any]], None] | None = None

    def _get_emit_fn(self) -> Callable[[str, dict[str, Any]], None]:
        """Lazy load event emission function."""
        if self._emit_fn is None:
            if self._event_emitter is not None:
                self._emit_fn = self._event_emitter
            else:
                try:
                    from app.coordination.event_router import emit_event

                    self._emit_fn = emit_event
                except ImportError:
                    # Fallback: no-op if event router not available
                    self._emit_fn = lambda *args, **kwargs: None
        return self._emit_fn

    def can_transition(
        self,
        current_state: JobLifecycleState | str,
        new_state: JobLifecycleState | str,
    ) -> bool:
        """Check if a transition is valid.

        Args:
            current_state: Current job state.
            new_state: Desired new state.

        Returns:
            True if the transition is allowed.
        """
        # Convert strings to enums
        if isinstance(current_state, str):
            try:
                current_state = JobLifecycleState(current_state)
            except ValueError:
                return False
        if isinstance(new_state, str):
            try:
                new_state = JobLifecycleState(new_state)
            except ValueError:
                return False

        # Same state is always "valid" (no-op)
        if current_state == new_state:
            return True

        # Check valid transitions
        allowed = VALID_TRANSITIONS.get(current_state, set())
        return new_state in allowed

    def get_transition_type(
        self,
        old_state: JobLifecycleState,
        new_state: JobLifecycleState,
    ) -> TransitionType:
        """Categorize a transition for event routing."""
        if new_state == JobLifecycleState.COMPLETED:
            return TransitionType.COMPLETION
        elif new_state == JobLifecycleState.FAILED:
            return TransitionType.FAILURE
        elif new_state == JobLifecycleState.CANCELLED:
            return TransitionType.CANCELLATION
        elif new_state.is_problem():
            return TransitionType.PROBLEM
        elif old_state.is_problem() and not new_state.is_problem():
            return TransitionType.RECOVERY
        else:
            return TransitionType.PROGRESS

    def transition(
        self,
        job_id: str,
        current_state: JobLifecycleState | str,
        new_state: JobLifecycleState | str,
        *,
        node_id: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> TransitionResult:
        """Attempt to transition a job to a new state.

        Args:
            job_id: Unique job identifier.
            current_state: Current job state.
            new_state: Desired new state.
            node_id: Node performing the transition.
            reason: Human-readable reason for transition.
            metadata: Additional metadata to include in events.
            force: If True, bypass validation (use with caution).

        Returns:
            TransitionResult indicating success or failure.
        """
        timestamp = time.time()
        metadata = metadata or {}

        # Convert strings to enums
        if isinstance(current_state, str):
            try:
                current_state = JobLifecycleState(current_state)
            except ValueError:
                return TransitionResult(
                    success=False,
                    job_id=job_id,
                    old_state=current_state,  # type: ignore
                    new_state=None,
                    error=f"Invalid current state: {current_state}",
                    timestamp=timestamp,
                    metadata=metadata,
                )
        if isinstance(new_state, str):
            try:
                new_state = JobLifecycleState(new_state)
            except ValueError:
                return TransitionResult(
                    success=False,
                    job_id=job_id,
                    old_state=current_state,
                    new_state=None,
                    error=f"Invalid new state: {new_state}",
                    timestamp=timestamp,
                    metadata=metadata,
                )

        # Same state is a no-op success
        if current_state == new_state:
            return TransitionResult(
                success=True,
                job_id=job_id,
                old_state=current_state,
                new_state=new_state,
                transition_type=None,
                timestamp=timestamp,
                metadata=metadata,
            )

        # Validate transition
        if not force and not self.can_transition(current_state, new_state):
            error_msg = (
                f"Invalid transition: {current_state.value} -> {new_state.value}. "
                f"Allowed: {[s.value for s in VALID_TRANSITIONS.get(current_state, set())]}"
            )
            logger.warning(f"[JobStateMachine] {error_msg} for job {job_id}")

            # Track failed transitions
            key = f"{current_state.value}->{new_state.value}"
            self._failure_counts[key] = self._failure_counts.get(key, 0) + 1

            return TransitionResult(
                success=False,
                job_id=job_id,
                old_state=current_state,
                new_state=new_state,
                error=error_msg,
                timestamp=timestamp,
                metadata=metadata,
            )

        # Valid transition
        transition_type = self.get_transition_type(current_state, new_state)

        # Track statistics
        key = f"{current_state.value}->{new_state.value}"
        self._transition_counts[key] = self._transition_counts.get(key, 0) + 1
        self._last_transition_time[job_id] = timestamp

        # Emit event if enabled
        if self._emit_events:
            self._emit_transition_event(
                job_id=job_id,
                old_state=current_state,
                new_state=new_state,
                transition_type=transition_type,
                node_id=node_id,
                reason=reason,
                timestamp=timestamp,
                metadata=metadata,
            )

        logger.debug(
            f"[JobStateMachine] Job {job_id}: {current_state.value} -> {new_state.value} "
            f"({transition_type.value}){' by ' + node_id if node_id else ''}"
        )

        return TransitionResult(
            success=True,
            job_id=job_id,
            old_state=current_state,
            new_state=new_state,
            transition_type=transition_type,
            timestamp=timestamp,
            metadata=metadata,
        )

    def _emit_transition_event(
        self,
        job_id: str,
        old_state: JobLifecycleState,
        new_state: JobLifecycleState,
        transition_type: TransitionType,
        node_id: str | None,
        reason: str | None,
        timestamp: float,
        metadata: dict[str, Any],
    ) -> None:
        """Emit an event for a state transition."""
        try:
            emit_fn = self._get_emit_fn()

            # Map transition type to event name
            event_map = {
                TransitionType.PROGRESS: "JOB_STATE_CHANGED",
                TransitionType.COMPLETION: "JOB_COMPLETED",
                TransitionType.FAILURE: "JOB_FAILED",
                TransitionType.CANCELLATION: "JOB_CANCELLED",
                TransitionType.PROBLEM: "JOB_PROBLEM_DETECTED",
                TransitionType.RECOVERY: "JOB_RECOVERED",
            }
            event_name = event_map.get(transition_type, "JOB_STATE_CHANGED")

            payload = {
                "job_id": job_id,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "transition_type": transition_type.value,
                "node_id": node_id,
                "reason": reason,
                "timestamp": timestamp,
                **metadata,
            }

            emit_fn(event_name, payload)

        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as e:
            logger.debug(f"[JobStateMachine] Failed to emit event: {e}")

    def get_allowed_transitions(
        self, state: JobLifecycleState | str
    ) -> set[JobLifecycleState]:
        """Get the set of states that can be transitioned to from the given state."""
        if isinstance(state, str):
            try:
                state = JobLifecycleState(state)
            except ValueError:
                return set()
        return VALID_TRANSITIONS.get(state, set())

    def get_statistics(self) -> dict[str, Any]:
        """Get state machine statistics for monitoring."""
        return {
            "transition_counts": dict(self._transition_counts),
            "failure_counts": dict(self._failure_counts),
            "total_transitions": sum(self._transition_counts.values()),
            "total_failures": sum(self._failure_counts.values()),
            "unique_jobs_tracked": len(self._last_transition_time),
        }

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self._transition_counts.clear()
        self._failure_counts.clear()
        self._last_transition_time.clear()


# Singleton instance
_state_machine: JobStateMachine | None = None


def get_job_state_machine() -> JobStateMachine:
    """Get the singleton JobStateMachine instance."""
    global _state_machine
    if _state_machine is None:
        _state_machine = JobStateMachine()
    return _state_machine


def reset_job_state_machine() -> None:
    """Reset the singleton instance (for testing)."""
    global _state_machine
    _state_machine = None


# Convenience functions for common transitions
def claim_job(
    job_id: str,
    node_id: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> TransitionResult:
    """Transition a job from PENDING to CLAIMED."""
    return get_job_state_machine().transition(
        job_id=job_id,
        current_state=JobLifecycleState.PENDING,
        new_state=JobLifecycleState.CLAIMED,
        node_id=node_id,
        reason="job claimed by worker",
        metadata=metadata,
    )


def start_job(
    job_id: str,
    node_id: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> TransitionResult:
    """Transition a job from CLAIMED to STARTING."""
    return get_job_state_machine().transition(
        job_id=job_id,
        current_state=JobLifecycleState.CLAIMED,
        new_state=JobLifecycleState.STARTING,
        node_id=node_id,
        reason="job starting",
        metadata=metadata,
    )


def mark_job_running(
    job_id: str,
    node_id: str,
    *,
    current_state: JobLifecycleState = JobLifecycleState.STARTING,
    metadata: dict[str, Any] | None = None,
) -> TransitionResult:
    """Transition a job to RUNNING."""
    return get_job_state_machine().transition(
        job_id=job_id,
        current_state=current_state,
        new_state=JobLifecycleState.RUNNING,
        node_id=node_id,
        reason="job running",
        metadata=metadata,
    )


def complete_job(
    job_id: str,
    node_id: str,
    *,
    current_state: JobLifecycleState = JobLifecycleState.RUNNING,
    metadata: dict[str, Any] | None = None,
) -> TransitionResult:
    """Transition a job to COMPLETED."""
    return get_job_state_machine().transition(
        job_id=job_id,
        current_state=current_state,
        new_state=JobLifecycleState.COMPLETED,
        node_id=node_id,
        reason="job completed successfully",
        metadata=metadata,
    )


def fail_job(
    job_id: str,
    *,
    current_state: JobLifecycleState,
    node_id: str | None = None,
    reason: str = "job failed",
    metadata: dict[str, Any] | None = None,
) -> TransitionResult:
    """Transition a job to FAILED."""
    return get_job_state_machine().transition(
        job_id=job_id,
        current_state=current_state,
        new_state=JobLifecycleState.FAILED,
        node_id=node_id,
        reason=reason,
        metadata=metadata,
    )


def cancel_job(
    job_id: str,
    *,
    current_state: JobLifecycleState,
    node_id: str | None = None,
    reason: str = "job cancelled",
    metadata: dict[str, Any] | None = None,
) -> TransitionResult:
    """Transition a job to CANCELLED."""
    return get_job_state_machine().transition(
        job_id=job_id,
        current_state=current_state,
        new_state=JobLifecycleState.CANCELLED,
        node_id=node_id,
        reason=reason,
        metadata=metadata,
    )


def mark_job_stale(
    job_id: str,
    *,
    current_state: JobLifecycleState = JobLifecycleState.CLAIMED,
    reason: str = "claim timeout exceeded",
    metadata: dict[str, Any] | None = None,
) -> TransitionResult:
    """Transition a job to STALE (claim timeout)."""
    return get_job_state_machine().transition(
        job_id=job_id,
        current_state=current_state,
        new_state=JobLifecycleState.STALE,
        reason=reason,
        metadata=metadata,
    )


def mark_job_stuck(
    job_id: str,
    *,
    current_state: JobLifecycleState = JobLifecycleState.RUNNING,
    reason: str = "execution timeout exceeded",
    metadata: dict[str, Any] | None = None,
) -> TransitionResult:
    """Transition a job to STUCK (running too long)."""
    return get_job_state_machine().transition(
        job_id=job_id,
        current_state=current_state,
        new_state=JobLifecycleState.STUCK,
        reason=reason,
        metadata=metadata,
    )


def mark_job_orphaned(
    job_id: str,
    *,
    current_state: JobLifecycleState = JobLifecycleState.RUNNING,
    reason: str = "lost contact with worker node",
    metadata: dict[str, Any] | None = None,
) -> TransitionResult:
    """Transition a job to ORPHANED (lost contact)."""
    return get_job_state_machine().transition(
        job_id=job_id,
        current_state=current_state,
        new_state=JobLifecycleState.ORPHANED,
        reason=reason,
        metadata=metadata,
    )


def reassign_job(
    job_id: str,
    *,
    current_state: JobLifecycleState,
    reason: str = "reassigning job",
    metadata: dict[str, Any] | None = None,
) -> TransitionResult:
    """Reassign a job back to PENDING for retry."""
    return get_job_state_machine().transition(
        job_id=job_id,
        current_state=current_state,
        new_state=JobLifecycleState.PENDING,
        reason=reason,
        metadata=metadata,
    )
