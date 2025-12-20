"""Optimization Orchestrator - Unified monitoring for CMAES/NAS/PBT runs (December 2025).

This module provides centralized monitoring and coordination for all optimization runs:
- CMAES (Covariance Matrix Adaptation Evolution Strategy) for hyperparameter tuning
- NAS (Neural Architecture Search) for architecture optimization
- PBT (Population Based Training) for dynamic hyperparameter scheduling

These optimization runs can consume significant compute resources ($2M+) and
previously had no unified monitoring or event handling.

Usage:
    from app.training.optimization_orchestrator import (
        OptimizationOrchestrator,
        get_optimization_orchestrator,
        wire_optimization_events,
    )

    # Wire all optimization events
    orchestrator = wire_optimization_events()

    # Check optimization status
    status = orchestrator.get_status()
    print(f"Active runs: {status['active_runs']}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization runs."""
    CMAES = "cmaes"
    NAS = "nas"
    PBT = "pbt"


class OptimizationState(Enum):
    """State of an optimization run."""
    PENDING = "pending"
    RUNNING = "running"
    GENERATION_COMPLETE = "generation_complete"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OptimizationRun:
    """Tracks a single optimization run."""
    run_id: str
    opt_type: OptimizationType
    state: OptimizationState
    config: str  # board_type_num_players config
    started_at: float
    updated_at: float
    current_generation: int = 0
    total_generations: int = 0
    best_score: float | None = None
    best_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "type": self.opt_type.value,
            "state": self.state.value,
            "config": self.config,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "current_generation": self.current_generation,
            "total_generations": self.total_generations,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "duration_seconds": time.time() - self.started_at,
        }


class OptimizationOrchestrator:
    """Unified orchestrator for all optimization runs.

    Subscribes to CMAES, NAS, and PBT events to track optimization lifecycles,
    record metrics, and coordinate resources.
    """

    def __init__(self, max_history: int = 100):
        self._active_runs: dict[str, OptimizationRun] = {}
        self._completed_runs: list[OptimizationRun] = []
        self._max_history = max_history
        self._subscribed = False
        self._event_counts: dict[str, int] = {}

    def subscribe_to_events(self) -> bool:
        """Subscribe to all optimization-related events.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()

            # CMAES events
            bus.subscribe(DataEventType.CMAES_TRIGGERED, self._on_cmaes_triggered)
            bus.subscribe(DataEventType.CMAES_COMPLETED, self._on_cmaes_completed)

            # NAS events
            bus.subscribe(DataEventType.NAS_TRIGGERED, self._on_nas_triggered)
            bus.subscribe(DataEventType.NAS_STARTED, self._on_nas_started)
            bus.subscribe(DataEventType.NAS_GENERATION_COMPLETE, self._on_nas_generation)
            bus.subscribe(DataEventType.NAS_COMPLETED, self._on_nas_completed)
            bus.subscribe(DataEventType.NAS_BEST_ARCHITECTURE, self._on_nas_best)

            # PBT events
            bus.subscribe(DataEventType.PBT_STARTED, self._on_pbt_started)
            bus.subscribe(DataEventType.PBT_GENERATION_COMPLETE, self._on_pbt_generation)
            bus.subscribe(DataEventType.PBT_COMPLETED, self._on_pbt_completed)

            self._subscribed = True
            logger.info("[OptimizationOrchestrator] Subscribed to all optimization events")
            return True

        except Exception as e:
            logger.warning(f"[OptimizationOrchestrator] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from all events."""
        if not self._subscribed:
            return

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()

            # Unsubscribe all
            for event_type in [
                DataEventType.CMAES_TRIGGERED, DataEventType.CMAES_COMPLETED,
                DataEventType.NAS_TRIGGERED, DataEventType.NAS_STARTED,
                DataEventType.NAS_GENERATION_COMPLETE, DataEventType.NAS_COMPLETED,
                DataEventType.NAS_BEST_ARCHITECTURE,
                DataEventType.PBT_STARTED, DataEventType.PBT_GENERATION_COMPLETE,
                DataEventType.PBT_COMPLETED,
            ]:
                try:
                    handler = getattr(self, f"_on_{event_type.value}")
                    bus.unsubscribe(event_type, handler)
                except (AttributeError, Exception):
                    pass

            self._subscribed = False
        except Exception:
            pass

    def _track_event(self, event_type: str) -> None:
        """Track event count for metrics."""
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1

    def _get_run_id(self, payload: dict[str, Any], opt_type: OptimizationType) -> str:
        """Generate or extract run ID from event payload."""
        if "run_id" in payload:
            return payload["run_id"]
        config = payload.get("config", payload.get("board_type", "unknown"))
        return f"{opt_type.value}_{config}_{int(time.time())}"

    # =========================================================================
    # CMAES Event Handlers
    # =========================================================================

    def _on_cmaes_triggered(self, event) -> None:
        """Handle CMAES_TRIGGERED event."""
        self._track_event("cmaes_triggered")
        payload = event.payload if hasattr(event, 'payload') else {}

        run_id = self._get_run_id(payload, OptimizationType.CMAES)
        config = payload.get("config", payload.get("board_type", "unknown"))

        run = OptimizationRun(
            run_id=run_id,
            opt_type=OptimizationType.CMAES,
            state=OptimizationState.RUNNING,
            config=config,
            started_at=time.time(),
            updated_at=time.time(),
            total_generations=payload.get("generations", 0),
            metadata=payload,
        )

        self._active_runs[run_id] = run
        logger.info(f"[OptimizationOrchestrator] CMAES started: {run_id} for {config}")
        self._emit_optimization_event("started", run)

    def _on_cmaes_completed(self, event) -> None:
        """Handle CMAES_COMPLETED event."""
        self._track_event("cmaes_completed")
        payload = event.payload if hasattr(event, 'payload') else {}

        run_id = self._find_active_run(OptimizationType.CMAES, payload)
        if run_id and run_id in self._active_runs:
            run = self._active_runs.pop(run_id)
            run.state = OptimizationState.COMPLETED
            run.updated_at = time.time()
            run.best_score = payload.get("best_score", payload.get("best_fitness"))
            run.best_params = payload.get("best_params", {})
            self._add_to_history(run)

            logger.info(
                f"[OptimizationOrchestrator] CMAES completed: {run_id} "
                f"(best_score={run.best_score})"
            )
            self._emit_optimization_event("completed", run)

    # =========================================================================
    # NAS Event Handlers
    # =========================================================================

    def _on_nas_triggered(self, event) -> None:
        """Handle NAS_TRIGGERED event."""
        self._track_event("nas_triggered")
        payload = event.payload if hasattr(event, 'payload') else {}

        run_id = self._get_run_id(payload, OptimizationType.NAS)
        config = payload.get("config", payload.get("board_type", "unknown"))

        run = OptimizationRun(
            run_id=run_id,
            opt_type=OptimizationType.NAS,
            state=OptimizationState.PENDING,
            config=config,
            started_at=time.time(),
            updated_at=time.time(),
            total_generations=payload.get("generations", 0),
            metadata=payload,
        )

        self._active_runs[run_id] = run
        logger.info(f"[OptimizationOrchestrator] NAS triggered: {run_id}")

    def _on_nas_started(self, event) -> None:
        """Handle NAS_STARTED event."""
        self._track_event("nas_started")
        payload = event.payload if hasattr(event, 'payload') else {}

        run_id = self._find_active_run(OptimizationType.NAS, payload)
        if run_id and run_id in self._active_runs:
            self._active_runs[run_id].state = OptimizationState.RUNNING
            self._active_runs[run_id].updated_at = time.time()
            logger.info(f"[OptimizationOrchestrator] NAS started: {run_id}")
            self._emit_optimization_event("started", self._active_runs[run_id])

    def _on_nas_generation(self, event) -> None:
        """Handle NAS_GENERATION_COMPLETE event."""
        self._track_event("nas_generation_complete")
        payload = event.payload if hasattr(event, 'payload') else {}

        run_id = self._find_active_run(OptimizationType.NAS, payload)
        if run_id and run_id in self._active_runs:
            run = self._active_runs[run_id]
            run.current_generation = payload.get("generation", run.current_generation + 1)
            run.updated_at = time.time()
            run.state = OptimizationState.GENERATION_COMPLETE

            if "best_score" in payload:
                run.best_score = payload["best_score"]

            logger.debug(
                f"[OptimizationOrchestrator] NAS generation {run.current_generation}: {run_id}"
            )

    def _on_nas_completed(self, event) -> None:
        """Handle NAS_COMPLETED event."""
        self._track_event("nas_completed")
        payload = event.payload if hasattr(event, 'payload') else {}

        run_id = self._find_active_run(OptimizationType.NAS, payload)
        if run_id and run_id in self._active_runs:
            run = self._active_runs.pop(run_id)
            run.state = OptimizationState.COMPLETED
            run.updated_at = time.time()
            run.best_score = payload.get("best_score")
            run.best_params = payload.get("best_architecture", {})
            self._add_to_history(run)

            logger.info(f"[OptimizationOrchestrator] NAS completed: {run_id}")
            self._emit_optimization_event("completed", run)

    def _on_nas_best(self, event) -> None:
        """Handle NAS_BEST_ARCHITECTURE event."""
        self._track_event("nas_best_architecture")
        payload = event.payload if hasattr(event, 'payload') else {}

        run_id = self._find_active_run(OptimizationType.NAS, payload)
        if run_id and run_id in self._active_runs:
            run = self._active_runs[run_id]
            run.best_params = payload.get("architecture", payload)
            run.best_score = payload.get("score", run.best_score)
            run.updated_at = time.time()

            logger.info(
                f"[OptimizationOrchestrator] NAS best architecture found: {run_id} "
                f"(score={run.best_score})"
            )

    # =========================================================================
    # PBT Event Handlers
    # =========================================================================

    def _on_pbt_started(self, event) -> None:
        """Handle PBT_STARTED event."""
        self._track_event("pbt_started")
        payload = event.payload if hasattr(event, 'payload') else {}

        run_id = self._get_run_id(payload, OptimizationType.PBT)
        config = payload.get("config", payload.get("board_type", "unknown"))

        run = OptimizationRun(
            run_id=run_id,
            opt_type=OptimizationType.PBT,
            state=OptimizationState.RUNNING,
            config=config,
            started_at=time.time(),
            updated_at=time.time(),
            total_generations=payload.get("generations", 0),
            metadata=payload,
        )

        self._active_runs[run_id] = run
        logger.info(f"[OptimizationOrchestrator] PBT started: {run_id}")
        self._emit_optimization_event("started", run)

    def _on_pbt_generation(self, event) -> None:
        """Handle PBT_GENERATION_COMPLETE event."""
        self._track_event("pbt_generation_complete")
        payload = event.payload if hasattr(event, 'payload') else {}

        run_id = self._find_active_run(OptimizationType.PBT, payload)
        if run_id and run_id in self._active_runs:
            run = self._active_runs[run_id]
            run.current_generation = payload.get("generation", run.current_generation + 1)
            run.updated_at = time.time()

            if "best_score" in payload:
                run.best_score = payload["best_score"]

            logger.debug(
                f"[OptimizationOrchestrator] PBT generation {run.current_generation}: {run_id}"
            )

    def _on_pbt_completed(self, event) -> None:
        """Handle PBT_COMPLETED event."""
        self._track_event("pbt_completed")
        payload = event.payload if hasattr(event, 'payload') else {}

        run_id = self._find_active_run(OptimizationType.PBT, payload)
        if run_id and run_id in self._active_runs:
            run = self._active_runs.pop(run_id)
            run.state = OptimizationState.COMPLETED
            run.updated_at = time.time()
            run.best_score = payload.get("best_score")
            run.best_params = payload.get("best_params", {})
            self._add_to_history(run)

            logger.info(f"[OptimizationOrchestrator] PBT completed: {run_id}")
            self._emit_optimization_event("completed", run)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _find_active_run(
        self,
        opt_type: OptimizationType,
        payload: dict[str, Any],
    ) -> str | None:
        """Find active run matching the event payload."""
        # Check for explicit run_id
        if "run_id" in payload:
            return payload["run_id"]

        # Find by config match
        config = payload.get("config", payload.get("board_type", ""))
        for run_id, run in self._active_runs.items():
            if run.opt_type == opt_type and (not config or run.config == config):
                return run_id

        # Return most recent run of this type
        for run_id, run in reversed(list(self._active_runs.items())):
            if run.opt_type == opt_type:
                return run_id

        return None

    def _add_to_history(self, run: OptimizationRun) -> None:
        """Add completed run to history."""
        self._completed_runs.append(run)
        if len(self._completed_runs) > self._max_history:
            self._completed_runs = self._completed_runs[-self._max_history:]

    def _emit_optimization_event(self, action: str, run: OptimizationRun) -> None:
        """Emit optimization lifecycle event."""
        try:
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            # Use a generic event type for optimization lifecycle
            event = DataEvent(
                event_type=DataEventType.QUALITY_SCORE_UPDATED,  # Reuse for now
                payload={
                    "event_subtype": f"optimization_{action}",
                    "optimization_type": run.opt_type.value,
                    "run_id": run.run_id,
                    "config": run.config,
                    "state": run.state.value,
                    "best_score": run.best_score,
                    "duration_seconds": time.time() - run.started_at,
                },
                source="optimization_orchestrator",
            )

            bus = get_event_bus()
            bus.publish_sync(event)

        except Exception as e:
            logger.debug(f"Failed to emit optimization event: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_active_runs(self) -> list[dict[str, Any]]:
        """Get all currently active optimization runs."""
        return [run.to_dict() for run in self._active_runs.values()]

    def get_completed_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recently completed runs."""
        return [run.to_dict() for run in self._completed_runs[-limit:]]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a specific run by ID."""
        if run_id in self._active_runs:
            return self._active_runs[run_id].to_dict()
        for run in self._completed_runs:
            if run.run_id == run_id:
                return run.to_dict()
        return None

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status summary."""
        active_by_type = {}
        for run in self._active_runs.values():
            key = run.opt_type.value
            active_by_type[key] = active_by_type.get(key, 0) + 1

        return {
            "subscribed": self._subscribed,
            "active_runs": len(self._active_runs),
            "active_by_type": active_by_type,
            "completed_runs": len(self._completed_runs),
            "event_counts": self._event_counts,
        }


# Singleton instance
_optimization_orchestrator: OptimizationOrchestrator | None = None


def get_optimization_orchestrator() -> OptimizationOrchestrator:
    """Get the global optimization orchestrator singleton."""
    global _optimization_orchestrator
    if _optimization_orchestrator is None:
        _optimization_orchestrator = OptimizationOrchestrator()
    return _optimization_orchestrator


def wire_optimization_events() -> OptimizationOrchestrator:
    """Wire all optimization events to the orchestrator.

    Returns:
        OptimizationOrchestrator instance
    """
    orchestrator = get_optimization_orchestrator()
    orchestrator.subscribe_to_events()

    logger.info(
        "[wire_optimization_events] Optimization events wired to orchestrator "
        "(CMAES, NAS, PBT)"
    )

    return orchestrator


def reset_optimization_orchestrator() -> None:
    """Reset the optimization orchestrator (for testing)."""
    global _optimization_orchestrator
    if _optimization_orchestrator:
        _optimization_orchestrator.unsubscribe()
    _optimization_orchestrator = None
