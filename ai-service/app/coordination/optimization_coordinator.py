"""OptimizationTriggeringCoordinator - Auto CMA-ES/NAS coordination (December 2025).

This module provides centralized coordination for hyperparameter optimization
and neural architecture search. It monitors training metrics, detects plateaus,
and automatically triggers optimization when appropriate.

Event Integration:
- Subscribes to PLATEAU_DETECTED: Trigger optimization on training plateau
- Subscribes to CMAES_TRIGGERED: Track CMA-ES start
- Subscribes to CMAES_COMPLETED: Track CMA-ES completion
- Subscribes to NAS_TRIGGERED: Track NAS start
- Subscribes to NAS_COMPLETED: Track NAS completion
- Subscribes to TRAINING_PROGRESS: Monitor for plateaus
- Subscribes to HYPERPARAMETER_UPDATED: Track parameter changes

Key Responsibilities:
1. Detect training plateaus automatically
2. Trigger CMA-ES for hyperparameter optimization
3. Trigger NAS for architecture search
4. Coordinate optimization across the cluster
5. Prevent concurrent optimizations

Usage:
    from app.coordination.optimization_coordinator import (
        OptimizationCoordinator,
        wire_optimization_events,
        get_optimization_coordinator,
    )

    # Wire optimization events
    coordinator = wire_optimization_events()

    # Check if optimization is running
    if not coordinator.is_optimization_running():
        coordinator.trigger_cmaes("loss_plateau")

    # Get optimization history
    history = coordinator.get_optimization_history()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization."""

    CMAES = "cmaes"  # CMA-ES hyperparameter optimization
    NAS = "nas"  # Neural Architecture Search
    PBT = "pbt"  # Population Based Training
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"


class OptimizationStatus(Enum):
    """Status of an optimization run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlateauType(Enum):
    """Types of detected plateaus."""

    LOSS = "loss"  # Training/validation loss not improving
    ELO = "elo"  # Elo not improving
    WIN_RATE = "win_rate"  # Win rate stagnant


@dataclass
class PlateauDetection:
    """Record of a detected plateau."""

    plateau_type: PlateauType
    metric_name: str
    current_value: float
    best_value: float
    epochs_since_improvement: int
    detected_at: float = field(default_factory=time.time)
    model_id: str = ""
    triggered_optimization: bool = False


@dataclass
class OptimizationRun:
    """Record of an optimization run."""

    run_id: str
    optimization_type: OptimizationType
    status: OptimizationStatus = OptimizationStatus.PENDING
    trigger_reason: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    # Configuration
    parameters_searched: list[str] = field(default_factory=list)
    search_space: dict[str, Any] = field(default_factory=dict)
    generations: int = 0
    population_size: int = 0

    # Results
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    iterations_completed: int = 0
    evaluations: int = 0

    # Error tracking
    error: str = ""

    @property
    def duration(self) -> float:
        """Get run duration in seconds."""
        if self.completed_at > 0:
            return self.completed_at - self.started_at
        return time.time() - self.started_at


@dataclass
class OptimizationStats:
    """Aggregate optimization statistics."""

    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    current_running: str | None = None
    total_evaluations: int = 0
    avg_run_duration: float = 0.0
    runs_by_type: dict[str, int] = field(default_factory=dict)
    plateaus_detected: int = 0
    last_optimization_time: float = 0.0


class OptimizationCoordinator:
    """Coordinates hyperparameter optimization and NAS across the cluster.

    Monitors training metrics, detects plateaus, and triggers optimization
    when appropriate while preventing concurrent conflicting runs.
    """

    def __init__(
        self,
        plateau_window: int = 10,  # Epochs to detect plateau
        plateau_threshold: float = 0.001,  # Min improvement to not be plateau
        min_epochs_between_optimization: int = 20,
        cooldown_seconds: float = 300.0,
        max_history: int = 100,
    ):
        """Initialize OptimizationCoordinator.

        Args:
            plateau_window: Number of epochs to check for plateau
            plateau_threshold: Minimum improvement to not be considered plateau
            min_epochs_between_optimization: Min epochs before re-triggering
            cooldown_seconds: Cooldown after optimization completes
            max_history: Maximum optimization runs to retain
        """
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.min_epochs_between_optimization = min_epochs_between_optimization
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history

        # Metric tracking for plateau detection
        self._metric_history: dict[str, deque] = {}  # metric_name -> values
        self._metric_best: dict[str, float] = {}
        self._epochs_since_improvement: dict[str, int] = {}

        # Optimization tracking
        self._current_optimization: OptimizationRun | None = None
        self._optimization_history: list[OptimizationRun] = []
        self._last_optimization_end: float = 0.0
        self._run_id_counter = 0

        # Plateau tracking
        self._plateaus: list[PlateauDetection] = []

        # Statistics
        self._total_evaluations = 0

        # Callbacks
        self._plateau_callbacks: list[Callable[[PlateauDetection], None]] = []
        self._optimization_callbacks: list[Callable[[OptimizationRun], None]] = []

        # Subscription state
        self._subscribed = False

    def subscribe_to_events(self) -> bool:
        """Subscribe to optimization-related events.

        Returns:
            True if successfully subscribed
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router
            from app.coordination.event_router import DataEventType

            router = get_router()

            router.subscribe(DataEventType.PLATEAU_DETECTED.value, self._on_plateau_detected)
            router.subscribe(DataEventType.CMAES_TRIGGERED.value, self._on_cmaes_triggered)
            router.subscribe(DataEventType.CMAES_COMPLETED.value, self._on_cmaes_completed)
            router.subscribe(DataEventType.NAS_TRIGGERED.value, self._on_nas_triggered)
            router.subscribe(DataEventType.NAS_COMPLETED.value, self._on_nas_completed)
            router.subscribe(DataEventType.TRAINING_PROGRESS.value, self._on_training_progress)
            router.subscribe(DataEventType.HYPERPARAMETER_UPDATED.value, self._on_hyperparameter_updated)

            self._subscribed = True
            logger.info("[OptimizationCoordinator] Subscribed to optimization events")
            return True

        except ImportError:
            logger.warning("[OptimizationCoordinator] data_events not available")
            return False
        except Exception as e:
            logger.error(f"[OptimizationCoordinator] Failed to subscribe: {e}")
            return False

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        self._run_id_counter += 1
        return f"opt_{int(time.time())}_{self._run_id_counter}"

    async def _on_plateau_detected(self, event) -> None:
        """Handle PLATEAU_DETECTED event - may auto-trigger CMAES."""
        payload = event.payload

        try:
            plateau_type = PlateauType(payload.get("plateau_type", "loss"))
        except ValueError:
            plateau_type = PlateauType.LOSS

        plateau = PlateauDetection(
            plateau_type=plateau_type,
            metric_name=payload.get("metric_name", ""),
            current_value=payload.get("current_value", 0.0),
            best_value=payload.get("best_value", 0.0),
            epochs_since_improvement=payload.get("epochs_since_improvement", 0),
            model_id=payload.get("model_id", ""),
        )

        self._plateaus.append(plateau)

        # Notify callbacks
        for callback in self._plateau_callbacks:
            try:
                callback(plateau)
            except Exception as e:
                logger.error(f"[OptimizationCoordinator] Plateau callback error: {e}")

        logger.info(
            f"[OptimizationCoordinator] Plateau detected: {plateau_type.value}, "
            f"metric={plateau.metric_name}, epochs={plateau.epochs_since_improvement}"
        )

        # Auto-trigger CMAES if enabled and conditions met (December 2025)
        if self._should_auto_trigger_cmaes(plateau):
            logger.info(
                f"[OptimizationCoordinator] Auto-triggering CMAES due to plateau: "
                f"{plateau.metric_name} stalled for {plateau.epochs_since_improvement} epochs"
            )
            self.trigger_cmaes(
                reason=f"auto_plateau_{plateau.plateau_type.value}",
                parameters=["learning_rate", "weight_decay", "batch_size"],
            )

    def _should_auto_trigger_cmaes(self, plateau: PlateauDetection) -> bool:
        """Check if CMAES should be auto-triggered for this plateau.

        Args:
            plateau: The detected plateau

        Returns:
            True if CMAES should be triggered
        """
        # Don't trigger if optimization already running
        if self._current_optimization is not None:
            return False

        # Don't trigger if we've done too many optimizations recently
        recent_cutoff = time.time() - 3600  # 1 hour
        recent_runs = [
            r for r in self._optimization_history
            if r.start_time > recent_cutoff
        ]
        if len(recent_runs) >= 3:
            return False

        # Trigger conditions:
        # - Loss plateau with 15+ epochs stalled
        # - ELO plateau with 10+ epochs stalled
        if plateau.plateau_type == PlateauType.LOSS:
            return plateau.epochs_since_improvement >= 15
        elif plateau.plateau_type == PlateauType.ELO:
            return plateau.epochs_since_improvement >= 10

        return False

    async def _on_cmaes_triggered(self, event) -> None:
        """Handle CMAES_TRIGGERED event."""
        payload = event.payload

        run = OptimizationRun(
            run_id=payload.get("run_id", self._generate_run_id()),
            optimization_type=OptimizationType.CMAES,
            status=OptimizationStatus.RUNNING,
            trigger_reason=payload.get("reason", "manual"),
            parameters_searched=payload.get("parameters", []),
            search_space=payload.get("search_space", {}),
            generations=payload.get("generations", 0),
            population_size=payload.get("population_size", 0),
        )

        self._current_optimization = run

        logger.info(
            f"[OptimizationCoordinator] CMA-ES triggered: {run.run_id}, "
            f"reason={run.trigger_reason}"
        )

    async def _on_cmaes_completed(self, event) -> None:
        """Handle CMAES_COMPLETED event."""
        payload = event.payload

        if self._current_optimization:
            run = self._current_optimization
            run.status = (
                OptimizationStatus.COMPLETED
                if payload.get("success", True)
                else OptimizationStatus.FAILED
            )
            run.completed_at = time.time()
            run.best_params = payload.get("best_params", {})
            run.best_score = payload.get("best_score", 0.0)
            run.iterations_completed = payload.get("generations_completed", 0)
            run.evaluations = payload.get("evaluations", 0)
            run.error = payload.get("error", "")

            self._total_evaluations += run.evaluations
            self._record_optimization(run)
            self._current_optimization = None
            self._last_optimization_end = time.time()

            logger.info(
                f"[OptimizationCoordinator] CMA-ES completed: {run.run_id}, "
                f"score={run.best_score:.4f}, evals={run.evaluations}"
            )

    async def _on_nas_triggered(self, event) -> None:
        """Handle NAS_TRIGGERED event."""
        payload = event.payload

        run = OptimizationRun(
            run_id=payload.get("run_id", self._generate_run_id()),
            optimization_type=OptimizationType.NAS,
            status=OptimizationStatus.RUNNING,
            trigger_reason=payload.get("reason", "manual"),
            search_space=payload.get("search_space", {}),
            generations=payload.get("generations", 0),
            population_size=payload.get("population_size", 0),
        )

        self._current_optimization = run

        logger.info(
            f"[OptimizationCoordinator] NAS triggered: {run.run_id}, "
            f"reason={run.trigger_reason}"
        )

    async def _on_nas_completed(self, event) -> None:
        """Handle NAS_COMPLETED event."""
        payload = event.payload

        if self._current_optimization:
            run = self._current_optimization
            run.status = (
                OptimizationStatus.COMPLETED
                if payload.get("success", True)
                else OptimizationStatus.FAILED
            )
            run.completed_at = time.time()
            run.best_params = payload.get("best_architecture", {})
            run.best_score = payload.get("best_score", 0.0)
            run.iterations_completed = payload.get("generations_completed", 0)
            run.evaluations = payload.get("evaluations", 0)
            run.error = payload.get("error", "")

            self._total_evaluations += run.evaluations
            self._record_optimization(run)
            self._current_optimization = None
            self._last_optimization_end = time.time()

            logger.info(
                f"[OptimizationCoordinator] NAS completed: {run.run_id}, "
                f"score={run.best_score:.4f}, evals={run.evaluations}"
            )

    async def _on_training_progress(self, event) -> None:
        """Handle TRAINING_PROGRESS to detect plateaus."""
        payload = event.payload

        # Track key metrics
        for metric in ["train_loss", "val_loss", "elo"]:
            if metric in payload:
                self._update_metric(metric, payload[metric])

    async def _on_hyperparameter_updated(self, event) -> None:
        """Handle HYPERPARAMETER_UPDATED event."""
        payload = event.payload

        logger.debug(
            f"[OptimizationCoordinator] Hyperparameter updated: "
            f"{payload.get('parameter')} = {payload.get('value')}"
        )

    def _update_metric(self, metric_name: str, value: float) -> None:
        """Update metric history and check for plateau."""
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = deque(maxlen=self.plateau_window)
            self._metric_best[metric_name] = value
            self._epochs_since_improvement[metric_name] = 0

        history = self._metric_history[metric_name]
        history.append(value)

        # Check for improvement (lower is better for loss)
        is_improvement = False
        if "loss" in metric_name:
            if value < self._metric_best[metric_name] - self.plateau_threshold:
                is_improvement = True
                self._metric_best[metric_name] = value
        else:
            if value > self._metric_best[metric_name] + self.plateau_threshold:
                is_improvement = True
                self._metric_best[metric_name] = value

        if is_improvement:
            self._epochs_since_improvement[metric_name] = 0
        else:
            self._epochs_since_improvement[metric_name] += 1

    def _record_optimization(self, run: OptimizationRun) -> None:
        """Record a completed optimization run."""
        self._optimization_history.append(run)

        # Trim history
        if len(self._optimization_history) > self.max_history:
            self._optimization_history = self._optimization_history[-self.max_history:]

        # Notify callbacks
        for callback in self._optimization_callbacks:
            try:
                callback(run)
            except Exception as e:
                logger.error(f"[OptimizationCoordinator] Optimization callback error: {e}")

    def _emit_optimization_triggered(self, run: OptimizationRun) -> None:
        """Emit CMAES_TRIGGERED or NAS_TRIGGERED event.

        January 2026: Migrated to safe_emit_event for consistent event handling.
        """
        from app.coordination.event_emission_helpers import safe_emit_event

        if run.optimization_type not in (OptimizationType.CMAES, OptimizationType.NAS):
            return

        optimization_type = run.optimization_type.value

        safe_emit_event(
            "OPTIMIZATION_TRIGGERED",
            {
                "optimization_type": optimization_type,
                "run_id": run.run_id,
                "reason": run.trigger_reason,
                "parameters_searched": len(run.parameters_searched),
                "search_space": run.search_space,
                "generations": run.generations,
                "population_size": run.population_size,
            },
            context="optimization_coordinator",
        )
        logger.debug("[OptimizationCoordinator] Emitted optimization triggered event")

    def is_optimization_running(self) -> bool:
        """Check if an optimization is currently running."""
        return self._current_optimization is not None

    def is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period after optimization."""
        if self._last_optimization_end == 0:
            return False
        return time.time() - self._last_optimization_end < self.cooldown_seconds

    def can_trigger_optimization(self) -> bool:
        """Check if optimization can be triggered."""
        if self.is_optimization_running():
            return False
        return not self.is_in_cooldown()

    def detect_plateau(self, metric_name: str) -> PlateauDetection | None:
        """Check if a metric is in plateau.

        Returns:
            PlateauDetection if plateau detected, None otherwise
        """
        if metric_name not in self._epochs_since_improvement:
            return None

        epochs = self._epochs_since_improvement[metric_name]
        if epochs < self.plateau_window:
            return None

        # Determine plateau type
        if "loss" in metric_name:
            plateau_type = PlateauType.LOSS
        elif "elo" in metric_name:
            plateau_type = PlateauType.ELO
        elif "win" in metric_name:
            plateau_type = PlateauType.WIN_RATE
        else:
            plateau_type = PlateauType.LOSS

        history = list(self._metric_history.get(metric_name, []))
        current = history[-1] if history else 0.0

        return PlateauDetection(
            plateau_type=plateau_type,
            metric_name=metric_name,
            current_value=current,
            best_value=self._metric_best.get(metric_name, 0.0),
            epochs_since_improvement=epochs,
        )

    def trigger_cmaes(
        self,
        reason: str,
        parameters: list[str] | None = None,
        search_space: dict | None = None,
        generations: int = 10,
        population_size: int = 8,
    ) -> OptimizationRun | None:
        """Trigger CMA-ES optimization.

        Returns:
            The created OptimizationRun, or None if cannot trigger
        """
        if not self.can_trigger_optimization():
            logger.warning("[OptimizationCoordinator] Cannot trigger CMAES: busy or cooldown")
            return None

        run = OptimizationRun(
            run_id=self._generate_run_id(),
            optimization_type=OptimizationType.CMAES,
            status=OptimizationStatus.RUNNING,
            trigger_reason=reason,
            parameters_searched=parameters or [],
            search_space=search_space or {},
            generations=generations,
            population_size=population_size,
        )

        self._current_optimization = run

        # Emit CMAES_TRIGGERED event (December 2025)
        self._emit_optimization_triggered(run)

        logger.info(f"[OptimizationCoordinator] Triggered CMA-ES: {run.run_id}")
        return run

    def trigger_nas(
        self,
        reason: str,
        search_space: dict | None = None,
        generations: int = 5,
        population_size: int = 4,
    ) -> OptimizationRun | None:
        """Trigger NAS optimization.

        Returns:
            The created OptimizationRun, or None if cannot trigger
        """
        if not self.can_trigger_optimization():
            logger.warning("[OptimizationCoordinator] Cannot trigger NAS: busy or cooldown")
            return None

        run = OptimizationRun(
            run_id=self._generate_run_id(),
            optimization_type=OptimizationType.NAS,
            status=OptimizationStatus.RUNNING,
            trigger_reason=reason,
            search_space=search_space or {},
            generations=generations,
            population_size=population_size,
        )

        self._current_optimization = run

        # Emit NAS_TRIGGERED event (December 2025)
        self._emit_optimization_triggered(run)

        logger.info(f"[OptimizationCoordinator] Triggered NAS: {run.run_id}")
        return run

    def cancel_optimization(self) -> bool:
        """Cancel the current optimization if running.

        Returns:
            True if cancelled, False if nothing to cancel
        """
        if self._current_optimization is None:
            return False

        self._current_optimization.status = OptimizationStatus.CANCELLED
        self._current_optimization.completed_at = time.time()
        self._record_optimization(self._current_optimization)
        self._current_optimization = None

        return True

    def on_plateau(self, callback: Callable[[PlateauDetection], None]) -> None:
        """Register callback for plateau detections."""
        self._plateau_callbacks.append(callback)

    def on_optimization_complete(self, callback: Callable[[OptimizationRun], None]) -> None:
        """Register callback for optimization completions."""
        self._optimization_callbacks.append(callback)

    def get_current_optimization(self) -> OptimizationRun | None:
        """Get the currently running optimization."""
        return self._current_optimization

    def get_optimization_history(self, limit: int = 50) -> list[OptimizationRun]:
        """Get recent optimization history."""
        return self._optimization_history[-limit:]

    def get_plateau_history(self, limit: int = 50) -> list[PlateauDetection]:
        """Get recent plateau detections."""
        return self._plateaus[-limit:]

    def get_stats(self) -> OptimizationStats:
        """Get aggregate optimization statistics."""
        successful = [r for r in self._optimization_history if r.status == OptimizationStatus.COMPLETED]
        failed = [r for r in self._optimization_history if r.status == OptimizationStatus.FAILED]

        # Count by type
        by_type: dict[str, int] = {}
        for run in self._optimization_history:
            by_type[run.optimization_type.value] = by_type.get(run.optimization_type.value, 0) + 1

        # Average duration
        durations = [r.duration for r in successful if r.duration > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return OptimizationStats(
            total_runs=len(self._optimization_history),
            successful_runs=len(successful),
            failed_runs=len(failed),
            current_running=self._current_optimization.run_id if self._current_optimization else None,
            total_evaluations=self._total_evaluations,
            avg_run_duration=avg_duration,
            runs_by_type=by_type,
            plateaus_detected=len(self._plateaus),
            last_optimization_time=self._last_optimization_end,
        )

    def get_status(self) -> dict[str, Any]:
        """Get coordinator status for monitoring."""
        stats = self.get_stats()

        return {
            "total_runs": stats.total_runs,
            "successful_runs": stats.successful_runs,
            "failed_runs": stats.failed_runs,
            "current_running": stats.current_running,
            "is_running": self.is_optimization_running(),
            "in_cooldown": self.is_in_cooldown(),
            "total_evaluations": stats.total_evaluations,
            "avg_run_duration": round(stats.avg_run_duration, 0),
            "runs_by_type": stats.runs_by_type,
            "plateaus_detected": stats.plateaus_detected,
            "subscribed": self._subscribed,
        }

    def health_check(self) -> HealthCheckResult:
        """Perform health check on optimization coordinator (December 2025).

        Returns:
            HealthCheckResult with status and metrics.

        December 2025 Session 2: Added exception handling.
        """
        try:
            stats = self.get_stats()

            # Calculate success rate
            total = stats.total_runs
            successful = stats.successful_runs
            success_rate = successful / max(total, 1)

            # Overall health criteria
            healthy = (
                self._subscribed  # Must be subscribed to events
                and success_rate >= 0.5  # At least 50% success rate
            )

            status = CoordinatorStatus.RUNNING if healthy else CoordinatorStatus.DEGRADED
            message = "" if healthy else f"Success rate {success_rate:.1%} below threshold"

            return HealthCheckResult(
                healthy=healthy,
                status=status,
                message=message,
                details={
                    "total_runs": total,
                    "successful_runs": successful,
                    "failed_runs": stats.failed_runs,
                    "success_rate": round(success_rate, 3),
                    "is_running": self.is_optimization_running(),
                    "in_cooldown": self.is_in_cooldown(),
                    "subscribed": self._subscribed,
                },
            )
        except Exception as e:
            logger.warning(f"[OptimizationCoordinator] health_check error: {e}")
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check error: {e}",
                details={"error": str(e)},
            )


# =============================================================================
# Singleton and convenience functions
# =============================================================================

_optimization_coordinator: OptimizationCoordinator | None = None


def get_optimization_coordinator() -> OptimizationCoordinator:
    """Get the global OptimizationCoordinator singleton."""
    global _optimization_coordinator
    if _optimization_coordinator is None:
        _optimization_coordinator = OptimizationCoordinator()
    return _optimization_coordinator


def wire_optimization_events() -> OptimizationCoordinator:
    """Wire optimization events to the coordinator.

    Returns:
        The wired OptimizationCoordinator instance
    """
    coordinator = get_optimization_coordinator()
    coordinator.subscribe_to_events()
    return coordinator


def is_optimization_running() -> bool:
    """Convenience function to check if optimization is running."""
    return get_optimization_coordinator().is_optimization_running()

def get_optimization_stats() -> OptimizationStats:
    """Convenience function to get optimization statistics."""
    return get_optimization_coordinator().get_stats()

def trigger_cmaes(reason: str = "manual") -> OptimizationRun | None:
    """Convenience function to trigger CMA-ES optimization."""
    return get_optimization_coordinator().trigger_cmaes(reason)

def trigger_nas(reason: str = "manual") -> OptimizationRun | None:
    """Convenience function to trigger NAS optimization."""
    return get_optimization_coordinator().trigger_nas(reason)


def trigger_hyperparameter_optimization(reason: str = "manual") -> OptimizationRun | None:
    """Convenience function to trigger CMA-ES."""
    return get_optimization_coordinator().trigger_cmaes(reason)


__all__ = [
    "OptimizationCoordinator",
    "OptimizationRun",
    "OptimizationStats",
    "OptimizationStatus",
    "OptimizationType",
    "PlateauDetection",
    "PlateauType",
    "get_optimization_coordinator",
    "get_optimization_stats",
    "is_optimization_running",
    "trigger_cmaes",
    "trigger_hyperparameter_optimization",
    "trigger_nas",
    "wire_optimization_events",
]
