"""Lifecycle Integration for Training Components.

Provides Initializable/Service wrappers for training components to enable:
- Coordinated startup/shutdown via LifecycleManager
- Health probing integration
- Event-driven lifecycle notifications

Usage:
    from app.training.lifecycle_integration import (
        BackgroundEvalService,
        BackgroundSelfplayService,
        TrainingLifecycleManager,
    )

    # Create lifecycle manager for training
    manager = TrainingLifecycleManager()

    # Register services
    manager.register_eval_service(model_getter, board_type)
    manager.register_selfplay_service(config)

    # Start all services in order
    await manager.start_all()

    # Health check
    health = await manager.check_health()

    # Graceful shutdown
    await manager.stop_all()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.health import HealthStatus
from app.core.lifecycle import LifecycleManager, Service

logger = logging.getLogger(__name__)

__all__ = [
    "BackgroundEvalService",
    "BackgroundSelfplayService",
    "DataCoordinatorService",
    "TrainingLifecycleManager",
]


# =============================================================================
# Background Eval Service
# =============================================================================

class BackgroundEvalService(Service):
    """Service wrapper for BackgroundEvaluator.

    Provides lifecycle management and health checking for the
    background evaluation component.

    Example:
        service = BackgroundEvalService(
            model_getter=lambda: trainer.get_model_info(),
            board_type=BoardType.SQUARE8,
            use_real_games=True,
        )

        await service.on_start()
        status = await service.check_health()
        await service.on_stop()
    """

    def __init__(
        self,
        model_getter: Callable[[], Any],
        board_type: Any | None = None,
        use_real_games: bool = False,
        eval_interval: int = 1000,
        games_per_eval: int = 20,
        use_event_driven: bool = True,
    ):
        super().__init__()
        self._model_getter = model_getter
        self._board_type = board_type
        self._use_real_games = use_real_games
        self._eval_interval = eval_interval
        self._games_per_eval = games_per_eval
        self._use_event_driven = use_event_driven
        self._evaluator: Any | None = None

    @property
    def name(self) -> str:
        return "background_eval"

    @property
    def dependencies(self) -> list[str]:
        # Depends on data coordinator if using real games
        if self._use_real_games:
            return ["data_coordinator"]
        return []

    async def on_start(self) -> None:
        """Start the background evaluator."""
        from app.training.background_eval import (
            BackgroundEvalConfig,
            BackgroundEvaluator,
        )

        config = BackgroundEvalConfig(
            eval_interval_steps=self._eval_interval,
            games_per_eval=self._games_per_eval,
        )

        self._evaluator = BackgroundEvaluator(
            model_getter=self._model_getter,
            config=config,
            board_type=self._board_type,
            use_real_games=self._use_real_games,
        )

        if self._use_event_driven:
            self._evaluator.start_event_driven()
        else:
            self._evaluator.start()

        logger.info(f"[{self.name}] Started background evaluator")

    async def on_stop(self) -> None:
        """Stop the background evaluator."""
        if self._evaluator:
            self._evaluator.stop()
            self._evaluator = None
        logger.info(f"[{self.name}] Stopped background evaluator")

    async def check_health(self) -> HealthStatus:
        """Check evaluator health."""
        if not self._evaluator:
            return HealthStatus.unhealthy("Evaluator not initialized")

        if not self._evaluator._running:
            return HealthStatus.unhealthy("Evaluator thread not running")

        # Check for evaluation progress
        eval_count = len(self._evaluator.eval_results)
        current_elo = self._evaluator.get_current_elo()
        best_elo = self._evaluator.best_elo

        # Check baseline gating
        passes_gating, failed_baselines, consecutive_failures = \
            self._evaluator.get_baseline_gating_status()

        if consecutive_failures >= 3:
            return HealthStatus.degraded(
                f"Consecutive baseline failures: {consecutive_failures}",
                failed_baselines=failed_baselines,
                current_elo=current_elo,
            )

        if self._evaluator.should_early_stop():
            return HealthStatus.degraded(
                "Elo drop detected - early stop recommended",
                current_elo=current_elo,
                best_elo=best_elo,
            )

        return HealthStatus.healthy(
            f"Running, Elo: {current_elo:.0f}",
            eval_count=eval_count,
            current_elo=current_elo,
            best_elo=best_elo,
            passes_gating=passes_gating,
        )

    def get_evaluator(self) -> Any | None:
        """Get the underlying evaluator instance."""
        return self._evaluator


# =============================================================================
# Background Selfplay Service
# =============================================================================

class BackgroundSelfplayService(Service):
    """Service wrapper for BackgroundSelfplayManager.

    Provides lifecycle management and health checking for the
    background selfplay component.

    Example:
        service = BackgroundSelfplayService(
            config={"board": "square8", "players": 2, "games_per_iter": 100},
        )

        await service.on_start()
        await service.start_iteration(iteration=1)
        status = await service.check_health()
        await service.on_stop()
    """

    def __init__(
        self,
        config: dict[str, Any],
        ai_service_root: Path | None = None,
    ):
        super().__init__()
        self._config = config
        self._ai_service_root = ai_service_root
        self._manager: Any | None = None

    @property
    def name(self) -> str:
        return "background_selfplay"

    @property
    def dependencies(self) -> list[str]:
        return []  # No dependencies for selfplay

    async def on_start(self) -> None:
        """Initialize the selfplay manager."""
        from app.training.background_selfplay import BackgroundSelfplayManager

        self._manager = BackgroundSelfplayManager(
            ai_service_root=self._ai_service_root,
        )
        logger.info(f"[{self.name}] Initialized selfplay manager")

    async def on_stop(self) -> None:
        """Stop any running selfplay and cleanup."""
        if self._manager:
            self._manager.cancel_current()
            self._manager = None
        logger.info(f"[{self.name}] Stopped selfplay manager")

    async def check_health(self) -> HealthStatus:
        """Check selfplay manager health."""
        if not self._manager:
            return HealthStatus.unhealthy("Manager not initialized")

        stats = self._manager.get_statistics()
        current_task = self._manager.get_current_task()

        if current_task and current_task.is_running():
            elapsed = current_task.elapsed_time()
            return HealthStatus.healthy(
                f"Running iteration {current_task.iteration}",
                iteration=current_task.iteration,
                elapsed_seconds=elapsed,
                games_requested=current_task.games_requested,
            )

        return HealthStatus.healthy(
            "Idle",
            total_runs=stats["total_runs"],
            completed=stats["completed"],
            failed=stats["failed"],
        )

    async def start_iteration(self, iteration: int) -> bool:
        """Start selfplay for an iteration.

        Args:
            iteration: Iteration number

        Returns:
            True if started successfully
        """
        if not self._manager:
            logger.error(f"[{self.name}] Manager not initialized")
            return False

        task = self._manager.start_background_selfplay(
            config=self._config,
            iteration=iteration,
        )
        return task is not None

    async def wait_for_current(
        self,
        timeout: float | None = None,
    ) -> tuple[bool, Path | None, int]:
        """Wait for current selfplay to complete."""
        if not self._manager:
            return True, None, 0

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._manager.wait_for_current(timeout),
        )

    def get_manager(self) -> Any | None:
        """Get the underlying manager instance."""
        return self._manager


# =============================================================================
# Data Coordinator Service
# =============================================================================

class DataCoordinatorService(Service):
    """Service wrapper for data coordination components.

    Provides lifecycle management for data loading, syncing, and buffering.
    """

    def __init__(
        self,
        config_key: str,
        db_path: Path | None = None,
        buffer_size: int = 10000,
    ):
        super().__init__()
        self._config_key = config_key
        self._db_path = db_path
        self._buffer_size = buffer_size
        self._coordinator: Any | None = None

    @property
    def name(self) -> str:
        return "data_coordinator"

    @property
    def dependencies(self) -> list[str]:
        return []  # Data coordinator is typically first

    async def on_start(self) -> None:
        """Initialize data coordinator."""
        # Try to import and initialize data coordinator
        try:
            from app.training.data_coordinator import DataCoordinator

            self._coordinator = DataCoordinator(
                config_key=self._config_key,
                db_path=self._db_path,
            )
            logger.info(f"[{self.name}] Initialized data coordinator for {self._config_key}")
        except ImportError:
            logger.warning(f"[{self.name}] DataCoordinator not available, using stub")
            self._coordinator = None

    async def on_stop(self) -> None:
        """Cleanup data coordinator."""
        if self._coordinator and hasattr(self._coordinator, 'close'):
            self._coordinator.close()
        self._coordinator = None
        logger.info(f"[{self.name}] Stopped data coordinator")

    async def check_health(self) -> HealthStatus:
        """Check data coordinator health."""
        if self._coordinator is None:
            return HealthStatus.degraded("Coordinator not available (stub mode)")

        # Check if coordinator has health method
        if hasattr(self._coordinator, 'get_status'):
            status = self._coordinator.get_status()
            return HealthStatus.healthy(
                "Data coordinator running",
                **status,
            )

        return HealthStatus.healthy("Data coordinator initialized")

    def get_coordinator(self) -> Any | None:
        """Get the underlying coordinator instance."""
        return self._coordinator


# =============================================================================
# Training Lifecycle Manager
# =============================================================================

class TrainingLifecycleManager:
    """Lifecycle manager specialized for training components.

    Coordinates startup/shutdown and health checking for all
    training-related services.

    Example:
        manager = TrainingLifecycleManager()

        # Register services
        manager.register_eval_service(model_getter, board_type)
        manager.register_selfplay_service(config)
        manager.register_data_service(config_key)

        # Start in dependency order
        await manager.start_all()

        # Health endpoint
        health = await manager.get_health_summary()

        # Graceful shutdown
        await manager.stop_all()
    """

    def __init__(self):
        self._lifecycle = LifecycleManager(
            health_timeout=10.0,
            shutdown_timeout=30.0,
        )
        self._services: dict[str, Service] = {}

    def register_eval_service(
        self,
        model_getter: Callable[[], Any],
        board_type: Any | None = None,
        use_real_games: bool = False,
        **kwargs: Any,
    ) -> BackgroundEvalService:
        """Register background evaluation service.

        Args:
            model_getter: Function returning model info
            board_type: Board type for real games
            use_real_games: Whether to play real games
            **kwargs: Additional eval config

        Returns:
            The registered service
        """
        service = BackgroundEvalService(
            model_getter=model_getter,
            board_type=board_type,
            use_real_games=use_real_games,
            **kwargs,
        )
        self._lifecycle.register(service)
        self._services["background_eval"] = service
        return service

    def register_selfplay_service(
        self,
        config: dict[str, Any],
        ai_service_root: Path | None = None,
    ) -> BackgroundSelfplayService:
        """Register background selfplay service.

        Args:
            config: Selfplay configuration
            ai_service_root: AI service root path

        Returns:
            The registered service
        """
        service = BackgroundSelfplayService(
            config=config,
            ai_service_root=ai_service_root,
        )
        self._lifecycle.register(service)
        self._services["background_selfplay"] = service
        return service

    def register_data_service(
        self,
        config_key: str,
        db_path: Path | None = None,
        **kwargs: Any,
    ) -> DataCoordinatorService:
        """Register data coordinator service.

        Args:
            config_key: Configuration key
            db_path: Database path
            **kwargs: Additional config

        Returns:
            The registered service
        """
        service = DataCoordinatorService(
            config_key=config_key,
            db_path=db_path,
            **kwargs,
        )
        self._lifecycle.register(service)
        self._services["data_coordinator"] = service
        return service

    def register_service(self, service: Service) -> None:
        """Register a custom service."""
        self._lifecycle.register(service)
        self._services[service.name] = service

    async def start_all(self) -> list[str]:
        """Start all registered services in dependency order.

        Returns:
            List of started service names
        """
        return await self._lifecycle.start_all()

    async def stop_all(self) -> None:
        """Stop all services in reverse order."""
        await self._lifecycle.stop_all()

    async def check_health(self) -> dict[str, HealthStatus]:
        """Check health of all services.

        Returns:
            Dict mapping service name to health status
        """
        result = await self._lifecycle.check_health()
        return result.components

    async def get_health_summary(self) -> dict[str, Any]:
        """Get health summary for API response.

        Returns:
            Dict suitable for JSON response
        """
        result = await self._lifecycle.check_health()
        return result.to_dict()

    def get_service(self, name: str) -> Service | None:
        """Get a registered service by name."""
        return self._services.get(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all services."""
        return self._lifecycle.get_status()

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._lifecycle.is_shutting_down


# =============================================================================
# Global Instance
# =============================================================================

_training_lifecycle: TrainingLifecycleManager | None = None


def get_training_lifecycle() -> TrainingLifecycleManager:
    """Get the global training lifecycle manager."""
    global _training_lifecycle
    if _training_lifecycle is None:
        _training_lifecycle = TrainingLifecycleManager()
    return _training_lifecycle


def reset_training_lifecycle() -> None:
    """Reset the global training lifecycle (for testing)."""
    global _training_lifecycle
    _training_lifecycle = None
