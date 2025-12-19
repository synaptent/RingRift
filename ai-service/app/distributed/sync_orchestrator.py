"""Unified Sync Orchestrator - Single entry point for all sync operations (December 2025).

This module provides a unified facade for coordinating all sync-related operations:
- Data sync (games, training data)
- Model sync (P2P model distribution)
- Elo sync (rating updates)
- Registry sync (model registry synchronization)

Instead of managing 5+ separate sync components, use SyncOrchestrator for:
- Unified initialization and shutdown
- Coordinated sync scheduling
- Cross-component health monitoring
- Event-driven sync triggers

Components wrapped:
- SyncCoordinator (app/distributed/sync_coordinator.py): Data sync execution
- SyncScheduler (app/coordination/sync_coordinator.py): Sync scheduling
- UnifiedDataSync (app/distributed/unified_data_sync.py): Unified data sync
- EloSyncManager (app/tournament/elo_sync_manager.py): Elo rating sync
- RegistrySyncManager (app/training/registry_sync_manager.py): Registry sync

Usage:
    from app.distributed.sync_orchestrator import (
        SyncOrchestrator,
        get_sync_orchestrator,
    )

    # Get orchestrator
    orchestrator = get_sync_orchestrator()

    # Initialize all sync components
    await orchestrator.initialize()

    # Run a full sync cycle
    result = await orchestrator.sync_all()

    # Get sync status
    status = orchestrator.get_status()

    # Cleanup
    await orchestrator.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SyncOrchestratorConfig:
    """Configuration for SyncOrchestrator."""

    # Data sync
    enable_data_sync: bool = True
    data_sync_interval_seconds: float = 300.0
    high_quality_priority: bool = True

    # Model sync
    enable_model_sync: bool = True
    model_sync_interval_seconds: float = 600.0

    # Elo sync
    enable_elo_sync: bool = True
    elo_sync_interval_seconds: float = 60.0

    # Registry sync
    enable_registry_sync: bool = True
    registry_sync_interval_seconds: float = 120.0

    # Quality-driven sync
    min_quality_for_priority_sync: float = 0.7
    max_games_per_sync: int = 500


@dataclass
class SyncOrchestratorState:
    """State tracking for sync orchestrator."""

    initialized: bool = False
    last_data_sync: float = 0.0
    last_model_sync: float = 0.0
    last_elo_sync: float = 0.0
    last_registry_sync: float = 0.0
    total_syncs: int = 0
    sync_errors: int = 0
    components_loaded: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class SyncResult:
    """Result of a sync operation."""

    component: str
    success: bool
    items_synced: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FullSyncResult:
    """Result of a full sync cycle."""

    success: bool = True
    total_items_synced: int = 0
    duration_seconds: float = 0.0
    component_results: List[SyncResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class SyncOrchestrator:
    """Unified orchestrator for all sync operations.

    Provides a single entry point for coordinating data sync, model sync,
    Elo sync, and registry sync across the distributed system.
    """

    def __init__(self, config: Optional[SyncOrchestratorConfig] = None):
        """Initialize sync orchestrator.

        Args:
            config: Configuration (default: SyncOrchestratorConfig())
        """
        self.config = config or SyncOrchestratorConfig()
        self.state = SyncOrchestratorState()

        # Component instances (lazy loaded)
        self._data_sync_coordinator = None
        self._sync_scheduler = None
        self._elo_sync_manager = None
        self._registry_sync_manager = None
        self._quality_sync_watcher = None

    async def initialize(self) -> bool:
        """Initialize all sync components.

        Returns:
            True if all components initialized successfully
        """
        if self.state.initialized:
            return True

        logger.info("[SyncOrchestrator] Initializing sync components...")
        start_time = time.time()
        components_loaded = []

        # Load data sync coordinator (required for data and model sync)
        if self.config.enable_data_sync or self.config.enable_model_sync:
            try:
                from app.distributed.sync_coordinator import SyncCoordinator

                self._data_sync_coordinator = SyncCoordinator.get_instance()
                components_loaded.append("data_sync_coordinator")
            except ImportError as e:
                logger.warning(f"[SyncOrchestrator] Could not load data_sync_coordinator: {e}")
                self.state.errors.append(f"data_sync_coordinator: {e}")

        # Load sync scheduler
        try:
            from app.coordination.sync_coordinator import get_sync_scheduler

            self._sync_scheduler = get_sync_scheduler()
            components_loaded.append("sync_scheduler")
        except ImportError as e:
            logger.warning(f"[SyncOrchestrator] Could not load sync_scheduler: {e}")
            self.state.errors.append(f"sync_scheduler: {e}")

        # Load Elo sync manager
        if self.config.enable_elo_sync:
            try:
                from app.tournament.elo_sync_manager import get_elo_sync_manager

                self._elo_sync_manager = get_elo_sync_manager()
                components_loaded.append("elo_sync_manager")
            except ImportError as e:
                logger.warning(f"[SyncOrchestrator] Could not load elo_sync_manager: {e}")
                self.state.errors.append(f"elo_sync_manager: {e}")

        # Load registry sync manager
        if self.config.enable_registry_sync:
            try:
                from app.training.registry_sync_manager import get_registry_sync_manager

                self._registry_sync_manager = get_registry_sync_manager()
                components_loaded.append("registry_sync_manager")
            except ImportError as e:
                logger.warning(f"[SyncOrchestrator] Could not load registry_sync_manager: {e}")
                self.state.errors.append(f"registry_sync_manager: {e}")

        # Wire quality events to sync priority
        if self.config.enable_data_sync and self.config.high_quality_priority:
            try:
                from app.distributed.sync_coordinator import wire_all_quality_events_to_sync

                self._quality_sync_watcher = wire_all_quality_events_to_sync(
                    min_quality_score=self.config.min_quality_for_priority_sync,
                    max_games_per_sync=self.config.max_games_per_sync,
                )
                components_loaded.append("quality_sync_watcher")
            except ImportError as e:
                logger.debug(f"[SyncOrchestrator] Quality sync watcher not available: {e}")

        self.state.components_loaded = components_loaded
        self.state.initialized = True

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[SyncOrchestrator] Initialized {len(components_loaded)} components "
            f"in {duration_ms:.1f}ms: {', '.join(components_loaded)}"
        )

        return len(self.state.errors) == 0

    async def shutdown(self) -> None:
        """Shutdown all sync components gracefully."""
        if not self.state.initialized:
            return

        logger.info("[SyncOrchestrator] Shutting down sync components...")

        # Unsubscribe quality watcher
        if self._quality_sync_watcher:
            try:
                self._quality_sync_watcher.unsubscribe()
            except Exception:
                pass

        self.state.initialized = False
        logger.info("[SyncOrchestrator] Shutdown complete")

    async def sync_data(self, categories: Optional[List[str]] = None) -> SyncResult:
        """Sync training data.

        Args:
            categories: Specific categories to sync (all if None)

        Returns:
            SyncResult with sync details
        """
        if not self._data_sync_coordinator:
            return SyncResult(
                component="data_sync",
                success=False,
                error="Data sync coordinator not available",
            )

        start_time = time.time()

        try:
            stats = await self._data_sync_coordinator.sync_all(
                categories=categories,
                sync_high_quality_first=self.config.high_quality_priority,
            )

            self.state.last_data_sync = time.time()

            return SyncResult(
                component="data_sync",
                success=True,
                items_synced=stats.files_synced,
                duration_seconds=time.time() - start_time,
                details={
                    "bytes_transferred": stats.bytes_transferred,
                    "high_quality_games": getattr(stats, "total_high_quality_games", 0),
                    "categories": list(stats.categories.keys()) if stats.categories else [],
                },
            )

        except Exception as e:
            logger.error(f"[SyncOrchestrator] Data sync failed: {e}")
            self.state.sync_errors += 1
            return SyncResult(
                component="data_sync",
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def sync_models(
        self,
        model_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> SyncResult:
        """Sync model checkpoints.

        Args:
            model_ids: Specific model IDs to sync (all if None)
            sources: Specific sources to sync from (auto-discovers if None)

        Returns:
            SyncResult with sync details
        """
        if not self._data_sync_coordinator:
            return SyncResult(
                component="model_sync",
                success=False,
                error="Data sync coordinator not available",
            )

        start_time = time.time()

        try:
            stats = await self._data_sync_coordinator.sync_models(
                model_ids=model_ids,
                sources=sources,
            )

            self.state.last_model_sync = time.time()

            return SyncResult(
                component="model_sync",
                success=True,
                items_synced=stats.files_synced,
                duration_seconds=time.time() - start_time,
                details={
                    "bytes_transferred": stats.bytes_transferred,
                    "transport_used": stats.transport_used,
                    "sources_tried": stats.sources_tried,
                    "errors": stats.errors,
                },
            )

        except Exception as e:
            logger.error(f"[SyncOrchestrator] Model sync failed: {e}")
            self.state.sync_errors += 1
            return SyncResult(
                component="model_sync",
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def sync_elo(self) -> SyncResult:
        """Sync Elo ratings.

        Returns:
            SyncResult with sync details
        """
        if not self._elo_sync_manager:
            return SyncResult(
                component="elo_sync",
                success=False,
                error="Elo sync manager not available",
            )

        start_time = time.time()

        try:
            # Elo sync manager may have different interfaces
            if hasattr(self._elo_sync_manager, "sync"):
                result = await self._elo_sync_manager.sync()
                items = getattr(result, "ratings_synced", 0)
            elif hasattr(self._elo_sync_manager, "sync_ratings"):
                items = await self._elo_sync_manager.sync_ratings()
            else:
                items = 0

            self.state.last_elo_sync = time.time()

            return SyncResult(
                component="elo_sync",
                success=True,
                items_synced=items,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"[SyncOrchestrator] Elo sync failed: {e}")
            self.state.sync_errors += 1
            return SyncResult(
                component="elo_sync",
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def sync_registry(self) -> SyncResult:
        """Sync model registry.

        Returns:
            SyncResult with sync details
        """
        if not self._registry_sync_manager:
            return SyncResult(
                component="registry_sync",
                success=False,
                error="Registry sync manager not available",
            )

        start_time = time.time()

        try:
            # Registry sync manager interface
            if hasattr(self._registry_sync_manager, "sync"):
                result = await self._registry_sync_manager.sync()
                items = getattr(result, "models_synced", 0)
            elif hasattr(self._registry_sync_manager, "sync_registry"):
                items = await self._registry_sync_manager.sync_registry()
            else:
                items = 0

            self.state.last_registry_sync = time.time()

            return SyncResult(
                component="registry_sync",
                success=True,
                items_synced=items,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"[SyncOrchestrator] Registry sync failed: {e}")
            self.state.sync_errors += 1
            return SyncResult(
                component="registry_sync",
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def sync_all(self) -> FullSyncResult:
        """Run a full sync cycle across all components.

        Returns:
            FullSyncResult with aggregated results
        """
        if not self.state.initialized:
            await self.initialize()

        start_time = time.time()
        result = FullSyncResult()

        # Sync data
        if self.config.enable_data_sync:
            data_result = await self.sync_data()
            result.component_results.append(data_result)
            if data_result.success:
                result.total_items_synced += data_result.items_synced
            else:
                result.success = False
                if data_result.error:
                    result.errors.append(f"data_sync: {data_result.error}")

        # Sync models
        if self.config.enable_model_sync:
            model_result = await self.sync_models()
            result.component_results.append(model_result)
            if model_result.success:
                result.total_items_synced += model_result.items_synced
            else:
                result.success = False
                if model_result.error:
                    result.errors.append(f"model_sync: {model_result.error}")

        # Sync Elo
        if self.config.enable_elo_sync:
            elo_result = await self.sync_elo()
            result.component_results.append(elo_result)
            if elo_result.success:
                result.total_items_synced += elo_result.items_synced
            else:
                result.success = False
                if elo_result.error:
                    result.errors.append(f"elo_sync: {elo_result.error}")

        # Sync registry
        if self.config.enable_registry_sync:
            registry_result = await self.sync_registry()
            result.component_results.append(registry_result)
            if registry_result.success:
                result.total_items_synced += registry_result.items_synced
            else:
                result.success = False
                if registry_result.error:
                    result.errors.append(f"registry_sync: {registry_result.error}")

        result.duration_seconds = time.time() - start_time
        self.state.total_syncs += 1

        logger.info(
            f"[SyncOrchestrator] Full sync complete: {result.total_items_synced} items, "
            f"{result.duration_seconds:.1f}s, success={result.success}"
        )

        # Emit SYNC_COMPLETE event (December 2025)
        await self._emit_sync_complete_event(result)

        return result

    async def _emit_sync_complete_event(self, result: FullSyncResult) -> None:
        """Emit SYNC_COMPLETE StageEvent after sync operations.

        Args:
            result: The full sync result
        """
        try:
            from app.coordination.stage_events import (
                StageEvent,
                StageCompletionResult,
                get_event_bus,
            )
            from datetime import datetime

            event_result = StageCompletionResult(
                event=StageEvent.SYNC_COMPLETE,
                success=result.success,
                iteration=self.state.total_syncs,
                timestamp=datetime.now().isoformat(),
                games_generated=result.total_items_synced,
                metadata={
                    "duration_seconds": result.duration_seconds,
                    "components": [r.component for r in result.component_results],
                    "errors": result.errors,
                },
            )

            bus = get_event_bus()
            await bus.emit(event_result)
            logger.debug("[SyncOrchestrator] Emitted SYNC_COMPLETE event")

        except ImportError:
            logger.debug("[SyncOrchestrator] StageEventBus not available")
        except Exception as e:
            logger.debug(f"[SyncOrchestrator] Event emission failed: {e}")

    async def run_scheduler(self, run_once: bool = False) -> None:
        """Run the sync scheduler loop.

        Continuously checks if syncs are due and runs them.
        Integrates with event bus for external triggers.

        Args:
            run_once: If True, run one sync cycle and exit
        """
        if not self.state.initialized:
            await self.initialize()

        logger.info("[SyncOrchestrator] Starting sync scheduler...")

        # Subscribe to events that should trigger syncs
        await self._subscribe_to_sync_triggers()

        while True:
            try:
                # Check each component for due syncs
                if self.needs_sync("data") and self.config.enable_data_sync:
                    await self.sync_data()

                if self.needs_sync("model") and self.config.enable_model_sync:
                    await self.sync_models()

                if self.needs_sync("elo") and self.config.enable_elo_sync:
                    await self.sync_elo()

                if self.needs_sync("registry") and self.config.enable_registry_sync:
                    await self.sync_registry()

                if run_once:
                    break

                # Sleep before next check (minimum of all intervals)
                min_interval = min(
                    self.config.data_sync_interval_seconds,
                    self.config.elo_sync_interval_seconds,
                    self.config.registry_sync_interval_seconds,
                    self.config.model_sync_interval_seconds,
                )
                await asyncio.sleep(min(min_interval / 4, 30.0))

            except asyncio.CancelledError:
                logger.info("[SyncOrchestrator] Scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"[SyncOrchestrator] Scheduler error: {e}")
                self.state.sync_errors += 1
                await asyncio.sleep(10.0)

    async def _subscribe_to_sync_triggers(self) -> None:
        """Subscribe to events that should trigger syncs."""
        try:
            from app.coordination.stage_events import (
                StageEvent,
                get_event_bus,
            )

            bus = get_event_bus()

            # Sync after selfplay completes (new games to sync)
            async def on_selfplay_complete(result):
                if result.success and result.games_generated > 0:
                    logger.info(
                        f"[SyncOrchestrator] Selfplay complete with {result.games_generated} games, "
                        "triggering data sync"
                    )
                    await self.sync_data()

            bus.subscribe(StageEvent.SELFPLAY_COMPLETE, on_selfplay_complete)

            # Sync after training completes (new model to sync)
            async def on_training_complete(result):
                if result.success and result.model_path:
                    logger.info(
                        f"[SyncOrchestrator] Training complete, triggering model sync"
                    )
                    await self.sync_models()
                    await self.sync_registry()

            bus.subscribe(StageEvent.TRAINING_COMPLETE, on_training_complete)

            # Sync after promotion (update registry)
            async def on_promotion_complete(result):
                if result.success:
                    logger.info(
                        f"[SyncOrchestrator] Promotion complete, triggering registry sync"
                    )
                    await self.sync_registry()

            bus.subscribe(StageEvent.PROMOTION_COMPLETE, on_promotion_complete)

            logger.debug("[SyncOrchestrator] Subscribed to sync trigger events")

        except ImportError:
            logger.debug("[SyncOrchestrator] Event bus not available for triggers")
        except Exception as e:
            logger.warning(f"[SyncOrchestrator] Failed to subscribe to triggers: {e}")

    def needs_sync(self, component: str) -> bool:
        """Check if a component needs sync based on interval.

        Args:
            component: Component name (data, elo, registry, model)

        Returns:
            True if sync is due
        """
        now = time.time()

        if component == "data":
            return now - self.state.last_data_sync >= self.config.data_sync_interval_seconds
        elif component == "elo":
            return now - self.state.last_elo_sync >= self.config.elo_sync_interval_seconds
        elif component == "registry":
            return now - self.state.last_registry_sync >= self.config.registry_sync_interval_seconds
        elif component == "model":
            return now - self.state.last_model_sync >= self.config.model_sync_interval_seconds

        return False

    def get_status(self) -> Dict[str, Any]:
        """Get sync orchestrator status.

        Returns:
            Dict with status information
        """
        return {
            "initialized": self.state.initialized,
            "components_loaded": self.state.components_loaded,
            "total_syncs": self.state.total_syncs,
            "sync_errors": self.state.sync_errors,
            "last_syncs": {
                "data": self.state.last_data_sync,
                "elo": self.state.last_elo_sync,
                "registry": self.state.last_registry_sync,
                "model": self.state.last_model_sync,
            },
            "config": {
                "data_sync_enabled": self.config.enable_data_sync,
                "elo_sync_enabled": self.config.enable_elo_sync,
                "registry_sync_enabled": self.config.enable_registry_sync,
                "model_sync_enabled": self.config.enable_model_sync,
                "high_quality_priority": self.config.high_quality_priority,
            },
            "errors": self.state.errors,
        }


# Singleton instance
_sync_orchestrator: Optional[SyncOrchestrator] = None


def get_sync_orchestrator(
    config: Optional[SyncOrchestratorConfig] = None,
) -> SyncOrchestrator:
    """Get the global sync orchestrator singleton.

    Args:
        config: Configuration (only used on first call)

    Returns:
        SyncOrchestrator instance
    """
    global _sync_orchestrator
    if _sync_orchestrator is None:
        _sync_orchestrator = SyncOrchestrator(config)
    return _sync_orchestrator


def reset_sync_orchestrator() -> None:
    """Reset the sync orchestrator singleton (for testing)."""
    global _sync_orchestrator
    _sync_orchestrator = None


__all__ = [
    "SyncOrchestrator",
    "SyncOrchestratorConfig",
    "SyncOrchestratorState",
    "SyncResult",
    "FullSyncResult",
    "get_sync_orchestrator",
    "reset_sync_orchestrator",
]
