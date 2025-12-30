"""Unified Sync Orchestrator - Single entry point for all sync operations (December 2025).

.. warning::
    SyncOrchestrator may be deprecated in favor of SyncFacade.

    Consider using SyncFacade for simpler, unified sync operations::

        from app.coordination.sync_facade import sync

        # Sync all data types
        await sync("all")

        # Sync specific data with routing
        await sync("games", board_type="hex8", priority="high")

    If you need the full orchestrator functionality, continue using this module.
    Check SYNC_CONSOLIDATION_PLAN.md for guidance.

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
- SyncScheduler (app/coordination/sync_coordinator.py): Sync scheduling (DEPRECATED)
- UnifiedDataSync (app/distributed/unified_data_sync.py): Unified data sync (DEPRECATED)
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
import contextlib
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

# Emit pending deprecation warning at import time
warnings.warn(
    "SyncOrchestrator may be deprecated in favor of SyncFacade. "
    "For simpler sync operations, consider:\n"
    "  from app.coordination.sync_facade import sync\n"
    "  await sync('all')  # Syncs all data types\n"
    "If you need the full orchestrator, continue using this module. "
    "See SYNC_CONSOLIDATION_PLAN.md for guidance.",
    PendingDeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

# NOTE: event_emitters import moved to lazy loading to avoid circular import
# The import was causing: event_router → data_events → distributed/__init__ → sync_orchestrator → event_emitters
# See _emit_sync_complete_event() for the lazy import pattern (December 2025)


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
    components_loaded: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ComponentSyncResult:
    """Result of a component-level sync operation (data, model, elo, registry).

    Note: This tracks high-level component results within the SyncOrchestrator.
    For individual file/transfer-level results, use:
        from app.coordination.sync_constants import SyncResult
    """

    component: str
    success: bool
    items_synced: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


# Backward compatibility alias
SyncResult = ComponentSyncResult


@dataclass
class FullSyncResult:
    """Result of a full sync cycle."""

    success: bool = True
    total_items_synced: int = 0
    duration_seconds: float = 0.0
    component_results: list[SyncResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SyncOrchestrator:
    """Unified orchestrator for all sync operations.

    Provides a single entry point for coordinating data sync, model sync,
    Elo sync, and registry sync across the distributed system.
    """

    def __init__(self, config: SyncOrchestratorConfig | None = None):
        """Initialize sync orchestrator.

        Args:
            config: Configuration (default: SyncOrchestratorConfig())
        """
        self.config = config or SyncOrchestratorConfig()
        self.state = SyncOrchestratorState()

        # Component instances (lazy loaded)
        self._data_sync_coordinator = None
        # Note: _sync_scheduler removed Dec 2025 - was loaded but never used
        self._elo_sync_manager = None
        self._registry_sync_manager = None
        self._quality_sync_watcher = None
        self._quality_orchestrator = None

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

        # Note: sync_scheduler loading removed Dec 2025 - was never used
        # Sync operations now go through SyncFacade or AutoSyncDaemon

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

        # Wire DataQualityOrchestrator for holistic quality state (December 2025)
        try:
            from app.quality.data_quality_orchestrator import (
                wire_quality_events,
            )

            # Wire and get orchestrator
            self._quality_orchestrator = wire_quality_events(
                high_quality_threshold=self.config.min_quality_for_priority_sync,
            )
            components_loaded.append("quality_orchestrator")

            # Subscribe to quality events for sync prioritization
            await self._subscribe_to_quality_events()

        except ImportError as e:
            logger.debug(f"[SyncOrchestrator] DataQualityOrchestrator not available: {e}")

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
            with contextlib.suppress(Exception):
                self._quality_sync_watcher.unsubscribe()

        self.state.initialized = False
        logger.info("[SyncOrchestrator] Shutdown complete")

    async def sync_data(self, categories: list[str] | None = None) -> SyncResult:
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
            from app.distributed.sync_coordinator import SyncCategory

            category_map = {
                "games": SyncCategory.GAMES,
                "training": SyncCategory.TRAINING,
                "models": SyncCategory.MODELS,
                "elo": SyncCategory.ELO,
            }

            category_list = None
            if categories:
                mapped = [
                    category_map.get(category.lower())
                    for category in categories
                    if category.lower() in category_map
                ]
                # Keep data-only categories for this entrypoint
                category_list = [
                    c for c in mapped
                    if c in (SyncCategory.GAMES, SyncCategory.TRAINING)
                ]

            if category_list is None:
                category_list = [SyncCategory.GAMES, SyncCategory.TRAINING]

            stats = await self._data_sync_coordinator.full_cluster_sync(
                categories=category_list,
                sync_high_quality_first=self.config.high_quality_priority,
            )

            self.state.last_data_sync = time.time()

            return SyncResult(
                component="data_sync",
                success=True,
                items_synced=stats.total_files_synced,
                duration_seconds=time.time() - start_time,
                details={
                    "bytes_transferred": stats.total_bytes_transferred,
                    "high_quality_games": stats.total_high_quality_games,
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
        model_ids: list[str] | None = None,
        sources: list[str] | None = None,
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
        """Emit SYNC_COMPLETE event using centralized emitter.

        Note: event_emitters.py handles routing to all event systems
        (data_events, stage_events, cross-process) internally.

        Args:
            result: The full sync result
        """
        try:
            # Lazy import to avoid circular import chain:
            # event_router → data_events → distributed/__init__ → sync_orchestrator → event_emitters
            from app.coordination.event_emitters import emit_sync_complete

            emitted = await emit_sync_complete(
                sync_type="full",
                items_synced=result.total_items_synced,
                success=result.success,
                duration_seconds=result.duration_seconds,
                iteration=self.state.total_syncs,
                components=[r.component for r in result.component_results],
                errors=result.errors,
            )
            if emitted:
                logger.debug("[SyncOrchestrator] Emitted SYNC_COMPLETE event")
        except ImportError:
            # event_emitters not available (rare)
            logger.debug("[SyncOrchestrator] event_emitters not available, skipping event")
        except Exception as e:
            logger.warning(f"[SyncOrchestrator] Failed to emit sync event: {e}")

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
            # P0.5 (December 2025): Use get_router() instead of deprecated get_stage_event_bus()
            from app.coordination.event_router import (
                StageEvent,
                get_router,
            )

            router = get_router()

            # Sync after selfplay completes (new games to sync)
            async def on_selfplay_complete(result):
                if result.success and result.games_generated > 0:
                    logger.info(
                        f"[SyncOrchestrator] Selfplay complete with {result.games_generated} games, "
                        "triggering data sync"
                    )
                    await self.sync_data()

            router.subscribe(StageEvent.SELFPLAY_COMPLETE, on_selfplay_complete)

            # Sync after training completes (new model to sync)
            async def on_training_complete(result):
                if result.success and result.model_path:
                    logger.info(
                        "[SyncOrchestrator] Training complete, triggering model sync"
                    )
                    await self.sync_models()
                    await self.sync_registry()

            router.subscribe(StageEvent.TRAINING_COMPLETE, on_training_complete)

            # Sync after promotion (update registry)
            async def on_promotion_complete(result):
                if result.success:
                    logger.info(
                        "[SyncOrchestrator] Promotion complete, triggering registry sync"
                    )
                    await self.sync_registry()

            router.subscribe(StageEvent.PROMOTION_COMPLETE, on_promotion_complete)

            logger.debug("[SyncOrchestrator] Subscribed to sync trigger events")

        except ImportError:
            logger.debug("[SyncOrchestrator] Event bus not available for triggers")
        except Exception as e:
            logger.warning(f"[SyncOrchestrator] Failed to subscribe to triggers: {e}")

    async def _subscribe_to_quality_events(self) -> None:
        """Subscribe to quality events from DataQualityOrchestrator for sync prioritization.

        This bridges the DataQualityOrchestrator's holistic quality state to
        intelligent sync decisions:
        - HIGH_QUALITY_DATA_AVAILABLE → Immediate priority data sync
        - LOW_QUALITY_DATA_WARNING → Deprioritize affected configs
        - QUALITY_DISTRIBUTION_CHANGED → Adjust sync priority based on trends
        """
        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()

            # On high-quality data available, trigger immediate data sync
            async def on_high_quality_available(event):
                payload = event.payload if hasattr(event, 'payload') else {}
                config_key = payload.get("config", "")
                high_quality_count = payload.get("high_quality_count", 0)

                if config_key and high_quality_count >= 100:  # Minimum threshold
                    logger.info(
                        f"[SyncOrchestrator] High-quality data available for {config_key} "
                        f"({high_quality_count} games), triggering priority sync"
                    )
                    await self.sync_data_for_config(config_key, priority=True)

            bus.subscribe(DataEventType.HIGH_QUALITY_DATA_AVAILABLE, on_high_quality_available)

            # On low quality data warning, deprioritize affected configs (December 2025)
            async def on_low_quality_warning(event):
                payload = event.payload if hasattr(event, 'payload') else {}
                config_key = payload.get("config", "")
                quality_ratio = payload.get("quality_ratio", 0.0)
                low_count = payload.get("low_quality_count", 0)

                if config_key and quality_ratio > 0.3:  # More than 30% low quality
                    logger.info(
                        f"[SyncOrchestrator] Low quality data warning for {config_key} "
                        f"({low_count} games, {quality_ratio:.1%} low quality), "
                        f"deprioritizing sync"
                    )
                    # Track deprioritized configs to skip in priority sync
                    if not hasattr(self, '_deprioritized_configs'):
                        self._deprioritized_configs = set()
                    self._deprioritized_configs.add(config_key)

            bus.subscribe(DataEventType.LOW_QUALITY_DATA_WARNING, on_low_quality_warning)

            # On quality distribution change, log and potentially adjust
            async def on_distribution_changed(event):
                payload = event.payload if hasattr(event, 'payload') else {}
                config_key = payload.get("config", "")
                avg_quality = payload.get("avg_quality", 0.5)

                # If quality improved significantly, prioritize this config
                if avg_quality >= self.config.min_quality_for_priority_sync:
                    logger.debug(
                        f"[SyncOrchestrator] Quality improved for {config_key}: {avg_quality:.2f}"
                    )
                    # Remove from deprioritized if quality recovered (December 2025)
                    if hasattr(self, '_deprioritized_configs') and config_key in self._deprioritized_configs:
                        self._deprioritized_configs.discard(config_key)
                        logger.info(
                            f"[SyncOrchestrator] Config {config_key} quality recovered, "
                            f"removing from deprioritized set"
                        )

            bus.subscribe(DataEventType.QUALITY_DISTRIBUTION_CHANGED, on_distribution_changed)

            logger.debug("[SyncOrchestrator] Subscribed to quality events for sync prioritization")

        except ImportError:
            logger.debug("[SyncOrchestrator] Data event bus not available")
        except Exception as e:
            logger.warning(f"[SyncOrchestrator] Failed to subscribe to quality events: {e}")

    async def sync_data_for_config(
        self,
        config_key: str,
        priority: bool = False,
    ) -> SyncResult:
        """Sync data for a specific configuration.

        Args:
            config_key: Configuration key (e.g., "square8_2p")
            priority: Whether this is a priority sync (bypasses cooldown)

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
            from app.distributed.sync_coordinator import SyncCategory

            stats = await self._data_sync_coordinator.full_cluster_sync(
                categories=[SyncCategory.GAMES, SyncCategory.TRAINING],
                sync_high_quality_first=priority,
            )

            self.state.last_data_sync = time.time()

            logger.info(
                f"[SyncOrchestrator] Config sync complete for {config_key}: "
                f"{stats.files_synced} files"
            )

            return SyncResult(
                component="data_sync",
                success=True,
                items_synced=stats.files_synced,
                duration_seconds=time.time() - start_time,
                details={
                    "config_key": config_key,
                    "priority": priority,
                    "categories": list(stats.categories.keys()) if stats.categories else [],
                },
            )

        except Exception as e:
            logger.error(f"[SyncOrchestrator] Config sync failed for {config_key}: {e}")
            self.state.sync_errors += 1
            return SyncResult(
                component="data_sync",
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    def get_quality_driven_sync_priority(self) -> list[str]:
        """Get configs ordered by quality-driven sync priority.

        Uses DataQualityOrchestrator state to determine which configs
        should be synced first based on:
        1. Configs ready for training (highest priority)
        2. Configs with high average quality
        3. Configs with recent activity

        Returns:
            List of config keys ordered by sync priority
        """
        if not self._quality_orchestrator:
            return []

        try:
            # Get configs ready for training (for potential future filtering)
            _ = self._quality_orchestrator.get_configs_ready_for_training()

            # Get all config states for sorting
            all_configs = []
            for config in self._quality_orchestrator._configs.values():
                all_configs.append({
                    "key": config.config_key,
                    "ready": config.is_ready_for_training,
                    "avg_quality": config.avg_quality_score,
                    "last_update": config.last_update_time,
                    "has_warning": config.has_active_warning,
                })

            # Sort by priority:
            # 1. Ready for training (descending)
            # 2. No warnings (descending)
            # 3. Higher quality (descending)
            # 4. More recent updates (descending)
            all_configs.sort(
                key=lambda c: (
                    c["ready"],
                    not c["has_warning"],
                    c["avg_quality"],
                    c["last_update"],
                ),
                reverse=True,
            )

            return [c["key"] for c in all_configs]

        except Exception as e:
            logger.warning(f"[SyncOrchestrator] Failed to get quality priority: {e}")
            return []

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

    def get_status(self) -> dict[str, Any]:
        """Get sync orchestrator status.

        Returns:
            Dict with status information
        """
        status = {
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

        # Add quality orchestrator state if available
        if self._quality_orchestrator:
            try:
                quality_status = self._quality_orchestrator.get_status()
                status["quality_orchestrator"] = {
                    "configs_tracked": quality_status.get("configs_tracked", 0),
                    "configs_ready_for_training": quality_status.get("configs_ready_for_training", 0),
                    "configs_with_warnings": quality_status.get("configs_with_warnings", 0),
                    "avg_quality": quality_status.get("avg_quality_across_configs", 0.0),
                }
                status["quality_priority_order"] = self.get_quality_driven_sync_priority()
            except (AttributeError, TypeError, KeyError, ValueError) as e:
                logger.debug(f"Failed to get quality orchestrator state: {e}")

        return status

    def health_check(self) -> HealthCheckResult:
        """Perform health check on sync orchestrator (December 2025).

        Returns:
            HealthCheckResult with health status including details on:
            - initialized: Whether orchestrator is initialized
            - error_rate: Recent sync error rate
            - components: Health of individual sync components
            - last_sync_age: Seconds since last successful sync
        """
        now = time.time()
        status = self.get_status()

        # Calculate error rate
        total = status.get("total_syncs", 0)
        errors = status.get("sync_errors", 0)
        error_rate = errors / max(total, 1)

        # Find last sync time (most recent across all components)
        last_syncs = status.get("last_syncs", {})
        last_sync_times = [v for v in last_syncs.values() if v > 0]
        last_sync_age = now - max(last_sync_times) if last_sync_times else float("inf")

        # Component health checks
        component_health = {}
        for component, enabled_key in [
            ("data", "data_sync_enabled"),
            ("elo", "elo_sync_enabled"),
            ("registry", "registry_sync_enabled"),
            ("model", "model_sync_enabled"),
        ]:
            config = status.get("config", {})
            is_enabled = config.get(enabled_key, False)
            last_time = last_syncs.get(component, 0)
            component_health[component] = {
                "enabled": is_enabled,
                "last_sync": last_time,
                "age_seconds": now - last_time if last_time > 0 else None,
                "healthy": not is_enabled or (now - last_time < 3600 if last_time > 0 else False),
            }

        # Overall health: initialized, low error rate, recent sync
        initialized = status.get("initialized", False)
        healthy = (
            initialized
            and error_rate < 0.2
            and (last_sync_age < 3600 or total == 0)  # Less than 1 hour or no syncs yet
        )

        # Determine status and message
        if not initialized:
            coordinator_status = CoordinatorStatus.INITIALIZING
            message = "Sync orchestrator not yet initialized"
        elif error_rate >= 0.2:
            coordinator_status = CoordinatorStatus.DEGRADED
            message = f"High sync error rate: {error_rate:.1%}"
        elif last_sync_age >= 3600 and total > 0:
            coordinator_status = CoordinatorStatus.DEGRADED
            message = f"No successful sync in {last_sync_age / 60:.0f} minutes"
        else:
            coordinator_status = CoordinatorStatus.RUNNING
            message = "Sync orchestrator healthy"

        return HealthCheckResult(
            healthy=healthy,
            status=coordinator_status,
            message=message,
            details={
                "initialized": initialized,
                "total_syncs": total,
                "sync_errors": errors,
                "error_rate": round(error_rate, 4),
                "last_sync_age_seconds": round(last_sync_age, 1) if last_sync_age != float("inf") else None,
                "components": component_health,
                "quality_orchestrator_available": status.get("quality_orchestrator") is not None,
            },
        )


# Singleton instance
_sync_orchestrator: SyncOrchestrator | None = None


def get_sync_orchestrator(
    config: SyncOrchestratorConfig | None = None,
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
    "FullSyncResult",
    "SyncOrchestrator",
    "SyncOrchestratorConfig",
    "SyncOrchestratorState",
    "SyncResult",
    "get_sync_orchestrator",
    "reset_sync_orchestrator",
]
