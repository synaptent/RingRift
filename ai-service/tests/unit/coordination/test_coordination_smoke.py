"""Smoke tests for coordination infrastructure.

These tests verify that the coordination components are properly wired
and can be initialized without errors. They focus on integration rather
than individual component behavior.

December 2025: Added after identifying event subscription gaps and
ensuring proper wiring of the coordination infrastructure.
"""

from __future__ import annotations

import pytest


class TestEventSubscriptionWiring:
    """Smoke tests for event subscription wiring."""

    def test_data_pipeline_orchestrator_subscriptions(self):
        """DataPipelineOrchestrator subscribes to all required events."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orch = DataPipelineOrchestrator()

        # Check the orchestrator has the required handler methods
        assert hasattr(orch, "_on_selfplay_complete")
        assert hasattr(orch, "_on_sync_complete")
        assert hasattr(orch, "_on_npz_export_complete")  # NPZ export handler
        assert hasattr(orch, "_on_training_complete")
        assert hasattr(orch, "_on_evaluation_complete")
        assert hasattr(orch, "_on_promotion_complete")

        # Check orphan games handlers (Dec 2025 fix)
        assert hasattr(orch, "_on_orphan_games_detected")
        assert hasattr(orch, "_on_orphan_games_registered")

        # Check exploration and sync feedback handlers (Dec 2025 fix)
        assert hasattr(orch, "_on_exploration_boost")
        assert hasattr(orch, "_on_sync_triggered")

    def test_sync_facade_methods(self):
        """SyncFacade has all required sync methods."""
        from app.coordination.sync_facade import SyncFacade, get_sync_facade

        facade = SyncFacade()

        # Core sync methods
        assert hasattr(facade, "sync")
        assert hasattr(facade, "trigger_priority_sync")  # Dec 2025 addition
        assert hasattr(facade, "get_stats")

        # Backend selection
        assert hasattr(facade, "_select_backend")
        assert hasattr(facade, "_execute_sync")

    def test_sync_router_node_recovery_handlers(self):
        """SyncRouter handles node recovery events."""
        from app.coordination.sync_router import SyncRouter

        router = SyncRouter()

        # Check node recovery handlers exist (Dec 2025 fix)
        assert hasattr(router, "_on_node_recovered")
        assert hasattr(router, "_on_host_online")
        assert hasattr(router, "_on_host_offline")

    def test_daemon_factory_has_all_types(self):
        """DaemonFactory has specs for critical daemon types."""
        from app.coordination.daemon_factory import get_daemon_factory
        from app.coordination.daemon_types import DaemonType

        factory = get_daemon_factory()

        # Check critical daemon types have specs
        critical_types = [
            DaemonType.AUTO_SYNC,
            DaemonType.QUEUE_POPULATOR,
            DaemonType.DATA_PIPELINE,
            DaemonType.FEEDBACK_LOOP,
        ]

        for daemon_type in critical_types:
            spec = factory.get_spec(daemon_type)
            assert spec is not None, f"Missing spec for {daemon_type}"


class TestClusterStatusMonitorAsync:
    """Tests for cluster status monitor async capabilities."""

    def test_async_methods_exist(self):
        """ClusterMonitor has async methods for non-blocking operation."""
        from app.coordination.cluster_status_monitor import ClusterMonitor

        monitor = ClusterMonitor()

        # Check async SSH method
        assert hasattr(monitor, "_async_run_ssh_command")
        assert hasattr(monitor, "_async_check_host_connectivity")

        # Check async status methods
        assert hasattr(monitor, "get_node_status_async")
        assert hasattr(monitor, "get_cluster_status_async")

    def test_run_forever_uses_async(self):
        """run_forever uses async status methods."""
        import inspect

        from app.coordination.cluster_status_monitor import ClusterMonitor

        monitor = ClusterMonitor()
        source = inspect.getsource(monitor.run_forever)

        # Verify it uses async version
        assert "get_cluster_status_async" in source
        assert "await" in source


class TestPathConfiguration:
    """Tests for centralized path configuration."""

    def test_all_paths_defined(self):
        """All standard paths are defined in paths module."""
        # Verify all paths are Path objects
        from pathlib import Path

        from app.utils.paths import (
            AI_SERVICE_ROOT,
            CONFIG_DIR,
            COORDINATION_DIR,
            DATA_DIR,
            GAMES_DIR,
            LOGS_DIR,
            MODELS_DIR,
            TRAINING_DIR,
        )

        assert isinstance(AI_SERVICE_ROOT, Path)
        assert isinstance(DATA_DIR, Path)
        assert isinstance(GAMES_DIR, Path)
        assert isinstance(TRAINING_DIR, Path)
        assert isinstance(COORDINATION_DIR, Path)
        assert isinstance(MODELS_DIR, Path)
        assert isinstance(LOGS_DIR, Path)
        assert isinstance(CONFIG_DIR, Path)

    def test_path_helper_functions(self):
        """Path helper functions work correctly."""
        from pathlib import Path

        from app.utils.paths import (
            get_games_db_path,
            get_selfplay_db_path,
            get_training_npz_path,
        )

        # Test helper functions return paths
        games_path = get_games_db_path("hex8_2p")
        training_path = get_training_npz_path("hex8_2p")
        selfplay_path = get_selfplay_db_path("hex8_2p")

        assert isinstance(games_path, Path)
        assert isinstance(training_path, Path)
        assert isinstance(selfplay_path, Path)

        # Verify path structure
        assert games_path.name == "hex8_2p.db"
        assert training_path.name == "hex8_2p.npz"
        assert selfplay_path.name == "selfplay_hex8_2p.db"


class TestCoordinationDefaults:
    """Tests for coordination defaults configuration."""

    def test_timeout_defaults_accessible(self):
        """Timeout defaults are accessible from coordination_defaults."""
        from app.config.coordination_defaults import (
            LockDefaults,
            SyncDefaults,
            TransportDefaults,
        )

        # Transport timeouts
        assert TransportDefaults.HTTP_TIMEOUT > 0
        assert TransportDefaults.SSH_TIMEOUT > 0
        assert TransportDefaults.CONNECT_TIMEOUT > 0

        # Lock timeouts
        assert LockDefaults.LOCK_TIMEOUT > 0
        assert LockDefaults.ACQUIRE_TIMEOUT > 0

        # Sync timeouts
        assert SyncDefaults.LOCK_TIMEOUT > 0

    def test_get_timeout_function(self):
        """get_timeout() returns appropriate values."""
        from app.config.coordination_defaults import get_timeout

        # Check common timeout lookups
        http_timeout = get_timeout("http")
        ssh_timeout = get_timeout("ssh")
        connect_timeout = get_timeout("connect")

        assert http_timeout > 0
        assert ssh_timeout > 0
        assert connect_timeout > 0


class TestDataEventTypes:
    """Tests for data event type coverage."""

    def test_orphan_games_events_exist(self):
        """ORPHAN_GAMES events are defined."""
        from app.distributed.data_events import DataEventType

        # Check orphan games events exist
        assert hasattr(DataEventType, "ORPHAN_GAMES_DETECTED")
        assert hasattr(DataEventType, "ORPHAN_GAMES_REGISTERED")

    def test_core_pipeline_events_exist(self):
        """Core pipeline events are defined."""
        from app.distributed.data_events import DataEventType

        # Check core pipeline events
        assert hasattr(DataEventType, "SELFPLAY_COMPLETE")
        assert hasattr(DataEventType, "DATA_SYNC_COMPLETED")
        assert hasattr(DataEventType, "TRAINING_COMPLETED")
        assert hasattr(DataEventType, "EVALUATION_COMPLETED")
        assert hasattr(DataEventType, "MODEL_PROMOTED")

    def test_new_pipeline_events_exist(self):
        """Newly added pipeline events are defined (Dec 2025 Phase 11)."""
        from app.distributed.data_events import DataEventType

        # Check new events added in Phase 11
        assert hasattr(DataEventType, "NEW_GAMES_AVAILABLE")
        assert hasattr(DataEventType, "REGRESSION_DETECTED")
        assert hasattr(DataEventType, "PROMOTION_FAILED")


class TestHealthCheckCoverage:
    """Tests for health_check() method coverage across coordinators.

    December 2025 Phase 11: Verifies that all critical coordinator classes
    have health_check() methods returning HealthCheckResult.
    """

    def test_base_orchestrator_health_check(self):
        """BaseOrchestrator has health_check() method."""
        from app.coordination.base_orchestrator import BaseOrchestrator

        assert hasattr(BaseOrchestrator, "health_check")

    def test_coordinator_base_health_check(self):
        """CoordinatorBase has health_check() method."""
        from app.coordination.coordinator_base import CoordinatorBase

        assert hasattr(CoordinatorBase, "health_check")

    def test_daemon_manager_health_check(self):
        """DaemonManager has health_check() method."""
        from app.coordination.daemon_manager import DaemonManager

        assert hasattr(DaemonManager, "health_check")

        # Verify it returns HealthCheckResult
        manager = DaemonManager.get_instance()
        try:
            result = manager.health_check()
            assert hasattr(result, "healthy")
            assert hasattr(result, "status")
            assert hasattr(result, "message")
            assert hasattr(result, "details")
        finally:
            DaemonManager.reset_instance()

    def test_daemon_lifecycle_manager_health_check(self):
        """DaemonLifecycleManager has health_check() method."""
        from app.coordination.daemon_lifecycle import DaemonLifecycleManager

        assert hasattr(DaemonLifecycleManager, "health_check")

    def test_health_check_orchestrator_health_check(self):
        """HealthCheckOrchestrator has its own health_check() method."""
        from app.coordination.health_check_orchestrator import HealthCheckOrchestrator

        assert hasattr(HealthCheckOrchestrator, "health_check")

    def test_multi_provider_orchestrator_health_check(self):
        """MultiProviderOrchestrator has health_check() method."""
        from app.coordination.multi_provider_orchestrator import MultiProviderOrchestrator

        assert hasattr(MultiProviderOrchestrator, "health_check")

    def test_orchestrator_registry_health_check(self):
        """OrchestratorRegistry has health_check() method."""
        from app.coordination.orchestrator_registry import OrchestratorRegistry

        assert hasattr(OrchestratorRegistry, "health_check")

    def test_sync_scheduler_inherits_health_check(self):
        """SyncScheduler inherits health_check() from CoordinatorBase."""
        from app.coordination.sync_coordinator import SyncScheduler

        assert hasattr(SyncScheduler, "health_check")


class TestDataPipelineEventHandlers:
    """Tests for DataPipelineOrchestrator event handler coverage.

    December 2025 Phase 11: Verifies new event handlers are present.
    """

    def test_new_games_available_handler(self):
        """DataPipelineOrchestrator has NEW_GAMES_AVAILABLE handler."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orch = DataPipelineOrchestrator()
        assert hasattr(orch, "_on_new_games_available")

    def test_regression_detected_handler(self):
        """DataPipelineOrchestrator has REGRESSION_DETECTED handler."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orch = DataPipelineOrchestrator()
        assert hasattr(orch, "_on_regression_detected")

    def test_promotion_failed_handler(self):
        """DataPipelineOrchestrator has PROMOTION_FAILED handler."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orch = DataPipelineOrchestrator()
        assert hasattr(orch, "_on_promotion_failed")

    def test_all_pipeline_handlers_present(self):
        """DataPipelineOrchestrator has all core pipeline handlers."""
        from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

        orch = DataPipelineOrchestrator()

        # Stage event handlers
        stage_handlers = [
            "_on_selfplay_complete",
            "_on_sync_complete",
            "_on_npz_export_complete",
            "_on_training_started",
            "_on_training_complete",
            "_on_training_failed",
            "_on_evaluation_complete",
            "_on_promotion_complete",
            "_on_iteration_complete",
        ]

        for handler in stage_handlers:
            assert hasattr(orch, handler), f"Missing handler: {handler}"

        # Data event handlers (Dec 2025 additions)
        data_handlers = [
            "_on_new_games_available",
            "_on_regression_detected",
            "_on_promotion_failed",
            "_on_orphan_games_detected",
            "_on_orphan_games_registered",
            "_on_exploration_boost",
            "_on_sync_triggered",
            "_on_data_stale",
        ]

        for handler in data_handlers:
            assert hasattr(orch, handler), f"Missing handler: {handler}"


class TestMultiModuleIntegration:
    """Integration tests for multi-module interactions.

    December 2025 Phase 11: Verifies that modules can work together correctly.
    """

    def test_health_check_result_consistency(self):
        """All health_check() methods return consistent HealthCheckResult."""
        from app.coordination.protocols import HealthCheckResult

        # Test that we can create HealthCheckResult with all fields
        result = HealthCheckResult(
            healthy=True,
            status="running",
            message="",
            details={"test": True},
        )
        assert result.healthy is True
        assert result.status == "running"
        assert result.message == ""
        assert result.details == {"test": True}

    def test_coordinator_base_subclass_gets_health_check(self):
        """Subclasses of CoordinatorBase automatically get health_check()."""
        from app.coordination.coordinator_base import CoordinatorBase

        # SyncScheduler is a subclass of CoordinatorBase
        from app.coordination.sync_coordinator import SyncScheduler

        scheduler = SyncScheduler()

        # Should have health_check from parent
        result = scheduler.health_check()
        assert hasattr(result, "healthy")
        assert hasattr(result, "details")

    def test_daemon_manager_can_check_all_coordinators(self):
        """DaemonManager can use health_check() on any coordinator."""
        from app.coordination.daemon_manager import DaemonManager
        from app.coordination.protocols import HealthCheckResult

        manager = DaemonManager.get_instance()
        try:
            # Manager itself has health_check
            result = manager.health_check()
            assert isinstance(result, HealthCheckResult)

            # Verify structure matches protocol
            assert hasattr(result, "healthy")
            assert hasattr(result, "status")
            assert hasattr(result, "message")
            assert hasattr(result, "details")
        finally:
            DaemonManager.reset_instance()
