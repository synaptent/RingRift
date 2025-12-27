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
