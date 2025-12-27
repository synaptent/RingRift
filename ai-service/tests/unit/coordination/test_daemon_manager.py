"""Tests for app.coordination.daemon_manager - Unified Daemon Manager.

This module tests the DaemonManager which coordinates lifecycle of all
background services including sync daemons, health checks, and event watchers.

Note: Async daemon lifecycle tests are limited to avoid timeout issues with
long-running background tasks.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.daemon_manager import (
    DaemonInfo,
    DaemonManager,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
    get_daemon_manager,
    reset_daemon_manager,
)


# =============================================================================
# DaemonType Tests
# =============================================================================


class TestDaemonType:
    """Tests for DaemonType enum."""

    def test_sync_daemons_defined(self):
        """Sync daemon types should be defined."""
        assert DaemonType.SYNC_COORDINATOR.value == "sync_coordinator"
        assert DaemonType.HIGH_QUALITY_SYNC.value == "high_quality_sync"
        assert DaemonType.ELO_SYNC.value == "elo_sync"
        assert DaemonType.MODEL_SYNC.value == "model_sync"

    def test_monitoring_daemons_defined(self):
        """Monitoring daemon types should be defined."""
        assert DaemonType.HEALTH_CHECK.value == "health_check"
        assert DaemonType.CLUSTER_MONITOR.value == "cluster_monitor"
        assert DaemonType.QUEUE_MONITOR.value == "queue_monitor"

    def test_event_daemons_defined(self):
        """Event processing daemon types should be defined."""
        assert DaemonType.EVENT_ROUTER.value == "event_router"
        assert DaemonType.CROSS_PROCESS_POLLER.value == "cross_process_poller"

    def test_pipeline_daemons_defined(self):
        """Pipeline daemon types should be defined."""
        assert DaemonType.DATA_PIPELINE.value == "data_pipeline"
        assert DaemonType.TRAINING_NODE_WATCHER.value == "training_node_watcher"

    def test_p2p_daemons_defined(self):
        """P2P service daemon types should be defined."""
        assert DaemonType.P2P_BACKEND.value == "p2p_backend"
        assert DaemonType.GOSSIP_SYNC.value == "gossip_sync"
        assert DaemonType.DATA_SERVER.value == "data_server"

    def test_daemon_count(self):
        """Should have expected number of daemon types."""
        assert len(DaemonType) >= 16


# =============================================================================
# DaemonState Tests
# =============================================================================


class TestDaemonState:
    """Tests for DaemonState enum."""

    def test_all_states_defined(self):
        """All states should be defined."""
        assert DaemonState.STOPPED.value == "stopped"
        assert DaemonState.STARTING.value == "starting"
        assert DaemonState.RUNNING.value == "running"
        assert DaemonState.STOPPING.value == "stopping"
        assert DaemonState.FAILED.value == "failed"
        assert DaemonState.RESTARTING.value == "restarting"
        assert DaemonState.IMPORT_FAILED.value == "import_failed"

    def test_state_count(self):
        """Should have exactly 7 states."""
        assert len(DaemonState) == 7


# =============================================================================
# DaemonInfo Tests
# =============================================================================


class TestDaemonInfo:
    """Tests for DaemonInfo dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        info = DaemonInfo(daemon_type=DaemonType.HEALTH_CHECK)

        assert info.daemon_type == DaemonType.HEALTH_CHECK
        assert info.state == DaemonState.STOPPED
        assert info.task is None
        assert info.start_time == 0.0
        assert info.restart_count == 0
        assert info.last_error is None
        assert info.health_check_interval == 60.0
        assert info.auto_restart is True
        assert info.max_restarts == 5
        assert info.restart_delay == 5.0
        assert info.depends_on == []

    def test_uptime_when_stopped(self):
        """uptime_seconds should be 0 when stopped."""
        info = DaemonInfo(daemon_type=DaemonType.HEALTH_CHECK)
        assert info.uptime_seconds == 0.0

    def test_uptime_when_running(self):
        """uptime_seconds should calculate correctly when running."""
        info = DaemonInfo(
            daemon_type=DaemonType.HEALTH_CHECK,
            state=DaemonState.RUNNING,
            start_time=time.time() - 10.0,
        )
        assert 9.5 < info.uptime_seconds < 11.0

    def test_uptime_with_future_start_time(self):
        """uptime_seconds should handle edge cases."""
        info = DaemonInfo(
            daemon_type=DaemonType.HEALTH_CHECK,
            state=DaemonState.STOPPED,  # Not running
            start_time=time.time() - 10.0,
        )
        # Not running, so uptime should be 0
        assert info.uptime_seconds == 0.0

    def test_dependencies_can_be_set(self):
        """Should accept dependency list."""
        info = DaemonInfo(
            daemon_type=DaemonType.DATA_PIPELINE,
            depends_on=[DaemonType.EVENT_ROUTER, DaemonType.SYNC_COORDINATOR],
        )
        assert len(info.depends_on) == 2
        assert DaemonType.EVENT_ROUTER in info.depends_on


# =============================================================================
# DaemonManagerConfig Tests
# =============================================================================


class TestDaemonManagerConfig:
    """Tests for DaemonManagerConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = DaemonManagerConfig()

        assert config.auto_start is False
        assert config.health_check_interval == 30.0
        assert config.shutdown_timeout == 10.0
        assert config.auto_restart_failed is True
        assert config.max_restart_attempts == 5

    def test_custom_config(self):
        """Should accept custom values."""
        config = DaemonManagerConfig(
            auto_start=True,
            health_check_interval=60.0,
            shutdown_timeout=30.0,
        )

        assert config.auto_start is True
        assert config.health_check_interval == 60.0
        assert config.shutdown_timeout == 30.0


# =============================================================================
# DaemonManager Init Tests
# =============================================================================


class TestDaemonManagerInit:
    """Tests for DaemonManager initialization."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_init_with_default_config(self):
        """Should initialize with default config."""
        manager = DaemonManager()

        assert manager.config is not None
        assert manager._running is False
        assert manager._health_task is None

    def test_init_with_custom_config(self):
        """Should accept custom config."""
        config = DaemonManagerConfig(
            health_check_interval=1.0,
            shutdown_timeout=2.0,
        )
        manager = DaemonManager(config)

        assert manager.config.health_check_interval == 1.0
        assert manager.config.shutdown_timeout == 2.0

    def test_default_factories_registered(self):
        """Default factories should be registered."""
        manager = DaemonManager()

        assert DaemonType.SYNC_COORDINATOR in manager._factories
        assert DaemonType.EVENT_ROUTER in manager._factories
        assert DaemonType.HEALTH_CHECK in manager._factories


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_get_instance_creates_singleton(self):
        """get_instance should create singleton."""
        config = DaemonManagerConfig()

        manager1 = DaemonManager.get_instance(config)
        manager2 = DaemonManager.get_instance()

        assert manager1 is manager2

    def test_reset_instance_clears_singleton(self):
        """reset_instance should clear singleton."""
        config = DaemonManagerConfig()

        manager1 = DaemonManager.get_instance(config)
        DaemonManager.reset_instance()
        manager2 = DaemonManager.get_instance(config)

        assert manager1 is not manager2

    def test_get_daemon_manager_function(self):
        """get_daemon_manager should return singleton."""
        config = DaemonManagerConfig()

        manager1 = get_daemon_manager(config)
        manager2 = get_daemon_manager()

        assert manager1 is manager2


# =============================================================================
# Factory Registration Tests
# =============================================================================


class TestFactoryRegistration:
    """Tests for daemon factory registration."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()

    def test_register_simple_factory(self):
        """Should register a simple factory."""
        manager = DaemonManager()

        async def my_factory():
            pass

        manager.register_factory(DaemonType.MODEL_SYNC, my_factory)

        assert DaemonType.MODEL_SYNC in manager._factories
        assert DaemonType.MODEL_SYNC in manager._daemons

    def test_register_factory_with_dependencies(self):
        """Should register factory with dependencies."""
        manager = DaemonManager()

        async def my_factory():
            pass

        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            my_factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        info = manager._daemons[DaemonType.DATA_PIPELINE]
        assert DaemonType.EVENT_ROUTER in info.depends_on

    def test_register_factory_with_config(self):
        """Should register factory with custom config."""
        manager = DaemonManager()

        async def my_factory():
            pass

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            my_factory,
            health_check_interval=120.0,
            auto_restart=False,
            max_restarts=10,
        )

        info = manager._daemons[DaemonType.MODEL_SYNC]
        assert info.health_check_interval == 120.0
        assert info.auto_restart is False
        assert info.max_restarts == 10


# =============================================================================
# Dependency Sorting Tests
# =============================================================================


class TestDependencySorting:
    """Tests for dependency-based sorting."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()

    def test_sort_by_dependencies(self):
        """Dependencies should be sorted correctly."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        result = manager._sort_by_dependencies([
            DaemonType.DATA_PIPELINE,
            DaemonType.EVENT_ROUTER,
        ])

        # EVENT_ROUTER should come before DATA_PIPELINE
        er_idx = result.index(DaemonType.EVENT_ROUTER)
        dp_idx = result.index(DaemonType.DATA_PIPELINE)
        assert er_idx < dp_idx

    def test_sort_handles_no_deps(self):
        """Should handle daemons with no dependencies."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.MODEL_SYNC, factory)
        manager.register_factory(DaemonType.ELO_SYNC, factory)

        result = manager._sort_by_dependencies([
            DaemonType.MODEL_SYNC,
            DaemonType.ELO_SYNC,
        ])

        assert len(result) == 2
        assert DaemonType.MODEL_SYNC in result
        assert DaemonType.ELO_SYNC in result

    def test_sort_circular_dependency(self):
        """Should handle circular dependencies gracefully."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager._daemons[DaemonType.MODEL_SYNC] = DaemonInfo(
            daemon_type=DaemonType.MODEL_SYNC,
            depends_on=[DaemonType.ELO_SYNC],
        )
        manager._daemons[DaemonType.ELO_SYNC] = DaemonInfo(
            daemon_type=DaemonType.ELO_SYNC,
            depends_on=[DaemonType.MODEL_SYNC],
        )
        manager._factories[DaemonType.MODEL_SYNC] = factory
        manager._factories[DaemonType.ELO_SYNC] = factory

        # Should not hang or crash
        result = manager._sort_by_dependencies([
            DaemonType.MODEL_SYNC,
            DaemonType.ELO_SYNC,
        ])

        assert len(result) == 2

    def test_sort_with_chain_dependencies(self):
        """Should handle A -> B -> C dependency chains."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.SYNC_COORDINATOR, factory)
        manager.register_factory(
            DaemonType.EVENT_ROUTER,
            factory,
            depends_on=[DaemonType.SYNC_COORDINATOR],
        )
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        result = manager._sort_by_dependencies([
            DaemonType.DATA_PIPELINE,
            DaemonType.SYNC_COORDINATOR,
            DaemonType.EVENT_ROUTER,
        ])

        sc_idx = result.index(DaemonType.SYNC_COORDINATOR)
        er_idx = result.index(DaemonType.EVENT_ROUTER)
        dp_idx = result.index(DaemonType.DATA_PIPELINE)

        assert sc_idx < er_idx < dp_idx


# =============================================================================
# Status Tests
# =============================================================================


class TestStatus:
    """Tests for status reporting."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()

    def test_get_status_empty(self):
        """Should return status for empty manager."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        status = manager.get_status()

        assert status["running"] is False
        assert status["daemons"] == {}
        assert status["summary"]["total"] == 0
        assert status["summary"]["running"] == 0

    def test_get_status_with_daemons(self):
        """Should return status for registered daemons."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.MODEL_SYNC, factory)

        status = manager.get_status()

        assert "model_sync" in status["daemons"]
        assert status["daemons"]["model_sync"]["state"] == "stopped"
        assert status["summary"]["total"] == 1
        assert status["summary"]["stopped"] == 1

    def test_get_status_counts_states(self):
        """Should correctly count daemon states."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        # Add daemons in different states
        manager._daemons[DaemonType.MODEL_SYNC] = DaemonInfo(
            daemon_type=DaemonType.MODEL_SYNC,
            state=DaemonState.RUNNING,
        )
        manager._daemons[DaemonType.ELO_SYNC] = DaemonInfo(
            daemon_type=DaemonType.ELO_SYNC,
            state=DaemonState.FAILED,
        )
        manager._daemons[DaemonType.HEALTH_CHECK] = DaemonInfo(
            daemon_type=DaemonType.HEALTH_CHECK,
            state=DaemonState.STOPPED,
        )

        status = manager.get_status()

        assert status["summary"]["total"] == 3
        assert status["summary"]["running"] == 1
        assert status["summary"]["failed"] == 1
        assert status["summary"]["stopped"] == 1

    def test_is_running_false_for_stopped(self):
        """is_running should return False for stopped daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.MODEL_SYNC, factory)
        assert manager.is_running(DaemonType.MODEL_SYNC) is False

    def test_is_running_true_for_running(self):
        """is_running should return True for running daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager._daemons[DaemonType.MODEL_SYNC] = DaemonInfo(
            daemon_type=DaemonType.MODEL_SYNC,
            state=DaemonState.RUNNING,
        )

        assert manager.is_running(DaemonType.MODEL_SYNC) is True

    def test_is_running_unknown_daemon(self):
        """is_running should return False for unknown daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        assert manager.is_running(DaemonType.MODEL_SYNC) is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()

    def test_sync_shutdown_no_loop(self):
        """_sync_shutdown should handle no running loop."""
        manager = DaemonManager()
        # Should not raise
        manager._sync_shutdown()


# =============================================================================
# Daemon Profiles Tests (December 2025)
# =============================================================================


class TestDaemonProfiles:
    """Tests for daemon profile definitions."""

    def test_coordinator_profile_exists(self):
        """Coordinator profile should be defined."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        assert "coordinator" in DAEMON_PROFILES

    def test_training_node_profile_exists(self):
        """Training node profile should be defined."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        assert "training_node" in DAEMON_PROFILES

    def test_ephemeral_profile_exists(self):
        """Ephemeral profile should be defined."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        assert "ephemeral" in DAEMON_PROFILES

    def test_selfplay_profile_exists(self):
        """Selfplay profile should be defined."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        assert "selfplay" in DAEMON_PROFILES

    def test_minimal_profile_exists(self):
        """Minimal profile should be defined."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        assert "minimal" in DAEMON_PROFILES

    def test_full_profile_exists(self):
        """Full profile should be defined."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        assert "full" in DAEMON_PROFILES

    def test_coordinator_has_event_router(self):
        """Coordinator profile should include event_router (critical dependency)."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        profile = DAEMON_PROFILES["coordinator"]
        assert DaemonType.EVENT_ROUTER in profile

    def test_training_node_has_event_router(self):
        """Training node profile should include event_router."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        profile = DAEMON_PROFILES["training_node"]
        assert DaemonType.EVENT_ROUTER in profile

    def test_ephemeral_has_ephemeral_sync(self):
        """Ephemeral profile should include ephemeral_sync."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        profile = DAEMON_PROFILES["ephemeral"]
        assert DaemonType.EPHEMERAL_SYNC in profile

    def test_minimal_is_truly_minimal(self):
        """Minimal profile should have only 1 daemon."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        profile = DAEMON_PROFILES["minimal"]
        assert len(profile) == 1
        assert DaemonType.EVENT_ROUTER in profile

    def test_full_profile_includes_all(self):
        """Full profile should include all daemon types."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        full_profile = DAEMON_PROFILES["full"]
        assert len(full_profile) == len(DaemonType)

    def test_profiles_have_no_duplicates(self):
        """Profiles should not have duplicate daemon types."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        for name, daemons in DAEMON_PROFILES.items():
            unique = set(daemons)
            assert len(unique) == len(daemons), f"Profile {name} has duplicates"

    def test_coordinator_has_critical_daemons(self):
        """Coordinator profile should include critical coordination daemons."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        profile = DAEMON_PROFILES["coordinator"]
        critical = [
            DaemonType.FEEDBACK_LOOP,
            DaemonType.CLUSTER_MONITOR,
            DaemonType.AUTO_SYNC,
        ]
        for daemon in critical:
            assert daemon in profile, f"Missing {daemon} in coordinator profile"

    def test_training_node_has_training_daemons(self):
        """Training node profile should include training-related daemons."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        profile = DAEMON_PROFILES["training_node"]
        training_daemons = [
            DaemonType.DATA_PIPELINE,
            DaemonType.EVALUATION,
            DaemonType.FEEDBACK_LOOP,
        ]
        for daemon in training_daemons:
            assert daemon in profile, f"Missing {daemon} in training_node profile"


# =============================================================================
# Start Profile Tests
# =============================================================================


class TestStartProfile:
    """Tests for start_profile function."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    @pytest.mark.asyncio
    async def test_start_profile_minimal(self):
        """start_profile should start minimal profile."""
        from app.coordination.daemon_manager import start_profile, DAEMON_PROFILES

        with patch.object(DaemonManager, 'start_all', new_callable=AsyncMock) as mock_start:
            mock_start.return_value = {DaemonType.EVENT_ROUTER: True}
            results = await start_profile("minimal")
            mock_start.assert_called_once()
            # Verify only minimal daemons passed
            call_args = mock_start.call_args[0][0]
            assert call_args == DAEMON_PROFILES["minimal"]

    @pytest.mark.asyncio
    async def test_start_profile_unknown_raises(self):
        """start_profile should raise ValueError for unknown profile."""
        from app.coordination.daemon_manager import start_profile

        with pytest.raises(ValueError) as exc_info:
            await start_profile("nonexistent_profile")
        assert "Unknown profile" in str(exc_info.value)
        assert "nonexistent_profile" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_start_profile_lists_available(self):
        """start_profile error should list available profiles."""
        from app.coordination.daemon_manager import start_profile, DAEMON_PROFILES

        with pytest.raises(ValueError) as exc_info:
            await start_profile("bad_profile")
        error_msg = str(exc_info.value)
        for profile_name in DAEMON_PROFILES.keys():
            assert profile_name in error_msg


# =============================================================================
# Startup Order Tests
# =============================================================================


class TestStartupOrder:
    """Tests for daemon startup order and dependency resolution."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_event_router_starts_first(self):
        """EVENT_ROUTER should be in first position when present."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        # Register with EVENT_ROUTER depending on nothing
        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )
        manager.register_factory(
            DaemonType.FEEDBACK_LOOP,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        sorted_types = manager._sort_by_dependencies([
            DaemonType.FEEDBACK_LOOP,
            DaemonType.DATA_PIPELINE,
            DaemonType.EVENT_ROUTER,
        ])

        # EVENT_ROUTER should be first
        assert sorted_types[0] == DaemonType.EVENT_ROUTER

    def test_multi_level_dependencies(self):
        """Multi-level dependencies should be correctly ordered."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        # Create A -> B -> C -> D chain
        manager.register_factory(DaemonType.EVENT_ROUTER, factory)  # Level 0
        manager.register_factory(
            DaemonType.SYNC_COORDINATOR,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],  # Level 1
        )
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            factory,
            depends_on=[DaemonType.SYNC_COORDINATOR],  # Level 2
        )
        manager.register_factory(
            DaemonType.FEEDBACK_LOOP,
            factory,
            depends_on=[DaemonType.DATA_PIPELINE],  # Level 3
        )

        sorted_types = manager._sort_by_dependencies([
            DaemonType.FEEDBACK_LOOP,
            DaemonType.DATA_PIPELINE,
            DaemonType.SYNC_COORDINATOR,
            DaemonType.EVENT_ROUTER,
        ])

        # Verify order
        er_idx = sorted_types.index(DaemonType.EVENT_ROUTER)
        sc_idx = sorted_types.index(DaemonType.SYNC_COORDINATOR)
        dp_idx = sorted_types.index(DaemonType.DATA_PIPELINE)
        fl_idx = sorted_types.index(DaemonType.FEEDBACK_LOOP)

        assert er_idx < sc_idx < dp_idx < fl_idx

    def test_diamond_dependencies(self):
        """Diamond dependency pattern should be handled correctly."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        # Create diamond: A -> B, A -> C, B -> D, C -> D
        manager.register_factory(DaemonType.EVENT_ROUTER, factory)  # A
        manager.register_factory(
            DaemonType.SYNC_COORDINATOR,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],  # B
        )
        manager.register_factory(
            DaemonType.HEALTH_CHECK,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],  # C
        )
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            factory,
            depends_on=[DaemonType.SYNC_COORDINATOR, DaemonType.HEALTH_CHECK],  # D
        )

        sorted_types = manager._sort_by_dependencies([
            DaemonType.DATA_PIPELINE,
            DaemonType.HEALTH_CHECK,
            DaemonType.SYNC_COORDINATOR,
            DaemonType.EVENT_ROUTER,
        ])

        # A must come before B and C, B and C must come before D
        a_idx = sorted_types.index(DaemonType.EVENT_ROUTER)
        b_idx = sorted_types.index(DaemonType.SYNC_COORDINATOR)
        c_idx = sorted_types.index(DaemonType.HEALTH_CHECK)
        d_idx = sorted_types.index(DaemonType.DATA_PIPELINE)

        assert a_idx < b_idx
        assert a_idx < c_idx
        assert b_idx < d_idx
        assert c_idx < d_idx

    def test_partial_dependency_list(self):
        """Sort should work with subset of registered daemons."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )
        manager.register_factory(DaemonType.MODEL_SYNC, factory)  # Not included in sort

        # Sort only 2 of 3 registered daemons
        sorted_types = manager._sort_by_dependencies([
            DaemonType.DATA_PIPELINE,
            DaemonType.EVENT_ROUTER,
        ])

        assert len(sorted_types) == 2
        assert DaemonType.MODEL_SYNC not in sorted_types


# =============================================================================
# Health Loop Tests
# =============================================================================


class TestHealthLoop:
    """Tests for health monitoring loop."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    @pytest.mark.asyncio
    async def test_health_loop_starts(self):
        """Health loop should start when first daemon starts."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            while True:
                await asyncio.sleep(0.1)

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)

        # Before start
        assert manager._health_task is None or manager._health_task.done()

        await manager.start(DaemonType.EVENT_ROUTER)

        # After start - health loop should be running
        assert manager._health_task is not None
        assert not manager._health_task.done()

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_health_loop_only_starts_once(self):
        """Health loop should not restart if already running."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            while True:
                await asyncio.sleep(0.1)

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        manager.register_factory(DaemonType.MODEL_SYNC, factory)

        await manager.start(DaemonType.EVENT_ROUTER)
        first_health_task = manager._health_task

        await manager.start(DaemonType.MODEL_SYNC)
        second_health_task = manager._health_task

        # Same task object
        assert first_health_task is second_health_task

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_ensure_health_loop_running_idempotent(self):
        """_ensure_health_loop_running should be safe to call multiple times."""
        manager = DaemonManager()

        await manager._ensure_health_loop_running()
        first_task = manager._health_task

        await manager._ensure_health_loop_running()
        second_task = manager._health_task

        assert first_task is second_task
        assert manager._running is True

        await manager.shutdown()


# =============================================================================
# Auto-Restart Tests
# =============================================================================


class TestAutoRestart:
    """Tests for auto-restart behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_auto_restart_default_enabled(self):
        """Auto-restart should be enabled by default."""
        manager = DaemonManager()

        async def factory():
            pass

        manager.register_factory(DaemonType.MODEL_SYNC, factory)
        info = manager._daemons[DaemonType.MODEL_SYNC]

        assert info.auto_restart is True

    def test_auto_restart_can_be_disabled(self):
        """Auto-restart can be disabled per daemon."""
        manager = DaemonManager()

        async def factory():
            pass

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            factory,
            auto_restart=False,
        )
        info = manager._daemons[DaemonType.MODEL_SYNC]

        assert info.auto_restart is False

    def test_max_restarts_configured(self):
        """Max restarts should be configurable."""
        manager = DaemonManager()

        async def factory():
            pass

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            factory,
            max_restarts=10,
        )
        info = manager._daemons[DaemonType.MODEL_SYNC]

        assert info.max_restarts == 10

    def test_restart_count_tracked(self):
        """Restart count should be tracked."""
        manager = DaemonManager()

        async def factory():
            pass

        manager.register_factory(DaemonType.MODEL_SYNC, factory)
        info = manager._daemons[DaemonType.MODEL_SYNC]

        # Initially 0
        assert info.restart_count == 0

        # Manually increment for testing
        info.restart_count += 1
        assert info.restart_count == 1


# =============================================================================
# Import Failure Handling Tests
# =============================================================================


class TestImportFailureHandling:
    """Tests for handling import failures during daemon startup."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_import_failed_state_exists(self):
        """IMPORT_FAILED state should be defined."""
        assert DaemonState.IMPORT_FAILED.value == "import_failed"

    @pytest.mark.asyncio
    async def test_failed_factory_sets_state(self):
        """Factory that raises should eventually set FAILED state.

        Note: Factory errors happen asynchronously. The start() method returns
        True when the task is created, but the factory error occurs in the
        background task and updates the state asynchronously.
        """
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def failing_factory():
            raise RuntimeError("Factory failed")

        manager.register_factory(DaemonType.MODEL_SYNC, failing_factory, auto_restart=False)

        # Start the daemon - returns True because task was created
        result = await manager.start(DaemonType.MODEL_SYNC)
        assert result is True  # Task creation succeeds

        # Wait for async factory to fail
        await asyncio.sleep(0.1)

        # Now check that the state reflects the failure
        info = manager._daemons[DaemonType.MODEL_SYNC]
        assert info.state == DaemonState.FAILED
        assert "Factory failed" in info.last_error

        await manager.shutdown()


# =============================================================================
# Get Dependents Tests
# =============================================================================


class TestGetDependents:
    """Tests for getting dependent daemons."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_get_dependents_direct(self):
        """_get_dependents should return direct dependents."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )
        manager.register_factory(
            DaemonType.FEEDBACK_LOOP,
            factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        dependents = manager._get_dependents(DaemonType.EVENT_ROUTER)

        assert DaemonType.DATA_PIPELINE in dependents
        assert DaemonType.FEEDBACK_LOOP in dependents

    def test_get_dependents_none(self):
        """_get_dependents should return empty for no dependents."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        async def factory():
            pass

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        manager.register_factory(DaemonType.MODEL_SYNC, factory)

        dependents = manager._get_dependents(DaemonType.EVENT_ROUTER)

        # MODEL_SYNC doesn't depend on EVENT_ROUTER
        assert DaemonType.MODEL_SYNC not in dependents


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestShutdown:
    """Tests for daemon shutdown behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_health_loop(self):
        """shutdown() should cancel health monitoring loop."""
        manager = DaemonManager()

        await manager._ensure_health_loop_running()
        assert manager._health_task is not None

        await manager.shutdown()

        assert manager._health_task.done() or manager._health_task.cancelled()

    @pytest.mark.asyncio
    async def test_shutdown_sets_not_running(self):
        """shutdown() should set _running to False."""
        manager = DaemonManager()

        await manager._ensure_health_loop_running()
        assert manager._running is True

        await manager.shutdown()

        assert manager._running is False

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_daemons(self):
        """shutdown() should stop all running daemons."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        stopped = []

        async def factory():
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                stopped.append(True)
                raise

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        manager.register_factory(DaemonType.MODEL_SYNC, factory)

        await manager.start(DaemonType.EVENT_ROUTER)
        await manager.start(DaemonType.MODEL_SYNC)

        await manager.shutdown()

        # Both daemons should have been stopped
        for daemon_type in [DaemonType.EVENT_ROUTER, DaemonType.MODEL_SYNC]:
            info = manager._daemons[daemon_type]
            assert info.state == DaemonState.STOPPED


# =============================================================================
# Validate Critical Subsystems Tests
# =============================================================================


class TestValidateCriticalSubsystems:
    """Tests for critical subsystem validation."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_validate_returns_list(self):
        """_validate_critical_subsystems should return a list."""
        manager = DaemonManager()
        errors = manager._validate_critical_subsystems()
        assert isinstance(errors, list)

    def test_validate_handles_missing_imports(self):
        """_validate_critical_subsystems should handle import failures."""
        manager = DaemonManager()
        # Should not raise even if subsystems fail to import
        errors = manager._validate_critical_subsystems()
        assert errors is not None
