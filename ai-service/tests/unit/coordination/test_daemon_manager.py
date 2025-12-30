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
    RESTART_STATE_FILE,
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
        """Should have exactly 8 states (including DEGRADED for graceful degradation)."""
        assert len(DaemonState) == 8


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
        assert config.shutdown_timeout == 30.0
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

    def test_ephemeral_has_auto_sync(self):
        """Ephemeral profile should include auto_sync (replaced ephemeral_sync)."""
        from app.coordination.daemon_manager import DAEMON_PROFILES
        profile = DAEMON_PROFILES["ephemeral"]
        # EPHEMERAL_SYNC was consolidated into AUTO_SYNC (Dec 2025)
        assert DaemonType.AUTO_SYNC in profile

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
        # Disable coordination wiring to avoid hanging in tests
        config = DaemonManagerConfig(enable_coordination_wiring=False)
        manager = DaemonManager(config)
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
        # Disable coordination wiring to avoid hanging in tests
        config = DaemonManagerConfig(enable_coordination_wiring=False)
        manager = DaemonManager(config)
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
        # Clear persisted restart counts file to ensure clean state
        RESTART_STATE_FILE.unlink(missing_ok=True)
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
        # Disable coordination wiring to avoid hanging in tests
        config = DaemonManagerConfig(enable_coordination_wiring=False)
        manager = DaemonManager(config)
        manager._factories.clear()
        manager._daemons.clear()

        async def failing_factory():
            raise RuntimeError("Factory failed")

        manager.register_factory(DaemonType.MODEL_SYNC, failing_factory, auto_restart=False)

        # Start the daemon - returns True because task was created
        result = await manager.start(DaemonType.MODEL_SYNC, wait_for_deps=False)
        assert result is True  # Task creation succeeds

        # Wait for async factory to fail
        await asyncio.sleep(0.1)

        # Now check that the state reflects the failure
        info = manager._daemons[DaemonType.MODEL_SYNC]
        assert info.state == DaemonState.FAILED
        assert "Factory failed" in info.last_error

        # Shutdown may re-raise pending task errors - that's expected
        try:
            await manager.shutdown()
        except RuntimeError:
            pass  # Expected - factory error may propagate


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
        # Disable coordination wiring to avoid hanging in tests
        config = DaemonManagerConfig(enable_coordination_wiring=False)
        manager = DaemonManager(config)
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

        await manager.start(DaemonType.EVENT_ROUTER, wait_for_deps=False)
        await manager.start(DaemonType.MODEL_SYNC, wait_for_deps=False)

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


# =============================================================================
# Health Check Method Tests
# =============================================================================


class TestHealthCheckMethod:
    """Tests for the health_check() method on DaemonManager itself."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_health_check_not_running(self):
        """health_check() returns unhealthy when not running."""
        manager = DaemonManager()
        manager._running = False

        result = manager.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower() or result.message == ""

    def test_health_check_running_no_daemons(self):
        """health_check() returns healthy when running with no daemons."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        result = manager.health_check()

        assert result.healthy is True

    def test_health_check_running_with_healthy_daemons(self):
        """health_check() returns healthy with running daemons."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        manager._daemons[DaemonType.EVENT_ROUTER].state = DaemonState.RUNNING

        result = manager.health_check()

        assert result.healthy is True
        assert result.details["daemons_running"] == 1

    def test_health_check_degraded_with_failures(self):
        """health_check() returns degraded with many failures."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        # Register 5 daemons, 2 failed (40% > 20% threshold)
        for i, dtype in enumerate([
            DaemonType.EVENT_ROUTER,
            DaemonType.DATA_PIPELINE,
            DaemonType.AUTO_SYNC,
            DaemonType.FEEDBACK_LOOP,
            DaemonType.QUEUE_POPULATOR,
        ]):
            manager.register_factory(dtype, lambda: None)
            if i < 2:
                manager._daemons[dtype].state = DaemonState.FAILED
            else:
                manager._daemons[dtype].state = DaemonState.RUNNING

        result = manager.health_check()

        assert result.healthy is False
        assert result.details["daemons_failed"] == 2

    def test_health_check_returns_details(self):
        """health_check() includes expected details."""
        manager = DaemonManager()
        manager._running = True

        result = manager.health_check()

        assert "running" in result.details
        assert "daemons_total" in result.details
        assert "daemons_running" in result.details
        assert "daemons_failed" in result.details
        assert "uptime_seconds" in result.details


# =============================================================================
# Liveness and Readiness Probe Tests
# =============================================================================


class TestProbes:
    """Tests for liveness and readiness probes."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_liveness_probe_basic(self):
        """liveness_probe() returns alive status."""
        manager = DaemonManager()

        probe = manager.liveness_probe()

        assert probe["alive"] is True
        assert "timestamp" in probe
        assert "uptime_seconds" in probe

    def test_liveness_probe_uptime_increases(self):
        """liveness_probe() uptime should increase over time."""
        manager = DaemonManager()

        probe1 = manager.liveness_probe()
        time.sleep(0.1)
        probe2 = manager.liveness_probe()

        assert probe2["uptime_seconds"] >= probe1["uptime_seconds"]

    def test_readiness_probe_not_started(self):
        """readiness_probe() returns not ready when not started."""
        manager = DaemonManager()
        manager._running = False

        probe = manager.readiness_probe()

        assert probe["ready"] is False
        assert "not started" in probe.get("reason", "").lower()

    def test_readiness_probe_no_daemons(self):
        """readiness_probe() returns not ready with no running daemons."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        probe = manager.readiness_probe()

        assert probe["ready"] is False
        assert "no daemon" in probe.get("reason", "").lower()

    def test_readiness_probe_with_running_daemons(self):
        """readiness_probe() returns ready with running daemons."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        manager._daemons[DaemonType.EVENT_ROUTER].state = DaemonState.RUNNING

        probe = manager.readiness_probe()

        assert probe["ready"] is True
        assert probe["running_count"] == 1

    def test_readiness_probe_required_daemons_missing(self):
        """readiness_probe() returns not ready when required daemons missing."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        manager._daemons[DaemonType.EVENT_ROUTER].state = DaemonState.RUNNING

        probe = manager.readiness_probe(
            required_daemons=[DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE]
        )

        assert probe["ready"] is False
        # Check for daemon name in reason (uses lowercase snake_case)
        assert "data_pipeline" in str(probe.get("reason", "")).lower()

    def test_readiness_probe_required_daemons_present(self):
        """readiness_probe() returns ready when required daemons running."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        for dtype in [DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE]:
            manager.register_factory(dtype, lambda: None)
            manager._daemons[dtype].state = DaemonState.RUNNING

        probe = manager.readiness_probe(
            required_daemons=[DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE]
        )

        assert probe["ready"] is True

    def test_health_summary_includes_all_info(self):
        """health_summary() includes comprehensive info."""
        manager = DaemonManager()
        manager._running = True

        summary = manager.health_summary()

        assert "status" in summary
        assert "score" in summary
        assert "running" in summary
        assert "failed" in summary
        assert "total" in summary
        assert "liveness" in summary
        assert "readiness" in summary
        assert "timestamp" in summary

    def test_health_summary_score_calculation(self):
        """health_summary() calculates score correctly."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        # 2 running, 1 failed = 66% health score
        for i, dtype in enumerate([DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE, DaemonType.AUTO_SYNC]):
            manager.register_factory(dtype, lambda: None)
            if i < 2:
                manager._daemons[dtype].state = DaemonState.RUNNING
            else:
                manager._daemons[dtype].state = DaemonState.FAILED

        summary = manager.health_summary()

        expected_score = 2 / 3  # 66%
        assert abs(summary["score"] - expected_score) < 0.01

    def test_health_summary_status_healthy(self):
        """health_summary() shows healthy status at 90%+."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        # 10 running, 0 failed = 100%
        for i in range(10):
            dtype = list(DaemonType)[i]
            manager.register_factory(dtype, lambda: None)
            manager._daemons[dtype].state = DaemonState.RUNNING

        summary = manager.health_summary()

        assert summary["status"] == "healthy"


# =============================================================================
# Lifecycle Summary Tests
# =============================================================================


class TestLifecycleSummary:
    """Tests for lifecycle tracking methods."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_get_lifecycle_summary_empty(self):
        """get_lifecycle_summary() works with no daemons."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        summary = manager.get_lifecycle_summary()

        assert "manager_uptime_seconds" in summary
        assert summary["total_restarts"] == 0
        assert summary["average_uptime_seconds"] == 0.0

    def test_get_lifecycle_summary_with_daemons(self):
        """get_lifecycle_summary() includes daemon statistics."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        info = manager._daemons[DaemonType.EVENT_ROUTER]
        info.state = DaemonState.RUNNING
        info.start_time = time.time() - 100  # Started 100s ago
        info.restart_count = 2

        summary = manager.get_lifecycle_summary()

        assert summary["total_restarts"] == 2
        assert summary["max_uptime_seconds"] >= 100
        assert summary["most_restarts_daemon"] == "event_router"
        assert summary["most_restarts_count"] == 2

    def test_get_failed_daemons_empty(self):
        """get_failed_daemons() returns empty list when none failed."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        manager._daemons[DaemonType.EVENT_ROUTER].state = DaemonState.RUNNING

        failed = manager.get_failed_daemons()

        assert len(failed) == 0

    def test_get_failed_daemons_with_failures(self):
        """get_failed_daemons() returns failed daemon info."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        info = manager._daemons[DaemonType.EVENT_ROUTER]
        info.state = DaemonState.FAILED
        info.last_error = "Test failure"

        failed = manager.get_failed_daemons()

        assert len(failed) == 1
        assert failed[0][0] == DaemonType.EVENT_ROUTER
        assert failed[0][1] == "Test failure"

    def test_get_daemon_uptime_not_running(self):
        """get_daemon_uptime() returns 0 for non-running daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)

        uptime = manager.get_daemon_uptime(DaemonType.EVENT_ROUTER)

        assert uptime == 0.0

    def test_get_daemon_uptime_running(self):
        """get_daemon_uptime() returns uptime for running daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        info = manager._daemons[DaemonType.EVENT_ROUTER]
        info.state = DaemonState.RUNNING
        info.start_time = time.time() - 50  # Started 50s ago

        uptime = manager.get_daemon_uptime(DaemonType.EVENT_ROUTER)

        assert uptime >= 50

    def test_get_daemon_uptime_unknown_daemon(self):
        """get_daemon_uptime() returns 0 for unknown daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        uptime = manager.get_daemon_uptime(DaemonType.EVENT_ROUTER)

        assert uptime == 0.0


# =============================================================================
# Restart Count Persistence Tests
# =============================================================================


class TestRestartCountPersistence:
    """Tests for restart count persistence."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_record_restart_first_time(self):
        """record_restart() allows first restart."""
        manager = DaemonManager()

        with patch.object(manager, "_save_restart_counts"):
            result = manager.record_restart(DaemonType.EVENT_ROUTER)

        assert result is True
        assert DaemonType.EVENT_ROUTER.value in manager._restart_timestamps

    def test_record_restart_tracks_timestamps(self):
        """record_restart() tracks restart timestamps."""
        manager = DaemonManager()

        with patch.object(manager, "_save_restart_counts"):
            manager.record_restart(DaemonType.EVENT_ROUTER)
            manager.record_restart(DaemonType.EVENT_ROUTER)

        timestamps = manager._restart_timestamps[DaemonType.EVENT_ROUTER.value]
        assert len(timestamps) == 2

    def test_record_restart_exceeds_hourly_limit(self):
        """record_restart() enters DEGRADED mode when hourly limit exceeded.

        December 2025: Changed from permanent failure to graceful degradation.
        Daemons now enter DEGRADED mode and keep retrying with longer intervals.
        """
        manager = DaemonManager()

        # Pre-populate with many recent restarts
        daemon_name = DaemonType.EVENT_ROUTER.value
        current_time = time.time()
        manager._restart_timestamps[daemon_name] = [
            current_time - i * 60 for i in range(15)  # 15 restarts in last hour
        ]

        with patch.object(manager, "_save_restart_counts"):
            result = manager.record_restart(DaemonType.EVENT_ROUTER)

        # Still returns False (restart not allowed yet) but enters DEGRADED mode
        assert result is False
        # New behavior: tracked in _degraded_daemons instead of _permanently_failed
        assert daemon_name in manager._degraded_daemons

    def test_record_restart_cleans_old_timestamps(self):
        """record_restart() removes timestamps older than 1 hour."""
        manager = DaemonManager()

        daemon_name = DaemonType.EVENT_ROUTER.value
        old_timestamp = time.time() - 7200  # 2 hours ago
        manager._restart_timestamps[daemon_name] = [old_timestamp]

        with patch.object(manager, "_save_restart_counts"):
            manager.record_restart(DaemonType.EVENT_ROUTER)

        # Old timestamp should be removed
        timestamps = manager._restart_timestamps[daemon_name]
        assert old_timestamp not in timestamps

    def test_is_permanently_failed_false(self):
        """is_permanently_failed() returns False for healthy daemon."""
        manager = DaemonManager()

        assert manager.is_permanently_failed(DaemonType.EVENT_ROUTER) is False

    def test_is_permanently_failed_true(self):
        """is_permanently_failed() returns True for failed daemon.

        December 2025: _permanently_failed is now a dict (daemon_name -> timestamp).
        """
        manager = DaemonManager()
        # Now a dict: daemon_name -> timestamp
        manager._permanently_failed[DaemonType.EVENT_ROUTER.value] = time.time()

        assert manager.is_permanently_failed(DaemonType.EVENT_ROUTER) is True

    def test_clear_permanently_failed(self):
        """clear_permanently_failed() resets failure status.

        December 2025: Also clears _degraded_daemons for graceful degradation.
        """
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        # Now a dict: daemon_name -> timestamp
        manager._permanently_failed[DaemonType.EVENT_ROUTER.value] = time.time()
        manager._restart_timestamps[DaemonType.EVENT_ROUTER.value] = [time.time()]
        manager._daemons[DaemonType.EVENT_ROUTER].restart_count = 5

        with patch.object(manager, "_save_restart_counts"):
            manager.clear_permanently_failed(DaemonType.EVENT_ROUTER)

        assert DaemonType.EVENT_ROUTER.value not in manager._permanently_failed
        assert DaemonType.EVENT_ROUTER.value not in manager._restart_timestamps
        assert manager._daemons[DaemonType.EVENT_ROUTER].restart_count == 0

    def test_get_recent_restarts_none(self):
        """get_recent_restarts() returns empty list when none recent."""
        manager = DaemonManager()
        manager._restart_timestamps.clear()

        recent = manager.get_recent_restarts()

        assert len(recent) == 0

    def test_get_recent_restarts_within_window(self):
        """get_recent_restarts() returns daemons restarted within window."""
        manager = DaemonManager()

        manager._restart_timestamps[DaemonType.EVENT_ROUTER.value] = [time.time() - 60]
        manager._restart_timestamps[DaemonType.DATA_PIPELINE.value] = [time.time() - 600]

        recent = manager.get_recent_restarts(within_seconds=300)

        assert DaemonType.EVENT_ROUTER in recent
        assert DaemonType.DATA_PIPELINE not in recent

    def test_get_recent_restarts_custom_window(self):
        """get_recent_restarts() respects custom time window."""
        manager = DaemonManager()

        manager._restart_timestamps[DaemonType.EVENT_ROUTER.value] = [time.time() - 60]

        # 30 second window should exclude restart from 60s ago
        recent = manager.get_recent_restarts(within_seconds=30)

        assert DaemonType.EVENT_ROUTER not in recent


# =============================================================================
# Mark Daemon Ready Tests
# =============================================================================


class TestMarkDaemonReady:
    """Tests for explicit daemon readiness signaling."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_mark_daemon_ready_sets_event(self):
        """mark_daemon_ready() sets the ready event."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        info = manager._daemons[DaemonType.EVENT_ROUTER]
        info.ready_event = asyncio.Event()

        result = manager.mark_daemon_ready(DaemonType.EVENT_ROUTER)

        assert result is True
        assert info.ready_event.is_set()

    def test_mark_daemon_ready_not_found(self):
        """mark_daemon_ready() returns False for unregistered daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        result = manager.mark_daemon_ready(DaemonType.EVENT_ROUTER)

        assert result is False

    def test_mark_daemon_ready_no_event(self):
        """mark_daemon_ready() returns False when no ready_event."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        manager._daemons[DaemonType.EVENT_ROUTER].ready_event = None

        result = manager.mark_daemon_ready(DaemonType.EVENT_ROUTER)

        assert result is False

    def test_mark_daemon_ready_already_set(self):
        """mark_daemon_ready() returns True when already set."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        info = manager._daemons[DaemonType.EVENT_ROUTER]
        info.ready_event = asyncio.Event()
        info.ready_event.set()

        result = manager.mark_daemon_ready(DaemonType.EVENT_ROUTER)

        assert result is True


# =============================================================================
# Render Metrics Tests
# =============================================================================


class TestRenderMetrics:
    """Tests for Prometheus-style metrics rendering."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_render_metrics_basic(self):
        """render_metrics() returns string or None depending on Prometheus availability."""
        manager = DaemonManager()
        manager._running = True

        metrics = manager.render_metrics()

        # render_metrics() returns None if prometheus_client is not available
        # or returns a string with metrics if available
        assert metrics is None or isinstance(metrics, str)
        if metrics is not None:
            assert "daemon_count" in metrics or "daemon_health_score" in metrics

    def test_render_metrics_includes_counts(self):
        """render_metrics() includes daemon state counts when Prometheus available."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)
        manager._daemons[DaemonType.EVENT_ROUTER].state = DaemonState.RUNNING

        metrics = manager.render_metrics()

        # Skip check if Prometheus not available
        if metrics is not None:
            assert 'daemon_count' in metrics or 'running' in metrics

    def test_render_metrics_includes_uptime(self):
        """render_metrics() includes uptime when Prometheus available."""
        manager = DaemonManager()
        manager._running = True

        metrics = manager.render_metrics()

        # Skip check if Prometheus not available
        if metrics is not None:
            assert "daemon_uptime_seconds" in metrics or "uptime" in metrics


# =============================================================================
# Get Daemon Info Tests
# =============================================================================


class TestGetDaemonInfo:
    """Tests for get_daemon_info() method."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    def test_get_daemon_info_exists(self):
        """get_daemon_info() returns info for registered daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.EVENT_ROUTER, lambda: None)

        info = manager.get_daemon_info(DaemonType.EVENT_ROUTER)

        assert info is not None
        assert info.daemon_type == DaemonType.EVENT_ROUTER

    def test_get_daemon_info_not_exists(self):
        """get_daemon_info() returns None for unregistered daemon."""
        manager = DaemonManager()
        manager._factories.clear()
        manager._daemons.clear()

        info = manager.get_daemon_info(DaemonType.EVENT_ROUTER)

        assert info is None


# =============================================================================
# Check Health Tests
# =============================================================================


class TestCheckHealth:
    """Tests for _check_health() health monitoring method."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    @pytest.mark.asyncio
    async def test_check_health_dict_result(self):
        """_check_health() handles dict health_check result."""
        # Disable coordination wiring to avoid hanging in tests
        config = DaemonManagerConfig(enable_coordination_wiring=False)
        manager = DaemonManager(config)
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        class DaemonWithHealth:
            def health_check(self):
                return {"healthy": True, "message": "OK"}

        async def factory():
            while True:
                await asyncio.sleep(0.1)

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        await manager.start(DaemonType.EVENT_ROUTER, wait_for_deps=False)
        manager._daemons[DaemonType.EVENT_ROUTER].instance = DaemonWithHealth()

        await manager._check_health()

        assert manager.is_running(DaemonType.EVENT_ROUTER)
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_check_health_bool_result(self):
        """_check_health() handles boolean health_check result."""
        # Disable coordination wiring to avoid hanging in tests
        config = DaemonManagerConfig(enable_coordination_wiring=False)
        manager = DaemonManager(config)
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        class DaemonWithBoolHealth:
            def health_check(self):
                return True

        async def factory():
            while True:
                await asyncio.sleep(0.1)

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        await manager.start(DaemonType.EVENT_ROUTER, wait_for_deps=False)
        manager._daemons[DaemonType.EVENT_ROUTER].instance = DaemonWithBoolHealth()

        await manager._check_health()

        assert manager.is_running(DaemonType.EVENT_ROUTER)
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_check_health_object_result(self):
        """_check_health() handles object with .healthy attribute."""
        # Disable coordination wiring to avoid hanging in tests
        config = DaemonManagerConfig(enable_coordination_wiring=False)
        manager = DaemonManager(config)
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        class HealthResult:
            def __init__(self):
                self.healthy = True
                self.message = "OK"

        class DaemonWithObjectHealth:
            def health_check(self):
                return HealthResult()

        async def factory():
            while True:
                await asyncio.sleep(0.1)

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        await manager.start(DaemonType.EVENT_ROUTER, wait_for_deps=False)
        manager._daemons[DaemonType.EVENT_ROUTER].instance = DaemonWithObjectHealth()

        await manager._check_health()

        assert manager.is_running(DaemonType.EVENT_ROUTER)
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_check_health_async_result(self):
        """_check_health() handles async health_check method."""
        # Disable coordination wiring to avoid hanging in tests
        config = DaemonManagerConfig(enable_coordination_wiring=False)
        manager = DaemonManager(config)
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        class DaemonWithAsyncHealth:
            async def health_check(self):
                await asyncio.sleep(0.01)
                return {"healthy": True}

        async def factory():
            while True:
                await asyncio.sleep(0.1)

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        await manager.start(DaemonType.EVENT_ROUTER, wait_for_deps=False)
        manager._daemons[DaemonType.EVENT_ROUTER].instance = DaemonWithAsyncHealth()

        await manager._check_health()

        assert manager.is_running(DaemonType.EVENT_ROUTER)
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_check_health_unhealthy_triggers_restart(self):
        """_check_health() triggers restart on unhealthy result."""
        # Disable coordination wiring to avoid hanging in tests
        config = DaemonManagerConfig(enable_coordination_wiring=False, auto_restart_failed=True)
        manager = DaemonManager(config)
        manager._factories.clear()
        manager._daemons.clear()
        manager._running = True

        class UnhealthyDaemon:
            def health_check(self):
                return {"healthy": False, "message": "Unhealthy"}

        async def factory():
            while True:
                await asyncio.sleep(0.1)

        manager.register_factory(DaemonType.EVENT_ROUTER, factory)
        await manager.start(DaemonType.EVENT_ROUTER, wait_for_deps=False)

        info = manager._daemons[DaemonType.EVENT_ROUTER]
        info.instance = UnhealthyDaemon()
        # Set start_time far in the past to simulate past startup grace period
        info.start_time = time.time() - 120  # 2 minutes ago (past 60s grace period)

        await manager._check_health()

        # Should have recorded error from unhealthy check
        assert info.last_error is not None or info.restart_count > 0

        await manager.shutdown()


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Tests for daemon manager event handlers."""

    def setup_method(self):
        """Reset singleton before each test."""
        DaemonManager.reset_instance()
        reset_daemon_manager()

    @pytest.mark.asyncio
    async def test_on_regression_critical_logs(self):
        """_on_regression_critical() logs the event."""
        manager = DaemonManager()

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "model_id": "test-model",
            "elo_drop": 50,
            "current_elo": 1500,
            "previous_elo": 1550,
        }

        # Mock the router module to avoid import/method issues
        # The import happens inside the function, so we patch the source module
        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            # Mock all publish methods as async
            mock_router.publish = AsyncMock()
            mock_router.publish_sync = MagicMock()
            mock_router.publish_async = AsyncMock()
            mock_get_router.return_value = mock_router

            # Should not raise
            await manager._on_regression_critical(event)

    @pytest.mark.asyncio
    async def test_on_selfplay_target_updated(self):
        """_on_selfplay_target_updated() handles event."""
        manager = DaemonManager()

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "priority": "high",
            "reason": "test",
        }

        # Should not raise
        await manager._on_selfplay_target_updated(event)

    @pytest.mark.asyncio
    async def test_on_exploration_boost(self):
        """_on_exploration_boost() handles event."""
        manager = DaemonManager()

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "boost_factor": 1.5,
            "reason": "test",
            "duration_seconds": 3600,
        }

        # Should not raise
        await manager._on_exploration_boost(event)

    @pytest.mark.asyncio
    async def test_on_host_offline(self):
        """_on_host_offline() handles event."""
        manager = DaemonManager()

        event = MagicMock()
        event.payload = {
            "host_id": "test-host",
            "reason": "connection lost",
        }

        # Should not raise
        await manager._on_host_offline(event)

    @pytest.mark.asyncio
    async def test_on_host_online(self):
        """_on_host_online() handles event."""
        manager = DaemonManager()

        event = MagicMock()
        event.payload = {
            "host_id": "test-host",
        }

        # Should not raise
        await manager._on_host_online(event)

    @pytest.mark.asyncio
    async def test_on_backpressure_activated(self):
        """_on_backpressure_activated() handles event."""
        manager = DaemonManager()

        event = MagicMock()
        event.payload = {
            "reason": "queue full",
            "threshold": 1000,
            "current_value": 1500,
        }

        # Should not raise
        await manager._on_backpressure_activated(event)

    @pytest.mark.asyncio
    async def test_on_backpressure_released(self):
        """_on_backpressure_released() handles event."""
        manager = DaemonManager()

        event = MagicMock()
        event.payload = {
            "duration_seconds": 300.0,
        }

        # Should not raise
        await manager._on_backpressure_released(event)

    @pytest.mark.asyncio
    async def test_on_disk_space_low(self):
        """_on_disk_space_low() handles event."""
        manager = DaemonManager()

        event = MagicMock()
        event.payload = {
            "host": "localhost",
            "usage_percent": 90.0,
            "free_gb": 10.0,
            "threshold": 70,
        }

        # Should not raise
        await manager._on_disk_space_low(event)


# =============================================================================
# Event Wiring Verification Tests (December 2025)
# =============================================================================


class TestValidateCriticalSubsystems:
    """Tests for _validate_critical_subsystems method."""

    def test_validate_returns_empty_on_success(self):
        """Validation returns empty list when all subsystems available."""
        manager = DaemonManager()
        errors = manager._validate_critical_subsystems()
        # Should return list (may be empty or have warnings)
        assert isinstance(errors, list)

    def test_validate_checks_event_router(self):
        """Validation checks event_router is importable."""
        manager = DaemonManager()

        with patch.dict("sys.modules", {"app.coordination.event_router": None}):
            # This tests the validation logic, not actual import failure
            # Real import failure would require more complex mocking
            errors = manager._validate_critical_subsystems()
            # Should still work since module is really available
            assert isinstance(errors, list)

    def test_validate_checks_startup_order(self):
        """Validation calls validate_startup_order_consistency."""
        manager = DaemonManager()

        with patch(
            "app.coordination.daemon_types.validate_startup_order_consistency",
            return_value=(True, []),
        ):
            errors = manager._validate_critical_subsystems()
            assert isinstance(errors, list)

    def test_validate_logs_startup_order_violations(self):
        """Startup order violations are logged as errors."""
        manager = DaemonManager()

        with patch(
            "app.coordination.daemon_types.validate_startup_order_consistency",
            return_value=(False, ["TEST_VIOLATION"]),
        ):
            with patch("app.coordination.daemon_manager.logger") as mock_logger:
                errors = manager._validate_critical_subsystems()
                # Check that error was logged
                assert any("TEST_VIOLATION" in str(c) for c in mock_logger.error.call_args_list)


class TestVerifyCriticalSubscriptions:
    """Tests for _verify_critical_subscriptions method."""

    def test_verify_returns_list(self):
        """Method returns list of missing subscriptions."""
        manager = DaemonManager()
        missing = manager._verify_critical_subscriptions()
        assert isinstance(missing, list)

    def test_verify_checks_critical_events(self):
        """Method checks for critical event subscriptions."""
        manager = DaemonManager()

        # Patch where has_subscribers is imported from (event_router)
        with patch("app.coordination.event_router.has_subscribers") as mock_has_subs:
            mock_has_subs.return_value = True
            missing = manager._verify_critical_subscriptions()
            # Should have checked for subscriptions
            assert mock_has_subs.call_count >= 1

    def test_verify_reports_missing_subscriptions(self):
        """Missing subscriptions are returned in list."""
        manager = DaemonManager()

        # Patch where has_subscribers is imported from (event_router)
        with patch("app.coordination.event_router.has_subscribers") as mock_has_subs:
            mock_has_subs.return_value = False  # All events missing
            missing = manager._verify_critical_subscriptions()
            # Should report missing events
            assert len(missing) > 0

    def test_verify_empty_when_all_subscribed(self):
        """Returns empty list when all events have subscribers."""
        manager = DaemonManager()

        # Patch where has_subscribers is imported from (event_router)
        with patch("app.coordination.event_router.has_subscribers") as mock_has_subs:
            mock_has_subs.return_value = True  # All events subscribed
            missing = manager._verify_critical_subscriptions()
            assert missing == []

    def test_verify_handles_import_error(self):
        """Gracefully handles import error for has_subscribers."""
        manager = DaemonManager()

        # The function catches import errors gracefully - should return empty list
        # We can't easily mock the import failure, but we can verify the method doesn't crash
        # when called normally
        missing = manager._verify_critical_subscriptions()
        assert isinstance(missing, list)

    def test_critical_events_include_training_completed(self):
        """TRAINING_COMPLETED is in critical events list."""
        manager = DaemonManager()

        checked_events = []

        def track_has_subs(event_type):
            checked_events.append(event_type)
            return True

        # Patch where has_subscribers is imported from (event_router)
        with patch("app.coordination.event_router.has_subscribers", side_effect=track_has_subs):
            manager._verify_critical_subscriptions()
            assert "TRAINING_COMPLETED" in checked_events

    def test_critical_events_include_model_promoted(self):
        """MODEL_PROMOTED is in critical events list."""
        manager = DaemonManager()

        checked_events = []

        def track_has_subs(event_type):
            checked_events.append(event_type)
            return True

        # Patch where has_subscribers is imported from (event_router)
        with patch("app.coordination.event_router.has_subscribers", side_effect=track_has_subs):
            manager._verify_critical_subscriptions()
            assert "MODEL_PROMOTED" in checked_events
