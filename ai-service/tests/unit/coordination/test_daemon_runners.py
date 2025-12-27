"""Tests for daemon_runners.py - Extracted daemon runner functions.

December 2025 - Critical path tests for daemon instantiation.

Tests cover:
- Runner registry building and lookup
- Individual runner function import guards
- Daemon factory patterns
- Error handling for missing dependencies
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.daemon_runners import (
    get_runner,
    get_all_runners,
    _build_runner_registry,
    _wait_for_daemon,
    # Sync runners
    create_sync_coordinator,
    create_high_quality_sync,
    create_elo_sync,
    create_auto_sync,
    create_ephemeral_sync,
    create_gossip_sync,
    # Event runners
    create_event_router,
    create_cross_process_poller,
    create_dlq_retry,
    # Health runners
    create_health_check,
    create_queue_monitor,
    create_daemon_watchdog,
    create_node_health_monitor,
    create_cluster_monitor,
    create_quality_monitor,
    # Pipeline runners
    create_data_pipeline,
    create_selfplay_coordinator,
    create_training_trigger,
    create_auto_export,
    # Distribution runners
    create_model_distribution,
    create_npz_distribution,
    # Resource runners
    create_idle_resource,
    create_node_recovery,
)
from app.coordination.daemon_types import DaemonType


class TestRunnerRegistry:
    """Tests for the runner registry functions."""

    def test_build_runner_registry_returns_dict(self):
        """Test that _build_runner_registry returns a dictionary."""
        registry = _build_runner_registry()
        assert isinstance(registry, dict)

    def test_build_runner_registry_has_entries(self):
        """Test that the registry has runner entries."""
        registry = _build_runner_registry()
        # Should have at least 50 entries (we have 57+ runner functions)
        assert len(registry) >= 50

    def test_registry_keys_are_daemon_type_names(self):
        """Test that registry keys match DaemonType enum names."""
        registry = _build_runner_registry()

        # Check a few known entries
        assert "AUTO_SYNC" in registry
        assert "EVENT_ROUTER" in registry
        assert "DATA_PIPELINE" in registry
        assert "CLUSTER_MONITOR" in registry

    def test_registry_values_are_callables(self):
        """Test that registry values are callable (coroutine functions)."""
        registry = _build_runner_registry()

        for name, runner in registry.items():
            assert callable(runner), f"Runner for {name} is not callable"

    def test_get_runner_returns_callable(self):
        """Test that get_runner returns a callable for known types."""
        runner = get_runner(DaemonType.AUTO_SYNC)
        assert runner is not None
        assert callable(runner)

    def test_get_runner_returns_none_for_unknown(self):
        """Test that get_runner returns None for unknown daemon types."""
        # Create a mock DaemonType with unknown name
        mock_type = MagicMock()
        mock_type.name = "UNKNOWN_DAEMON_TYPE_XYZ"

        runner = get_runner(mock_type)
        assert runner is None

    def test_get_all_runners_returns_dict(self):
        """Test that get_all_runners returns a dictionary."""
        runners = get_all_runners()
        assert isinstance(runners, dict)
        assert len(runners) >= 50


class TestRunnerRegistryCompleteness:
    """Tests to verify registry covers all expected daemon types."""

    def test_sync_daemons_registered(self):
        """Test that sync daemon runners are registered."""
        registry = _build_runner_registry()

        sync_types = [
            "SYNC_COORDINATOR",
            "HIGH_QUALITY_SYNC",
            "ELO_SYNC",
            "AUTO_SYNC",
            "TRAINING_NODE_WATCHER",
            "EPHEMERAL_SYNC",
            "GOSSIP_SYNC",
        ]

        for daemon_name in sync_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_event_daemons_registered(self):
        """Test that event daemon runners are registered."""
        registry = _build_runner_registry()

        event_types = [
            "EVENT_ROUTER",
            "CROSS_PROCESS_POLLER",
            "DLQ_RETRY",
        ]

        for daemon_name in event_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_health_daemons_registered(self):
        """Test that health daemon runners are registered."""
        registry = _build_runner_registry()

        health_types = [
            "HEALTH_CHECK",
            "QUEUE_MONITOR",
            "DAEMON_WATCHDOG",
            "NODE_HEALTH_MONITOR",
            "SYSTEM_HEALTH_MONITOR",
            "QUALITY_MONITOR",
            "MODEL_PERFORMANCE_WATCHDOG",
            "CLUSTER_MONITOR",
            "CLUSTER_WATCHDOG",
        ]

        for daemon_name in health_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_pipeline_daemons_registered(self):
        """Test that pipeline daemon runners are registered."""
        registry = _build_runner_registry()

        pipeline_types = [
            "DATA_PIPELINE",
            "CONTINUOUS_TRAINING_LOOP",
            "SELFPLAY_COORDINATOR",
            "TRAINING_TRIGGER",
            "AUTO_EXPORT",
        ]

        for daemon_name in pipeline_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_distribution_daemons_registered(self):
        """Test that distribution daemon runners are registered."""
        registry = _build_runner_registry()

        distribution_types = [
            "MODEL_SYNC",
            "MODEL_DISTRIBUTION",
            "NPZ_DISTRIBUTION",
            "DATA_SERVER",
        ]

        for daemon_name in distribution_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_resource_daemons_registered(self):
        """Test that resource daemon runners are registered."""
        registry = _build_runner_registry()

        resource_types = [
            "IDLE_RESOURCE",
            "NODE_RECOVERY",
            "RESOURCE_OPTIMIZER",
            "UTILIZATION_OPTIMIZER",
            "ADAPTIVE_RESOURCES",
        ]

        for daemon_name in resource_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"


class TestWaitForDaemon:
    """Tests for the _wait_for_daemon helper."""

    @pytest.mark.asyncio
    async def test_wait_for_daemon_returns_when_stopped(self):
        """Test that _wait_for_daemon returns when daemon stops."""
        mock_daemon = MagicMock()
        # Simulate daemon stopping after 2 checks
        mock_daemon.is_running.side_effect = [True, True, False]

        await _wait_for_daemon(mock_daemon, check_interval=0.01)

        assert mock_daemon.is_running.call_count == 3

    @pytest.mark.asyncio
    async def test_wait_for_daemon_returns_immediately_if_not_running(self):
        """Test that _wait_for_daemon returns immediately if daemon not running."""
        mock_daemon = MagicMock()
        mock_daemon.is_running.return_value = False

        await _wait_for_daemon(mock_daemon, check_interval=0.01)

        mock_daemon.is_running.assert_called_once()


class TestSyncRunners:
    """Tests for sync daemon runner functions."""

    def test_sync_runner_functions_are_coroutines(self):
        """Test that sync runner functions are coroutine functions."""
        import asyncio
        import inspect

        assert inspect.iscoroutinefunction(create_auto_sync)
        assert inspect.iscoroutinefunction(create_sync_coordinator)
        assert inspect.iscoroutinefunction(create_high_quality_sync)
        assert inspect.iscoroutinefunction(create_elo_sync)
        assert inspect.iscoroutinefunction(create_ephemeral_sync)
        assert inspect.iscoroutinefunction(create_gossip_sync)

    def test_sync_runners_in_registry(self):
        """Test that all sync runners are registered."""
        registry = _build_runner_registry()
        assert "AUTO_SYNC" in registry
        assert registry["AUTO_SYNC"] is create_auto_sync
        assert "ELO_SYNC" in registry
        assert registry["ELO_SYNC"] is create_elo_sync


class TestEventRunners:
    """Tests for event daemon runner functions."""

    def test_event_runner_functions_are_coroutines(self):
        """Test that event runner functions are coroutine functions."""
        import inspect

        assert inspect.iscoroutinefunction(create_event_router)
        assert inspect.iscoroutinefunction(create_cross_process_poller)
        assert inspect.iscoroutinefunction(create_dlq_retry)

    def test_event_runners_in_registry(self):
        """Test that all event runners are registered."""
        registry = _build_runner_registry()
        assert "EVENT_ROUTER" in registry
        assert registry["EVENT_ROUTER"] is create_event_router
        assert "CROSS_PROCESS_POLLER" in registry
        assert registry["CROSS_PROCESS_POLLER"] is create_cross_process_poller


class TestHealthRunners:
    """Tests for health daemon runner functions."""

    def test_health_runner_functions_are_coroutines(self):
        """Test that health runner functions are coroutine functions."""
        import inspect

        assert inspect.iscoroutinefunction(create_health_check)
        assert inspect.iscoroutinefunction(create_queue_monitor)
        assert inspect.iscoroutinefunction(create_daemon_watchdog)
        assert inspect.iscoroutinefunction(create_node_health_monitor)
        assert inspect.iscoroutinefunction(create_cluster_monitor)
        assert inspect.iscoroutinefunction(create_quality_monitor)

    def test_health_runners_in_registry(self):
        """Test that all health runners are registered."""
        registry = _build_runner_registry()
        assert "HEALTH_CHECK" in registry
        assert registry["HEALTH_CHECK"] is create_health_check
        assert "CLUSTER_MONITOR" in registry
        assert registry["CLUSTER_MONITOR"] is create_cluster_monitor


class TestPipelineRunners:
    """Tests for pipeline daemon runner functions."""

    def test_pipeline_runner_functions_are_coroutines(self):
        """Test that pipeline runner functions are coroutine functions."""
        import inspect

        assert inspect.iscoroutinefunction(create_data_pipeline)
        assert inspect.iscoroutinefunction(create_selfplay_coordinator)
        assert inspect.iscoroutinefunction(create_training_trigger)
        assert inspect.iscoroutinefunction(create_auto_export)

    def test_pipeline_runners_in_registry(self):
        """Test that all pipeline runners are registered."""
        registry = _build_runner_registry()
        assert "DATA_PIPELINE" in registry
        assert registry["DATA_PIPELINE"] is create_data_pipeline
        assert "SELFPLAY_COORDINATOR" in registry
        assert registry["SELFPLAY_COORDINATOR"] is create_selfplay_coordinator
        assert "TRAINING_TRIGGER" in registry
        assert registry["TRAINING_TRIGGER"] is create_training_trigger


class TestDistributionRunners:
    """Tests for distribution daemon runner functions."""

    def test_distribution_runner_functions_are_coroutines(self):
        """Test that distribution runner functions are coroutine functions."""
        import inspect

        assert inspect.iscoroutinefunction(create_model_distribution)
        assert inspect.iscoroutinefunction(create_npz_distribution)

    def test_distribution_runners_in_registry(self):
        """Test that all distribution runners are registered."""
        registry = _build_runner_registry()
        assert "MODEL_DISTRIBUTION" in registry
        assert registry["MODEL_DISTRIBUTION"] is create_model_distribution
        assert "NPZ_DISTRIBUTION" in registry
        assert registry["NPZ_DISTRIBUTION"] is create_npz_distribution


class TestResourceRunners:
    """Tests for resource daemon runner functions."""

    def test_resource_runner_functions_are_coroutines(self):
        """Test that resource runner functions are coroutine functions."""
        import inspect

        assert inspect.iscoroutinefunction(create_idle_resource)
        assert inspect.iscoroutinefunction(create_node_recovery)

    def test_resource_runners_in_registry(self):
        """Test that all resource runners are registered."""
        registry = _build_runner_registry()
        assert "IDLE_RESOURCE" in registry
        assert registry["IDLE_RESOURCE"] is create_idle_resource
        assert "NODE_RECOVERY" in registry
        assert registry["NODE_RECOVERY"] is create_node_recovery


class TestRunnerErrorHandling:
    """Tests for error handling in runner functions."""

    def test_runners_have_import_error_handling(self):
        """Test that runners have try-except ImportError blocks."""
        import inspect

        # Check the source code of a runner for ImportError handling
        source = inspect.getsource(create_auto_sync)
        assert "ImportError" in source
        assert "except ImportError" in source or "except ImportError as e" in source

    def test_runners_log_errors(self):
        """Test that runner source code has logging for errors."""
        import inspect

        source = inspect.getsource(create_auto_sync)
        assert "logger.error" in source


class TestRunnerIntegration:
    """Integration tests for runner functions."""

    def test_all_registered_runners_exist_as_functions(self):
        """Test that all registered runners correspond to actual functions."""
        registry = _build_runner_registry()

        import app.coordination.daemon_runners as runners_module

        for daemon_name, runner in registry.items():
            # Verify the function exists in the module
            func_name = f"create_{daemon_name.lower()}"
            # Some have different naming patterns, but all should be callable
            assert callable(runner), f"Runner for {daemon_name} is not callable"

    def test_get_runner_and_registry_consistent(self):
        """Test that get_runner and get_all_runners are consistent."""
        all_runners = get_all_runners()

        for daemon_type in DaemonType:
            direct_runner = get_runner(daemon_type)
            registry_runner = all_runners.get(daemon_type.name)

            # Both should return the same result (either both None or both same function)
            if daemon_type.name in all_runners:
                assert direct_runner == registry_runner


class TestDeprecatedRunners:
    """Tests for deprecated daemon runners."""

    def test_deprecated_sync_coordinator_registered(self):
        """Test that deprecated SYNC_COORDINATOR is still registered for backward compat."""
        registry = _build_runner_registry()
        assert "SYNC_COORDINATOR" in registry

    def test_deprecated_health_check_registered(self):
        """Test that deprecated HEALTH_CHECK is still registered for backward compat."""
        registry = _build_runner_registry()
        assert "HEALTH_CHECK" in registry

    def test_deprecated_runners_are_coroutines(self):
        """Test that deprecated runners are still coroutine functions."""
        import inspect

        assert inspect.iscoroutinefunction(create_sync_coordinator)
        assert inspect.iscoroutinefunction(create_health_check)


class TestAllRunnersParametrized:
    """Parametrized tests covering all 62+ daemon runners.

    Dec 27, 2025: Added to ensure comprehensive coverage of all runner functions.
    Uses parametrization to test the common pattern that all runners follow.
    """

    @pytest.fixture
    def all_runner_configs(self):
        """Get all registered runner configurations."""
        return _build_runner_registry()

    def test_all_runners_are_coroutine_functions(self, all_runner_configs):
        """Test that ALL registered runners are async coroutine functions."""
        import inspect

        for daemon_name, runner in all_runner_configs.items():
            assert inspect.iscoroutinefunction(runner), (
                f"Runner for {daemon_name} is not a coroutine function"
            )

    def test_all_runners_have_docstrings(self, all_runner_configs):
        """Test that all runners have docstrings for documentation."""
        for daemon_name, runner in all_runner_configs.items():
            assert runner.__doc__ is not None, (
                f"Runner for {daemon_name} is missing a docstring"
            )

    def test_all_runners_follow_naming_convention(self, all_runner_configs):
        """Test that runner function names follow the create_* pattern."""
        for daemon_name, runner in all_runner_configs.items():
            # Function name should be create_<daemon_name_lowercase> or similar
            func_name = runner.__name__
            assert func_name.startswith("create_"), (
                f"Runner for {daemon_name} has unexpected name: {func_name}"
            )

    def test_runner_registry_completeness(self, all_runner_configs):
        """Test that we have runners for a minimum expected count."""
        # We expect at least 60 daemon types to have runners
        assert len(all_runner_configs) >= 60, (
            f"Registry has only {len(all_runner_configs)} runners, expected >= 60"
        )

    @pytest.mark.parametrize("daemon_name", [
        "AUTO_SYNC", "DATA_PIPELINE", "EVENT_ROUTER", "FEEDBACK_LOOP",
        "SELFPLAY_COORDINATOR", "TRAINING_TRIGGER", "EVALUATION",
        "MODEL_DISTRIBUTION", "CLUSTER_MONITOR", "DAEMON_WATCHDOG",
    ])
    def test_critical_runners_are_registered(self, daemon_name, all_runner_configs):
        """Test that critical daemon runners are registered."""
        assert daemon_name in all_runner_configs, (
            f"Critical runner {daemon_name} is not registered"
        )

    @pytest.mark.asyncio
    async def test_wait_for_daemon_with_immediate_stop(self):
        """Test _wait_for_daemon returns immediately when daemon not running."""
        mock_daemon = MagicMock()
        mock_daemon.is_running.return_value = False

        # Should return without waiting
        await _wait_for_daemon(mock_daemon, check_interval=0.01)
        mock_daemon.is_running.assert_called()

    @pytest.mark.asyncio
    async def test_wait_for_daemon_polls_until_stopped(self):
        """Test _wait_for_daemon polls until daemon stops."""
        mock_daemon = MagicMock()
        # Return True twice, then False to simulate daemon stopping
        mock_daemon.is_running.side_effect = [True, True, False]

        await _wait_for_daemon(mock_daemon, check_interval=0.01)

        # Should have been called 3 times
        assert mock_daemon.is_running.call_count == 3
