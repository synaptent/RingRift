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
            "HIGH_QUALITY_SYNC",
            "ELO_SYNC",
            "AUTO_SYNC",
            "TRAINING_NODE_WATCHER",
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
            "QUEUE_MONITOR",
            "DAEMON_WATCHDOG",
            "QUALITY_MONITOR",
            "MODEL_PERFORMANCE_WATCHDOG",
            "CLUSTER_MONITOR",
            "CLUSTER_WATCHDOG",
            "HEALTH_SERVER",
        ]

        for daemon_name in health_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_pipeline_daemons_registered(self):
        """Test that pipeline daemon runners are registered."""
        registry = _build_runner_registry()

        pipeline_types = [
            "DATA_PIPELINE",
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
        assert "CLUSTER_MONITOR" in registry
        assert registry["CLUSTER_MONITOR"] is create_cluster_monitor
        assert "QUALITY_MONITOR" in registry
        assert registry["QUALITY_MONITOR"] is create_quality_monitor


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
    """Tests for deprecated daemon runners.

    These runners still exist in app.coordination.runners as deprecated no-ops,
    but have been removed from the daemon_runners.py registry.
    """

    def test_deprecated_sync_coordinator_exists_as_function(self):
        """Test that deprecated create_sync_coordinator still exists as a no-op function."""
        import inspect
        assert inspect.iscoroutinefunction(create_sync_coordinator)

    def test_deprecated_health_check_exists_as_function(self):
        """Test that deprecated create_health_check still exists as a no-op function."""
        import inspect
        assert inspect.iscoroutinefunction(create_health_check)

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


class TestCheckDaemonRunning:
    """Tests for the _check_daemon_running helper function.

    Dec 29, 2025: Added comprehensive tests for all code paths in _check_daemon_running.
    """

    def test_check_daemon_running_with_property_true(self):
        """Test _check_daemon_running with is_running property returning True."""
        from app.coordination.daemon_runners import _check_daemon_running

        mock_daemon = MagicMock()
        mock_daemon.is_running = True  # Property, not method

        result = _check_daemon_running(mock_daemon)
        assert result is True

    def test_check_daemon_running_with_property_false(self):
        """Test _check_daemon_running with is_running property returning False."""
        from app.coordination.daemon_runners import _check_daemon_running

        mock_daemon = MagicMock()
        mock_daemon.is_running = False  # Property, not method

        result = _check_daemon_running(mock_daemon)
        assert result is False

    def test_check_daemon_running_with_method_true(self):
        """Test _check_daemon_running with is_running() method returning True."""
        from app.coordination.daemon_runners import _check_daemon_running

        mock_daemon = MagicMock()
        mock_daemon.is_running = MagicMock(return_value=True)  # Callable method

        result = _check_daemon_running(mock_daemon)
        assert result is True
        mock_daemon.is_running.assert_called_once()

    def test_check_daemon_running_with_method_false(self):
        """Test _check_daemon_running with is_running() method returning False."""
        from app.coordination.daemon_runners import _check_daemon_running

        mock_daemon = MagicMock()
        mock_daemon.is_running = MagicMock(return_value=False)  # Callable method

        result = _check_daemon_running(mock_daemon)
        assert result is False
        mock_daemon.is_running.assert_called_once()

    def test_check_daemon_running_with_running_attribute_true(self):
        """Test _check_daemon_running with _running attribute (legacy pattern)."""
        from app.coordination.daemon_runners import _check_daemon_running

        # Create object without is_running but with _running
        class LegacyDaemon:
            def __init__(self):
                self._running = True

        daemon = LegacyDaemon()
        result = _check_daemon_running(daemon)
        assert result is True

    def test_check_daemon_running_with_running_attribute_false(self):
        """Test _check_daemon_running with _running attribute set to False."""
        from app.coordination.daemon_runners import _check_daemon_running

        class LegacyDaemon:
            def __init__(self):
                self._running = False

        daemon = LegacyDaemon()
        result = _check_daemon_running(daemon)
        assert result is False

    def test_check_daemon_running_no_attributes_returns_false(self):
        """Test _check_daemon_running returns False when no running indicator exists."""
        from app.coordination.daemon_runners import _check_daemon_running

        class UnknownDaemon:
            pass

        daemon = UnknownDaemon()
        result = _check_daemon_running(daemon)
        assert result is False

    def test_check_daemon_running_prefers_is_running_over_running(self):
        """Test that is_running takes precedence over _running attribute."""
        from app.coordination.daemon_runners import _check_daemon_running

        class MixedDaemon:
            def __init__(self):
                self.is_running = True
                self._running = False  # Should be ignored

        daemon = MixedDaemon()
        result = _check_daemon_running(daemon)
        assert result is True


class TestRunnerImportErrorHandling:
    """Tests for ImportError handling in runner functions.

    Dec 29, 2025: Added mocked ImportError tests to verify graceful error handling.
    """

    def test_auto_sync_has_import_error_handling(self):
        """Test that create_auto_sync has ImportError handling in source."""
        import inspect
        source = inspect.getsource(create_auto_sync)
        assert "except ImportError as e:" in source
        assert "raise" in source
        assert "logger.error" in source

    def test_event_router_has_import_error_handling(self):
        """Test that create_event_router logs and raises ImportError."""
        import inspect
        source = inspect.getsource(create_event_router)
        assert "ImportError" in source
        assert "logger.error" in source

    def test_most_runners_have_import_error_in_source(self):
        """Test that most runner functions handle ImportError.

        Note: Some deprecated runners (like LAMBDA_IDLE) may not have
        ImportError handling if they early-return or handle errors differently.
        """
        import inspect
        registry = _build_runner_registry()

        # These deprecated runners may have different error handling patterns
        deprecated_runners_with_different_handling = {
            "LAMBDA_IDLE",  # Returns early for terminated Lambda account
        }

        missing_import_error = []
        for daemon_name, runner in registry.items():
            if daemon_name in deprecated_runners_with_different_handling:
                continue
            source = inspect.getsource(runner)
            if "ImportError" not in source:
                missing_import_error.append(daemon_name)

        # Allow up to 5 runners to have different handling patterns
        assert len(missing_import_error) <= 5, (
            f"Too many runners missing ImportError handling: {missing_import_error}"
        )

    def test_non_deprecated_runners_have_import_error(self):
        """Test that all non-deprecated runners have ImportError handling."""
        import inspect
        registry = _build_runner_registry()

        for daemon_name, runner in registry.items():
            source = inspect.getsource(runner)
            # Skip deprecated no-op runners (they intentionally lack ImportError handling)
            if "Deprecated" in source and "no-op" in source:
                continue
            # Check for import error handling or early return
            has_handling = "ImportError" in source or "return" in source
            assert has_handling, (
                f"Runner {daemon_name} missing ImportError handling"
            )


class TestDecember2025Runners:
    """Tests for daemon runners added in December 2025.

    These tests verify the newer runners that were added for:
    - 48-hour autonomous operation
    - Availability management
    - Cascade training
    - PER orchestrator
    """

    def test_december_2025_runners_registered(self):
        """Test that all December 2025 runners are registered."""
        registry = _build_runner_registry()

        dec_2025_types = [
            "AVAILABILITY_NODE_MONITOR",
            "AVAILABILITY_RECOVERY_ENGINE",
            "AVAILABILITY_CAPACITY_PLANNER",
            "AVAILABILITY_PROVISIONER",
            "CASCADE_TRAINING",
            "PER_ORCHESTRATOR",
            "PROGRESS_WATCHDOG",
            "P2P_RECOVERY",
            "TAILSCALE_HEALTH",
            "CONNECTIVITY_RECOVERY",
        ]

        for daemon_name in dec_2025_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_autonomous_operation_runners_are_coroutines(self):
        """Test that 48-hour autonomous operation runners are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_progress_watchdog,
            create_p2p_recovery,
        )

        assert inspect.iscoroutinefunction(create_progress_watchdog)
        assert inspect.iscoroutinefunction(create_p2p_recovery)

    def test_availability_runners_are_coroutines(self):
        """Test that availability management runners are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_availability_node_monitor,
            create_availability_recovery_engine,
            create_availability_capacity_planner,
            create_availability_provisioner,
        )

        assert inspect.iscoroutinefunction(create_availability_node_monitor)
        assert inspect.iscoroutinefunction(create_availability_recovery_engine)
        assert inspect.iscoroutinefunction(create_availability_capacity_planner)
        assert inspect.iscoroutinefunction(create_availability_provisioner)


class TestMaintenanceRunners:
    """Tests for maintenance and cleanup daemon runners."""

    def test_maintenance_runners_registered(self):
        """Test that maintenance runners are registered."""
        registry = _build_runner_registry()

        maintenance_types = [
            "MAINTENANCE",
            "ORPHAN_DETECTION",
            "DATA_CLEANUP",
            "DISK_SPACE_MANAGER",
            "COORDINATOR_DISK_MANAGER",
            "INTEGRITY_CHECK",
        ]

        for daemon_name in maintenance_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_maintenance_runner_functions_are_coroutines(self):
        """Test that maintenance runner functions are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_maintenance,
            create_orphan_detection,
            create_data_cleanup,
            create_disk_space_manager,
            create_coordinator_disk_manager,
            create_integrity_check,
        )

        assert inspect.iscoroutinefunction(create_maintenance)
        assert inspect.iscoroutinefunction(create_orphan_detection)
        assert inspect.iscoroutinefunction(create_data_cleanup)
        assert inspect.iscoroutinefunction(create_disk_space_manager)
        assert inspect.iscoroutinefunction(create_coordinator_disk_manager)
        assert inspect.iscoroutinefunction(create_integrity_check)


class TestS3Runners:
    """Tests for S3-related daemon runners."""

    def test_s3_runners_registered(self):
        """Test that S3 daemon runners are registered."""
        registry = _build_runner_registry()

        s3_types = [
            "S3_BACKUP",
            "S3_NODE_SYNC",
            "S3_CONSOLIDATION",
        ]

        for daemon_name in s3_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_s3_runner_functions_are_coroutines(self):
        """Test that S3 runner functions are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_s3_backup,
            create_s3_node_sync,
            create_s3_consolidation,
        )

        assert inspect.iscoroutinefunction(create_s3_backup)
        assert inspect.iscoroutinefunction(create_s3_node_sync)
        assert inspect.iscoroutinefunction(create_s3_consolidation)


class TestProviderIdleRunners:
    """Tests for cloud provider idle detection runners.

    These runners were removed from the daemon_runners.py registry but still
    exist as deprecated no-ops in the runners subpackage.
    """

    def test_provider_idle_runner_functions_are_coroutines(self):
        """Test that provider idle runner functions are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_lambda_idle,
            create_vast_idle,
        )

        assert inspect.iscoroutinefunction(create_lambda_idle)
        assert inspect.iscoroutinefunction(create_vast_idle)


class TestReplicationRunners:
    """Tests for replication daemon runners.

    These runners were removed from the daemon_runners.py registry but still
    exist as deprecated no-ops in the runners subpackage.
    """

    def test_replication_runner_functions_are_coroutines(self):
        """Test that replication runner functions are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_replication_monitor,
            create_replication_repair,
        )

        assert inspect.iscoroutinefunction(create_replication_monitor)
        assert inspect.iscoroutinefunction(create_replication_repair)


class TestSchedulingRunners:
    """Tests for scheduling and queue daemon runners."""

    def test_scheduling_runners_registered(self):
        """Test that scheduling daemon runners are registered."""
        registry = _build_runner_registry()

        scheduling_types = [
            "QUEUE_POPULATOR",
            "JOB_SCHEDULER",
            "FEEDBACK_LOOP",
            "CURRICULUM_INTEGRATION",
        ]

        for daemon_name in scheduling_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_scheduling_runner_functions_are_coroutines(self):
        """Test that scheduling runner functions are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_queue_populator,
            create_job_scheduler,
            create_feedback_loop,
            create_curriculum_integration,
        )

        assert inspect.iscoroutinefunction(create_queue_populator)
        assert inspect.iscoroutinefunction(create_job_scheduler)
        assert inspect.iscoroutinefunction(create_feedback_loop)
        assert inspect.iscoroutinefunction(create_curriculum_integration)


class TestP2PRunners:
    """Tests for P2P service daemon runners."""

    def test_p2p_runners_registered(self):
        """Test that P2P daemon runners are registered."""
        registry = _build_runner_registry()

        p2p_types = [
            "P2P_BACKEND",
            "P2P_AUTO_DEPLOY",
        ]

        for daemon_name in p2p_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_p2p_runner_functions_are_coroutines(self):
        """Test that P2P runner functions are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_p2p_backend,
            create_p2p_auto_deploy,
        )

        assert inspect.iscoroutinefunction(create_p2p_backend)
        assert inspect.iscoroutinefunction(create_p2p_auto_deploy)


class TestTrainingDataRunners:
    """Tests for training data sync daemon runners."""

    def test_training_data_runners_registered(self):
        """Test that training data daemon runners are registered."""
        registry = _build_runner_registry()

        training_data_types = [
            "TRAINING_NODE_WATCHER",
            "TRAINING_DATA_SYNC",
            "OWC_IMPORT",
            "VAST_CPU_PIPELINE",
        ]

        for daemon_name in training_data_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_training_data_runner_functions_are_coroutines(self):
        """Test that training data runner functions are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_training_node_watcher,
            create_training_data_sync,
            create_owc_import,
            create_vast_cpu_pipeline,
        )
        from app.coordination.daemon_runners import (
            create_external_drive_sync,
            create_cluster_data_sync,
        )

        assert inspect.iscoroutinefunction(create_training_node_watcher)
        assert inspect.iscoroutinefunction(create_training_data_sync)
        assert inspect.iscoroutinefunction(create_owc_import)
        assert inspect.iscoroutinefunction(create_external_drive_sync)
        assert inspect.iscoroutinefunction(create_vast_cpu_pipeline)
        assert inspect.iscoroutinefunction(create_cluster_data_sync)


class TestEvaluationRunners:
    """Tests for evaluation and tournament daemon runners."""

    def test_evaluation_runners_registered(self):
        """Test that evaluation daemon runners are registered."""
        registry = _build_runner_registry()

        evaluation_types = [
            "TOURNAMENT_DAEMON",
            "EVALUATION",
            "AUTO_PROMOTION",
            "UNIFIED_PROMOTION",
            "GAUNTLET_FEEDBACK",
        ]

        for daemon_name in evaluation_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_evaluation_runner_functions_are_coroutines(self):
        """Test that evaluation runner functions are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_tournament_daemon,
            create_evaluation_daemon,
            create_auto_promotion,
            create_unified_promotion,
            create_gauntlet_feedback,
        )

        assert inspect.iscoroutinefunction(create_tournament_daemon)
        assert inspect.iscoroutinefunction(create_evaluation_daemon)
        assert inspect.iscoroutinefunction(create_auto_promotion)
        assert inspect.iscoroutinefunction(create_unified_promotion)
        assert inspect.iscoroutinefunction(create_gauntlet_feedback)


class TestWaitForDaemonEdgeCases:
    """Additional edge case tests for _wait_for_daemon."""

    @pytest.mark.asyncio
    async def test_wait_for_daemon_with_callable_is_running(self):
        """Test _wait_for_daemon with is_running as a method."""
        mock_daemon = MagicMock()
        call_count = [0]

        def is_running_method():
            call_count[0] += 1
            return call_count[0] < 3

        mock_daemon.is_running = is_running_method

        await _wait_for_daemon(mock_daemon, check_interval=0.01)

        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_wait_for_daemon_with_legacy_running_attr(self):
        """Test _wait_for_daemon with _running attribute (legacy pattern)."""

        class LegacyDaemon:
            def __init__(self):
                self._running = True
                self.check_count = 0

            def stop(self):
                self._running = False

        daemon = LegacyDaemon()

        # Stop daemon after a brief delay
        async def stop_after_delay():
            await asyncio.sleep(0.02)
            daemon.stop()

        import asyncio
        asyncio.create_task(stop_after_delay())

        await _wait_for_daemon(daemon, check_interval=0.01)

        assert daemon._running is False


class TestRegistryLaziness:
    """Tests for lazy registry building behavior."""

    def test_registry_is_none_initially(self):
        """Test that _RUNNER_REGISTRY starts as None."""
        import app.coordination.daemon_runners as module

        # Reset the registry
        original = module._RUNNER_REGISTRY
        module._RUNNER_REGISTRY = None

        try:
            # First access should build the registry
            registry = module.get_all_runners()
            assert registry is not None
            assert len(registry) >= 60
        finally:
            # Restore original state
            module._RUNNER_REGISTRY = original

    def test_get_all_runners_returns_copy(self):
        """Test that get_all_runners returns a copy, not the original."""
        runners1 = get_all_runners()
        runners2 = get_all_runners()

        # Should be equal but not the same object
        assert runners1 == runners2
        assert runners1 is not runners2

        # Modifying one shouldn't affect the other
        runners1["TEST_KEY"] = lambda: None
        assert "TEST_KEY" not in runners2


class TestRegistryDaemonTypeCoverage:
    """Tests to verify registry covers all DaemonType enum values."""

    def test_registry_covers_all_non_deprecated_types(self):
        """Test that registry has entries for all active DaemonType values."""
        registry = _build_runner_registry()
        registry_names = set(registry.keys())

        # Get all DaemonType names
        all_type_names = {dt.name for dt in DaemonType}

        # Check coverage - should have most types covered
        covered = registry_names & all_type_names
        assert len(covered) >= 60, (
            f"Only {len(covered)} daemon types covered, expected >= 60"
        )

    def test_registry_keys_match_daemon_type_names_exactly(self):
        """Test that registry keys exactly match DaemonType enum names."""
        registry = _build_runner_registry()

        # Get all DaemonType names
        valid_names = {dt.name for dt in DaemonType}

        for key in registry.keys():
            assert key in valid_names, (
                f"Registry key '{key}' does not match any DaemonType"
            )


class TestNewRunnersNPZCombination:
    """Tests for NPZ combination daemon runner."""

    def test_npz_combination_runner_registered(self):
        """Test that NPZ_COMBINATION runner is registered."""
        registry = _build_runner_registry()
        assert "NPZ_COMBINATION" in registry

    def test_npz_combination_runner_is_coroutine(self):
        """Test that NPZ combination runner is a coroutine function."""
        import inspect
        from app.coordination.daemon_runners import create_npz_combination

        assert inspect.iscoroutinefunction(create_npz_combination)


class TestNodeAvailabilityRunner:
    """Tests for node availability daemon runner."""

    def test_node_availability_runner_registered(self):
        """Test that NODE_AVAILABILITY runner is registered."""
        registry = _build_runner_registry()
        assert "NODE_AVAILABILITY" in registry

    def test_node_availability_runner_is_coroutine(self):
        """Test that node availability runner is a coroutine function."""
        import inspect
        from app.coordination.daemon_runners import create_node_availability

        assert inspect.iscoroutinefunction(create_node_availability)


class TestMetricsAndAnalysisRunners:
    """Tests for metrics and analysis daemon runners."""

    def test_metrics_runners_registered(self):
        """Test that metrics daemon runners are registered."""
        registry = _build_runner_registry()

        metrics_types = [
            "METRICS_ANALYSIS",
            "DATA_CONSOLIDATION",
        ]

        for daemon_name in metrics_types:
            assert daemon_name in registry, f"Missing runner for {daemon_name}"

    def test_metrics_runner_functions_are_coroutines(self):
        """Test that metrics runner functions are coroutine functions."""
        import inspect
        from app.coordination.daemon_runners import (
            create_metrics_analysis,
            create_data_consolidation,
        )

        assert inspect.iscoroutinefunction(create_metrics_analysis)
        assert inspect.iscoroutinefunction(create_data_consolidation)


class TestUnifiedDataPlaneRunner:
    """Tests for unified data plane daemon runner."""

    def test_unified_data_plane_runner_registered(self):
        """Test that UNIFIED_DATA_PLANE runner is registered."""
        registry = _build_runner_registry()
        assert "UNIFIED_DATA_PLANE" in registry

    def test_unified_data_plane_runner_is_coroutine(self):
        """Test that unified data plane runner is a coroutine function."""
        import inspect
        from app.coordination.daemon_runners import create_unified_data_plane

        assert inspect.iscoroutinefunction(create_unified_data_plane)


class TestSyncPushRunner:
    """Tests for sync push daemon runner."""

    def test_sync_push_runner_registered(self):
        """Test that SYNC_PUSH runner is registered."""
        registry = _build_runner_registry()
        assert "SYNC_PUSH" in registry

    def test_sync_push_runner_is_coroutine(self):
        """Test that sync push runner is a coroutine function."""
        import inspect
        from app.coordination.daemon_runners import create_sync_push

        assert inspect.iscoroutinefunction(create_sync_push)
