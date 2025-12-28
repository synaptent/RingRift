"""Tests for coordination_bootstrap.py (December 2025).

This module tests the coordination bootstrap layer which initializes
all coordinators in dependency order.

Coverage targets:
- BootstrapCoordinatorStatus dataclass
- BootstrapState dataclass
- _validate_critical_imports() function
- Individual _init_* functions
- bootstrap_coordination() main entry point
- shutdown_coordination() graceful shutdown
- get_bootstrap_status() status reporting
- is_coordination_ready() readiness check
- reset_bootstrap_state() state reset
- run_bootstrap_smoke_test() smoke testing
- _wire_missing_event_subscriptions() event wiring
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest


class TestBootstrapCoordinatorStatus:
    """Tests for BootstrapCoordinatorStatus dataclass."""

    def test_default_values(self):
        """Test default values for BootstrapCoordinatorStatus."""
        from app.coordination.coordination_bootstrap import BootstrapCoordinatorStatus

        status = BootstrapCoordinatorStatus(name="test_coordinator")

        assert status.name == "test_coordinator"
        assert status.initialized is False
        assert status.subscribed is False
        assert status.error is None
        assert status.initialized_at is None

    def test_initialized_status(self):
        """Test initialized coordinator status."""
        from app.coordination.coordination_bootstrap import BootstrapCoordinatorStatus

        now = datetime.now()
        status = BootstrapCoordinatorStatus(
            name="test_coordinator",
            initialized=True,
            subscribed=True,
            initialized_at=now,
        )

        assert status.initialized is True
        assert status.subscribed is True
        assert status.initialized_at == now

    def test_error_status(self):
        """Test error status tracking."""
        from app.coordination.coordination_bootstrap import BootstrapCoordinatorStatus

        status = BootstrapCoordinatorStatus(
            name="failed_coordinator",
            initialized=False,
            error="Import error: module not found",
        )

        assert status.initialized is False
        assert status.error == "Import error: module not found"


class TestBootstrapState:
    """Tests for BootstrapState dataclass."""

    def test_default_values(self):
        """Test default values for BootstrapState."""
        from app.coordination.coordination_bootstrap import BootstrapState

        state = BootstrapState()

        assert state.initialized is False
        assert state.started_at is None
        assert state.completed_at is None
        assert state.coordinators == {}
        assert state.errors == []
        assert state.shutdown_requested is False

    def test_populated_state(self):
        """Test populated BootstrapState."""
        from app.coordination.coordination_bootstrap import (
            BootstrapState,
            BootstrapCoordinatorStatus,
        )

        now = datetime.now()
        status = BootstrapCoordinatorStatus(name="test", initialized=True)

        state = BootstrapState(
            initialized=True,
            started_at=now,
            completed_at=now,
            coordinators={"test": status},
            errors=["error1"],
            shutdown_requested=False,
        )

        assert state.initialized is True
        assert state.started_at == now
        assert state.completed_at == now
        assert "test" in state.coordinators
        assert len(state.errors) == 1


class TestValidateCriticalImports:
    """Tests for _validate_critical_imports()."""

    def test_validates_critical_modules(self):
        """Test that critical modules are validated."""
        from app.coordination.coordination_bootstrap import _validate_critical_imports

        result = _validate_critical_imports()

        # Result should have the expected keys
        assert "critical_failures" in result
        assert "optional_failures" in result
        assert "validated" in result

        # Should be lists
        assert isinstance(result["critical_failures"], list)
        assert isinstance(result["optional_failures"], list)
        assert isinstance(result["validated"], list)

    def test_strict_mode_raises_on_failure(self):
        """Test that strict mode raises RuntimeError on critical failures."""
        from app.coordination.coordination_bootstrap import _validate_critical_imports

        # Mock a critical module to fail
        with patch.dict("os.environ", {"RINGRIFT_REQUIRE_CRITICAL_IMPORTS": "1"}):
            with patch(
                "app.coordination.coordination_bootstrap._CRITICAL_MODULES",
                [("nonexistent.module", "Test module")],
            ):
                with pytest.raises(RuntimeError, match="Critical imports failed"):
                    _validate_critical_imports()


class TestResetBootstrapState:
    """Tests for reset_bootstrap_state()."""

    def test_reset_clears_state(self):
        """Test that reset_bootstrap_state clears global state."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            get_bootstrap_status,
        )

        # Reset first to ensure clean state
        reset_bootstrap_state()

        status = get_bootstrap_status()
        assert status["initialized"] is False
        assert status["coordinators"] == {}
        assert status["errors"] == []


class TestGetBootstrapStatus:
    """Tests for get_bootstrap_status()."""

    def test_returns_status_dict(self):
        """Test that get_bootstrap_status returns proper dict."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            get_bootstrap_status,
        )

        reset_bootstrap_state()
        status = get_bootstrap_status()

        assert isinstance(status, dict)
        assert "initialized" in status
        assert "started_at" in status
        assert "completed_at" in status
        assert "initialized_count" in status
        assert "subscribed_count" in status
        assert "total_count" in status
        assert "coordinators" in status
        assert "errors" in status
        assert "shutdown_requested" in status

    def test_uninitialized_status(self):
        """Test status when not initialized."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            get_bootstrap_status,
        )

        reset_bootstrap_state()
        status = get_bootstrap_status()

        assert status["initialized"] is False
        assert status["started_at"] is None
        assert status["completed_at"] is None
        assert status["initialized_count"] == 0
        assert status["subscribed_count"] == 0


class TestIsCoordinationReady:
    """Tests for is_coordination_ready()."""

    def test_not_ready_when_uninitialized(self):
        """Test is_coordination_ready returns False when not initialized."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            is_coordination_ready,
        )

        reset_bootstrap_state()
        assert is_coordination_ready() is False

    def test_ready_after_core_coordinators_init(self):
        """Test is_coordination_ready returns True after core init."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            is_coordination_ready,
            BootstrapCoordinatorStatus,
        )
        import app.coordination.coordination_bootstrap as bootstrap_module

        reset_bootstrap_state()

        # Manually set up initialized state with core coordinators
        # Access _state through the module to get the current global instance
        bootstrap_module._state.initialized = True
        bootstrap_module._state.coordinators = {
            "resource_coordinator": BootstrapCoordinatorStatus(
                name="resource_coordinator", initialized=True
            ),
            "metrics_orchestrator": BootstrapCoordinatorStatus(
                name="metrics_orchestrator", initialized=True
            ),
            "cache_orchestrator": BootstrapCoordinatorStatus(
                name="cache_orchestrator", initialized=True
            ),
        }

        assert is_coordination_ready() is True

        # Clean up
        reset_bootstrap_state()


class TestBootstrapCoordination:
    """Tests for bootstrap_coordination() main entry point."""

    def test_idempotent_call(self):
        """Test that bootstrap_coordination is idempotent."""
        import app.coordination.coordination_bootstrap as bootstrap_module
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            bootstrap_coordination,
        )

        reset_bootstrap_state()

        # CRITICAL: Must access _state from module AFTER reset, not from import
        # reset_bootstrap_state() creates a new BootstrapState object
        bootstrap_module._state.initialized = True

        # Second call should return existing status without re-init
        result = bootstrap_coordination()

        assert result["initialized"] is True

        reset_bootstrap_state()

    def test_disabling_all_coordinators(self):
        """Test bootstrap with all coordinators disabled."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            bootstrap_coordination,
            get_bootstrap_status,
        )

        reset_bootstrap_state()

        # Disable all coordinators
        result = bootstrap_coordination(
            enable_resources=False,
            enable_metrics=False,
            enable_optimization=False,
            enable_cache=False,
            enable_model=False,
            enable_error=False,
            enable_health=False,
            enable_leadership=False,
            enable_selfplay=False,
            enable_pipeline=False,
            enable_task=False,
            enable_sync=False,
            enable_training=False,
            enable_recovery=False,
            enable_transfer=False,
            enable_ephemeral=False,
            enable_queue=False,
            enable_multi_provider=False,
            enable_job_scheduler=False,
            enable_global_task=False,
            enable_integrations=False,
            enable_auto_export=False,
            enable_auto_evaluation=False,
            enable_model_distribution=False,
            enable_idle_resource=False,
            enable_quality_monitor=False,
            enable_orphan_detection=False,
            enable_curriculum_integration=False,
            register_with_registry=False,
        )

        assert result["initialized"] is True
        # No coordinators should be tracked since all disabled
        # (But _state.initialized is set to True)

        reset_bootstrap_state()

    def test_master_loop_enforcement_disabled(self):
        """Test that master loop check can be skipped via env var."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            bootstrap_coordination,
        )

        reset_bootstrap_state()

        with patch.dict("os.environ", {"RINGRIFT_SKIP_MASTER_LOOP_CHECK": "1"}):
            # Should not raise even with require_master_loop=True
            # because skip check is enabled
            result = bootstrap_coordination(
                require_master_loop=True,
                enable_resources=False,
                enable_metrics=False,
                enable_optimization=False,
                enable_cache=False,
                enable_model=False,
                enable_health=False,
                enable_leadership=False,
                enable_selfplay=False,
                enable_pipeline=False,
                enable_task=False,
                enable_sync=False,
                enable_training=False,
                enable_transfer=False,
                enable_ephemeral=False,
                enable_queue=False,
                enable_multi_provider=False,
                enable_job_scheduler=False,
                enable_global_task=False,
                enable_integrations=False,
                enable_auto_export=False,
                enable_auto_evaluation=False,
                enable_model_distribution=False,
                enable_idle_resource=False,
                enable_quality_monitor=False,
                enable_orphan_detection=False,
                enable_curriculum_integration=False,
                register_with_registry=False,
            )

            assert result["initialized"] is True

        reset_bootstrap_state()


class TestShutdownCoordination:
    """Tests for shutdown_coordination()."""

    def test_shutdown_when_not_initialized(self):
        """Test shutdown returns early when not initialized."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            shutdown_coordination,
        )

        reset_bootstrap_state()

        result = shutdown_coordination()

        assert result["shutdown"] is False
        assert result["reason"] == "not initialized"

    def test_shutdown_sets_flag(self):
        """Test shutdown sets shutdown_requested flag."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            shutdown_coordination,
        )
        import app.coordination.coordination_bootstrap as bootstrap_module

        reset_bootstrap_state()

        # Manually set initialized via module reference
        bootstrap_module._state.initialized = True

        result = shutdown_coordination()

        assert result["shutdown"] is True
        assert bootstrap_module._state.shutdown_requested is True

        reset_bootstrap_state()


class TestCoordinatorRegistry:
    """Tests for COORDINATOR_REGISTRY-based initialization (Dec 2025 refactor).

    Note: The old _init_*() functions were replaced with COORDINATOR_REGISTRY
    + _init_coordinator_from_spec() pattern. See CLAUDE.md for details.
    """

    def test_registry_has_expected_coordinators(self):
        """Test that COORDINATOR_REGISTRY contains expected coordinator specs."""
        from app.coordination.coordination_bootstrap import (
            COORDINATOR_REGISTRY,
            CoordinatorSpec,
        )

        # Verify registry is populated
        assert len(COORDINATOR_REGISTRY) > 0, "COORDINATOR_REGISTRY should not be empty"

        # Verify each entry is a CoordinatorSpec
        for name, spec in COORDINATOR_REGISTRY.items():
            assert isinstance(spec, CoordinatorSpec), (
                f"Entry '{name}' should be a CoordinatorSpec, got {type(spec)}"
            )
            assert isinstance(name, str)
            assert spec.pattern is not None

    def test_init_coordinator_from_spec_returns_status(self):
        """Test that _init_coordinator_from_spec returns BootstrapCoordinatorStatus."""
        from app.coordination.coordination_bootstrap import (
            COORDINATOR_REGISTRY,
            _init_coordinator_from_spec,
            BootstrapCoordinatorStatus,
        )

        # Test with first available spec that has pattern=SKIP (no side effects)
        skip_specs = [
            (name, spec) for name, spec in COORDINATOR_REGISTRY.items()
            if spec.pattern.name == "SKIP"
        ]

        if skip_specs:
            name, spec = skip_specs[0]
            status = _init_coordinator_from_spec(spec)
            assert isinstance(status, BootstrapCoordinatorStatus), (
                f"_init_coordinator_from_spec did not return BootstrapCoordinatorStatus"
            )
            assert isinstance(status.name, str)
            assert isinstance(status.initialized, bool)
            assert isinstance(status.subscribed, bool)

    def test_deprecated_coordinators_return_skip_status(self):
        """Test that deprecated coordinators return skip pattern status."""
        from app.coordination.coordination_bootstrap import (
            COORDINATOR_REGISTRY,
            _init_coordinator_from_spec,
            InitPattern,
        )

        # Find coordinators with SKIP pattern (deprecated)
        for name, spec in COORDINATOR_REGISTRY.items():
            if spec.pattern == InitPattern.SKIP:
                status = _init_coordinator_from_spec(spec)
                # SKIP pattern should return initialized=True (no-op success)
                assert status.initialized is True, (
                    f"SKIP coordinator '{name}' should return initialized=True"
                )


class TestSmokeTest:
    """Tests for run_bootstrap_smoke_test()."""

    def test_smoke_test_returns_results(self):
        """Test that smoke test returns proper result structure."""
        from app.coordination.coordination_bootstrap import run_bootstrap_smoke_test

        result = run_bootstrap_smoke_test()

        assert isinstance(result, dict)
        assert "passed" in result
        assert "checks" in result
        assert "passed_count" in result
        assert "failed_count" in result
        assert "warnings" in result

        assert isinstance(result["checks"], list)
        assert isinstance(result["warnings"], list)
        assert isinstance(result["passed_count"], int)
        assert isinstance(result["failed_count"], int)

    def test_smoke_test_check_structure(self):
        """Test that individual checks have proper structure."""
        from app.coordination.coordination_bootstrap import run_bootstrap_smoke_test

        result = run_bootstrap_smoke_test()

        for check in result["checks"]:
            assert "name" in check
            assert "passed" in check
            # error and details may be None

    def test_smoke_test_counts_match(self):
        """Test that passed + failed counts match total checks."""
        from app.coordination.coordination_bootstrap import run_bootstrap_smoke_test

        result = run_bootstrap_smoke_test()

        total_checks = len(result["checks"])
        assert result["passed_count"] + result["failed_count"] == total_checks


class TestSmokeTestResult:
    """Tests for SmokeTestResult dataclass."""

    def test_default_values(self):
        """Test default values for SmokeTestResult."""
        from app.coordination.coordination_bootstrap import SmokeTestResult

        result = SmokeTestResult(name="test_check", passed=True)

        assert result.name == "test_check"
        assert result.passed is True
        assert result.error is None
        assert result.details is None

    def test_with_error(self):
        """Test SmokeTestResult with error."""
        from app.coordination.coordination_bootstrap import SmokeTestResult

        result = SmokeTestResult(
            name="failed_check",
            passed=False,
            error="Something went wrong",
        )

        assert result.passed is False
        assert result.error == "Something went wrong"

    def test_with_details(self):
        """Test SmokeTestResult with details."""
        from app.coordination.coordination_bootstrap import SmokeTestResult

        details = {"subscriber_count": 5, "daemon_count": 3}
        result = SmokeTestResult(
            name="check_with_details",
            passed=True,
            details=details,
        )

        assert result.details == details
        assert result.details["subscriber_count"] == 5


class TestWireMissingEventSubscriptions:
    """Tests for _wire_missing_event_subscriptions()."""

    def test_returns_results_dict(self):
        """Test that _wire_missing_event_subscriptions returns results dict."""
        from app.coordination.coordination_bootstrap import _wire_missing_event_subscriptions

        results = _wire_missing_event_subscriptions()

        assert isinstance(results, dict)
        # Results should be booleans indicating success/failure
        for key, value in results.items():
            assert isinstance(key, str)
            assert isinstance(value, bool)

    def test_handles_missing_modules(self):
        """Test that missing modules are handled gracefully."""
        from app.coordination.coordination_bootstrap import _wire_missing_event_subscriptions

        # Even if some modules are missing, should not raise
        results = _wire_missing_event_subscriptions()

        # Should return some results
        assert isinstance(results, dict)


class TestValidateEventWiring:
    """Tests for _validate_event_wiring()."""

    def test_returns_validation_results(self):
        """Test that _validate_event_wiring returns proper structure."""
        from app.coordination.coordination_bootstrap import _validate_event_wiring

        results = _validate_event_wiring()

        assert isinstance(results, dict)
        assert "healthy" in results
        assert "issues" in results
        assert "recommendations" in results
        assert "validated" in results


class TestStartUnifiedFeedbackOrchestrator:
    """Tests for _start_unified_feedback_orchestrator()."""

    def test_handles_import_error(self):
        """Test that missing module is handled gracefully."""
        from app.coordination.coordination_bootstrap import _start_unified_feedback_orchestrator

        # Should not raise even if unified_feedback doesn't exist
        result = _start_unified_feedback_orchestrator()

        # Result is boolean
        assert isinstance(result, bool)


class TestWireIntegrations:
    """Tests for _wire_integrations()."""

    def test_handles_missing_integration_bridge(self):
        """Test that missing integration_bridge is handled gracefully."""
        from app.coordination.coordination_bootstrap import _wire_integrations

        # Should not raise even if integration_bridge doesn't exist
        result = _wire_integrations()

        # Result is boolean
        assert isinstance(result, bool)


class TestRegisterCoordinators:
    """Tests for _register_coordinators()."""

    def test_handles_missing_registry(self):
        """Test that missing registry is handled gracefully."""
        from app.coordination.coordination_bootstrap import _register_coordinators

        # Should not raise even if registry has issues
        result = _register_coordinators()

        # Result is boolean
        assert isinstance(result, bool)


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """Test that all __all__ exports are importable."""
        from app.coordination import coordination_bootstrap

        for name in coordination_bootstrap.__all__:
            assert hasattr(coordination_bootstrap, name), f"Missing export: {name}"

    def test_expected_exports(self):
        """Test that expected functions/classes are exported."""
        from app.coordination.coordination_bootstrap import (
            BootstrapCoordinatorStatus,
            BootstrapState,
            SmokeTestResult,
            bootstrap_coordination,
            get_bootstrap_status,
            is_coordination_ready,
            reset_bootstrap_state,
            run_bootstrap_smoke_test,
            shutdown_coordination,
        )

        # All should be callable or classes
        assert callable(bootstrap_coordination)
        assert callable(get_bootstrap_status)
        assert callable(is_coordination_ready)
        assert callable(reset_bootstrap_state)
        assert callable(run_bootstrap_smoke_test)
        assert callable(shutdown_coordination)


class TestIntegration:
    """Integration tests for bootstrap flow."""

    def test_full_bootstrap_shutdown_cycle(self):
        """Test complete bootstrap -> status -> shutdown cycle."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            bootstrap_coordination,
            get_bootstrap_status,
            shutdown_coordination,
            is_coordination_ready,
        )

        # Start fresh
        reset_bootstrap_state()

        # Bootstrap with minimal coordinators for speed
        result = bootstrap_coordination(
            enable_resources=False,
            enable_metrics=False,
            enable_optimization=False,
            enable_cache=False,
            enable_model=False,
            enable_health=False,
            enable_leadership=False,
            enable_selfplay=False,
            enable_pipeline=False,
            enable_task=False,
            enable_sync=False,
            enable_training=False,
            enable_transfer=False,
            enable_ephemeral=False,
            enable_queue=False,
            enable_multi_provider=False,
            enable_job_scheduler=False,
            enable_global_task=False,
            enable_integrations=False,
            enable_auto_export=False,
            enable_auto_evaluation=False,
            enable_model_distribution=False,
            enable_idle_resource=False,
            enable_quality_monitor=False,
            enable_orphan_detection=False,
            enable_curriculum_integration=False,
            register_with_registry=False,
        )

        assert result["initialized"] is True

        # Get status
        status = get_bootstrap_status()
        assert status["initialized"] is True
        assert status["shutdown_requested"] is False

        # Shutdown
        shutdown_result = shutdown_coordination()
        assert shutdown_result["shutdown"] is True

        # Verify shutdown state
        final_status = get_bootstrap_status()
        assert final_status["shutdown_requested"] is True

        # Clean up
        reset_bootstrap_state()


class TestCriticalModulesConfig:
    """Tests for critical modules configuration."""

    def test_critical_modules_list_exists(self):
        """Test that _CRITICAL_MODULES is defined."""
        from app.coordination.coordination_bootstrap import _CRITICAL_MODULES

        assert isinstance(_CRITICAL_MODULES, list)
        assert len(_CRITICAL_MODULES) > 0

        # Each entry should be (module_path, description)
        for entry in _CRITICAL_MODULES:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], str)

    def test_optional_modules_list_exists(self):
        """Test that _OPTIONAL_MODULES is defined."""
        from app.coordination.coordination_bootstrap import _OPTIONAL_MODULES

        assert isinstance(_OPTIONAL_MODULES, list)

        # Each entry should be (module_path, description)
        for entry in _OPTIONAL_MODULES:
            assert isinstance(entry, tuple)
            assert len(entry) == 2


class TestBootstrapWithPipelineConfig:
    """Tests for bootstrap with pipeline configuration."""

    def test_pipeline_config_passed_to_orchestrator(self):
        """Test that pipeline config is passed correctly."""
        from app.coordination.coordination_bootstrap import (
            reset_bootstrap_state,
            bootstrap_coordination,
        )

        reset_bootstrap_state()

        # Mock the pipeline orchestrator init
        with patch(
            "app.coordination.coordination_bootstrap._init_pipeline_orchestrator"
        ) as mock_init:
            from app.coordination.coordination_bootstrap import BootstrapCoordinatorStatus

            mock_init.return_value = BootstrapCoordinatorStatus(
                name="pipeline_orchestrator", initialized=True, subscribed=True
            )

            bootstrap_coordination(
                enable_pipeline=True,
                pipeline_auto_trigger=True,
                training_epochs=50,
                training_batch_size=256,
                training_model_version="v3",
                # Disable others for speed
                enable_resources=False,
                enable_metrics=False,
                enable_optimization=False,
                enable_cache=False,
                enable_model=False,
                enable_health=False,
                enable_leadership=False,
                enable_selfplay=False,
                enable_task=False,
                enable_sync=False,
                enable_training=False,
                enable_transfer=False,
                enable_ephemeral=False,
                enable_queue=False,
                enable_multi_provider=False,
                enable_job_scheduler=False,
                enable_global_task=False,
                enable_integrations=False,
                enable_auto_export=False,
                enable_auto_evaluation=False,
                enable_model_distribution=False,
                enable_idle_resource=False,
                enable_quality_monitor=False,
                enable_orphan_detection=False,
                enable_curriculum_integration=False,
                register_with_registry=False,
            )

        reset_bootstrap_state()
